//! Source/sink nodes backed by an `ffmpeg` child process.
//!
//! Rather than binding to libav*, these spawn the `ffmpeg` CLI and exchange
//! YUV4MPEG2 with it over a pipe (`-f yuv4mpegpipe`), decoded and encoded by
//! the same [`super::y4m`] code the file nodes use. This keeps the crate free
//! of native dependencies: ffmpeg is a *runtime* requirement of these two
//! node kinds only, checked when they are built ([`BuildError::Unavailable`])
//! and again when they start ([`NodeError::Message`]).
//!
//! ffmpeg's stderr is drained on a background thread (otherwise a chatty
//! ffmpeg could fill the pipe and deadlock against us) and its tail is quoted
//! in any error, since that is where the actual reason for a failure lives.

use std::io::{BufWriter, Read};
use std::path::PathBuf;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::OnceLock;
use std::thread::JoinHandle;

use super::y4m::{Chroma, Y4mError, Y4mHeader, Y4mReader, Y4mWriter};
use super::{frame_in_ports, frame_out_ports, parse_fps};
use crate::data::Payload;
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet};
use crate::graph::Params;

/// Name of the program spawned. Resolved through `PATH`.
const FFMPEG: &str = "ffmpeg";

/// How much of ffmpeg's stderr to keep for error messages.
const STDERR_TAIL: usize = 4096;

/// Whether `ffmpeg` can be launched. Probed once per process (by running
/// `ffmpeg -version`) and cached, because graph validation builds every node
/// more than once.
pub fn ffmpeg_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new(FFMPEG)
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

fn require_ffmpeg(kind: &str) -> Result<(), BuildError> {
    if ffmpeg_available() {
        Ok(())
    } else {
        Err(BuildError::Unavailable(format!(
            "{kind} needs the '{FFMPEG}' program, which was not found on PATH"
        )))
    }
}

fn spawn(cmd: &mut Command, kind: &str) -> Result<Child, NodeError> {
    cmd.stderr(Stdio::piped()).spawn().map_err(|e| {
        NodeError::Message(format!(
            "{kind}: could not start '{FFMPEG}' ({e}); is it installed and on PATH?"
        ))
    })
}

/// Drain a child's stderr on a thread, keeping only the last few KiB.
fn drain_stderr(stderr: Option<ChildStderr>) -> Option<JoinHandle<String>> {
    let mut stderr = stderr?;
    Some(std::thread::spawn(move || {
        let mut tail: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 1024];
        while let Ok(n) = stderr.read(&mut chunk) {
            if n == 0 {
                break;
            }
            tail.extend_from_slice(&chunk[..n]);
            if tail.len() > STDERR_TAIL {
                tail.drain(..tail.len() - STDERR_TAIL);
            }
        }
        String::from_utf8_lossy(&tail).trim().to_string()
    }))
}

/// Collect the drained stderr. Only call once the child has exited (or its
/// stderr is otherwise closed), or this blocks until it does.
fn stderr_text(handle: &mut Option<JoinHandle<String>>) -> String {
    handle
        .take()
        .and_then(|h| h.join().ok())
        .unwrap_or_default()
}

fn describe_failure(kind: &str, what: &str, status: Option<ExitStatus>, stderr: &str) -> String {
    let mut msg = format!("{kind}: {what}");
    if let Some(s) = status {
        msg.push_str(&format!(" (ffmpeg {s})"));
    }
    if !stderr.is_empty() {
        msg.push_str(": ");
        msg.push_str(stderr);
    }
    msg
}

// --- source ------------------------------------------------------------------

/// A running decoder: `ffmpeg -i <path> -f yuv4mpegpipe -` read through
/// [`Y4mReader`].
struct Decoder {
    child: Child,
    reader: Y4mReader<ChildStdout>,
    stderr: Option<JoinHandle<String>>,
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // We may stop reading before ffmpeg has finished writing (reset, or the
        // graph is dropped mid-stream). Kill it rather than wait for a writer
        // that is blocked on a pipe nobody will drain.
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = stderr_text(&mut self.stderr);
    }
}

/// Decodes any file ffmpeg can read into 3-channel `u8` RGB frames.
pub struct FfmpegSource {
    path: PathBuf,
    decoder: Option<Decoder>,
    exhausted: bool,
}

impl FfmpegSource {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            decoder: None,
            exhausted: false,
        }
    }

    fn start(&self) -> Result<Decoder, NodeError> {
        let mut child = spawn(
            Command::new(FFMPEG)
                .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
                .arg(&self.path)
                // 4:4:4 so ffmpeg's own (filtered) chroma upsampling is used
                // rather than our nearest-neighbour one.
                .args(["-f", "yuv4mpegpipe", "-pix_fmt", "yuv444p", "-"])
                .stdin(Stdio::null())
                .stdout(Stdio::piped()),
            "ffmpeg_source",
        )?;
        let stderr = drain_stderr(child.stderr.take());
        let stdout = child.stdout.take().expect("stdout was piped");
        match Y4mReader::new(stdout) {
            Ok(reader) => Ok(Decoder {
                child,
                reader,
                stderr,
            }),
            Err(e) => {
                // Usually ffmpeg could not open the input and wrote nothing;
                // its stderr says why.
                let status = child.wait().ok();
                let mut stderr = stderr;
                Err(NodeError::Message(describe_failure(
                    "ffmpeg_source",
                    &format!("reading '{}' failed: {e}", self.path.display()),
                    status,
                    &stderr_text(&mut stderr),
                )))
            }
        }
    }
}

impl TryFrom<&Params> for FfmpegSource {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        // Not checked for existence: ffmpeg inputs may be URLs or devices.
        let path = PathBuf::from(p.get_str("path")?);
        require_ffmpeg("ffmpeg_source")?;
        Ok(Self::new(path))
    }
}

impl Node for FfmpegSource {
    fn ports(&self) -> PortSet {
        frame_out_ports()
    }

    fn eval(&mut self, _inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        if self.exhausted {
            return Err(NodeError::EndOfStream);
        }
        if self.decoder.is_none() {
            self.decoder = Some(self.start()?);
        }
        let dec = self.decoder.as_mut().expect("started above");
        match dec.reader.read_rgb() {
            Ok(Some(frame)) => {
                outputs.set("out", Payload::Frame(frame));
                Ok(())
            }
            Ok(None) => {
                // Clean end of the pipe. Distinguish "the video ended" from
                // "ffmpeg died part-way" by its exit status.
                self.exhausted = true;
                let mut dec = self.decoder.take().expect("started above");
                let status = dec.child.wait().ok();
                if status.is_some_and(|s| s.success()) {
                    Err(NodeError::EndOfStream)
                } else {
                    Err(NodeError::Message(describe_failure(
                        "ffmpeg_source",
                        &format!("decoding '{}' failed", self.path.display()),
                        status,
                        &stderr_text(&mut dec.stderr),
                    )))
                }
            }
            Err(e) => {
                // Do not restart on the next evaluation: that would silently
                // loop a corrupt input from frame 0.
                self.exhausted = true;
                let mut dec = self.decoder.take().expect("started above");
                let _ = dec.child.kill();
                let status = dec.child.wait().ok();
                Err(NodeError::Message(describe_failure(
                    "ffmpeg_source",
                    &e.to_string(),
                    status,
                    &stderr_text(&mut dec.stderr),
                )))
            }
        }
    }

    /// Rewind: stop the current decoder; the next evaluation starts over.
    fn reset(&mut self) {
        self.decoder = None;
        self.exhausted = false;
    }
}

// --- sink --------------------------------------------------------------------

/// A running encoder: `ffmpeg -f yuv4mpegpipe -i - <path>` fed through
/// [`Y4mWriter`].
struct Encoder {
    child: Child,
    /// `None` once stdin has been closed by [`Encoder::finish`].
    writer: Option<Y4mWriter<BufWriter<ChildStdin>>>,
    stderr: Option<JoinHandle<String>>,
    finished: bool,
}

impl Encoder {
    /// Close ffmpeg's stdin and wait for it to write the trailer and exit.
    /// Idempotent; returns ffmpeg's complaint if it did not exit cleanly.
    fn finish(&mut self, path: &std::path::Path) -> Result<(), String> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let flushed = match self.writer.take() {
            // Dropping the stdin handle is what signals EOF to ffmpeg.
            Some(w) => w
                .into_inner()
                .map_err(|e| e.to_string())
                .and_then(|bw| bw.into_inner().map(drop).map_err(|e| e.to_string())),
            None => Ok(()),
        };
        let status = self.child.wait().ok();
        let stderr = stderr_text(&mut self.stderr);
        match (flushed, status) {
            (Ok(()), Some(s)) if s.success() => Ok(()),
            (flushed, status) => Err(describe_failure(
                "ffmpeg_sink",
                &format!(
                    "encoding '{}' failed{}",
                    path.display(),
                    flushed.err().map(|e| format!(" ({e})")).unwrap_or_default()
                ),
                status,
                &stderr,
            )),
        }
    }
}

/// Encodes incoming frames to any file ffmpeg can write (container and codec
/// chosen by ffmpeg from the extension, e.g. `.mp4` → H.264 when available).
///
/// ffmpeg is started on the first frame, with the stream size taken from it.
/// The output is only complete once the sink is reset or dropped — see the
/// module docs on finalization.
pub struct FfmpegSink {
    path: PathBuf,
    fps: (u32, u32),
    pix_fmt: String,
    encoder: Option<Encoder>,
}

impl FfmpegSink {
    pub fn new(path: impl Into<PathBuf>, fps: (u32, u32), pix_fmt: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            fps,
            pix_fmt: pix_fmt.into(),
            encoder: None,
        }
    }

    /// Finalize the output now (close ffmpeg's input and wait for it), rather
    /// than on drop. A later frame starts a new encode, overwriting the file.
    pub fn finish(&mut self) -> Result<(), NodeError> {
        match self.encoder.take() {
            Some(mut enc) => enc.finish(&self.path).map_err(NodeError::Message),
            None => Ok(()),
        }
    }

    fn start(&self, header: Y4mHeader) -> Result<Encoder, NodeError> {
        let mut child = spawn(
            Command::new(FFMPEG)
                .args(["-y", "-hide_banner", "-loglevel", "error"])
                .args(["-f", "yuv4mpegpipe", "-i", "-", "-pix_fmt", &self.pix_fmt])
                .arg(&self.path)
                .stdin(Stdio::piped())
                .stdout(Stdio::null()),
            "ffmpeg_sink",
        )?;
        let stderr = drain_stderr(child.stderr.take());
        let stdin = child.stdin.take().expect("stdin was piped");
        Ok(Encoder {
            child,
            writer: Some(Y4mWriter::new(BufWriter::new(stdin), header)),
            stderr,
            finished: false,
        })
    }
}

impl TryFrom<&Params> for FfmpegSink {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        let path = PathBuf::from(p.get_str("path")?);
        let fps = parse_fps(p, "fps", (25, 1))?;
        // yuv420p is the most widely playable choice for mp4/H.264.
        let pix_fmt = p.get("pix_fmt").map_or("yuv420p", String::as_str);
        require_ffmpeg("ffmpeg_sink")?;
        Ok(Self::new(path, fps, pix_fmt))
    }
}

impl Node for FfmpegSink {
    fn ports(&self) -> PortSet {
        frame_in_ports()
    }

    fn eval(&mut self, inputs: &Inputs, _outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;
        if self.encoder.is_none() {
            // Refuse a frame we could not encode *before* launching ffmpeg,
            // so a bad first frame does not leave an empty encode behind.
            if frame.as_u8().is_none() || !matches!(frame.channels, 1 | 3) {
                return Err(NodeError::Message(format!(
                    "ffmpeg_sink: expected a 1- or 3-channel u8 frame, got {}-channel {:?}",
                    frame.channels,
                    frame.dtype()
                )));
            }
            // Send 4:4:4 (or mono) so the only chroma downsampling is
            // ffmpeg's own, to whatever `pix_fmt` asks for.
            let chroma = if frame.channels == 1 {
                Chroma::Mono
            } else {
                Chroma::C444
            };
            let header = Y4mHeader::new(frame.width, frame.height, self.fps, chroma);
            self.encoder = Some(self.start(header)?);
        }
        let enc = self.encoder.as_mut().expect("started above");
        let writer = enc.writer.as_mut().expect("encoder not finished");
        match writer.write_frame(frame) {
            Ok(()) => Ok(()),
            // A frame we refused (wrong size, f32, ...): nothing was sent, so
            // the encode is intact and keeps going with later valid frames.
            Err(e @ (Y4mError::Format(_) | Y4mError::Unsupported(_))) => {
                Err(NodeError::Message(format!("ffmpeg_sink: {e}")))
            }
            // An I/O error is typically a broken pipe because ffmpeg gave up;
            // finish() reaps it and returns its stderr, which explains why.
            Err(e) => {
                let mut enc = self.encoder.take().expect("started above");
                Err(match enc.finish(&self.path) {
                    Err(why) => NodeError::Message(format!("{why} (while writing: {e})")),
                    Ok(()) => NodeError::Message(format!("ffmpeg_sink: {e}")),
                })
            }
        }
    }

    /// Finalize the current output; the next frame starts a fresh encode.
    fn reset(&mut self) {
        if let Err(e) = self.finish() {
            eprintln!("{e}");
        }
    }
}

impl Drop for FfmpegSink {
    fn drop(&mut self) {
        // A Drop impl cannot return the error, and silently losing a failed
        // encode (an unplayable mp4) is worse than a line on stderr.
        if let Err(e) = self.finish() {
            eprintln!("{e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sink_builds_without_spawning_and_drop_is_a_noop() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        }
        let mut p = Params::new();
        p.insert("path".into(), "/nonexistent/dir/out.mp4".into());
        let mut sink = FfmpegSink::try_from(&p).unwrap();
        // Never received a frame, so there is nothing to finalize.
        assert!(sink.finish().is_ok());
    }

    #[test]
    fn missing_input_reports_ffmpeg_stderr() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        }
        let mut src = FfmpegSource::new("/definitely/not/here.mp4");
        let err = src
            .eval(&Inputs::default(), &mut Outputs::new())
            .unwrap_err();
        let NodeError::Message(msg) = err else {
            panic!("expected a message, got {err:?}");
        };
        assert!(msg.contains("ffmpeg_source"), "{msg}");
        assert!(msg.contains("No such file"), "{msg}");
    }
}
