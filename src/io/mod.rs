//! Video I/O: source and sink nodes that move frames between a graph and the
//! outside world.
//!
//! Everything goes through one codec, [`y4m`] (YUV4MPEG2), which is generic
//! over `Read`/`Write`. The file nodes here use it on a `File`; the
//! [`ffmpeg`] nodes use it on the stdin/stdout pipes of an `ffmpeg` child
//! process, which buys every container and codec ffmpeg knows without linking
//! any of them.
//!
//! | kind            | ports            | params                               |
//! |-----------------|------------------|--------------------------------------|
//! | `y4m_source`    | → `out`          | `path`                               |
//! | `y4m_sink`      | `in` →           | `path`, `fps`?, `chroma`?            |
//! | `ffmpeg_source` | → `out`          | `path`                               |
//! | `ffmpeg_sink`   | `in` →           | `path`, `fps`?, `pix_fmt`?           |
//!
//! Sources emit 3-channel `u8` RGB frames and report
//! [`NodeError::EndOfStream`] once exhausted (drive them with
//! [`crate::exec::Runtime::run_until_eos`]). Sinks accept 3-channel RGB or
//! 1-channel grey `u8` frames and take the stream's size from the first frame.
//!
//! **Laziness.** Nothing touches the filesystem or spawns a process when a node
//! is *built*, only when it first evaluates. That matters because validation
//! builds a throwaway instance of every node just to ask for its ports
//! ([`crate::exec::Registry::ports_of`]); an eager sink would truncate its
//! output file and an eager ffmpeg source would launch a decoder for nothing.
//! Construction only does cheap checks (does the input file exist, is ffmpeg
//! on `PATH`) so a broken graph still fails at build time.
//!
//! **Finalization.** A sink finishes its output when it is reset
//! ([`crate::exec::Runtime::reset`]) or dropped (drop the `Runtime`). For
//! `ffmpeg_sink` that closes ffmpeg's stdin and waits for it to exit, which is
//! what writes an mp4's index — skipping it leaves an unplayable file.
//!
//! [`NodeError::EndOfStream`]: crate::exec::NodeError::EndOfStream

pub mod ffmpeg;
pub mod y4m;

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use crate::data::{Payload, PayloadKind};
use crate::exec::{
    BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec, Registry,
};
use crate::graph::Params;

pub use self::ffmpeg::{FfmpegSink, FfmpegSource, ffmpeg_available};
pub use self::y4m::{Chroma, Y4mError, Y4mHeader, Y4mReader, Y4mWriter};

/// Register the video I/O node kinds (called by
/// [`crate::exec::builtin_registry`]).
pub fn register_io(reg: &mut Registry) {
    reg.register("y4m_source", |p| {
        Ok(Box::new(Y4mSource::try_from(p)?) as Box<dyn Node>)
    });
    reg.register("y4m_sink", |p| {
        Ok(Box::new(Y4mSink::try_from(p)?) as Box<dyn Node>)
    });
    reg.register("ffmpeg_source", |p| {
        Ok(Box::new(FfmpegSource::try_from(p)?) as Box<dyn Node>)
    });
    reg.register("ffmpeg_sink", |p| {
        Ok(Box::new(FfmpegSink::try_from(p)?) as Box<dyn Node>)
    });
}

/// Parse a frame-rate parameter: `30`, `30000/1001` or `29.97`.
///
/// Returns `default` when the key is absent. Decimals become a `/1000`
/// rational, which is exact for the rates people actually type.
pub fn parse_fps(p: &Params, key: &str, default: (u32, u32)) -> Result<(u32, u32), BuildError> {
    let Some(raw) = p.get(key) else {
        return Ok(default);
    };
    let bad = || BuildError::BadParam {
        key: key.to_string(),
        value: raw.clone(),
        expected: "frame rate (e.g. 30, 30000/1001, 29.97)",
    };
    let fps = if let Some((n, d)) = raw.split_once('/') {
        (
            n.trim().parse().map_err(|_| bad())?,
            d.trim().parse().map_err(|_| bad())?,
        )
    } else if let Ok(n) = raw.trim().parse::<u32>() {
        (n, 1)
    } else {
        let f: f64 = raw.trim().parse().map_err(|_| bad())?;
        if !(f.is_finite() && f > 0.0 && f * 1000.0 <= u32::MAX as f64) {
            return Err(bad());
        }
        ((f * 1000.0).round() as u32, 1000)
    };
    if fps.0 == 0 || fps.1 == 0 {
        return Err(bad());
    }
    Ok(fps)
}

fn frame_out_ports() -> PortSet {
    PortSet::new(vec![], vec![PortSpec::new("out", PayloadKind::Frame)])
}

fn frame_in_ports() -> PortSet {
    PortSet::new(vec![PortSpec::new("in", PayloadKind::Frame)], vec![])
}

/// Reads frames from a `.y4m` file, one per evaluation.
pub struct Y4mSource {
    path: PathBuf,
    reader: Option<Y4mReader<File>>,
    /// Set once the file is exhausted, so later evaluations keep reporting
    /// end of stream instead of reopening (and looping) the file.
    exhausted: bool,
}

impl Y4mSource {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            reader: None,
            exhausted: false,
        }
    }

    fn fail(&self, e: impl std::fmt::Display) -> NodeError {
        NodeError::Message(format!("y4m_source '{}': {e}", self.path.display()))
    }
}

impl TryFrom<&Params> for Y4mSource {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        let path = PathBuf::from(p.get_str("path")?);
        // Cheap existence check so a typo fails at build time, not mid-run.
        if !path.is_file() {
            return Err(BuildError::Unavailable(format!(
                "y4m_source input '{}' is not a readable file",
                path.display()
            )));
        }
        Ok(Self::new(path))
    }
}

impl Node for Y4mSource {
    fn ports(&self) -> PortSet {
        frame_out_ports()
    }

    fn eval(&mut self, _inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        if self.exhausted {
            return Err(NodeError::EndOfStream);
        }
        if self.reader.is_none() {
            let file = File::open(&self.path).map_err(|e| self.fail(e))?;
            self.reader = Some(Y4mReader::new(file).map_err(|e| self.fail(e))?);
        }
        let reader = self.reader.as_mut().expect("opened above");
        match reader.read_rgb() {
            Ok(Some(frame)) => {
                outputs.set("out", Payload::Frame(frame));
                Ok(())
            }
            Ok(None) => {
                self.exhausted = true;
                self.reader = None;
                Err(NodeError::EndOfStream)
            }
            Err(e) => {
                // A corrupt stream cannot be resynchronized; stop here rather
                // than read garbage (or loop) on the next evaluation.
                self.exhausted = true;
                self.reader = None;
                Err(self.fail(e))
            }
        }
    }

    /// Rewind: the next evaluation reopens the file from the first frame.
    fn reset(&mut self) {
        self.reader = None;
        self.exhausted = false;
    }
}

/// Writes incoming frames to a `.y4m` file.
///
/// The file is created (truncating any existing one) on the first frame, and
/// its header is taken from that frame's size. Each frame is flushed as it is
/// written, so the file is always a valid y4m stream of the frames so far.
pub struct Y4mSink {
    path: PathBuf,
    fps: (u32, u32),
    /// `None` = choose from the first frame (1-channel → mono, else 4:2:0).
    chroma: Option<Chroma>,
    writer: Option<Y4mWriter<BufWriter<File>>>,
}

impl Y4mSink {
    pub fn new(path: impl Into<PathBuf>, fps: (u32, u32), chroma: Option<Chroma>) -> Self {
        Self {
            path: path.into(),
            fps,
            chroma,
            writer: None,
        }
    }

    fn fail(&self, e: impl std::fmt::Display) -> NodeError {
        NodeError::Message(format!("y4m_sink '{}': {e}", self.path.display()))
    }
}

impl TryFrom<&Params> for Y4mSink {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        let path = PathBuf::from(p.get_str("path")?);
        let fps = parse_fps(p, "fps", (25, 1))?;
        let chroma = match p.get("chroma").map(String::as_str) {
            None => None,
            Some("420") => Some(Chroma::C420),
            Some("444") => Some(Chroma::C444),
            Some("mono") => Some(Chroma::Mono),
            Some(other) => {
                return Err(BuildError::BadParam {
                    key: "chroma".to_string(),
                    value: other.to_string(),
                    expected: "420|444|mono",
                });
            }
        };
        Ok(Self::new(path, fps, chroma))
    }
}

impl Node for Y4mSink {
    fn ports(&self) -> PortSet {
        frame_in_ports()
    }

    fn eval(&mut self, inputs: &Inputs, _outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;
        if self.writer.is_none() {
            let chroma = self.chroma.unwrap_or(if frame.channels == 1 {
                Chroma::Mono
            } else {
                Chroma::C420
            });
            let header = Y4mHeader::new(frame.width, frame.height, self.fps, chroma);
            let file = File::create(&self.path).map_err(|e| self.fail(e))?;
            self.writer = Some(Y4mWriter::new(BufWriter::new(file), header));
        }
        let writer = self.writer.as_mut().expect("created above");
        let result = writer.write_frame(frame).and_then(|()| writer.flush());
        result.map_err(|e| self.fail(e))
    }

    /// Finish the current file; the next frame starts (and truncates) it anew.
    fn reset(&mut self) {
        if let Some(mut w) = self.writer.take() {
            let _ = w.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(kv: &[(&str, &str)]) -> Params {
        kv.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn nodes_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Y4mSource>();
        assert_send::<Y4mSink>();
        assert_send::<FfmpegSource>();
        assert_send::<FfmpegSink>();
    }

    #[test]
    fn fps_parsing() {
        let p = params(&[("a", "30"), ("b", "30000/1001"), ("c", "29.97"), ("d", "0")]);
        assert_eq!(parse_fps(&p, "a", (1, 1)).unwrap(), (30, 1));
        assert_eq!(parse_fps(&p, "b", (1, 1)).unwrap(), (30000, 1001));
        assert_eq!(parse_fps(&p, "c", (1, 1)).unwrap(), (29970, 1000));
        assert_eq!(parse_fps(&p, "missing", (25, 1)).unwrap(), (25, 1));
        assert!(matches!(
            parse_fps(&p, "d", (1, 1)),
            Err(BuildError::BadParam { .. })
        ));
    }

    #[test]
    fn source_requires_existing_file() {
        let err = Y4mSource::try_from(&params(&[("path", "/definitely/not/here.y4m")]))
            .err()
            .unwrap();
        assert!(matches!(err, BuildError::Unavailable(_)));
        assert_eq!(
            Y4mSource::try_from(&Params::new()).err().unwrap(),
            BuildError::MissingParam("path".to_string())
        );
    }

    #[test]
    fn building_a_sink_does_not_touch_the_filesystem() {
        let dir = std::env::temp_dir().join(format!("pipe-graph-lazy-{}", std::process::id()));
        let path = dir.join("never.y4m");
        let sink = Y4mSink::try_from(&params(&[("path", path.to_str().unwrap())])).unwrap();
        drop(sink);
        assert!(!path.exists());
    }

    #[test]
    fn sink_rejects_bad_chroma() {
        let err = Y4mSink::try_from(&params(&[("path", "x.y4m"), ("chroma", "422")]))
            .err()
            .unwrap();
        assert!(matches!(err, BuildError::BadParam { .. }));
    }
}
