//! Video I/O end to end: Y4M files and ffmpeg pipes driven by
//! `Runtime::run_until_eos`.
//!
//! The Y4M tests write their fixtures from Rust and need nothing installed.
//! The ffmpeg tests skip (with a note on stderr) when `ffmpeg`/`ffprobe` are
//! not on PATH, since CI runners may not have them.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use pipe_graph::data::Frame;
use pipe_graph::exec::{BuildError, Runtime, ScheduleError, ValidationError, builtin_registry};
use pipe_graph::graph::{Graph, NodeId, NodeSpec, Params, PortId};
use pipe_graph::io::{Chroma, Y4mHeader, Y4mReader, Y4mWriter, ffmpeg_available};

fn node(id: &str, kind: &str, params: &[(&str, &str)]) -> NodeSpec {
    let mut p = Params::new();
    for (k, v) in params {
        p.insert(k.to_string(), v.to_string());
    }
    NodeSpec {
        id: NodeId(id.to_string()),
        kind: kind.to_string(),
        params: p,
    }
}

fn port(node: &str, port: &str) -> (NodeId, PortId) {
    (NodeId(node.to_string()), PortId(port.to_string()))
}

/// A fresh, empty scratch directory unique to this test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pipe-graph-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// Frame `i` of the fixture: flat colour (so 4:2:0 is lossless up to
/// rounding) that changes per frame, so frame order is checkable.
fn fixture_colour(i: u32) -> (u8, u8, u8) {
    (200, (i * 20) as u8, 255 - (i * 20) as u8)
}

fn write_fixture(path: &Path, frames: u32, w: u32, h: u32) {
    let file = File::create(path).unwrap();
    let mut wr = Y4mWriter::new(file, Y4mHeader::new(w, h, (10, 1), Chroma::C420));
    for i in 0..frames {
        let px = vec![fixture_colour(i); (w * h) as usize];
        wr.write_frame(&Frame::from_rgb8(w, h, px)).unwrap();
    }
    wr.into_inner().unwrap();
}

fn read_all(path: &Path) -> (Y4mHeader, Vec<Frame>) {
    let mut r = Y4mReader::new(File::open(path).unwrap()).unwrap();
    let mut frames = Vec::new();
    while let Some(f) = r.read_rgb().unwrap() {
        frames.push(f);
    }
    (r.header().clone(), frames)
}

fn close(a: (u8, u8, u8), b: (u8, u8, u8), tol: u8) -> bool {
    a.0.abs_diff(b.0) <= tol && a.1.abs_diff(b.1) <= tol && a.2.abs_diff(b.2) <= tol
}

/// `src -> clear_channel(red) -> sink`.
fn clear_red_graph(src_kind: &str, input: &Path, sink_kind: &str, output: &Path) -> Graph {
    let mut g = Graph::new();
    g.add_node(node("src", src_kind, &[("path", s(input))]))
        .unwrap();
    g.add_node(node("clear", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node(
        "sink",
        sink_kind,
        &[("path", s(output)), ("fps", "10")],
    ))
    .unwrap();
    g.connect(port("src", "out"), port("clear", "in")).unwrap();
    g.connect(port("clear", "out"), port("sink", "in")).unwrap();
    g
}

#[test]
fn y4m_pipeline_runs_to_end_of_stream() {
    let dir = scratch("y4m-pipeline");
    let (input, output) = (dir.join("in.y4m"), dir.join("out.y4m"));
    write_fixture(&input, 6, 9, 5);

    let g = clear_red_graph("y4m_source", &input, "y4m_sink", &output);
    let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
    // Building (twice: validation + instantiation) must not create the output.
    assert!(!output.exists());

    assert_eq!(rt.run_until_eos(1000).unwrap(), 6);
    // Exhausted sources stay exhausted.
    assert_eq!(rt.run_until_eos(1000).unwrap(), 0);
    let err = rt.run_once().unwrap_err();
    assert!(err.is_end_of_stream(), "{err}");
    drop(rt);

    let (header, frames) = read_all(&output);
    assert_eq!((header.width, header.height), (9, 5));
    assert_eq!((header.fps_num, header.fps_den), (10, 1));
    assert_eq!(frames.len(), 6);
    for (i, f) in frames.iter().enumerate() {
        let (_, g, b) = fixture_colour(i as u32);
        for px in f.to_rgb8() {
            assert!(close(px, (0, g, b), 4), "frame {i}: {px:?}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn run_until_eos_respects_max_and_reset_rewinds() {
    let dir = scratch("y4m-max");
    let (input, output) = (dir.join("in.y4m"), dir.join("out.y4m"));
    write_fixture(&input, 5, 4, 4);

    let g = clear_red_graph("y4m_source", &input, "y4m_sink", &output);
    let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
    assert_eq!(rt.run_until_eos(2).unwrap(), 2);
    assert_eq!(rt.run_until_eos(100).unwrap(), 3);

    // Reset rewinds the source and finishes the sink's file; the next run
    // rewrites it from the first frame.
    rt.reset();
    assert_eq!(rt.run_until_eos(100).unwrap(), 5);
    drop(rt);
    assert_eq!(read_all(&output).1.len(), 5);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn shortest_source_ends_the_run() {
    // Two independent branches of different lengths. The run stops when the
    // shorter one is exhausted; the longer branch may have processed one
    // extra frame in the abandoned final pass if it is ordered first (see
    // `Runtime::run_until_eos`).
    let dir = scratch("y4m-two-sources");
    let (a_in, b_in) = (dir.join("a.y4m"), dir.join("b.y4m"));
    let (a_out, b_out) = (dir.join("a_out.y4m"), dir.join("b_out.y4m"));
    write_fixture(&a_in, 3, 2, 2);
    write_fixture(&b_in, 5, 2, 2);

    let mut g = Graph::new();
    for (id, path) in [("a", &a_in), ("b", &b_in)] {
        g.add_node(node(id, "y4m_source", &[("path", s(path))]))
            .unwrap();
    }
    for (id, path) in [("a_sink", &a_out), ("b_sink", &b_out)] {
        g.add_node(node(id, "y4m_sink", &[("path", s(path))]))
            .unwrap();
    }
    g.connect(port("a", "out"), port("a_sink", "in")).unwrap();
    g.connect(port("b", "out"), port("b_sink", "in")).unwrap();

    let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
    assert_eq!(rt.run_until_eos(100).unwrap(), 3);
    drop(rt);
    assert_eq!(read_all(&a_out).1.len(), 3);
    let b_frames = read_all(&b_out).1.len();
    assert!(b_frames == 3 || b_frames == 4, "{b_frames}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn missing_input_file_fails_at_build_time() {
    let mut g = Graph::new();
    g.add_node(node(
        "src",
        "y4m_source",
        &[("path", "/definitely/not/here.y4m")],
    ))
    .unwrap();
    let err = Runtime::instantiate(&g, &builtin_registry()).err().unwrap();
    assert!(
        matches!(
            err,
            ScheduleError::Validation(ValidationError::Build {
                error: BuildError::Unavailable(_),
                ..
            })
        ),
        "{err}"
    );
}

// --- ffmpeg ------------------------------------------------------------------

fn have_ffmpeg() -> bool {
    let probe = Command::new("ffprobe").arg("-version").output();
    if ffmpeg_available() && probe.is_ok_and(|o| o.status.success()) {
        true
    } else {
        eprintln!("skipping: ffmpeg/ffprobe not on PATH");
        false
    }
}

/// `(width, height, frame count)` of the first video stream, per ffprobe.
fn probe(path: &Path) -> (u32, u32, u32) {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-count_frames", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height,nb_read_frames"])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    let field = |k: &str| -> u32 {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .unwrap_or_else(|| panic!("no {k} in {text}"))
            .trim()
            .parse()
            .unwrap()
    };
    (field("width"), field("height"), field("nb_read_frames"))
}

/// ffmpeg's `testsrc` pattern, 10 frames of 64x48, as 4:2:0 y4m.
fn testsrc(path: &Path) {
    let status = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y", "-f", "lavfi"])
        .args(["-i", "testsrc=size=64x48:rate=10", "-frames:v", "10"])
        .args(["-pix_fmt", "yuv420p"])
        .arg(path)
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn ffmpeg_generated_y4m_through_rust_pipeline() {
    if !have_ffmpeg() {
        return;
    }
    let dir = scratch("ffmpeg-y4m");
    let (input, output) = (dir.join("in.y4m"), dir.join("out.y4m"));
    testsrc(&input);

    let g = clear_red_graph("y4m_source", &input, "y4m_sink", &output);
    let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
    assert_eq!(rt.run_until_eos(1000).unwrap(), 10);
    drop(rt);

    // ffmpeg agrees on what we wrote.
    assert_eq!(probe(&output), (64, 48, 10));

    // Spot-check: (40, 30) lies inside testsrc's flat green bar on frame 0.
    let (_, before) = read_all(&input);
    let (_, after) = read_all(&output);
    let at = |f: &Frame| f.to_rgb8()[30 * 64 + 40];
    assert!(
        close(at(&before[0]), (0, 255, 0), 4),
        "{:?}",
        at(&before[0])
    );
    assert!(close(at(&after[0]), (0, 255, 0), 4), "{:?}", at(&after[0]));
    // And a magenta pixel (20, 10) loses its red.
    let px_in = before[0].to_rgb8()[10 * 64 + 20];
    let px_out = after[0].to_rgb8()[10 * 64 + 20];
    assert!(close(px_in, (255, 0, 255), 4), "{px_in:?}");
    assert!(close(px_out, (0, 0, 255), 6), "{px_out:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ffmpeg_source_to_ffmpeg_sink_writes_a_playable_mp4() {
    if !have_ffmpeg() {
        return;
    }
    let dir = scratch("ffmpeg-mp4");
    let (input, output) = (dir.join("in.y4m"), dir.join("out.mp4"));
    testsrc(&input);

    let g = clear_red_graph("ffmpeg_source", &input, "ffmpeg_sink", &output);
    let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
    assert_eq!(rt.run_until_eos(1000).unwrap(), 10);
    // Dropping the runtime closes ffmpeg's stdin and waits for it to write
    // the mp4 index; probing before this would see a truncated file.
    drop(rt);

    assert_eq!(probe(&output), (64, 48, 10));

    // Read it back through ffmpeg_source as well.
    let mut g = Graph::new();
    g.add_node(node("src", "ffmpeg_source", &[("path", s(&output))]))
        .unwrap();
    g.add_node(node(
        "sink",
        "y4m_sink",
        &[("path", s(&dir.join("back.y4m")))],
    ))
    .unwrap();
    g.connect(port("src", "out"), port("sink", "in")).unwrap();
    let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
    assert_eq!(rt.run_until_eos(1000).unwrap(), 10);
    drop(rt);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ffmpeg_source_reports_unreadable_input() {
    if !have_ffmpeg() {
        return;
    }
    let mut g = Graph::new();
    g.add_node(node(
        "src",
        "ffmpeg_source",
        &[("path", "/definitely/not/here.mp4")],
    ))
    .unwrap();
    let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
    let err = rt.run_until_eos(10).unwrap_err();
    assert!(!err.is_end_of_stream());
    assert!(err.to_string().contains("No such file"), "{err}");
}
