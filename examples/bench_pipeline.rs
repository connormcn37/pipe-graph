//! Benchmark: a realistic per-pixel workload for the scheduler.
//!
//! Builds a 1080p graph
//!
//! ```text
//!            ┌─ blur0 ─ gain0 ─┐
//!   split ───┼─ blur1 ─ gain1 ─┼─── merge
//!            └─ blur2 ─ gain2 ─┘
//! ```
//!
//! and times `run_once` per frame for both a `u8` and an `f32` RGB input.
//! The three branches are independent, so this is the shape that parallel
//! component execution and chain fusion are meant to speed up; run it before
//! and after those land to compare.
//!
//! Usage: `cargo run --release --example bench_pipeline [frames] [radius]`
//! (defaults: 30 frames, radius 4).

use std::time::{Duration, Instant};

use pipe_graph::data::{Frame, FrameData, Payload};
use pipe_graph::exec::{Runtime, builtin_registry};
use pipe_graph::graph::{Graph, NodeId, NodeSpec, Params, PortId};

const W: u32 = 1920;
const H: u32 = 1080;

fn node(id: &str, kind: &str, params: &[(&str, String)]) -> NodeSpec {
    let mut p = Params::new();
    for (k, v) in params {
        p.insert(k.to_string(), v.clone());
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

fn build_graph(radius: u32) -> Graph {
    let mut g = Graph::new();
    g.add_node(node("split", "split", &[("channels", "3".into())]))
        .unwrap();
    g.add_node(node("merge", "merge", &[("channels", "3".into())]))
        .unwrap();
    // Slightly different gain per channel so the branches do distinct work.
    let gains = ["1.10", "0.95", "1.25"];
    for (c, gain) in gains.iter().enumerate() {
        let blur = format!("blur{c}");
        let gn = format!("gain{c}");
        g.add_node(node(&blur, "box_blur", &[("radius", radius.to_string())]))
            .unwrap();
        g.add_node(node(&gn, "gain", &[("factor", gain.to_string())]))
            .unwrap();
        g.connect(port("split", &format!("out{c}")), port(&blur, "in"))
            .unwrap();
        g.connect(port(&blur, "out"), port(&gn, "in")).unwrap();
        g.connect(port(&gn, "out"), port("merge", &format!("in{c}")))
            .unwrap();
    }
    g
}

/// A deterministic, non-flat test pattern (gradients + a checker).
fn pattern_u8() -> Vec<u8> {
    let mut buf = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            let checker = if ((x / 64) + (y / 64)) % 2 == 0 {
                40
            } else {
                0
            };
            buf.push(((x * 255 / W) as u8).saturating_add(checker));
            buf.push((y * 255 / H) as u8);
            buf.push(((x + y) % 256) as u8);
        }
    }
    buf
}

fn bench(label: &str, rt: &mut Runtime, input: Frame, frames: u32) {
    rt.set_input(&NodeId("split".into()), "in", Payload::Frame(input));

    // Warm-up: first run pays for page faults / allocator growth.
    rt.run_once().expect("warm-up run failed");

    let mut times = Vec::with_capacity(frames as usize);
    for _ in 0..frames {
        let t = Instant::now();
        rt.run_once().expect("run failed");
        times.push(t.elapsed());
    }

    let out = rt
        .output(&NodeId("merge".into()), "out")
        .and_then(Payload::as_frame)
        .expect("merge produced no frame");
    assert_eq!((out.width, out.height, out.channels), (W, H, 3));

    times.sort();
    let total: Duration = times.iter().sum();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mean = ms(total) / frames as f64;
    println!(
        "{label:>4}: mean {mean:7.2} ms/frame  median {:7.2}  min {:7.2}  max {:7.2}  ({:.1} fps)",
        ms(times[times.len() / 2]),
        ms(times[0]),
        ms(times[times.len() - 1]),
        1e3 / mean,
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let frames: u32 = args
        .next()
        .map(|s| s.parse().expect("frames must be a positive integer"))
        .unwrap_or(30)
        .max(1);
    let radius: u32 = args
        .next()
        .map(|s| s.parse().expect("radius must be a non-negative integer"))
        .unwrap_or(4);

    println!(
        "bench_pipeline: {W}x{H} RGB, split -> 3x(box_blur r={radius} -> gain) -> merge, {frames} frames"
    );

    let reg = builtin_registry();
    let graph = build_graph(radius);

    let u8_buf = pattern_u8();
    let f32_buf: Vec<f32> = u8_buf.iter().map(|&v| v as f32 / 255.0).collect();

    let mut rt = Runtime::instantiate(&graph, &reg).expect("graph should instantiate");
    bench(
        "u8",
        &mut rt,
        Frame::from_data(W, H, 3, FrameData::U8(u8_buf)),
        frames,
    );

    let mut rt = Runtime::instantiate(&graph, &reg).expect("graph should instantiate");
    bench(
        "f32",
        &mut rt,
        Frame::from_data(W, H, 3, FrameData::F32(f32_buf)),
        frames,
    );
}
