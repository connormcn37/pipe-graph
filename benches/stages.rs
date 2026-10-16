//! Per-stage throughput: every builtin per-pixel stage evaluated directly.
//!
//! Each benchmark builds one node through [`builtin_registry`] (so params are
//! parsed exactly as a graph file would parse them) and times a single
//! [`Node::eval`] call on a 1920x1080 frame, with no scheduler involved. That
//! isolates the pixel loop from scheduling/fusion overhead, which
//! `benches/scheduler.rs` and `benches/pipeline.rs` cover instead.
//!
//! What a sample includes: `eval` allocating its output frame, and that frame
//! being dropped. Both are part of what a stage costs inside the runtime too
//! (the out-of-place path allocates per evaluation), and keeping every output
//! alive until the end of a sample (`iter_with_large_drop`) would hold
//! hundreds of megabytes per sample for no gain in accuracy.
//!
//! Throughput is reported in bytes of *input* frame data per second (for
//! `crop`, the bytes of the cropped region, since that is all it reads), so
//! `u8` and `f32` rows are comparable as memory bandwidth: an `f32` frame is
//! four times the bytes of the same `u8` image.
//!
//! Run with `cargo bench --bench stages`; filter with e.g.
//! `cargo bench --bench stages -- box_blur`.

mod common;

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pipe_graph::data::{Frame, FrameData, Payload};
use pipe_graph::exec::{Inputs, Node, Outputs, builtin_registry};
use pipe_graph::graph::PortId;

use common::{H, W, frame_f32, frame_u8, spec};

/// Which element type a case feeds the stage.
#[derive(Clone, Copy)]
enum Dtype {
    U8,
    F32,
}

impl Dtype {
    fn name(self) -> &'static str {
        match self {
            Dtype::U8 => "u8",
            Dtype::F32 => "f32",
        }
    }

    fn frame(self, channels: u32) -> Frame {
        match self {
            Dtype::U8 => frame_u8(channels),
            Dtype::F32 => frame_f32(channels),
        }
    }
}

/// Bytes of pixel data in `frame`, for [`Throughput::Bytes`].
fn frame_bytes(frame: &Frame) -> u64 {
    match frame.data() {
        FrameData::U8(v) => v.len() as u64,
        FrameData::F32(v) => (v.len() * size_of::<f32>()) as u64,
    }
}

/// Build the [`Inputs`] for one evaluation from `(port, frame)` pairs.
///
/// Kept in one place on purpose: it is the only spot in the benches that
/// touches how `Inputs` is constructed, so if that representation changes only
/// this helper needs updating.
fn inputs(ports: Vec<(String, Frame)>) -> Inputs {
    let map: HashMap<PortId, Payload> = ports
        .into_iter()
        .map(|(p, f)| (PortId(p), Payload::Frame(f)))
        .collect();
    Inputs::new(map)
}

/// Time `node.eval` on `inputs` as `group/<id>` with the given throughput.
fn bench_eval(
    c: &mut Criterion,
    group: &str,
    id: &str,
    mut node: Box<dyn Node>,
    inputs: Inputs,
    bytes: u64,
) {
    // Fail loudly up front (with the stage's own message) rather than
    // benchmarking an error path.
    node.eval(&inputs, &mut Outputs::new())
        .unwrap_or_else(|e| panic!("{group}/{id}: eval failed: {e}"));

    let mut g = c.benchmark_group(group);
    g.throughput(Throughput::Bytes(bytes));
    g.bench_function(BenchmarkId::from_parameter(id), |b| {
        b.iter(|| {
            let mut out = Outputs::new();
            node.eval(black_box(&inputs), &mut out).unwrap();
            black_box(out)
        })
    });
    g.finish();
}

/// A 1-in/1-out stage with ports `in`/`out`, fed a 3-channel frame.
fn bench_unary(c: &mut Criterion, group: &str, kind: &str, params: &[(&str, &str)], dt: Dtype) {
    let node = builtin_registry()
        .build(&spec("n", kind, params))
        .unwrap_or_else(|e| panic!("building {kind}: {e:?}"));
    let frame = dt.frame(3);
    let bytes = frame_bytes(&frame);
    bench_eval(
        c,
        group,
        dt.name(),
        node,
        inputs(vec![("in".into(), frame)]),
        bytes,
    );
}

fn per_pixel(c: &mut Criterion) {
    for dt in [Dtype::U8, Dtype::F32] {
        bench_unary(c, "gain", "gain", &[("factor", "1.1")], dt);
        bench_unary(c, "grayscale", "grayscale", &[], dt);
        bench_unary(c, "invert", "invert", &[], dt);
        bench_unary(c, "threshold", "threshold", &[("level", "0.5")], dt);
        bench_unary(c, "box_blur_r1", "box_blur", &[("radius", "1")], dt);
        bench_unary(c, "box_blur_r4", "box_blur", &[("radius", "4")], dt);
        bench_unary(
            c,
            "clear_channel",
            "clear_channel",
            &[("channel", "red")],
            dt,
        );
    }
}

fn crop(c: &mut Criterion) {
    // The centred half-size region: big enough to be a real copy, and offset
    // so every row copy starts mid-row rather than at the buffer start.
    let (x, y, w, h) = (W / 4, H / 4, W / 2, H / 2);
    let params = [
        ("x", x.to_string()),
        ("y", y.to_string()),
        ("w", w.to_string()),
        ("h", h.to_string()),
    ];
    let params: Vec<(&str, &str)> = params.iter().map(|(k, v)| (*k, v.as_str())).collect();
    for dt in [Dtype::U8, Dtype::F32] {
        let node = builtin_registry()
            .build(&spec("n", "crop", &params))
            .unwrap();
        let frame = dt.frame(3);
        // Bytes actually read: the cropped region, not the whole frame.
        let bytes = frame_bytes(&frame) * u64::from(w * h) / u64::from(W * H);
        bench_eval(
            c,
            "crop",
            dt.name(),
            node,
            inputs(vec![("in".into(), frame)]),
            bytes,
        );
    }
}

fn cast(c: &mut Criterion) {
    // A cast is only meaningful across dtypes, so the two cases are the two
    // directions rather than "u8" and "f32" of the same operation.
    for (id, src, target) in [
        ("u8_to_f32", Dtype::U8, "f32"),
        ("f32_to_u8", Dtype::F32, "u8"),
    ] {
        let node = builtin_registry()
            .build(&spec("n", "cast", &[("dtype", target)]))
            .unwrap();
        let frame = src.frame(3);
        let bytes = frame_bytes(&frame);
        bench_eval(
            c,
            "cast",
            id,
            node,
            inputs(vec![("in".into(), frame)]),
            bytes,
        );
    }
}

fn split_merge(c: &mut Criterion) {
    for dt in [Dtype::U8, Dtype::F32] {
        // split: one interleaved RGB frame -> three planes.
        let node = builtin_registry()
            .build(&spec("n", "split", &[("channels", "3")]))
            .unwrap();
        let frame = dt.frame(3);
        let bytes = frame_bytes(&frame);
        bench_eval(
            c,
            "split",
            dt.name(),
            node,
            inputs(vec![("in".into(), frame)]),
            bytes,
        );

        // merge: three planes -> one interleaved RGB frame. Same total bytes
        // as the split input, so the two rows are directly comparable.
        let node = builtin_registry()
            .build(&spec("n", "merge", &[("channels", "3")]))
            .unwrap();
        let planes: Vec<(String, Frame)> =
            (0..3).map(|i| (format!("in{i}"), dt.frame(1))).collect();
        let bytes = planes.iter().map(|(_, f)| frame_bytes(f)).sum();
        bench_eval(c, "merge", dt.name(), node, inputs(planes), bytes);
    }
}

/// Settings shared by every stage bench.
///
/// One eval is milliseconds, so 20 samples over ~2s gives stable medians while
/// keeping the whole file (~30 benchmarks) to roughly two minutes.
fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = stages;
    config = config();
    targets = per_pixel, crop, cast, split_merge
}
criterion_main!(stages);
