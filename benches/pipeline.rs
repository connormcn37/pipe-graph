//! End-to-end pipeline: a realistic 1080p graph through the full runtime.
//!
//! The graph is the one `examples/bench_pipeline.rs` times by hand:
//!
//! ```text
//!            ┌─ blur0 ─ gain0 ─┐
//!   split ───┼─ blur1 ─ gain1 ─┼─── merge
//!            └─ blur2 ─ gain2 ─┘
//! ```
//!
//! (box blur radius 4). The three branches are independent and individually
//! expensive, which is the case parallel mode exists for, and each
//! `blur -> gain` pair is a fusable run. So the four
//! `{serial, parallel} x {fusion off, on}` rows here show what each scheduler
//! feature buys on real work, where `benches/scheduler.rs` shows what it
//! costs on no work.
//!
//! Throughput is input frame bytes per second; one iteration is one frame.
//!
//! Run with `cargo bench --bench pipeline`.

mod common;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pipe_graph::data::Payload;
use pipe_graph::exec::{ExecMode, Runtime, builtin_registry};
use pipe_graph::graph::{Graph, NodeId};

use common::{H, W, frame_bytes, frame_f32, frame_u8, port, spec};

/// Blur radius for every branch, matching the example's default.
const RADIUS: &str = "4";

fn build_graph() -> Graph {
    let mut g = Graph::new();
    g.add_node(spec("split", "split", &[("channels", "3")]))
        .unwrap();
    g.add_node(spec("merge", "merge", &[("channels", "3")]))
        .unwrap();
    // Slightly different gain per channel so the branches do distinct work.
    let gains = ["1.10", "0.95", "1.25"];
    for (c, gain) in gains.iter().enumerate() {
        let blur = format!("blur{c}");
        let gn = format!("gain{c}");
        g.add_node(spec(&blur, "box_blur", &[("radius", RADIUS)]))
            .unwrap();
        g.add_node(spec(&gn, "gain", &[("factor", gain)])).unwrap();
        g.connect(port("split", &format!("out{c}")), port(&blur, "in"))
            .unwrap();
        g.connect(port(&blur, "out"), port(&gn, "in")).unwrap();
        g.connect(port(&gn, "out"), port("merge", &format!("in{c}")))
            .unwrap();
    }
    g
}

fn pipeline(c: &mut Criterion) {
    let reg = builtin_registry();
    let graph = build_graph();
    let configs = [
        ("serial", ExecMode::Serial, false),
        ("serial+fused", ExecMode::Serial, true),
        ("parallel", ExecMode::parallel(), false),
        ("parallel+fused", ExecMode::parallel(), true),
    ];

    for (dtype, input) in [("u8", frame_u8(3)), ("f32", frame_f32(3))] {
        let mut group = c.benchmark_group(format!("split_blur_gain_merge_{dtype}"));
        group.throughput(Throughput::Bytes(frame_bytes(&input)));
        for (label, mode, fusion) in configs {
            let mut rt = Runtime::instantiate(&graph, &reg).expect("graph should instantiate");
            rt.set_exec_mode(mode);
            rt.set_fusion(fusion);
            // The external input stays set across runs, so every iteration
            // processes the same frame from scratch.
            rt.set_input(&NodeId("split".into()), "in", Payload::Frame(input.clone()));

            // Untimed first run: pays one-off page faults and allocator
            // growth, and checks the graph actually produced a full frame.
            rt.run_once().expect("run_once failed");
            let out = rt
                .output(&NodeId("merge".into()), "out")
                .and_then(Payload::as_frame)
                .expect("merge produced no frame");
            assert_eq!((out.width, out.height, out.channels), (W, H, 3));

            group.bench_function(BenchmarkId::from_parameter(label), |b| {
                b.iter(|| black_box(&mut rt).run_once().unwrap())
            });
        }
        group.finish();
    }
}

/// A frame takes tens of milliseconds here, so criterion's minimum of 10
/// samples over ~4s is plenty; eight benchmarks then finish in about a minute.
fn config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4))
}

criterion_group! {
    name = benches;
    config = config();
    targets = pipeline
}
criterion_main!(benches);
