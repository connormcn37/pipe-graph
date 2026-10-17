//! GPU placement (H2): the scheduler runs GPU-capable nodes on a device,
//! moves frames across only where the graph crosses between CPU and GPU, and
//! submits a pass's GPU work at once — with the same results as the CPU.
#![cfg(feature = "gpu")]

use pipe_graph::data::{Frame, FrameData, Location, Meta, Payload};
use pipe_graph::exec::{Backend, ExecMode, Runtime, builtin_registry};
use pipe_graph::gpu::GpuContext;
use pipe_graph::graph::{Graph, NodeId, NodeSpec, Params, PortId};

/// A context, or `None` (test skipped) without an adapter — unless
/// `PIPE_GRAPH_REQUIRE_GPU` is set, as in CI.
fn context() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) if std::env::var_os("PIPE_GRAPH_REQUIRE_GPU").is_some() => {
            panic!("PIPE_GRAPH_REQUIRE_GPU is set but {e}")
        }
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn node(id: &str, kind: &str, params: &[(&str, &str)]) -> NodeSpec {
    NodeSpec {
        id: NodeId(id.into()),
        kind: kind.into(),
        params: params
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<Params>(),
    }
}

fn port(n: &str, p: &str) -> (NodeId, PortId) {
    (NodeId(n.into()), PortId(p.into()))
}

fn id(s: &str) -> NodeId {
    NodeId(s.into())
}

/// `a` invert → `b` clear_channel → `c` invert → `d` invert → `e` gain.
///
/// `clear_channel` has no GPU path, so this crosses CPU/GPU in both
/// directions, includes a lone GPU-capable node (`a`) and a GPU run (`c`,
/// `d`, `e`), and puts `clear_channel` — which edits frames in place when
/// fused — right after a GPU node.
fn mixed_chain() -> Graph {
    let mut g = Graph::new();
    g.add_node(node("a", "invert", &[])).unwrap();
    g.add_node(node("b", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node("c", "invert", &[])).unwrap();
    g.add_node(node("d", "invert", &[])).unwrap();
    g.add_node(node("e", "gain", &[("factor", "0.5")])).unwrap();
    for (from, to) in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")] {
        g.connect(port(from, "out"), port(to, "in")).unwrap();
    }
    g
}

/// 7x5 RGB, so neither the byte count nor the pixel count is a whole number
/// of GPU words or workgroups.
fn input() -> Frame {
    Frame::from_data(7, 5, 3, FrameData::U8((0..105).map(|v| v * 2).collect())).with_meta(Meta {
        frame_index: Some(4),
        ..Meta::default()
    })
}

fn run(g: &Graph, backend: Backend, mode: ExecMode, fusion: bool) -> Runtime {
    let mut rt = Runtime::instantiate(g, &builtin_registry()).unwrap();
    rt.set_backend(backend);
    rt.set_exec_mode(mode);
    rt.set_fusion(fusion);
    for n in ["a", "b", "c", "d"] {
        rt.watch(&id(n), "out");
    }
    rt.set_input(&id("a"), "in", Payload::Frame(input()));
    rt.run_once().unwrap();
    rt
}

fn output_on_cpu(rt: &Runtime, n: &str) -> Frame {
    rt.output(&id(n), "out")
        .unwrap()
        .as_frame()
        .unwrap()
        .clone()
        .into_cpu()
        .unwrap()
}

#[test]
fn every_backend_mode_and_fusion_setting_agrees_with_the_cpu() {
    let Some(ctx) = context() else { return };
    let g = mixed_chain();
    let reference = run(&g, Backend::Cpu, ExecMode::Serial, false);

    for backend in [Backend::Gpu(ctx.clone()), Backend::Auto(ctx.clone())] {
        for mode in [ExecMode::Serial, ExecMode::Parallel { threads: 2 }] {
            for fusion in [false, true] {
                let rt = run(&g, backend.clone(), mode, fusion);
                for n in ["a", "b", "c", "d", "e"] {
                    assert_eq!(
                        output_on_cpu(&rt, n),
                        output_on_cpu(&reference, n),
                        "{n} under {backend:?} {mode:?} fusion={fusion}"
                    );
                }
            }
        }
    }
}

#[test]
fn placement_follows_the_policy_and_frames_follow_placement() {
    let Some(ctx) = context() else { return };
    let g = mixed_chain();
    let at = |rt: &Runtime, n: &str| rt.placement(&id(n)).unwrap();
    let lives = |rt: &Runtime, n: &str| {
        rt.output(&id(n), "out")
            .unwrap()
            .as_frame()
            .unwrap()
            .location()
    };

    // Everything GPU-capable on the GPU; outputs stay where they were made.
    let rt = run(&g, Backend::Gpu(ctx.clone()), ExecMode::Serial, true);
    let expected = [
        ("a", Location::Gpu),
        ("b", Location::Cpu),
        ("c", Location::Gpu),
        ("d", Location::Gpu),
        ("e", Location::Gpu),
    ];
    for (n, loc) in expected {
        assert_eq!(at(&rt, n), loc, "placement of {n}");
        assert_eq!(lives(&rt, n), loc, "output of {n}");
    }

    // Auto leaves the lone invert `a` on the CPU, moves the `c`→`d`→`e` run.
    let rt = run(&g, Backend::Auto(ctx.clone()), ExecMode::Serial, true);
    for (n, loc) in [
        ("a", Location::Cpu),
        ("b", Location::Cpu),
        ("c", Location::Gpu),
        ("d", Location::Gpu),
        ("e", Location::Gpu),
    ] {
        assert_eq!(at(&rt, n), loc, "auto placement of {n}");
    }

    // And back: switching to the CPU backend moves everything home.
    let mut rt = run(&g, Backend::Gpu(ctx), ExecMode::Serial, true);
    rt.set_backend(Backend::Cpu);
    rt.run_once().unwrap();
    for n in ["a", "c", "d", "e"] {
        assert_eq!((at(&rt, n), lives(&rt, n)), (Location::Cpu, Location::Cpu));
    }
    assert_eq!(rt.placement(&id("nope")), None);
}

#[test]
fn a_gpu_chain_is_one_submission_per_pass() {
    let Some(ctx) = context() else { return };
    let mut g = Graph::new();
    for n in ["i1", "i2", "i3", "i4"] {
        g.add_node(node(n, "invert", &[])).unwrap();
    }
    for (from, to) in [("i1", "i2"), ("i2", "i3"), ("i3", "i4")] {
        g.connect(port(from, "out"), port(to, "in")).unwrap();
    }
    let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
    rt.set_backend(Backend::Gpu(ctx.clone()));
    rt.set_input(&id("i1"), "in", Payload::Frame(input()));

    let before = ctx.submissions();
    rt.tick(3).unwrap();
    assert_eq!(
        ctx.submissions() - before,
        3,
        "4 kernels per pass, 1 submit"
    );

    // Four inversions: the input back again, and reading it is one more
    // submission (the copy to a mappable buffer).
    let out = output_on_cpu(&rt, "i4");
    assert_eq!(out, input());
    assert_eq!(ctx.submissions() - before, 4);
}

#[test]
fn metadata_survives_the_gpu_path() {
    let Some(ctx) = context() else { return };
    let rt = run(&mixed_chain(), Backend::Gpu(ctx), ExecMode::Serial, true);
    for n in ["a", "b", "c", "d", "e"] {
        let out = rt.output(&id(n), "out").unwrap();
        assert_eq!(out.meta().unwrap().frame_index, Some(4), "{n}");
    }
}

#[test]
fn a_cpu_stage_editing_in_place_gets_a_downloaded_frame() {
    // invert (GPU) → clear_channel (CPU, fused, edits its input in place):
    // the GPU frame it is handed is uniquely owned, so the in-place path is
    // taken — and must download before touching samples.
    let Some(ctx) = context() else { return };
    let mut g = Graph::new();
    g.add_node(node("inv", "invert", &[])).unwrap();
    g.add_node(node("clr", "clear_channel", &[("channel", "green")]))
        .unwrap();
    g.connect(port("inv", "out"), port("clr", "in")).unwrap();

    let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
    rt.set_backend(Backend::Gpu(ctx));
    assert!(rt.fusion_enabled());
    rt.set_input(
        &id("inv"),
        "in",
        Payload::Frame(Frame::from_rgb8(2, 1, vec![(0, 0, 0), (10, 20, 30)])),
    );
    rt.run_once().unwrap();
    let out = rt.output(&id("clr"), "out").unwrap().as_frame().unwrap();
    assert_eq!(out.location(), Location::Cpu);
    assert_eq!(out.to_rgb8(), vec![(255, 0, 255), (245, 0, 225)]);
}
