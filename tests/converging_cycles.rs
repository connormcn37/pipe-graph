//! Feedback loops stop iterating once they converge.
//!
//! A cyclic component runs Jacobi iterations until an iteration reproduces the
//! previous one's outputs (within [`Convergence`]), with `max_iters` as a hard
//! cap. These tests pin the iteration counts via [`Runtime::last_iterations_of`]
//! in serial and parallel mode, with and without fusion.

use pipe_graph::data::{Frame, FrameData, Payload, PayloadKind};
use pipe_graph::exec::{
    Convergence, DEFAULT_MAX_ITERS, ExecMode, Inputs, Node, NodeError, Outputs, PortSet, PortSpec,
    Registry, Runtime, builtin_registry,
};
use pipe_graph::graph::{Graph, NodeId, NodeSpec, Params, PortId};

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

fn id(s: &str) -> NodeId {
    NodeId(s.to_string())
}

/// `out = min(prev + 1, cap)` on a scalar self-loop: from empty feedback it
/// produces 1, 2, ..., cap, cap — settling after `cap + 1` iterations.
struct Saturating {
    cap: f64,
}

impl Node for Saturating {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("prev", PayloadKind::Scalar)],
            vec![PortSpec::new("out", PayloadKind::Scalar)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let prev = inputs
            .get("prev")
            .and_then(Payload::as_scalar)
            .unwrap_or(0.0);
        outputs.set("out", Payload::Scalar((prev + 1.0).min(self.cap)));
        Ok(())
    }
}

/// `out = prev + 1`: never settles.
struct Counter;

impl Node for Counter {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("prev", PayloadKind::Scalar)],
            vec![PortSpec::new("out", PayloadKind::Scalar)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let prev = inputs
            .get("prev")
            .and_then(Payload::as_scalar)
            .unwrap_or(0.0);
        outputs.set("out", Payload::Scalar(prev + 1.0));
        Ok(())
    }
}

/// `out = prev / 2 + 1`: approaches 2 asymptotically, never exactly reaching
/// it in a handful of iterations (the gap halves each time).
struct HalfStep;

impl Node for HalfStep {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("prev", PayloadKind::Scalar)],
            vec![PortSpec::new("out", PayloadKind::Scalar)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let prev = inputs
            .get("prev")
            .and_then(Payload::as_scalar)
            .unwrap_or(0.0);
        outputs.set("out", Payload::Scalar(prev / 2.0 + 1.0));
        Ok(())
    }
}

/// Frame analogue of [`HalfStep`]: every element moves halfway towards `1.0`.
/// Starts from a 2x1 single-channel zero frame when there is no feedback yet.
struct FrameHalfStep;

impl Node for FrameHalfStep {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("prev", PayloadKind::Frame)],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let prev: Vec<f32> = inputs
            .get("prev")
            .and_then(Payload::as_frame)
            .and_then(Frame::as_f32)
            .map(<[f32]>::to_vec)
            .unwrap_or_else(|| vec![0.0, 0.0]);
        let next = prev.iter().map(|v| v + (1.0 - v) / 2.0).collect();
        outputs.set(
            "out",
            Payload::Frame(Frame::from_data(2, 1, 1, FrameData::F32(next))),
        );
        Ok(())
    }
}

fn registry() -> Registry {
    let mut reg = builtin_registry();
    reg.register("saturating", |_| {
        Ok(Box::new(Saturating { cap: 2.0 }) as Box<dyn Node>)
    });
    reg.register("counter", |_| Ok(Box::new(Counter) as Box<dyn Node>));
    reg.register("half_step", |_| Ok(Box::new(HalfStep) as Box<dyn Node>));
    reg.register("frame_half_step", |_| {
        Ok(Box::new(FrameHalfStep) as Box<dyn Node>)
    });
    reg
}

fn self_loop(kind: &str) -> Runtime {
    let mut g = Graph::new();
    g.add_node(node("k", kind, &[])).unwrap();
    g.connect(port("k", "out"), port("k", "prev")).unwrap();
    Runtime::instantiate(&g, &registry()).unwrap()
}

#[test]
fn settling_loop_stops_after_three_iterations() {
    // 1, 2, 2: the third iteration reproduces the second, so it stops there.
    let mut rt = self_loop("saturating");
    rt.run_once().unwrap();
    assert_eq!(rt.output(&id("k"), "out").unwrap().as_scalar(), Some(2.0));
    assert_eq!(rt.last_iterations_of(&id("k")), Some(3));

    // Next pass starts from the settled buffer: 2, 2 — two iterations, the
    // minimum needed to see a fixed point.
    rt.run_once().unwrap();
    assert_eq!(rt.last_iterations_of(&id("k")), Some(2));
    assert_eq!(rt.output(&id("k"), "out").unwrap().as_scalar(), Some(2.0));

    // reset clears the diagnostics along with the buffers.
    rt.reset();
    assert_eq!(rt.last_iterations_of(&id("k")), Some(0));
    assert_eq!(rt.last_iterations_of(&id("nope")), None);
}

#[test]
fn never_settling_loop_runs_to_max_iters() {
    let mut rt = self_loop("counter");
    rt.run_once().unwrap();
    assert_eq!(rt.last_iterations_of(&id("k")), Some(DEFAULT_MAX_ITERS));
    assert_eq!(
        rt.output(&id("k"), "out").unwrap().as_scalar(),
        Some(f64::from(DEFAULT_MAX_ITERS))
    );

    rt.set_max_iters(4);
    rt.run_once().unwrap();
    assert_eq!(rt.last_iterations_of(&id("k")), Some(4));
}

#[test]
fn convergence_off_always_runs_max_iters() {
    let mut rt = self_loop("saturating");
    rt.set_convergence(Convergence::OFF);
    rt.run_once().unwrap();
    assert_eq!(rt.last_iterations_of(&id("k")), Some(DEFAULT_MAX_ITERS));
    assert_eq!(rt.output(&id("k"), "out").unwrap().as_scalar(), Some(2.0));
}

#[test]
fn max_iters_still_caps_a_converging_loop() {
    let mut rt = self_loop("saturating");
    rt.set_max_iters(2);
    rt.run_once().unwrap();
    // 1, 2: capped before the repeat that would have revealed convergence.
    assert_eq!(rt.last_iterations_of(&id("k")), Some(2));
    assert_eq!(rt.output(&id("k"), "out").unwrap().as_scalar(), Some(2.0));
}

#[test]
fn scalar_tolerance_stops_an_asymptotic_loop_early() {
    // Exact: 1, 1.5, 1.75, ... needs dozens of iterations to hit 2.0 in f64,
    // so it runs to the cap.
    let mut exact = self_loop("half_step");
    exact.run_once().unwrap();
    assert_eq!(exact.last_iterations_of(&id("k")), Some(DEFAULT_MAX_ITERS));

    // Successive differences are 0.5, 0.25, 0.125, 0.0625, ...: with
    // eps = 0.1 the 4th iteration (diff 0.125) still differs, the 5th
    // (diff 0.0625) is within tolerance.
    let mut loose = self_loop("half_step");
    loose.set_convergence(Convergence {
        scalar_eps: 0.1,
        ..Convergence::EXACT
    });
    loose.run_once().unwrap();
    assert_eq!(loose.last_iterations_of(&id("k")), Some(5));
    let out = loose.output(&id("k"), "out").unwrap().as_scalar().unwrap();
    assert!((out - 1.9375).abs() < 1e-12, "{out}");
}

#[test]
fn frame_tolerance_stops_an_asymptotic_loop_early() {
    let mut exact = self_loop("frame_half_step");
    exact.run_once().unwrap();
    assert_eq!(exact.last_iterations_of(&id("k")), Some(DEFAULT_MAX_ITERS));

    // Per element: 0.5, 0.75, 0.875, 0.9375 (gap to 1.0 halves each time).
    // Successive differences 0.25, 0.125, 0.0625: the 4th iteration is the
    // first within eps = 0.1.
    let mut loose = self_loop("frame_half_step");
    loose.set_convergence(Convergence {
        frame_eps: 0.1,
        ..Convergence::EXACT
    });
    loose.run_once().unwrap();
    assert_eq!(loose.last_iterations_of(&id("k")), Some(4));
}

/// A chain feeding a two-node loop, plus an independent counter loop:
///
/// ```text
/// a (clear red) -> b (clear green) -> c (clear blue)   [fusable chain]
/// x <-> y  (saturating, 2-cycle)
/// k (counter, self-loop)
/// ```
fn mixed() -> Runtime {
    let mut g = Graph::new();
    g.add_node(node("a", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node("b", "clear_channel", &[("channel", "green")]))
        .unwrap();
    g.add_node(node("c", "clear_channel", &[("channel", "blue")]))
        .unwrap();
    g.add_node(node("x", "saturating", &[])).unwrap();
    g.add_node(node("y", "saturating", &[])).unwrap();
    g.add_node(node("k", "counter", &[])).unwrap();
    g.connect(port("a", "out"), port("b", "in")).unwrap();
    g.connect(port("b", "out"), port("c", "in")).unwrap();
    g.connect(port("x", "out"), port("y", "prev")).unwrap();
    g.connect(port("y", "out"), port("x", "prev")).unwrap();
    g.connect(port("k", "out"), port("k", "prev")).unwrap();
    let mut rt = Runtime::instantiate(&g, &registry()).unwrap();
    rt.set_input(
        &id("a"),
        "in",
        Payload::Frame(Frame::from_rgb8(1, 1, vec![(9, 9, 9)])),
    );
    rt
}

#[test]
fn convergence_is_identical_across_exec_modes_and_fusion() {
    let mut reference = None;
    for mode in [
        ExecMode::Serial,
        ExecMode::Parallel { threads: 1 },
        ExecMode::Parallel { threads: 4 },
    ] {
        for fusion in [true, false] {
            let mut rt = mixed();
            rt.set_exec_mode(mode);
            rt.set_fusion(fusion);
            rt.run_once().unwrap();

            // Jacobi on the 2-cycle: (x, y) = (1, 1), (2, 2), (2, 2).
            assert_eq!(rt.last_iterations_of(&id("x")), Some(3), "{mode:?}");
            assert_eq!(rt.last_iterations_of(&id("y")), Some(3), "{mode:?}");
            assert_eq!(
                rt.last_iterations_of(&id("k")),
                Some(DEFAULT_MAX_ITERS),
                "{mode:?}"
            );
            // Acyclic components report 0.
            assert_eq!(rt.last_iterations_of(&id("a")), Some(0));

            let snapshot = (
                rt.last_iterations().to_vec(),
                rt.output(&id("y"), "out").and_then(Payload::as_scalar),
                rt.output(&id("k"), "out").and_then(Payload::as_scalar),
                rt.output(&id("c"), "out")
                    .and_then(Payload::as_frame)
                    .cloned(),
            );
            assert_eq!(snapshot.1, Some(2.0));
            match &reference {
                None => reference = Some(snapshot),
                Some(r) => assert_eq!(&snapshot, r, "{mode:?}, fusion {fusion}"),
            }
        }
    }
}
