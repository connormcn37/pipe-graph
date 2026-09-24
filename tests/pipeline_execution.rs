//! End-to-end execution tests: build a graph, compile + run it, assert outputs.
//!
//! This is the headline milestone — an arbitrary graph runs to a hand-computed
//! result, including fan-out, multi-port split/merge, and a feedback loop.

use pipe_graph::data::{Frame, FrameData, Payload, PayloadKind};
use pipe_graph::exec::{
    Inputs, Node, NodeError, Outputs, PortSet, PortSpec, Registry, Runtime, builtin_registry,
};
use pipe_graph::graph::{Graph, NodeId, NodeSpec, Params, PortId};
use std::sync::Arc;

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

#[test]
fn linear_chain_clears_all_channels() {
    // a: clear red -> b: clear green -> c: clear blue.
    let mut g = Graph::new();
    g.add_node(node("a", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node("b", "clear_channel", &[("channel", "green")]))
        .unwrap();
    g.add_node(node("c", "clear_channel", &[("channel", "blue")]))
        .unwrap();
    g.connect(port("a", "out"), port("b", "in")).unwrap();
    g.connect(port("b", "out"), port("c", "in")).unwrap();

    let reg = builtin_registry();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_input(
        &id("a"),
        "in",
        Payload::Frame(Frame::from_rgb8(1, 1, vec![(255, 255, 255)])),
    );
    rt.run_once().unwrap();

    let out = rt.output(&id("c"), "out").unwrap().as_frame().unwrap();
    assert_eq!(out.to_rgb8(), vec![(0, 0, 0)]);
}

#[test]
fn fan_out_feeds_two_consumers() {
    // h: clear red; h.out fans out to a (clear green) and b (clear blue).
    let mut g = Graph::new();
    g.add_node(node("h", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node("a", "clear_channel", &[("channel", "green")]))
        .unwrap();
    g.add_node(node("b", "clear_channel", &[("channel", "blue")]))
        .unwrap();
    g.connect(port("h", "out"), port("a", "in")).unwrap();
    g.connect(port("h", "out"), port("b", "in")).unwrap();

    let reg = builtin_registry();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_input(
        &id("h"),
        "in",
        Payload::Frame(Frame::from_rgb8(1, 1, vec![(255, 255, 255)])),
    );
    rt.run_once().unwrap();

    // h -> (0,255,255); a also clears green -> (0,0,255); b also clears blue -> (0,255,0).
    assert_eq!(
        rt.output(&id("a"), "out")
            .unwrap()
            .as_frame()
            .unwrap()
            .to_rgb8(),
        vec![(0, 0, 255)]
    );
    assert_eq!(
        rt.output(&id("b"), "out")
            .unwrap()
            .as_frame()
            .unwrap()
            .to_rgb8(),
        vec![(0, 255, 0)]
    );
}

#[test]
fn diamond_split_then_merge_round_trips() {
    // s (split, k=3) -> out0..out2 -> m (merge, k=3) in0..in2.
    let mut g = Graph::new();
    g.add_node(node("s", "split", &[("channels", "3")]))
        .unwrap();
    g.add_node(node("m", "merge", &[("channels", "3")]))
        .unwrap();
    g.connect(port("s", "out0"), port("m", "in0")).unwrap();
    g.connect(port("s", "out1"), port("m", "in1")).unwrap();
    g.connect(port("s", "out2"), port("m", "in2")).unwrap();

    let src = Frame::from_data(
        2,
        2,
        3,
        FrameData::U8(vec![1, 10, 100, 2, 20, 101, 3, 30, 102, 4, 40, 103]),
    );

    let reg = builtin_registry();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_input(&id("s"), "in", Payload::Frame(src.clone()));
    rt.run_once().unwrap();

    assert_eq!(
        rt.output(&id("m"), "out").unwrap().as_frame().unwrap(),
        &src
    );
}

/// A stateless node whose only "state" is the feedback edge: out = prev + 1.
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

#[test]
fn feedback_loop_converges_over_iterations() {
    // A single node with a self-loop out -> prev. Each iteration adds 1.
    let mut g = Graph::new();
    g.add_node(node("k", "counter", &[])).unwrap();
    g.connect(port("k", "out"), port("k", "prev")).unwrap();

    let mut reg = Registry::new();
    reg.register("counter", |_| Ok(Box::new(Counter) as Box<dyn Node>));

    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_max_iters(5);
    rt.reset();
    rt.run_once().unwrap();

    // Starting from empty feedback: 1,2,3,4,5.
    assert_eq!(rt.output(&id("k"), "out").unwrap().as_scalar(), Some(5.0));

    // Streaming: buffers persist, so another pass continues 6..10.
    rt.run_once().unwrap();
    assert_eq!(rt.output(&id("k"), "out").unwrap().as_scalar(), Some(10.0));
}

#[test]
fn missing_source_input_surfaces_as_run_error() {
    // 'a' needs a frame on "in" but none is injected.
    let mut g = Graph::new();
    g.add_node(node("a", "clear_channel", &[("channel", "red")]))
        .unwrap();

    let reg = builtin_registry();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    let err = rt.run_once().unwrap_err();
    assert_eq!(err.node, id("a"));
    assert_eq!(err.error, NodeError::MissingInput("in".to_string()));
}

#[test]
fn tap_observes_latest_output_without_blocking() {
    // A self-loop counter, one iteration per run_once.
    let mut g = Graph::new();
    g.add_node(node("k", "counter", &[])).unwrap();
    g.connect(port("k", "out"), port("k", "prev")).unwrap();

    let mut reg = Registry::new();
    reg.register("counter", |_| Ok(Box::new(Counter) as Box<dyn Node>));

    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_max_iters(1);
    let tap = rt.add_tap(&id("k"), "out");
    rt.reset();

    // Run three ticks without reading the tap in between; latest-value wins.
    rt.tick(3).unwrap();
    assert_eq!(tap.latest().unwrap().as_scalar(), Some(3.0));

    // The tap also matches the runtime's captured output.
    assert_eq!(
        rt.output(&id("k"), "out").unwrap().as_scalar(),
        tap.latest().unwrap().as_scalar()
    );
}

/// A source that broadcasts one still image forever — the "static image into a
/// stream" case. It caches the payload once and republishes the *same* `Arc`
/// every evaluation via `set_shared`.
struct StillSource {
    frame: Arc<Payload>,
}

impl Node for StillSource {
    fn ports(&self) -> PortSet {
        PortSet::new(vec![], vec![PortSpec::new("out", PayloadKind::Frame)])
    }

    fn eval(&mut self, _inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        outputs.set_shared("out", self.frame.clone());
        Ok(())
    }
}

/// A source that rebuilds an identical frame from scratch each evaluation.
/// Byte-for-byte the same output as `StillSource`, but a fresh allocation.
struct RebuildingSource;

impl Node for RebuildingSource {
    fn ports(&self) -> PortSet {
        PortSet::new(vec![], vec![PortSpec::new("out", PayloadKind::Frame)])
    }

    fn eval(&mut self, _inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        outputs.set(
            "out",
            Payload::Frame(Frame::from_rgb8(1, 1, vec![(7, 7, 7)])),
        );
        Ok(())
    }
}

#[test]
fn still_source_keeps_buffer_identity_while_seq_advances() {
    // A still broadcast into a stream: every tick is a real publish (the stream
    // is live), but the pixels never change — so a consumer should be able to
    // skip re-uploading the buffer without comparing any pixels.
    let still = Arc::new(Payload::Frame(Frame::from_rgb8(1, 1, vec![(7, 7, 7)])));

    let mut g = Graph::new();
    g.add_node(node("still", "still", &[])).unwrap();

    let mut reg = Registry::new();
    let shared = still.clone();
    reg.register("still", move |_| {
        Ok(Box::new(StillSource {
            frame: shared.clone(),
        }) as Box<dyn Node>)
    });

    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    let tap = rt.add_tap(&id("still"), "out");
    assert_eq!(tap.seq(), 0, "nothing published yet");

    rt.tick(3).unwrap();

    // Liveness: three publishes landed, so a poller sees three changes...
    let (seq, latest) = tap.latest_with_seq();
    assert_eq!(seq, 3);

    // ...but the payload is literally the same allocation the node cached, so
    // an expensive consumer can bail out on a pointer comparison.
    let latest = latest.expect("the tap holds the still");
    assert!(
        Arc::ptr_eq(&latest, &still),
        "set_shared must preserve buffer identity through the scheduler"
    );

    // The always-on output capture sees the same allocation too.
    assert!(Arc::ptr_eq(
        &rt.output_arc(&id("still"), "out").unwrap(),
        &still
    ));
}

#[test]
fn rebuilt_frames_are_distinct_allocations_even_when_equal() {
    // The contrast case: identical pixels, but a fresh buffer each tick, so
    // pointer identity correctly reports "this is not the buffer you had".
    let mut g = Graph::new();
    g.add_node(node("src", "rebuild", &[])).unwrap();

    let mut reg = Registry::new();
    reg.register("rebuild", |_| {
        Ok(Box::new(RebuildingSource) as Box<dyn Node>)
    });

    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    let tap = rt.add_tap(&id("src"), "out");

    rt.run_once().unwrap();
    let first = tap.latest().unwrap();
    rt.run_once().unwrap();
    let second = tap.latest().unwrap();

    assert_eq!(
        first.as_frame().unwrap(),
        second.as_frame().unwrap(),
        "the frames are equal by value"
    );
    assert!(
        !Arc::ptr_eq(&first, &second),
        "but they are distinct allocations, so a consumer must re-read"
    );
    assert_eq!(tap.seq(), 2);
}

/// `a -> b -> c`, each clearing one channel. Only `c`'s output is terminal.
fn three_stage_chain() -> (Graph, Registry) {
    let mut g = Graph::new();
    g.add_node(node("a", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node("b", "clear_channel", &[("channel", "green")]))
        .unwrap();
    g.add_node(node("c", "clear_channel", &[("channel", "blue")]))
        .unwrap();
    g.connect(port("a", "out"), port("b", "in")).unwrap();
    g.connect(port("b", "out"), port("c", "in")).unwrap();
    (g, builtin_registry())
}

fn white_pixel() -> Payload {
    Payload::Frame(Frame::from_rgb8(1, 1, vec![(255, 255, 255)]))
}

#[test]
fn results_are_captured_but_intermediates_are_opt_in() {
    let (g, reg) = three_stage_chain();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_input(&id("a"), "in", white_pixel());
    rt.run_once().unwrap();

    // The pipeline's result never leaves its component, so it costs nothing to
    // keep and is captured without asking.
    assert!(rt.is_observed(&id("c"), "out"));
    assert!(rt.output(&id("c"), "out").is_some());

    // The intermediates are dropped once the consuming node has read them.
    assert!(!rt.is_observed(&id("a"), "out"));
    assert!(rt.output(&id("a"), "out").is_none());
    assert!(rt.output(&id("b"), "out").is_none());
}

#[test]
fn watching_an_intermediate_makes_it_readable_and_unwatching_drops_it() {
    let (g, reg) = three_stage_chain();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_input(&id("a"), "in", white_pixel());

    rt.watch(&id("b"), "out");
    rt.run_once().unwrap();

    // a cleared red, b cleared green.
    assert_eq!(
        rt.output(&id("b"), "out")
            .unwrap()
            .as_frame()
            .unwrap()
            .to_rgb8(),
        vec![(0, 0, 255)]
    );

    rt.unwatch(&id("b"), "out");
    assert!(
        rt.output(&id("b"), "out").is_none(),
        "a value that will no longer be refreshed must not linger"
    );
    assert!(rt.output(&id("c"), "out").is_some(), "c is unaffected");
}

#[test]
fn capture_all_restores_unconditional_capture() {
    let (g, reg) = three_stage_chain();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_input(&id("a"), "in", white_pixel());

    rt.set_capture_all(true);
    rt.run_once().unwrap();
    assert!(rt.output(&id("a"), "out").is_some());
    assert!(rt.output(&id("b"), "out").is_some());

    rt.set_capture_all(false);
    rt.run_once().unwrap();
    assert!(rt.output(&id("a"), "out").is_none());
}

#[test]
fn a_live_tap_keeps_a_port_observed_on_its_own() {
    let (g, reg) = three_stage_chain();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();

    let tap = rt.add_tap(&id("b"), "out");
    assert!(rt.is_observed(&id("b"), "out"));

    // `unwatch` undoes `watch`, and nothing else: the tap still needs the value.
    rt.unwatch(&id("b"), "out");
    assert!(
        rt.is_observed(&id("b"), "out"),
        "unwatch must not silently starve a live tap"
    );

    rt.set_input(&id("a"), "in", white_pixel());
    rt.run_once().unwrap();
    assert!(tap.latest().is_some());

    assert_eq!(rt.remove_taps(&id("b"), "out"), 1);
    assert!(!rt.is_observed(&id("b"), "out"));
}

#[test]
fn observation_splits_a_fusable_run_only_where_it_looks() {
    // a -> b -> c -> d: one structural chain of four.
    let mut g = Graph::new();
    for (name, channel) in [("a", "red"), ("b", "green"), ("c", "blue"), ("d", "red")] {
        g.add_node(node(name, "clear_channel", &[("channel", channel)]))
            .unwrap();
    }
    g.connect(port("a", "out"), port("b", "in")).unwrap();
    g.connect(port("b", "out"), port("c", "in")).unwrap();
    g.connect(port("c", "out"), port("d", "in")).unwrap();

    let reg = builtin_registry();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();

    // `d`'s output is captured, but a run's own result has to materialize
    // anyway, so it is not a barrier: the whole chain is one run.
    assert_eq!(rt.fusable_runs(), &[vec![0, 1, 2, 3]]);

    // Previewing b forces b's value to exist. The run splits there, and there
    // only -- you pay one materialization for the frame you asked to see.
    rt.watch(&id("b"), "out");
    assert_eq!(rt.fusable_runs(), &[vec![0, 1], vec![2, 3]]);

    // Stop looking and the run heals.
    rt.unwatch(&id("b"), "out");
    assert_eq!(rt.fusable_runs(), &[vec![0, 1, 2, 3]]);

    // Capturing everything blocks every fusion, as it must.
    rt.set_capture_all(true);
    assert!(rt.fusable_runs().is_empty());
}

#[test]
fn attaching_a_tap_mid_stream_preserves_pipeline_state() {
    // The scenario the opt-in capture exists to serve: a pipeline is streaming
    // and someone clicks "preview" on a node. Re-deriving what to capture must
    // not rebuild nodes or edge buffers, or a feedback loop would silently
    // restart mid-convergence.
    let mut g = Graph::new();
    g.add_node(node("k", "counter", &[])).unwrap();
    g.connect(port("k", "out"), port("k", "prev")).unwrap();

    let mut reg = Registry::new();
    reg.register("counter", |_| Ok(Box::new(Counter) as Box<dyn Node>));

    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_max_iters(1);
    rt.reset();

    for _ in 0..3 {
        rt.run_once().unwrap();
    }
    assert_eq!(rt.output(&id("k"), "out").unwrap().as_scalar(), Some(3.0));

    // Preview attached mid-stream.
    let tap = rt.add_tap(&id("k"), "out");

    rt.run_once().unwrap();
    assert_eq!(
        rt.output(&id("k"), "out").unwrap().as_scalar(),
        Some(4.0),
        "the accumulator kept counting instead of restarting at 1"
    );
    assert_eq!(tap.latest().unwrap().as_scalar(), Some(4.0));

    // Detaching mid-stream is equally non-destructive.
    rt.remove_taps(&id("k"), "out");
    rt.run_once().unwrap();
    assert_eq!(rt.output(&id("k"), "out").unwrap().as_scalar(), Some(5.0));
}

/// Stage-level (no runtime) check that split and merge are inverses.
#[test]
fn split_and_merge_stages_invert_each_other() {
    use pipe_graph::stages::{merge::MergeStage, split::SplitStage};
    use std::collections::HashMap;

    let src = Frame::from_data(
        2,
        2,
        3,
        FrameData::U8(vec![1, 10, 100, 2, 20, 101, 3, 30, 102, 4, 40, 103]),
    );

    let mut split = SplitStage::new(3);
    let mut m = HashMap::new();
    m.insert(PortId("in".to_string()), Payload::Frame(src.clone()));
    let mut split_out = Outputs::new();
    split.eval(&Inputs::new(m), &mut split_out).unwrap();
    let ch0 = split_out.get("out0").unwrap().as_frame().unwrap().clone();
    assert_eq!(ch0.channels, 1);
    assert_eq!(ch0.as_u8().unwrap(), &[1, 2, 3, 4]);

    let mut merge = MergeStage::new(3);
    let mut m = HashMap::new();
    for i in 0..3 {
        let ch = split_out.get(&format!("out{i}")).unwrap().clone();
        m.insert(PortId(format!("in{i}")), ch);
    }
    let mut merged = Outputs::new();
    merge.eval(&Inputs::new(m), &mut merged).unwrap();
    assert_eq!(merged.get("out").unwrap().as_frame().unwrap(), &src);
}
