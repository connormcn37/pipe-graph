//! End-to-end execution tests: build a graph, compile + run it, assert outputs.
//!
//! This is the headline milestone — an arbitrary graph runs to a hand-computed
//! result, including fan-out, multi-port split/merge, and a feedback loop.

use pipe_graph::data::{Frame, FrameData, Payload, PayloadKind};
use pipe_graph::exec::{
    Inputs, Node, NodeError, Outputs, PortSet, PortSpec, Registry, Runtime, builtin_registry,
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
