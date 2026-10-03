//! End-to-end tests for the per-pixel stage pack, built through
//! `builtin_registry` exactly as a serialized graph would be.

use pipe_graph::data::{Frame, FrameData, Payload};
use pipe_graph::exec::{Runtime, builtin_registry};
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
fn u8_chain_grayscale_invert_blur_gain_threshold() {
    // gray -> inv -> blur(r=1) -> gain(1.5) -> thr(0.9)
    let mut g = Graph::new();
    g.add_node(node("gray", "grayscale", &[])).unwrap();
    g.add_node(node("inv", "invert", &[])).unwrap();
    g.add_node(node("blur", "box_blur", &[("radius", "1")]))
        .unwrap();
    g.add_node(node("gain", "gain", &[("factor", "1.5")]))
        .unwrap();
    g.add_node(node("thr", "threshold", &[("level", "0.9")]))
        .unwrap();
    g.connect(port("gray", "out"), port("inv", "in")).unwrap();
    g.connect(port("inv", "out"), port("blur", "in")).unwrap();
    g.connect(port("blur", "out"), port("gain", "in")).unwrap();
    g.connect(port("gain", "out"), port("thr", "in")).unwrap();

    let reg = builtin_registry();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.watch(&id("gain"), "out");
    rt.set_input(
        &id("gray"),
        "in",
        Payload::Frame(Frame::from_rgb8(
            3,
            1,
            vec![(255, 255, 255), (0, 0, 0), (255, 0, 0)],
        )),
    );
    rt.run_once().unwrap();

    // grayscale:  [255, 0, 76]
    // invert:     [0, 255, 179]
    // box_blur r=1 on a single row (vertical clamp triples each row sum, /9):
    //   x0: (0+0+255)/3       = 85
    //   x1: (0+255+179)/3     = 144.67 -> 145
    //   x2: (255+179+179)/3   = 204.33 -> 204
    // gain 1.5:   [127.5 -> 128, 217.5 -> 218, 306 -> 255]
    let gained = rt.output(&id("gain"), "out").unwrap().as_frame().unwrap();
    assert_eq!(gained.channels, 1);
    assert_eq!(gained.as_u8().unwrap(), &[128, 218, 255]);

    // threshold 0.9 -> cut at 229.5: only the saturated pixel is on.
    let out = rt.output(&id("thr"), "out").unwrap().as_frame().unwrap();
    assert_eq!(out.as_u8().unwrap(), &[0, 0, 255]);
}

#[test]
fn f32_split_blur_gain_merge() {
    // The benchmark's shape, in miniature: split -> 3x(blur -> gain) -> merge.
    let mut g = Graph::new();
    g.add_node(node("split", "split", &[("channels", "3")]))
        .unwrap();
    g.add_node(node("merge", "merge", &[("channels", "3")]))
        .unwrap();
    for (c, factor) in ["1", "2", "0.5"].iter().enumerate() {
        let blur = format!("blur{c}");
        let gain = format!("gain{c}");
        g.add_node(node(&blur, "box_blur", &[("radius", "1")]))
            .unwrap();
        g.add_node(node(&gain, "gain", &[("factor", factor)]))
            .unwrap();
        g.connect(port("split", &format!("out{c}")), port(&blur, "in"))
            .unwrap();
        g.connect(port(&blur, "out"), port(&gain, "in")).unwrap();
        g.connect(port(&gain, "out"), port("merge", &format!("in{c}")))
            .unwrap();
    }

    let reg = builtin_registry();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    // 3x1 RGB: R = [0, .3, .6], G = flat .3, B = [.6, 0, 0].
    rt.set_input(
        &id("split"),
        "in",
        Payload::Frame(Frame::from_data(
            3,
            1,
            3,
            FrameData::F32(vec![
                0.0, 0.3, 0.6, //
                0.3, 0.3, 0.0, //
                0.6, 0.3, 0.0,
            ]),
        )),
    );
    rt.run_once().unwrap();

    // R blur: [.1, .3, .5] * 1
    // G blur: flat .3 * 2 = .6
    // B blur: [(.6+.6+0)/3, (.6+0+0)/3, 0] = [.4, .2, 0] * .5 = [.2, .1, 0]
    let want = [
        0.1, 0.6, 0.2, //
        0.3, 0.6, 0.1, //
        0.5, 0.6, 0.0,
    ];
    let out = rt.output(&id("merge"), "out").unwrap().as_frame().unwrap();
    assert_eq!((out.width, out.height, out.channels), (3, 1, 3));
    for (i, (got, want)) in out.as_f32().unwrap().iter().zip(want).enumerate() {
        assert!((got - want).abs() < 1e-6, "sample {i}: {got} vs {want}");
    }
}

#[test]
fn new_kinds_are_registered() {
    let reg = builtin_registry();
    for kind in ["gain", "grayscale", "invert", "threshold", "box_blur"] {
        assert!(reg.contains(kind), "{kind} not registered");
    }
}
