//! Fused-run execution: `run_once` evaluates each `Runtime::fusable_runs()` run
//! as a unit, handing payloads node to node without materializing the private
//! edges between them, and evaluating 1-in/1-out nodes in place when the
//! payload they receive is uniquely owned.
//!
//! Every test here compares against the unfused path (`set_fusion(false)`):
//! fusion is an optimization, so anything observable must be identical.

use std::sync::{Arc, Mutex};

use pipe_graph::data::{Frame, FrameData, Payload, PayloadKind};
use pipe_graph::exec::{
    ExecMode, Inputs, Node, NodeError, Outputs, PortSet, PortSpec, Registry, Runtime,
    builtin_registry,
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

fn frame_ports() -> PortSet {
    PortSet::new(
        vec![PortSpec::new("in", PayloadKind::Frame)],
        vec![PortSpec::new("out", PayloadKind::Frame)],
    )
}

/// How a probe node was evaluated, and the address of the pixel buffer it
/// published. Equal addresses along a chain mean no clone happened between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Via {
    Eval,
    InPlace,
}

type Log = Arc<Mutex<Vec<(String, Via, usize)>>>;

fn buf_addr(frame: &Frame) -> usize {
    frame.as_u8().expect("probe frames are u8").as_ptr() as usize
}

/// 1-in/1-out frame node that bumps the first byte and logs how it ran.
struct Probe {
    name: String,
    log: Log,
}

impl Probe {
    fn apply(&self, frame: &mut Frame, via: Via) {
        let px = frame.as_u8_mut().expect("u8 frame");
        px[0] = px[0].wrapping_add(1);
        self.log
            .lock()
            .unwrap()
            .push((self.name.clone(), via, buf_addr(frame)));
    }
}

impl Node for Probe {
    fn ports(&self) -> PortSet {
        frame_ports()
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let mut frame = inputs.frame("in")?.clone();
        self.apply(&mut frame, Via::Eval);
        outputs.set("out", Payload::Frame(frame));
        Ok(())
    }

    fn eval_in_place(&mut self, payload: &mut Payload) -> Option<Result<(), NodeError>> {
        let Payload::Frame(frame) = payload else {
            return None;
        };
        self.apply(frame, Via::InPlace);
        Some(Ok(()))
    }
}

/// Publishes the first frame it ever saw, as the *same* cached `Arc` every
/// evaluation — so its output is never uniquely owned downstream.
struct Sticky {
    cached: Option<Arc<Payload>>,
    log: Log,
}

impl Node for Sticky {
    fn ports(&self) -> PortSet {
        frame_ports()
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        if self.cached.is_none() {
            self.cached = Some(Arc::new(Payload::Frame(inputs.frame("in")?.clone())));
        }
        let cached = self.cached.clone().unwrap();
        let addr = buf_addr(cached.as_frame().unwrap());
        self.log
            .lock()
            .unwrap()
            .push(("sticky".to_string(), Via::Eval, addr));
        outputs.set_shared("out", cached);
        Ok(())
    }
}

/// Scalar node with two inputs: out = a + 10 * b.
struct Weigh;

impl Node for Weigh {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![
                PortSpec::new("a", PayloadKind::Scalar),
                PortSpec::new("b", PayloadKind::Scalar),
            ],
            vec![PortSpec::new("out", PayloadKind::Scalar)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let get = |p: &str| {
            inputs
                .get(p)
                .and_then(Payload::as_scalar)
                .ok_or_else(|| NodeError::MissingInput(p.to_string()))
        };
        outputs.set("out", Payload::Scalar(get("a")? + 10.0 * get("b")?));
        Ok(())
    }
}

/// Scalar 1-in/1-out: out = in + 1, with an in-place path.
struct Inc;

impl Node for Inc {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Scalar)],
            vec![PortSpec::new("out", PayloadKind::Scalar)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let v = inputs
            .get("in")
            .and_then(Payload::as_scalar)
            .ok_or_else(|| NodeError::MissingInput("in".to_string()))?;
        outputs.set("out", Payload::Scalar(v + 1.0));
        Ok(())
    }

    fn eval_in_place(&mut self, payload: &mut Payload) -> Option<Result<(), NodeError>> {
        match payload {
            Payload::Scalar(v) => {
                *v += 1.0;
                Some(Ok(()))
            }
            _ => None,
        }
    }
}

/// The builtin registry plus the test nodes above, all logging to `log`.
fn registry(log: &Log) -> Registry {
    let mut reg = builtin_registry();
    let probe_log = log.clone();
    reg.register("probe", move |p| {
        Ok(Box::new(Probe {
            name: p.get("name").cloned().unwrap_or_default(),
            log: probe_log.clone(),
        }) as Box<dyn Node>)
    });
    let sticky_log = log.clone();
    reg.register("sticky", move |_| {
        Ok(Box::new(Sticky {
            cached: None,
            log: sticky_log.clone(),
        }) as Box<dyn Node>)
    });
    reg.register("weigh", |_| Ok(Box::new(Weigh) as Box<dyn Node>));
    reg.register("inc", |_| Ok(Box::new(Inc) as Box<dyn Node>));
    reg
}

fn runtime(g: &Graph, fusion: bool) -> Runtime {
    let log = Log::default();
    let mut rt = Runtime::instantiate(g, &registry(&log)).unwrap();
    rt.set_fusion(fusion);
    rt
}

/// A mixed pipeline: a builtin chain (clear -> crop -> cast -> cast -> clear)
/// feeding a split -> merge pair, i.e. in-place-capable stages, a stage that
/// declines in-place (a real cast), and multi-edge links between two nodes.
fn mixed_pipeline() -> Graph {
    let mut g = Graph::new();
    g.add_node(node("r", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node(
        "crop",
        "crop",
        &[("x", "1"), ("y", "0"), ("w", "2"), ("h", "2")],
    ))
    .unwrap();
    g.add_node(node("f", "cast", &[("dtype", "f32")])).unwrap();
    g.add_node(node("u", "cast", &[("dtype", "u8")])).unwrap();
    g.add_node(node("same", "cast", &[("dtype", "u8")]))
        .unwrap();
    g.add_node(node("g", "clear_channel", &[("channel", "green")]))
        .unwrap();
    g.add_node(node("s", "split", &[("channels", "3")]))
        .unwrap();
    g.add_node(node("m", "merge", &[("channels", "3")]))
        .unwrap();
    g.connect(port("r", "out"), port("crop", "in")).unwrap();
    g.connect(port("crop", "out"), port("f", "in")).unwrap();
    g.connect(port("f", "out"), port("u", "in")).unwrap();
    g.connect(port("u", "out"), port("same", "in")).unwrap();
    g.connect(port("same", "out"), port("g", "in")).unwrap();
    g.connect(port("g", "out"), port("s", "in")).unwrap();
    for c in 0..3 {
        g.connect(port("s", &format!("out{c}")), port("m", &format!("in{c}")))
            .unwrap();
    }
    g
}

fn source_frame(seed: u8) -> Frame {
    let px: Vec<(u8, u8, u8)> = (0..9u8)
        .map(|i| (seed.wrapping_add(i), seed ^ i, i.wrapping_mul(29)))
        .collect();
    Frame::from_rgb8(3, 3, px)
}

#[test]
fn fused_outputs_equal_unfused_outputs() {
    let g = mixed_pipeline();
    let mut fused = runtime(&g, true);
    let mut plain = runtime(&g, false);
    assert!(fused.fusion_enabled());
    assert!(
        !fused.fusable_runs().is_empty(),
        "the pipeline has fusable runs"
    );

    for seed in [7u8, 200, 33] {
        for rt in [&mut fused, &mut plain] {
            rt.set_input(&id("r"), "in", Payload::Frame(source_frame(seed)));
            rt.run_once().unwrap();
        }
        let a = fused.output(&id("m"), "out").unwrap().as_frame().unwrap();
        let b = plain.output(&id("m"), "out").unwrap().as_frame().unwrap();
        assert_eq!(a, b);
        assert_eq!((a.width, a.height), (2, 2));
    }
}

#[test]
fn intermediate_buffers_stay_empty_when_fused() {
    let g = mixed_pipeline();
    let links = [
        (port("r", "out"), port("crop", "in")),
        (port("crop", "out"), port("f", "in")),
        (port("same", "out"), port("g", "in")),
        (port("s", "out1"), port("m", "in1")),
    ];

    let mut rt = runtime(&g, false);
    rt.set_input(&id("r"), "in", Payload::Frame(source_frame(1)));
    rt.run_once().unwrap();
    for (from, to) in &links {
        assert!(
            !rt.edge_buffer(from, to).unwrap().is_empty(),
            "unfused path materializes {from:?} -> {to:?}"
        );
    }

    // Switching fusion on mid-stream also clears what the unfused run left.
    rt.set_fusion(true);
    rt.run_once().unwrap();
    for (from, to) in &links {
        assert!(
            rt.edge_buffer(from, to).unwrap().is_empty(),
            "fused path never materializes {from:?} -> {to:?}"
        );
    }
    assert!(rt.output(&id("m"), "out").is_some());
}

#[test]
fn tapping_a_mid_run_port_still_materializes_it() {
    // a -> b -> c -> d, all one run until b.out is tapped.
    let mut g = Graph::new();
    for (n, ch) in [("a", "red"), ("b", "green"), ("c", "blue"), ("d", "red")] {
        g.add_node(node(n, "clear_channel", &[("channel", ch)]))
            .unwrap();
    }
    g.connect(port("a", "out"), port("b", "in")).unwrap();
    g.connect(port("b", "out"), port("c", "in")).unwrap();
    g.connect(port("c", "out"), port("d", "in")).unwrap();

    let mut fused = runtime(&g, true);
    let mut plain = runtime(&g, false);
    let tap = fused.add_tap(&id("b"), "out");
    let plain_tap = plain.add_tap(&id("b"), "out");
    assert_eq!(fused.fusable_runs().len(), 2, "the tap splits the run");

    let src = Frame::from_rgb8(1, 1, vec![(9, 8, 7)]);
    for rt in [&mut fused, &mut plain] {
        rt.set_input(&id("a"), "in", Payload::Frame(src.clone()));
        rt.run_once().unwrap();
    }

    let seen = tap.latest().unwrap();
    assert_eq!(seen.as_frame().unwrap().to_rgb8(), vec![(0, 0, 7)]);
    assert_eq!(
        seen.as_frame().unwrap(),
        plain_tap.latest().unwrap().as_frame().unwrap()
    );
    assert_eq!(tap.seq(), 1);
    assert!(fused.output(&id("b"), "out").is_some());
    // The tapped link exists; the links on either side of it do not.
    assert!(
        !fused
            .edge_buffer(&port("b", "out"), &port("c", "in"))
            .unwrap()
            .is_empty()
    );
    assert!(
        fused
            .edge_buffer(&port("a", "out"), &port("b", "in"))
            .unwrap()
            .is_empty()
    );
    assert!(
        fused
            .edge_buffer(&port("c", "out"), &port("d", "in"))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        fused.output(&id("d"), "out").unwrap().as_frame().unwrap(),
        plain.output(&id("d"), "out").unwrap().as_frame().unwrap()
    );
}

fn probe_chain(head_kind: &str) -> Graph {
    let mut g = Graph::new();
    g.add_node(node("p1", head_kind, &[("name", "p1")]))
        .unwrap();
    g.add_node(node("p2", "probe", &[("name", "p2")])).unwrap();
    g.add_node(node("p3", "probe", &[("name", "p3")])).unwrap();
    g.connect(port("p1", "out"), port("p2", "in")).unwrap();
    g.connect(port("p2", "out"), port("p3", "in")).unwrap();
    g
}

fn run_logged(g: &Graph, fusion: bool, frame: Frame) -> (Runtime, Vec<(String, Via, usize)>) {
    let log = Log::default();
    let mut rt = Runtime::instantiate(g, &registry(&log)).unwrap();
    rt.set_fusion(fusion);
    rt.set_input(&id("p1"), "in", Payload::Frame(frame));
    rt.run_once().unwrap();
    let entries = log.lock().unwrap().clone();
    (rt, entries)
}

#[test]
fn clone_probe_shows_in_place_reuse_along_a_fused_run() {
    let g = probe_chain("probe");
    let src = Frame::from_rgb8(2, 1, vec![(10, 0, 0), (0, 0, 0)]);

    let (fused_rt, fused) = run_logged(&g, true, src.clone());
    let (plain_rt, plain) = run_logged(&g, false, src);

    // Unfused: every stage evaluates on its own copy.
    let vias: Vec<Via> = plain.iter().map(|e| e.1).collect();
    assert_eq!(vias, vec![Via::Eval; 3]);
    let mut addrs: Vec<usize> = plain.iter().map(|e| e.2).collect();
    addrs.dedup();
    assert_eq!(addrs.len(), 3, "three separate buffers: {plain:?}");

    // Fused: the head copies its (shared) external input once; the rest of
    // the run transforms that same buffer in place.
    let vias: Vec<Via> = fused.iter().map(|e| e.1).collect();
    assert_eq!(vias, vec![Via::Eval, Via::InPlace, Via::InPlace]);
    assert!(
        fused.iter().all(|e| e.2 == fused[0].2),
        "one buffer end to end: {fused:?}"
    );
    let out = fused_rt.output(&id("p3"), "out").unwrap();
    assert_eq!(buf_addr(out.as_frame().unwrap()), fused[0].2);

    assert_eq!(
        out.as_frame().unwrap(),
        plain_rt
            .output(&id("p3"), "out")
            .unwrap()
            .as_frame()
            .unwrap()
    );
    assert_eq!(out.as_frame().unwrap().as_u8().unwrap()[0], 13);
}

#[test]
fn a_shared_payload_is_copied_before_in_place_mutation() {
    // The head republishes one cached Arc; mutating it in place would corrupt
    // the cache and change every later run's result.
    let g = probe_chain("sticky");
    let log = Log::default();
    let mut rt = Runtime::instantiate(&g, &registry(&log)).unwrap();
    assert!(rt.fusion_enabled());
    let src = Frame::from_rgb8(1, 1, vec![(5, 0, 0)]);
    rt.set_input(&id("p1"), "in", Payload::Frame(src));

    for _ in 0..3 {
        rt.run_once().unwrap();
        let out = rt.output(&id("p3"), "out").unwrap();
        assert_eq!(out.as_frame().unwrap().as_u8().unwrap()[0], 7);
    }

    let entries = log.lock().unwrap().clone();
    let cached = entries[0].2;
    for chunk in entries.chunks(3) {
        assert_eq!(chunk[0].2, cached, "sticky keeps its buffer");
        assert_eq!(chunk[1].1, Via::InPlace);
        assert_ne!(chunk[1].2, cached, "p2 worked on a copy");
        assert_eq!(chunk[2].2, chunk[1].2, "p3 reused p2's buffer");
    }
}

#[test]
fn members_with_several_inputs_fall_back_to_eval() {
    // inc -> weigh, where weigh.a comes from inc and weigh.b either from inc
    // too (two edges on one link) or from an external input.
    let mut g = Graph::new();
    g.add_node(node("i", "inc", &[])).unwrap();
    g.add_node(node("w", "weigh", &[])).unwrap();
    g.add_node(node("j", "inc", &[])).unwrap();
    g.connect(port("i", "out"), port("w", "a")).unwrap();
    g.connect(port("i", "out"), port("w", "b")).unwrap();
    g.connect(port("w", "out"), port("j", "in")).unwrap();

    let mut ext = Graph::new();
    ext.add_node(node("i", "inc", &[])).unwrap();
    ext.add_node(node("w", "weigh", &[])).unwrap();
    ext.add_node(node("j", "inc", &[])).unwrap();
    ext.connect(port("i", "out"), port("w", "a")).unwrap();
    ext.connect(port("w", "out"), port("j", "in")).unwrap();

    for (graph, expected) in [(&g, 2.0 + 20.0 + 1.0), (&ext, 2.0 + 30.0 + 1.0)] {
        let mut results = Vec::new();
        for fusion in [true, false] {
            let mut rt = runtime(graph, fusion);
            if fusion {
                assert_eq!(rt.fusable_runs().len(), 1);
                assert_eq!(rt.fusable_runs()[0].len(), 3);
            }
            rt.set_input(&id("i"), "in", Payload::Scalar(1.0));
            rt.set_input(&id("w"), "b", Payload::Scalar(3.0));
            rt.run_once().unwrap();
            results.push(rt.output(&id("j"), "out").unwrap().as_scalar().unwrap());
        }
        assert_eq!(results, vec![expected, expected]);
    }
}

#[test]
fn errors_name_the_same_node_fused_or_not() {
    // The crop rectangle does not fit, so the run's middle member fails.
    let mut g = Graph::new();
    g.add_node(node("a", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node(
        "c",
        "crop",
        &[("x", "0"), ("y", "0"), ("w", "5"), ("h", "1")],
    ))
    .unwrap();
    g.add_node(node("z", "clear_channel", &[("channel", "blue")]))
        .unwrap();
    g.connect(port("a", "out"), port("c", "in")).unwrap();
    g.connect(port("c", "out"), port("z", "in")).unwrap();

    let mut errors = Vec::new();
    for fusion in [true, false] {
        let mut rt = runtime(&g, fusion);
        rt.set_input(
            &id("a"),
            "in",
            Payload::Frame(Frame::from_rgb8(1, 1, vec![(1, 2, 3)])),
        );
        errors.push(rt.run_once().unwrap_err());
    }
    assert_eq!(errors[0], errors[1]);
    assert_eq!(errors[0].node, id("c"));
}

#[test]
fn fused_f32_chain_agrees_across_ticks() {
    // Repeated ticks through a fused cast -> crop -> cast chain agree with the
    // serial path, including an f32 payload cropped in place.
    let mut g = Graph::new();
    g.add_node(node("f", "cast", &[("dtype", "f32")])).unwrap();
    g.add_node(node(
        "c",
        "crop",
        &[("x", "0"), ("y", "1"), ("w", "1"), ("h", "1")],
    ))
    .unwrap();
    g.add_node(node("u", "cast", &[("dtype", "u8")])).unwrap();
    g.connect(port("f", "out"), port("c", "in")).unwrap();
    g.connect(port("c", "out"), port("u", "in")).unwrap();

    let src = Frame::from_data(1, 2, 1, FrameData::U8(vec![10, 200]));
    let mut outs = Vec::new();
    for fusion in [true, false] {
        let mut rt = runtime(&g, fusion);
        rt.set_input(&id("f"), "in", Payload::Frame(src.clone()));
        rt.tick(3).unwrap();
        outs.push(
            rt.output(&id("u"), "out")
                .unwrap()
                .as_frame()
                .unwrap()
                .clone(),
        );
    }
    assert_eq!(outs[0], outs[1]);
    assert_eq!(outs[0].as_u8().unwrap(), &[200]);
}

#[test]
fn parallel_mode_runs_each_fused_branch_in_place_on_its_own() {
    // `src` fans out to two fused runs, a1 -> a2 and b1 -> b2. In parallel
    // mode each run is one unit of work: its head copies the (shared) fan-out
    // value once and its tail reuses that buffer, exactly as in serial mode.
    let mut g = Graph::new();
    for n in ["src", "a1", "a2", "b1", "b2"] {
        g.add_node(node(n, "probe", &[("name", n)])).unwrap();
    }
    g.connect(port("src", "out"), port("a1", "in")).unwrap();
    g.connect(port("src", "out"), port("b1", "in")).unwrap();
    g.connect(port("a1", "out"), port("a2", "in")).unwrap();
    g.connect(port("b1", "out"), port("b2", "in")).unwrap();

    let run = |mode: ExecMode, fusion: bool| {
        let log = Log::default();
        let mut rt = Runtime::instantiate(&g, &registry(&log)).unwrap();
        rt.set_exec_mode(mode);
        rt.set_fusion(fusion);
        rt.set_input(
            &id("src"),
            "in",
            Payload::Frame(Frame::from_rgb8(2, 1, vec![(1, 0, 0), (0, 0, 0)])),
        );
        rt.tick(2).unwrap();
        let entries = log.lock().unwrap().clone();
        (rt, entries)
    };

    let (par, log) = run(ExecMode::Parallel { threads: 2 }, true);
    assert_eq!(par.fusable_runs().len(), 2);
    let entry = |name: &str| -> Vec<(Via, usize)> {
        log.iter()
            .filter(|e| e.0 == name)
            .map(|e| (e.1, e.2))
            .collect()
    };
    for (head, tail) in [("a1", "a2"), ("b1", "b2")] {
        let (h, t) = (entry(head), entry(tail));
        assert_eq!(h.len(), 2, "{head} ran once per tick");
        for (h, t) in h.iter().zip(&t) {
            assert_eq!(h.0, Via::Eval, "{head} copies the shared fan-out value");
            assert_eq!(t.0, Via::InPlace, "{tail} reuses it");
            assert_eq!(h.1, t.1, "{head} -> {tail} is one buffer");
        }
        assert!(
            par.edge_buffer(&port(head, "out"), &port(tail, "in"))
                .unwrap()
                .is_empty(),
            "{head} -> {tail} never materializes"
        );
    }

    let (serial, _) = run(ExecMode::Serial, false);
    for n in ["a2", "b2"] {
        assert_eq!(
            par.output(&id(n), "out").unwrap().as_frame().unwrap(),
            serial.output(&id(n), "out").unwrap().as_frame().unwrap()
        );
    }
}
