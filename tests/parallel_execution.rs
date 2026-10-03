//! `ExecMode::Parallel` must be an optimization only: identical outputs and
//! tap sequence numbers to `ExecMode::Serial`, with independent branches
//! actually running at the same time.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pipe_graph::data::{Frame, Payload, PayloadKind};
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

/// Self-looping counter: out = prev + 1. Gives the graph a cyclic component.
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

/// A fan-out + diamond + an independent feedback loop:
///
/// ```text
///        +--> a (clear green)
/// h -----+--> b (clear blue)
/// (red)  +--> sp (split 3) ==out0..2==> m (merge 3)
///
/// k (counter, self-loop)
/// ```
fn mixed_graph() -> (Graph, Registry) {
    let mut g = Graph::new();
    g.add_node(node("h", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node("a", "clear_channel", &[("channel", "green")]))
        .unwrap();
    g.add_node(node("b", "clear_channel", &[("channel", "blue")]))
        .unwrap();
    g.add_node(node("sp", "split", &[("channels", "3")]))
        .unwrap();
    g.add_node(node("m", "merge", &[("channels", "3")]))
        .unwrap();
    g.add_node(node("k", "counter", &[])).unwrap();
    g.connect(port("h", "out"), port("a", "in")).unwrap();
    g.connect(port("h", "out"), port("b", "in")).unwrap();
    g.connect(port("h", "out"), port("sp", "in")).unwrap();
    for c in 0..3 {
        g.connect(port("sp", &format!("out{c}")), port("m", &format!("in{c}")))
            .unwrap();
    }
    g.connect(port("k", "out"), port("k", "prev")).unwrap();

    let mut reg = builtin_registry();
    reg.register("counter", |_| Ok(Box::new(Counter) as Box<dyn Node>));
    (g, reg)
}

/// Everything observable after a run, for comparing modes.
#[derive(Debug, PartialEq)]
struct Snapshot {
    frames: Vec<Option<Frame>>,
    counter: Option<f64>,
    taps: Vec<(u64, Option<Frame>)>,
}

fn run_mixed(mode: ExecMode, ticks: u32) -> Snapshot {
    let (g, reg) = mixed_graph();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_exec_mode(mode);
    assert_eq!(rt.exec_mode(), mode);
    rt.set_max_iters(3);

    let tapped = [("h", "out"), ("a", "out"), ("sp", "out1"), ("m", "out")];
    let taps: Vec<_> = tapped.iter().map(|(n, p)| rt.add_tap(&id(n), p)).collect();

    let px: Vec<(u8, u8, u8)> = (0..16).map(|i| (i * 3, i * 5 + 1, 200 - i * 7)).collect();
    rt.set_input(&id("h"), "in", Payload::Frame(Frame::from_rgb8(4, 4, px)));
    rt.tick(ticks).unwrap();

    let frame = |n: &str, p: &str| rt.output(&id(n), p).and_then(Payload::as_frame).cloned();
    Snapshot {
        frames: vec![
            frame("h", "out"),
            frame("a", "out"),
            frame("b", "out"),
            frame("sp", "out1"),
            frame("m", "out"),
        ],
        counter: rt.output(&id("k"), "out").and_then(Payload::as_scalar),
        taps: taps
            .iter()
            .map(|t| {
                let (seq, v) = t.latest_with_seq();
                (seq, v.and_then(|p| p.as_frame().cloned()))
            })
            .collect(),
    }
}

#[test]
fn default_mode_is_serial() {
    let (g, reg) = mixed_graph();
    let rt = Runtime::instantiate(&g, &reg).unwrap();
    assert_eq!(rt.exec_mode(), ExecMode::Serial);
}

#[test]
fn parallel_matches_serial_outputs_and_tap_seqs() {
    let serial = run_mixed(ExecMode::Serial, 3);

    // Sanity-check the reference so a vacuous match (all `None`) can't pass.
    assert!(serial.frames.iter().all(Option::is_some));
    assert_eq!(serial.counter, Some(9.0)); // 3 ticks x 3 iterations
    assert!(serial.taps.iter().all(|(seq, v)| *seq == 3 && v.is_some()));
    assert_eq!(
        serial.frames[4], serial.frames[0],
        "split->merge round-trips"
    );

    // Every concurrency cap, including degenerate ones, must agree exactly.
    for threads in [0, 1, 2, 3, 64] {
        let parallel = run_mixed(ExecMode::Parallel { threads }, 3);
        assert_eq!(parallel, serial, "threads = {threads}");
    }
    assert_eq!(run_mixed(ExecMode::parallel(), 3), serial);
}

#[test]
fn switching_mode_mid_stream_keeps_state() {
    // Feedback state and tap seqs carry over when the mode changes.
    let (g, reg) = mixed_graph();
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    rt.set_max_iters(1);
    let tap = rt.add_tap(&id("k"), "out");
    rt.set_input(
        &id("h"),
        "in",
        Payload::Frame(Frame::from_rgb8(1, 1, vec![(9, 9, 9)])),
    );
    rt.tick(2).unwrap();
    rt.set_exec_mode(ExecMode::Parallel { threads: 4 });
    rt.tick(2).unwrap();
    rt.set_exec_mode(ExecMode::Serial);
    rt.tick(1).unwrap();
    assert_eq!(tap.seq(), 5);
    assert_eq!(rt.output(&id("k"), "out").unwrap().as_scalar(), Some(5.0));
}

/// Fails on demand, to check error reporting is the same in both modes.
struct Failing;

impl Node for Failing {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Any)],
            vec![PortSpec::new("out", PayloadKind::Scalar)],
        )
    }

    fn eval(&mut self, _inputs: &Inputs, _outputs: &mut Outputs) -> Result<(), NodeError> {
        Err(NodeError::Message("boom".to_string()))
    }
}

#[test]
fn a_failing_branch_reports_the_same_error_in_both_modes() {
    let mut g = Graph::new();
    g.add_node(node("src", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node("ok", "clear_channel", &[("channel", "green")]))
        .unwrap();
    g.add_node(node("bad", "failing", &[])).unwrap();
    g.connect(port("src", "out"), port("ok", "in")).unwrap();
    g.connect(port("src", "out"), port("bad", "in")).unwrap();
    let mut reg = builtin_registry();
    reg.register("failing", |_| Ok(Box::new(Failing) as Box<dyn Node>));

    let run = |mode| {
        let mut rt = Runtime::instantiate(&g, &reg).unwrap();
        rt.set_exec_mode(mode);
        rt.set_input(
            &id("src"),
            "in",
            Payload::Frame(Frame::from_rgb8(1, 1, vec![(1, 2, 3)])),
        );
        rt.run_once().unwrap_err()
    };
    let serial = run(ExecMode::Serial);
    assert_eq!(serial.node, id("bad"));
    assert_eq!(run(ExecMode::Parallel { threads: 4 }), serial);
}

/// Records when each evaluation started and finished.
type Log = Arc<Mutex<Vec<(String, Instant, Instant)>>>;

/// Sleeps for a fixed time, then forwards its scalar input. Stands in for an
/// expensive per-frame filter.
struct Sleeper {
    name: String,
    nap: Duration,
    log: Log,
}

impl Node for Sleeper {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Scalar)],
            vec![PortSpec::new("out", PayloadKind::Scalar)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let start = Instant::now();
        std::thread::sleep(self.nap);
        let v = inputs.get("in").and_then(Payload::as_scalar).unwrap_or(0.0);
        outputs.set("out", Payload::Scalar(v + 1.0));
        self.log
            .lock()
            .unwrap()
            .push((self.name.clone(), start, Instant::now()));
        Ok(())
    }
}

/// `src` fans out to two 100ms sleepers, `x` and `y`.
fn named_sleepy_runtime(log: &Log) -> Runtime {
    let mut g = Graph::new();
    g.add_node(node("src", "sleeper", &[("name", "src"), ("ms", "0")]))
        .unwrap();
    g.add_node(node("x", "sleeper", &[("name", "x"), ("ms", "100")]))
        .unwrap();
    g.add_node(node("y", "sleeper", &[("name", "y"), ("ms", "100")]))
        .unwrap();
    g.connect(port("src", "out"), port("x", "in")).unwrap();
    g.connect(port("src", "out"), port("y", "in")).unwrap();

    let mut reg = Registry::new();
    let shared = log.clone();
    reg.register("sleeper", move |p| {
        Ok(Box::new(Sleeper {
            name: p.get("name").cloned().unwrap_or_default(),
            nap: Duration::from_millis(p.get("ms").and_then(|s| s.parse().ok()).unwrap_or(0)),
            log: shared.clone(),
        }) as Box<dyn Node>)
    });
    Runtime::instantiate(&g, &reg).unwrap()
}

/// The `(start, end)` interval logged for node `name`.
fn span(log: &Log, name: &str) -> (Instant, Instant) {
    let log = log.lock().unwrap();
    let (_, s, e) = log
        .iter()
        .find(|(n, _, _)| n == name)
        .unwrap_or_else(|| panic!("{name} never ran"));
    (*s, *e)
}

#[test]
fn parallel_branches_overlap_in_time() {
    let log: Log = Arc::default();
    let mut rt = named_sleepy_runtime(&log);
    rt.set_exec_mode(ExecMode::Parallel { threads: 2 });

    let t0 = Instant::now();
    rt.run_once().unwrap();
    let elapsed = t0.elapsed();

    // The robust check: each branch started before the other finished. This
    // holds no matter how slow or loaded the machine is.
    let (xs, xe) = span(&log, "x");
    let (ys, ye) = span(&log, "y");
    assert!(xs < ye && ys < xe, "x and y did not overlap");

    // And the payoff: the wall-clock time beats running the sleeps in turn.
    // Serial is at least 200ms by construction, so that is the bound: it leaves
    // ~100ms of slack for thread start-up on a loaded machine.
    assert!(
        elapsed < Duration::from_millis(200),
        "parallel run took {elapsed:?}"
    );

    // Results are the same either way.
    assert_eq!(rt.output(&id("x"), "out").unwrap().as_scalar(), Some(2.0));
    assert_eq!(rt.output(&id("y"), "out").unwrap().as_scalar(), Some(2.0));
}

#[test]
fn serial_and_single_thread_branches_do_not_overlap() {
    // The control for the test above: with no concurrency allowed, the two
    // sleeps are strictly sequential.
    for mode in [ExecMode::Serial, ExecMode::Parallel { threads: 1 }] {
        let log: Log = Arc::default();
        let mut rt = named_sleepy_runtime(&log);
        rt.set_exec_mode(mode);
        rt.run_once().unwrap();

        let (xs, xe) = span(&log, "x");
        let (ys, ye) = span(&log, "y");
        assert!(xe <= ys || ye <= xs, "{mode:?}: x and y overlapped");
    }
}

/// `src` fans out to two fused runs of 50ms sleepers: x1 -> x2 and y1 -> y2.
fn fused_sleepy_runtime(log: &Log) -> Runtime {
    let mut g = Graph::new();
    g.add_node(node("src", "sleeper", &[("name", "src"), ("ms", "0")]))
        .unwrap();
    for n in ["x1", "x2", "y1", "y2"] {
        g.add_node(node(n, "sleeper", &[("name", n), ("ms", "50")]))
            .unwrap();
    }
    g.connect(port("src", "out"), port("x1", "in")).unwrap();
    g.connect(port("src", "out"), port("y1", "in")).unwrap();
    g.connect(port("x1", "out"), port("x2", "in")).unwrap();
    g.connect(port("y1", "out"), port("y2", "in")).unwrap();

    let mut reg = Registry::new();
    let shared = log.clone();
    reg.register("sleeper", move |p| {
        Ok(Box::new(Sleeper {
            name: p.get("name").cloned().unwrap_or_default(),
            nap: Duration::from_millis(p.get("ms").and_then(|s| s.parse().ok()).unwrap_or(0)),
            log: shared.clone(),
        }) as Box<dyn Node>)
    });
    Runtime::instantiate(&g, &reg).unwrap()
}

#[test]
fn fused_branches_run_concurrently() {
    // Each branch is a fused run, so parallel mode must schedule it as one
    // unit; if fused runs were evaluated one after another, the two branches
    // could not overlap.
    let log: Log = Arc::default();
    let mut rt = fused_sleepy_runtime(&log);
    assert!(rt.fusion_enabled());
    assert_eq!(rt.fusable_runs().len(), 2, "x1->x2 and y1->y2 fuse");
    rt.set_exec_mode(ExecMode::Parallel { threads: 2 });

    let t0 = Instant::now();
    rt.run_once().unwrap();
    let elapsed = t0.elapsed();

    // Whole branches overlap: x's run started before y's finished and vice versa.
    let (x_start, x_end) = (span(&log, "x1").0, span(&log, "x2").1);
    let (y_start, y_end) = (span(&log, "y1").0, span(&log, "y2").1);
    assert!(
        x_start < y_end && y_start < x_end,
        "fused branches did not overlap"
    );
    // Within a branch, members still run in order.
    assert!(span(&log, "x1").1 <= span(&log, "x2").0);
    assert!(span(&log, "y1").1 <= span(&log, "y2").0);
    // Serial needs 4 x 50ms = 200ms; overlapping branches need ~100ms.
    assert!(
        elapsed < Duration::from_millis(200),
        "parallel fused run took {elapsed:?}"
    );

    assert_eq!(rt.output(&id("x2"), "out").unwrap().as_scalar(), Some(3.0));
    assert_eq!(rt.output(&id("y2"), "out").unwrap().as_scalar(), Some(3.0));
    assert!(
        rt.edge_buffer(&port("x1", "out"), &port("x2", "in"))
            .unwrap()
            .is_empty(),
        "the private link inside a fused run never materializes"
    );
}

/// Fan-out into fused branches built from real stages, one of them tapped
/// mid-run (which splits its run), plus a split -> merge pair — itself a
/// fused run whose second member has three inputs, so it falls back to `eval`.
///
/// ```text
///        +--> a (gain) --> c (invert) --> d (threshold)      a->c->d fuses
/// h -----+--> e (grayscale) --> f (box_blur) --> g (gain)    tap on f: e->f fuses, g alone
/// (red)  +--> sp (split 3) ==out0..2==> m (merge 3)         sp->m fuses
/// ```
fn fused_fanout_graph() -> Graph {
    type Spec<'a> = (&'a str, &'a str, &'a [(&'a str, &'a str)]);
    let mut g = Graph::new();
    let nodes: [Spec; 9] = [
        ("h", "clear_channel", &[("channel", "red")]),
        ("a", "gain", &[("factor", "1.5")]),
        ("c", "invert", &[]),
        ("d", "threshold", &[("level", "0.5")]),
        ("e", "grayscale", &[]),
        ("f", "box_blur", &[("radius", "1")]),
        ("g", "gain", &[("factor", "0.5")]),
        ("sp", "split", &[("channels", "3")]),
        ("m", "merge", &[("channels", "3")]),
    ];
    for (n, kind, params) in nodes {
        g.add_node(node(n, kind, params)).unwrap();
    }
    for (from, to) in [
        ("h", "a"),
        ("a", "c"),
        ("c", "d"),
        ("h", "e"),
        ("e", "f"),
        ("f", "g"),
        ("h", "sp"),
    ] {
        g.connect(port(from, "out"), port(to, "in")).unwrap();
    }
    for c in 0..3 {
        g.connect(port("sp", &format!("out{c}")), port("m", &format!("in{c}")))
            .unwrap();
    }
    g
}

#[test]
fn parallel_with_fusion_matches_serial_without() {
    let snapshot = |mode: ExecMode, fusion: bool| {
        let mut rt = Runtime::instantiate(&fused_fanout_graph(), &builtin_registry()).unwrap();
        rt.set_exec_mode(mode);
        rt.set_fusion(fusion);
        let tap = rt.add_tap(&id("f"), "out");
        if fusion {
            assert_eq!(rt.fusable_runs().len(), 3, "a->c->d, e->f and sp->m");
        }
        let px: Vec<(u8, u8, u8)> = (0..16).map(|i| (i * 9, i * 13 + 3, 250 - i * 11)).collect();
        rt.set_input(&id("h"), "in", Payload::Frame(Frame::from_rgb8(4, 4, px)));
        rt.tick(3).unwrap();
        let frame = |n: &str| {
            rt.output(&id(n), "out")
                .and_then(Payload::as_frame)
                .cloned()
        };
        let (seq, v) = tap.latest_with_seq();
        (
            ["d", "f", "g", "m"].map(frame),
            seq,
            v.and_then(|p| p.as_frame().cloned()),
        )
    };

    let reference = snapshot(ExecMode::Serial, false);
    assert!(reference.0.iter().all(Option::is_some));
    assert_eq!(reference.1, 3);
    for mode in [
        ExecMode::Serial,
        ExecMode::Parallel { threads: 1 },
        ExecMode::Parallel { threads: 2 },
        ExecMode::Parallel { threads: 8 },
    ] {
        for fusion in [false, true] {
            assert_eq!(
                snapshot(mode, fusion),
                reference,
                "{mode:?} with fusion={fusion}"
            );
        }
    }
}
