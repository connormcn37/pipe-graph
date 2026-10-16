//! Inputs are presented without copying: a consumer sees the very allocation
//! its producer published (or the caller injected), not a clone of it.
//!
//! Before this, `Runtime::gather` deep-copied every input payload into each
//! node on every evaluation — a full frame copy per edge per run, paid even
//! by stages that only read their input to compute a fresh output.

use std::sync::{Arc, Mutex};

use pipe_graph::data::{Frame, Payload, PayloadKind};
use pipe_graph::exec::{
    ExecMode, Inputs, Node, NodeError, Outputs, PortSet, PortSpec, Registry, Runtime,
};
use pipe_graph::graph::{Graph, NodeId, NodeSpec, Params, PortId};

type Seen = Arc<Mutex<Vec<(String, Arc<Payload>)>>>;

/// Records the handle it was given, and forwards a fresh 1x1 frame.
struct Recorder {
    name: String,
    seen: Seen,
}

impl Node for Recorder {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Frame)],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let shared = inputs
            .get_shared("in")
            .ok_or_else(|| NodeError::MissingInput("in".into()))?;
        self.seen.lock().unwrap().push((self.name.clone(), shared));
        outputs.set(
            "out",
            Payload::Frame(Frame::from_rgb8(1, 1, vec![(1, 2, 3)])),
        );
        Ok(())
    }
}

fn spec(id: &str) -> NodeSpec {
    let mut params = Params::new();
    params.insert("name".into(), id.into());
    NodeSpec {
        id: NodeId(id.into()),
        kind: "recorder".into(),
        params,
    }
}

fn port(n: &str, p: &str) -> (NodeId, PortId) {
    (NodeId(n.into()), PortId(p.into()))
}

fn registry(seen: &Seen) -> Registry {
    let mut reg = Registry::new();
    let seen = seen.clone();
    reg.register("recorder", move |p| {
        Ok(Box::new(Recorder {
            name: p.get("name").cloned().unwrap_or_default(),
            seen: seen.clone(),
        }) as Box<dyn Node>)
    });
    reg
}

/// `src` fans out to `a` and `b` (so neither link is fusable: every node is
/// evaluated through the ordinary `gather` path).
fn fan_out() -> Graph {
    let mut g = Graph::new();
    for n in ["src", "a", "b"] {
        g.add_node(spec(n)).unwrap();
    }
    g.connect(port("src", "out"), port("a", "in")).unwrap();
    g.connect(port("src", "out"), port("b", "in")).unwrap();
    g
}

fn handle_seen_by(seen: &Seen, name: &str) -> Vec<Arc<Payload>> {
    seen.lock()
        .unwrap()
        .iter()
        .filter(|(n, _)| n == name)
        .map(|(_, h)| h.clone())
        .collect()
}

#[test]
fn consumers_see_the_producers_allocation() {
    for mode in [ExecMode::Serial, ExecMode::Parallel { threads: 2 }] {
        let seen = Seen::default();
        let mut rt = Runtime::instantiate(&fan_out(), &registry(&seen)).unwrap();
        rt.set_exec_mode(mode);
        rt.watch(&NodeId("src".into()), "out");
        rt.set_input(
            &NodeId("src".into()),
            "in",
            Payload::Frame(Frame::from_rgb8(1, 1, vec![(9, 9, 9)])),
        );
        rt.run_once().unwrap();

        let published = rt.output_arc(&NodeId("src".into()), "out").unwrap();
        for consumer in ["a", "b"] {
            let got = handle_seen_by(&seen, consumer);
            assert_eq!(got.len(), 1, "{mode:?}: {consumer} ran once");
            assert!(
                Arc::ptr_eq(&got[0], &published),
                "{mode:?}: {consumer} was handed a copy, not the producer's value"
            );
        }
        // Both consumers share one allocation.
        assert!(Arc::ptr_eq(
            &handle_seen_by(&seen, "a")[0],
            &handle_seen_by(&seen, "b")[0]
        ));
    }
}

#[test]
fn an_injected_input_is_not_copied_per_run() {
    let seen = Seen::default();
    let mut rt = Runtime::instantiate(&fan_out(), &registry(&seen)).unwrap();
    rt.set_input(
        &NodeId("src".into()),
        "in",
        Payload::Frame(Frame::from_rgb8(1, 1, vec![(9, 9, 9)])),
    );
    rt.tick(3).unwrap();

    let got = handle_seen_by(&seen, "src");
    assert_eq!(got.len(), 3);
    assert!(
        got.windows(2).all(|w| Arc::ptr_eq(&w[0], &w[1])),
        "every run sees the one injected allocation"
    );
}
