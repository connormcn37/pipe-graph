//! Scheduler overhead: `Runtime::run_once` on graphs of near-free nodes.
//!
//! The nodes here do one `f64` add, so practically all of the measured time is
//! the runtime itself: gathering inputs, committing outputs to edge buffers,
//! capture bookkeeping, and (in parallel mode) handing work to threads. Two
//! shapes stress different parts of it:
//!
//! - `chain_100`: `src -> n1 -> ... -> n99`, a 100-node linear chain. Purely
//!   sequential, so parallel mode cannot win; it measures per-node cost and,
//!   with fusion on, whether fusing a long run of 1-in/1-out nodes saves
//!   anything for nodes that do not opt into in-place evaluation.
//! - `fanout_100`: one source feeding 99 independent sinks (100 nodes, like
//!   the chain). The widest possible level, so it measures the cost of parallel dispatch against
//!   doing 100 trivial evals inline.
//!
//! Each shape runs in all four `{serial, parallel} x {fusion off, on}`
//! combinations. Throughput is reported as elements = nodes, so criterion's
//! "elem/s" reads directly as nodes scheduled per second.
//!
//! Run with `cargo bench --bench scheduler`.

mod common;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pipe_graph::data::{Payload, PayloadKind};
use pipe_graph::exec::{
    ExecMode, Inputs, Node, NodeError, Outputs, PortSet, PortSpec, Registry, Runtime,
};
use pipe_graph::graph::{Graph, NodeId};

use common::{port, spec};

/// Number of nodes in each benchmarked graph.
const N: usize = 100;

/// A source with no inputs that emits a constant scalar on `out`.
struct Const;

impl Node for Const {
    fn ports(&self) -> PortSet {
        PortSet::new(vec![], vec![PortSpec::new("out", PayloadKind::Scalar)])
    }

    fn eval(&mut self, _: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        outputs.set("out", Payload::Scalar(1.0));
        Ok(())
    }
}

/// `out = in + 1`: the cheapest node that still reads and writes a port, so
/// the scheduler cannot skip any of its data plumbing.
struct AddOne;

impl Node for AddOne {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Scalar)],
            vec![PortSpec::new("out", PayloadKind::Scalar)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let x = inputs
            .get("in")
            .and_then(Payload::as_scalar)
            .ok_or_else(|| NodeError::MissingInput("in".to_string()))?;
        outputs.set("out", Payload::Scalar(x + 1.0));
        Ok(())
    }
}

/// A registry holding only the two bench nodes, so nothing here depends on
/// builtin stage behaviour.
fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register("const", |_| Ok(Box::new(Const) as Box<dyn Node>));
    reg.register("add_one", |_| Ok(Box::new(AddOne) as Box<dyn Node>));
    reg
}

/// `src -> n1 -> ... -> n{N-1}`: `N` nodes in one linear chain.
fn chain() -> Graph {
    let mut g = Graph::new();
    g.add_node(spec("src", "const", &[])).unwrap();
    let mut prev = "src".to_string();
    for i in 1..N {
        let id = format!("n{i}");
        g.add_node(spec(&id, "add_one", &[])).unwrap();
        g.connect(port(&prev, "out"), port(&id, "in")).unwrap();
        prev = id;
    }
    g
}

/// `src -> {n1 .. n{N-1}}`: one source fanned out to `N - 1` sinks (`N`
/// nodes in total, matching [`chain`]).
fn fanout() -> Graph {
    let mut g = Graph::new();
    g.add_node(spec("src", "const", &[])).unwrap();
    for i in 1..N {
        let id = format!("n{i}");
        g.add_node(spec(&id, "add_one", &[])).unwrap();
        g.connect(port("src", "out"), port(&id, "in")).unwrap();
    }
    g
}

/// The four execution configurations every graph is measured under.
fn configs() -> [(&'static str, ExecMode, bool); 4] {
    [
        ("serial", ExecMode::Serial, false),
        ("serial+fused", ExecMode::Serial, true),
        ("parallel", ExecMode::parallel(), false),
        ("parallel+fused", ExecMode::parallel(), true),
    ]
}

fn bench_graph(c: &mut Criterion, name: &str, graph: &Graph, check: (&str, f64)) {
    let reg = registry();
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(N as u64));
    for (label, mode, fusion) in configs() {
        let mut rt = Runtime::instantiate(graph, &reg).expect("graph should instantiate");
        rt.set_exec_mode(mode);
        rt.set_fusion(fusion);

        // One untimed run doubles as a correctness check: a misconfigured
        // graph that errored or skipped nodes would otherwise benchmark as
        // suspiciously fast.
        rt.run_once().expect("run_once failed");
        let got = rt
            .output(&NodeId(check.0.to_string()), "out")
            .and_then(Payload::as_scalar);
        assert_eq!(got, Some(check.1), "{name}/{label}: wrong result");

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| black_box(&mut rt).run_once().unwrap())
        });
    }
    group.finish();
}

fn chain_100(c: &mut Criterion) {
    // src emits 1.0 and each of the N-1 add_one nodes adds 1.
    let last = format!("n{}", N - 1);
    bench_graph(c, "chain_100", &chain(), (&last, N as f64));
}

fn fanout_100(c: &mut Criterion) {
    let last = format!("n{}", N - 1);
    bench_graph(c, "fanout_100", &fanout(), (&last, 2.0));
}

/// A run is microseconds, so criterion fits many iterations per sample; 1s of
/// measurement keeps this file well under a minute.
fn config() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
}

criterion_group! {
    name = scheduler;
    config = config();
    targets = chain_100, fanout_100
}
criterion_main!(scheduler);
