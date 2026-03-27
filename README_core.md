# Core runtime notes (WIP)

## Graph validation
Use `Graph::validate(opts)` (or `Graph::validate_strict()`) to catch structural problems early:
- empty node ids
- empty node kinds
- missing node references
- empty port names
- duplicate edges
- optional: require every node has at least one incident edge
- optional: disallow exact self-loop edges

## Cycles: SCC decomposition
Cycles are expected (feedback loops). Use:
- `Graph::strongly_connected_components()` (Tarjan)
- `Graph::component_graph()` to collapse SCCs into a DAG
- `Graph::plan(opts)` for validation + SCC collapse + topo order

The plan marks which components are cyclic so an executor can run:
- acyclic SCCs in topo order
- cyclic SCCs with a tick/iterative scheduler

Quick usage sketch:

```rust
use pipe_graph::graph::Graph;
use pipe_graph::graph::GraphValidationOptions;

let g = Graph::new();
let plan = g.plan(GraphValidationOptions::default()).unwrap();

let linear = plan.acyclic_node_order();
let cycles = plan.cyclic_node_groups();
```

## Tap API (latest-value)
A **tap** is a lightweight way to observe values flowing through the graph without blocking producers.

Also see `src/bin/tap_demo.rs` for a runnable example (node+port tap + edge tap).

### `Tap<T>` (latest + sequence)
`Tap<T>` stores only the latest published value (and a monotonic seq counter):

```rust
use pipe_graph::tap::Tap;

let tap: Tap<u32> = Tap::new();
assert!(tap.latest().is_none());

let seq = tap.publish(123);
assert_eq!(seq, 1);
assert_eq!(*tap.latest().unwrap(), 123);

// avoid pairing the wrong seq/value across two lock acquisitions
let (s, v) = tap.latest_with_seq();
assert_eq!(s, 1);
assert_eq!(*v.unwrap(), 123);
```

### `TapRegistry<T>` (stable attachment points)
Use a registry when you want “the tap for X” without explicitly passing tap handles around:

```rust
use pipe_graph::tap::{TapPoint, TapRegistry};
use pipe_graph::graph::{NodeId, PortId};

let r: TapRegistry<u32> = TapRegistry::new();

// node+port tap
let t1 = r.tap(NodeId("ema".into()), PortId("out".into()));

// edge tap (per-consumer observation)
let t2 = r.tap_edge(
    NodeId("ema".into()), PortId("out".into()),
    NodeId("sink".into()), PortId("in".into()),
);

// registry is keyed by TapPoint; asking twice returns the same Tap
let same = r.tap_at(TapPoint::node_port(NodeId("ema".into()), PortId("out".into())));
```

This is meant to be the minimal primitive for previews/debugging; bounded streams can come later.
