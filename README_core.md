# Core runtime notes (WIP)

## Graph validation
Use `Graph::validate(opts)` to catch structural problems early:
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

## Tap API (latest-value)
A **tap** is a lightweight way to observe values flowing through the graph without blocking producers.

`Tap<T>` stores only the latest published value:

```rust
use pipe_graph::tap::Tap;

let tap: Tap<u32> = Tap::new();
assert!(tap.latest().is_none());

let seq = tap.publish(123);
assert_eq!(seq, 1);
assert_eq!(*tap.latest().unwrap(), 123);
```

This is meant to be the minimal primitive for previews/debugging; bounded streams can come later.
