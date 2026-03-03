# pipe-graph

A dependency-light **pipeline graph runtime** (data-agnostic: video, audio, control, etc.), with room for an optional editor/UI layer later.

This repo currently focuses on the headless core:
- graph model + structural validation
- SCC decomposition (cycles/feedback loops are allowed)
- planning into a component DAG (topo order)
- a minimal "tap" API for live preview/debugging (latest-value)

## Build / test

```bash
cargo test
```

## Core concepts (current)

### Cycles via SCC planning
Graphs may contain cycles. We collapse them into strongly-connected components (SCCs) so an executor can:
- run **acyclic components** in topological order
- run **cyclic components** with a tick/iterative scheduler

See: `README_core.md` for the current planning/validation API.

### Tap API (latest-value)
A tap is a lightweight way to observe values without blocking producers.

```rust
use pipe_graph::tap::Tap;

let tap: Tap<u32> = Tap::new();
assert!(tap.latest().is_none());

tap.publish(123);
assert_eq!(*tap.latest().unwrap(), 123);
```

## Docs
- `README_core.md` (graph validation, SCC planning, tap API)
