# pipe-graph
node graph editor for data (video) pipelines

## Concept:

a program for video data pipeline processing. The base class which most objects share inheritance is named `Entity`, which at minimum has: 
  - a string `label` such that no two entities are allowed to have the same label, thus it can be used as unique id 
  - `inputs`: a list of input labels referencing other entity instances 
  - `connect()`: tells the entity to (re)initialize from input labels 
  - `disconnect()`: release pointers, stop 

A `Stage` is an `Entity` with extra properties: 

  - `parameters`: a dictionary of key:value pairs used by that stage to tune whatever transformation is applied to the input data. 

`Stage` also has functions:
  - `get_last_frame()`: returns most recent output frame created by this stage. Used for "pull" based data flow 
  - `push_frame()`: used for "push" based data flow 

There are several different classes representing different kinds of stages such as: 
  - `CropStage` (crop input frame), 
  - `CastStage` (change data type between float,uint8), 
  - `SplitStage` (given an input frame of shape (width, height, k) create k outputs (width,height) )
  - `MergeStage` (given k inputs of shape (width,height) create 1 output of shape (width,height,k) 

A `Pipeline` is an `Entity` that can be initialized by giving a list of stage types with labels and parameters to initialize them and connect them together. It additionally has the functions: 

  - `start()`: orchestrates the stages data processing sequentially 
  - `stop()`: stop processing.

## Architecture (current implementation)

The core is a **data-agnostic, headless runtime** (`default` build pulls in no
Bevy); UI frontends adapt to it. The pieces:

- **`data`** — `Frame` is a channel/dtype-generic interleaved buffer (`u8`/`f32`,
  any channel count); `Payload` (`Frame` | `Scalar` | `Bytes`) is what flows
  along edges. `from_rgb8`/`to_rgb8` bridge the simple RGB case.
- **`graph`** — pure topology: `Graph` of `NodeSpec { id, kind, params }` and
  `Connection`s between named `(node, port)` endpoints. Cycles are allowed
  (feedback loops). Dependency-light; the editor mirrors these types.
- **`traits::Processor`** — the original single-in/single-out `&mut Frame`
  transform (e.g. `ClearChannel`, `ProcessList`), still supported.
- **`exec`** — the execution layer:
  - `Node` — the real execution unit: named N-in/M-out ports (`ports()`) and
    `eval(inputs) -> outputs`. `ProcessorNode<P>` adapts any `Processor` into a
    1-in/1-out node for free.
  - `Registry` — maps a node's `kind` string to a constructor, parsing
    `params` into typed config; `validate()` checks ports and payload kinds.
  - `EdgeBuffer` — where a payload lives between nodes (pull-based, `Arc`-shared).
  - `Runtime` / `compile()` — decomposes the graph into strongly-connected
    components (Tarjan), runs acyclic parts in topological order and cyclic
    parts with a bounded tick loop (feedback edges read the previous tick).
    The resulting `Plan` keeps the *condensation* (the DAG between components)
    rather than only the flattened order, so later passes can see what is
    independent (`deps`/`successors`) and what is chained one-to-one
    (`linear_chains`) — the structural precondition for running branches
    concurrently, or fusing a run of stages without materializing the values
    in between. `Runtime::fusable_runs` cuts those chains at whatever is
    actually being watched, and is re-derived from the plan alone, so
    attaching a preview to a running pipeline never rebuilds nodes or edge
    buffers (a feedback loop keeps converging).
  - **Capture is opt-in.** A port whose value never leaves its component is
    captured from the start — a pipeline's results, and anything circulating
    inside a feedback loop. An *intermediate* is dropped once the consuming
    node has read it, so `Runtime::output` returns `None` for one until
    `watch`/`add_tap` asks for it (`set_capture_all` restores the
    unconditional behaviour while debugging). What is observed is exactly what
    a fusing executor may not eliminate, so the two are one set rather than
    two.
  - `Tap` — non-blocking latest-value previews on any output port, carrying a
    monotonic `seq`. `seq` counts *publishes*, so a still image broadcast into
    a stream still reads as live; `Arc::ptr_eq` on the payload answers the
    separate question of whether the buffer actually changed. Read both at
    once with `latest_with_seq()`. A node that re-emits a cached buffer should
    publish it with `Outputs::set_shared` to keep that pointer identity stable
    through the scheduler.
- **`stages`** — the registered node kinds: `crop`, `cast`, `split`, `merge`,
  `blend`, `image_read`, `image_write`, plus the `Processor`-backed
  `clear_channel`, `grayscale` and `invert`.
- **`editor`** — Bevy-free controller logic: `EditorCommand`/`apply_command`
  (route user intents through the core `Graph`) and `view_diff` (which node
  views a frontend should spawn/despawn to mirror the graph).
- **`systems`** (Bevy-only, feature-gated) — a thin view/controller:
  `PipeGraphEditorPlugin` holds the `Graph` in a `GraphResource`, applies queued
  `EditorCommand`s, and syncs `NodeView` entities to match. The core never
  depends on Bevy; ECS entities are views, not the data model.

Realizing the vision above: an `Entity`'s `label` is a `NodeId`; its `inputs` /
`connect` / `disconnect` are core `Graph` operations; a `Stage`'s `parameters`
are `NodeSpec.params`; `get_last_frame` / `push_frame` are now
`EdgeBuffer::get_last` / `push`; and a `Pipeline` is a `Graph` executed by a
`Runtime`.

### Pipeline files

A pipeline is a YAML (or TOML) file of nodes and `node.port` edges; see
[`sample.yaml`](sample.yaml):

```yaml
nodes:
  - id: split
    kind: split
    params:
      channels: "3"
  - id: merge
    kind: merge
    params:
      channels: "3"
edges:
  - from: split.out0
    to: merge.in0
  # ...
```

### Try it

```sh
cargo run -- check sample.yaml   # parse, build every node, validate ports
cargo run -- run sample.yaml     # push a 2x2 test frame into split.in, print merge.out
cargo run -- run sample.yaml --watch   # re-run whenever the file changes
cargo run -- run pipe.yaml --input-node a --input-port in --output-node b --output-port out
cargo run --features bevy -- edit sample.yaml   # Bevy editor (view scaffold only; draws nothing yet)
cargo test                       # headless test suite
cargo test --features bevy       # also compile/run the Bevy-gated code
```
