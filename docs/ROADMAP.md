# Roadmap: from demo to tool

**Goal of this phase:** someone can launch the editor, drop in a video file,
build a pipeline from a palette, watch it run at the source's frame rate —
on the GPU where a stage supports it — save it, and run the same file
headlessly from a CLI, fast enough that the stages, not the plumbing, are the
bottleneck.

Each unit below is roughly one PR. Units within a phase are mostly
independent, so a phase can be worked as a parallel batch.

## Decisions

| | Decision |
|---|---|
| Dependencies | Small, well-known crates are fine (`png`, `criterion`, `proptest`, `pollster`, …). The default build stays headless with no dependencies; anything heavier sits behind a feature. |
| Platforms | Linux first. macOS and Windows are checked headless in CI. GPU tests run on Linux with software Vulkan (lavapipe). |
| Workspace split | Deferred to Phase 4, once the data model has settled. |
| Audio | Video first. Audio exists as a payload type and port kind (stubs) so graphs can carry it; processing comes later. |
| GPU | Frames can live on the GPU. Every stage gets a CPU and a GPU implementation, kept in agreement by a shared conformance harness. |

## GPU architecture

**Technology:** wgpu (pinned to the version Bevy uses, so the editor and the
runtime can share one device) with WGSL compute shaders, behind a `gpu`
feature.

**One `Frame` type.** Shape, dtype and metadata are always available; only
the pixel storage varies:

```rust
pub struct Frame { shape: Shape, meta: Meta, storage: Storage }

pub enum Storage {
    Cpu(FrameData),
    #[cfg(feature = "gpu")]
    Gpu(GpuBuffer),                      // wgpu buffer + layout
    External(Arc<dyn ExternalStorage>),  // open-ended: hardware decoder
}                                        // surfaces, CUDA, Bevy textures…

impl Frame {
    fn location(&self) -> Location;          // Cpu | Gpu | External
    fn to_cpu(&self, ctx) -> Cow<FrameData>; // borrow if already there
    fn to_gpu(&self, ctx) -> Cow<GpuBuffer>; // otherwise transfer
}
```

A closed enum for the storage we own (stage code pattern-matches and gets the
concrete buffer, and the compiler checks every location is handled), plus one
trait-object variant for storage we don't. A trait-only design would push
stages into runtime downcasts to reach the bytes or buffer they actually work
on.

**Status (H1–H3):** `Storage`, `Location`, `ExternalStorage`, `to_cpu` /
`to_gpu` / `into_cpu` / `into_gpu`, `GpuContext` (own or borrowed device),
`GpuBuffer` (u8 packed four per word), cached WGSL `Kernel`s, and
`Node::eval_gpu` exist. Every built-in stage except the legacy
`clear_channel` processor has a GPU path (H3), bit-identical to its CPU path
except `f32` `box_blur` (sums in `f32`, not `f64`) and `f32` `grayscale`
(possible fused multiply-add), checked by `tests/gpu_conformance.rs`.
`Runtime::set_backend(Cpu | Gpu(ctx) | Auto(ctx))` places nodes (`Auto`
keeps a lone GPU-capable stage on the CPU), every CPU evaluation is handed
CPU frames (downloading where needed, including before in-place edits), and
GPU work is recorded into one command buffer per pass and submitted once.
Not yet: caching a download shared by several CPU consumers (each downloads
for itself), and a GPU-backed editor (D8).

**Two implementations per stage.** `Node::eval` is the CPU path; an optional
`Node::eval_gpu` is the GPU path. Stages without one run on the CPU.

**Placement.** `Runtime::set_backend(Cpu | Gpu | Auto)` places each node, and
the scheduler inserts uploads/downloads where the graph crosses between the
two. A fused chain on the GPU is one command submission per run. Previews
download a GPU frame only when something is watching.

**Later (Phase 4):** describe simple per-pixel stages as expressions that
generate both the Rust and the WGSL implementation, which is also what shader
fusion of whole chains needs.

## Phase 1 — Foundations

**1a. Data model** (sequential: these all reshape `Payload`/`Frame`/`Inputs`)

| ID | Unit |
|---|---|
| A1 | Zero-copy inputs: `Inputs` holds `Arc<Payload>`; nodes read without copying |
| A2 | Frame metadata: pts, frame index, duration, colour space |
| A8 | Audio stubs: `Payload::Audio`, `PayloadKind::Audio` |
| H1 | GPU foundation: `gpu` feature, context, `Storage` enum, transfers, lavapipe CI |

**1b. Independent units** (parallel)

| ID | Unit |
|---|---|
| A5 | Feedback loops stop when they converge |
| A6 | criterion benchmarks + baseline |
| E2 | Graph format v2: version header, layout positions |
| E3 | Parameter schemas for every node kind |
| F1 | Docs build with `-D warnings` in CI |
| F2 | Fuzzing for the Y4M and graph-text parsers |
| D1 | Editor pan, zoom, fit-to-view |
| D9 | Bundled Unicode font for the editor |

## Phase 2 — Capability

- ~~**H2** GPU scheduler: placement, automatic transfers, one submit per fused chain.~~ Done: one submit per pass.
- ~~**H3** GPU implementations of the existing stages, with the conformance harness.~~ Done.
- **C1–C7** New stages, each with CPU + GPU implementations: colour conversion, geometry (resize…), combining frames, frame → scalar measurements, scalar-controlled params, temporal effects, generic convolution.
- **B1–B4** Video I/O: probing, real-time pacing, seek/loop/range, image sequences.
- **A3** Buffer pool (CPU and GPU). **A7** Randomized equivalence tests across backends and modes. **E1** Real CLI.

## Phase 3 — Tool

- **A4** Pipelined streaming (overlapping uploads with processing on the GPU).
- **B5** Live capture. **C8** Auto-exposure demo, on the GPU.
- **D2–D8** Editor: save/open, node palette, full param editing, undo/redo, multi-select, errors on the canvas, more previews (GPU frames drawn via the device shared with Bevy).
- **E5** User guide. **F4** CI on more platforms. **F5** Release `v0.2.0`.

## Phase 4 — Stretch

- Shader fusion of GPU chains (expression-defined stages).
- **F3** Workspace split. **B6** Audio processing. **E4** Subgraphs. Scripting nodes.

## Risks

- **wgpu upgrades** move in lockstep with Bevy; a Bevy upgrade is a planned step.
- **CPU/GPU float differences** are made explicit by per-stage tolerances in the conformance harness.
- **Download cost for previews** is why previews download only when watched, and why device sharing with Bevy matters.
- **Lavapipe is slow**: fine for correctness, meaningless for benchmarks; GPU speed needs real hardware.
