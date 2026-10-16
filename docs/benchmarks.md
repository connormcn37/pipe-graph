# Benchmarks

Criterion benchmarks live in `benches/`:

| Bench | What it measures |
| --- | --- |
| `benches/stages.rs` | Every builtin per-pixel stage, one `Node::eval` on a 1920x1080 frame (`u8` and `f32`), no scheduler. Throughput in input bytes/s. |
| `benches/scheduler.rs` | `Runtime::run_once` overhead on 100 near-free scalar nodes: a 100-node chain and a 1→99 fan-out, in serial/parallel × fusion off/on. Throughput in nodes/s. |
| `benches/pipeline.rs` | The 1080p `split → 3×(box_blur r=4 → gain) → merge` graph (same shape as `examples/bench_pipeline.rs`), `u8` and `f32`, serial/parallel × fusion off/on. One iteration = one frame. |

## Reproduce

```sh
cargo bench                       # all three, a few minutes
cargo bench --bench stages        # one file
cargo bench --bench stages -- box_blur   # filter by name
```

Results and an HTML report land in `target/criterion/` (open
`target/criterion/report/index.html`). Criterion compares each run against the
previous one in the same target directory, so for a before/after comparison
run once on the base commit and once on the change, on the same machine.

For a CI run, trigger the **Bench** workflow manually (Actions → Bench → Run
workflow); it uploads `target/criterion/` as an artifact. It is deliberately not
part of the regular CI: shared runners are too noisy for regression gating.

## Baseline (2026-10-16)

> **Not a performance claim.** Recorded once in a shared 4-core Linux
> container while several other build jobs were compiling on the same machine,
> so absolute numbers are heavily inflated and noisy (confidence intervals of
> ±10% or worse; parallel-mode numbers in particular are distorted because the
> cores were already busy). Use these only as an order-of-magnitude reference
> and to sanity-check that a fresh run is in the same ballpark. Rust 1.94,
> release profile, criterion 0.8; library code at `main` commit `15b790a`.

Values are criterion's point estimate (middle of the reported interval). The
whole `cargo bench` (including the release build) took about 6 minutes.

### Stages (`cargo bench --bench stages`), 1920x1080x3

| Stage | u8 time | u8 throughput | f32 time | f32 throughput |
| --- | ---: | ---: | ---: | ---: |
| gain (factor 1.1) | 2.12 ms | 2.73 GiB/s | 2.38 ms | 9.75 GiB/s |
| grayscale | 3.28 ms | 1.76 GiB/s | 2.37 ms | 9.76 GiB/s |
| invert | 0.57 ms | 10.1 GiB/s | 2.36 ms | 9.81 GiB/s |
| threshold (level 0.5) | 1.98 ms | 2.93 GiB/s | 2.21 ms | 10.5 GiB/s |
| box_blur r=1 | 73.7 ms | 81 MiB/s | 72.0 ms | 330 MiB/s |
| box_blur r=4 | 62.0 ms | 96 MiB/s | 74.0 ms | 321 MiB/s |
| clear_channel (red) | 1.48 ms | 3.92 GiB/s | 3.95 ms | 5.87 GiB/s |
| crop (centred 960x540) | 1.39 ms | 1.05 GiB/s | 1.65 ms | 3.52 GiB/s |
| split (3 planes) | 4.46 ms | 1.30 GiB/s | 6.88 ms | 3.37 GiB/s |
| merge (3 planes) | 4.05 ms | 1.43 GiB/s | 7.27 ms | 3.19 GiB/s |

| Cast | time | throughput |
| --- | ---: | ---: |
| u8 → f32 | 1.85 ms | 3.14 GiB/s |
| f32 → u8 | 22.7 ms | 1.02 GiB/s |

Throughput is bytes of input read per second (for crop, the cropped region),
so an `f32` frame counts four times the bytes of the same `u8` image.

### Scheduler overhead (`cargo bench --bench scheduler`), 100 nodes

| Graph | serial | serial+fused | parallel | parallel+fused |
| --- | ---: | ---: | ---: | ---: |
| chain_100 | 32.2 µs | 86.9 µs | 44.9 µs | 69.6 µs |
| fanout_100 (1 → 99) | 47.8 µs | 48.2 µs | 4.26 ms | 3.40 ms |

### Pipeline (`cargo bench --bench pipeline`), one 1080p frame

| Input | serial | serial+fused | parallel | parallel+fused |
| --- | ---: | ---: | ---: | ---: |
| u8 | 83.0 ms | 92.6 ms | 72.1 ms | 70.1 ms |
| f32 | 141.1 ms | 149.2 ms | 167.3 ms | 91.0 ms |

### Observations worth re-checking on a quiet machine

These may be artefacts of the loaded container, but are the first things to
look at with a clean run:

- **Parallel fan-out of cheap nodes is ~90× slower than serial**
  (4.3 ms vs 48 µs for 99 trivial siblings). Losing on cheap nodes is
  expected and documented on `ExecMode::Parallel`, but this suggests a
  per-node dispatch cost in the tens of microseconds.
- **Fusion makes the scalar chain slower** (32 → 87 µs serial). These nodes
  do not implement `eval_in_place`, so fusion can only add bookkeeping here.
- **`box_blur` costs about the same at r=1 and r=4**, consistent with its
  O(1)-per-pixel sliding window, but it is ~30× slower than the other
  per-pixel stages and dominates the pipeline bench.
- **`cast f32 → u8` is ~12× slower than `u8 → f32`** (rounding and clamping
  per element).
- **Parallel + fusion is the only clear pipeline win** (f32: 141 → 91 ms);
  the other combinations sit within noise of serial on four busy cores.
