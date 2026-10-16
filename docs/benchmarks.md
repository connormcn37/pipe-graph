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
> container while seven other build jobs were compiling on the same machine,
> so absolute numbers are heavily inflated and noisy (confidence intervals of
> ±10% or worse; parallel-mode numbers in particular are distorted because the
> cores were already busy). Use these only as an order-of-magnitude reference
> and to sanity-check that a fresh run is in the same ballpark. Rust 1.94,
> release profile, criterion 0.8, commit on branch `phase1/criterion-benches`.

Values are criterion's point estimate (middle of the reported interval).

### Stages (`cargo bench --bench stages`), 1920x1080x3

| Stage | u8 time | u8 throughput | f32 time | f32 throughput |
| --- | ---: | ---: | ---: | ---: |
| gain (factor 1.1) | 2.04 ms | 2.84 GiB/s | 2.56 ms | 9.07 GiB/s |
| grayscale | 3.82 ms | 1.52 GiB/s | 2.20 ms | 10.5 GiB/s |
| invert | 0.58 ms | 9.93 GiB/s | 2.43 ms | 9.53 GiB/s |
| threshold (level 0.5) | 2.07 ms | 2.80 GiB/s | 2.25 ms | 10.3 GiB/s |
| box_blur r=1 | 65.7 ms | 90 MiB/s | 78.0 ms | 304 MiB/s |
| box_blur r=4 | 67.0 ms | 89 MiB/s | 79.9 ms | 297 MiB/s |
| clear_channel (red) | 1.29 ms | 4.48 GiB/s | 4.70 ms | 4.93 GiB/s |
| crop (centred 960x540) | 1.22 ms | 1.18 GiB/s | 1.83 ms | 3.16 GiB/s |
| split (3 planes) | 4.84 ms | 1.20 GiB/s | 6.67 ms | 3.48 GiB/s |
| merge (3 planes) | 5.44 ms | 1.07 GiB/s | 7.36 ms | 3.15 GiB/s |

| Cast | time | throughput |
| --- | ---: | ---: |
| u8 → f32 | 2.41 ms | 2.41 GiB/s |
| f32 → u8 | 22.4 ms | 1.03 GiB/s |

Throughput is bytes of input read per second (for crop, the cropped region),
so an `f32` frame counts four times the bytes of the same `u8` image.

### Scheduler overhead (`cargo bench --bench scheduler`), 100 nodes

| Graph | serial | serial+fused | parallel | parallel+fused |
| --- | ---: | ---: | ---: | ---: |
| chain_100 | 35.1 µs | 96.8 µs | 41.9 µs | 61.4 µs |
| fanout_100 | 39.8 µs | 51.0 µs | 13.5 ms | 29.7 ms |

### Pipeline (`cargo bench --bench pipeline`), one 1080p frame

| Input | serial | serial+fused | parallel | parallel+fused |
| --- | ---: | ---: | ---: | ---: |
| u8 | 119.5 ms | 104.9 ms | 128.3 ms | 107.3 ms |
| f32 | 144.6 ms | 157.7 ms | 145.3 ms | 148.0 ms |

### Observations worth re-checking on a quiet machine

These may be artefacts of the loaded container, but are the first things to
look at with a clean run:

- **Parallel fan-out of cheap nodes is ~300× slower than serial**
  (13.5 ms vs 40 µs for 99 trivial siblings). Expected to lose on cheap nodes,
  but not by this much: it suggests per-node thread dispatch cost in the
  millisecond range when cores are contended.
- **Fusion makes the scalar chain slower** (35 → 97 µs serial). These nodes
  do not implement `eval_in_place`, so fusion can only add bookkeeping here.
- **`box_blur` costs the same at r=1 and r=4**, consistent with its O(1)
  per-pixel sliding window, but it is ~30× slower than the other per-pixel
  stages and dominates the pipeline bench.
- **`cast f32 → u8` is ~10× slower than `u8 → f32`** (rounding + clamping per
  element).
- **Parallel mode gives no speedup on the pipeline** here; with four busy
  cores that is unsurprising, and is the main number to re-measure.
