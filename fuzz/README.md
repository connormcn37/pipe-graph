# Fuzzing pipe-graph's untrusted-input parsers

Two parsers read bytes nobody vetted, and neither may ever panic, hang, or
allocate unboundedly. Whatever the input, they must return an error:

| target        | parser                                  | input comes from               |
|---------------|-----------------------------------------|--------------------------------|
| `y4m_reader`  | `pipe_graph::io::Y4mReader`             | `.y4m` files, ffmpeg pipes     |
| `graph_text`  | `pipe_graph::graph::Graph::from_text`   | `.graph` files, editors, users |

`graph_text` also checks a round-trip property: whenever `from_text`
accepts an input, `to_text` followed by `from_text` must give the same graph.

This directory is a standalone [cargo-fuzz] crate. It depends on the main
crate by path, but the main crate has no workspace and does not depend on it,
and its own empty `[workspace]` table keeps it out of any parent workspace.
So the stable `cargo build/clippy/test` commands at the repo root never build
it. Keep it that way: do not add it as a member of a root workspace.
Running the fuzzers needs nightly, because libFuzzer instrumentation uses
`-Z` flags. Plain compilation works on stable, and CI runs
`cargo clippy --manifest-path fuzz/Cargo.toml` so API changes cannot silently
break the targets.

Both targets call the shared property checks in `checks.rs`. The stable
smoke test includes the same file.

## Always-on smoke tests

`tests/fuzz_smoke.rs` runs in normal CI on stable. It is a deterministic,
dependency-free mutation loop over the same seeds and regressions, with a few
thousand cases per parser in about a second. It cannot find what coverage
guidance finds, but it catches regressions on every `cargo test`.

## Setup (once)

```sh
rustup toolchain install nightly --profile minimal
cargo +nightly install cargo-fuzz --locked
```

The repo's `rust-toolchain.toml` pins stable, so pass `+nightly` explicitly.

## Running

From the repo root:

```sh
# Seeds are committed under fuzz/seeds/<target>. libFuzzer writes new inputs
# to the first corpus directory and only reads the others.
mkdir -p fuzz/corpus/y4m_reader fuzz/corpus/graph_text
cargo +nightly fuzz run -O y4m_reader fuzz/corpus/y4m_reader \
    fuzz/seeds/y4m_reader fuzz/regressions/y4m_reader \
    -- -max_total_time=600 -max_len=16384 -malloc_limit_mb=64
cargo +nightly fuzz run -O graph_text fuzz/corpus/graph_text fuzz/seeds/graph_text \
    -- -max_total_time=600
```

* `-max_len=16384` lets Y4M inputs hold a header plus a few small frames. The
  default of 4096 is fine too.
* `-malloc_limit_mb=64` turns any single allocation over 64 MiB into a
  crash. Before a frame's bytes arrive, the reader allocates at most 1 MiB
  for it (`FIRST_CHUNK` in `src/io/y4m.rs`), and a 16 KiB input can't
  justify more than that. A bigger allocation means a header-declared size
  is being trusted.
* Use `-jobs=N -workers=N` to run in parallel. Use `cargo +nightly fuzz
  build -O` to only compile.

`fuzz/corpus`, `fuzz/artifacts` and `fuzz/coverage` are git-ignored.

## When a crash is found

libFuzzer saves the input as `fuzz/artifacts/<target>/crash-<sha1>` (or
`oom-*` / `timeout-*`). Then:

1. Reproduce: `cargo +nightly fuzz run <target> fuzz/artifacts/<target>/crash-...`.
   Run `cargo +nightly fuzz tmin <target> <file>` to minimise the input.
2. Fix the parser so it returns an error, and add a focused regression test
   next to the parser (see `huge_declared_frame_does_not_allocate_up_front` in
   `src/io/y4m.rs`).
3. Commit the minimised input as `fuzz/regressions/<target>/<short-name>`.
   `tests/fuzz_smoke.rs` replays every file there verbatim and also uses it as
   a mutation seed, so the bug stays fixed in stable CI.

[cargo-fuzz]: https://rust-fuzz.github.io/book/cargo-fuzz.html
