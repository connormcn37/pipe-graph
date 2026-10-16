//! Stable, deterministic fuzz smoke tests for the parsers of untrusted input.
//!
//! The real fuzzers live in `fuzz/` (cargo-fuzz, nightly only; see
//! `fuzz/README.md`). They find bugs, but they do not run in normal CI. This
//! file is the cheap, always-on complement: a dependency-free mutation loop
//! over the same seed corpus (`fuzz/seeds/`) plus every committed crash
//! reproducer (`fuzz/regressions/`), checking the same properties, which are
//! literally the same code (`fuzz/checks.rs`):
//!
//! * `Y4mReader` never panics, stops after finitely many frames, and (checked
//!   here only, via a tracking allocator) never makes an allocation far larger
//!   than its input: a header may *declare* a 3 GiB frame, and that must not
//!   be trusted until the bytes arrive.
//! * `Graph::from_text` never panics, and whatever it accepts survives a
//!   `to_text` -> `from_text` round trip unchanged.
//!
//! The RNG is a fixed-seed xorshift, so a failure reproduces exactly, and each
//! case runs under `catch_unwind` so the report names the offending input
//! instead of just "a test panicked".

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use pipe_graph::graph::Graph;
use pipe_graph::io::Y4mReader;

#[path = "../fuzz/checks.rs"]
mod checks;

/// Mutated inputs per parser. A few thousand keeps each test well under a
/// second in debug builds while still hitting every mutation kind many times.
const CASES: usize = 4000;

// --- allocation tracking -----------------------------------------------------
//
// To assert "no unbounded allocation" we need to see allocation sizes. This
// test binary installs a thin wrapper over the system allocator that records
// the largest single request made *on the current thread*, so tests running
// in parallel threads do not see each other's allocations.

struct TrackingAlloc;

thread_local! {
    // `const` init: no lazy initialisation (which could itself allocate and
    // recurse into the allocator) and no destructor.
    static LARGEST_ALLOC: Cell<usize> = const { Cell::new(0) };
}

fn note_alloc(size: usize) {
    // `try_with`: the slot may be unavailable while a thread is shutting down.
    let _ = LARGEST_ALLOC.try_with(|m| m.set(m.get().max(size)));
}

// SAFETY: every method delegates to `System` unchanged; we only observe sizes.
unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_alloc(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc;

/// Run `f` and return the largest single allocation it made on this thread.
fn largest_alloc_during(f: impl FnOnce()) -> usize {
    LARGEST_ALLOC.with(|m| m.set(0));
    f();
    LARGEST_ALLOC.with(|m| m.get())
}

// --- deterministic mutation engine -------------------------------------------

/// xorshift64: tiny, fast, good enough to drive mutations, and reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform-ish in `0..n` (`n > 0`); modulo bias is irrelevant here.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

/// Produce one mutated input: start from a random seed, then apply 1..=4
/// random edits. `dict` holds format-specific tokens worth splicing in (the
/// mutation engine knows nothing else about either format).
fn mutate(rng: &mut Rng, seeds: &[Vec<u8>], dict: &[&[u8]]) -> Vec<u8> {
    let mut buf = rng.pick(seeds).clone();
    for _ in 0..=rng.below(4) {
        let len = buf.len();
        match rng.below(8) {
            // Flip one bit.
            0 if len > 0 => {
                let i = rng.below(len);
                buf[i] ^= 1 << rng.below(8);
            }
            // Overwrite a byte with an arbitrary value.
            1 if len > 0 => {
                let i = rng.below(len);
                buf[i] = rng.next() as u8;
            }
            // Insert a few random bytes.
            2 => {
                let at = rng.below(len + 1);
                let n = 1 + rng.below(8);
                let bytes: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
                buf.splice(at..at, bytes);
            }
            // Delete a range.
            3 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(16)).min(len);
                buf.drain(at..end);
            }
            // Truncate.
            4 => buf.truncate(rng.below(len + 1)),
            // Duplicate a chunk in place (repeats lines, frames, tokens).
            5 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(64)).min(len);
                let chunk = buf[at..end].to_vec();
                let to = rng.below(len + 1);
                buf.splice(to..to, chunk);
            }
            // Splice: keep a prefix of this input, append a suffix of another.
            6 => {
                let other = rng.pick(seeds);
                let cut = rng.below(len + 1);
                let from = rng.below(other.len() + 1);
                buf.truncate(cut);
                buf.extend_from_slice(&other[from..]);
            }
            // Insert a dictionary token.
            _ => {
                let at = rng.below(len + 1);
                let tok = rng.pick(dict);
                buf.splice(at..at, tok.iter().copied());
            }
        }
    }
    buf
}

// --- corpus ------------------------------------------------------------------

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Every file directly inside `dir` (missing directories yield nothing), in a
/// stable order so the RNG sequence maps to the same inputs on every machine.
fn files_in(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_file())
            .collect(),
        Err(_) => return Vec::new(),
    };
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let bytes = std::fs::read(&p).unwrap();
            (p, bytes)
        })
        .collect()
}

/// Seed corpus for a cargo-fuzz target, plus its committed crash reproducers.
fn corpus(target: &str) -> Vec<Vec<u8>> {
    let mut all = files_in(&repo().join("fuzz/seeds").join(target));
    assert!(!all.is_empty(), "no seeds for {target}");
    all.extend(files_in(&repo().join("fuzz/regressions").join(target)));
    all.into_iter().map(|(_, b)| b).collect()
}

/// Run `check` on every corpus input verbatim, then on `CASES` mutants,
/// naming the input of the first case that panics.
fn fuzz(target: &str, rng_seed: u64, dict: &[&[u8]], check: impl Fn(&[u8])) {
    let seeds = corpus(target);
    let mut rng = Rng(rng_seed);
    let inputs = seeds
        .iter()
        .cloned()
        .chain((0..CASES).map(|_| mutate(&mut rng, &seeds, dict)));
    for (case, input) in inputs.enumerate() {
        if panic::catch_unwind(AssertUnwindSafe(|| check(&input))).is_err() {
            panic!(
                "{target}: case {case} panicked on input ({} bytes): b\"{}\"",
                input.len(),
                input.escape_ascii()
            );
        }
    }
}

// --- Y4M reader ----------------------------------------------------------------

/// Bytes Y4M mutations should splice in: tags, sizes near the limits, and the
/// keywords that move the parser between header, frame line and payload.
const Y4M_DICT: &[&[u8]] = &[
    b"YUV4MPEG2 ",
    b"FRAME\n",
    b"FRAME Ixyz\n",
    b"\n",
    b" ",
    b" W",
    b" H",
    b" W32768",
    b" H32768",
    b" W32769",
    b" W4294967295",
    b" W0",
    b"18446744073709551616",
    b" F0:0",
    b" F30000:1001",
    b" C420jpeg",
    b" C444",
    b" Cmono",
    b" C422",
    b" XCOLORRANGE=FULL",
    b"\xff\xfe",
];

/// Drive a reader over `data` to the end, checking every bound we promise.
fn check_y4m(data: &[u8]) {
    let largest = largest_alloc_during(|| checks::y4m(data));
    // Before any payload arrives the reader may allocate a 1 MiB first chunk
    // (`FIRST_CHUNK` in src/io/y4m.rs); beyond that, memory must track input.
    // Buffers at most double what was received, and an RGB frame is 3x its
    // planar bytes, so 16x input is a comfortable ceiling.
    let bound = (1 << 20) + 16 * data.len();
    assert!(
        largest <= bound,
        "allocated {largest} bytes for a {}-byte input",
        data.len()
    );
}

#[test]
fn y4m_reader_survives_mutated_streams() {
    fuzz("y4m_reader", 0x9E37_79B9_7F4A_7C15, Y4M_DICT, check_y4m);
}

#[test]
fn y4m_header_declaring_a_huge_frame_is_not_trusted() {
    // Regression: a 3 GiB frame buffer used to be allocated from the header
    // alone, before a single payload byte had been read.
    for header in [
        &b"YUV4MPEG2 W32768 H32768 C444\nFRAME\n\x10\x10"[..],
        b"YUV4MPEG2 W32768 H32768 C444\n",
        b"YUV4MPEG2 W32768 H32768 C420jpeg\nFRAME\n",
    ] {
        check_y4m(header);
    }
}

// --- graph text parser -------------------------------------------------------

/// Graph-text tokens: keywords, the arrow, every delimiter and escape, line
/// endings, and non-ASCII text.
const GRAPH_DICT: &[&[u8]] = &[
    b"node ",
    b"edge ",
    b" -> ",
    b"->",
    b".",
    b"=",
    b"\"",
    b"\\",
    b"\\\"",
    b"\\n",
    b"\\x",
    b"#",
    b"\n",
    b"\r\n",
    b"\r",
    b"\t",
    b" ",
    b"\"\"",
    b"\xc3\xa9",
    b"\xff",
    b"node a b\n",
    b"edge a.out -> a.in\n",
];

#[test]
fn graph_text_survives_mutated_files() {
    fuzz(
        "graph_text",
        0xD1B5_4A32_D192_ED03,
        GRAPH_DICT,
        checks::graph_text,
    );
}

#[test]
fn graph_seed_corpus_parses() {
    // The seeds are only useful as seeds if they are valid: otherwise the
    // mutation loop above would mostly be exploring error paths.
    for (path, bytes) in files_in(&repo().join("fuzz/seeds/graph_text")) {
        let text = String::from_utf8(bytes).unwrap();
        if let Err(e) = Graph::from_text(&text) {
            panic!("{}: {e}", path.display());
        }
    }
}

#[test]
fn y4m_seed_corpus_decodes() {
    for (path, bytes) in files_in(&repo().join("fuzz/seeds/y4m_reader")) {
        let mut r = Y4mReader::new(&bytes[..]).unwrap();
        let mut frames = 0;
        while r
            .read_rgb()
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .is_some()
        {
            frames += 1;
        }
        assert!(frames > 0, "{}: no frames", path.display());
    }
}
