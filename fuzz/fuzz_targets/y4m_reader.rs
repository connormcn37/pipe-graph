//! Feed arbitrary bytes to `Y4mReader` as if they came from a file or an
//! ffmpeg pipe. The reader must return an error or a clean EOF for every
//! input: never panic, hang, or allocate far beyond the input's size.
//!
//! The progress and shape checks live in `checks.rs` (shared with the stable
//! smoke test); libFuzzer itself enforces time and memory limits (`-timeout`,
//! `-rss_limit_mb`, `-malloc_limit_mb`).
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../checks.rs"]
mod checks;

fuzz_target!(|data: &[u8]| checks::y4m(data));
