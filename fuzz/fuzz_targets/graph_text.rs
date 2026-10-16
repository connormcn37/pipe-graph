//! Parse arbitrary text with `Graph::from_text`. It must never panic, and
//! whenever it accepts an input, serializing the result and parsing that again
//! must give the same graph. The checks live in `checks.rs`, shared with the
//! stable smoke test.
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../checks.rs"]
mod checks;

fuzz_target!(|data: &[u8]| checks::graph_text(data));
