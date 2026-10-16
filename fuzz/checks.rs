//! The properties both fuzz harnesses check, defined once.
//!
//! Included by path (`#[path = ...] mod checks;`) into the cargo-fuzz targets
//! in `fuzz_targets/` *and* into the stable smoke test `tests/fuzz_smoke.rs`,
//! so libFuzzer and normal CI can never drift apart on what "correct" means.
//! Each function panics if, and only if, the parser misbehaves on `data`.

// Each including crate uses only some of these functions.
#![allow(dead_code)]

use std::io::Cursor;

use pipe_graph::graph::Graph;
use pipe_graph::io::Y4mReader;

/// Drive a [`Y4mReader`] over `data` until EOF or the first error.
///
/// Every frame must match its header's shape. Each frame consumes at least
/// `"FRAME\n"` plus one payload byte, so more frames than input bytes would
/// mean the reader has stopped making progress (a hang, in a pipe).
pub fn y4m(data: &[u8]) {
    let Ok(mut reader) = Y4mReader::new(Cursor::new(data)) else {
        return;
    };
    let (w, h) = (reader.header().width, reader.header().height);
    let mut frames = 0usize;
    while let Ok(Some(frame)) = reader.read_rgb() {
        assert_eq!((frame.width, frame.height, frame.channels), (w, h, 3));
        frames += 1;
        assert!(
            frames <= data.len(),
            "no progress after {frames} frames from {} bytes",
            data.len()
        );
    }
}

/// Parse `data` (lossy UTF-8, so invalid bytes still reach the lexer as
/// U+FFFD) as graph text. If accepted, `to_text` then `from_text` must give
/// the same graph.
///
/// "Same" is checked by comparing `to_text` output. That output is
/// deterministic and writes every serialized field, so this is a structural
/// equality check that keeps working as the format grows new fields.
pub fn graph_text(data: &[u8]) {
    let text = String::from_utf8_lossy(data);
    let Ok(graph) = Graph::from_text(&text) else {
        return;
    };
    let written = graph.to_text();
    let reparsed = match Graph::from_text(&written) {
        Ok(g) => g,
        Err(e) => panic!("to_text output does not parse ({e}):\n{written}"),
    };
    assert_eq!(graph.nodes.len(), reparsed.nodes.len());
    assert_eq!(graph.edges.len(), reparsed.edges.len());
    assert_eq!(written, reparsed.to_text(), "round trip changed the graph");
}
