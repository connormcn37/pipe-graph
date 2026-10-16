//! Helpers shared by the criterion benches.
//!
//! Pulled in by each bench with `mod common;`. Cargo only auto-discovers
//! top-level files in `benches/` as targets, so this directory module is not
//! built as a bench of its own.
//!
//! Every bench compiles this file separately and none of them uses all of it,
//! so unused-item lints are silenced for the module as a whole rather than
//! sprinkling `#[allow]` on each helper (CI runs clippy with `-D warnings`).
#![allow(dead_code)]

use pipe_graph::data::{Frame, FrameData};
use pipe_graph::graph::{NodeId, NodeSpec, Params, PortId};

/// Frame width used by the per-pixel benches: 1080p is the size the project
/// actually targets, and large enough that per-call overhead is negligible
/// next to the pixel loop.
pub const W: u32 = 1920;
/// Frame height; see [`W`].
pub const H: u32 = 1080;

/// Build a [`NodeSpec`] from string params, exactly as a graph file would.
pub fn spec(id: &str, kind: &str, params: &[(&str, &str)]) -> NodeSpec {
    let mut p = Params::new();
    for (k, v) in params {
        p.insert(k.to_string(), v.to_string());
    }
    NodeSpec {
        id: NodeId(id.to_string()),
        kind: kind.to_string(),
        params: p,
    }
}

/// A `(node, port)` endpoint for [`pipe_graph::graph::Graph::connect`].
pub fn port(node: &str, port: &str) -> (NodeId, PortId) {
    (NodeId(node.to_string()), PortId(port.to_string()))
}

/// A deterministic, non-flat `channels`-channel test pattern (gradients plus a
/// checker), the same family as `examples/bench_pipeline.rs`.
///
/// Non-flat matters: a constant image could let a stage (or the branch
/// predictor) take an unrepresentatively cheap path, e.g. threshold always
/// landing on the same side.
pub fn pattern_u8(w: u32, h: u32, channels: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * channels) as usize);
    for y in 0..h {
        for x in 0..w {
            let checker = if ((x / 64) + (y / 64)) % 2 == 0 {
                40
            } else {
                0
            };
            for c in 0..channels {
                let v = match c % 3 {
                    0 => ((x * 255 / w) as u8).saturating_add(checker),
                    1 => (y * 255 / h) as u8,
                    _ => ((x + y) % 256) as u8,
                };
                buf.push(v);
            }
        }
    }
    buf
}

/// A `W`x`H` frame of the test pattern as `u8`.
pub fn frame_u8(channels: u32) -> Frame {
    Frame::from_data(W, H, channels, FrameData::U8(pattern_u8(W, H, channels)))
}

/// The same pattern as [`frame_u8`], normalized to `f32` in `[0, 1]`.
pub fn frame_f32(channels: u32) -> Frame {
    let data = pattern_u8(W, H, channels)
        .into_iter()
        .map(|v| v as f32 / 255.0)
        .collect();
    Frame::from_data(W, H, channels, FrameData::F32(data))
}
