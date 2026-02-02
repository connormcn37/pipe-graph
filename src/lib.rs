//! pipe-graph
//!
//! Goal: a data-agnostic pipeline graph runtime (video, audio, control, etc.).
//! UI/editor frontends (e.g. Bevy) should adapt to this core instead of owning it.
//!
//! Design notes (early / evolving):
//! - Graphs may contain cycles (feedback loops), e.g. auto-exposure / auto-focus.
//! - For scheduling/execution, we can decompose the graph into strongly-connected
//!   components (SCCs). Acyclic components can be executed in topological order
//!   and are candidates for "linear chain" fusion. Cyclic components can be
//!   isolated and executed with a tick/iterative scheduler.
//! - "Taps" (live previews) should be first-class and non-blocking:
//!   any stage output can publish "latest value" and/or a bounded stream.

pub mod data;
pub mod processors;
#[cfg(feature = "bevy")]
pub mod systems;
pub mod traits;

pub mod graph;
