//! Edge buffers: where a payload lives between the node that produces it and
//! the node(s) that consume it.
//!
//! The scheduler owns one `EdgeBuffer` per graph edge. A producing node
//! `push`es its output; consuming nodes `get_last` it. This is the concrete
//! home of the pull-based dataflow model (and the reincarnation of the old
//! `Stage::get_last_frame` / `push_frame` stubs):
//!
//! - **Acyclic** edges are written once per run, then read in topological order.
//! - **Cyclic** (feedback) edges are read *before* they're written within a
//!   tick, so `get_last` returns the previous tick's value — empty on tick 0
//!   until a node seeds it.
//!
//! Payloads are stored behind `Arc`, so fan-out to multiple consumers and taps
//! (Phase 7) share the value without cloning the underlying frame.

use std::sync::Arc;

use crate::data::Payload;

/// A single-slot "latest value" buffer for one edge.
#[derive(Debug, Default, Clone)]
pub struct EdgeBuffer {
    latest: Option<Arc<Payload>>,
}

impl EdgeBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a freshly produced payload, replacing any previous value.
    pub fn push(&mut self, payload: Payload) {
        self.latest = Some(Arc::new(payload));
    }

    /// Store an already-shared payload (cheap fan-out from one output to many
    /// edges).
    pub fn push_arc(&mut self, payload: Arc<Payload>) {
        self.latest = Some(payload);
    }

    /// The most recently pushed payload, or `None` if nothing has been pushed.
    pub fn get_last(&self) -> Option<&Payload> {
        self.latest.as_deref()
    }

    /// A cheap shared handle to the latest payload (for fan-out / taps).
    pub fn get_last_arc(&self) -> Option<Arc<Payload>> {
        self.latest.clone()
    }

    pub fn is_empty(&self) -> bool {
        self.latest.is_none()
    }

    /// Drop the stored value (e.g. when resetting before a fresh run).
    pub fn clear(&mut self) {
        self.latest = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Frame;

    #[test]
    fn empty_by_default() {
        let b = EdgeBuffer::new();
        assert!(b.is_empty());
        assert!(b.get_last().is_none());
    }

    #[test]
    fn push_then_get_last() {
        let mut b = EdgeBuffer::new();
        b.push(Payload::Scalar(2.5));
        assert_eq!(b.get_last().unwrap().as_scalar(), Some(2.5));
        assert!(!b.is_empty());
    }

    #[test]
    fn push_replaces_previous() {
        let mut b = EdgeBuffer::new();
        b.push(Payload::Scalar(1.0));
        b.push(Payload::Scalar(2.0));
        assert_eq!(b.get_last().unwrap().as_scalar(), Some(2.0));
    }

    #[test]
    fn arc_sharing_does_not_clone_frame() {
        let mut b = EdgeBuffer::new();
        b.push(Payload::Frame(Frame::from_rgb8(1, 1, vec![(1, 2, 3)])));
        let a = b.get_last_arc().unwrap();
        let a2 = b.get_last_arc().unwrap();
        // Both handles point at the same allocation.
        assert!(Arc::ptr_eq(&a, &a2));
    }

    #[test]
    fn clear_empties() {
        let mut b = EdgeBuffer::new();
        b.push(Payload::Scalar(1.0));
        b.clear();
        assert!(b.is_empty());
    }
}
