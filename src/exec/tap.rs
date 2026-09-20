//! Taps: first-class, non-blocking live previews of a node's output port.
//!
//! A `Tap` is a cheap, cloneable handle onto the *latest* value published on
//! one port. The scheduler publishes by briefly locking and overwriting a
//! shared cell, so a slow observer never stalls the pipeline — it simply reads
//! whatever the most recent value is (latest-value / drop-oldest semantics). A
//! bounded streaming ring can be layered on later; latest-value covers the
//! common "show me the current frame" preview case.
//!
//! Because payloads are stored behind `Arc`, publishing to a tap is a refcount
//! bump, not a frame copy.

use std::sync::{Arc, Mutex};

use crate::data::Payload;

/// A shareable handle to the latest value on a tapped port.
#[derive(Clone, Default)]
pub struct Tap {
    latest: Arc<Mutex<Option<Arc<Payload>>>>,
}

impl Tap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish a new value (called by the scheduler). Overwrites the previous.
    pub fn publish(&self, payload: Arc<Payload>) {
        *self.latest.lock().expect("tap mutex poisoned") = Some(payload);
    }

    /// Read the most recently published value, if any. Never blocks the
    /// producer; the consumer just sees the latest.
    pub fn latest(&self) -> Option<Arc<Payload>> {
        self.latest.lock().expect("tap mutex poisoned").clone()
    }

    /// Drop the currently held value.
    pub fn clear(&self) {
        *self.latest.lock().expect("tap mutex poisoned") = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_then_latest() {
        let t = Tap::new();
        assert!(t.latest().is_none());
        t.publish(Arc::new(Payload::Scalar(3.0)));
        assert_eq!(t.latest().unwrap().as_scalar(), Some(3.0));
    }

    #[test]
    fn publish_overwrites() {
        let t = Tap::new();
        t.publish(Arc::new(Payload::Scalar(1.0)));
        t.publish(Arc::new(Payload::Scalar(2.0)));
        assert_eq!(t.latest().unwrap().as_scalar(), Some(2.0));
    }

    #[test]
    fn clone_shares_the_same_cell() {
        let a = Tap::new();
        let b = a.clone();
        a.publish(Arc::new(Payload::Scalar(9.0)));
        assert_eq!(b.latest().unwrap().as_scalar(), Some(9.0));
    }
}
