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
//!
//! # Change detection
//!
//! Each tap carries a monotonic **sequence number** that counts *publishes*,
//! not content changes. A consumer polling the tap gets two independent,
//! cheap questions answered:
//!
//! 1. *Did the producer publish since I last looked?* — compare [`Tap::seq`].
//!    A source that re-broadcasts a static image bumps `seq` every tick, which
//!    is what you want for liveness ("the stream is alive").
//! 2. *Is this the same buffer I already have?* — `Arc::ptr_eq` against the
//!    payload you kept. A node that republishes a cached `Arc` (see
//!    [`crate::exec::Outputs::set_shared`]) keeps pointer identity stable, so
//!    an expensive consumer (a GPU texture upload, say) can skip the work
//!    without hashing or comparing pixels.
//!
//! Always read the pair through [`Tap::latest_with_seq`] rather than calling
//! [`Tap::seq`] and [`Tap::latest`] separately: two separate locks can
//! interleave with a publish and pair a value with the wrong sequence number.

use std::sync::{Arc, Mutex};

use crate::data::Payload;

#[derive(Debug, Default)]
struct TapInner {
    seq: u64,
    latest: Option<Arc<Payload>>,
}

/// A shareable handle to the latest value on a tapped port.
#[derive(Clone, Default)]
pub struct Tap {
    inner: Arc<Mutex<TapInner>>,
}

impl Tap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish a new value (called by the scheduler). Overwrites the previous
    /// one and returns the resulting sequence number.
    ///
    /// Sequence numbers start at 1; a tap that has never published reports 0.
    pub fn publish(&self, payload: Arc<Payload>) -> u64 {
        let mut inner = self.inner.lock().expect("tap mutex poisoned");
        inner.seq = inner.seq.wrapping_add(1);
        inner.latest = Some(payload);
        inner.seq
    }

    /// Read the most recently published value, if any. Never blocks the
    /// producer; the consumer just sees the latest.
    pub fn latest(&self) -> Option<Arc<Payload>> {
        self.inner
            .lock()
            .expect("tap mutex poisoned")
            .latest
            .clone()
    }

    /// The number of times this tap has been published to (or cleared).
    ///
    /// `0` means nothing has ever been published. Prefer
    /// [`Tap::latest_with_seq`] when you also need the value.
    pub fn seq(&self) -> u64 {
        self.inner.lock().expect("tap mutex poisoned").seq
    }

    /// Read `(seq, latest)` under a single lock.
    ///
    /// This is the call a polling consumer should use: taking `seq()` and
    /// `latest()` separately lets a publish land between them, pairing a value
    /// with the wrong sequence number.
    pub fn latest_with_seq(&self) -> (u64, Option<Arc<Payload>>) {
        let inner = self.inner.lock().expect("tap mutex poisoned");
        (inner.seq, inner.latest.clone())
    }

    /// Drop the currently held value.
    ///
    /// This bumps the sequence number: dropping the value *is* a change, and a
    /// consumer polling by `seq` would otherwise keep displaying a stale frame
    /// the tap no longer holds.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("tap mutex poisoned");
        inner.seq = inner.seq.wrapping_add(1);
        inner.latest = None;
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

    #[test]
    fn seq_starts_at_zero_and_counts_publishes() {
        let t = Tap::new();
        assert_eq!(t.seq(), 0);
        assert_eq!(t.publish(Arc::new(Payload::Scalar(1.0))), 1);
        assert_eq!(t.publish(Arc::new(Payload::Scalar(2.0))), 2);
        assert_eq!(t.seq(), 2);
    }

    #[test]
    fn seq_counts_publishes_not_content_changes() {
        // A source re-broadcasting a static image bumps seq every tick even
        // though the payload is identical — that's liveness, by design.
        let t = Tap::new();
        let still = Arc::new(Payload::Scalar(7.0));
        t.publish(still.clone());
        t.publish(still.clone());
        t.publish(still.clone());
        assert_eq!(t.seq(), 3);

        // Pointer identity is the separate question, and it stayed stable, so a
        // consumer can skip re-uploading the same buffer.
        assert!(Arc::ptr_eq(&t.latest().unwrap(), &still));
    }

    #[test]
    fn latest_with_seq_agrees_with_the_parts() {
        let t = Tap::new();
        assert_eq!(t.latest_with_seq().0, 0);
        assert!(t.latest_with_seq().1.is_none());

        t.publish(Arc::new(Payload::Scalar(5.0)));
        let (seq, value) = t.latest_with_seq();
        assert_eq!(seq, t.seq());
        assert_eq!(value.unwrap().as_scalar(), Some(5.0));
    }

    #[test]
    fn clear_bumps_seq_so_pollers_observe_it() {
        let t = Tap::new();
        t.publish(Arc::new(Payload::Scalar(1.0)));
        let before = t.seq();

        t.clear();

        let (after, value) = t.latest_with_seq();
        assert!(value.is_none());
        assert_ne!(
            after, before,
            "a poller comparing seq must notice the clear"
        );
    }
}
