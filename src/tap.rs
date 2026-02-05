//! Tap API (early).
//!
//! A *tap* is a lightweight, non-blocking way to observe values flowing through
//! the graph (for previews, debugging, and UI inspection).
//!
//! This first cut focuses on the simplest primitive: **latest-value** storage.
//! Producers can `publish()` new values; consumers can `latest()` to fetch the
//! most recently published value.
//!
//! Design goals:
//! - Non-blocking publish path (no unbounded buffering).
//! - Cloneable handles that can be passed around freely.
//! - Dependency-light (std only) so core remains headless/minimal.

use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct TapInner<T> {
    seq: u64,
    latest: Option<Arc<T>>,
}

/// A tap that stores the latest published value.
///
/// Notes:
/// - Values are stored behind an `Arc` so `latest()` is cheap and does not copy.
/// - `publish()` overwrites the previous value (last-write-wins).
#[derive(Debug, Clone)]
pub struct Tap<T> {
    inner: Arc<Mutex<TapInner<T>>>,
}

impl<T> Default for Tap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Tap<T> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TapInner {
                seq: 0,
                latest: None,
            })),
        }
    }

    /// Publish a new value and return the resulting monotonic sequence number.
    ///
    /// Sequence numbers start at 1.
    pub fn publish(&self, value: T) -> u64 {
        let mut g = self.inner.lock().expect("tap mutex poisoned");
        g.seq = g.seq.wrapping_add(1);
        g.latest = Some(Arc::new(value));
        g.seq
    }

    /// Current sequence number.
    pub fn seq(&self) -> u64 {
        self.inner.lock().expect("tap mutex poisoned").seq
    }

    /// Fetch the latest published value (if any).
    pub fn latest(&self) -> Option<Arc<T>> {
        self.inner
            .lock()
            .expect("tap mutex poisoned")
            .latest
            .as_ref()
            .cloned()
    }

    /// Clear the stored latest value (sequence number remains unchanged).
    pub fn clear(&self) {
        self.inner.lock().expect("tap mutex poisoned").latest = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_starts_empty() {
        let t: Tap<u32> = Tap::new();
        assert_eq!(t.seq(), 0);
        assert!(t.latest().is_none());
    }

    #[test]
    fn tap_publish_overwrites_latest() {
        let t = Tap::new();
        let s1 = t.publish(1);
        let s2 = t.publish(2);

        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(*t.latest().unwrap(), 2);
    }

    #[test]
    fn tap_can_be_cloned_and_shared() {
        let t1 = Tap::new();
        let t2 = t1.clone();

        t1.publish("hello".to_string());
        assert_eq!(t2.latest().as_deref().map(|s| s.as_str()), Some("hello"));
    }
}
