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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::graph::{NodeId, PortId};

/// Where in the graph a tap is attached.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TapPoint {
    /// Tap a named output or input port on a node.
    NodePort { node: NodeId, port: PortId },
    /// Tap a specific edge (from node+port to node+port).
    ///
    /// Useful when a node has multiple outgoing edges from the same port and you
    /// want to observe per-consumer values.
    Edge {
        from_node: NodeId,
        from_port: PortId,
        to_node: NodeId,
        to_port: PortId,
    },
}

impl TapPoint {
    pub fn node_port(node: NodeId, port: PortId) -> Self {
        Self::NodePort { node, port }
    }

    pub fn edge(
        from_node: NodeId,
        from_port: PortId,
        to_node: NodeId,
        to_port: PortId,
    ) -> Self {
        Self::Edge {
            from_node,
            from_port,
            to_node,
            to_port,
        }
    }
}

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
#[derive(Debug)]
pub struct Tap<T> {
    inner: Arc<Mutex<TapInner<T>>>,
}

impl<T> Clone for Tap<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Registry for taps keyed by attachment point.
///
/// This gives the runtime/editor a stable place to grab "the tap for X" without
/// having to plumb `Tap<T>` handles everywhere.
#[derive(Debug)]
pub struct TapRegistry<T> {
    inner: Arc<Mutex<HashMap<TapPoint, Tap<T>>>>,
}

impl<T> Default for TapRegistry<T> {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<T> Clone for TapRegistry<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
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

impl<T> TapRegistry<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get (or create) the tap for a given attachment point.
    pub fn tap_at(&self, point: TapPoint) -> Tap<T> {
        let mut g = self.inner.lock().expect("tap registry mutex poisoned");
        g.entry(point).or_insert_with(Tap::new).clone()
    }

    /// Convenience: tap a `(node, port)`.
    pub fn tap(&self, node: NodeId, port: PortId) -> Tap<T> {
        self.tap_at(TapPoint::node_port(node, port))
    }

    /// Convenience: tap a specific edge.
    pub fn tap_edge(
        &self,
        from_node: NodeId,
        from_port: PortId,
        to_node: NodeId,
        to_port: PortId,
    ) -> Tap<T> {
        self.tap_at(TapPoint::edge(from_node, from_port, to_node, to_port))
    }

    pub fn remove(&self, point: &TapPoint) -> Option<Tap<T>> {
        self.inner
            .lock()
            .expect("tap registry mutex poisoned")
            .remove(point)
    }

    pub fn points(&self) -> Vec<TapPoint> {
        self.inner
            .lock()
            .expect("tap registry mutex poisoned")
            .keys()
            .cloned()
            .collect()
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

    #[test]
    fn registry_returns_same_tap_for_same_point() {
        let r: TapRegistry<u32> = TapRegistry::new();

        let a1 = r.tap(NodeId("a".into()), PortId("out".into()));
        let a2 = r.tap(NodeId("a".into()), PortId("out".into()));

        a1.publish(7);
        assert_eq!(*a2.latest().unwrap(), 7);
    }

    #[test]
    fn registry_supports_edge_taps() {
        let r: TapRegistry<&'static str> = TapRegistry::new();

        let t1 = r.tap_edge(
            NodeId("a".into()),
            PortId("out".into()),
            NodeId("b".into()),
            PortId("in".into()),
        );
        let t2 = r.tap_edge(
            NodeId("a".into()),
            PortId("out".into()),
            NodeId("b".into()),
            PortId("in".into()),
        );

        t1.publish("hi");
        assert_eq!(t2.latest().map(|x| *x), Some("hi"));
    }
}
