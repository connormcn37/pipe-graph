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
use std::fmt;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use crate::graph::{Graph, NodeId, PortId};

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

impl fmt::Display for TapPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TapPoint::NodePort { node, port } => write!(f, "{}.{}", node.0, port.0),
            TapPoint::Edge {
                from_node,
                from_port,
                to_node,
                to_port,
            } => write!(f, "{}.{} -> {}.{}", from_node.0, from_port.0, to_node.0, to_port.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TapPointParseError {
    Empty,
    BadFormat(String),
}

fn parse_endpoint(s: &str) -> Result<(NodeId, PortId), TapPointParseError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(TapPointParseError::Empty);
    }
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 2 {
        return Err(TapPointParseError::BadFormat(s.to_string()));
    }
    let node = parts[0].trim();
    let port = parts[1].trim();
    if node.is_empty() || port.is_empty() {
        return Err(TapPointParseError::BadFormat(s.to_string()));
    }
    Ok((NodeId(node.to_string()), PortId(port.to_string())))
}

impl FromStr for TapPoint {
    type Err = TapPointParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err(TapPointParseError::Empty);
        }

        if let Some((lhs, rhs)) = s.split_once("->") {
            let (from_node, from_port) = parse_endpoint(lhs)?;
            let (to_node, to_port) = parse_endpoint(rhs)?;
            return Ok(TapPoint::edge(from_node, from_port, to_node, to_port));
        }

        let (node, port) = parse_endpoint(s)?;
        Ok(TapPoint::node_port(node, port))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TapPointValidationError {
    MissingNode { node: NodeId },
    MissingPort { node: NodeId, port: PortId },
    MissingEdge {
        from_node: NodeId,
        from_port: PortId,
        to_node: NodeId,
        to_port: PortId,
    },
}

/// Validate that a tap point refers to something that exists in the graph.
///
/// Note: core `Graph` does not currently model port schemas per node kind.
/// This validator checks only:
/// - node ids exist
/// - port ids are non-empty
/// - for edge taps: an exact edge exists with those endpoints
pub fn validate_tap_point(g: &Graph, p: &TapPoint) -> Result<(), TapPointValidationError> {
    match p {
        TapPoint::NodePort { node, port } => {
            if !g.nodes.contains_key(node) {
                return Err(TapPointValidationError::MissingNode { node: node.clone() });
            }
            if port.0.trim().is_empty() {
                return Err(TapPointValidationError::MissingPort {
                    node: node.clone(),
                    port: port.clone(),
                });
            }
            Ok(())
        }
        TapPoint::Edge {
            from_node,
            from_port,
            to_node,
            to_port,
        } => {
            if !g.nodes.contains_key(from_node) {
                return Err(TapPointValidationError::MissingNode {
                    node: from_node.clone(),
                });
            }
            if !g.nodes.contains_key(to_node) {
                return Err(TapPointValidationError::MissingNode {
                    node: to_node.clone(),
                });
            }
            if from_port.0.trim().is_empty() {
                return Err(TapPointValidationError::MissingPort {
                    node: from_node.clone(),
                    port: from_port.clone(),
                });
            }
            if to_port.0.trim().is_empty() {
                return Err(TapPointValidationError::MissingPort {
                    node: to_node.clone(),
                    port: to_port.clone(),
                });
            }

            let exists = g.edges.values().any(|c| {
                c.from.0 == *from_node
                    && c.from.1 == *from_port
                    && c.to.0 == *to_node
                    && c.to.1 == *to_port
            });

            if !exists {
                return Err(TapPointValidationError::MissingEdge {
                    from_node: from_node.clone(),
                    from_port: from_port.clone(),
                    to_node: to_node.clone(),
                    to_port: to_port.clone(),
                });
            }

            Ok(())
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

    /// Fetch `(seq, latest)` under a single lock.
    ///
    /// This avoids a TOCTOU race where callers do `seq()` and `latest()` in two
    /// separate calls and end up pairing a value with the wrong sequence.
    pub fn latest_with_seq(&self) -> (u64, Option<Arc<T>>) {
        let g = self.inner.lock().expect("tap mutex poisoned");
        (g.seq, g.latest.as_ref().cloned())
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

    /// Parse a `TapPoint` from a string and get (or create) its tap.
    ///
    /// Supported formats:
    /// - `node.port`
    /// - `from_node.from_port -> to_node.to_port`
    pub fn tap_at_str(&self, s: &str) -> Result<Tap<T>, TapPointParseError> {
        let p: TapPoint = s.parse()?;
        Ok(self.tap_at(p))
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

    /// Parse a `TapPoint` from a string and remove it from the registry.
    pub fn remove_str(&self, s: &str) -> Result<Option<Tap<T>>, TapPointParseError> {
        let p: TapPoint = s.parse()?;
        Ok(self.remove(&p))
    }

    pub fn points(&self) -> Vec<TapPoint> {
        self.inner
            .lock()
            .expect("tap registry mutex poisoned")
            .keys()
            .cloned()
            .collect()
    }

    /// Deterministic list of tap attachment points.
    pub fn points_sorted(&self) -> Vec<TapPoint> {
        let mut pts = self.points();
        pts.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
        pts
    }

    /// Validate every registered tap point against a graph.
    ///
    /// Returns `Ok(())` if all tap points refer to existing nodes/ports/edges.
    /// Otherwise returns a stable list of `(point, error)` pairs.
    pub fn validate_against(
        &self,
        g: &Graph,
    ) -> Result<(), Vec<(TapPoint, TapPointValidationError)>> {
        let mut errs: Vec<(TapPoint, TapPointValidationError)> = Vec::new();
        for p in self.points_sorted() {
            if let Err(e) = validate_tap_point(g, &p) {
                errs.push((p, e));
            }
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Connection, EdgeId, NodeSpec, Params};

    #[test]
    fn tap_point_parse_node_port() {
        let p: TapPoint = "a.out".parse().unwrap();
        assert_eq!(p, TapPoint::node_port(NodeId("a".into()), PortId("out".into())));
    }

    #[test]
    fn tap_point_parse_edge_variants() {
        let p1: TapPoint = "a.out->b.in".parse().unwrap();
        let p2: TapPoint = "a.out -> b.in".parse().unwrap();
        assert_eq!(p1, p2);
        assert_eq!(
            p1,
            TapPoint::edge(
                NodeId("a".into()),
                PortId("out".into()),
                NodeId("b".into()),
                PortId("in".into())
            )
        );
    }

    fn small_graph() -> Graph {
        let mut g = Graph::new();
        g.nodes.insert(
            NodeId("a".into()),
            NodeSpec {
                id: NodeId("a".into()),
                kind: "src".into(),
                params: Params::new(),
            },
        );
        g.nodes.insert(
            NodeId("b".into()),
            NodeSpec {
                id: NodeId("b".into()),
                kind: "sink".into(),
                params: Params::new(),
            },
        );
        g.edges.insert(
            EdgeId(1),
            Connection {
                from: (NodeId("a".into()), PortId("out".into())),
                to: (NodeId("b".into()), PortId("in".into())),
            },
        );
        g
    }

    #[test]
    fn validate_tap_point_node_port_exists() {
        let g = small_graph();
        let p = TapPoint::node_port(NodeId("a".into()), PortId("out".into()));
        assert_eq!(validate_tap_point(&g, &p), Ok(()));
    }

    #[test]
    fn validate_tap_point_edge_requires_exact_edge() {
        let g = small_graph();
        let ok = TapPoint::edge(
            NodeId("a".into()),
            PortId("out".into()),
            NodeId("b".into()),
            PortId("in".into()),
        );
        assert_eq!(validate_tap_point(&g, &ok), Ok(()));

        let missing = TapPoint::edge(
            NodeId("a".into()),
            PortId("out".into()),
            NodeId("b".into()),
            PortId("in2".into()),
        );
        assert!(matches!(
            validate_tap_point(&g, &missing),
            Err(TapPointValidationError::MissingEdge { .. })
        ));
    }

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

        let (seq, v) = t.latest_with_seq();
        assert_eq!(seq, 2);
        assert_eq!(*v.unwrap(), 2);
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
    fn registry_can_create_taps_from_strings() {
        let r: TapRegistry<u32> = TapRegistry::new();

        let t1 = r.tap_at_str("a.out").unwrap();
        let t2 = r.tap_at_str("a.out").unwrap();

        t1.publish(9);
        assert_eq!(*t2.latest().unwrap(), 9);

        let e1 = r.tap_at_str("a.out -> b.in").unwrap();
        let e2 = r.tap_edge(
            NodeId("a".into()),
            PortId("out".into()),
            NodeId("b".into()),
            PortId("in".into()),
        );

        e1.publish(11);
        assert_eq!(*e2.latest().unwrap(), 11);
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

    #[test]
    fn registry_validate_against_graph_reports_missing_points() {
        let g = small_graph();
        let r: TapRegistry<u32> = TapRegistry::new();

        // ok: node+port exists
        let _ = r.tap(NodeId("a".into()), PortId("out".into()));
        // not ok: edge doesn't exist
        let _ = r.tap_edge(
            NodeId("a".into()),
            PortId("out".into()),
            NodeId("b".into()),
            PortId("in2".into()),
        );

        let err = r.validate_against(&g).unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0].0.to_string(), "a.out -> b.in2");
        assert!(matches!(err[0].1, TapPointValidationError::MissingEdge { .. }));
    }
}
