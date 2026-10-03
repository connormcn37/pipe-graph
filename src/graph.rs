//! Core graph types (runtime-facing).
//!
//! This module is intentionally small and dependency-light.
//! The idea is that UI layers (Bevy/egui/etc.) can mirror these types.

use std::collections::{HashMap, HashSet};

mod text;
pub use text::{ParseError, ParseErrorKind};

/// Stable identifier for a node/stage in the graph.
///
/// Early version uses a string label (matches README intent: unique label).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub String);

/// Identifier for an input or output port.
///
/// Ports are named so they can map cleanly to UI pins.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PortId(pub String);

impl std::borrow::Borrow<str> for PortId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u64);

#[derive(Debug, Clone)]
pub struct Connection {
    pub from: (NodeId, PortId),
    pub to: (NodeId, PortId),
}

/// Parameter map for configuring stages.
pub type Params = HashMap<String, String>;

/// A node specification (runtime graph). UI/editor should produce these.
#[derive(Debug, Clone)]
pub struct NodeSpec {
    pub id: NodeId,
    /// Stage "type" (e.g. "crop", "cast", "merge").
    pub kind: String,
    pub params: Params,
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    pub nodes: HashMap<NodeId, NodeSpec>,
    pub edges: HashMap<EdgeId, Connection>,
    next_edge_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    DuplicateNodeId(String),
    MissingNode(String),
    SelfLoopNotAllowed(String),
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, spec: NodeSpec) -> Result<(), GraphError> {
        if self.nodes.contains_key(&spec.id) {
            return Err(GraphError::DuplicateNodeId(spec.id.0));
        }
        self.nodes.insert(spec.id.clone(), spec);
        Ok(())
    }

    pub fn connect(
        &mut self,
        from: (NodeId, PortId),
        to: (NodeId, PortId),
    ) -> Result<EdgeId, GraphError> {
        if !self.nodes.contains_key(&from.0) {
            return Err(GraphError::MissingNode(from.0.0));
        }
        if !self.nodes.contains_key(&to.0) {
            return Err(GraphError::MissingNode(to.0.0));
        }

        // NOTE: cycles are allowed, but we may still want to treat self-loops specially.
        // Keep this permissive for now.

        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        self.edges.insert(id.clone(), Connection { from, to });
        Ok(id)
    }

    /// Remove a node and every edge touching it. Returns whether the node
    /// existed. Needed by a live editor (and by graph re-compilation).
    pub fn remove_node(&mut self, id: &NodeId) -> bool {
        let existed = self.nodes.remove(id).is_some();
        self.edges
            .retain(|_, conn| &conn.from.0 != id && &conn.to.0 != id);
        existed
    }

    /// Remove a single edge by id. Returns whether it existed.
    pub fn disconnect(&mut self, edge: &EdgeId) -> bool {
        self.edges.remove(edge).is_some()
    }

    /// Set (insert or overwrite) one parameter on a node, returning the value it
    /// replaced, if any.
    ///
    /// The graph stores parameters as opaque strings and does not check them:
    /// whether `"purple"` is a valid `channel` is the registry's call, made when
    /// the graph is validated/instantiated. Keeping the check out of here lets
    /// an editor hold a half-typed, temporarily invalid value in the
    /// authoritative graph and report the problem, rather than silently
    /// discarding what the user entered.
    pub fn set_param(
        &mut self,
        node: &NodeId,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Option<String>, GraphError> {
        let spec = self
            .nodes
            .get_mut(node)
            .ok_or_else(|| GraphError::MissingNode(node.0.clone()))?;
        Ok(spec.params.insert(key.into(), value.into()))
    }

    /// Remove one parameter from a node, returning its previous value
    /// (`Ok(None)` when the node exists but had no such parameter).
    pub fn remove_param(&mut self, node: &NodeId, key: &str) -> Result<Option<String>, GraphError> {
        let spec = self
            .nodes
            .get_mut(node)
            .ok_or_else(|| GraphError::MissingNode(node.0.clone()))?;
        Ok(spec.params.remove(key))
    }

    /// Returns a set of node ids referenced by edges but missing from `nodes`.
    pub fn dangling_references(&self) -> HashSet<NodeId> {
        let mut out = HashSet::new();
        for conn in self.edges.values() {
            if !self.nodes.contains_key(&conn.from.0) {
                out.insert(conn.from.0.clone());
            }
            if !self.nodes.contains_key(&conn.to.0) {
                out.insert(conn.to.0.clone());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_with(id: &str) -> Graph {
        let mut g = Graph::new();
        g.add_node(NodeSpec {
            id: NodeId(id.to_string()),
            kind: "crop".to_string(),
            params: Params::new(),
        })
        .unwrap();
        g
    }

    #[test]
    fn set_param_inserts_then_overwrites() {
        let mut g = graph_with("a");
        let a = NodeId("a".to_string());
        assert_eq!(g.set_param(&a, "w", "4"), Ok(None));
        assert_eq!(g.set_param(&a, "w", "8"), Ok(Some("4".to_string())));
        assert_eq!(g.nodes[&a].params["w"], "8");
    }

    #[test]
    fn remove_param_reports_previous_value() {
        let mut g = graph_with("a");
        let a = NodeId("a".to_string());
        g.set_param(&a, "w", "4").unwrap();
        assert_eq!(g.remove_param(&a, "w"), Ok(Some("4".to_string())));
        assert_eq!(g.remove_param(&a, "w"), Ok(None));
        assert!(g.nodes[&a].params.is_empty());
    }

    #[test]
    fn param_edits_on_missing_node_are_errors() {
        let mut g = Graph::new();
        let ghost = NodeId("ghost".to_string());
        assert_eq!(
            g.set_param(&ghost, "w", "1"),
            Err(GraphError::MissingNode("ghost".to_string()))
        );
        assert_eq!(
            g.remove_param(&ghost, "w"),
            Err(GraphError::MissingNode("ghost".to_string()))
        );
    }
}
