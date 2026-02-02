//! Core graph types (runtime-facing).
//!
//! This module is intentionally small and dependency-light.
//! The idea is that UI layers (Bevy/egui/etc.) can mirror these types.

use std::collections::{HashMap, HashSet};

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
            return Err(GraphError::MissingNode(from.0 .0));
        }
        if !self.nodes.contains_key(&to.0) {
            return Err(GraphError::MissingNode(to.0 .0));
        }

        // NOTE: cycles are allowed, but we may still want to treat self-loops specially.
        // Keep this permissive for now.

        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        self.edges.insert(id.clone(), Connection { from, to });
        Ok(id)
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
