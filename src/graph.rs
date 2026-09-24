//! Core graph types (runtime-facing).
//!
//! This module is intentionally small and dependency-light.
//! The idea is that UI layers (Bevy/egui/etc.) can mirror these types.

use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

/// Stable identifier for a node/stage in the graph.
///
/// Early version uses a string label (matches README intent: unique label).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

/// Identifier for an input or output port.
///
/// Ports are named so they can map cleanly to UI pins.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortId(pub String);

impl std::borrow::Borrow<str> for PortId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub from: (NodeId, PortId),
    pub to: (NodeId, PortId),
}

/// Parameter map for configuring stages.
pub type Params = HashMap<String, String>;

/// A node specification (runtime graph). UI/editor should produce these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSpec {
    pub id: NodeId,
    /// Stage "type" (e.g. "crop", "cast", "merge").
    pub kind: String,
    #[serde(default)]
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
    ParseError(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::DuplicateNodeId(id) => write!(f, "Duplicate node id: {}", id),
            GraphError::MissingNode(id) => write!(f, "Missing node: {}", id),
            GraphError::SelfLoopNotAllowed(id) => write!(f, "Self loop not allowed on: {}", id),
            GraphError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for GraphError {}

/// A human-readable pipeline definition for TOML/YAML files.
#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineDef {
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    pub edges: Vec<ConnectionDef>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConnectionDef {
    pub from: String,
    pub to: String,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads a Graph from a YAML string.
    pub fn from_yaml(s: &str) -> Result<Self, GraphError> {
        let def: PipelineDef = serde_yaml::from_str(s)
            .map_err(|e| GraphError::ParseError(e.to_string()))?;
        Self::from_def(&def)
    }

    /// Loads a Graph from a TOML string.
    pub fn from_toml(s: &str) -> Result<Self, GraphError> {
        let def: PipelineDef = toml::from_str(s)
            .map_err(|e| GraphError::ParseError(e.to_string()))?;
        Self::from_def(&def)
    }

    /// Converts a PipelineDef into a Graph instance.
    fn from_def(def: &PipelineDef) -> Result<Self, GraphError> {
        let mut g = Graph::new();
        for node in &def.nodes {
            g.add_node(node.clone())?;
        }
        for edge in &def.edges {
            let from_parts: Vec<&str> = edge.from.split('.').collect();
            let to_parts: Vec<&str> = edge.to.split('.').collect();
            if from_parts.len() != 2 || to_parts.len() != 2 {
                return Err(GraphError::ParseError(format!(
                    "Invalid edge format, expected 'node.port', got from: '{}', to: '{}'",
                    edge.from, edge.to
                )));
            }
            g.connect(
                (NodeId(from_parts[0].to_string()), PortId(from_parts[1].to_string())),
                (NodeId(to_parts[0].to_string()), PortId(to_parts[1].to_string())),
            )?;
        }
        Ok(g)
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
