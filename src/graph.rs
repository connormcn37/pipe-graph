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
}

/// Errors returned by [`Graph::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphValidationError {
    /// An edge references a node id that does not exist.
    MissingNode { node: NodeId },
    /// An edge references an empty port id ("" after trimming).
    MissingPort { node: NodeId, port: PortId },
    /// An edge connects a node+port to itself.
    SelfLoop { node: NodeId, port: PortId },
    /// Two edges have identical endpoints (from,to).
    DuplicateEdge { from: (NodeId, PortId), to: (NodeId, PortId) },
    /// Node has no incident edges (isolated) when required by options.
    UnconnectedNode { node: NodeId },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GraphValidationOptions {
    /// If true, a node must appear in at least one edge (in or out).
    pub require_all_nodes_connected: bool,
    /// If true, disallow edges that connect a port to itself.
    pub disallow_self_loops: bool,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the graph structure without mutating it.
    ///
    /// This is intentionally conservative: it checks only invariants that are
    /// knowable from the core `Graph` model (nodes, edges, and port *names*).
    pub fn validate(&self, opts: GraphValidationOptions) -> Result<(), Vec<GraphValidationError>> {
        let mut errs: Vec<GraphValidationError> = Vec::new();

        // Endpoint-level checks + duplicate detection.
        let mut seen: HashSet<((NodeId, PortId), (NodeId, PortId))> = HashSet::new();
        for conn in self.edges.values() {
            // Missing nodes.
            if !self.nodes.contains_key(&conn.from.0) {
                errs.push(GraphValidationError::MissingNode {
                    node: conn.from.0.clone(),
                });
            }
            if !self.nodes.contains_key(&conn.to.0) {
                errs.push(GraphValidationError::MissingNode {
                    node: conn.to.0.clone(),
                });
            }

            // Missing ports (we only know the name exists / is non-empty).
            if conn.from.1 .0.trim().is_empty() {
                errs.push(GraphValidationError::MissingPort {
                    node: conn.from.0.clone(),
                    port: conn.from.1.clone(),
                });
            }
            if conn.to.1 .0.trim().is_empty() {
                errs.push(GraphValidationError::MissingPort {
                    node: conn.to.0.clone(),
                    port: conn.to.1.clone(),
                });
            }

            // Self-loop checks.
            if opts.disallow_self_loops && conn.from == conn.to {
                errs.push(GraphValidationError::SelfLoop {
                    node: conn.from.0.clone(),
                    port: conn.from.1.clone(),
                });
            }

            // Duplicate edges.
            let key = (conn.from.clone(), conn.to.clone());
            if !seen.insert(key.clone()) {
                errs.push(GraphValidationError::DuplicateEdge {
                    from: key.0,
                    to: key.1,
                });
            }
        }

        if opts.require_all_nodes_connected {
            let mut connected: HashSet<NodeId> = HashSet::new();
            for conn in self.edges.values() {
                connected.insert(conn.from.0.clone());
                connected.insert(conn.to.0.clone());
            }
            for id in self.nodes.keys() {
                if !connected.contains(id) {
                    errs.push(GraphValidationError::UnconnectedNode { node: id.clone() });
                }
            }
        }

        if errs.is_empty() {
            Ok(())
        } else {
            // De-dupe identical errors to keep output stable.
            errs.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
            errs.dedup();
            Err(errs)
        }
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

    fn node(id: &str) -> NodeSpec {
        NodeSpec {
            id: NodeId(id.to_string()),
            kind: "noop".to_string(),
            params: Params::default(),
        }
    }

    #[test]
    fn validate_ok_minimal_connected_graph() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("b".into()), PortId("in".into())),
        )
        .unwrap();

        g.validate(GraphValidationOptions {
            require_all_nodes_connected: true,
            disallow_self_loops: true,
        })
        .unwrap();
    }

    #[test]
    fn validate_reports_duplicate_edges() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("b".into()), PortId("in".into())),
        )
        .unwrap();

        // Add a second identical edge by bypassing connect() (edge IDs are distinct).
        g.edges.insert(
            EdgeId(999),
            Connection {
                from: (NodeId("a".into()), PortId("out".into())),
                to: (NodeId("b".into()), PortId("in".into())),
            },
        );

        let errs = g.validate(GraphValidationOptions::default()).unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, GraphValidationError::DuplicateEdge { .. })));
    }

    #[test]
    fn validate_reports_missing_ports() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("".into())),
            (NodeId("b".into()), PortId("in".into())),
        )
        .unwrap();

        let errs = g.validate(GraphValidationOptions::default()).unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, GraphValidationError::MissingPort { .. })));
    }

    #[test]
    fn validate_reports_unconnected_node_when_required() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        // b is isolated
        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("a".into()), PortId("in".into())),
        )
        .unwrap();

        let errs = g
            .validate(GraphValidationOptions {
                require_all_nodes_connected: true,
                disallow_self_loops: false,
            })
            .unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, GraphValidationError::UnconnectedNode { node } if node.0 == "b")));
    }

    #[test]
    fn validate_reports_self_loop_when_disallowed() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("p".into())),
            (NodeId("a".into()), PortId("p".into())),
        )
        .unwrap();

        let errs = g
            .validate(GraphValidationOptions {
                require_all_nodes_connected: false,
                disallow_self_loops: true,
            })
            .unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, GraphValidationError::SelfLoop { .. })));
    }
}
