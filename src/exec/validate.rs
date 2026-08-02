//! Port-aware validation of a `Graph` against a `Registry`.
//!
//! Kept in the exec layer (not on `Graph`) so `graph::Graph` stays pure and
//! dependency-light: `exec` knows about the registry and port model, the graph
//! layer does not. Validation resolves every node's declared ports via the
//! registry and checks each connection for port existence, payload-kind
//! compatibility, and single-writer inputs (fan-out is fine, fan-in is not).

use std::collections::{HashMap, HashSet};

use crate::data::PayloadKind;
use crate::exec::{BuildError, PortSet, Registry};
use crate::graph::{Graph, NodeId, PortId};

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// A node referenced by an edge is not present in the graph.
    MissingNode(NodeId),
    /// A node's `kind` has no registered constructor (or failed to build).
    Build { node: NodeId, error: BuildError },
    /// An edge's source port is not a declared output of its node.
    UnknownOutputPort { node: NodeId, port: PortId },
    /// An edge's destination port is not a declared input of its node.
    UnknownInputPort { node: NodeId, port: PortId },
    /// An edge connects incompatible payload kinds.
    KindMismatch {
        from: (NodeId, PortId),
        to: (NodeId, PortId),
        out_kind: PayloadKind,
        in_kind: PayloadKind,
    },
    /// More than one edge targets the same input port.
    InputAlreadyConnected { node: NodeId, port: PortId },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingNode(n) => write!(f, "edge references missing node '{}'", n.0),
            ValidationError::Build { node, error } => {
                write!(f, "node '{}': {error}", node.0)
            }
            ValidationError::UnknownOutputPort { node, port } => {
                write!(f, "node '{}' has no output port '{}'", node.0, port.0)
            }
            ValidationError::UnknownInputPort { node, port } => {
                write!(f, "node '{}' has no input port '{}'", node.0, port.0)
            }
            ValidationError::KindMismatch {
                from,
                to,
                out_kind,
                in_kind,
            } => write!(
                f,
                "edge {}:{} -> {}:{} connects {out_kind:?} to {in_kind:?}",
                from.0.0, from.1.0, to.0.0, to.1.0
            ),
            ValidationError::InputAlreadyConnected { node, port } => write!(
                f,
                "input '{}:{}' already has an incoming edge",
                node.0, port.0
            ),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validate every connection in `graph` against the ports declared by `reg`.
pub fn validate(graph: &Graph, reg: &Registry) -> Result<(), ValidationError> {
    // Resolve each node's port set once.
    let mut ports: HashMap<&NodeId, PortSet> = HashMap::with_capacity(graph.nodes.len());
    for (id, spec) in &graph.nodes {
        let ps = reg.ports_of(spec).map_err(|error| ValidationError::Build {
            node: id.clone(),
            error,
        })?;
        ports.insert(id, ps);
    }

    let mut connected_inputs: HashSet<(NodeId, PortId)> = HashSet::new();

    for conn in graph.edges.values() {
        let (from_node, from_port) = &conn.from;
        let (to_node, to_port) = &conn.to;

        let from_ports = ports
            .get(from_node)
            .ok_or_else(|| ValidationError::MissingNode(from_node.clone()))?;
        let to_ports = ports
            .get(to_node)
            .ok_or_else(|| ValidationError::MissingNode(to_node.clone()))?;

        let out = from_ports.find_output(&from_port.0).ok_or_else(|| {
            ValidationError::UnknownOutputPort {
                node: from_node.clone(),
                port: from_port.clone(),
            }
        })?;
        let inp =
            to_ports
                .find_input(&to_port.0)
                .ok_or_else(|| ValidationError::UnknownInputPort {
                    node: to_node.clone(),
                    port: to_port.clone(),
                })?;

        if !inp.kind.accepts(out.kind) {
            return Err(ValidationError::KindMismatch {
                from: conn.from.clone(),
                to: conn.to.clone(),
                out_kind: out.kind,
                in_kind: inp.kind,
            });
        }

        if !connected_inputs.insert((to_node.clone(), to_port.clone())) {
            return Err(ValidationError::InputAlreadyConnected {
                node: to_node.clone(),
                port: to_port.clone(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::builtin_registry;
    use crate::graph::{NodeSpec, Params};

    fn clear(id: &str) -> NodeSpec {
        let mut p = Params::new();
        p.insert("channel".to_string(), "red".to_string());
        NodeSpec {
            id: NodeId(id.to_string()),
            kind: "clear_channel".to_string(),
            params: p,
        }
    }

    fn port(node: &str, port: &str) -> (NodeId, PortId) {
        (NodeId(node.to_string()), PortId(port.to_string()))
    }

    #[test]
    fn valid_chain_passes() {
        let reg = builtin_registry();
        let mut g = Graph::new();
        g.add_node(clear("a")).unwrap();
        g.add_node(clear("b")).unwrap();
        g.connect(port("a", "out"), port("b", "in")).unwrap();
        assert!(validate(&g, &reg).is_ok());
    }

    #[test]
    fn unknown_input_port_is_rejected() {
        let reg = builtin_registry();
        let mut g = Graph::new();
        g.add_node(clear("a")).unwrap();
        g.add_node(clear("b")).unwrap();
        g.connect(port("a", "out"), port("b", "nope")).unwrap();
        assert_eq!(
            validate(&g, &reg).unwrap_err(),
            ValidationError::UnknownInputPort {
                node: NodeId("b".to_string()),
                port: PortId("nope".to_string()),
            }
        );
    }

    #[test]
    fn fan_in_is_rejected() {
        let reg = builtin_registry();
        let mut g = Graph::new();
        g.add_node(clear("a")).unwrap();
        g.add_node(clear("b")).unwrap();
        g.add_node(clear("c")).unwrap();
        // Two writers into b's single "in" port.
        g.connect(port("a", "out"), port("b", "in")).unwrap();
        g.connect(port("c", "out"), port("b", "in")).unwrap();
        assert_eq!(
            validate(&g, &reg).unwrap_err(),
            ValidationError::InputAlreadyConnected {
                node: NodeId("b".to_string()),
                port: PortId("in".to_string()),
            }
        );
    }

    #[test]
    fn unknown_kind_is_rejected() {
        let reg = builtin_registry();
        let mut g = Graph::new();
        g.add_node(NodeSpec {
            id: NodeId("x".to_string()),
            kind: "does_not_exist".to_string(),
            params: Params::new(),
        })
        .unwrap();
        assert!(matches!(
            validate(&g, &reg).unwrap_err(),
            ValidationError::Build { .. }
        ));
    }
}
