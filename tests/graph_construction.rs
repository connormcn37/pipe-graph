//! Integration tests for the core `Graph` data structure.
//!
//! These exercise the construction/validation API that exists today
//! (`add_node`, `connect`, `dangling_references`) including the error paths,
//! so later phases that build execution on top can't silently regress it.

use pipe_graph::graph::{Graph, GraphError, NodeId, NodeSpec, Params, PortId};

fn node(id: &str, kind: &str) -> NodeSpec {
    NodeSpec {
        id: NodeId(id.to_string()),
        kind: kind.to_string(),
        params: Params::new(),
    }
}

fn port(node: &str, port: &str) -> (NodeId, PortId) {
    (NodeId(node.to_string()), PortId(port.to_string()))
}

#[test]
fn add_nodes_and_connect() {
    let mut g = Graph::new();
    g.add_node(node("a", "source")).unwrap();
    g.add_node(node("b", "sink")).unwrap();

    let edge = g.connect(port("a", "out"), port("b", "in")).unwrap();

    assert_eq!(g.nodes.len(), 2);
    assert!(g.edges.contains_key(&edge));
    assert!(g.dangling_references().is_empty());
}

#[test]
fn duplicate_node_id_is_rejected() {
    let mut g = Graph::new();
    g.add_node(node("a", "source")).unwrap();

    let err = g.add_node(node("a", "other")).unwrap_err();

    assert_eq!(err, GraphError::DuplicateNodeId("a".to_string()));
    assert_eq!(g.nodes.len(), 1);
}

#[test]
fn connect_to_missing_node_is_rejected() {
    let mut g = Graph::new();
    g.add_node(node("a", "source")).unwrap();

    let err = g
        .connect(port("a", "out"), port("ghost", "in"))
        .unwrap_err();

    assert_eq!(err, GraphError::MissingNode("ghost".to_string()));
    assert!(g.edges.is_empty());
}

#[test]
fn connect_from_missing_node_is_rejected() {
    let mut g = Graph::new();
    g.add_node(node("b", "sink")).unwrap();

    let err = g
        .connect(port("ghost", "out"), port("b", "in"))
        .unwrap_err();

    assert_eq!(err, GraphError::MissingNode("ghost".to_string()));
}

#[test]
fn cycles_are_permitted() {
    // The runtime is explicitly designed to allow feedback loops; the graph
    // layer must not reject them at construction time.
    let mut g = Graph::new();
    g.add_node(node("a", "k")).unwrap();
    g.add_node(node("b", "k")).unwrap();

    g.connect(port("a", "out"), port("b", "in")).unwrap();
    g.connect(port("b", "out"), port("a", "in")).unwrap();

    assert_eq!(g.edges.len(), 2);
}

#[test]
fn remove_node_drops_touching_edges() {
    let mut g = Graph::new();
    g.add_node(node("a", "k")).unwrap();
    g.add_node(node("b", "k")).unwrap();
    g.add_node(node("c", "k")).unwrap();
    g.connect(port("a", "out"), port("b", "in")).unwrap();
    g.connect(port("b", "out"), port("c", "in")).unwrap();

    assert!(g.remove_node(&NodeId("b".to_string())));
    assert_eq!(g.nodes.len(), 2);
    // Both edges touched b, so both are gone.
    assert!(g.edges.is_empty());
    // Removing again is a no-op.
    assert!(!g.remove_node(&NodeId("b".to_string())));
}

#[test]
fn disconnect_removes_single_edge() {
    let mut g = Graph::new();
    g.add_node(node("a", "k")).unwrap();
    g.add_node(node("b", "k")).unwrap();
    let e = g.connect(port("a", "out"), port("b", "in")).unwrap();

    assert!(g.disconnect(&e));
    assert!(g.edges.is_empty());
    assert!(!g.disconnect(&e));
}

#[test]
fn edges_get_distinct_ids() {
    let mut g = Graph::new();
    g.add_node(node("a", "k")).unwrap();
    g.add_node(node("b", "k")).unwrap();

    let e1 = g.connect(port("a", "out"), port("b", "in")).unwrap();
    let e2 = g.connect(port("a", "out2"), port("b", "in2")).unwrap();

    assert_ne!(e1, e2);
}
