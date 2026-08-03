//! Editor controller logic — Bevy-free and unit-testable.
//!
//! This layer sits between the authoritative [`crate::graph::Graph`] and any UI
//! frontend (the Bevy layer in [`crate::systems`], or a test). It expresses two
//! things the plan calls for, without depending on any UI toolkit:
//!
//! - **Commands** ([`EditorCommand`] / [`apply_command`]): user intents that
//!   mutate the graph, routed through the core `Graph` API so the graph stays
//!   the single source of truth.
//! - **View sync** ([`view_diff`]): given the graph and the set of node views a
//!   frontend currently shows, compute which views to spawn and which to
//!   despawn so the view mirrors the graph.
//!
//! Keeping this here (rather than inside the Bevy systems) means the important
//! logic is covered by the headless test suite; the Bevy layer only has to wire
//! it into ECS.

use std::collections::HashSet;

use crate::graph::{EdgeId, Graph, GraphError, NodeId, NodeSpec, PortId};

/// A user/editor intent to mutate the authoritative graph.
#[derive(Debug, Clone)]
pub enum EditorCommand {
    AddNode(NodeSpec),
    Connect {
        from: (NodeId, PortId),
        to: (NodeId, PortId),
    },
    RemoveNode(NodeId),
    Disconnect(EdgeId),
}

/// The result of applying an [`EditorCommand`], so a UI can report success or
/// surface why a change was rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum CommandOutcome {
    NodeAdded(NodeId),
    Connected(EdgeId),
    NodeRemoved(NodeId),
    Disconnected(EdgeId),
    /// The graph rejected the change (e.g. duplicate id, missing endpoint).
    Rejected(GraphError),
    /// The target node/edge did not exist.
    NotFound,
}

/// Apply one command to the authoritative graph via the core `Graph` API.
pub fn apply_command(graph: &mut Graph, command: EditorCommand) -> CommandOutcome {
    match command {
        EditorCommand::AddNode(spec) => {
            let id = spec.id.clone();
            match graph.add_node(spec) {
                Ok(()) => CommandOutcome::NodeAdded(id),
                Err(e) => CommandOutcome::Rejected(e),
            }
        }
        EditorCommand::Connect { from, to } => match graph.connect(from, to) {
            Ok(edge) => CommandOutcome::Connected(edge),
            Err(e) => CommandOutcome::Rejected(e),
        },
        EditorCommand::RemoveNode(id) => {
            if graph.remove_node(&id) {
                CommandOutcome::NodeRemoved(id)
            } else {
                CommandOutcome::NotFound
            }
        }
        EditorCommand::Disconnect(edge) => {
            if graph.disconnect(&edge) {
                CommandOutcome::Disconnected(edge)
            } else {
                CommandOutcome::NotFound
            }
        }
    }
}

/// Which node views a frontend should spawn/despawn so its views mirror the
/// graph, given the node ids it currently shows (`present`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ViewDiff {
    /// Graph nodes with no view yet.
    pub spawn: Vec<NodeId>,
    /// Views whose node no longer exists in the graph.
    pub despawn: Vec<NodeId>,
}

impl ViewDiff {
    pub fn is_empty(&self) -> bool {
        self.spawn.is_empty() && self.despawn.is_empty()
    }
}

/// Compute the view diff between `graph` and the currently-shown node ids.
/// Results are sorted by id for deterministic behavior.
pub fn view_diff(graph: &Graph, present: &HashSet<NodeId>) -> ViewDiff {
    let mut spawn: Vec<NodeId> = graph
        .nodes
        .keys()
        .filter(|id| !present.contains(*id))
        .cloned()
        .collect();

    let mut despawn: Vec<NodeId> = present
        .iter()
        .filter(|id| !graph.nodes.contains_key(*id))
        .cloned()
        .collect();

    spawn.sort_by(|a, b| a.0.cmp(&b.0));
    despawn.sort_by(|a, b| a.0.cmp(&b.0));

    ViewDiff { spawn, despawn }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Params;

    fn spec(id: &str) -> NodeSpec {
        NodeSpec {
            id: NodeId(id.to_string()),
            kind: "clear_channel".to_string(),
            params: Params::new(),
        }
    }

    fn port(node: &str, port: &str) -> (NodeId, PortId) {
        (NodeId(node.to_string()), PortId(port.to_string()))
    }

    #[test]
    fn add_node_command_mutates_graph() {
        let mut g = Graph::new();
        let outcome = apply_command(&mut g, EditorCommand::AddNode(spec("a")));
        assert_eq!(outcome, CommandOutcome::NodeAdded(NodeId("a".to_string())));
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn duplicate_add_is_rejected() {
        let mut g = Graph::new();
        apply_command(&mut g, EditorCommand::AddNode(spec("a")));
        let outcome = apply_command(&mut g, EditorCommand::AddNode(spec("a")));
        assert_eq!(
            outcome,
            CommandOutcome::Rejected(GraphError::DuplicateNodeId("a".to_string()))
        );
    }

    #[test]
    fn connect_and_disconnect_round_trip() {
        let mut g = Graph::new();
        apply_command(&mut g, EditorCommand::AddNode(spec("a")));
        apply_command(&mut g, EditorCommand::AddNode(spec("b")));

        let edge = match apply_command(
            &mut g,
            EditorCommand::Connect {
                from: port("a", "out"),
                to: port("b", "in"),
            },
        ) {
            CommandOutcome::Connected(e) => e,
            other => panic!("expected Connected, got {other:?}"),
        };
        assert_eq!(g.edges.len(), 1);

        let outcome = apply_command(&mut g, EditorCommand::Disconnect(edge.clone()));
        assert_eq!(outcome, CommandOutcome::Disconnected(edge));
        assert!(g.edges.is_empty());
    }

    #[test]
    fn remove_missing_node_reports_not_found() {
        let mut g = Graph::new();
        let outcome = apply_command(
            &mut g,
            EditorCommand::RemoveNode(NodeId("ghost".to_string())),
        );
        assert_eq!(outcome, CommandOutcome::NotFound);
    }

    #[test]
    fn view_diff_spawns_new_and_despawns_removed() {
        let mut g = Graph::new();
        apply_command(&mut g, EditorCommand::AddNode(spec("a")));
        apply_command(&mut g, EditorCommand::AddNode(spec("b")));

        // Nothing shown yet -> spawn both (sorted).
        let diff = view_diff(&g, &HashSet::new());
        assert_eq!(
            diff.spawn,
            vec![NodeId("a".to_string()), NodeId("b".to_string())]
        );
        assert!(diff.despawn.is_empty());

        // 'a' and a stale 'z' shown -> spawn b, despawn z.
        let present: HashSet<NodeId> = [NodeId("a".to_string()), NodeId("z".to_string())]
            .into_iter()
            .collect();
        let diff = view_diff(&g, &present);
        assert_eq!(diff.spawn, vec![NodeId("b".to_string())]);
        assert_eq!(diff.despawn, vec![NodeId("z".to_string())]);

        // Fully in sync -> empty.
        let present: HashSet<NodeId> = [NodeId("a".to_string()), NodeId("b".to_string())]
            .into_iter()
            .collect();
        assert!(view_diff(&g, &present).is_empty());
    }
}
