//! Editor controller logic — Bevy-free and unit-testable.
//!
//! This layer sits between the authoritative [`crate::graph::Graph`] and any UI
//! frontend (the Bevy layer in [`crate::systems`], or a test). It expresses the
//! things an editor needs, without depending on any UI toolkit:
//!
//! - **Commands** ([`EditorCommand`] / [`apply_command`]): user intents that
//!   mutate the graph, routed through the core `Graph` API so the graph stays
//!   the single source of truth.
//! - **Layout** ([`Layout`]): where each node sits on the canvas. Positions are
//!   editor state, not pipeline state, so they live beside the graph rather
//!   than in [`crate::graph::NodeSpec`] — moving a node must never force the
//!   pipeline to rebuild. [`apply_command_with_layout`] routes
//!   [`EditorCommand::MoveNode`] there and keeps the layout in step with
//!   node additions/removals.
//! - **Live session** ([`LiveSession`]): a graph + layout + running
//!   [`crate::exec::Runtime`] kept in sync. Every topology or parameter edit
//!   re-instantiates the runtime, carrying external inputs, taps and watches
//!   across, and reports build/validation/run errors instead of panicking.
//! - **View sync** ([`view_diff`]): given the graph and the set of node views a
//!   frontend currently shows, compute which views to spawn and which to
//!   despawn so the view mirrors the graph.
//!
//! Keeping this here (rather than inside the Bevy systems) means the important
//! logic is covered by the headless test suite; the Bevy layer only has to wire
//! it into ECS.

use std::collections::HashSet;

use crate::graph::{EdgeId, Graph, GraphError, NodeId, NodeSpec, PortId};

mod layout;
pub use self::layout::*;

mod session;
pub use self::session::*;

/// A user/editor intent to mutate the authoritative graph (or, for
/// [`EditorCommand::MoveNode`], the editor's [`Layout`]).
#[derive(Debug, Clone)]
pub enum EditorCommand {
    AddNode(NodeSpec),
    Connect {
        from: (NodeId, PortId),
        to: (NodeId, PortId),
    },
    RemoveNode(NodeId),
    Disconnect(EdgeId),
    /// Set (insert or overwrite) one parameter on a node. The value is not
    /// checked here; an invalid value surfaces when the graph is instantiated
    /// (see [`LiveSession::last_error`]).
    SetParam {
        node: NodeId,
        key: String,
        value: String,
    },
    /// Remove one parameter from a node.
    RemoveParam {
        node: NodeId,
        key: String,
    },
    /// Move a node on the canvas. Layout-only: it never touches the graph, so
    /// it never triggers a pipeline rebuild.
    MoveNode {
        node: NodeId,
        to: (f32, f32),
    },
}

/// The result of applying an [`EditorCommand`], so a UI can report success or
/// surface why a change was rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum CommandOutcome {
    NodeAdded(NodeId),
    Connected(EdgeId),
    NodeRemoved(NodeId),
    Disconnected(EdgeId),
    /// A parameter was set; `previous` is the value it replaced, if any.
    ParamSet {
        node: NodeId,
        key: String,
        previous: Option<String>,
    },
    /// A parameter was removed; `previous` is the value it had.
    ParamRemoved {
        node: NodeId,
        key: String,
        previous: String,
    },
    /// A node was moved in the [`Layout`]; `previous` is its old position
    /// (`None` if it had not been placed yet).
    NodeMoved {
        node: NodeId,
        previous: Option<(f32, f32)>,
    },
    /// The command is valid but only concerns editor state outside the graph
    /// (a [`EditorCommand::MoveNode`] given to plain [`apply_command`], which
    /// has no layout to move). The graph is unchanged; use
    /// [`apply_command_with_layout`] or [`LiveSession::apply`] to apply it.
    NoGraphChange,
    /// The graph rejected the change (e.g. duplicate id, missing endpoint).
    Rejected(GraphError),
    /// The target node/edge/parameter did not exist.
    NotFound,
}

impl CommandOutcome {
    /// Whether this outcome changed the graph's topology or parameters, i.e.
    /// whether a running pipeline built from the graph is now out of date.
    ///
    /// Note a [`CommandOutcome::ParamSet`] that wrote the same value back is
    /// still reported as a change here (the outcome does not carry the new
    /// value); [`LiveSession::apply`] filters that case out itself.
    pub fn changes_graph(&self) -> bool {
        matches!(
            self,
            CommandOutcome::NodeAdded(_)
                | CommandOutcome::Connected(_)
                | CommandOutcome::NodeRemoved(_)
                | CommandOutcome::Disconnected(_)
                | CommandOutcome::ParamSet { .. }
                | CommandOutcome::ParamRemoved { .. }
        )
    }
}

/// Apply one command to the authoritative graph via the core `Graph` API.
///
/// [`EditorCommand::MoveNode`] has no graph effect: it yields
/// [`CommandOutcome::NoGraphChange`] for an existing node (or
/// [`CommandOutcome::NotFound`]) and leaves the graph untouched. Callers that
/// keep a [`Layout`] should use [`apply_command_with_layout`] instead.
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
        EditorCommand::SetParam { node, key, value } => {
            match graph.set_param(&node, key.clone(), value) {
                Ok(previous) => CommandOutcome::ParamSet {
                    node,
                    key,
                    previous,
                },
                Err(_) => CommandOutcome::NotFound,
            }
        }
        EditorCommand::RemoveParam { node, key } => match graph.remove_param(&node, &key) {
            Ok(Some(previous)) => CommandOutcome::ParamRemoved {
                node,
                key,
                previous,
            },
            Ok(None) | Err(_) => CommandOutcome::NotFound,
        },
        EditorCommand::MoveNode { node, .. } => {
            if graph.nodes.contains_key(&node) {
                CommandOutcome::NoGraphChange
            } else {
                CommandOutcome::NotFound
            }
        }
    }
}

/// Apply one command to a graph *and* its editor [`Layout`].
///
/// [`EditorCommand::MoveNode`] moves the node in `layout`; every other command
/// goes through [`apply_command`], after which the layout follows the graph:
/// an added node is auto-placed (see [`Layout::place_missing`]) and a removed
/// node's position is dropped. Existing positions are never shuffled, so an
/// edit does not rearrange nodes the user has arranged by hand.
pub fn apply_command_with_layout(
    graph: &mut Graph,
    layout: &mut Layout,
    command: EditorCommand,
) -> CommandOutcome {
    if let EditorCommand::MoveNode { node, to } = command {
        if !graph.nodes.contains_key(&node) {
            return CommandOutcome::NotFound;
        }
        let previous = layout.set_position(node.clone(), to);
        return CommandOutcome::NodeMoved { node, previous };
    }

    let outcome = apply_command(graph, command);
    match &outcome {
        CommandOutcome::NodeAdded(_) => {
            layout.place_missing(graph);
        }
        CommandOutcome::NodeRemoved(id) => {
            layout.remove(id);
        }
        _ => {}
    }
    outcome
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

    fn id(s: &str) -> NodeId {
        NodeId(s.to_string())
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
    fn set_param_reports_previous_and_mutates_graph() {
        let mut g = Graph::new();
        apply_command(&mut g, EditorCommand::AddNode(spec("a")));
        let set = |value: &str| EditorCommand::SetParam {
            node: id("a"),
            key: "channel".to_string(),
            value: value.to_string(),
        };

        assert_eq!(
            apply_command(&mut g, set("red")),
            CommandOutcome::ParamSet {
                node: id("a"),
                key: "channel".to_string(),
                previous: None,
            }
        );
        assert_eq!(
            apply_command(&mut g, set("blue")),
            CommandOutcome::ParamSet {
                node: id("a"),
                key: "channel".to_string(),
                previous: Some("red".to_string()),
            }
        );
        assert_eq!(g.nodes[&id("a")].params["channel"], "blue");
    }

    #[test]
    fn set_param_on_missing_node_is_not_found() {
        let mut g = Graph::new();
        let outcome = apply_command(
            &mut g,
            EditorCommand::SetParam {
                node: id("ghost"),
                key: "k".to_string(),
                value: "v".to_string(),
            },
        );
        assert_eq!(outcome, CommandOutcome::NotFound);
    }

    #[test]
    fn remove_param_round_trip() {
        let mut g = Graph::new();
        apply_command(&mut g, EditorCommand::AddNode(spec("a")));
        g.set_param(&id("a"), "channel", "red").unwrap();
        let remove = || EditorCommand::RemoveParam {
            node: id("a"),
            key: "channel".to_string(),
        };

        assert_eq!(
            apply_command(&mut g, remove()),
            CommandOutcome::ParamRemoved {
                node: id("a"),
                key: "channel".to_string(),
                previous: "red".to_string(),
            }
        );
        // Removing an absent key is NotFound, not a silent success.
        assert_eq!(apply_command(&mut g, remove()), CommandOutcome::NotFound);
    }

    #[test]
    fn move_node_via_plain_apply_command_leaves_graph_alone() {
        let mut g = Graph::new();
        apply_command(&mut g, EditorCommand::AddNode(spec("a")));
        let before = format!("{:?}", g.nodes[&id("a")]);

        let outcome = apply_command(
            &mut g,
            EditorCommand::MoveNode {
                node: id("a"),
                to: (1.0, 2.0),
            },
        );
        assert_eq!(outcome, CommandOutcome::NoGraphChange);
        assert!(!outcome.changes_graph());
        assert_eq!(format!("{:?}", g.nodes[&id("a")]), before);

        let outcome = apply_command(
            &mut g,
            EditorCommand::MoveNode {
                node: id("ghost"),
                to: (1.0, 2.0),
            },
        );
        assert_eq!(outcome, CommandOutcome::NotFound);
    }

    #[test]
    fn layout_routing_moves_places_and_forgets_nodes() {
        let mut g = Graph::new();
        let mut layout = Layout::new();

        apply_command_with_layout(&mut g, &mut layout, EditorCommand::AddNode(spec("a")));
        let placed = layout.position(&id("a")).expect("added node is placed");

        let outcome = apply_command_with_layout(
            &mut g,
            &mut layout,
            EditorCommand::MoveNode {
                node: id("a"),
                to: (40.0, 50.0),
            },
        );
        assert_eq!(
            outcome,
            CommandOutcome::NodeMoved {
                node: id("a"),
                previous: Some(placed),
            }
        );
        assert_eq!(layout.position(&id("a")), Some((40.0, 50.0)));

        // Moving a node that does not exist does not invent a position.
        let outcome = apply_command_with_layout(
            &mut g,
            &mut layout,
            EditorCommand::MoveNode {
                node: id("ghost"),
                to: (0.0, 0.0),
            },
        );
        assert_eq!(outcome, CommandOutcome::NotFound);
        assert_eq!(layout.position(&id("ghost")), None);

        apply_command_with_layout(&mut g, &mut layout, EditorCommand::RemoveNode(id("a")));
        assert_eq!(layout.position(&id("a")), None);
        assert!(layout.is_empty());
    }

    #[test]
    fn changes_graph_classifies_outcomes() {
        assert!(CommandOutcome::NodeAdded(id("a")).changes_graph());
        assert!(
            CommandOutcome::ParamRemoved {
                node: id("a"),
                key: "k".to_string(),
                previous: "v".to_string(),
            }
            .changes_graph()
        );
        assert!(
            !CommandOutcome::NodeMoved {
                node: id("a"),
                previous: None,
            }
            .changes_graph()
        );
        assert!(!CommandOutcome::NotFound.changes_graph());
        assert!(!CommandOutcome::Rejected(GraphError::MissingNode("a".into())).changes_graph());
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
