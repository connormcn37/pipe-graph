//! Bevy editor layer: a thin *view/controller* over the authoritative core
//! graph. The core (`data`/`graph`/`exec`/`stages`/`editor`) never depends on
//! Bevy; this module — compiled only under `--features bevy` — mirrors the
//! `Graph` into ECS and routes user intents back into it.
//!
//! Design (per the roadmap): the `Graph` is the single source of truth, held in
//! a [`GraphResource`]. ECS entities are *views* ([`NodeView`]), not the data
//! model. Editor intents are queued as [`crate::editor::EditorCommand`]s in
//! [`EditorCommands`] and applied to the graph; a sync system then spawns or
//! despawns node views so what's on screen matches the graph. All the graph
//! logic lives in the Bevy-free [`crate::editor`] module and is tested there;
//! the systems here are glue.
//!
//! The layer is split into three plugins so each can run where it makes sense:
//!
//! - [`PipeGraphEditorPlugin`] (this file) — the headless core: resources,
//!   command application and view sync. Works under `MinimalPlugins`.
//! - [`PipeGraphInteractPlugin`] ([`interact`]) — turns an abstract pointer
//!   state ([`EditorPointer`]) into node moves, selection and queued
//!   `EditorCommand`s. Also headless: it never reads windows or devices, so
//!   tests drive it by writing the pointer resource directly.
//! - [`PipeGraphRenderPlugin`] ([`render`]) — the only part that needs a
//!   window/renderer: fills [`EditorPointer`] from mouse/keyboard, gives views
//!   sprites and labels, and draws pins and edges with gizmos.
//!
//! Geometry shared by rendering and hit testing lives in [`layout`] as plain
//! functions, so "where is this pin" has exactly one answer.

pub mod interact;
pub mod layout;
pub mod render;

use bevy::prelude::*;

use crate::editor::{EditorCommand, apply_command, view_diff};
use crate::exec::{PortSet, builtin_registry};
use crate::graph::{Graph, NodeId};

pub use self::interact::{
    DragState, EditorDrag, EditorPointer, EditorSelection, PipeGraphInteractPlugin, Selection,
    handle_pointer,
};
pub use self::render::PipeGraphRenderPlugin;

/// The authoritative graph, wrapped as a Bevy resource. Everything the editor
/// draws or runs derives from this.
#[derive(Resource, Default)]
pub struct GraphResource(pub Graph);

/// Queue of pending editor intents, drained each frame by [`apply_editor_commands`].
/// UI code pushes onto this instead of mutating the graph directly.
#[derive(Resource, Default)]
pub struct EditorCommands {
    pub queue: Vec<EditorCommand>,
}

/// A view entity mirroring one graph node. Maps an ECS entity to a `NodeId`.
///
/// The view's on-screen position (box center) is its `Transform` translation.
/// Position is deliberately *not* stored in the graph: layout is a property of
/// one editor's view, not of the pipeline being described.
#[derive(Component, Debug, Clone)]
pub struct NodeView {
    pub id: NodeId,
}

/// The visual shape of a node view: its kind (for the label), the ports it
/// declares (for pins), and the resulting box size.
///
/// Computed once at spawn from the registry. Ports can depend on params (e.g.
/// Split's channel count), but no existing `EditorCommand` edits params, so a
/// node's shape is fixed for the lifetime of its view. A kind the registry does
/// not know — or params it rejects — yields an empty `PortSet`: the node still
/// renders as a pin-less box rather than disappearing or panicking.
#[derive(Component, Debug, Clone)]
pub struct NodeShape {
    pub kind: String,
    pub ports: PortSet,
    pub size: Vec2,
}

impl NodeShape {
    pub fn new(kind: impl Into<String>, ports: PortSet) -> Self {
        let size = layout::node_size(&ports);
        Self {
            kind: kind.into(),
            ports,
            size,
        }
    }
}

/// Plugin wiring the headless editor core (resources, command application and
/// view sync) into a Bevy `App`. Safe under `MinimalPlugins`.
pub struct PipeGraphEditorPlugin;

/// System set for the core systems, so input handling can be ordered before
/// them (intents queued this frame are applied this frame).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EditorCoreSystems;

impl Plugin for PipeGraphEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GraphResource>()
            .init_resource::<EditorCommands>()
            // Apply intents first, then reconcile the views with the new graph.
            .add_systems(
                Update,
                (apply_editor_commands, sync_graph_views)
                    .chain()
                    .in_set(EditorCoreSystems),
            );
    }
}

/// Drain queued editor commands and apply them to the authoritative graph.
pub fn apply_editor_commands(
    mut commands: ResMut<EditorCommands>,
    mut graph: ResMut<GraphResource>,
) {
    if commands.queue.is_empty() {
        return;
    }
    for command in commands.queue.drain(..) {
        // Outcomes are intentionally ignored here; a real UI would surface them.
        let _ = apply_command(&mut graph.0, command);
    }
}

/// Spawn a [`NodeView`] for each new graph node and despawn views whose node is
/// gone, so the ECS view mirrors the graph.
///
/// New views get a [`NodeShape`] and a `Transform` placed by
/// [`layout::auto_layout`]; existing views keep wherever the user dragged them.
pub fn sync_graph_views(
    graph: Res<GraphResource>,
    views: Query<(Entity, &NodeView)>,
    mut commands: Commands,
) {
    let present: std::collections::HashSet<NodeId> =
        views.iter().map(|(_, v)| v.id.clone()).collect();

    let diff = view_diff(&graph.0, &present);
    if diff.is_empty() {
        return;
    }

    if !diff.spawn.is_empty() {
        // Built per spawn batch rather than held as a resource: the registry's
        // constructors are not `Send + Sync`, and spawns are rare.
        let registry = builtin_registry();
        let positions = layout::auto_layout(&graph.0);
        for id in diff.spawn {
            let Some(spec) = graph.0.nodes.get(&id) else {
                continue;
            };
            let ports = registry.ports_of(spec).unwrap_or_default();
            let pos = positions.get(&id).copied().unwrap_or_default();
            commands.spawn((
                NodeView { id },
                NodeShape::new(spec.kind.clone(), ports),
                Transform::from_translation(pos.extend(0.0)),
            ));
        }
    }
    if !diff.despawn.is_empty() {
        for (entity, view) in views.iter() {
            if diff.despawn.contains(&view.id) {
                commands.entity(entity).despawn();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{NodeSpec, Params};

    fn spec(id: &str) -> NodeSpec {
        NodeSpec {
            id: NodeId(id.to_string()),
            kind: "clear_channel".to_string(),
            params: Params::new(),
        }
    }

    fn node_view_count(app: &mut App) -> usize {
        let mut q = app.world_mut().query::<&NodeView>();
        q.iter(app.world()).count()
    }

    #[test]
    fn views_mirror_graph_node_lifecycle() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(PipeGraphEditorPlugin);

        // Add a node via a command; after an update a view should exist.
        app.world_mut()
            .resource_mut::<EditorCommands>()
            .queue
            .push(EditorCommand::AddNode(spec("a")));
        app.update();
        assert_eq!(node_view_count(&mut app), 1);

        // Adding a second node yields a second view.
        app.world_mut()
            .resource_mut::<EditorCommands>()
            .queue
            .push(EditorCommand::AddNode(spec("b")));
        app.update();
        assert_eq!(node_view_count(&mut app), 2);

        // Removing a node despawns exactly its view.
        app.world_mut()
            .resource_mut::<EditorCommands>()
            .queue
            .push(EditorCommand::RemoveNode(NodeId("a".to_string())));
        app.update();
        assert_eq!(node_view_count(&mut app), 1);
    }

    #[test]
    fn spawned_views_get_shape_from_registry_and_unknown_kinds_get_no_pins() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(PipeGraphEditorPlugin);

        let mut split = Params::new();
        split.insert("channels".to_string(), "3".to_string());
        let queue = &mut app.world_mut().resource_mut::<EditorCommands>().queue;
        queue.push(EditorCommand::AddNode(NodeSpec {
            id: NodeId("s".to_string()),
            kind: "split".to_string(),
            params: split,
        }));
        queue.push(EditorCommand::AddNode(NodeSpec {
            id: NodeId("mystery".to_string()),
            kind: "not_a_real_kind".to_string(),
            params: Params::new(),
        }));
        app.update();

        let mut q = app
            .world_mut()
            .query::<(&NodeView, &NodeShape, &Transform)>();
        let mut shapes: Vec<(String, usize, usize)> = q
            .iter(app.world())
            .map(|(v, s, _)| (v.id.0.clone(), s.ports.inputs.len(), s.ports.outputs.len()))
            .collect();
        shapes.sort();
        assert_eq!(
            shapes,
            vec![("mystery".to_string(), 0, 0), ("s".to_string(), 1, 3)]
        );
    }
}
