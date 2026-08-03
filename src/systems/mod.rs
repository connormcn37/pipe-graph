//! Bevy editor layer: a thin *view/controller* over the authoritative core
//! graph. The core (`data`/`graph`/`exec`/`stages`/`editor`) never depends on
//! Bevy; this module — compiled only under `--features bevy` — mirrors the
//! `Graph` into ECS and routes user intents back into it.
//!
//! Design (per the roadmap): the `Graph` is the single source of truth, held in
//! a [`GraphResource`]. ECS entities are *views* ([`NodeView`]), not the data
//! model. Editor intents are queued as [`crate::editor::EditorCommand`]s in
//! [`EditorCommands`] and applied to the graph; a sync system then spawns or
//! despawns node views so what's on screen matches the graph. All the actual
//! logic lives in the Bevy-free [`crate::editor`] module and is tested there;
//! the systems here are glue.

use bevy::prelude::*;

use crate::editor::{EditorCommand, apply_command, view_diff};
use crate::graph::{Graph, NodeId};

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
#[derive(Component, Debug, Clone)]
pub struct NodeView {
    pub id: NodeId,
}

/// Plugin wiring the editor resources and systems into a Bevy `App`.
pub struct PipeGraphEditorPlugin;

impl Plugin for PipeGraphEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GraphResource>()
            .init_resource::<EditorCommands>()
            // Apply intents first, then reconcile the views with the new graph.
            .add_systems(Update, (apply_editor_commands, sync_graph_views).chain());
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

    for id in diff.spawn {
        commands.spawn(NodeView { id });
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
}
