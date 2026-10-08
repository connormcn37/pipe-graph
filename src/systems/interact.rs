//! Editor interaction: pointer → selection, node drags and queued
//! [`EditorCommand`]s.
//!
//! The handler here never reads windows, cameras or input devices. It consumes
//! an abstract [`EditorPointer`] (cursor in *world* space plus button/key edge
//! flags) which the render plugin fills from real input each frame. That
//! indirection is what makes the interaction logic testable headlessly: a test
//! writes `EditorPointer`, calls `app.update()`, and inspects the queued
//! commands — no window, no camera, no event plumbing.
//!
//! Gestures:
//! - press on an output (or input) pin, release on a pin of the opposite side
//!   → queue [`EditorCommand::Connect`] (always oriented output → input; an
//!   input that is already fed has its old edge `Disconnect`ed first);
//! - press on a node body → select it, raise it to the front and drag it
//!   (moves its `Transform`); releasing queues [`EditorCommand::MoveNode`] so
//!   the session's layout records where it was dropped;
//! - press near an edge line → select the edge;
//! - press on empty space → clear the selection;
//! - Delete/Backspace → queue [`EditorCommand::RemoveNode`] or
//!   [`EditorCommand::Disconnect`] for the selection.
//!
//! The graph is never mutated here; everything goes through the command queue
//! so the core stays the single source of truth.

use std::collections::HashMap;

use bevy::prelude::*;

use super::layout::{
    EDGE_HIT_DISTANCE, PinSide, distance_to_segment, pin_at, point_in_rect, port_anchor,
    world_to_canvas,
};
use super::{
    EditorCommands, EditorCoreSystems, GraphResource, NodeShape, NodeView, PipeGraphEditorPlugin,
};
use crate::editor::EditorCommand;
use crate::graph::{EdgeId, Graph, NodeId, PortId};

/// Abstract pointer state for one frame, in world coordinates.
///
/// The `just_*` and `delete_just_pressed` flags are one-shot: [`handle_pointer`]
/// clears them after acting on them, so a test (or any other producer) can set
/// a flag once without it firing again on the next frame.
#[derive(Resource, Debug, Clone, Default)]
pub struct EditorPointer {
    /// Cursor position in world space, or `None` if it is outside the window.
    pub world: Option<Vec2>,
    /// The primary button is currently held.
    pub pressed: bool,
    /// The primary button went down this frame.
    pub just_pressed: bool,
    /// The primary button went up this frame.
    pub just_released: bool,
    /// The delete gesture (Delete/Backspace) fired this frame.
    pub delete_just_pressed: bool,
}

/// What the user has selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    Node(NodeId),
    Edge(EdgeId),
}

/// The current selection, if any.
#[derive(Resource, Debug, Clone, Default)]
pub struct EditorSelection(pub Option<Selection>);

/// An in-progress pointer drag.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum DragState {
    #[default]
    Idle,
    /// Moving a node view; `grab_offset` keeps the box from jumping so its
    /// center sits under the cursor.
    MoveNode { entity: Entity, grab_offset: Vec2 },
    /// Dragging a wire out of a pin; `cursor` is the loose end, for drawing.
    Wire {
        node: NodeId,
        side: PinSide,
        port: PortId,
        cursor: Vec2,
    },
}

/// The current drag, if any.
#[derive(Resource, Debug, Clone, Default)]
pub struct EditorDrag(pub DragState);

/// Headless interaction plugin. Adds [`PipeGraphEditorPlugin`] if it is not
/// already present, and runs [`handle_pointer`] before the core systems so
/// commands queued by a gesture are applied in the same frame.
pub struct PipeGraphInteractPlugin;

impl Plugin for PipeGraphInteractPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<PipeGraphEditorPlugin>() {
            app.add_plugins(PipeGraphEditorPlugin);
        }
        app.init_resource::<EditorPointer>()
            .init_resource::<EditorSelection>()
            .init_resource::<EditorDrag>()
            .add_systems(Update, handle_pointer.before(EditorCoreSystems));
    }
}

/// A node view's hit-test data, captured once per frame.
struct NodeHit {
    entity: Entity,
    id: NodeId,
    center: Vec2,
    z: f32,
    shape: NodeShape,
}

/// Collect every node view, front-most first (highest `z`, ties broken by id
/// so picking is deterministic when boxes overlap at the same depth).
fn collect_nodes<'a>(
    views: impl Iterator<Item = (Entity, &'a NodeView, &'a NodeShape, &'a Transform)>,
) -> Vec<NodeHit> {
    let mut hits: Vec<NodeHit> = views
        .map(|(entity, view, shape, tf)| NodeHit {
            entity,
            id: view.id.clone(),
            center: tf.translation.truncate(),
            z: tf.translation.z,
            shape: shape.clone(),
        })
        .collect();
    hits.sort_by(|a, b| b.z.total_cmp(&a.z).then_with(|| a.id.0.cmp(&b.id.0)));
    hits
}

/// What a click landed on, among node views.
enum NodePick {
    /// A pin of a node: `(node, side, port)`.
    Pin(NodeId, PinSide, PortId),
    /// The body of the node at this index of the `collect_nodes` list.
    Body(usize),
}

/// The front-most node feature under `point`. Each node is tested as a whole
/// (pins, then body) before moving to the node behind it, so a pin hidden
/// under another node's box can't be grabbed through it. Pins come first
/// within a node because they straddle its border.
fn pick_node(nodes: &[NodeHit], point: Vec2) -> Option<NodePick> {
    nodes.iter().enumerate().find_map(|(i, n)| {
        if let Some((side, port)) = pin_at(point, n.center, &n.shape.ports) {
            Some(NodePick::Pin(n.id.clone(), side, port))
        } else if point_in_rect(point, n.center, n.shape.size) {
            Some(NodePick::Body(i))
        } else {
            None
        }
    })
}

/// Distance in `z` between stacked node views. Labels sit at `+1` relative
/// to their node, so a step of 2 keeps one node's labels below the next
/// node's box.
pub const Z_STEP: f32 = 2.0;

/// Both endpoints of every edge whose nodes have views, in world space.
pub fn edge_segments(
    graph: &Graph,
    nodes: &HashMap<NodeId, (Vec2, &NodeShape)>,
) -> Vec<(EdgeId, Vec2, Vec2)> {
    let mut out: Vec<(EdgeId, Vec2, Vec2)> = graph
        .edges
        .iter()
        .filter_map(|(id, conn)| {
            let (fc, fs) = nodes.get(&conn.from.0)?;
            let (tc, ts) = nodes.get(&conn.to.0)?;
            Some((
                id.clone(),
                port_anchor(*fc, &fs.ports, PinSide::Output, &conn.from.1.0),
                port_anchor(*tc, &ts.ports, PinSide::Input, &conn.to.1.0),
            ))
        })
        .collect();
    out.sort_by_key(|(id, _, _)| id.0);
    out
}

/// The edge nearest to `point` within [`EDGE_HIT_DISTANCE`].
fn edge_hit(graph: &Graph, nodes: &[NodeHit], point: Vec2) -> Option<EdgeId> {
    let lookup: HashMap<NodeId, (Vec2, &NodeShape)> = nodes
        .iter()
        .map(|n| (n.id.clone(), (n.center, &n.shape)))
        .collect();
    edge_segments(graph, &lookup)
        .into_iter()
        .map(|(id, a, b)| (id, distance_to_segment(point, a, b)))
        .filter(|(_, d)| *d <= EDGE_HIT_DISTANCE)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(id, _)| id)
}

/// Queue a connection, keeping each input single-fed.
///
/// An input port accepts exactly one edge (validation rejects a second with
/// `InputAlreadyConnected`), so dropping a wire on an occupied input
/// *replaces* its existing edge — the usual node-editor behavior — instead of
/// silently producing a graph that can no longer run. Re-dropping the exact
/// existing wire is a no-op.
fn queue_connect(
    graph: &Graph,
    queue: &mut Vec<EditorCommand>,
    from: (NodeId, PortId),
    to: (NodeId, PortId),
) {
    let mut existing: Vec<(&EdgeId, bool)> = graph
        .edges
        .iter()
        .filter(|(_, e)| e.to == to)
        .map(|(id, e)| (id, e.from == from))
        .collect();
    if existing.iter().any(|(_, same)| *same) {
        return;
    }
    existing.sort_by_key(|(id, _)| id.0);
    for (id, _) in existing {
        queue.push(EditorCommand::Disconnect(id.clone()));
    }
    queue.push(EditorCommand::Connect { from, to });
}

/// Turn the frame's [`EditorPointer`] into selection changes, node moves and
/// queued [`EditorCommand`]s. See the module docs for the gesture list.
pub fn handle_pointer(
    mut pointer: ResMut<EditorPointer>,
    mut drag: ResMut<EditorDrag>,
    mut selection: ResMut<EditorSelection>,
    mut queue: ResMut<EditorCommands>,
    graph: Res<GraphResource>,
    mut views: Query<(Entity, &NodeView, &NodeShape, &mut Transform)>,
) {
    let cursor = pointer.world;

    if pointer.just_pressed
        && let Some(c) = cursor
    {
        let nodes = collect_nodes(views.iter());
        let pick = pick_node(&nodes, c);
        if let Some(NodePick::Pin(node, side, port)) = pick {
            drag.0 = DragState::Wire {
                node,
                side,
                port,
                cursor: c,
            };
        } else if let Some(NodePick::Body(i)) = pick {
            let hit = &nodes[i];
            selection.0 = Some(Selection::Node(hit.id.clone()));
            drag.0 = DragState::MoveNode {
                entity: hit.entity,
                grab_offset: hit.center - c,
            };
            // Bring the grabbed node to the front, re-packing everyone's z
            // into 0, Z_STEP, 2*Z_STEP, … so depth stays bounded however
            // many times nodes are raised.
            let order = std::iter::once(hit.entity)
                .chain(nodes.iter().map(|n| n.entity).filter(|e| *e != hit.entity));
            let top = (nodes.len() as f32 - 1.0) * Z_STEP;
            for (rank, entity) in order.enumerate() {
                if let Ok((_, _, _, mut tf)) = views.get_mut(entity) {
                    tf.translation.z = top - rank as f32 * Z_STEP;
                }
            }
        } else if let Some(edge) = edge_hit(&graph.0, &nodes, c) {
            selection.0 = Some(Selection::Edge(edge));
            drag.0 = DragState::Idle;
        } else {
            selection.0 = None;
            drag.0 = DragState::Idle;
        }
    }

    if pointer.pressed
        && let Some(c) = cursor
    {
        match &mut drag.0 {
            DragState::MoveNode {
                entity,
                grab_offset,
            } => {
                if let Ok((_, _, _, mut tf)) = views.get_mut(*entity) {
                    let target = c + *grab_offset;
                    tf.translation.x = target.x;
                    tf.translation.y = target.y;
                }
            }
            DragState::Wire { cursor, .. } => *cursor = c,
            DragState::Idle => {}
        }
    }

    if pointer.just_released {
        match std::mem::take(&mut drag.0) {
            DragState::Wire {
                node, side, port, ..
            } => {
                if let Some(c) = cursor {
                    let nodes = collect_nodes(views.iter());
                    if let Some(NodePick::Pin(t_node, t_side, t_port)) = pick_node(&nodes, c)
                        && t_side == side.opposite()
                    {
                        let (from, to) = match side {
                            PinSide::Output => ((node, port), (t_node, t_port)),
                            PinSide::Input => ((t_node, t_port), (node, port)),
                        };
                        queue_connect(&graph.0, &mut queue.queue, from, to);
                    }
                }
            }
            // Record where the node was dropped in the session's layout, so
            // the position survives the view being rebuilt (e.g. by a param
            // edit that changes its ports) and belongs to the editing session
            // rather than to one ECS entity.
            DragState::MoveNode { entity, .. } => {
                if let Ok((_, view, _, tf)) = views.get(entity) {
                    queue.queue.push(EditorCommand::MoveNode {
                        node: view.id.clone(),
                        to: world_to_canvas(tf.translation.truncate()),
                    });
                }
            }
            DragState::Idle => {}
        }
        drag.0 = DragState::Idle;
    }

    if pointer.delete_just_pressed
        && let Some(sel) = selection.0.take()
    {
        queue.queue.push(match sel {
            Selection::Node(id) => EditorCommand::RemoveNode(id),
            Selection::Edge(edge) => EditorCommand::Disconnect(edge),
        });
    }

    pointer.just_pressed = false;
    pointer.just_released = false;
    pointer.delete_just_pressed = false;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::EditorCommand;
    use crate::graph::{NodeSpec, Params};
    use crate::systems::layout::{pin_offset, side_ports};

    fn node(id: &str, kind: &str, params: &[(&str, &str)]) -> EditorCommand {
        let mut p = Params::new();
        for (k, v) in params {
            p.insert(k.to_string(), v.to_string());
        }
        EditorCommand::AddNode(NodeSpec {
            id: NodeId(id.to_string()),
            kind: kind.to_string(),
            params: p,
        })
    }

    /// An app with the headless editor + interaction plugins and two nodes:
    /// `a` (clear_channel, 1 in / 1 out) and `b` (merge, 2 in / 1 out).
    fn app_with_two_nodes() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(PipeGraphInteractPlugin);
        let q = &mut app.world_mut().resource_mut::<EditorCommands>().queue;
        q.push(node("a", "clear_channel", &[("channel", "red")]));
        q.push(node("b", "merge", &[("channels", "2")]));
        app.update();
        // Pin the views at known positions, independent of auto-layout.
        place(&mut app, "a", Vec2::new(0.0, 0.0));
        place(&mut app, "b", Vec2::new(400.0, 0.0));
        app
    }

    fn view_entity(app: &mut App, id: &str) -> Entity {
        let mut q = app.world_mut().query::<(Entity, &NodeView)>();
        q.iter(app.world())
            .find(|(_, v)| v.id.0 == id)
            .map(|(e, _)| e)
            .unwrap_or_else(|| panic!("no view for {id}"))
    }

    fn place(app: &mut App, id: &str, pos: Vec2) {
        let e = view_entity(app, id);
        app.world_mut().get_mut::<Transform>(e).unwrap().translation = pos.extend(0.0);
    }

    fn center(app: &mut App, id: &str) -> Vec2 {
        let e = view_entity(app, id);
        app.world()
            .get::<Transform>(e)
            .unwrap()
            .translation
            .truncate()
    }

    /// World position of `port` on `side` of node `id`.
    fn pin(app: &mut App, id: &str, side: PinSide, port: &str) -> Vec2 {
        let e = view_entity(app, id);
        let shape = app.world().get::<NodeShape>(e).unwrap().clone();
        let idx = side_ports(&shape.ports, side)
            .position(|p| p.0 == port)
            .unwrap();
        center(app, id) + pin_offset(&shape.ports, side, idx)
    }

    fn press(app: &mut App, at: Vec2) {
        *app.world_mut().resource_mut::<EditorPointer>() = EditorPointer {
            world: Some(at),
            pressed: true,
            just_pressed: true,
            ..default()
        };
        app.update();
    }

    fn drag_to(app: &mut App, at: Vec2) {
        let mut p = app.world_mut().resource_mut::<EditorPointer>();
        p.world = Some(at);
        p.pressed = true;
        app.update();
    }

    fn release(app: &mut App, at: Vec2) {
        *app.world_mut().resource_mut::<EditorPointer>() = EditorPointer {
            world: Some(at),
            just_released: true,
            ..default()
        };
        app.update();
    }

    fn press_delete(app: &mut App) {
        app.world_mut()
            .resource_mut::<EditorPointer>()
            .delete_just_pressed = true;
        app.update();
    }

    /// The graph's edges as sorted `(from, from_port, to, to_port)` tuples.
    fn edges(app: &App) -> Vec<(String, String, String, String)> {
        let g = &app.world().resource::<GraphResource>().0;
        let mut v: Vec<_> = g
            .edges
            .values()
            .map(|c| {
                (
                    c.from.0.0.clone(),
                    c.from.1.0.clone(),
                    c.to.0.0.clone(),
                    c.to.1.0.clone(),
                )
            })
            .collect();
        v.sort();
        v
    }

    #[test]
    fn drag_output_pin_to_input_pin_queues_connect() {
        let mut app = app_with_two_nodes();

        // Run only the interaction system so we can observe the raw command
        // before the core applies it.
        let out = pin(&mut app, "a", PinSide::Output, "out");
        let in1 = pin(&mut app, "b", PinSide::Input, "in1");
        *app.world_mut().resource_mut::<EditorPointer>() = EditorPointer {
            world: Some(out),
            pressed: true,
            just_pressed: true,
            ..default()
        };
        app.world_mut().run_system_cached(handle_pointer).unwrap();
        assert!(matches!(
            app.world().resource::<EditorDrag>().0,
            DragState::Wire {
                side: PinSide::Output,
                ..
            }
        ));
        *app.world_mut().resource_mut::<EditorPointer>() = EditorPointer {
            world: Some(in1 + Vec2::new(2.0, 1.0)),
            just_released: true,
            ..default()
        };
        app.world_mut().run_system_cached(handle_pointer).unwrap();

        let queued = &app.world().resource::<EditorCommands>().queue;
        assert_eq!(queued.len(), 1);
        match &queued[0] {
            EditorCommand::Connect { from, to } => {
                assert_eq!((from.0.0.as_str(), from.1.0.as_str()), ("a", "out"));
                assert_eq!((to.0.0.as_str(), to.1.0.as_str()), ("b", "in1"));
            }
            other => panic!("expected Connect, got {other:?}"),
        }
        assert_eq!(app.world().resource::<EditorDrag>().0, DragState::Idle);

        // And through the full schedule, the graph gains the edge.
        app.update();
        assert_eq!(
            edges(&app),
            vec![(
                "a".to_string(),
                "out".to_string(),
                "b".to_string(),
                "in1".to_string()
            )]
        );
    }

    #[test]
    fn wire_dragged_from_input_is_oriented_output_to_input() {
        let mut app = app_with_two_nodes();
        let in0 = pin(&mut app, "b", PinSide::Input, "in0");
        let out = pin(&mut app, "a", PinSide::Output, "out");
        press(&mut app, in0);
        drag_to(&mut app, (in0 + out) / 2.0);
        release(&mut app, out);
        assert_eq!(
            edges(&app),
            vec![(
                "a".to_string(),
                "out".to_string(),
                "b".to_string(),
                "in0".to_string()
            )]
        );
    }

    #[test]
    fn invalid_wire_drops_are_ignored() {
        let mut app = app_with_two_nodes();
        let out_a = pin(&mut app, "a", PinSide::Output, "out");
        let out_b = pin(&mut app, "b", PinSide::Output, "out");

        // Output → output: rejected.
        press(&mut app, out_a);
        release(&mut app, out_b);
        // Output → empty space: rejected.
        press(&mut app, out_a);
        release(&mut app, Vec2::new(-1000.0, -1000.0));
        assert!(edges(&app).is_empty());

        // Same wire twice: only one edge.
        let in0 = pin(&mut app, "b", PinSide::Input, "in0");
        press(&mut app, out_a);
        release(&mut app, in0);
        press(&mut app, out_a);
        release(&mut app, in0);
        assert_eq!(edges(&app).len(), 1);
    }

    #[test]
    fn wiring_an_occupied_input_replaces_its_edge() {
        let mut app = app_with_two_nodes();
        // A third node to act as a second source.
        app.world_mut()
            .resource_mut::<EditorCommands>()
            .queue
            .push(node("c", "clear_channel", &[("channel", "blue")]));
        app.update();
        place(&mut app, "c", Vec2::new(0.0, -200.0));

        let in0 = pin(&mut app, "b", PinSide::Input, "in0");
        let out_a = pin(&mut app, "a", PinSide::Output, "out");
        let out_c = pin(&mut app, "c", PinSide::Output, "out");
        press(&mut app, out_a);
        release(&mut app, in0);
        press(&mut app, out_c);
        release(&mut app, in0);
        assert_eq!(
            edges(&app),
            vec![(
                "c".to_string(),
                "out".to_string(),
                "b".to_string(),
                "in0".to_string()
            )]
        );
    }

    #[test]
    fn overlapping_nodes_pick_the_front_one_and_grabbing_raises_it() {
        let mut app = app_with_two_nodes();
        // Overlap b onto a; b is in front.
        place(&mut app, "b", Vec2::new(30.0, 0.0));
        let eb = view_entity(&mut app, "b");
        app.world_mut()
            .get_mut::<Transform>(eb)
            .unwrap()
            .translation
            .z = 10.0;

        // a's output pin lies under b's body: the click must hit b, not wire a.
        let hidden = pin(&mut app, "a", PinSide::Output, "out");
        press(&mut app, hidden);
        assert!(matches!(
            app.world().resource::<EditorDrag>().0,
            DragState::MoveNode { entity, .. } if entity == eb
        ));
        release(&mut app, hidden);

        // Grab a by a part b does not cover; it comes to the front.
        let a_left = center(&mut app, "a") + Vec2::new(-75.0, 10.0);
        press(&mut app, a_left);
        release(&mut app, a_left);
        let ea = view_entity(&mut app, "a");
        let za = app.world().get::<Transform>(ea).unwrap().translation.z;
        let zb = app.world().get::<Transform>(eb).unwrap().translation.z;
        assert!(za > zb, "grabbed node should be in front ({za} vs {zb})");
        assert_eq!(za - zb, Z_STEP);
    }

    #[test]
    fn dragging_a_node_body_moves_its_transform() {
        let mut app = app_with_two_nodes();
        // Grab slightly off-center: the offset must be preserved.
        let grab = center(&mut app, "a") + Vec2::new(10.0, 5.0);
        press(&mut app, grab);
        assert_eq!(
            app.world().resource::<EditorSelection>().0,
            Some(Selection::Node(NodeId("a".to_string())))
        );
        drag_to(&mut app, grab + Vec2::new(50.0, -30.0));
        release(&mut app, grab + Vec2::new(50.0, -30.0));
        assert_eq!(center(&mut app, "a"), Vec2::new(50.0, -30.0));
        // The other node is untouched.
        assert_eq!(center(&mut app, "b"), Vec2::new(400.0, 0.0));
    }

    #[test]
    fn select_node_and_delete_removes_it() {
        let mut app = app_with_two_nodes();
        let c = center(&mut app, "a");
        press(&mut app, c);
        release(&mut app, c);

        // Observe the raw command from the interaction system alone.
        app.world_mut()
            .resource_mut::<EditorPointer>()
            .delete_just_pressed = true;
        app.world_mut().run_system_cached(handle_pointer).unwrap();
        let queued = &app.world().resource::<EditorCommands>().queue;
        assert!(matches!(
            queued.as_slice(),
            [EditorCommand::RemoveNode(id)] if id.0 == "a"
        ));
        assert_eq!(app.world().resource::<EditorSelection>().0, None);

        app.update();
        let mut q = app.world_mut().query::<&NodeView>();
        let ids: Vec<String> = q.iter(app.world()).map(|v| v.id.0.clone()).collect();
        assert_eq!(ids, vec!["b".to_string()]);
    }

    #[test]
    fn click_edge_and_delete_disconnects_it() {
        let mut app = app_with_two_nodes();
        let out = pin(&mut app, "a", PinSide::Output, "out");
        let in0 = pin(&mut app, "b", PinSide::Input, "in0");
        press(&mut app, out);
        release(&mut app, in0);
        assert_eq!(edges(&app).len(), 1);

        // Click the middle of the wire, a few units off the line.
        let mid = (out + in0) / 2.0 + Vec2::new(0.0, 3.0);
        press(&mut app, mid);
        release(&mut app, mid);
        assert!(matches!(
            app.world().resource::<EditorSelection>().0,
            Some(Selection::Edge(_))
        ));

        press_delete(&mut app);
        assert!(edges(&app).is_empty());
        // Both nodes survive.
        let mut q = app.world_mut().query::<&NodeView>();
        assert_eq!(q.iter(app.world()).count(), 2);
    }

    #[test]
    fn click_empty_space_clears_selection_and_delete_is_a_no_op() {
        let mut app = app_with_two_nodes();
        let c = center(&mut app, "a");
        press(&mut app, c);
        release(&mut app, c);
        press(&mut app, Vec2::new(-1000.0, 1000.0));
        release(&mut app, Vec2::new(-1000.0, 1000.0));
        assert_eq!(app.world().resource::<EditorSelection>().0, None);

        press_delete(&mut app);
        let mut q = app.world_mut().query::<&NodeView>();
        assert_eq!(q.iter(app.world()).count(), 2);
    }

    #[test]
    fn unknown_kind_node_is_selectable_and_deletable() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(PipeGraphInteractPlugin);
        app.world_mut()
            .resource_mut::<EditorCommands>()
            .queue
            .push(node("ghost", "no_such_kind", &[]));
        app.update();

        let c = center(&mut app, "ghost");
        press(&mut app, c);
        release(&mut app, c);
        press_delete(&mut app);
        let mut q = app.world_mut().query::<&NodeView>();
        assert_eq!(q.iter(app.world()).count(), 0);
    }
}
