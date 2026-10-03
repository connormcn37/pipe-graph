//! Editor rendering and device input — the only part of the editor that needs
//! a window and a renderer.
//!
//! - [`gather_pointer_input`] translates mouse/keyboard state and the cursor's
//!   window position (via the 2D camera) into the abstract
//!   [`EditorPointer`] the headless interaction logic consumes.
//! - [`attach_node_visuals`] gives each newly spawned [`NodeView`] a filled
//!   sprite box, a title label and per-port labels.
//! - [`draw_graph`] draws box outlines (highlighting the selection), pins,
//!   edges and the wire being dragged, with immediate-mode gizmos — they are
//!   re-derived from the graph every frame, so there is no edge entity state to
//!   keep in sync.
//!
//! This plugin needs `DefaultPlugins` (gizmos, sprites, text, input, windows).
//! Tests use [`super::PipeGraphInteractPlugin`] under `MinimalPlugins` instead.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::window::PrimaryWindow;

use super::interact::{
    DragState, EditorDrag, EditorPointer, EditorSelection, PipeGraphInteractPlugin, Selection,
    edge_segments, handle_pointer,
};
use super::layout::{HEADER_HEIGHT, PIN_RADIUS, PinSide, pin_offset, pin_positions, side_ports};
use super::{GraphResource, NodeShape, NodeView};
use crate::graph::NodeId;

const BOX_FILL: Color = Color::srgb(0.16, 0.17, 0.21);
const BOX_OUTLINE: Color = Color::srgb(0.45, 0.47, 0.55);
const SELECTED: Color = Color::srgb(1.0, 0.78, 0.25);
const INPUT_PIN: Color = Color::srgb(0.35, 0.75, 1.0);
const OUTPUT_PIN: Color = Color::srgb(0.55, 0.95, 0.5);
const EDGE: Color = Color::srgb(0.8, 0.8, 0.85);
const WIRE: Color = Color::srgb(1.0, 1.0, 1.0);
const TITLE: Color = Color::srgb(0.95, 0.95, 0.95);
const PORT_LABEL: Color = Color::srgb(0.7, 0.72, 0.78);

/// Rendering + device-input plugin. Adds [`PipeGraphInteractPlugin`] (and
/// through it [`super::PipeGraphEditorPlugin`]) if not already present.
/// Requires `DefaultPlugins` and a `Camera2d` (the editor example spawns one).
pub struct PipeGraphRenderPlugin;

impl Plugin for PipeGraphRenderPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<PipeGraphInteractPlugin>() {
            app.add_plugins(PipeGraphInteractPlugin);
        }
        app.add_systems(
            Update,
            (
                gather_pointer_input.before(handle_pointer),
                attach_node_visuals,
                draw_graph.after(handle_pointer),
            ),
        );
    }
}

/// Fill [`EditorPointer`] from the primary window's cursor (converted to world
/// space through the 2D camera), the left mouse button and Delete/Backspace.
///
/// Edge flags are OR-ed in rather than overwritten: `handle_pointer` clears
/// them once consumed, so nothing is lost if both ever ran out of step.
pub fn gather_pointer_input(
    mut pointer: ResMut<EditorPointer>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) {
    pointer.world = windows
        .single()
        .ok()
        .and_then(Window::cursor_position)
        .and_then(|screen| {
            let (camera, transform) = cameras.single().ok()?;
            camera.viewport_to_world_2d(transform, screen).ok()
        });
    pointer.pressed = mouse.pressed(MouseButton::Left);
    pointer.just_pressed |= mouse.just_pressed(MouseButton::Left);
    pointer.just_released |= mouse.just_released(MouseButton::Left);
    pointer.delete_just_pressed |= keys.any_just_pressed([KeyCode::Delete, KeyCode::Backspace]);
}

/// Give each new node view a filled box sprite, a title and port labels. The
/// labels are children, so they follow the node when dragged and are
/// despawned with it.
pub fn attach_node_visuals(
    mut commands: Commands,
    added: Query<(Entity, &NodeView, &NodeShape), Added<NodeView>>,
) {
    for (entity, view, shape) in &added {
        let title = format!("{} ({})", view.id.0, shape.kind);
        let mut node = commands.entity(entity);
        node.insert(Sprite::from_color(BOX_FILL, shape.size));
        node.with_child((
            Text2d::new(title),
            TextFont::from_font_size(14.0),
            TextColor(TITLE),
            Transform::from_xyz(0.0, shape.size.y / 2.0 - HEADER_HEIGHT / 2.0, 1.0),
        ));
        for side in [PinSide::Input, PinSide::Output] {
            // Labels sit just inside the box, next to their pin.
            let (anchor, inset) = match side {
                PinSide::Input => (Anchor::CENTER_LEFT, 10.0),
                PinSide::Output => (Anchor::CENTER_RIGHT, -10.0),
            };
            for (i, port) in side_ports(&shape.ports, side).enumerate() {
                let at = pin_offset(&shape.ports, side, i) + Vec2::new(inset, 0.0);
                node.with_child((
                    Text2d::new(port.0.clone()),
                    TextFont::from_font_size(11.0),
                    TextColor(PORT_LABEL),
                    anchor,
                    Transform::from_translation(at.extend(1.0)),
                ));
            }
        }
    }
}

/// Draw outlines, pins, edges and the in-progress wire with gizmos.
pub fn draw_graph(
    mut gizmos: Gizmos,
    graph: Res<GraphResource>,
    selection: Res<EditorSelection>,
    drag: Res<EditorDrag>,
    views: Query<(&NodeView, &NodeShape, &Transform)>,
) {
    let selected = selection.0.as_ref();
    let mut lookup: HashMap<NodeId, (Vec2, &NodeShape)> = HashMap::new();

    for (view, shape, tf) in &views {
        let center = tf.translation.truncate();
        lookup.insert(view.id.clone(), (center, shape));

        let is_selected = selected == Some(&Selection::Node(view.id.clone()));
        let outline = if is_selected { SELECTED } else { BOX_OUTLINE };
        gizmos.rect_2d(center, shape.size, outline);
        if is_selected {
            gizmos.rect_2d(center, shape.size + Vec2::splat(4.0), SELECTED);
        }
        for (side, _, pos) in pin_positions(center, &shape.ports) {
            let color = match side {
                PinSide::Input => INPUT_PIN,
                PinSide::Output => OUTPUT_PIN,
            };
            gizmos.circle_2d(pos, PIN_RADIUS, color);
        }
    }

    for (id, a, b) in edge_segments(&graph.0, &lookup) {
        let color = if selected == Some(&Selection::Edge(id)) {
            SELECTED
        } else {
            EDGE
        };
        gizmos.line_2d(a, b, color);
    }

    if let DragState::Wire {
        node,
        side,
        port,
        cursor,
    } = &drag.0
        && let Some((center, shape)) = lookup.get(node)
    {
        let start = super::layout::port_anchor(*center, &shape.ports, *side, &port.0);
        gizmos.line_2d(start, *cursor, WIRE);
    }
}
