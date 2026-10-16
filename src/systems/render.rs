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
//! - [`gather_key_input`] fills [`EditorKeys`] with this frame's key presses,
//!   in order; [`super::route_keys`] decides what they mean (parameter
//!   editing, play/pause, delete). [`update_inspector_text`] draws the
//!   parameter inspector panel.
//! - [`navigate_camera`] pans (middle- or right-drag) and zooms (wheel,
//!   about the cursor) the 2D camera; [`fit_camera`] frames every node on
//!   startup and whenever [`FitView`] is requested (`F`). The math is in
//!   [`super::camera`]; these only read devices and write the camera.
//! - [`update_status_text`] shows play state and the session's last error in a
//!   screen-space line; [`update_preview`] shows the selected node's latest
//!   output frame (from [`EditorPreview`]'s tap) under its box, re-uploading
//!   the texture only when the tap's sequence number moves.
//!
//! All text uses the bundled Unicode font ([`super::font::EditorFont`]), not
//! Bevy's ASCII-only default, so non-ASCII ids, values and paths render.
//!
//! This plugin needs `DefaultPlugins` (gizmos, sprites, text, input, windows).
//! Tests use [`super::PipeGraphInteractPlugin`] under `MinimalPlugins` instead.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::sprite::Anchor;
use bevy::window::PrimaryWindow;

use super::camera::{
    FitView, PIXELS_PER_LINE, boxes_bounds, center_keeping, fit_rect, screen_to_world,
    wheel_zoom_factor, zoom_about,
};
use super::font::{EditorFont, setup_editor_font};
use super::inspect::{EditKey, EditorKeys, ParamInspector, inspector_text};
use super::interact::{
    DragState, EditorDrag, EditorPointer, EditorSelection, PipeGraphInteractPlugin, Selection,
    edge_segments, handle_pointer,
};
use super::layout::{
    KIND_FONT_SIZE, PIN_RADIUS, PinSide, TITLE_FONT_SIZE, fit_label, header_line_offsets,
    header_text_width, pin_offset, pin_positions, side_ports,
};
use super::preview::frame_to_rgba8;
use super::{EditorPreview, EditorRun, EditorStatus, GraphResource, NodeShape, NodeView};
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
const STATUS: Color = Color::srgb(0.85, 0.87, 0.92);
const STATUS_ERROR: Color = Color::srgb(1.0, 0.45, 0.4);

/// Width a preview image is drawn at, in world units; height follows the
/// frame's aspect ratio.
const PREVIEW_WIDTH: f32 = 200.0;
/// Gap between a node's box and its preview below it.
const PREVIEW_GAP: f32 = 12.0;

/// Rendering + device-input plugin. Adds [`PipeGraphInteractPlugin`] (and
/// through it [`super::PipeGraphEditorPlugin`]) if not already present.
/// Requires `DefaultPlugins` and a `Camera2d` (the editor example spawns one).
pub struct PipeGraphRenderPlugin;

impl Plugin for PipeGraphRenderPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<PipeGraphInteractPlugin>() {
            app.add_plugins(PipeGraphInteractPlugin);
        }
        app.add_systems(PreStartup, setup_editor_font)
            .add_systems(Startup, (spawn_status_text, spawn_inspector_text))
            .add_systems(
                Update,
                (
                    (gather_pointer_input, gather_key_input).before(super::route_keys),
                    update_inspector_text.after(super::EditorCoreSystems),
                    attach_node_visuals,
                    draw_graph.after(handle_pointer),
                    update_status_text.after(super::EditorCoreSystems),
                    update_preview.after(super::EditorCoreSystems),
                    (fit_camera.after(super::route_keys), navigate_camera).chain(),
                ),
            );
    }
}

/// Fill [`EditorPointer`] from the primary window's cursor (converted to world
/// space through the 2D camera) and the left mouse button. Keys are gathered
/// separately ([`gather_key_input`]) and routed by [`super::route_keys`].
///
/// Edge flags are OR-ed in rather than overwritten: `handle_pointer` clears
/// them once consumed, so nothing is lost if both ever ran out of step.
pub fn gather_pointer_input(
    mut pointer: ResMut<EditorPointer>,
    mouse: Res<ButtonInput<MouseButton>>,
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
}

/// Fill [`EditorKeys`] with this frame's key presses, in the order they
/// happened. Auto-repeat is kept for typed characters and Backspace/Delete
/// (holding them should keep going) but dropped for Tab, Enter, Esc and Space,
/// so holding Enter does not alternate between starting and applying an edit,
/// nor holding Space flicker play/pause.
pub fn gather_key_input(mut events: MessageReader<KeyboardInput>, mut keys: ResMut<EditorKeys>) {
    for event in events.read() {
        if event.state != ButtonState::Pressed {
            continue;
        }
        let (pressed, repeats): (Vec<EditKey>, bool) = match &event.logical_key {
            Key::Tab => (vec![EditKey::Tab], false),
            Key::Enter => (vec![EditKey::Enter], false),
            Key::Escape => (vec![EditKey::Escape], false),
            Key::Space => (vec![EditKey::Char(' ')], false),
            Key::Backspace => (vec![EditKey::Backspace], true),
            Key::Delete => (vec![EditKey::Delete], true),
            _ => (
                event
                    .text
                    .iter()
                    .flat_map(|text| text.chars())
                    .filter(|c| !c.is_control())
                    .map(EditKey::Char)
                    .collect(),
                true,
            ),
        };
        if repeats || !event.repeat {
            keys.0.extend(pressed);
        }
    }
}

/// Give each new node view a filled box sprite, a title and port labels. The
/// labels are children, so they follow the node when dragged and are
/// despawned with it.
pub fn attach_node_visuals(
    mut commands: Commands,
    font: Res<EditorFont>,
    added: Query<(Entity, &NodeView, &NodeShape), Added<NodeView>>,
) {
    for (entity, view, shape) in &added {
        // Two header lines, each shortened to fit the fixed-width box: the
        // id (what the user named it) and, smaller and dimmer, its kind.
        let (title_y, kind_y) = header_line_offsets(shape.size.y);
        let mut node = commands.entity(entity);
        node.insert(Sprite::from_color(BOX_FILL, shape.size));
        node.with_child((
            Text2d::new(fit_label(&view.id.0, TITLE_FONT_SIZE, header_text_width())),
            font.text_font(TITLE_FONT_SIZE),
            TextColor(TITLE),
            Transform::from_xyz(0.0, title_y, 1.0),
        ));
        node.with_child((
            Text2d::new(fit_label(&shape.kind, KIND_FONT_SIZE, header_text_width())),
            font.text_font(KIND_FONT_SIZE),
            TextColor(PORT_LABEL),
            Transform::from_xyz(0.0, kind_y, 1.0),
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
                    font.text_font(11.0),
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

/// Marker for the screen-space status line.
#[derive(Component)]
pub struct StatusText;

pub fn spawn_status_text(mut commands: Commands, font: Res<EditorFont>) {
    commands.spawn((
        StatusText,
        Text::new(""),
        font.text_font(14.0),
        TextColor(STATUS),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));
}

/// Show play state, the session's last error and any rejected command.
pub fn update_status_text(
    run: Res<EditorRun>,
    status: Res<EditorStatus>,
    mut text: Query<(&mut Text, &mut TextColor), With<StatusText>>,
) {
    if !run.is_changed() && !status.is_changed() {
        return;
    }
    let Ok((mut text, mut color)) = text.single_mut() else {
        return;
    };
    let mut line = if run.playing {
        format!("playing (frame {}) - Space to pause", run.frames)
    } else {
        "paused - Space to play".to_string()
    };
    for problem in [&status.error, &status.rejected].into_iter().flatten() {
        line.push('\n');
        line.push_str(problem);
    }
    text.0 = line;
    color.0 = if status.error.is_some() || status.rejected.is_some() {
        STATUS_ERROR
    } else {
        STATUS
    };
}

/// Marker for the preview sprite (one, reused across selections).
#[derive(Component)]
pub struct PreviewSprite {
    /// `(EditorPreview::generation, tap seq)` the current texture was built
    /// from. The generation matters: a newly selected node gets a fresh tap
    /// whose `seq` restarts, and could otherwise collide with the old one's.
    shown: (u64, u64),
}

/// Show the selected node's latest output frame under its box.
///
/// The sprite is spawned on first use and hidden while there is nothing to
/// show (no selection, a node without a frame output, nothing published yet,
/// or a frame with no picture). The texture is only rebuilt when the tap's
/// `(generation, seq)` moves, so a paused pipeline costs nothing per frame.
pub fn update_preview(
    mut commands: Commands,
    preview: Res<EditorPreview>,
    mut images: ResMut<Assets<Image>>,
    views: Query<(&NodeView, &NodeShape, &Transform), Without<PreviewSprite>>,
    mut sprite: Query<(
        &mut PreviewSprite,
        &mut Sprite,
        &mut Transform,
        &mut Visibility,
    )>,
) {
    let latest = preview.tap.as_ref().map(|tap| tap.latest_with_seq());
    let frame = latest
        .as_ref()
        .and_then(|(_, payload)| payload.as_ref()?.as_frame().cloned());
    let anchor = preview.target.as_ref().and_then(|(node, _)| {
        views
            .iter()
            .find(|(view, _, _)| &view.id == node)
            .map(|(_, shape, tf)| (tf.translation, shape.size))
    });

    let (Some(frame), Some((center, size)), Some((seq, _))) = (frame, anchor, latest) else {
        if let Ok((_, _, _, mut vis)) = sprite.single_mut() {
            *vis = Visibility::Hidden;
        }
        return;
    };
    let Some((w, h, rgba)) = frame_to_rgba8(&frame) else {
        // Not showable (e.g. more than four channels): hide rather than keep
        // showing the previous selection's picture.
        if let Ok((_, _, _, mut vis)) = sprite.single_mut() {
            *vis = Visibility::Hidden;
        }
        return;
    };
    let shown = (preview.generation, seq);
    let draw = Vec2::new(PREVIEW_WIDTH, PREVIEW_WIDTH * h as f32 / w as f32);
    let at = Vec3::new(
        center.x,
        center.y - size.y / 2.0 - PREVIEW_GAP - draw.y / 2.0,
        center.z,
    );
    let image = Image::new(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );

    match sprite.single_mut() {
        Ok((mut marker, mut sprite, mut tf, mut vis)) => {
            if marker.shown != shown {
                images.remove(&sprite.image);
                sprite.image = images.add(image);
                marker.shown = shown;
            }
            sprite.custom_size = Some(draw);
            tf.translation = at;
            *vis = Visibility::Visible;
        }
        Err(_) => {
            commands.spawn((
                PreviewSprite { shown },
                Sprite {
                    image: images.add(image),
                    custom_size: Some(draw),
                    ..default()
                },
                Transform::from_translation(at),
                Visibility::Visible,
            ));
        }
    }
}

/// Marker for the screen-space parameter inspector panel.
#[derive(Component)]
pub struct InspectorText;

pub fn spawn_inspector_text(mut commands: Commands, font: Res<EditorFont>) {
    commands.spawn((
        InspectorText,
        Text::new(""),
        font.text_font(14.0),
        TextColor(STATUS),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            right: Val::Px(10.0),
            ..default()
        },
    ));
}

/// Show the selected node's parameters (see [`super::inspect`]); empty when
/// nothing is selected. Highlighted while a value is being edited.
pub fn update_inspector_text(
    inspector: Res<ParamInspector>,
    graph: Res<GraphResource>,
    mut text: Query<(&mut Text, &mut TextColor), With<InspectorText>>,
) {
    if !inspector.is_changed() && !graph.is_changed() {
        return;
    }
    let Ok((mut text, mut color)) = text.single_mut() else {
        return;
    };
    text.0 = inspector_text(&inspector, &graph.0).unwrap_or_default();
    color.0 = if inspector.is_editing() {
        SELECTED
    } else {
        STATUS
    };
}

/// The projection's orthographic scale (the zoom), mutably, or `None` for a
/// perspective or custom projection, which the editor does not navigate.
fn ortho_scale(projection: &mut Projection) -> Option<&mut f32> {
    match projection {
        Projection::Orthographic(ortho) => Some(&mut ortho.scale),
        _ => None,
    }
}

/// Pan and zoom the editor camera from the mouse.
///
/// - **Pan**: pressing the middle or right button remembers the world point
///   under the cursor; while either is held, the camera is re-centered every
///   frame so that point stays exactly under the cursor
///   ([`center_keeping`]). Re-solving from the grab point, rather than adding
///   up per-frame deltas, means no drift and no dependence on raw mouse
///   motion (which some platforms report unaccelerated). The left button is
///   left alone: it belongs to selection, node drags and wiring.
/// - **Zoom**: the wheel zooms about the world point under the cursor (or the
///   view center when the cursor is outside the window), clamped by
///   [`zoom_about`]. Pixel-unit deltas (touchpads) are converted to notches.
///
/// Works in screen space from the camera's current `Transform` and the
/// logical viewport size, assuming Bevy's default 2D projection
/// (`ScalingMode::WindowSize`, centered origin) that `Camera2d` requires.
pub fn navigate_camera(
    mouse: Res<ButtonInput<MouseButton>>,
    scroll: Res<AccumulatedMouseScroll>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<(&Camera, &mut Transform, &mut Projection), With<Camera2d>>,
    mut grab: Local<Option<Vec2>>,
) {
    let Ok((camera, mut tf, mut projection)) = cameras.single_mut() else {
        return;
    };
    let (Some(viewport), Some(scale)) =
        (camera.logical_viewport_size(), ortho_scale(&mut projection))
    else {
        return;
    };
    let cursor = windows.single().ok().and_then(Window::cursor_position);
    let mut center = tf.translation.truncate();

    if scroll.delta.y != 0.0 {
        let lines = match scroll.unit {
            MouseScrollUnit::Line => scroll.delta.y,
            MouseScrollUnit::Pixel => scroll.delta.y / PIXELS_PER_LINE,
        };
        let anchor = cursor.map_or(center, |c| screen_to_world(center, *scale, viewport, c));
        (center, *scale) = zoom_about(center, *scale, anchor, wheel_zoom_factor(lines));
    }

    let panning = [MouseButton::Middle, MouseButton::Right];
    if !mouse.any_pressed(panning) {
        *grab = None;
    } else if let Some(cursor) = cursor {
        match *grab {
            // Solved after any zoom above, so zooming mid-pan keeps the
            // grabbed point under the cursor too.
            Some(world) => center = center_keeping(world, cursor, *scale, viewport),
            None => *grab = Some(screen_to_world(center, *scale, viewport, cursor)),
        }
    }

    // Write only on change so `Transform` change detection stays quiet.
    if center != tf.translation.truncate() {
        tf.translation.x = center.x;
        tf.translation.y = center.y;
    }
}

/// A node view's box as [`fit_camera`] reads it.
type NodeBox = (&'static NodeShape, &'static Transform);

/// Frame every node box ([`fit_rect`]): once automatically, as soon as the
/// first nodes exist and the viewport size is known, and again whenever
/// [`FitView`] is requested. A request with no nodes is dropped; one made
/// before the camera knows its viewport waits for it.
pub fn fit_camera(
    mut fit: ResMut<FitView>,
    views: Query<NodeBox, (With<NodeView>, Without<Camera2d>)>,
    mut cameras: Query<(&Camera, &mut Transform, &mut Projection), With<Camera2d>>,
    mut startup_done: Local<bool>,
) {
    if *startup_done && !fit.requested {
        return;
    }
    let Some(bounds) = boxes_bounds(
        views
            .iter()
            .map(|(shape, tf)| (tf.translation.truncate(), shape.size)),
    ) else {
        // Nothing to frame; keep waiting for the first nodes at startup.
        if fit.requested {
            fit.requested = false;
        }
        return;
    };
    let Ok((camera, mut tf, mut projection)) = cameras.single_mut() else {
        return;
    };
    let Some((center, new_scale)) = camera
        .logical_viewport_size()
        .and_then(|viewport| fit_rect(bounds, viewport))
    else {
        return;
    };
    let Some(scale) = ortho_scale(&mut projection) else {
        return;
    };
    *scale = new_scale;
    tf.translation.x = center.x;
    tf.translation.y = center.y;
    *startup_done = true;
    fit.requested = false;
}
