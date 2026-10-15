//! Bevy editor layer: a thin *view/controller* over a live editing session.
//! The core (`data`/`graph`/`exec`/`stages`/`editor`) never depends on Bevy;
//! this module — compiled only under `--features bevy` — mirrors the session's
//! graph into ECS, routes user intents back into it, and runs the pipeline so
//! the editor shows live output.
//!
//! Design: a [`LiveSession`] is the single source of truth, held as the
//! [`EditorSession`] non-send resource. It owns the `Graph`, the canvas
//! [`Layout`](crate::editor::Layout), and a `Runtime` that is rebuilt after
//! every topology or parameter edit. ECS entities are *views* ([`NodeView`]),
//! not the data model. Editor intents are queued as
//! [`crate::editor::EditorCommand`]s in [`EditorCommands`] and applied through
//! the session; [`GraphResource`] is a read-only mirror of its graph for
//! systems that only draw or hit-test; a sync system then spawns, reshapes or
//! despawns node views so what's on screen matches the graph. All graph,
//! layout and rebuild logic lives in the Bevy-free [`crate::editor`] module and
//! is tested there; the systems here are glue.
//!
//! The layer is split into three plugins so each can run where it makes sense:
//!
//! - [`PipeGraphEditorPlugin`] (this file) — the headless core: the session,
//!   command application, view sync, running ([`EditorRun`]), the selected
//!   node's preview tap ([`EditorPreview`]) and a status summary
//!   ([`EditorStatus`]). Works under `MinimalPlugins`.
//! - [`PipeGraphInteractPlugin`] ([`interact`], [`inspect`]) — turns an
//!   abstract pointer state ([`EditorPointer`]) into node moves, selection and
//!   queued `EditorCommand`s, and routes abstract keystrokes ([`EditorKeys`])
//!   to parameter edits through the [`ParamInspector`] or to shortcuts
//!   (play/pause, delete). Also headless: it never reads windows or devices,
//!   so tests drive it by writing those resources.
//! - [`PipeGraphRenderPlugin`] ([`render`]) — the only part that needs a
//!   window/renderer: fills [`EditorPointer`] from mouse/keyboard, gives views
//!   sprites and labels, draws pins and edges with gizmos, and shows the
//!   status line, the parameter inspector and the selected node's preview.
//!
//! Geometry shared by rendering and hit testing lives in [`layout`] as plain
//! functions, so "where is this pin" has exactly one answer.

pub mod inspect;
pub mod interact;
pub mod layout;
pub mod preview;
pub mod render;

use bevy::prelude::*;

use crate::data::PayloadKind;
use crate::editor::{EditorCommand, LiveSession, SessionError, view_diff};
use crate::exec::{PortSet, Tap, builtin_registry};
use crate::graph::{Graph, NodeId, Params, PortId};

pub use self::inspect::{EditKey, EditorKeys, ParamInspector, route_keys};
pub use self::interact::{
    DragState, EditorDrag, EditorPointer, EditorSelection, PipeGraphInteractPlugin, Selection,
    handle_pointer,
};
pub use self::render::PipeGraphRenderPlugin;

/// The authoritative editing session: graph, canvas layout and live runtime.
///
/// A *non-send* resource, because a session owns a `Registry` and a `Runtime`
/// whose nodes are `Send` but not `Sync`; Bevy runs the systems that touch it
/// on the main thread. Read it with `NonSend<EditorSession>`, and change the
/// graph by queueing [`EditorCommands`] rather than calling
/// [`LiveSession::apply`] directly, so views and the [`GraphResource`] mirror
/// stay in step. Calling the session's own non-graph methods directly —
/// `set_input`, `watch`, `set_max_iters`, taps — is fine.
///
/// [`PipeGraphEditorPlugin`] inserts an empty session over
/// [`builtin_registry`] unless the app already has one, so an app can start
/// from its own registry or graph by inserting an `EditorSession` first.
pub struct EditorSession(pub LiveSession);

impl Default for EditorSession {
    fn default() -> Self {
        Self(LiveSession::new(builtin_registry()))
    }
}

/// Read-only mirror of the session's graph, refreshed whenever a command
/// changes it. Drawing and hit-testing read this rather than the non-send
/// session so they can stay ordinary (parallelizable) systems. Writing to it
/// has no effect on the pipeline and is overwritten by the next edit.
#[derive(Resource, Default)]
pub struct GraphResource(pub Graph);

/// Queue of pending editor intents, drained each frame by [`apply_editor_commands`].
/// UI code pushes onto this instead of mutating the graph directly.
#[derive(Resource, Default)]
pub struct EditorCommands {
    pub queue: Vec<EditorCommand>,
}

/// Whether the pipeline is running. While `playing`, [`run_session`] calls
/// [`LiveSession::run_once`] once per frame.
///
/// Playback pauses itself when a source reports end of stream; other run
/// failures are shown in [`EditorStatus`] but do not pause, so fixing the
/// offending parameter resumes output without another click.
#[derive(Resource, Debug, Clone, Default)]
pub struct EditorRun {
    pub playing: bool,
    /// Frames run since the app started.
    pub frames: u64,
}

/// A displayable summary of the session, refreshed every frame. Plain strings,
/// so the renderer needs no access to the non-send session.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct EditorStatus {
    /// The session's last build or run error, if any.
    pub error: Option<String>,
    /// Why the most recent queued command was rejected, if it was.
    pub rejected: Option<String>,
}

/// The live-preview tap on the selected node's first frame output.
///
/// [`sync_preview`] moves the tap as the selection changes; the renderer reads
/// it with [`Tap::latest_with_seq`] and re-uploads the image only when the
/// sequence number moves.
#[derive(Resource, Clone, Default)]
pub struct EditorPreview {
    pub target: Option<(NodeId, PortId)>,
    pub tap: Option<Tap>,
    /// Bumped every time `tap` is replaced. A fresh tap's sequence numbers
    /// restart from zero, so a consumer caching "the frame I last showed" must
    /// key it on `(generation, seq)` rather than `seq` alone.
    pub generation: u64,
}

/// A view entity mirroring one graph node. Maps an ECS entity to a `NodeId`.
///
/// The view's on-screen position (box center) is its `Transform` translation.
/// A new view starts at the node's slot in the session's
/// [`Layout`](crate::editor::Layout); dragging it records the new spot back
/// there via [`EditorCommand::MoveNode`]. Position is deliberately *not*
/// stored in the graph: layout is a property of one editor's view, not of the
/// pipeline being described.
#[derive(Component, Debug, Clone)]
pub struct NodeView {
    pub id: NodeId,
}

/// The visual shape of a node view: its kind (for the label), the ports it
/// declares (for pins), and the resulting box size.
///
/// Computed from the registry. Ports can depend on params (e.g. Split's channel
/// count), so `params` records what the shape was built from; when a
/// [`EditorCommand::SetParam`] changes them, [`sync_graph_views`] rebuilds the
/// view. A kind the registry does not know — or params it rejects — yields an
/// empty `PortSet`: the node still renders as a pin-less box rather than
/// disappearing or panicking.
#[derive(Component, Debug, Clone)]
pub struct NodeShape {
    pub kind: String,
    pub params: Params,
    pub ports: PortSet,
    pub size: Vec2,
}

impl NodeShape {
    pub fn new(kind: impl Into<String>, params: Params, ports: PortSet) -> Self {
        let size = layout::node_size(&ports);
        Self {
            kind: kind.into(),
            params,
            ports,
            size,
        }
    }
}

/// Plugin wiring the headless editor core (session, command application, view
/// sync, running and preview) into a Bevy `App`. Safe under `MinimalPlugins`.
pub struct PipeGraphEditorPlugin;

/// System set for the core systems, so input handling can be ordered before
/// them (intents queued this frame are applied this frame).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EditorCoreSystems;

impl Plugin for PipeGraphEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_non_send_resource::<EditorSession>()
            .init_resource::<GraphResource>()
            .init_resource::<EditorCommands>()
            .init_resource::<EditorRun>()
            .init_resource::<EditorStatus>()
            .init_resource::<EditorPreview>()
            .init_resource::<EditorSelection>()
            // Apply intents first, then reconcile the views with the new
            // graph, then run the (possibly rebuilt) pipeline.
            .add_systems(
                Update,
                (
                    apply_editor_commands,
                    sync_graph_views,
                    sync_preview,
                    run_session,
                )
                    .chain()
                    .in_set(EditorCoreSystems),
            );
    }
}

/// Drain queued editor commands into the session (one rebuild for the whole
/// batch), refresh the [`GraphResource`] mirror if the graph changed, and
/// note a rejected command in [`EditorStatus`].
///
/// Removing the previewed node also drops its tap inside the session, so the
/// preview target is forgotten here; if the batch re-adds a node under the same
/// id (a "replace node" edit), [`sync_preview`] then taps the new one.
pub fn apply_editor_commands(
    mut commands: ResMut<EditorCommands>,
    mut session: NonSendMut<EditorSession>,
    mut graph: ResMut<GraphResource>,
    mut status: ResMut<EditorStatus>,
    mut preview: ResMut<EditorPreview>,
) {
    // Mirror on the first frame too: an app may have inserted a session that
    // already holds a graph.
    let first_frame = graph.is_added();
    if commands.queue.is_empty() && !first_frame {
        return;
    }
    let outcomes = session.0.apply_all(commands.queue.drain(..));
    let previewed_removed = outcomes.iter().any(|o| match (o, &preview.target) {
        (crate::editor::CommandOutcome::NodeRemoved(n), Some((target, _))) => n == target,
        _ => false,
    });
    if previewed_removed {
        preview.target = None;
        preview.tap = None;
    }
    if let Some(rejected) = outcomes.iter().rev().find_map(|o| match o {
        crate::editor::CommandOutcome::Rejected(e) => Some(format!("{e:?}")),
        _ => None,
    }) {
        status.rejected = Some(rejected);
    } else if !outcomes.is_empty() {
        status.rejected = None;
    }
    if first_frame || outcomes.iter().any(|o| o.changes_graph()) {
        graph.0 = session.0.graph().clone();
    }
}

/// Spawn a [`NodeView`] for each new graph node, rebuild views whose kind or
/// params changed (their ports may have), and despawn views whose node is gone,
/// so the ECS view mirrors the graph.
///
/// New views get a [`NodeShape`] and a `Transform` at the node's position in
/// the session's layout; rebuilt views keep wherever the user dragged them.
pub fn sync_graph_views(
    graph: Res<GraphResource>,
    session: NonSend<EditorSession>,
    views: Query<(Entity, &NodeView, &NodeShape, &Transform)>,
    mut commands: Commands,
) {
    if !graph.is_changed() {
        return;
    }

    let present: std::collections::HashSet<NodeId> =
        views.iter().map(|(_, v, _, _)| v.id.clone()).collect();
    let diff = view_diff(&graph.0, &present);

    let registry = session.0.registry();
    let shape_of = |id: &NodeId| {
        let spec = graph.0.nodes.get(id)?;
        let ports = registry.ports_of(spec).unwrap_or_default();
        Some(NodeShape::new(
            spec.kind.clone(),
            spec.params.clone(),
            ports,
        ))
    };

    for id in diff.spawn {
        let Some(shape) = shape_of(&id) else {
            continue;
        };
        let pos = session
            .0
            .layout()
            .position(&id)
            .map(layout::canvas_to_world)
            .unwrap_or_default();
        commands.spawn((
            NodeView { id },
            shape,
            Transform::from_translation(pos.extend(0.0)),
        ));
    }

    for (entity, view, shape, tf) in views.iter() {
        if diff.despawn.contains(&view.id) {
            commands.entity(entity).despawn();
            continue;
        }
        let Some(spec) = graph.0.nodes.get(&view.id) else {
            continue;
        };
        if spec.kind != shape.kind || spec.params != shape.params {
            // Respawn rather than patch: the renderer builds pin labels once
            // per view, and a fresh entity gets them rebuilt for free.
            commands.entity(entity).despawn();
            if let Some(shape) = shape_of(&view.id) {
                commands.spawn((
                    NodeView {
                        id: view.id.clone(),
                    },
                    shape,
                    *tf,
                ));
            }
        }
    }
}

/// Keep [`EditorPreview`]'s tap on the selected node's first output that can
/// carry a frame, detaching it from the previous target.
pub fn sync_preview(
    selection: Res<EditorSelection>,
    graph: Res<GraphResource>,
    mut session: NonSendMut<EditorSession>,
    mut preview: ResMut<EditorPreview>,
) {
    if !selection.is_changed() && !graph.is_changed() {
        return;
    }
    let target = match &selection.0 {
        Some(Selection::Node(id)) => graph.0.nodes.get(id).and_then(|spec| {
            let ports = session.0.registry().ports_of(spec).ok()?;
            let port = ports
                .outputs
                .iter()
                .find(|p| matches!(p.kind, PayloadKind::Frame | PayloadKind::Any))?;
            Some((id.clone(), port.id.clone()))
        }),
        _ => None,
    };
    if target == preview.target {
        return;
    }
    if let Some((node, port)) = preview.target.take() {
        session.0.remove_taps(&node, &port.0);
    }
    preview.tap = target
        .as_ref()
        .map(|(node, port)| session.0.add_tap(node, &port.0));
    preview.target = target;
    preview.generation += 1;
}

/// While [`EditorRun::playing`], run the pipeline once per frame; always
/// refresh [`EditorStatus::error`] from the session.
pub fn run_session(
    mut session: NonSendMut<EditorSession>,
    mut run: ResMut<EditorRun>,
    mut status: ResMut<EditorStatus>,
) {
    if run.playing && session.0.runtime().is_some() {
        match session.0.run_once() {
            Ok(()) => run.frames += 1,
            Err(SessionError::Run(e)) if e.is_end_of_stream() => run.playing = false,
            Err(_) => {}
        }
    }
    let error = session.0.last_error().map(ToString::to_string);
    if status.error != error {
        status.error = error;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NodeSpec;

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

    fn with_params(id: &str, kind: &str, params: &[(&str, &str)]) -> EditorCommand {
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

    fn id(s: &str) -> NodeId {
        NodeId(s.to_string())
    }

    fn headless() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(PipeGraphEditorPlugin);
        app
    }

    fn queue(app: &mut App, commands: impl IntoIterator<Item = EditorCommand>) {
        app.world_mut()
            .resource_mut::<EditorCommands>()
            .queue
            .extend(commands);
        app.update();
    }

    fn session(app: &App) -> &LiveSession {
        &app.world().non_send_resource::<EditorSession>().0
    }

    fn view(app: &mut App, node: &str) -> (Entity, NodeShape, Vec2) {
        let mut q = app
            .world_mut()
            .query::<(Entity, &NodeView, &NodeShape, &Transform)>();
        let found: Vec<_> = q
            .iter(app.world())
            .filter(|(_, v, _, _)| v.id.0 == node)
            .map(|(e, _, s, t)| (e, s.clone(), t.translation.truncate()))
            .collect();
        assert_eq!(found.len(), 1, "exactly one view for {node}");
        found.into_iter().next().unwrap()
    }

    fn red_pixel() -> crate::data::Payload {
        crate::data::Payload::Frame(crate::data::Frame::from_rgb8(1, 1, vec![(200, 100, 50)]))
    }

    #[test]
    fn commands_go_through_the_session_which_builds_a_runtime() {
        let mut app = headless();
        queue(
            &mut app,
            [
                with_params("a", "clear_channel", &[("channel", "red")]),
                with_params("b", "clear_channel", &[("channel", "green")]),
                EditorCommand::Connect {
                    from: (id("a"), PortId("out".into())),
                    to: (id("b"), PortId("in".into())),
                },
            ],
        );
        let s = session(&app);
        assert_eq!(s.graph().nodes.len(), 2);
        assert_eq!(s.graph().edges.len(), 1);
        assert!(s.runtime().is_some(), "{:?}", s.last_error());
        // The mirror tracks the session.
        let mirror = &app.world().resource::<GraphResource>().0;
        assert_eq!(mirror.nodes.len(), 2);
        assert_eq!(mirror.edges.len(), 1);
    }

    #[test]
    fn new_views_start_at_their_layout_position() {
        let mut app = headless();
        queue(
            &mut app,
            [
                with_params("a", "clear_channel", &[("channel", "red")]),
                with_params("b", "clear_channel", &[("channel", "green")]),
                EditorCommand::Connect {
                    from: (id("a"), PortId("out".into())),
                    to: (id("b"), PortId("in".into())),
                },
            ],
        );
        for n in ["a", "b"] {
            let canvas = session(&app).layout().position(&id(n)).unwrap();
            assert_eq!(view(&mut app, n).2, layout::canvas_to_world(canvas));
        }
        // b sits to the right of a (data flows left to right).
        assert!(view(&mut app, "b").2.x > view(&mut app, "a").2.x);
    }

    #[test]
    fn a_param_edit_that_changes_ports_reshapes_the_view_in_place() {
        let mut app = headless();
        queue(&mut app, [with_params("s", "split", &[("channels", "3")])]);
        let (e, shape, _) = view(&mut app, "s");
        assert_eq!(shape.ports.outputs.len(), 3);
        let dragged = Vec2::new(123.0, -45.0);
        app.world_mut().get_mut::<Transform>(e).unwrap().translation = dragged.extend(0.0);

        queue(
            &mut app,
            [EditorCommand::SetParam {
                node: id("s"),
                key: "channels".into(),
                value: "2".into(),
            }],
        );
        let (_, shape, pos) = view(&mut app, "s");
        assert_eq!(shape.ports.outputs.len(), 2);
        assert_eq!(shape.params.get("channels").map(String::as_str), Some("2"));
        assert_eq!(pos, dragged, "a rebuilt view keeps where it was dragged");
    }

    #[test]
    fn dropping_a_dragged_node_records_it_in_the_layout() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(PipeGraphInteractPlugin);
        queue(
            &mut app,
            [with_params("a", "clear_channel", &[("channel", "red")])],
        );
        let (_, _, start) = view(&mut app, "a");

        let grab = start + Vec2::new(0.0, -5.0);
        let target = Vec2::new(300.0, -200.0);
        for pointer in [
            EditorPointer {
                world: Some(grab),
                pressed: true,
                just_pressed: true,
                ..default()
            },
            EditorPointer {
                world: Some(grab + target),
                pressed: true,
                ..default()
            },
            EditorPointer {
                world: Some(grab + target),
                just_released: true,
                ..default()
            },
        ] {
            *app.world_mut().resource_mut::<EditorPointer>() = pointer;
            app.update();
        }

        let (_, _, end) = view(&mut app, "a");
        assert_eq!(end, start + target);
        assert_eq!(
            session(&app).layout().position(&id("a")),
            Some(layout::world_to_canvas(end))
        );
        assert!(
            session(&app).runtime().is_some(),
            "a move is layout-only and never drops the runtime"
        );
    }

    #[test]
    fn playing_runs_the_pipeline_and_build_errors_show_in_status() {
        let mut app = headless();
        queue(
            &mut app,
            [with_params("a", "clear_channel", &[("channel", "red")])],
        );
        app.world_mut()
            .non_send_resource_mut::<EditorSession>()
            .0
            .set_input(&id("a"), "in", red_pixel());

        app.update();
        assert!(
            session(&app).output(&id("a"), "out").is_none(),
            "paused by default"
        );

        app.world_mut().resource_mut::<EditorRun>().playing = true;
        app.update();
        app.update();
        assert_eq!(app.world().resource::<EditorRun>().frames, 2);
        let out = session(&app).output(&id("a"), "out").unwrap();
        assert_eq!(out.as_frame().unwrap().to_rgb8(), vec![(0, 100, 50)]);
        assert_eq!(app.world().resource::<EditorStatus>().error, None);

        // A bad param: no runtime, the error is shown, nothing runs.
        queue(
            &mut app,
            [EditorCommand::SetParam {
                node: id("a"),
                key: "channel".into(),
                value: "purple".into(),
            }],
        );
        let status = app.world().resource::<EditorStatus>().clone();
        assert!(
            status.error.as_deref().unwrap_or("").contains("purple"),
            "{status:?}"
        );
        assert_eq!(app.world().resource::<EditorRun>().frames, 2);

        // Fixing it resumes output without touching playback.
        queue(
            &mut app,
            [EditorCommand::SetParam {
                node: id("a"),
                key: "channel".into(),
                value: "blue".into(),
            }],
        );
        assert_eq!(app.world().resource::<EditorStatus>().error, None);
        assert_eq!(app.world().resource::<EditorRun>().frames, 3);
        let out = session(&app).output(&id("a"), "out").unwrap();
        assert_eq!(out.as_frame().unwrap().to_rgb8(), vec![(200, 100, 0)]);
    }

    #[test]
    fn a_rejected_command_is_reported() {
        let mut app = headless();
        queue(
            &mut app,
            [with_params("a", "clear_channel", &[("channel", "red")])],
        );
        queue(
            &mut app,
            [with_params("a", "clear_channel", &[("channel", "red")])],
        );
        let rejected = app.world().resource::<EditorStatus>().rejected.clone();
        assert!(rejected.unwrap_or_default().contains("Duplicate"));
    }

    /// Emits one frame, then reports end of stream until reset.
    struct OneShot(bool);

    impl crate::exec::Node for OneShot {
        fn ports(&self) -> PortSet {
            PortSet::new(
                vec![],
                vec![crate::exec::PortSpec::new("out", PayloadKind::Frame)],
            )
        }

        fn eval(
            &mut self,
            _: &crate::exec::Inputs,
            outputs: &mut crate::exec::Outputs,
        ) -> Result<(), crate::exec::NodeError> {
            if std::mem::replace(&mut self.0, true) {
                return Err(crate::exec::NodeError::EndOfStream);
            }
            outputs.set("out", red_pixel());
            Ok(())
        }
    }

    #[test]
    fn end_of_stream_pauses_playback_and_a_preinserted_session_is_kept() {
        let mut registry = builtin_registry();
        registry.register("one_shot", |_| {
            Ok(Box::new(OneShot(false)) as Box<dyn crate::exec::Node>)
        });
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_non_send_resource(EditorSession(LiveSession::new(registry)))
            .add_plugins(PipeGraphEditorPlugin);
        queue(&mut app, [with_params("src", "one_shot", &[])]);
        assert!(
            session(&app).runtime().is_some(),
            "the plugin kept the app's session (and its registry)"
        );

        app.world_mut().resource_mut::<EditorRun>().playing = true;
        app.update();
        assert!(app.world().resource::<EditorRun>().playing);
        app.update();
        let run = app.world().resource::<EditorRun>();
        assert!(!run.playing, "end of stream pauses");
        assert_eq!(run.frames, 1);
    }

    #[test]
    fn selecting_a_node_moves_the_preview_tap_to_it() {
        let mut app = headless();
        queue(
            &mut app,
            [
                with_params("a", "clear_channel", &[("channel", "red")]),
                with_params("b", "clear_channel", &[("channel", "green")]),
                EditorCommand::Connect {
                    from: (id("a"), PortId("out".into())),
                    to: (id("b"), PortId("in".into())),
                },
            ],
        );
        app.world_mut()
            .non_send_resource_mut::<EditorSession>()
            .0
            .set_input(&id("a"), "in", red_pixel());
        app.world_mut().resource_mut::<EditorRun>().playing = true;

        // Previewing an intermediate works even though capture is opt-in:
        // the tap is what makes `a`'s output observed.
        app.world_mut().resource_mut::<EditorSelection>().0 = Some(Selection::Node(id("a")));
        app.update();
        let preview = app.world().resource::<EditorPreview>().clone();
        assert_eq!(preview.target, Some((id("a"), PortId("out".into()))));
        let (seq, value) = preview.tap.as_ref().unwrap().latest_with_seq();
        assert_eq!(seq, 1);
        assert_eq!(
            value.unwrap().as_frame().unwrap().to_rgb8(),
            vec![(0, 100, 50)]
        );

        // The tap survives a rebuild caused by an unrelated param edit.
        queue(
            &mut app,
            [EditorCommand::SetParam {
                node: id("b"),
                key: "channel".into(),
                value: "blue".into(),
            }],
        );
        assert_eq!(preview.tap.as_ref().unwrap().seq(), 2);

        // Deselecting detaches it.
        app.world_mut().resource_mut::<EditorSelection>().0 = None;
        app.update();
        let preview = app.world().resource::<EditorPreview>();
        assert!(preview.target.is_none() && preview.tap.is_none());
    }

    #[test]
    fn replacing_the_previewed_node_keeps_its_preview_live() {
        let mut app = headless();
        queue(
            &mut app,
            [with_params("a", "clear_channel", &[("channel", "red")])],
        );
        app.world_mut()
            .non_send_resource_mut::<EditorSession>()
            .0
            .set_input(&id("a"), "in", red_pixel());
        app.world_mut().resource_mut::<EditorRun>().playing = true;
        app.world_mut().resource_mut::<EditorSelection>().0 = Some(Selection::Node(id("a")));
        app.update();
        let first = app.world().resource::<EditorPreview>().generation;

        // Remove and re-add `a` in one batch: the session drops the old tap.
        queue(
            &mut app,
            [
                EditorCommand::RemoveNode(id("a")),
                with_params("a", "clear_channel", &[("channel", "green")]),
            ],
        );
        app.world_mut()
            .non_send_resource_mut::<EditorSession>()
            .0
            .set_input(&id("a"), "in", red_pixel());
        app.update();

        let preview = app.world().resource::<EditorPreview>().clone();
        assert_eq!(preview.target, Some((id("a"), PortId("out".into()))));
        assert!(
            preview.generation > first,
            "a fresh tap is a new generation"
        );
        let (_, value) = preview.tap.as_ref().unwrap().latest_with_seq();
        assert_eq!(
            value.unwrap().as_frame().unwrap().to_rgb8(),
            vec![(200, 0, 50)],
            "the new node's output, not a frozen frame from the old one"
        );
    }
}
