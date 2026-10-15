//! Interactive node editor demo.
//!
//! ```sh
//! cargo run --features bevy --example editor
//! ```
//!
//! Opens a window showing a small sample graph (split → 3 × clear_channel →
//! merge) running live on a generated test pattern:
//!
//! - click a node to select it and see its latest output frame under it;
//! - the selected node's parameters are listed top right: Tab picks one,
//!   Enter edits it (type, Backspace), Enter again applies it and the
//!   pipeline rebuilds live, Esc cancels — try `channels` on `split`/`merge`
//!   or `channel` on a `clear_*` node;
//! - Space pauses/resumes the pipeline (the status line, top left, shows the
//!   frame count and any build or run error);
//! - drag a node's body to move it; drag from an output pin (right side) to an
//!   input pin (left side) to connect; click a node or an edge and press
//!   Delete/Backspace to remove it. Every edit rebuilds the pipeline, and
//!   previews keep updating across the rebuild.
//!
//! The sample graph is built by queueing ordinary `EditorCommand`s — the same
//! path the UI uses — so the example also exercises the command pipeline.

use bevy::prelude::*;
use pipe_graph::data::{Frame, Payload};
use pipe_graph::editor::EditorCommand;
use pipe_graph::graph::{NodeId, NodeSpec, Params, PortId};
use pipe_graph::systems::{
    EditorCommands, EditorRun, EditorSelection, EditorSession, PipeGraphEditorPlugin,
    PipeGraphRenderPlugin, Selection,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "pipe-graph editor".to_string(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((PipeGraphEditorPlugin, PipeGraphRenderPlugin))
        .insert_resource(EditorRun {
            playing: true,
            ..default()
        })
        // Start with the result selected, so its preview is on screen at once.
        .insert_resource(EditorSelection(Some(Selection::Node(NodeId(
            "merge".to_string(),
        )))))
        .add_systems(
            Startup,
            (spawn_camera, queue_sample_graph, feed_test_pattern),
        )
        .run();
}

fn spawn_camera(mut commands: Commands) {
    // Center the view on the sample's auto-layout: three columns (x = 0, 220,
    // 440) and three rows (y = 0, -120, -240), with room for previews below.
    commands.spawn((Camera2d, Transform::from_xyz(220.0, -180.0, 0.0)));
}

/// A 160x90 RGB gradient with a checker, fed into the split. The session
/// remembers it and re-injects it after every rebuild.
fn feed_test_pattern(mut session: NonSendMut<EditorSession>) {
    let (w, h) = (160u32, 90u32);
    let px = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let checker = if (x / 16 + y / 16) % 2 == 0 { 60 } else { 0 };
                (
                    (x * 255 / w) as u8,
                    (y * 255 / h) as u8,
                    (255 - x * 255 / w) as u8 / 2 + checker,
                )
            })
        })
        .collect();
    session.0.set_input(
        &NodeId("split".to_string()),
        "in",
        Payload::Frame(Frame::from_rgb8(w, h, px)),
    );
}

fn spec(id: &str, kind: &str, params: &[(&str, &str)]) -> NodeSpec {
    let mut p = Params::new();
    for (k, v) in params {
        p.insert(k.to_string(), v.to_string());
    }
    NodeSpec {
        id: NodeId(id.to_string()),
        kind: kind.to_string(),
        params: p,
    }
}

fn port(node: &str, port: &str) -> (NodeId, PortId) {
    (NodeId(node.to_string()), PortId(port.to_string()))
}

fn queue_sample_graph(mut queue: ResMut<EditorCommands>) {
    let q = &mut queue.queue;
    q.push(EditorCommand::AddNode(spec(
        "split",
        "split",
        &[("channels", "3")],
    )));
    q.push(EditorCommand::AddNode(spec(
        "merge",
        "merge",
        &[("channels", "3")],
    )));
    for (i, channel) in ["red", "green", "blue"].into_iter().enumerate() {
        let id = format!("clear_{channel}");
        q.push(EditorCommand::AddNode(spec(
            &id,
            "clear_channel",
            &[("channel", channel)],
        )));
        q.push(EditorCommand::Connect {
            from: port("split", &format!("out{i}")),
            to: port(&id, "in"),
        });
        q.push(EditorCommand::Connect {
            from: port(&id, "out"),
            to: port("merge", &format!("in{i}")),
        });
    }
}
