//! Interactive node editor demo.
//!
//! ```sh
//! cargo run --features bevy --example editor
//! ```
//!
//! Opens a window showing a small sample graph (split → 3 × clear_channel →
//! merge). Drag a node's body to move it; drag from an output pin (right side)
//! to an input pin (left side) to connect; click a node or an edge and press
//! Delete/Backspace to remove it.
//!
//! The sample graph is built by queueing ordinary `EditorCommand`s — the same
//! path the UI uses — so the example also exercises the command pipeline. It is
//! a topology demo only; nothing is executed.

use bevy::prelude::*;
use pipe_graph::editor::EditorCommand;
use pipe_graph::graph::{NodeId, NodeSpec, Params, PortId};
use pipe_graph::systems::{EditorCommands, PipeGraphEditorPlugin, PipeGraphRenderPlugin};

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
        .add_systems(Startup, (spawn_camera, queue_sample_graph))
        .run();
}

fn spawn_camera(mut commands: Commands) {
    // Center the view on the auto-layout's three columns (x = 0, 260, 520).
    commands.spawn((Camera2d, Transform::from_xyz(260.0, 0.0, 0.0)));
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
