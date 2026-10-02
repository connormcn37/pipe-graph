//! End-to-end editor session: drive a `LiveSession` the way a UI would —
//! build a graph by commands, feed it, run it, tweak parameters live — and
//! check that outputs follow the edits, that preview taps survive rebuilds,
//! and that invalid edits are reported instead of panicking.

use pipe_graph::data::{Frame, Payload};
use pipe_graph::editor::{CommandOutcome, EditorCommand, LiveSession, SessionError};
use pipe_graph::exec::builtin_registry;
use pipe_graph::graph::{NodeId, NodeSpec, Params, PortId};

fn id(s: &str) -> NodeId {
    NodeId(s.to_string())
}

fn node(name: &str, kind: &str, params: &[(&str, &str)]) -> NodeSpec {
    NodeSpec {
        id: id(name),
        kind: kind.to_string(),
        params: params
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<Params>(),
    }
}

fn set(node: &str, key: &str, value: &str) -> EditorCommand {
    EditorCommand::SetParam {
        node: id(node),
        key: key.to_string(),
        value: value.to_string(),
    }
}

/// The tap's latest frame as `(width, height, pixels)`.
fn tapped(tap: &pipe_graph::exec::Tap) -> (u32, u32, Vec<(u8, u8, u8)>) {
    let latest = tap.latest().expect("tap has a value");
    let f = latest.as_frame().expect("frame payload");
    (f.width, f.height, f.to_rgb8())
}

#[test]
fn live_session_follows_param_edits_and_keeps_taps() {
    let mut s = LiveSession::new(builtin_registry());

    // add -> connect
    let outcomes = s.apply_all([
        EditorCommand::AddNode(node("clear", "clear_channel", &[("channel", "red")])),
        EditorCommand::AddNode(node(
            "crop",
            "crop",
            &[("x", "0"), ("y", "0"), ("w", "1"), ("h", "1")],
        )),
        EditorCommand::Connect {
            from: (id("clear"), PortId("out".to_string())),
            to: (id("crop"), PortId("in".to_string())),
        },
    ]);
    assert!(outcomes.iter().all(CommandOutcome::changes_graph));
    assert!(s.last_error().is_none(), "{:?}", s.last_error());

    // Every node was placed as it was added. Edits never shuffle placed
    // nodes, so a "tidy up" relayout is what puts the crop to the right of
    // what feeds it.
    assert_eq!(s.layout().len(), 2);
    let tidy = pipe_graph::editor::Layout::auto(s.graph());
    *s.layout_mut() = tidy;
    let (cx, _) = s.layout().position(&id("clear")).unwrap();
    let (kx, _) = s.layout().position(&id("crop")).unwrap();
    assert!(kx > cx);

    // set_input -> run
    let src = Frame::from_rgb8(2, 1, vec![(10, 20, 30), (40, 50, 60)]);
    s.set_input(&id("clear"), "in", Payload::Frame(src));
    let tap = s.add_tap(&id("crop"), "out");
    s.run_once().unwrap();
    assert_eq!(tapped(&tap), (1, 1, vec![(0, 20, 30)]));
    let seq_before = tap.seq();

    // set_param (crop window) -> run: the old tap handle sees the new output.
    s.apply(set("crop", "x", "1"));
    assert!(s.last_error().is_none());
    s.run_once().unwrap();
    assert!(tap.seq() > seq_before);
    assert_eq!(tapped(&tap), (1, 1, vec![(0, 50, 60)]));

    // Widen the crop and change the upstream channel in one compound edit.
    s.apply_all([
        set("crop", "x", "0"),
        set("crop", "w", "2"),
        set("clear", "channel", "blue"),
    ]);
    s.run_once().unwrap();
    assert_eq!(tapped(&tap), (2, 1, vec![(10, 20, 0), (40, 50, 0)]));
    let output = s.output(&id("crop"), "out").unwrap().as_frame().unwrap();
    assert_eq!(output.width, 2);

    // Moving a node is layout-only: no rebuild, so the output is still there.
    s.apply(EditorCommand::MoveNode {
        node: id("crop"),
        to: (500.0, 80.0),
    });
    assert_eq!(s.layout().position(&id("crop")), Some((500.0, 80.0)));
    assert!(s.output(&id("crop"), "out").is_some());
}

#[test]
fn invalid_edits_are_reported_not_panicked() {
    let mut s = LiveSession::new(builtin_registry());
    s.apply(EditorCommand::AddNode(node(
        "clear",
        "clear_channel",
        &[("channel", "red")],
    )));
    s.set_input(
        &id("clear"),
        "in",
        Payload::Frame(Frame::from_rgb8(1, 1, vec![(255, 255, 255)])),
    );
    let tap = s.add_tap(&id("clear"), "out");
    s.run_once().unwrap();
    let good_seq = tap.seq();

    // Bad parameter value: the graph keeps it, the runtime is dropped, and the
    // reason is in last_error.
    let outcome = s.apply(set("clear", "channel", "purple"));
    assert!(matches!(outcome, CommandOutcome::ParamSet { .. }));
    assert!(s.runtime().is_none());
    let err = s.last_error().expect("bad param is reported").clone();
    assert!(matches!(err, SessionError::Schedule(_)));
    assert!(err.to_string().contains("purple"), "{err}");
    assert_eq!(s.run_once(), Err(err));
    // Previews freeze on the last good frame.
    assert_eq!(tap.seq(), good_seq);
    assert!(tap.latest().is_some());

    // Fixing it rebuilds; inputs and the tap come back.
    s.apply(set("clear", "channel", "green"));
    assert!(s.last_error().is_none());
    s.run_once().unwrap();
    assert!(tap.seq() > good_seq);
    assert_eq!(tapped(&tap).2, vec![(255, 0, 255)]);

    // Unknown kind: same policy.
    s.apply(EditorCommand::AddNode(node("mystery", "warp_drive", &[])));
    assert!(matches!(s.last_error(), Some(SessionError::Schedule(_))));
    assert!(s.run_once().is_err());
    s.apply(EditorCommand::RemoveNode(id("mystery")));
    assert!(s.last_error().is_none());
    s.run_once().unwrap();

    // A command the graph rejects outright is reported via its outcome only.
    let outcome = s.apply(EditorCommand::AddNode(node(
        "clear",
        "clear_channel",
        &[("channel", "red")],
    )));
    assert!(matches!(outcome, CommandOutcome::Rejected(_)));
    assert!(s.last_error().is_none());
    assert!(s.runtime().is_some());
}
