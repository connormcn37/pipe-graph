//! Keyboard routing and the parameter inspector: view and edit the selected
//! node's parameters from the keyboard.
//!
//! Like [`super::interact`], nothing here reads devices. Keystrokes arrive, in
//! order, as an abstract [`EditorKeys`] list that the render plugin fills from
//! real keyboard input, so tests drive the inspector by writing that resource
//! and calling `app.update()`.
//!
//! [`route_keys`] is the one place that decides what a key means, which is
//! what keeps typing a value from also firing shortcuts:
//!
//! - **While a value is being edited** the keyboard belongs to it: typed
//!   characters (Space included) and Backspace edit the text, Enter applies
//!   it as an [`EditorCommand::SetParam`] (the session rebuilds the pipeline,
//!   and an invalid value shows up in [`super::EditorStatus`]), Esc abandons
//!   it.
//! - **Otherwise**, with a node selected, Tab highlights its next parameter
//!   and Enter starts editing the highlighted one, pre-filled with its value.
//!   Space toggles [`EditorRun::playing`], and Delete/Backspace delete the
//!   selection (by raising [`EditorPointer::delete_just_pressed`] for
//!   [`super::handle_pointer`]).
//!
//! Keys are applied in the order they were pressed, so a fast "Enter, 2,
//! Enter" within one frame does what it would do across three.

use bevy::prelude::*;

use super::interact::{EditorPointer, EditorSelection, Selection};
use super::{EditorCommands, EditorRun, GraphResource};
use crate::editor::EditorCommand;
use crate::graph::{Graph, NodeId};

/// Longest parameter value the inspector accepts from the keyboard. Values
/// are short (numbers, names, paths); the cap keeps a stuck key or a paste
/// from growing the panel without bound.
pub const MAX_VALUE_LEN: usize = 256;

/// A key, as the editor sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditKey {
    /// A printable character (including space).
    Char(char),
    Backspace,
    Delete,
    Tab,
    Enter,
    Escape,
}

/// This frame's key presses, in order. One-shot: [`route_keys`] drains it.
#[derive(Resource, Debug, Clone, Default)]
pub struct EditorKeys(pub Vec<EditKey>);

/// Inspector state: which node it shows, which parameter is highlighted, and
/// the value being typed, if any.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct ParamInspector {
    pub node: Option<NodeId>,
    /// Index into [`params_of`] for `node`.
    pub field: usize,
    /// `Some(buffer)` while a value is being edited.
    pub editing: Option<String>,
}

impl ParamInspector {
    /// Whether keystrokes currently belong to a value being edited.
    pub fn is_editing(&self) -> bool {
        self.editing.is_some()
    }
}

/// A node's parameters as `(key, value)`, sorted by key so the order (and
/// therefore what Tab steps through) is stable.
pub fn params_of(graph: &Graph, node: &NodeId) -> Vec<(String, String)> {
    let Some(spec) = graph.nodes.get(node) else {
        return Vec::new();
    };
    let mut params: Vec<(String, String)> = spec
        .params
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    params.sort();
    params
}

/// Route this frame's [`EditorKeys`]: to the value being edited, to the
/// inspector, or to the editor's shortcuts. Runs before pointer handling (so
/// a Delete it passes on is acted on this frame) and before the core systems
/// (so an applied edit is rebuilt this frame).
pub fn route_keys(
    selection: Res<EditorSelection>,
    graph: Res<GraphResource>,
    mut keys: ResMut<EditorKeys>,
    mut inspector: ResMut<ParamInspector>,
    mut queue: ResMut<EditorCommands>,
    mut pointer: ResMut<EditorPointer>,
    mut run: ResMut<EditorRun>,
) {
    let keys = std::mem::take(&mut keys.0);

    // Work on a copy and write back only on change, so the panel (which
    // redraws on `is_changed`) is not rebuilt every frame.
    let mut state = inspector.clone();
    let selected = match &selection.0 {
        Some(Selection::Node(id)) => Some(id.clone()),
        _ => None,
    };
    if state.node != selected {
        // A new selection abandons any half-typed edit.
        state = ParamInspector {
            node: selected,
            ..default()
        };
    }
    let params = state
        .node
        .as_ref()
        .map(|node| params_of(&graph.0, node))
        .unwrap_or_default();
    if params.is_empty() {
        state.field = 0;
        state.editing = None;
    } else {
        state.field = state.field.min(params.len() - 1);
    }

    for key in keys {
        if let Some(buffer) = state.editing.as_mut() {
            match key {
                EditKey::Char(c) if buffer.chars().count() < MAX_VALUE_LEN => buffer.push(c),
                EditKey::Char(_) | EditKey::Tab | EditKey::Delete => {}
                EditKey::Backspace => {
                    buffer.pop();
                }
                EditKey::Escape => state.editing = None,
                EditKey::Enter => {
                    let value = state.editing.take().unwrap_or_default();
                    let (key, current) = &params[state.field];
                    if &value != current
                        && let Some(node) = state.node.clone()
                    {
                        queue.queue.push(EditorCommand::SetParam {
                            node,
                            key: key.clone(),
                            value,
                        });
                    }
                }
            }
            continue;
        }
        match key {
            EditKey::Tab if !params.is_empty() => {
                state.field = (state.field + 1) % params.len();
            }
            EditKey::Enter if !params.is_empty() => {
                state.editing = Some(params[state.field].1.clone());
            }
            EditKey::Char(' ') => run.playing = !run.playing,
            EditKey::Backspace | EditKey::Delete => pointer.delete_just_pressed = true,
            _ => {}
        }
    }

    inspector.set_if_neq(state);
}

/// The inspector panel's text for the current state, or `None` when no node
/// is selected. The highlighted parameter is marked with `>`; a value being
/// edited is shown with a trailing `_` cursor.
pub fn inspector_text(inspector: &ParamInspector, graph: &Graph) -> Option<String> {
    let node = inspector.node.as_ref()?;
    let kind = graph.nodes.get(node).map_or("?", |s| s.kind.as_str());
    let mut text = format!("{} ({kind})", node.0);
    let params = params_of(graph, node);
    if params.is_empty() {
        text.push_str("\n  (no parameters)");
        return Some(text);
    }
    for (i, (key, value)) in params.iter().enumerate() {
        let here = i == inspector.field;
        let shown = match (&inspector.editing, here) {
            (Some(buffer), true) => format!("{buffer}_"),
            _ => value.clone(),
        };
        let mark = if here { '>' } else { ' ' };
        text.push_str(&format!("\n{mark} {key} = {shown}"));
    }
    text.push_str(if inspector.is_editing() {
        "\n[Enter] apply  [Esc] cancel"
    } else {
        "\n[Tab] next  [Enter] edit"
    });
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{NodeSpec, Params};
    use crate::systems::{EditorSession, NodeShape, NodeView, PipeGraphInteractPlugin};

    use EditKey::*;

    fn id(s: &str) -> NodeId {
        NodeId(s.to_string())
    }

    fn add(id_: &str, kind: &str, params: &[(&str, &str)]) -> EditorCommand {
        EditorCommand::AddNode(NodeSpec {
            id: id(id_),
            kind: kind.to_string(),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<Params>(),
        })
    }

    /// Headless app with a merge node (one param) and a crop node (four).
    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(PipeGraphInteractPlugin);
        app.world_mut()
            .resource_mut::<EditorCommands>()
            .queue
            .extend([
                add("m", "merge", &[("channels", "3")]),
                add(
                    "c",
                    "crop",
                    &[("x", "0"), ("y", "0"), ("w", "2"), ("h", "2")],
                ),
            ]);
        app.update();
        app
    }

    fn select(app: &mut App, node: &str) {
        app.world_mut().resource_mut::<EditorSelection>().0 = Some(Selection::Node(id(node)));
        app.update();
    }

    fn press(app: &mut App, keys: &[EditKey]) {
        app.world_mut().resource_mut::<EditorKeys>().0 = keys.to_vec();
        app.update();
    }

    fn inspector(app: &App) -> ParamInspector {
        app.world().resource::<ParamInspector>().clone()
    }

    fn param(app: &App, node: &str, key: &str) -> Option<String> {
        app.world()
            .non_send_resource::<EditorSession>()
            .0
            .graph()
            .nodes
            .get(&id(node))?
            .params
            .get(key)
            .cloned()
    }

    fn panel(app: &App) -> Option<String> {
        inspector_text(&inspector(app), &app.world().resource::<GraphResource>().0)
    }

    fn has_node(app: &App, node: &str) -> bool {
        app.world()
            .resource::<GraphResource>()
            .0
            .nodes
            .contains_key(&id(node))
    }

    #[test]
    fn follows_the_selection_and_tab_cycles_sorted_params() {
        let mut app = app();
        select(&mut app, "c");
        assert_eq!(inspector(&app).node, Some(id("c")));
        let keys: Vec<String> = params_of(&app.world().resource::<GraphResource>().0, &id("c"))
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(keys, ["h", "w", "x", "y"]);
        for expected in [1, 2, 3, 0] {
            press(&mut app, &[Tab]);
            assert_eq!(inspector(&app).field, expected);
        }
        press(&mut app, &[Tab]);
        select(&mut app, "m");
        assert_eq!(
            inspector(&app).field,
            0,
            "a new selection starts at the top"
        );
    }

    #[test]
    fn enter_edit_enter_applies_a_set_param_and_rebuilds_the_view() {
        let mut app = app();
        select(&mut app, "m");
        press(&mut app, &[Enter]);
        assert_eq!(inspector(&app).editing.as_deref(), Some("3"));
        press(&mut app, &[Backspace, Char('2')]);
        assert_eq!(inspector(&app).editing.as_deref(), Some("2"));
        press(&mut app, &[Enter]);
        assert!(!inspector(&app).is_editing());
        assert_eq!(param(&app, "m", "channels").as_deref(), Some("2"));

        let mut q = app.world_mut().query::<(&NodeView, &NodeShape)>();
        let inputs = q
            .iter(app.world())
            .find(|(v, _)| v.id.0 == "m")
            .map(|(_, s)| s.ports.inputs.len());
        assert_eq!(inputs, Some(2));
    }

    #[test]
    fn keys_within_one_frame_apply_in_order() {
        let mut app = app();
        select(&mut app, "m");
        // Start editing, erase "3", type "5", erase it, type "7", apply —
        // all in one frame, exactly as if spread across several.
        press(
            &mut app,
            &[Enter, Backspace, Char('5'), Backspace, Char('7'), Enter],
        );
        assert!(!inspector(&app).is_editing());
        assert_eq!(param(&app, "m", "channels").as_deref(), Some("7"));
    }

    #[test]
    fn while_editing_space_and_backspace_are_text_not_shortcuts() {
        let mut app = app();
        select(&mut app, "m");
        let playing = app.world().resource::<EditorRun>().playing;
        // Enter then Backspace in the same frame: the Backspace edits the
        // value; it must not delete the selected node.
        press(&mut app, &[Enter, Backspace, Backspace, Char(' ')]);
        assert!(
            has_node(&app, "m"),
            "Backspace while editing deleted the node"
        );
        assert_eq!(inspector(&app).editing.as_deref(), Some(" "));
        assert_eq!(app.world().resource::<EditorRun>().playing, playing);
    }

    #[test]
    fn when_not_editing_space_toggles_play_and_backspace_deletes() {
        let mut app = app();
        select(&mut app, "m");
        let playing = app.world().resource::<EditorRun>().playing;
        press(&mut app, &[Char(' ')]);
        assert_eq!(app.world().resource::<EditorRun>().playing, !playing);
        press(&mut app, &[Backspace]);
        assert!(!has_node(&app, "m"));
        app.update();
        assert_eq!(panel(&app), None, "nothing selected, no panel");
    }

    #[test]
    fn escape_and_reselecting_abandon_an_edit() {
        let mut app = app();
        select(&mut app, "m");
        press(&mut app, &[Enter, Char('9'), Escape]);
        assert!(!inspector(&app).is_editing());
        assert_eq!(param(&app, "m", "channels").as_deref(), Some("3"));

        press(&mut app, &[Enter, Char('9')]);
        select(&mut app, "c");
        assert!(!inspector(&app).is_editing());
        assert_eq!(param(&app, "m", "channels").as_deref(), Some("3"));
    }

    #[test]
    fn applying_an_unchanged_value_queues_nothing() {
        use bevy::ecs::system::RunSystemOnce;
        let mut app = app();
        select(&mut app, "m");
        press(&mut app, &[Enter]);
        // Run only the router so the raw queue is observable.
        app.world_mut().resource_mut::<EditorKeys>().0 = vec![Enter];
        app.world_mut().run_system_once(route_keys).unwrap();
        assert!(app.world().resource::<EditorCommands>().queue.is_empty());
    }

    #[test]
    fn values_are_capped_in_length() {
        let mut app = app();
        select(&mut app, "m");
        let flood = vec![Char('9'); MAX_VALUE_LEN + 50];
        press(&mut app, &[&[Enter][..], &flood].concat());
        assert_eq!(
            inspector(&app).editing.map(|b| b.chars().count()),
            Some(MAX_VALUE_LEN)
        );
    }

    #[test]
    fn an_idle_inspector_is_not_marked_changed_every_frame() {
        let mut app = app();
        select(&mut app, "c");
        app.update();
        let tick = app.world().resource_ref::<ParamInspector>().last_changed();
        app.update();
        app.update();
        assert_eq!(
            app.world().resource_ref::<ParamInspector>().last_changed(),
            tick
        );
    }

    #[test]
    fn panel_text_marks_the_field_and_shows_the_edit_cursor() {
        let mut app = app();
        assert_eq!(panel(&app), None);
        select(&mut app, "m");
        press(&mut app, &[Enter]);
        assert_eq!(
            panel(&app).unwrap(),
            "m (merge)\n> channels = 3_\n[Enter] apply  [Esc] cancel"
        );
    }

    #[test]
    fn a_node_without_params_shows_a_note_and_ignores_enter() {
        let mut app = app();
        app.world_mut()
            .resource_mut::<EditorCommands>()
            .queue
            .push(add("bare", "not_a_real_kind", &[]));
        app.update();
        select(&mut app, "bare");
        press(&mut app, &[Enter, Tab]);
        assert!(!inspector(&app).is_editing());
        assert!(panel(&app).unwrap().ends_with("(no parameters)"));
    }
}
