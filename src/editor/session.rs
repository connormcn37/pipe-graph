//! A live editing session: graph + layout + a running pipeline, kept in sync.
//!
//! A [`Runtime`] is built from a snapshot of a graph; it cannot absorb a
//! topology or parameter change in place. An editor, though, wants to tweak a
//! parameter and see the result on the next frame. [`LiveSession`] bridges the
//! two: every edit that changes the graph re-instantiates the runtime, and the
//! things the *user* attached to the old runtime are carried across:
//!
//! - **External inputs** set with [`LiveSession::set_input`] are re-injected.
//! - **Taps** from [`LiveSession::add_tap`] are re-attached with
//!   [`Runtime::attach_tap`], so a [`Tap`] handle a UI already holds keeps
//!   receiving values — a preview panel does not need to know a rebuild
//!   happened.
//! - **Watches** from [`LiveSession::watch`] are re-applied.
//! - The iteration bound from [`LiveSession::set_max_iters`] is re-applied.
//!
//! Taps and watches are keyed by `(node, port)` and re-attached only where that
//! port still exists. A binding whose port disappeared (e.g. a parameter
//! change shrank a node's outputs) is kept dormant and comes back if the port
//! does; a binding whose *node* was removed is dropped.
//!
//! What is *not* carried across is node-internal state and in-flight edge
//! buffers: a rebuilt pipeline starts cold, exactly like a freshly
//! instantiated one (a feedback loop re-converges from scratch).
//!
//! # Errors never panic
//!
//! Editing passes through invalid states all the time — a node added before
//! its parameters are filled in, a half-typed value, an unknown kind. The
//! graph accepts such edits (it is the authoritative record of what the user
//! entered), and the session reports why it cannot run instead of failing:
//!
//! - If re-instantiation fails, the runtime is **dropped** (`runtime()` is
//!   `None`) and [`LiveSession::last_error`] holds the
//!   [`SessionError::Schedule`]. Keeping the previous runtime running would
//!   silently compute with parameters the graph no longer has. Tap handles
//!   keep their last published value, so previews freeze on the last good
//!   frame rather than going blank. The next edit that makes the graph valid
//!   rebuilds and clears the error.
//! - If a run fails, `last_error` holds the [`SessionError::Run`]; the runtime
//!   stays, and the next successful run clears the error.
//!
//! Commands the graph itself rejects (a duplicate id, an edge to a missing
//! node) leave the graph unchanged and are reported only through the returned
//! [`CommandOutcome`]; they do not touch `last_error`, which describes the
//! state of the pipeline, not the last keystroke.

use std::collections::{HashMap, HashSet};

use crate::data::Payload;
use crate::editor::{CommandOutcome, EditorCommand, Layout, apply_command_with_layout};
use crate::exec::{Registry, RunError, Runtime, ScheduleError, Tap};
use crate::graph::{Graph, NodeId, PortId};

/// Why a [`LiveSession`] is not producing output.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionError {
    /// The graph failed validation or a node failed to build; there is no
    /// runtime until an edit fixes it.
    Schedule(ScheduleError),
    /// The most recent run failed.
    Run(RunError),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Schedule(e) => write!(f, "cannot build pipeline: {e}"),
            SessionError::Run(e) => write!(f, "run failed: {e}"),
        }
    }
}

impl std::error::Error for SessionError {}

/// A tap the session re-attaches across rebuilds.
struct TapBinding {
    node: NodeId,
    port: PortId,
    tap: Tap,
}

/// An editable graph with a live [`Runtime`] that follows every edit.
///
/// See the module docs for what survives a rebuild and how errors are
/// reported.
pub struct LiveSession {
    graph: Graph,
    layout: Layout,
    registry: Registry,
    /// `None` exactly when the graph cannot be instantiated; `last_error` then
    /// holds the [`SessionError::Schedule`] explaining why.
    runtime: Option<Runtime>,
    last_error: Option<SessionError>,
    inputs: HashMap<NodeId, HashMap<PortId, Payload>>,
    taps: Vec<TapBinding>,
    watched: HashSet<(NodeId, PortId)>,
    max_iters: Option<u32>,
}

impl LiveSession {
    /// An empty session building nodes from `registry`.
    pub fn new(registry: Registry) -> Self {
        Self::with_graph(Graph::new(), registry)
    }

    /// A session over an existing graph, auto-laid-out and instantiated. A
    /// graph that does not build is accepted; check [`LiveSession::last_error`].
    pub fn with_graph(graph: Graph, registry: Registry) -> Self {
        let layout = Layout::auto(&graph);
        let mut session = Self {
            graph,
            layout,
            registry,
            runtime: None,
            last_error: None,
            inputs: HashMap::new(),
            taps: Vec::new(),
            watched: HashSet::new(),
            max_iters: None,
        };
        session.rebuild();
        session
    }

    /// The authoritative graph. Mutate it through [`LiveSession::apply`] so
    /// the runtime follows.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Direct layout access (e.g. for [`Layout::relayout`]). Layout never
    /// affects the pipeline, so this cannot desynchronize the runtime.
    pub fn layout_mut(&mut self) -> &mut Layout {
        &mut self.layout
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// The current runtime, or `None` while the graph does not build.
    pub fn runtime(&self) -> Option<&Runtime> {
        self.runtime.as_ref()
    }

    /// Mutable access to the current runtime. Anything attached through this
    /// handle directly (taps, watches, inputs) is **not** remembered and is lost
    /// on the next rebuild; use the session's own methods for that.
    pub fn runtime_mut(&mut self) -> Option<&mut Runtime> {
        self.runtime.as_mut()
    }

    /// Why the session is not (fully) working, if it isn't: a build/validation
    /// error (no runtime) or the last run's error.
    pub fn last_error(&self) -> Option<&SessionError> {
        self.last_error.as_ref()
    }

    /// Apply one editor command, rebuilding the runtime if it changed the
    /// graph's topology or parameters.
    ///
    /// Layout-only commands ([`EditorCommand::MoveNode`]), rejected commands,
    /// and a [`EditorCommand::SetParam`] that writes back the value already
    /// there do not rebuild, so they never reset a running pipeline.
    pub fn apply(&mut self, command: EditorCommand) -> CommandOutcome {
        let (outcome, changed) = self.apply_without_rebuild(command);
        if changed {
            self.rebuild();
        }
        outcome
    }

    /// Apply several commands, rebuilding at most once at the end. Use this
    /// for compound edits (paste, undo of a group) so the pipeline is not
    /// re-instantiated — and validated against half-finished states — after
    /// each step.
    pub fn apply_all(
        &mut self,
        commands: impl IntoIterator<Item = EditorCommand>,
    ) -> Vec<CommandOutcome> {
        let mut changed = false;
        let outcomes = commands
            .into_iter()
            .map(|command| {
                let (outcome, c) = self.apply_without_rebuild(command);
                changed |= c;
                outcome
            })
            .collect();
        if changed {
            self.rebuild();
        }
        outcomes
    }

    fn apply_without_rebuild(&mut self, command: EditorCommand) -> (CommandOutcome, bool) {
        // A SetParam that writes the current value back is a no-op for the
        // pipeline; the outcome alone cannot tell, so remember the new value.
        let new_value = match &command {
            EditorCommand::SetParam { value, .. } => Some(value.clone()),
            _ => None,
        };
        let outcome = apply_command_with_layout(&mut self.graph, &mut self.layout, command);
        let changed = match &outcome {
            CommandOutcome::ParamSet { previous, .. } => *previous != new_value,
            other => other.changes_graph(),
        };
        // Drop a removed node's bindings now, not at rebuild time: inside an
        // `apply_all` batch the id may be re-added (a "replace node" edit),
        // and the new node must not inherit the old one's inputs or previews.
        if let CommandOutcome::NodeRemoved(node) = &outcome {
            self.inputs.remove(node);
            self.taps.retain(|b| &b.node != node);
            self.watched.retain(|(n, _)| n != node);
        }
        (outcome, changed)
    }

    /// Inject a value on a node's input port. Remembered and re-injected after
    /// every rebuild until the node is removed.
    pub fn set_input(&mut self, node: &NodeId, port: &str, payload: Payload) {
        if let Some(rt) = self.runtime.as_mut() {
            rt.set_input(node, port, payload.clone());
        }
        self.inputs
            .entry(node.clone())
            .or_default()
            .insert(PortId(port.to_string()), payload);
    }

    /// Attach a live-preview tap to `(node, port)`. The returned handle keeps
    /// receiving values across rebuilds for as long as the node exists.
    ///
    /// A tap can be added while the graph does not build, or on a port that
    /// does not exist (yet); it starts publishing once the port does.
    pub fn add_tap(&mut self, node: &NodeId, port: &str) -> Tap {
        let tap = Tap::new();
        let binding = TapBinding {
            node: node.clone(),
            port: PortId(port.to_string()),
            tap: tap.clone(),
        };
        if self.port_exists(node, port)
            && let Some(rt) = self.runtime.as_mut()
        {
            rt.attach_tap(node, port, tap.clone());
        }
        self.taps.push(binding);
        tap
    }

    /// Detach every tap on `(node, port)`. Returns how many were removed.
    pub fn remove_taps(&mut self, node: &NodeId, port: &str) -> usize {
        let before = self.taps.len();
        self.taps.retain(|b| !(&b.node == node && b.port.0 == port));
        if let Some(rt) = self.runtime.as_mut() {
            rt.remove_taps(node, port);
        }
        before - self.taps.len()
    }

    /// Capture an intermediate port for [`LiveSession::output`]; survives
    /// rebuilds like a tap.
    pub fn watch(&mut self, node: &NodeId, port: &str) {
        if self.port_exists(node, port)
            && let Some(rt) = self.runtime.as_mut()
        {
            rt.watch(node, port);
        }
        self.watched
            .insert((node.clone(), PortId(port.to_string())));
    }

    /// Undo a [`LiveSession::watch`].
    pub fn unwatch(&mut self, node: &NodeId, port: &str) {
        self.watched
            .remove(&(node.clone(), PortId(port.to_string())));
        if let Some(rt) = self.runtime.as_mut() {
            rt.unwatch(node, port);
        }
    }

    /// Bound on iterations per cyclic component; survives rebuilds.
    pub fn set_max_iters(&mut self, n: u32) {
        self.max_iters = Some(n);
        if let Some(rt) = self.runtime.as_mut() {
            rt.set_max_iters(n);
        }
    }

    /// A node's last output on `port` (see [`Runtime::output`]); `None` while
    /// there is no runtime.
    pub fn output(&self, node: &NodeId, port: &str) -> Option<&Payload> {
        self.runtime.as_ref()?.output(node, port)
    }

    /// Run the pipeline once.
    ///
    /// Returns the build error if there is no runtime. A run failure is also
    /// stored in [`LiveSession::last_error`]; a successful run clears a stored
    /// run failure.
    pub fn run_once(&mut self) -> Result<(), SessionError> {
        self.tick(1)
    }

    /// Run the pipeline `n` times; see [`LiveSession::run_once`].
    pub fn tick(&mut self, n: u32) -> Result<(), SessionError> {
        let Some(rt) = self.runtime.as_mut() else {
            // Invariant: no runtime means the last rebuild failed and recorded why.
            return Err(self
                .last_error
                .clone()
                .expect("a session without a runtime always records a schedule error"));
        };
        match rt.tick(n) {
            Ok(()) => {
                if matches!(self.last_error, Some(SessionError::Run(_))) {
                    self.last_error = None;
                }
                Ok(())
            }
            Err(e) => {
                let err = SessionError::Run(e);
                self.last_error = Some(err.clone());
                Err(err)
            }
        }
    }

    /// Re-instantiate the runtime from the current graph and carry the user's
    /// bindings across. See the module docs for the policy.
    fn rebuild(&mut self) {
        // Forget bindings for nodes not in the graph (removed nodes were
        // already purged in `apply_without_rebuild`; this also catches
        // bindings made for ids that never existed).
        let graph = &self.graph;
        self.inputs.retain(|node, _| graph.nodes.contains_key(node));
        self.taps.retain(|b| graph.nodes.contains_key(&b.node));
        self.watched
            .retain(|(node, _)| graph.nodes.contains_key(node));

        match Runtime::instantiate(&self.graph, &self.registry) {
            Ok(mut rt) => {
                if let Some(n) = self.max_iters {
                    rt.set_max_iters(n);
                }
                for (node, ports) in &self.inputs {
                    for (port, payload) in ports {
                        rt.set_input(node, &port.0, payload.clone());
                    }
                }
                for b in &self.taps {
                    if self.port_exists(&b.node, &b.port.0) {
                        rt.attach_tap(&b.node, &b.port.0, b.tap.clone());
                    }
                }
                for (node, port) in &self.watched {
                    if self.port_exists(node, &port.0) {
                        rt.watch(node, &port.0);
                    }
                }
                self.runtime = Some(rt);
                self.last_error = None;
            }
            Err(e) => {
                self.runtime = None;
                self.last_error = Some(SessionError::Schedule(e));
            }
        }
    }

    /// Whether `node` currently declares an output `port`.
    fn port_exists(&self, node: &NodeId, port: &str) -> bool {
        self.graph
            .nodes
            .get(node)
            .and_then(|spec| self.registry.ports_of(spec).ok())
            .is_some_and(|ports| ports.find_output(port).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Frame;
    use crate::exec::builtin_registry;
    use crate::graph::{NodeSpec, Params};

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

    fn white() -> Payload {
        Payload::Frame(Frame::from_rgb8(1, 1, vec![(255, 255, 255)]))
    }

    fn rgb(session: &LiveSession, n: &str) -> Vec<(u8, u8, u8)> {
        session
            .output(&id(n), "out")
            .and_then(Payload::as_frame)
            .expect("frame output")
            .to_rgb8()
    }

    fn set_channel(n: &str, ch: &str) -> EditorCommand {
        EditorCommand::SetParam {
            node: id(n),
            key: "channel".to_string(),
            value: ch.to_string(),
        }
    }

    #[test]
    fn empty_session_builds_and_runs() {
        let mut s = LiveSession::new(builtin_registry());
        assert!(s.runtime().is_some());
        assert!(s.last_error().is_none());
        s.run_once().unwrap();
    }

    #[test]
    fn param_change_rebuilds_and_keeps_inputs() {
        let mut s = LiveSession::new(builtin_registry());
        s.apply(EditorCommand::AddNode(node(
            "a",
            "clear_channel",
            &[("channel", "red")],
        )));
        s.set_input(&id("a"), "in", white());
        s.run_once().unwrap();
        assert_eq!(rgb(&s, "a"), vec![(0, 255, 255)]);

        s.apply(set_channel("a", "blue"));
        assert!(s.last_error().is_none());
        // Output is cleared by the rebuild; the input was carried across.
        assert!(s.output(&id("a"), "out").is_none());
        s.run_once().unwrap();
        assert_eq!(rgb(&s, "a"), vec![(255, 255, 0)]);
    }

    #[test]
    fn layout_only_and_noop_edits_do_not_rebuild() {
        let mut s = LiveSession::new(builtin_registry());
        s.apply(EditorCommand::AddNode(node(
            "a",
            "clear_channel",
            &[("channel", "red")],
        )));
        s.set_input(&id("a"), "in", white());
        s.run_once().unwrap();

        // A rebuild would clear the captured output; these must not.
        let moved = s.apply(EditorCommand::MoveNode {
            node: id("a"),
            to: (3.0, 4.0),
        });
        assert!(matches!(moved, CommandOutcome::NodeMoved { .. }));
        assert_eq!(s.layout().position(&id("a")), Some((3.0, 4.0)));
        s.apply(set_channel("a", "red"));
        s.apply(EditorCommand::RemoveNode(id("ghost")));
        assert!(s.output(&id("a"), "out").is_some());
    }

    #[test]
    fn bad_param_drops_runtime_and_reports_then_recovers() {
        let mut s = LiveSession::new(builtin_registry());
        s.apply(EditorCommand::AddNode(node(
            "a",
            "clear_channel",
            &[("channel", "red")],
        )));
        s.set_input(&id("a"), "in", white());
        let tap = s.add_tap(&id("a"), "out");
        s.run_once().unwrap();
        let seq = tap.seq();

        s.apply(set_channel("a", "purple"));
        assert!(s.runtime().is_none());
        assert!(matches!(s.last_error(), Some(SessionError::Schedule(_))));
        assert!(matches!(s.run_once(), Err(SessionError::Schedule(_))));
        // The graph holds what the user typed.
        assert_eq!(s.graph().nodes[&id("a")].params["channel"], "purple");
        // The preview froze on the last good frame.
        assert_eq!(tap.seq(), seq);
        assert!(tap.latest().is_some());

        s.apply(set_channel("a", "green"));
        assert!(s.last_error().is_none());
        s.run_once().unwrap();
        assert!(tap.seq() > seq);
        let latest = tap.latest().unwrap();
        assert_eq!(latest.as_frame().unwrap().to_rgb8(), vec![(255, 0, 255)]);
    }

    #[test]
    fn taps_and_watches_follow_rebuilds_and_die_with_their_node() {
        let mut s = LiveSession::new(builtin_registry());
        s.apply_all([
            EditorCommand::AddNode(node("a", "clear_channel", &[("channel", "red")])),
            EditorCommand::AddNode(node("b", "clear_channel", &[("channel", "green")])),
            EditorCommand::Connect {
                from: (id("a"), PortId("out".to_string())),
                to: (id("b"), PortId("in".to_string())),
            },
        ]);
        s.set_input(&id("a"), "in", white());
        s.watch(&id("a"), "out");
        let tap = s.add_tap(&id("a"), "out");

        s.apply(set_channel("b", "blue"));
        s.run_once().unwrap();
        // The intermediate is captured (watch survived) and the tap published.
        assert_eq!(rgb(&s, "a"), vec![(0, 255, 255)]);
        assert_eq!(tap.seq(), 1);

        s.unwatch(&id("a"), "out");
        assert_eq!(s.remove_taps(&id("a"), "out"), 1);
        s.run_once().unwrap();
        assert_eq!(tap.seq(), 1, "a removed tap is no longer published to");

        let tap = s.add_tap(&id("a"), "out");
        s.apply(EditorCommand::RemoveNode(id("a")));
        assert!(s.taps.is_empty());
        assert!(s.inputs.is_empty());
        assert_eq!(tap.seq(), 0);
    }

    #[test]
    fn replacing_a_node_in_one_batch_drops_its_bindings() {
        let mut s = LiveSession::new(builtin_registry());
        s.apply(EditorCommand::AddNode(node(
            "a",
            "clear_channel",
            &[("channel", "red")],
        )));
        s.set_input(&id("a"), "in", white());
        s.watch(&id("a"), "out");
        let old_tap = s.add_tap(&id("a"), "out");

        s.apply_all([
            EditorCommand::RemoveNode(id("a")),
            EditorCommand::AddNode(node("a", "clear_channel", &[("channel", "blue")])),
        ]);
        assert!(s.inputs.is_empty());
        assert!(s.taps.is_empty());
        assert!(s.watched.is_empty());
        // The replacement has no input, so running fails rather than silently
        // processing the old node's frame; the old tap stays silent.
        assert!(s.run_once().is_err());
        assert_eq!(old_tap.seq(), 0);
    }

    #[test]
    fn tap_on_unknown_port_stays_dormant() {
        let mut s = LiveSession::new(builtin_registry());
        s.apply(EditorCommand::AddNode(node(
            "a",
            "clear_channel",
            &[("channel", "red")],
        )));
        s.set_input(&id("a"), "in", white());
        let tap = s.add_tap(&id("a"), "nope");
        s.run_once().unwrap();
        assert_eq!(tap.seq(), 0);
        // Still remembered, so it would attach if the port appeared.
        assert_eq!(s.taps.len(), 1);
    }

    #[test]
    fn run_errors_are_recorded_and_cleared() {
        // A crop larger than its input fails at run time, not build time.
        let mut s = LiveSession::new(builtin_registry());
        s.apply(EditorCommand::AddNode(node(
            "c",
            "crop",
            &[("x", "0"), ("y", "0"), ("w", "2"), ("h", "2")],
        )));
        s.set_input(&id("c"), "in", white());
        assert!(matches!(s.run_once(), Err(SessionError::Run(_))));
        assert!(matches!(s.last_error(), Some(SessionError::Run(_))));
        assert!(s.runtime().is_some(), "a run error keeps the runtime");

        s.apply_all([
            EditorCommand::SetParam {
                node: id("c"),
                key: "w".to_string(),
                value: "1".to_string(),
            },
            EditorCommand::SetParam {
                node: id("c"),
                key: "h".to_string(),
                value: "1".to_string(),
            },
        ]);
        s.run_once().unwrap();
        assert!(s.last_error().is_none());
    }

    #[test]
    fn with_graph_lays_out_existing_nodes() {
        let mut g = Graph::new();
        g.add_node(node("a", "clear_channel", &[("channel", "red")]))
            .unwrap();
        let s = LiveSession::with_graph(g, builtin_registry());
        assert_eq!(s.layout().position(&id("a")), Some((0.0, 0.0)));
        assert!(s.runtime().is_some());
    }
}
