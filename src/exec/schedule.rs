//! Scheduler: compile a `Graph` into an execution `Plan` and run it.
//!
//! Compilation follows the design notes in `lib.rs`:
//! 1. Build a node-level dependency graph from the edges.
//! 2. Decompose it into strongly-connected components (Tarjan).
//! 3. Order the condensation topologically (sources first).
//! 4. A single-node SCC with no self-loop is an [`Component::Acyclic`] step,
//!    run once. Any larger SCC (or a self-loop) is a [`Component::Cyclic`]
//!    step, run with a bounded tick loop.
//!
//! Data lives in per-edge [`EdgeBuffer`]s. A node reads its inputs from the
//! buffers of incoming edges (plus any externally-injected inputs) and writes
//! outputs to the buffers of outgoing edges. Cyclic components use a
//! Jacobi-style update — every node in the component is evaluated against the
//! previous iteration's buffers, then all results are committed together — so a
//! feedback edge naturally reads the previous tick's value.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::data::Payload;
use crate::exec::{
    BuildError, EdgeBuffer, Inputs, Node, Outputs, Registry, Tap, ValidationError, validate,
};
use crate::graph::{EdgeId, Graph, NodeId, PortId};

/// Default bound on iterations for a cyclic component within one `run_once`.
pub const DEFAULT_MAX_ITERS: u32 = 16;

/// One execution step in a compiled [`Plan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Component {
    /// A single node with no feedback: evaluate once.
    Acyclic(NodeId),
    /// A strongly-connected component (or self-loop): iterate to settle.
    Cyclic { nodes: Vec<NodeId> },
}

/// A compiled execution order for a graph, plus the structure it was derived
/// from.
///
/// [`Plan::components`] is the flat order a single-threaded run follows. The
/// *condensation* — the DAG of dependencies between those components — is kept
/// alongside it rather than discarded, because the flat order answers only "is
/// this order legal?" while the DAG answers the questions optimizations need:
///
/// - **What can run concurrently?** Components with no path between them are
///   independent. A diamond (split → two branches → merge) looks like a straight
///   line once flattened; in the DAG the branches are visibly parallel.
/// - **What can be fused?** A run of components chained one-to-one
///   ([`Plan::linear_chains`]) produces intermediates that nothing else reads,
///   so a future executor could evaluate the run as a unit and skip
///   materializing the values in between.
///
/// Note the graph as a whole may be cyclic; it is the *condensation* that is
/// always acyclic, since each cycle is collapsed into one [`Component::Cyclic`].
#[derive(Debug, Clone)]
pub struct Plan {
    components: Vec<Component>,
    /// Condensation edges as `(producer, consumer)` indices into `components`.
    /// Sorted and deduplicated.
    deps: Vec<(usize, usize)>,
    /// Which component each node was placed in.
    node_component: HashMap<NodeId, usize>,
}

impl Plan {
    /// The components in the order a serial run executes them (sources first).
    pub fn components(&self) -> &[Component] {
        &self.components
    }

    /// Dependency edges between components, as indices into [`Plan::components`].
    pub fn deps(&self) -> &[(usize, usize)] {
        &self.deps
    }

    /// Which component a node belongs to.
    pub fn component_of(&self, node: &NodeId) -> Option<usize> {
        self.node_component.get(node).copied()
    }

    /// Components that consume this one's output.
    pub fn successors(&self, component: usize) -> Vec<usize> {
        self.deps
            .iter()
            .filter(|(from, _)| *from == component)
            .map(|(_, to)| *to)
            .collect()
    }

    /// Components this one consumes from.
    pub fn predecessors(&self, component: usize) -> Vec<usize> {
        self.deps
            .iter()
            .filter(|(_, to)| *to == component)
            .map(|(from, _)| *from)
            .collect()
    }

    /// Maximal runs of components chained strictly one-to-one: every component
    /// in a run feeds exactly one successor, which in turn is fed by only that
    /// component. Returned as indices into [`Plan::components`]; runs shorter
    /// than two components are omitted, and the runs are disjoint.
    ///
    /// These are *structural* fusion candidates. Because the values passed
    /// along a run have exactly one producer and one consumer, an executor
    /// could evaluate the whole run as a unit without materializing the
    /// intermediates — fewer buffers, less latency.
    ///
    /// Two caveats before acting on this:
    ///
    /// - Only [`Component::Acyclic`] components participate. A cyclic component
    ///   iterates internally to settle, so it is a barrier at both its edges.
    /// - **A `Plan` does not know what is observed.** Anything watching a value
    ///   inside a run is a fusion barrier, and observation is a property of the
    ///   [`Runtime`], not the topology. Use [`Runtime::fusable_runs`] for the
    ///   runs that survive what is actually being watched; this method reports
    ///   the structure they are cut from.
    pub fn linear_chains(&self) -> Vec<Vec<usize>> {
        let n = self.components.len();
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(from, to) in &self.deps {
            succ[from].push(to);
            pred[to].push(from);
        }

        let acyclic = |i: usize| matches!(self.components[i], Component::Acyclic(_));

        // The component `i` feeds, if that link is private to the two of them.
        let next_in_chain = |i: usize| -> Option<usize> {
            if !acyclic(i) || succ[i].len() != 1 {
                return None;
            }
            let j = succ[i][0];
            (pred[j].len() == 1 && acyclic(j)).then_some(j)
        };

        let mut chains = Vec::new();
        for (start, feeding) in pred.iter().enumerate() {
            if !acyclic(start) {
                continue;
            }
            // Skip components that continue a chain begun earlier.
            if feeding.len() == 1 && next_in_chain(feeding[0]) == Some(start) {
                continue;
            }
            let mut chain = vec![start];
            while let Some(next) = next_in_chain(*chain.last().expect("non-empty")) {
                chain.push(next);
            }
            if chain.len() > 1 {
                chains.push(chain);
            }
        }
        chains
    }
}

/// Errors from turning a graph into a runnable [`Runtime`].
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleError {
    Validation(ValidationError),
    Build { node: NodeId, error: BuildError },
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScheduleError::Validation(e) => write!(f, "validation: {e}"),
            ScheduleError::Build { node, error } => write!(f, "building '{}': {error}", node.0),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// A node failed while the graph was running.
#[derive(Debug, Clone, PartialEq)]
pub struct RunError {
    pub node: NodeId,
    pub error: crate::exec::NodeError,
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node '{}' failed: {}", self.node.0, self.error)
    }
}

impl std::error::Error for RunError {}

/// Compile the topology of `graph` into an ordered [`Plan`].
///
/// This is pure topology — it needs no registry and does not validate ports.
/// [`Runtime::instantiate`] validates first, so by the time nodes run the graph
/// is well-formed; edges with unknown endpoints are ignored here defensively.
pub fn compile(graph: &Graph) -> Plan {
    // Stable node indexing (sorted for deterministic output).
    let mut ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
    ids.sort_by(|a, b| a.0.cmp(&b.0));
    let index: HashMap<&NodeId, usize> = ids.iter().enumerate().map(|(i, id)| (id, i)).collect();
    let n = ids.len();

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut has_self_loop = vec![false; n];
    for conn in graph.edges.values() {
        let (Some(&u), Some(&v)) = (index.get(&conn.from.0), index.get(&conn.to.0)) else {
            continue;
        };
        if u == v {
            has_self_loop[u] = true;
        }
        adj[u].push(v);
    }
    for a in &mut adj {
        a.sort_unstable();
        a.dedup();
    }

    // Tarjan yields SCCs sinks-first (reverse topological); reverse for sources-first.
    let sccs = tarjan_scc(n, &adj);

    // Component index of every node, in the same sources-first order.
    let mut comp_of = vec![usize::MAX; n];
    for (ci, comp) in sccs.iter().rev().enumerate() {
        for &node in comp {
            comp_of[node] = ci;
        }
    }

    let mut components = Vec::with_capacity(sccs.len());
    let mut node_component: HashMap<NodeId, usize> = HashMap::with_capacity(n);
    for (ci, comp) in sccs.iter().rev().enumerate() {
        for &node in comp {
            node_component.insert(ids[node].clone(), ci);
        }
        if comp.len() == 1 && !has_self_loop[comp[0]] {
            components.push(Component::Acyclic(ids[comp[0]].clone()));
        } else {
            let mut nodes: Vec<NodeId> = comp.iter().map(|&i| ids[i].clone()).collect();
            nodes.sort_by(|a, b| a.0.cmp(&b.0));
            components.push(Component::Cyclic { nodes });
        }
    }

    // Condensation: keep the node-level edges that cross component boundaries.
    // Edges inside a component are the cycle itself and carry no ordering.
    let mut deps: Vec<(usize, usize)> = Vec::new();
    for (u, targets) in adj.iter().enumerate() {
        for &v in targets {
            if comp_of[u] != comp_of[v] {
                deps.push((comp_of[u], comp_of[v]));
            }
        }
    }
    deps.sort_unstable();
    deps.dedup();

    Plan {
        components,
        deps,
        node_component,
    }
}

/// Iterative Tarjan's strongly-connected-components. Returns components in the
/// order they finish (sinks first / reverse topological).
fn tarjan_scc(n: usize, adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    const UNVISITED: usize = usize::MAX;

    let mut indices = vec![UNVISITED; n];
    let mut lowlink = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut sccs: Vec<Vec<usize>> = Vec::new();
    let mut counter = 0usize;

    for start in 0..n {
        if indices[start] != UNVISITED {
            continue;
        }
        // DFS frames: (node, next-neighbor-cursor).
        let mut call_stack: Vec<(usize, usize)> = vec![(start, 0)];
        while let Some(&(v, cursor)) = call_stack.last() {
            if cursor == 0 {
                indices[v] = counter;
                lowlink[v] = counter;
                counter += 1;
                stack.push(v);
                on_stack[v] = true;
            }

            if cursor < adj[v].len() {
                call_stack.last_mut().unwrap().1 += 1;
                let w = adj[v][cursor];
                if indices[w] == UNVISITED {
                    call_stack.push((w, 0));
                } else if on_stack[w] {
                    lowlink[v] = lowlink[v].min(indices[w]);
                }
            } else {
                // Finished exploring v.
                if lowlink[v] == indices[v] {
                    let mut comp = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        comp.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(comp);
                }
                call_stack.pop();
                if let Some(&(parent, _)) = call_stack.last() {
                    lowlink[parent] = lowlink[parent].min(lowlink[v]);
                }
            }
        }
    }

    sccs
}

/// One edge and its data slot.
struct Edge {
    from: (NodeId, PortId),
    to: (NodeId, PortId),
    buffer: EdgeBuffer,
}

/// An instantiated, runnable graph.
pub struct Runtime {
    plan: Plan,
    nodes: HashMap<NodeId, Box<dyn Node>>,
    edges: Vec<Edge>,
    /// edge indices whose destination is this node.
    incoming: HashMap<NodeId, Vec<usize>>,
    /// edge indices whose source is this node.
    outgoing: HashMap<NodeId, Vec<usize>>,
    /// Externally injected inputs (graph "sources"): node → port → value.
    external: HashMap<NodeId, HashMap<PortId, Arc<Payload>>>,
    /// Last outputs produced by each node (an always-on tap; read via `output`).
    node_outputs: HashMap<NodeId, HashMap<PortId, Arc<Payload>>>,
    /// Live-preview taps registered on (node, output-port) pairs.
    taps: HashMap<NodeId, HashMap<PortId, Vec<Tap>>>,
    /// Ports a caller asked to capture, via [`Runtime::watch`] or
    /// [`Runtime::add_tap`].
    watched: HashMap<NodeId, HashSet<PortId>>,
    /// Ports captured for free: their value never leaves its own component, so
    /// it is materialized either way. Computed once at instantiation.
    terminal: HashMap<NodeId, HashSet<PortId>>,
    /// Capture every port regardless of the above (debugging escape hatch).
    capture_all: bool,
    /// Runs of components that could be fused given what is currently
    /// observed. Derived from the plan and the observed set; see
    /// [`Runtime::fusable_runs`].
    fusable: Vec<Vec<usize>>,
    max_iters: u32,
}

impl Runtime {
    /// Validate, compile, and instantiate `graph` against `reg`.
    pub fn instantiate(graph: &Graph, reg: &Registry) -> Result<Self, ScheduleError> {
        validate(graph, reg).map_err(ScheduleError::Validation)?;
        let plan = compile(graph);

        let mut nodes = HashMap::with_capacity(graph.nodes.len());
        for (id, spec) in &graph.nodes {
            let node = reg.build(spec).map_err(|error| ScheduleError::Build {
                node: id.clone(),
                error,
            })?;
            nodes.insert(id.clone(), node);
        }

        // Deterministic edge order by EdgeId.
        let mut edge_ids: Vec<&EdgeId> = graph.edges.keys().collect();
        edge_ids.sort_by_key(|e| e.0);

        let mut edges = Vec::with_capacity(edge_ids.len());
        let mut incoming: HashMap<NodeId, Vec<usize>> = HashMap::new();
        let mut outgoing: HashMap<NodeId, Vec<usize>> = HashMap::new();
        for eid in edge_ids {
            let conn = &graph.edges[eid];
            let i = edges.len();
            edges.push(Edge {
                from: conn.from.clone(),
                to: conn.to.clone(),
                buffer: EdgeBuffer::new(),
            });
            outgoing.entry(conn.from.0.clone()).or_default().push(i);
            incoming.entry(conn.to.0.clone()).or_default().push(i);
        }

        let terminal = terminal_ports(graph, &plan, &nodes);

        let mut runtime = Self {
            plan,
            nodes,
            edges,
            incoming,
            outgoing,
            external: HashMap::new(),
            node_outputs: HashMap::new(),
            taps: HashMap::new(),
            watched: HashMap::new(),
            terminal,
            capture_all: false,
            fusable: Vec::new(),
            max_iters: DEFAULT_MAX_ITERS,
        };
        runtime.recompute_fusable();
        Ok(runtime)
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Bound on iterations per cyclic component within one `run_once`.
    pub fn set_max_iters(&mut self, n: u32) {
        self.max_iters = n;
    }

    /// Inject a value on a node's input port (feeds "source" nodes whose input
    /// has no incoming edge). Persists across runs until overwritten.
    pub fn set_input(&mut self, node: &NodeId, port: &str, payload: Payload) {
        self.external
            .entry(node.clone())
            .or_default()
            .insert(PortId(port.to_string()), Arc::new(payload));
    }

    /// The last value a node produced on `port`, if it is being captured.
    ///
    /// Capture is opt-in for intermediates. A port whose value leaves its
    /// component is dropped once the consuming node has read it, so this
    /// returns `None` for one unless [`Runtime::watch`] or
    /// [`Runtime::add_tap`] asked for it — check with
    /// [`Runtime::is_observed`] if a `None` is surprising, or turn on
    /// [`Runtime::set_capture_all`] while debugging.
    ///
    /// A pipeline's results need no opt-in: a port whose value never leaves
    /// its component is captured from the start, which covers both a leaf
    /// node's output and anything circulating inside a feedback loop.
    pub fn output(&self, node: &NodeId, port: &str) -> Option<&Payload> {
        self.node_outputs.get(node)?.get(port).map(Arc::as_ref)
    }

    /// A cheap shared handle to a node's last output on `port`. Subject to the
    /// same opt-in capture rule as [`Runtime::output`].
    pub fn output_arc(&self, node: &NodeId, port: &str) -> Option<Arc<Payload>> {
        self.node_outputs.get(node)?.get(port).cloned()
    }

    /// Attach a non-blocking live-preview tap to a node's output `port`.
    ///
    /// Returns a cloneable [`Tap`] handle that always reflects the latest value
    /// produced on that port. Multiple taps may observe the same port.
    ///
    /// A live tap makes the port observed on its own, independently of
    /// [`Runtime::watch`]: [`Runtime::output`] can read it, and a fusing
    /// executor knows the value has to exist. Attaching a tap never rebuilds
    /// nodes or edge buffers, so it is safe mid-stream: see
    /// [`Runtime::fusable_runs`].
    pub fn add_tap(&mut self, node: &NodeId, port: &str) -> Tap {
        let tap = Tap::new();
        self.taps
            .entry(node.clone())
            .or_default()
            .entry(PortId(port.to_string()))
            .or_default()
            .push(tap.clone());
        self.recompute_fusable();
        tap
    }

    /// Detach every tap on `port`, and drop the value captured for it if
    /// nothing else is observing that port.
    ///
    /// Returns how many taps were removed. Their handles keep working as
    /// values; they simply stop being published to.
    pub fn remove_taps(&mut self, node: &NodeId, port: &str) -> usize {
        let removed = self
            .taps
            .get_mut(node)
            .and_then(|ports| ports.remove(port))
            .map_or(0, |taps| taps.len());
        if self.taps.get(node).is_some_and(HashMap::is_empty) {
            self.taps.remove(node);
        }
        if removed > 0 {
            if !self.is_observed(node, port)
                && let Some(captured) = self.node_outputs.get_mut(node)
            {
                captured.remove(port);
            }
            self.recompute_fusable();
        }
        removed
    }

    /// Capture the value produced on `port` so [`Runtime::output`] can read it.
    ///
    /// Ports whose value never leaves its component are captured already (a
    /// pipeline's results, and anything inside a feedback loop); this is for
    /// probing an *intermediate*, which is otherwise dropped as soon as the
    /// consuming node has read it.
    pub fn watch(&mut self, node: &NodeId, port: &str) {
        self.watched
            .entry(node.clone())
            .or_default()
            .insert(PortId(port.to_string()));
        self.recompute_fusable();
    }

    /// Undo a [`Runtime::watch`], dropping the value already captured there so
    /// no stale frame is served from a port that will no longer be refreshed.
    ///
    /// Has no effect on a port that is observed for another reason — a live
    /// tap, or a value that never leaves its component. Use
    /// [`Runtime::remove_taps`] to detach taps.
    pub fn unwatch(&mut self, node: &NodeId, port: &str) {
        if let Some(ports) = self.watched.get_mut(node) {
            ports.remove(port);
            if ports.is_empty() {
                self.watched.remove(node);
            }
        }
        if !self.is_observed(node, port)
            && let Some(captured) = self.node_outputs.get_mut(node)
        {
            captured.remove(port);
        }
        self.recompute_fusable();
    }

    /// Capture every port, restoring the unconditional behaviour this runtime
    /// had before capture became opt-in. Convenient while debugging a graph;
    /// it pins one payload per port and blocks every fusion opportunity, so it
    /// is not the setting to ship.
    pub fn set_capture_all(&mut self, enabled: bool) {
        self.capture_all = enabled;
        self.recompute_fusable();
    }

    /// Whether `port`'s value is captured for [`Runtime::output`].
    pub fn is_observed(&self, node: &NodeId, port: &str) -> bool {
        self.capture_all
            || self.terminal.get(node).is_some_and(|p| p.contains(port))
            || self.watched.get(node).is_some_and(|p| p.contains(port))
            || self.taps.get(node).is_some_and(|p| p.contains_key(port))
    }

    /// Runs of components that could be evaluated as a unit, given what is
    /// currently observed. Indices into [`Plan::components`].
    ///
    /// This is [`Plan::linear_chains`] cut at every point something is
    /// watching. A chain `a -> b -> c -> d` with `b`'s output tapped yields
    /// `[a, b]` and `[c, d]`: tapping forces `b`'s value to exist, so the run
    /// splits there and nowhere else — you pay one materialization for the
    /// frame you asked to see.
    ///
    /// Recomputed whenever the observed set changes, from the plan and the
    /// observed set alone. Nodes and edge buffers are never rebuilt, so
    /// attaching a preview to a running pipeline cannot reset a node's internal
    /// state or a feedback loop mid-convergence. Changing the graph's
    /// *topology* is a different matter and still needs a fresh
    /// [`Runtime::instantiate`].
    pub fn fusable_runs(&self) -> &[Vec<usize>] {
        &self.fusable
    }

    fn observes(&self, node: &NodeId, port: &PortId) -> bool {
        self.capture_all
            || self.terminal.get(node).is_some_and(|p| p.contains(port))
            || self.watched.get(node).is_some_and(|p| p.contains(port))
            || self.taps.get(node).is_some_and(|p| p.contains_key(port))
    }

    /// True when nothing observes any value passed from component `from` to
    /// component `to`, so the two could be evaluated without the intermediate
    /// ever existing. Checks the specific ports carrying the link, not the
    /// nodes as a whole — a split whose other outputs are watched can still
    /// fuse along the branch nobody is looking at.
    fn link_is_private(&self, from: usize, to: usize) -> bool {
        !self.edges.iter().any(|edge| {
            self.plan.component_of(&edge.from.0) == Some(from)
                && self.plan.component_of(&edge.to.0) == Some(to)
                && self.observes(&edge.from.0, &edge.from.1)
        })
    }

    /// Cut one structural chain into the runs that survive observation.
    fn split_chain(&self, chain: &[usize]) -> Vec<Vec<usize>> {
        let mut runs: Vec<Vec<usize>> = Vec::new();
        let mut current: Vec<usize> = vec![chain[0]];
        for link in chain.windows(2) {
            if !self.link_is_private(link[0], link[1]) {
                // A run of one component is not a fusion; drop it.
                if current.len() > 1 {
                    runs.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
            current.push(link[1]);
        }
        if current.len() > 1 {
            runs.push(current);
        }
        runs
    }

    fn recompute_fusable(&mut self) {
        self.fusable = self
            .plan
            .linear_chains()
            .iter()
            .flat_map(|chain| self.split_chain(chain))
            .collect();
    }

    /// Clear all edge buffers and captured outputs, and reset node state.
    pub fn reset(&mut self) {
        for e in &mut self.edges {
            e.buffer.clear();
        }
        self.node_outputs.clear();
        for ports in self.taps.values() {
            for taps in ports.values() {
                for tap in taps {
                    tap.clear();
                }
            }
        }
        for node in self.nodes.values_mut() {
            node.reset();
        }
    }

    /// Execute the whole plan once (acyclic steps once; cyclic steps iterate).
    pub fn run_once(&mut self) -> Result<(), RunError> {
        for ci in 0..self.plan.components.len() {
            match self.plan.components[ci].clone() {
                Component::Acyclic(id) => self.eval_node(&id)?,
                Component::Cyclic { nodes } => self.run_cyclic(&nodes)?,
            }
        }
        Ok(())
    }

    /// Run the plan `n` times (e.g. advancing a stream by n frames).
    pub fn tick(&mut self, n: u32) -> Result<(), RunError> {
        for _ in 0..n {
            self.run_once()?;
        }
        Ok(())
    }

    fn eval_node(&mut self, id: &NodeId) -> Result<(), RunError> {
        let inputs = self.gather(id);
        let mut outputs = Outputs::new();
        {
            let node = self.nodes.get_mut(id).expect("compiled node exists");
            node.eval(&inputs, &mut outputs).map_err(|error| RunError {
                node: id.clone(),
                error,
            })?;
        }
        self.commit(id, outputs);
        Ok(())
    }

    fn run_cyclic(&mut self, nodes: &[NodeId]) -> Result<(), RunError> {
        for _ in 0..self.max_iters {
            // Gather+eval every node against the *current* buffers first...
            let mut pending: Vec<(NodeId, Outputs)> = Vec::with_capacity(nodes.len());
            for id in nodes {
                let inputs = self.gather(id);
                let mut outputs = Outputs::new();
                let node = self.nodes.get_mut(id).expect("compiled node exists");
                node.eval(&inputs, &mut outputs).map_err(|error| RunError {
                    node: id.clone(),
                    error,
                })?;
                pending.push((id.clone(), outputs));
            }
            // ...then commit them together (previous-tick feedback semantics).
            for (id, outputs) in pending {
                self.commit(&id, outputs);
            }
        }
        Ok(())
    }

    fn gather(&self, id: &NodeId) -> Inputs {
        let mut values: HashMap<PortId, Payload> = HashMap::new();
        // External inputs first; edge values override where both exist.
        if let Some(ext) = self.external.get(id) {
            for (port, value) in ext {
                values.insert(port.clone(), value.as_ref().clone());
            }
        }
        if let Some(edge_ids) = self.incoming.get(id) {
            for &i in edge_ids {
                if let Some(payload) = self.edges[i].buffer.get_last() {
                    values.insert(self.edges[i].to.1.clone(), payload.clone());
                }
            }
        }
        Inputs::new(values)
    }

    fn commit(&mut self, id: &NodeId, outputs: Outputs) {
        // Each output payload is already shared: one Arc spans the port's whole
        // fan-out, its taps, and `node_outputs`. Taking it as-is (rather than
        // re-wrapping) preserves pointer identity across evaluations, so a node
        // that republished a cached buffer via `Outputs::set_shared` stays
        // detectable downstream with `Arc::ptr_eq`.
        let arced: HashMap<PortId, Arc<Payload>> = outputs.into_map();

        let out_edges = self.outgoing.get(id).cloned().unwrap_or_default();
        for i in out_edges {
            let port = self.edges[i].from.1.clone();
            if let Some(shared) = arced.get(&port) {
                self.edges[i].buffer.push_arc(shared.clone());
            }
        }

        // Publish to any live-preview taps (non-blocking: a quick lock+store).
        if let Some(node_taps) = self.taps.get(id) {
            for (port, shared) in &arced {
                if let Some(taps) = node_taps.get(port) {
                    for tap in taps {
                        tap.publish(shared.clone());
                    }
                }
            }
        }

        // Capture only what something is actually observing. An unobserved
        // intermediate is exactly what a fusing executor is free to never
        // materialize, so keeping it here would defeat the optimization (and
        // pins a frame in memory for a reader that never comes).
        let captured: HashMap<PortId, Arc<Payload>> = arced
            .into_iter()
            .filter(|(port, _)| self.observes(id, port))
            .collect();
        if captured.is_empty() {
            self.node_outputs.remove(id);
        } else {
            self.node_outputs.insert(id.clone(), captured);
        }
    }
}

/// Output ports whose value never leaves its own component.
///
/// Either nothing consumes the value, or only nodes inside the same component
/// do — and a cyclic component materializes its members' outputs each
/// iteration regardless. Capturing these is therefore free: no fusion
/// opportunity is lost, because there was none to lose.
fn terminal_ports(
    graph: &Graph,
    plan: &Plan,
    nodes: &HashMap<NodeId, Box<dyn Node>>,
) -> HashMap<NodeId, HashSet<PortId>> {
    let mut terminal: HashMap<NodeId, HashSet<PortId>> = HashMap::new();
    for (id, node) in nodes {
        let component = plan.component_of(id);
        for spec in node.ports().outputs {
            let escapes = graph.edges.values().any(|conn| {
                conn.from.0 == *id
                    && conn.from.1 == spec.id
                    && plan.component_of(&conn.to.0) != component
            });
            if !escapes {
                terminal.entry(id.clone()).or_default().insert(spec.id);
            }
        }
    }
    terminal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{NodeSpec, Params};

    fn spec(id: &str, kind: &str) -> NodeSpec {
        NodeSpec {
            id: NodeId(id.to_string()),
            kind: kind.to_string(),
            params: Params::new(),
        }
    }

    fn port(node: &str, port: &str) -> (NodeId, PortId) {
        (NodeId(node.to_string()), PortId(port.to_string()))
    }

    #[test]
    fn linear_chain_compiles_to_ordered_acyclic_steps() {
        let mut g = Graph::new();
        g.add_node(spec("a", "k")).unwrap();
        g.add_node(spec("b", "k")).unwrap();
        g.add_node(spec("c", "k")).unwrap();
        g.connect(port("a", "out"), port("b", "in")).unwrap();
        g.connect(port("b", "out"), port("c", "in")).unwrap();

        let plan = compile(&g);
        assert_eq!(
            plan.components(),
            &[
                Component::Acyclic(NodeId("a".to_string())),
                Component::Acyclic(NodeId("b".to_string())),
                Component::Acyclic(NodeId("c".to_string())),
            ]
        );
    }

    #[test]
    fn two_cycle_compiles_to_one_cyclic_component() {
        let mut g = Graph::new();
        g.add_node(spec("a", "k")).unwrap();
        g.add_node(spec("b", "k")).unwrap();
        g.connect(port("a", "out"), port("b", "in")).unwrap();
        g.connect(port("b", "out"), port("a", "in")).unwrap();

        let plan = compile(&g);
        assert_eq!(
            plan.components(),
            &[Component::Cyclic {
                nodes: vec![NodeId("a".to_string()), NodeId("b".to_string())],
            }]
        );
    }

    #[test]
    fn self_loop_is_cyclic() {
        let mut g = Graph::new();
        g.add_node(spec("a", "k")).unwrap();
        g.connect(port("a", "out"), port("a", "in")).unwrap();

        let plan = compile(&g);
        assert_eq!(
            plan.components(),
            &[Component::Cyclic {
                nodes: vec![NodeId("a".to_string())],
            }]
        );
    }

    /// A diamond: `s` fans out to `x` and `y`, which both feed `m`.
    fn diamond() -> Graph {
        let mut g = Graph::new();
        for id in ["s", "x", "y", "m"] {
            g.add_node(spec(id, "k")).unwrap();
        }
        g.connect(port("s", "out"), port("x", "in")).unwrap();
        g.connect(port("s", "out"), port("y", "in")).unwrap();
        g.connect(port("x", "out"), port("m", "a")).unwrap();
        g.connect(port("y", "out"), port("m", "b")).unwrap();
        g
    }

    #[test]
    fn condensation_edges_survive_compilation() {
        let plan = compile(&diamond());
        let at = |id: &str| plan.component_of(&NodeId(id.to_string())).unwrap();

        // `x` and `y` are independent, so which of them takes the lower index
        // is arbitrary; compare the edge set, not an incidental numbering.
        let mut expected = vec![
            (at("s"), at("x")),
            (at("s"), at("y")),
            (at("x"), at("m")),
            (at("y"), at("m")),
        ];
        expected.sort_unstable();
        assert_eq!(plan.deps(), expected);

        let sorted = |mut v: Vec<usize>| {
            v.sort_unstable();
            v
        };
        assert_eq!(
            sorted(plan.successors(at("s"))),
            sorted(vec![at("x"), at("y")])
        );
        assert_eq!(
            sorted(plan.predecessors(at("m"))),
            sorted(vec![at("x"), at("y")])
        );
        assert!(plan.predecessors(at("s")).is_empty());
    }

    #[test]
    fn parallel_branches_are_visible_in_the_dag_but_not_the_flat_order() {
        let plan = compile(&diamond());
        let at = |id: &str| plan.component_of(&NodeId(id.to_string())).unwrap();

        // Flattened, x and y are simply adjacent steps — indistinguishable from
        // a dependency. The DAG shows neither feeds the other.
        assert!(!plan.successors(at("x")).contains(&at("y")));
        assert!(!plan.successors(at("y")).contains(&at("x")));
    }

    #[test]
    fn duplicate_edges_between_two_nodes_yield_one_dependency() {
        // Two ports of `b` fed from `a`: still a single component dependency.
        let mut g = Graph::new();
        g.add_node(spec("a", "k")).unwrap();
        g.add_node(spec("b", "k")).unwrap();
        g.connect(port("a", "out"), port("b", "one")).unwrap();
        g.connect(port("a", "out"), port("b", "two")).unwrap();

        let plan = compile(&g);
        assert_eq!(plan.deps().len(), 1);
    }

    #[test]
    fn straight_run_is_one_fusable_chain() {
        let mut g = Graph::new();
        for id in ["a", "b", "c"] {
            g.add_node(spec(id, "k")).unwrap();
        }
        g.connect(port("a", "out"), port("b", "in")).unwrap();
        g.connect(port("b", "out"), port("c", "in")).unwrap();

        let plan = compile(&g);
        assert_eq!(plan.linear_chains(), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn fan_out_and_fan_in_break_chains() {
        // In the diamond nothing is chained one-to-one: `s` has two consumers
        // and `m` has two producers, so each branch stands alone.
        let plan = compile(&diamond());
        assert!(
            plan.linear_chains().is_empty(),
            "no component pair is a private producer/consumer link"
        );
    }

    #[test]
    fn a_chain_stops_at_a_cycle() {
        // a -> b -> c, with b self-looping: b is cyclic, so it is a barrier and
        // no run of two or more acyclic components remains.
        let mut g = Graph::new();
        for id in ["a", "b", "c"] {
            g.add_node(spec(id, "k")).unwrap();
        }
        g.connect(port("a", "out"), port("b", "in")).unwrap();
        g.connect(port("b", "out"), port("b", "in")).unwrap();
        g.connect(port("b", "out"), port("c", "in")).unwrap();

        let plan = compile(&g);
        let at = |id: &str| plan.component_of(&NodeId(id.to_string())).unwrap();
        assert!(matches!(
            plan.components()[at("b")],
            Component::Cyclic { .. }
        ));
        assert!(plan.linear_chains().is_empty());
    }

    #[test]
    fn chains_are_disjoint_and_maximal() {
        // Two independent runs: a->b->c and p->q.
        let mut g = Graph::new();
        for id in ["a", "b", "c", "p", "q"] {
            g.add_node(spec(id, "k")).unwrap();
        }
        g.connect(port("a", "out"), port("b", "in")).unwrap();
        g.connect(port("b", "out"), port("c", "in")).unwrap();
        g.connect(port("p", "out"), port("q", "in")).unwrap();

        let plan = compile(&g);
        let chains = plan.linear_chains();
        assert_eq!(chains.len(), 2);

        let mut lens: Vec<usize> = chains.iter().map(Vec::len).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![2, 3]);

        // No component appears in two chains.
        let mut seen: Vec<usize> = chains.concat();
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total);
    }
}
