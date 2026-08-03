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

use std::collections::HashMap;
use std::sync::Arc;

use crate::data::Payload;
use crate::exec::{
    BuildError, EdgeBuffer, Inputs, Node, Outputs, Registry, ValidationError, validate,
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

/// A compiled execution order for a graph.
#[derive(Debug, Clone)]
pub struct Plan {
    components: Vec<Component>,
}

impl Plan {
    pub fn components(&self) -> &[Component] {
        &self.components
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

    let mut components = Vec::with_capacity(sccs.len());
    for comp in sccs.into_iter().rev() {
        if comp.len() == 1 && !has_self_loop[comp[0]] {
            components.push(Component::Acyclic(ids[comp[0]].clone()));
        } else {
            let mut nodes: Vec<NodeId> = comp.into_iter().map(|i| ids[i].clone()).collect();
            nodes.sort_by(|a, b| a.0.cmp(&b.0));
            components.push(Component::Cyclic { nodes });
        }
    }

    Plan { components }
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

        Ok(Self {
            plan,
            nodes,
            edges,
            incoming,
            outgoing,
            external: HashMap::new(),
            node_outputs: HashMap::new(),
            max_iters: DEFAULT_MAX_ITERS,
        })
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

    /// The last value a node produced on `port`, if any.
    pub fn output(&self, node: &NodeId, port: &str) -> Option<&Payload> {
        self.node_outputs.get(node)?.get(port).map(Arc::as_ref)
    }

    /// A cheap shared handle to a node's last output on `port`.
    pub fn output_arc(&self, node: &NodeId, port: &str) -> Option<Arc<Payload>> {
        self.node_outputs.get(node)?.get(port).cloned()
    }

    /// Clear all edge buffers and captured outputs, and reset node state.
    pub fn reset(&mut self) {
        for e in &mut self.edges {
            e.buffer.clear();
        }
        self.node_outputs.clear();
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
        // Share each output payload across its fan-out edges via one Arc.
        let arced: HashMap<PortId, Arc<Payload>> = outputs
            .into_map()
            .into_iter()
            .map(|(port, payload)| (port, Arc::new(payload)))
            .collect();

        let out_edges = self.outgoing.get(id).cloned().unwrap_or_default();
        for i in out_edges {
            let port = self.edges[i].from.1.clone();
            if let Some(shared) = arced.get(&port) {
                self.edges[i].buffer.push_arc(shared.clone());
            }
        }

        self.node_outputs.insert(id.clone(), arced);
    }
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
}
