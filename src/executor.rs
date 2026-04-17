//! Executor sketch (WIP).
//!
//! This is intentionally minimal: the goal is to show *where* `GraphPlan` fits
//! and how cyclic SCCs get isolated for iterative/tick scheduling.
//!
//! Future work:
//! - Build per-node processors from `NodeSpec.kind`
//! - Route values along edges (port-aware)
//! - Define a typed value model / frame model
//! - Attach taps at node+port or edge

use crate::graph::{ComponentId, Graph, GraphValidationOptions, NodeId, PortId};
use crate::plan::{GraphPlan, GraphPlanError};
use crate::tap::{Tap, TapPoint, TapRegistry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickMode {
    /// Execute all acyclic components once in topological order.
    Once,
    /// Execute with an explicit number of ticks; cyclic components run each tick.
    Ticks(u64),
}

/// A trace event emitted by the executor.
///
/// This is the first (tiny) integration point between execution and the `Tap` API.
/// UIs can subscribe to a `Tap<ExecutionEvent>` to see what the scheduler is doing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionEvent {
    pub tick: u64,
    pub component: ComponentId,
    pub node: NodeId,
}

/// Stable tap point used for executor trace events.
///
/// This is intentionally a reserved/"system" node id so UIs can attach without
/// needing to know any user graph node ids.
pub fn executor_trace_point() -> TapPoint {
    TapPoint::node_port(NodeId("__executor".into()), PortId("trace".into()))
}

/// An executor that owns a compiled plan.
#[derive(Debug, Clone)]
pub struct Executor {
    pub plan: GraphPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorError {
    Plan(GraphPlanError),
}

impl Executor {
    /// Compile a graph into an executor.
    pub fn compile(g: &Graph, opts: GraphValidationOptions) -> Result<Self, ExecutorError> {
        let plan = g.plan(opts).map_err(ExecutorError::Plan)?;
        Ok(Self { plan })
    }

    /// Compile using the common UI/editor validation preset.
    pub fn compile_strict(g: &Graph) -> Result<Self, ExecutorError> {
        let plan = g.plan_strict().map_err(ExecutorError::Plan)?;
        Ok(Self { plan })
    }

    /// Run the graph according to the chosen tick mode.
    ///
    /// This is currently a no-op skeleton that only demonstrates scheduling.
    pub fn run(&self, mode: TickMode) {
        self.run_with_trace(mode, None)
    }

    /// Run the graph, optionally publishing a trace event for each executed node.
    ///
    /// The trace tap stores only the *latest* event (by design). If you want a
    /// history, a UI layer can buffer these events itself.
    pub fn run_with_trace(&self, mode: TickMode, trace: Option<&Tap<ExecutionEvent>>) {
        match mode {
            TickMode::Once => {
                self.run_acyclic_once(0, trace);
            }
            TickMode::Ticks(n) => {
                for tick in 0..n {
                    self.run_tick(tick, trace);
                }
            }
        }
    }

    /// Run the graph and publish trace events into a [`TapRegistry`].
    ///
    /// This gives UIs a stable attachment point without passing tap handles:
    ///
    /// - tap point: `__executor.trace` (see [`executor_trace_point`])
    pub fn run_with_registry(&self, mode: TickMode, reg: &TapRegistry<ExecutionEvent>) {
        let t = reg.tap_at(executor_trace_point());
        self.run_with_trace(mode, Some(&t));
    }

    fn run_acyclic_once(&self, tick: u64, trace: Option<&Tap<ExecutionEvent>>) {
        for cid in self.plan.acyclic_component_order() {
            self.exec_component(tick, cid, trace);
        }
    }

    fn run_tick(&self, tick: u64, trace: Option<&Tap<ExecutionEvent>>) {
        // One reasonable default: execute acyclic SCCs in topo order each tick,
        // then execute cyclic SCCs (or vice versa). The real choice will depend
        // on dataflow semantics (push vs pull, statefulness, etc.).
        for &cid in &self.plan.component_order {
            if !self.is_cyclic(cid) {
                self.exec_component(tick, cid, trace);
            }
        }

        for &cid in &self.plan.cyclic_components {
            self.exec_component(tick, cid, trace);
        }
    }

    fn is_cyclic(&self, cid: ComponentId) -> bool {
        self.plan.is_cyclic_component(cid)
    }

    fn exec_component(&self, tick: u64, cid: ComponentId, trace: Option<&Tap<ExecutionEvent>>) {
        // Placeholder scheduling demo: execute nodes in this SCC in a stable order.
        //
        // Notes:
        // - For acyclic SCCs, this is a reasonable default.
        // - For cyclic SCCs, a real engine likely needs a policy (fixed-point
        //   iteration, limited inner iters, pull/push semantics, etc.). Here we
        //   just demonstrate that cycles are isolated *as SCCs* and can be run
        //   as a unit.
        let Some(comp) = self.plan.component(cid) else {
            return;
        };

        for nid in &comp.nodes {
            self.exec_node(tick, cid, nid, trace);
        }
    }

    fn exec_node(
        &self,
        tick: u64,
        cid: ComponentId,
        nid: &NodeId,
        trace: Option<&Tap<ExecutionEvent>>,
    ) {
        // Placeholder. Eventually: look up processor for node kind and execute.
        if let Some(t) = trace {
            t.publish(ExecutionEvent {
                tick,
                component: cid,
                node: nid.clone(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{NodeId, NodeSpec, Params, PortId};
    use crate::tap::{Tap, TapRegistry};

    fn node(id: &str) -> NodeSpec {
        NodeSpec {
            id: NodeId(id.to_string()),
            kind: "noop".to_string(),
            params: Params::default(),
        }
    }

    #[test]
    fn compile_executor_from_valid_graph() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("b".into()), PortId("in".into())),
        )
        .unwrap();

        let ex = Executor::compile(&g, GraphValidationOptions::default()).unwrap();
        assert_eq!(ex.plan.component_graph.components.len(), 2);

        let ex2 = Executor::compile_strict(&g).unwrap();
        assert_eq!(ex2.plan.component_graph.components.len(), 2);
    }

    #[test]
    fn executor_can_publish_trace_events_via_tap() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("b".into()), PortId("in".into())),
        )
        .unwrap();

        let ex = Executor::compile(&g, GraphValidationOptions::default()).unwrap();
        let trace: Tap<ExecutionEvent> = Tap::new();

        ex.run_with_trace(TickMode::Once, Some(&trace));

        let (_, last) = trace.latest_with_seq();
        let last = last.expect("expected at least one trace event");
        // last executed node should be "b" (acyclic topo order)
        assert_eq!(last.tick, 0);
        assert_eq!(last.node, NodeId("b".into()));
    }

    #[test]
    fn executor_can_publish_trace_events_via_registry() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("b".into()), PortId("in".into())),
        )
        .unwrap();

        let ex = Executor::compile(&g, GraphValidationOptions::default()).unwrap();
        let reg: TapRegistry<ExecutionEvent> = TapRegistry::new();

        ex.run_with_registry(TickMode::Once, &reg);

        let t = reg.tap_at(executor_trace_point());
        let (_, last) = t.latest_with_seq();
        let last = last.expect("expected at least one trace event");
        assert_eq!(last.node, NodeId("b".into()));
    }
}

