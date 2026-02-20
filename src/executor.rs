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

use crate::graph::{ComponentId, Graph, GraphValidationOptions};
use crate::plan::{GraphPlan, GraphPlanError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickMode {
    /// Execute all acyclic components once in topological order.
    Once,
    /// Execute with an explicit number of ticks; cyclic components run each tick.
    Ticks(u64),
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

    /// Run the graph according to the chosen tick mode.
    ///
    /// This is currently a no-op skeleton that only demonstrates scheduling.
    pub fn run(&self, mode: TickMode) {
        match mode {
            TickMode::Once => {
                self.run_acyclic_once();
            }
            TickMode::Ticks(n) => {
                for _ in 0..n {
                    self.run_tick();
                }
            }
        }
    }

    fn run_acyclic_once(&self) {
        for &cid in &self.plan.component_order {
            if self.is_cyclic(cid) {
                // Skip cyclic SCCs in one-shot mode.
                continue;
            }
            self.exec_component(cid);
        }
    }

    fn run_tick(&self) {
        // One reasonable default: execute acyclic SCCs in topo order each tick,
        // then execute cyclic SCCs (or vice versa). The real choice will depend
        // on dataflow semantics (push vs pull, statefulness, etc.).
        for &cid in &self.plan.component_order {
            if !self.is_cyclic(cid) {
                self.exec_component(cid);
            }
        }

        for &cid in &self.plan.cyclic_components {
            self.exec_component(cid);
        }
    }

    fn is_cyclic(&self, cid: ComponentId) -> bool {
        self.plan
            .component_graph
            .components
            .iter()
            .find(|c| c.id == cid)
            .map(|c| c.is_cyclic)
            .unwrap_or(false)
    }

    fn exec_component(&self, _cid: ComponentId) {
        // Placeholder. Eventually: execute nodes in this SCC in a policy-defined order.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{NodeId, NodeSpec, Params, PortId};

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
    }
}
