//! Graph compilation / planning.
//!
//! This layer turns a user-authored [`Graph`](crate::graph::Graph) into a form
//! that's convenient for schedulers/executors.
//!
//! Current responsibilities:
//! - Run structural validation (core invariants).
//! - Decompose into SCCs and produce a component DAG.
//! - Topologically order the component DAG.

use std::collections::{BTreeSet, HashMap};

use crate::graph::{
    ComponentGraph, ComponentId, Graph, GraphValidationError, GraphValidationOptions, NodeId,
};

#[derive(Debug, Clone)]
pub struct GraphPlan {
    pub component_graph: ComponentGraph,
    /// Topological order of components in `component_graph`.
    pub component_order: Vec<ComponentId>,
    /// Components that represent cycles.
    pub cyclic_components: Vec<ComponentId>,
}

impl GraphPlan {
    /// Convenience: return an acyclic execution order of node ids.
    ///
    /// This flattens `component_order` and skips cyclic components.
    ///
    /// Notes:
    /// - Node order within a component is stable (sorted by `NodeId.0`).
    /// - Cyclic SCCs are intentionally excluded; a tick-based scheduler should
    ///   decide how to order/iterate nodes inside them.
    pub fn acyclic_node_order(&self) -> Vec<NodeId> {
        let mut out: Vec<NodeId> = Vec::new();
        for &cid in &self.component_order {
            if self.cyclic_components.contains(&cid) {
                continue;
            }
            if let Some(comp) = self.component_graph.components.iter().find(|c| c.id == cid) {
                out.extend(comp.nodes.iter().cloned());
            }
        }
        out
    }

    /// Convenience: return the cyclic SCCs as groups of node ids (stable).
    pub fn cyclic_node_groups(&self) -> Vec<Vec<NodeId>> {
        let mut out: Vec<Vec<NodeId>> = Vec::new();
        for &cid in &self.cyclic_components {
            if let Some(comp) = self.component_graph.components.iter().find(|c| c.id == cid) {
                out.push(comp.nodes.clone());
            }
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphPlanError {
    Validation(Vec<GraphValidationError>),
    /// Should be impossible because SCC-collapsed graphs are DAGs.
    InternalCycle,
}

impl Graph {
    /// Compile this graph into a schedulable plan.
    pub fn plan(&self, opts: GraphValidationOptions) -> Result<GraphPlan, GraphPlanError> {
        if let Err(errs) = self.validate(opts) {
            return Err(GraphPlanError::Validation(errs));
        }

        let component_graph = self.component_graph();
        let component_order =
            topo_sort_components(&component_graph).ok_or(GraphPlanError::InternalCycle)?;

        let mut cyclic_components: Vec<ComponentId> = component_graph
            .components
            .iter()
            .filter(|c| c.is_cyclic)
            .map(|c| c.id)
            .collect();
        cyclic_components.sort();

        Ok(GraphPlan {
            component_graph,
            component_order,
            cyclic_components,
        })
    }

    /// Convenience: compile using the common UI/editor validation preset.
    pub fn plan_strict(&self) -> Result<GraphPlan, GraphPlanError> {
        self.plan(GraphValidationOptions::strict())
    }
}

fn topo_sort_components(cg: &ComponentGraph) -> Option<Vec<ComponentId>> {
    // Kahn's algorithm. Tie-break deterministically by ComponentId.
    let mut indegree: HashMap<ComponentId, usize> = HashMap::new();
    for c in &cg.components {
        indegree.insert(c.id, 0);
    }
    for &(_, to) in &cg.edges {
        *indegree.entry(to).or_insert(0) += 1;
    }

    let mut q: BTreeSet<ComponentId> = indegree
        .iter()
        .filter_map(|(&cid, &deg)| if deg == 0 { Some(cid) } else { None })
        .collect();

    let mut out: Vec<ComponentId> = Vec::with_capacity(cg.components.len());

    // adjacency from component edges
    let mut adj: HashMap<ComponentId, Vec<ComponentId>> = HashMap::new();
    for &(a, b) in &cg.edges {
        adj.entry(a).or_default().push(b);
    }

    while let Some(&cid) = q.iter().next() {
        q.remove(&cid);
        out.push(cid);

        if let Some(nexts) = adj.get(&cid) {
            // deterministic traversal of outgoing edges
            let mut nexts = nexts.clone();
            nexts.sort();
            for n in nexts {
                let e = indegree.get_mut(&n).unwrap();
                *e -= 1;
                if *e == 0 {
                    q.insert(n);
                }
            }
        }
    }

    if out.len() == cg.components.len() {
        Some(out)
    } else {
        None
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
    fn plan_orders_components_and_reports_cyclic() {
        // a <-> b (cycle) -> c
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_node(node("c")).unwrap();

        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("b".into()), PortId("in".into())),
        )
        .unwrap();
        g.connect(
            (NodeId("b".into()), PortId("out".into())),
            (NodeId("a".into()), PortId("in".into())),
        )
        .unwrap();
        g.connect(
            (NodeId("b".into()), PortId("out2".into())),
            (NodeId("c".into()), PortId("in".into())),
        )
        .unwrap();

        let plan = g.plan(GraphValidationOptions::default()).unwrap();
        assert_eq!(plan.component_graph.components.len(), 2);
        assert_eq!(plan.cyclic_components.len(), 1);

        let ca = *plan
            .component_graph
            .node_component
            .get(&NodeId("a".into()))
            .unwrap();
        let cc = *plan
            .component_graph
            .node_component
            .get(&NodeId("c".into()))
            .unwrap();

        // topo order must place cycle component before c
        let posa = plan.component_order.iter().position(|&x| x == ca).unwrap();
        let posc = plan.component_order.iter().position(|&x| x == cc).unwrap();
        assert!(posa < posc);

        // helper methods
        assert_eq!(plan.acyclic_node_order(), vec![NodeId("c".into())]);
        assert_eq!(plan.cyclic_node_groups(), vec![vec![NodeId("a".into()), NodeId("b".into())]]);
    }

    #[test]
    fn plan_fails_on_validation_errors() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        // dangling reference: missing node "b"
        g.edges.insert(
            crate::graph::EdgeId(1),
            crate::graph::Connection {
                from: (NodeId("a".into()), PortId("out".into())),
                to: (NodeId("b".into()), PortId("in".into())),
            },
        );

        let err = g.plan(GraphValidationOptions::default()).unwrap_err();
        assert!(matches!(err, GraphPlanError::Validation(_)));
    }
}
