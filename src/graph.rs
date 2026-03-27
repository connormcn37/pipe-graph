//! Core graph types (runtime-facing).
//!
//! This module is intentionally small and dependency-light.
//! The idea is that UI layers (Bevy/egui/etc.) can mirror these types.

use std::collections::{HashMap, HashSet};

/// Stable identifier for a node/stage in the graph.
///
/// Early version uses a string label (matches README intent: unique label).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub String);

/// Identifier for an input or output port.
///
/// Ports are named so they can map cleanly to UI pins.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PortId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u64);

/// Identifier for a strongly-connected component in the decomposed graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ComponentId(pub usize);

#[derive(Debug, Clone)]
pub struct Connection {
    pub from: (NodeId, PortId),
    pub to: (NodeId, PortId),
}

/// Parameter map for configuring stages.
pub type Params = HashMap<String, String>;

/// A node specification (runtime graph). UI/editor should produce these.
#[derive(Debug, Clone)]
pub struct NodeSpec {
    pub id: NodeId,
    /// Stage "type" (e.g. "crop", "cast", "merge").
    pub kind: String,
    pub params: Params,
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    pub nodes: HashMap<NodeId, NodeSpec>,
    pub edges: HashMap<EdgeId, Connection>,
    next_edge_id: u64,
}

/// A single SCC in the graph.
#[derive(Debug, Clone)]
pub struct Component {
    pub id: ComponentId,
    pub nodes: Vec<NodeId>,
    /// True if the component contains a cycle (size>1 or explicit self-loop).
    pub is_cyclic: bool,
}

/// DAG of SCCs (each SCC collapsed to a single vertex).
#[derive(Debug, Clone)]
pub struct ComponentGraph {
    pub components: Vec<Component>,
    /// Map each node id to its SCC id.
    pub node_component: HashMap<NodeId, ComponentId>,
    /// Component-level edges (from_scc -> to_scc).
    pub edges: HashSet<(ComponentId, ComponentId)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    DuplicateNodeId(String),
    MissingNode(String),
}

/// Errors returned by [`Graph::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphValidationError {
    /// A node id is empty ("" after trimming).
    EmptyNodeId { node: NodeId },
    /// A node kind is empty ("" after trimming).
    EmptyNodeKind { node: NodeId },
    /// An edge references a node id that does not exist.
    MissingNode { node: NodeId },
    /// The graph contains at least one cycle (SCC with >1 node or explicit self-loop).
    CyclicComponent { nodes: Vec<NodeId> },
    /// An edge references an empty port id ("" after trimming).
    MissingPort { node: NodeId, port: PortId },
    /// An edge connects a node+port to itself.
    SelfLoop { node: NodeId, port: PortId },
    /// Two edges have identical endpoints (from,to).
    DuplicateEdge {
        from: (NodeId, PortId),
        to: (NodeId, PortId),
    },
    /// More than one edge targets the same input endpoint (node,port).
    MultipleInboundToPort { node: NodeId, port: PortId },
    /// Node has no incident edges (isolated) when required by options.
    UnconnectedNode { node: NodeId },
}

#[derive(Debug, Clone, Copy)]
pub struct GraphValidationOptions {
    /// If true, a node must appear in at least one edge (in or out).
    pub require_all_nodes_connected: bool,
    /// If true, disallow edges that connect a port to itself.
    pub disallow_self_loops: bool,
    /// If true, disallow multiple inbound edges to the same input port.
    ///
    /// This is a common constraint in node/pin UIs where each input pin accepts
    /// exactly one connection.
    pub disallow_multiple_inbound_to_port: bool,
    /// If true, disallow any cycles in the graph.
    pub disallow_cycles: bool,
}

impl Default for GraphValidationOptions {
    fn default() -> Self {
        Self {
            require_all_nodes_connected: false,
            disallow_self_loops: false,
            disallow_multiple_inbound_to_port: false,
            disallow_cycles: false,
        }
    }
}

impl GraphValidationOptions {
    /// A reasonable "UI/editor" preset: catch the most common structural issues.
    pub fn strict() -> Self {
        Self {
            require_all_nodes_connected: true,
            disallow_self_loops: true,
            disallow_multiple_inbound_to_port: true,
            disallow_cycles: false,
        }
    }

    /// Strict + acyclic (useful for DAG-only schedulers).
    pub fn strict_acyclic() -> Self {
        Self {
            disallow_cycles: true,
            ..Self::strict()
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the graph structure without mutating it.
    ///
    /// This is intentionally conservative: it checks only invariants that are
    /// knowable from the core `Graph` model (nodes, edges, and port *names*).
    pub fn validate(&self, opts: GraphValidationOptions) -> Result<(), Vec<GraphValidationError>> {
        let mut errs: Vec<GraphValidationError> = Vec::new();

        // Node-level checks.
        for spec in self.nodes.values() {
            if spec.id.0.trim().is_empty() {
                errs.push(GraphValidationError::EmptyNodeId {
                    node: spec.id.clone(),
                });
            }
            if spec.kind.trim().is_empty() {
                errs.push(GraphValidationError::EmptyNodeKind {
                    node: spec.id.clone(),
                });
            }
        }

        // Endpoint-level checks + duplicate detection.
        let mut seen: HashSet<((NodeId, PortId), (NodeId, PortId))> = HashSet::new();
        let mut inbound_seen: HashSet<(NodeId, PortId)> = HashSet::new();

        for conn in self.edges.values() {
            // Missing nodes.
            if !self.nodes.contains_key(&conn.from.0) {
                errs.push(GraphValidationError::MissingNode {
                    node: conn.from.0.clone(),
                });
            }
            if !self.nodes.contains_key(&conn.to.0) {
                errs.push(GraphValidationError::MissingNode {
                    node: conn.to.0.clone(),
                });
            }

            // Missing ports (we only know the name exists / is non-empty).
            if conn.from.1.0.trim().is_empty() {
                errs.push(GraphValidationError::MissingPort {
                    node: conn.from.0.clone(),
                    port: conn.from.1.clone(),
                });
            }
            if conn.to.1.0.trim().is_empty() {
                errs.push(GraphValidationError::MissingPort {
                    node: conn.to.0.clone(),
                    port: conn.to.1.clone(),
                });
            }

            // Self-loop checks.
            if opts.disallow_self_loops && conn.from == conn.to {
                errs.push(GraphValidationError::SelfLoop {
                    node: conn.from.0.clone(),
                    port: conn.from.1.clone(),
                });
            }

            // Duplicate edges.
            let key = (conn.from.clone(), conn.to.clone());
            if !seen.insert(key.clone()) {
                errs.push(GraphValidationError::DuplicateEdge {
                    from: key.0,
                    to: key.1,
                });
            }

            // Multiple inbound edges to the same input endpoint.
            if opts.disallow_multiple_inbound_to_port {
                let to_key = conn.to.clone();
                if !inbound_seen.insert(to_key.clone()) {
                    errs.push(GraphValidationError::MultipleInboundToPort {
                        node: to_key.0,
                        port: to_key.1,
                    });
                }
            }
        }

        if opts.require_all_nodes_connected {
            let mut connected: HashSet<NodeId> = HashSet::new();
            for conn in self.edges.values() {
                connected.insert(conn.from.0.clone());
                connected.insert(conn.to.0.clone());
            }
            for id in self.nodes.keys() {
                if !connected.contains(id) {
                    errs.push(GraphValidationError::UnconnectedNode { node: id.clone() });
                }
            }
        }

        if opts.disallow_cycles {
            // Use SCC decomposition to detect cycles.
            for c in self.component_graph().components {
                if c.is_cyclic {
                    errs.push(GraphValidationError::CyclicComponent { nodes: c.nodes });
                }
            }
        }

        if errs.is_empty() {
            Ok(())
        } else {
            // De-dupe identical errors to keep output stable.
            errs.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
            errs.dedup();
            Err(errs)
        }
    }

    /// Convenience: validate using the common UI/editor strict preset.
    pub fn validate_strict(&self) -> Result<(), Vec<GraphValidationError>> {
        self.validate(GraphValidationOptions::strict())
    }

    pub fn add_node(&mut self, spec: NodeSpec) -> Result<(), GraphError> {
        if self.nodes.contains_key(&spec.id) {
            return Err(GraphError::DuplicateNodeId(spec.id.0));
        }
        self.nodes.insert(spec.id.clone(), spec);
        Ok(())
    }

    /// Strongly connected components (Tarjan).
    ///
    /// Returns components in a stable-ish order (deterministic for a given set of
    /// node ids and edges), but callers should not rely on a specific ordering.
    pub fn strongly_connected_components(&self) -> Vec<Vec<NodeId>> {
        // Build adjacency for node->node based on edges (ports ignored here).
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for id in self.nodes.keys() {
            adj.entry(id.clone()).or_default();
        }
        for conn in self.edges.values() {
            // Skip dangling endpoints; those are handled by validate().
            if self.nodes.contains_key(&conn.from.0) && self.nodes.contains_key(&conn.to.0) {
                adj.entry(conn.from.0.clone())
                    .or_default()
                    .push(conn.to.0.clone());
            }
        }

        // Tarjan SCC.
        let mut index: usize = 0;
        let mut indices: HashMap<NodeId, usize> = HashMap::new();
        let mut lowlink: HashMap<NodeId, usize> = HashMap::new();
        let mut stack: Vec<NodeId> = Vec::new();
        let mut on_stack: HashSet<NodeId> = HashSet::new();
        let mut out: Vec<Vec<NodeId>> = Vec::new();

        // Deterministic traversal: sort node ids by string.
        let mut nodes: Vec<NodeId> = adj.keys().cloned().collect();
        nodes.sort_by(|a, b| a.0.cmp(&b.0));

        fn strongconnect(
            v: NodeId,
            index: &mut usize,
            indices: &mut HashMap<NodeId, usize>,
            lowlink: &mut HashMap<NodeId, usize>,
            stack: &mut Vec<NodeId>,
            on_stack: &mut HashSet<NodeId>,
            adj: &HashMap<NodeId, Vec<NodeId>>,
            out: &mut Vec<Vec<NodeId>>,
        ) {
            indices.insert(v.clone(), *index);
            lowlink.insert(v.clone(), *index);
            *index += 1;
            stack.push(v.clone());
            on_stack.insert(v.clone());

            if let Some(nexts) = adj.get(&v) {
                // Deterministic traversal of outgoing edges.
                let mut nexts = nexts.clone();
                nexts.sort_by(|a, b| a.0.cmp(&b.0));
                for w in nexts {
                    if !indices.contains_key(&w) {
                        strongconnect(
                            w.clone(),
                            index,
                            indices,
                            lowlink,
                            stack,
                            on_stack,
                            adj,
                            out,
                        );
                        let lw = *lowlink.get(&w).unwrap();
                        let lv = lowlink.get_mut(&v).unwrap();
                        *lv = (*lv).min(lw);
                    } else if on_stack.contains(&w) {
                        let iw = *indices.get(&w).unwrap();
                        let lv = lowlink.get_mut(&v).unwrap();
                        *lv = (*lv).min(iw);
                    }
                }
            }

            // If v is a root node, pop the stack and output an SCC.
            let lv = *lowlink.get(&v).unwrap();
            let iv = *indices.get(&v).unwrap();
            if lv == iv {
                let mut scc: Vec<NodeId> = Vec::new();
                loop {
                    let w = stack.pop().expect("tarjan stack underflow");
                    on_stack.remove(&w);
                    scc.push(w.clone());
                    if w == v {
                        break;
                    }
                }
                // Stable node ordering inside SCC.
                scc.sort_by(|a, b| a.0.cmp(&b.0));
                out.push(scc);
            }
        }

        for v in nodes {
            if !indices.contains_key(&v) {
                strongconnect(
                    v,
                    &mut index,
                    &mut indices,
                    &mut lowlink,
                    &mut stack,
                    &mut on_stack,
                    &adj,
                    &mut out,
                );
            }
        }

        out
    }

    /// Collapse SCCs into a component DAG.
    pub fn component_graph(&self) -> ComponentGraph {
        let sccs = self.strongly_connected_components();

        let mut node_component: HashMap<NodeId, ComponentId> = HashMap::new();
        for (i, comp) in sccs.iter().enumerate() {
            let cid = ComponentId(i);
            for n in comp {
                node_component.insert(n.clone(), cid);
            }
        }

        let mut edges: HashSet<(ComponentId, ComponentId)> = HashSet::new();
        for conn in self.edges.values() {
            let Some(&a) = node_component.get(&conn.from.0) else {
                continue;
            };
            let Some(&b) = node_component.get(&conn.to.0) else {
                continue;
            };
            if a != b {
                edges.insert((a, b));
            }
        }

        let mut components: Vec<Component> = Vec::with_capacity(sccs.len());
        for (i, nodes) in sccs.into_iter().enumerate() {
            let cid = ComponentId(i);
            let has_self_loop = if nodes.len() == 1 {
                let n = &nodes[0];
                self.edges.values().any(|c| c.from.0 == *n && c.to.0 == *n)
            } else {
                false
            };
            components.push(Component {
                id: cid,
                is_cyclic: nodes.len() > 1 || has_self_loop,
                nodes,
            });
        }

        ComponentGraph {
            components,
            node_component,
            edges,
        }
    }

    pub fn connect(
        &mut self,
        from: (NodeId, PortId),
        to: (NodeId, PortId),
    ) -> Result<EdgeId, GraphError> {
        if !self.nodes.contains_key(&from.0) {
            return Err(GraphError::MissingNode(from.0.0));
        }
        if !self.nodes.contains_key(&to.0) {
            return Err(GraphError::MissingNode(to.0.0));
        }

        // NOTE: cycles are allowed, but we may still want to treat self-loops specially.
        // Keep this permissive for now.

        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        self.edges.insert(id.clone(), Connection { from, to });
        Ok(id)
    }

    /// Returns a set of node ids referenced by edges but missing from `nodes`.
    pub fn dangling_references(&self) -> HashSet<NodeId> {
        let mut out = HashSet::new();
        for conn in self.edges.values() {
            if !self.nodes.contains_key(&conn.from.0) {
                out.insert(conn.from.0.clone());
            }
            if !self.nodes.contains_key(&conn.to.0) {
                out.insert(conn.to.0.clone());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str) -> NodeSpec {
        NodeSpec {
            id: NodeId(id.to_string()),
            kind: "noop".to_string(),
            params: Params::default(),
        }
    }

    #[test]
    fn validate_ok_minimal_connected_graph() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("b".into()), PortId("in".into())),
        )
        .unwrap();

        g.validate(GraphValidationOptions {
            require_all_nodes_connected: true,
            disallow_self_loops: true,
            disallow_multiple_inbound_to_port: false,
            disallow_cycles: false,
        })
        .unwrap();
    }

    #[test]
    fn strict_options_enable_common_ui_invariants() {
        let opts = GraphValidationOptions::strict();
        assert!(opts.require_all_nodes_connected);
        assert!(opts.disallow_self_loops);
        assert!(opts.disallow_multiple_inbound_to_port);
        assert!(!opts.disallow_cycles);

        let opts2 = GraphValidationOptions::strict_acyclic();
        assert!(opts2.disallow_cycles);
    }

    #[test]
    fn validate_reports_empty_node_id_and_kind() {
        let mut g = Graph::new();
        g.add_node(NodeSpec {
            id: NodeId("   ".into()),
            kind: "noop".into(),
            params: Params::default(),
        })
        .unwrap();
        g.add_node(NodeSpec {
            id: NodeId("x".into()),
            kind: "   ".into(),
            params: Params::default(),
        })
        .unwrap();

        let errs = g.validate(GraphValidationOptions::default()).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphValidationError::EmptyNodeId { .. }))
        );
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphValidationError::EmptyNodeKind { .. }))
        );
    }

    #[test]
    fn validate_reports_duplicate_edges() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("b".into()), PortId("in".into())),
        )
        .unwrap();

        // Add a second identical edge by bypassing connect() (edge IDs are distinct).
        g.edges.insert(
            EdgeId(999),
            Connection {
                from: (NodeId("a".into()), PortId("out".into())),
                to: (NodeId("b".into()), PortId("in".into())),
            },
        );

        let errs = g.validate(GraphValidationOptions::default()).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphValidationError::DuplicateEdge { .. }))
        );
    }

    #[test]
    fn validate_reports_multiple_inbound_to_same_port_when_enabled() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_node(node("c")).unwrap();

        // a -> c.in and b -> c.in
        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("c".into()), PortId("in".into())),
        )
        .unwrap();
        g.connect(
            (NodeId("b".into()), PortId("out".into())),
            (NodeId("c".into()), PortId("in".into())),
        )
        .unwrap();

        let errs = g
            .validate(GraphValidationOptions {
                disallow_multiple_inbound_to_port: true,
                ..GraphValidationOptions::default()
            })
            .unwrap_err();

        assert!(errs.iter().any(|e| {
            matches!(
                e,
                GraphValidationError::MultipleInboundToPort { node, port }
                    if node.0 == "c" && port.0 == "in"
            )
        }));
    }

    #[test]
    fn validate_reports_missing_ports() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("".into())),
            (NodeId("b".into()), PortId("in".into())),
        )
        .unwrap();

        let errs = g.validate(GraphValidationOptions::default()).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphValidationError::MissingPort { .. }))
        );
    }

    #[test]
    fn validate_reports_unconnected_node_when_required() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        // b is isolated
        g.connect(
            (NodeId("a".into()), PortId("out".into())),
            (NodeId("a".into()), PortId("in".into())),
        )
        .unwrap();

        let errs = g
            .validate(GraphValidationOptions {
                require_all_nodes_connected: true,
                disallow_self_loops: false,
                disallow_multiple_inbound_to_port: false,
                disallow_cycles: false,
            })
            .unwrap_err();
        assert!(
            errs.iter().any(
                |e| matches!(e, GraphValidationError::UnconnectedNode { node } if node.0 == "b")
            )
        );
    }

    #[test]
    fn validate_reports_self_loop_when_disallowed() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.connect(
            (NodeId("a".into()), PortId("p".into())),
            (NodeId("a".into()), PortId("p".into())),
        )
        .unwrap();

        let errs = g
            .validate(GraphValidationOptions {
                require_all_nodes_connected: false,
                disallow_self_loops: true,
                disallow_multiple_inbound_to_port: false,
                disallow_cycles: false,
            })
            .unwrap_err();
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphValidationError::SelfLoop { .. }))
        );
    }

    #[test]
    fn scc_decomposes_acyclic_chain_into_singletons() {
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
            (NodeId("c".into()), PortId("in".into())),
        )
        .unwrap();

        let mut sccs = g.strongly_connected_components();
        sccs.sort_by(|a, b| a[0].0.cmp(&b[0].0));
        assert_eq!(sccs.len(), 3);
        assert_eq!(sccs[0], vec![NodeId("a".into())]);
        assert_eq!(sccs[1], vec![NodeId("b".into())]);
        assert_eq!(sccs[2], vec![NodeId("c".into())]);
    }

    #[test]
    fn scc_groups_cycle() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
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

        let sccs = g.strongly_connected_components();
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0], vec![NodeId("a".into()), NodeId("b".into())]);

        let errs = g
            .validate(GraphValidationOptions {
                disallow_cycles: true,
                ..GraphValidationOptions::default()
            })
            .unwrap_err();
        assert!(errs.iter().any(|e| matches!(
            e,
            GraphValidationError::CyclicComponent { .. }
        )));
    }

    #[test]
    fn component_graph_collapses_sccs_and_builds_dag_edges() {
        let mut g = Graph::new();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_node(node("c")).unwrap();

        // a <-> b cycle, plus edge b -> c
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

        let cg = g.component_graph();
        assert_eq!(cg.components.len(), 2);

        let ca = *cg.node_component.get(&NodeId("a".into())).unwrap();
        let cb = *cg.node_component.get(&NodeId("b".into())).unwrap();
        let cc = *cg.node_component.get(&NodeId("c".into())).unwrap();
        assert_eq!(ca, cb);
        assert_ne!(ca, cc);

        assert!(cg.edges.contains(&(ca, cc)));
        assert_eq!(cg.edges.len(), 1);

        let cyc = cg.components.iter().find(|c| c.id == ca).unwrap();
        assert!(cyc.is_cyclic);
    }
}
