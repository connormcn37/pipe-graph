//! Canvas layout: where each node sits in the editor.
//!
//! Positions are editor state, not pipeline state. They live here rather than
//! in [`crate::graph::NodeSpec`] so that dragging a node around never looks
//! like a pipeline edit (and never forces a runtime rebuild), and so the core
//! graph stays free of presentation concerns.
//!
//! # Auto-placement
//!
//! New nodes need *somewhere* to appear. [`auto_grid`] assigns each node a
//! grid slot that reads left-to-right in data-flow order:
//!
//! - **column** = longest-path depth from the graph's sources. Longest (not
//!   shortest) path, so every edge points strictly rightwards: a node always
//!   sits to the right of everything feeding it. Cycles have no longest path,
//!   so depth is computed over the *condensation* from
//!   [`crate::exec::compile`] — every node of a feedback loop shares the
//!   loop's column.
//! - **row** = the node's index within its column, sorting by id, so placement
//!   is stable and independent of `HashMap` iteration order.

use std::collections::HashMap;

use crate::exec::compile;
use crate::graph::{Graph, NodeId};

/// Horizontal distance between auto-placed columns, in canvas units.
pub const COLUMN_SPACING: f32 = 220.0;
/// Vertical distance between auto-placed rows, in canvas units.
pub const ROW_SPACING: f32 = 120.0;

/// A cell in the auto-placement grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridSlot {
    pub column: usize,
    pub row: usize,
}

impl GridSlot {
    /// The canvas position of this slot's origin.
    pub fn position(self) -> (f32, f32) {
        (
            self.column as f32 * COLUMN_SPACING,
            self.row as f32 * ROW_SPACING,
        )
    }
}

/// Compute the auto-placement grid slot of every node in `graph`.
///
/// See the module docs for the rules. Pure topology: it needs no registry, so
/// a graph that does not validate can still be laid out.
pub fn auto_grid(graph: &Graph) -> HashMap<NodeId, GridSlot> {
    let plan = compile(graph);
    let n = plan.components().len();

    // Components come sources-first (a topological order of the condensation),
    // so every predecessor's depth is final by the time we reach a component.
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(from, to) in plan.deps() {
        preds[to].push(from);
    }
    let mut depth = vec![0usize; n];
    for c in 0..n {
        depth[c] = preds[c].iter().map(|&p| depth[p] + 1).max().unwrap_or(0);
    }

    let mut columns: HashMap<usize, Vec<&NodeId>> = HashMap::new();
    for id in graph.nodes.keys() {
        let column = plan.component_of(id).map_or(0, |c| depth[c]);
        columns.entry(column).or_default().push(id);
    }

    let mut grid = HashMap::with_capacity(graph.nodes.len());
    for (column, mut ids) in columns {
        ids.sort_by(|a, b| a.0.cmp(&b.0));
        for (row, id) in ids.into_iter().enumerate() {
            grid.insert(id.clone(), GridSlot { column, row });
        }
    }
    grid
}

/// Node positions on the editor canvas (`NodeId` → `(x, y)`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Layout {
    positions: HashMap<NodeId, (f32, f32)>,
}

impl Layout {
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh layout with every node of `graph` at its [`auto_grid`] slot.
    pub fn auto(graph: &Graph) -> Self {
        Self {
            positions: auto_grid(graph)
                .into_iter()
                .map(|(id, slot)| (id, slot.position()))
                .collect(),
        }
    }

    /// Discard every position (including hand-placed ones) and auto-place the
    /// whole graph again — an editor's "tidy up" action.
    pub fn relayout(&mut self, graph: &Graph) {
        *self = Self::auto(graph);
    }

    pub fn position(&self, node: &NodeId) -> Option<(f32, f32)> {
        self.positions.get(node).copied()
    }

    /// Place `node` at `pos`, returning its previous position.
    pub fn set_position(&mut self, node: NodeId, pos: (f32, f32)) -> Option<(f32, f32)> {
        self.positions.insert(node, pos)
    }

    /// Forget `node`'s position, returning it.
    pub fn remove(&mut self, node: &NodeId) -> Option<(f32, f32)> {
        self.positions.remove(node)
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&NodeId, (f32, f32))> {
        self.positions.iter().map(|(id, pos)| (id, *pos))
    }

    /// Drop positions of nodes no longer in `graph`.
    pub fn retain_graph(&mut self, graph: &Graph) {
        self.positions.retain(|id, _| graph.nodes.contains_key(id));
    }

    /// Auto-place every node of `graph` that has no position yet, leaving
    /// already-placed nodes exactly where they are. Returns the newly placed
    /// ids, sorted.
    ///
    /// A new node goes in its [`auto_grid`] column, in the first row slot of
    /// that column not already occupied (within half a grid cell) by an
    /// existing node. Using the first *free* slot rather than the node's
    /// auto-grid row avoids stacking a new node on top of one that was placed
    /// earlier, since rows shift as nodes are added. On an empty layout this
    /// yields exactly [`Layout::auto`].
    pub fn place_missing(&mut self, graph: &Graph) -> Vec<NodeId> {
        let mut missing: Vec<(GridSlot, NodeId)> = auto_grid(graph)
            .into_iter()
            .filter(|(id, _)| !self.positions.contains_key(id))
            .map(|(id, slot)| (slot, id))
            .collect();
        missing.sort_by(|(sa, a), (sb, b)| {
            (sa.column, sa.row)
                .cmp(&(sb.column, sb.row))
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut placed = Vec::with_capacity(missing.len());
        for (slot, id) in missing {
            let mut row = 0;
            let pos = loop {
                let candidate = GridSlot {
                    column: slot.column,
                    row,
                }
                .position();
                if !self.is_occupied(candidate) {
                    break candidate;
                }
                row += 1;
            };
            self.positions.insert(id.clone(), pos);
            placed.push(id);
        }
        placed.sort_by(|a, b| a.0.cmp(&b.0));
        placed
    }

    fn is_occupied(&self, (x, y): (f32, f32)) -> bool {
        self.positions.values().any(|&(px, py)| {
            (px - x).abs() < COLUMN_SPACING / 2.0 && (py - y).abs() < ROW_SPACING / 2.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{NodeSpec, Params, PortId};

    fn id(s: &str) -> NodeId {
        NodeId(s.to_string())
    }

    fn graph(nodes: &[&str], edges: &[(&str, &str)]) -> Graph {
        let mut g = Graph::new();
        for n in nodes {
            g.add_node(NodeSpec {
                id: id(n),
                kind: "clear_channel".to_string(),
                params: Params::new(),
            })
            .unwrap();
        }
        for (a, b) in edges {
            g.connect(
                (id(a), PortId("out".to_string())),
                (id(b), PortId("in".to_string())),
            )
            .unwrap();
        }
        g
    }

    fn slot(column: usize, row: usize) -> GridSlot {
        GridSlot { column, row }
    }

    #[test]
    fn columns_follow_longest_path_depth() {
        // a -> b -> c, plus a shortcut a -> c: c must still be right of b.
        let g = graph(&["a", "b", "c"], &[("a", "b"), ("b", "c"), ("a", "c")]);
        let grid = auto_grid(&g);
        assert_eq!(grid[&id("a")], slot(0, 0));
        assert_eq!(grid[&id("b")], slot(1, 0));
        assert_eq!(grid[&id("c")], slot(2, 0));
    }

    #[test]
    fn rows_are_sorted_by_id_within_a_column() {
        // Two sources and a fan-out; ids chosen so insertion order != sorted.
        let g = graph(&["z", "m", "src"], &[("src", "z"), ("src", "m")]);
        let grid = auto_grid(&g);
        assert_eq!(grid[&id("src")], slot(0, 0));
        assert_eq!(grid[&id("m")], slot(1, 0));
        assert_eq!(grid[&id("z")], slot(1, 1));
    }

    #[test]
    fn a_feedback_loop_shares_one_column() {
        // src -> {x <-> y} -> sink
        let g = graph(
            &["src", "x", "y", "sink"],
            &[("src", "x"), ("x", "y"), ("y", "x"), ("y", "sink")],
        );
        let grid = auto_grid(&g);
        assert_eq!(grid[&id("src")].column, 0);
        assert_eq!(grid[&id("x")], slot(1, 0));
        assert_eq!(grid[&id("y")], slot(1, 1));
        assert_eq!(grid[&id("sink")].column, 2);
    }

    #[test]
    fn auto_layout_maps_slots_to_positions() {
        let g = graph(&["a", "b"], &[("a", "b")]);
        let layout = Layout::auto(&g);
        assert_eq!(layout.position(&id("a")), Some((0.0, 0.0)));
        assert_eq!(layout.position(&id("b")), Some((COLUMN_SPACING, 0.0)));
    }

    #[test]
    fn place_missing_on_empty_layout_matches_auto() {
        let g = graph(&["a", "b", "c", "d"], &[("a", "b"), ("a", "c"), ("c", "d")]);
        let mut layout = Layout::new();
        let placed = layout.place_missing(&g);
        assert_eq!(placed, vec![id("a"), id("b"), id("c"), id("d")]);
        assert_eq!(layout, Layout::auto(&g));
    }

    #[test]
    fn place_missing_keeps_existing_and_avoids_overlap() {
        // Column 0 holds a and c; the user moved a somewhere else entirely.
        let mut g = graph(&["a", "c"], &[]);
        let mut layout = Layout::auto(&g);
        layout.set_position(id("a"), (999.0, 999.0));
        let c_before = layout.position(&id("c")).unwrap(); // row 1

        // b sorts between a and c, so its auto-grid row (1) is c's slot.
        g.add_node(NodeSpec {
            id: id("b"),
            kind: "clear_channel".to_string(),
            params: Params::new(),
        })
        .unwrap();
        assert_eq!(layout.place_missing(&g), vec![id("b")]);

        assert_eq!(layout.position(&id("a")), Some((999.0, 999.0)));
        assert_eq!(layout.position(&id("c")), Some(c_before));
        // Row 0 was vacated by the move, so b takes it rather than landing on c.
        assert_eq!(layout.position(&id("b")), Some((0.0, 0.0)));

        // Nothing left to place.
        assert!(layout.place_missing(&g).is_empty());
    }

    #[test]
    fn retain_and_relayout() {
        let mut g = graph(&["a", "b"], &[("a", "b")]);
        let mut layout = Layout::auto(&g);
        layout.set_position(id("b"), (5.0, 5.0));

        g.remove_node(&id("a"));
        layout.retain_graph(&g);
        assert_eq!(layout.len(), 1);
        assert_eq!(layout.position(&id("a")), None);

        layout.relayout(&g);
        assert_eq!(layout.position(&id("b")), Some((0.0, 0.0)));
    }
}
