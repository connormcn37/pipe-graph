//! Pure editor geometry: node box sizing, pin placement, hit testing, and the
//! mapping between the session's canvas layout and world space.
//!
//! Nothing here touches the ECS — every function takes plain values (`Vec2`,
//! [`PortSet`]) and returns plain values — so the fiddly geometry is
//! covered by ordinary unit tests, and both the renderer ([`super::render`])
//! and the input handler ([`super::interact`]) agree on where a pin *is* by
//! construction: they call the same functions. It only uses `bevy::math` for
//! `Vec2`, which is why it lives under the `bevy` feature rather than in core.
//!
//! Coordinate convention: world space, +Y up (Bevy's 2D convention). A node's
//! position is the *center* of its box (its `Transform` translation).

use bevy::math::Vec2;

use crate::exec::PortSet;
use crate::graph::PortId;

/// Width of every node box. Fixed so labels line up and pins sit on a
/// predictable edge; height grows with the port count instead.
pub const NODE_WIDTH: f32 = 170.0;
/// Space at the top of a box reserved for the node's title label.
pub const HEADER_HEIGHT: f32 = 28.0;
/// Vertical distance between consecutive pins on one side.
pub const PIN_SPACING: f32 = 22.0;
/// Drawn radius of a pin.
pub const PIN_RADIUS: f32 = 5.0;
/// Pick radius of a pin. Larger than the drawn radius: pins are small targets
/// and a forgiving grab radius makes wiring far less frustrating.
pub const PIN_HIT_RADIUS: f32 = 10.0;
/// How close (in world units) a click must be to an edge line to select it.
pub const EDGE_HIT_DISTANCE: f32 = 6.0;

/// Which side of a node box a pin is on. Inputs are on the left, outputs on
/// the right, so data visually flows left-to-right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PinSide {
    Input,
    Output,
}

impl PinSide {
    /// The side a wire started on this side must end on.
    pub fn opposite(self) -> Self {
        match self {
            PinSide::Input => PinSide::Output,
            PinSide::Output => PinSide::Input,
        }
    }
}

/// Size of a node's box for the given ports. Always at least one pin row tall
/// so a node with no ports (e.g. an unknown kind) still gets a clickable box.
pub fn node_size(ports: &PortSet) -> Vec2 {
    let rows = ports.inputs.len().max(ports.outputs.len()).max(1);
    Vec2::new(NODE_WIDTH, HEADER_HEIGHT + rows as f32 * PIN_SPACING)
}

/// Offset of pin `index` on `side` from the node's center.
pub fn pin_offset(ports: &PortSet, side: PinSide, index: usize) -> Vec2 {
    let size = node_size(ports);
    let x = match side {
        PinSide::Input => -size.x / 2.0,
        PinSide::Output => size.x / 2.0,
    };
    let rows_top = size.y / 2.0 - HEADER_HEIGHT;
    let y = rows_top - (index as f32 + 0.5) * PIN_SPACING;
    Vec2::new(x, y)
}

/// The ports on one side of a [`PortSet`], in declaration order.
pub fn side_ports(ports: &PortSet, side: PinSide) -> impl Iterator<Item = &PortId> {
    let list = match side {
        PinSide::Input => &ports.inputs,
        PinSide::Output => &ports.outputs,
    };
    list.iter().map(|p| &p.id)
}

/// Every pin of a node as `(side, port, world position)`, given its center.
pub fn pin_positions(center: Vec2, ports: &PortSet) -> Vec<(PinSide, &PortId, Vec2)> {
    let mut out = Vec::with_capacity(ports.inputs.len() + ports.outputs.len());
    for side in [PinSide::Input, PinSide::Output] {
        for (i, id) in side_ports(ports, side).enumerate() {
            out.push((side, id, center + pin_offset(ports, side, i)));
        }
    }
    out
}

/// Where an edge attaches to a node: the named pin if the node declares it,
/// otherwise the middle of that side of the box.
///
/// The fallback matters because the graph is allowed to reference ports that
/// the registry does not know (unknown node kinds, or params that fail to
/// build); such edges must still be drawable and clickable, not dropped.
pub fn port_anchor(center: Vec2, ports: &PortSet, side: PinSide, port: &str) -> Vec2 {
    match side_ports(ports, side).position(|p| p.0 == port) {
        Some(i) => center + pin_offset(ports, side, i),
        None => {
            let half = node_size(ports).x / 2.0;
            let x = match side {
                PinSide::Input => -half,
                PinSide::Output => half,
            };
            center + Vec2::new(x, 0.0)
        }
    }
}

/// Whether `point` lies inside (or on the border of) the axis-aligned rect of
/// `size` centered at `center`.
pub fn point_in_rect(point: Vec2, center: Vec2, size: Vec2) -> bool {
    let d = (point - center).abs();
    d.x <= size.x / 2.0 && d.y <= size.y / 2.0
}

/// The pin of a node (centered at `center`) under `point`, if any. When pick
/// circles overlap, the nearest pin wins.
pub fn pin_at(point: Vec2, center: Vec2, ports: &PortSet) -> Option<(PinSide, PortId)> {
    pin_positions(center, ports)
        .into_iter()
        .map(|(side, id, pos)| (side, id, pos.distance(point)))
        .filter(|(_, _, d)| *d <= PIN_HIT_RADIUS)
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .map(|(side, id, _)| (side, id.clone()))
}

/// Shortest distance from `p` to the segment `a`–`b`. Used to pick edges.
pub fn distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 == 0.0 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Map a position in the session's canvas [`Layout`](crate::editor::Layout)
/// (+Y *down*, rows grow downward like a page) to world space (+Y up).
///
/// The layout is the editor-independent record of where nodes sit; this is
/// the one place the two conventions meet, so the auto-placed grid reads
/// top-to-bottom on screen exactly as it does on the canvas.
pub fn canvas_to_world((x, y): (f32, f32)) -> Vec2 {
    Vec2::new(x, -y)
}

/// Inverse of [`canvas_to_world`].
pub fn world_to_canvas(world: Vec2) -> (f32, f32) {
    (world.x, -world.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::PayloadKind;
    use crate::exec::PortSpec;

    fn ports(inputs: &[&str], outputs: &[&str]) -> PortSet {
        PortSet::new(
            inputs
                .iter()
                .map(|p| PortSpec::new(*p, PayloadKind::Frame))
                .collect(),
            outputs
                .iter()
                .map(|p| PortSpec::new(*p, PayloadKind::Frame))
                .collect(),
        )
    }

    #[test]
    fn size_grows_with_the_busier_side_and_has_a_floor() {
        let empty = node_size(&PortSet::default());
        assert_eq!(empty, Vec2::new(NODE_WIDTH, HEADER_HEIGHT + PIN_SPACING));

        let three_out = node_size(&ports(&["in"], &["a", "b", "c"]));
        assert_eq!(three_out.y, HEADER_HEIGHT + 3.0 * PIN_SPACING);
    }

    #[test]
    fn pins_sit_on_box_edges_below_the_header() {
        let p = ports(&["in"], &["o0", "o1"]);
        let size = node_size(&p);

        let input = pin_offset(&p, PinSide::Input, 0);
        assert_eq!(input.x, -size.x / 2.0);

        let o0 = pin_offset(&p, PinSide::Output, 0);
        let o1 = pin_offset(&p, PinSide::Output, 1);
        assert_eq!(o0.x, size.x / 2.0);
        // Pins stack downward and stay inside the box, under the header.
        assert!(o0.y > o1.y);
        assert!(o0.y < size.y / 2.0 - HEADER_HEIGHT);
        assert!(o1.y > -size.y / 2.0);
    }

    #[test]
    fn pin_positions_lists_every_pin_once() {
        let p = ports(&["a", "b"], &["out"]);
        let pins = pin_positions(Vec2::new(10.0, 20.0), &p);
        assert_eq!(pins.len(), 3);
        assert_eq!(pins[0].0, PinSide::Input);
        assert_eq!(pins[2].0, PinSide::Output);
        assert_eq!(
            pins[2].2,
            Vec2::new(10.0, 20.0) + pin_offset(&p, PinSide::Output, 0)
        );
    }

    #[test]
    fn point_in_rect_includes_border_and_excludes_outside() {
        let c = Vec2::new(100.0, 50.0);
        let s = Vec2::new(20.0, 10.0);
        assert!(point_in_rect(c, c, s));
        assert!(point_in_rect(Vec2::new(110.0, 55.0), c, s));
        assert!(!point_in_rect(Vec2::new(110.1, 50.0), c, s));
        assert!(!point_in_rect(Vec2::new(100.0, 44.0), c, s));
    }

    #[test]
    fn pin_at_finds_the_pin_under_the_cursor() {
        let p = ports(&["in"], &["out0", "out1"]);
        let c = Vec2::ZERO;
        let out1 = c + pin_offset(&p, PinSide::Output, 1);

        assert_eq!(
            pin_at(out1 + Vec2::new(3.0, -2.0), c, &p),
            Some((PinSide::Output, PortId("out1".to_string())))
        );
        let input = c + pin_offset(&p, PinSide::Input, 0);
        assert_eq!(
            pin_at(input, c, &p),
            Some((PinSide::Input, PortId("in".to_string())))
        );
        // The box center is not a pin.
        assert_eq!(pin_at(c, c, &p), None);
        // A node with no ports has no pins at all.
        assert_eq!(pin_at(input, c, &PortSet::default()), None);
    }

    #[test]
    fn port_anchor_falls_back_to_side_midpoint_for_unknown_ports() {
        let p = ports(&["in"], &["out"]);
        let c = Vec2::new(5.0, 5.0);
        assert_eq!(
            port_anchor(c, &p, PinSide::Output, "out"),
            c + pin_offset(&p, PinSide::Output, 0)
        );
        assert_eq!(
            port_anchor(c, &p, PinSide::Input, "nope"),
            c + Vec2::new(-NODE_WIDTH / 2.0, 0.0)
        );
    }

    #[test]
    fn segment_distance() {
        let a = Vec2::ZERO;
        let b = Vec2::new(10.0, 0.0);
        assert_eq!(distance_to_segment(Vec2::new(5.0, 3.0), a, b), 3.0);
        // Beyond an endpoint, distance is to that endpoint.
        assert_eq!(distance_to_segment(Vec2::new(13.0, 4.0), a, b), 5.0);
        // Degenerate segment.
        assert_eq!(distance_to_segment(Vec2::new(3.0, 4.0), a, a), 5.0);
    }

    #[test]
    fn canvas_and_world_round_trip_with_y_flipped() {
        assert_eq!(canvas_to_world((220.0, 120.0)), Vec2::new(220.0, -120.0));
        assert_eq!(world_to_canvas(Vec2::new(220.0, -120.0)), (220.0, 120.0));
        let p = (13.5, -7.25);
        assert_eq!(world_to_canvas(canvas_to_world(p)), p);
    }
}
