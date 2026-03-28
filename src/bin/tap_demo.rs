//! Minimal TapRegistry demo.
//!
//! Run:
//!   cargo run --bin tap_demo

use pipe_graph::graph::{NodeId, PortId};
use pipe_graph::tap::TapRegistry;

fn main() {
    let taps: TapRegistry<String> = TapRegistry::new();

    // A producer publishes values to a node+port tap.
    let out = taps.tap(NodeId("node_a".into()), PortId("out".into()));
    let seq1 = out.publish("hello".to_string());

    // Somewhere else, a consumer retrieves the same tap point.
    let out2 = taps.tap(NodeId("node_a".into()), PortId("out".into()));
    let (seq2, v) = {
        let (s, opt) = out2.latest_with_seq();
        let v = opt
            .as_deref()
            .map(|x| x.as_str().to_string())
            .unwrap_or("<empty>".to_string());
        (s, v)
    };

    // Edge-level taps are also supported.
    let edge_tap = taps.tap_edge(
        NodeId("node_a".into()),
        PortId("out".into()),
        NodeId("node_b".into()),
        PortId("in".into()),
    );
    edge_tap.publish("edge-preview".to_string());

    println!("node tap publish seq: {seq1}");
    println!("node tap latest_with_seq: ({seq2}, {v:?})");
    println!("points (sorted):");
    for p in taps.points_sorted() {
        println!("  - {p}");
    }
}
