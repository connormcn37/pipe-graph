//! Minimal TapRegistry demo.
//!
//! Run:
//!   cargo run --bin tap_demo

use pipe_graph::graph::{NodeId, PortId};
use pipe_graph::tap::TapRegistry;

fn main() {
    let taps: TapRegistry<String> = TapRegistry::new();

    // A producer publishes values.
    let out = taps.tap(NodeId("node_a".into()), PortId("out".into()));
    out.publish("hello".to_string());

    // Somewhere else, a consumer retrieves the same tap point.
    let out2 = taps.tap(NodeId("node_a".into()), PortId("out".into()));
    let v = out2.latest().map(|x| x.as_str().to_string());

    println!("latest: {v:?}");
    println!("points: {:?}", taps.points());
}
