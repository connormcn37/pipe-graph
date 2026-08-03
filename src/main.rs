use pipe_graph::{data, processors, traits::Processor};

fn main() {
    let mut frame = data::Frame::from_rgb8(
        3,
        3,
        vec![
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (192, 192, 192),
            (128, 128, 128),
            (64, 64, 64),
        ],
    );

    println!("Original Frame: {:?}", frame);
    let mut processor_list = processors::ProcessList::new();
    processor_list.add_processor(processors::ClearChannel(processors::Channel::Red));
    processor_list.add_processor(processors::ClearChannel(processors::Channel::Green));

    let mut processor_list2 = processors::ProcessList::new();
    processor_list2.add_processor(processor_list);
    processor_list2.add_processor(processors::ClearChannel(processors::Channel::Blue));

    processor_list2.process(&mut frame);
    println!("Processed Frame: {:?}", frame);

    graph_runtime_demo();

    // use bevy::prelude::*;

    // let mut app = App::new();

    // app.add_plugins(DefaultPlugins);

    // app.add_systems(Startup, setup);

    // app.add_systems(
    //     Update,
    //     (
    //         handle_crop_stage,
    //         handle_cast_stage,
    //         handle_split_stage,
    //         handle_merge_stage,
    //     )
    //         .chain(),
    // );

    // app.run();
}

/// Demonstrates the node-graph runtime: describe a graph declaratively, then
/// compile and execute it. Here a 2x2x3 frame is split into its three channels
/// and merged back, round-tripping to the original.
fn graph_runtime_demo() {
    use pipe_graph::data::{Frame, FrameData, Payload};
    use pipe_graph::exec::{Runtime, builtin_registry};
    use pipe_graph::graph::{Graph, NodeId, NodeSpec, Params, PortId};

    fn node(id: &str, kind: &str, params: &[(&str, &str)]) -> NodeSpec {
        let mut p = Params::new();
        for (k, v) in params {
            p.insert(k.to_string(), v.to_string());
        }
        NodeSpec {
            id: NodeId(id.to_string()),
            kind: kind.to_string(),
            params: p,
        }
    }
    let port = |n: &str, p: &str| (NodeId(n.to_string()), PortId(p.to_string()));

    let mut graph = Graph::new();
    graph
        .add_node(node("split", "split", &[("channels", "3")]))
        .unwrap();
    graph
        .add_node(node("merge", "merge", &[("channels", "3")]))
        .unwrap();
    graph
        .connect(port("split", "out0"), port("merge", "in0"))
        .unwrap();
    graph
        .connect(port("split", "out1"), port("merge", "in1"))
        .unwrap();
    graph
        .connect(port("split", "out2"), port("merge", "in2"))
        .unwrap();

    let source = Frame::from_data(
        2,
        2,
        3,
        FrameData::U8(vec![1, 10, 100, 2, 20, 101, 3, 30, 102, 4, 40, 103]),
    );

    let registry = builtin_registry();
    let mut runtime = Runtime::instantiate(&graph, &registry).expect("valid graph");
    runtime.set_input(
        &NodeId("split".to_string()),
        "in",
        Payload::Frame(source.clone()),
    );
    runtime.run_once().expect("run");

    let output = runtime
        .output(&NodeId("merge".to_string()), "out")
        .and_then(Payload::as_frame)
        .expect("merge produced a frame");

    println!("\nGraph runtime demo (split -> merge):");
    println!("  input : {:?}", source.data());
    println!("  output: {:?}", output.data());
    println!("  round-trip ok: {}", output == &source);
}

#[cfg(feature = "bevy")]
#[allow(dead_code)] // wired up in Phase 8 (Bevy editor); kept as scaffold
fn setup(mut commands: bevy::prelude::Commands) {
    commands.spawn((
        data::Object {
            id: "A".to_string(),
        },
        data::Stage {
            parameters: std::collections::HashMap::new(),
        },
        data::Crop,
    ));

    commands.spawn((
        data::Object {
            id: "B - won't show up because no Stage attached".to_string(),
        },
        data::Cast,
    ));

    let mut parameters = std::collections::HashMap::new();
    parameters.insert("key".to_string(), "value".to_string());
    commands.spawn((
        data::Object {
            id: "C".to_string(),
        },
        data::Stage { parameters },
        data::Split,
    ));
}
