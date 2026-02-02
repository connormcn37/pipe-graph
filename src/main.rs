use pipe_graph::{data, processors, traits::Processor};

fn main() {
    let mut frame = data::Frame {
        width: 3,
        height: 3,
        pixels: vec![
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
    };

    println!("Original Frame: {:?}", frame);
    let mut processor_list = processors::ProcessList::new();
    processor_list.add_processor(processors::ClearChannel(processors::Channel::Red));
    processor_list.add_processor(processors::ClearChannel(processors::Channel::Green));

    let mut processor_list2 = processors::ProcessList::new();
    processor_list2.add_processor(processor_list);
    processor_list2.add_processor(processors::ClearChannel(processors::Channel::Blue));

    processor_list2.process(&mut frame);
    println!("Processed Frame: {:?}", frame);

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

#[cfg(feature = "bevy")]
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
