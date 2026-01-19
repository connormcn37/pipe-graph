use crate::systems::{
    handle_cast_stage, handle_crop_stage, handle_merge_stage, handle_split_stage,
};

pub mod data;
pub mod systems;

fn main() {
    use bevy::prelude::*;

    let mut app = App::new();

    app.add_plugins(DefaultPlugins);

    app.add_systems(Startup, setup);

    app.add_systems(
        Update,
        (
            handle_crop_stage,
            handle_cast_stage,
            handle_split_stage,
            handle_merge_stage,
        )
            .chain(),
    );

    app.run();
}

fn setup(mut commands: bevy::prelude::Commands) {
    commands.spawn((
        data::Object {
            id: "A".to_string(),
        },
        data::Stage {
            parameters: bevy::platform::collections::HashMap::new(),
        },
        data::Crop,
    ));

    commands.spawn((
        data::Object {
            id: "B - won't show up because no Stage attached".to_string(),
        },
        data::Cast,
    ));

    let mut parameters = bevy::platform::collections::HashMap::new();
    parameters.insert("key".to_string(), "value".to_string());
    commands.spawn((
        data::Object {
            id: "C".to_string(),
        },
        data::Stage { parameters },
        data::Split,
    ));
}
