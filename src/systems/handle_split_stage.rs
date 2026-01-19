use bevy::prelude::*;

use crate::data::{Object, Split, Stage};

pub fn handle_split_stage(q_objects: Query<(&Object, &Stage), With<Split>>) {
    for (object, stage) in q_objects.iter() {
        println!("SplitStage object {:?} stage {:?}", object, stage);
    }
}
