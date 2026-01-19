use bevy::prelude::*;

use crate::data::{Merge, Object, Stage};

pub fn handle_merge_stage(q_objects: Query<(&Object, &Stage), With<Merge>>) {
    for (object, stage) in q_objects.iter() {
        println!("MergeStage object {:?} stage {:?}", object, stage);
    }
}
