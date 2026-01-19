use bevy::prelude::*;

use crate::data::{Cast, Object, Stage};

pub fn handle_cast_stage(q_objects: Query<(&Object, &Stage), With<Cast>>) {
    for (object, stage) in q_objects.iter() {
        println!("CastStage object {:?} stage {:?}", object, stage);
    }
}
