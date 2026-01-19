use bevy::prelude::*;

use crate::data::{Crop, Object, Stage};

pub fn handle_crop_stage(q_objects: Query<(&Object, &Stage), With<Crop>>) {
    for (object, stage) in q_objects.iter() {
        println!("CropStage object {:?} stage {:?}", object, stage);
    }
}
