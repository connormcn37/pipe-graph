use bevy::{platform::collections::HashMap, prelude::*};

#[derive(Component, Debug)]
pub struct Stage {
    pub parameters: HashMap<String, String>,
}

impl Stage {
    pub fn get_last_frame() {
        todo!()
    }

    pub fn push_frame() {
        todo!()
    }
}

#[derive(Component, Debug)]
pub struct Crop;

#[derive(Component, Debug)]
pub struct Cast;

#[derive(Component, Debug)]
pub struct Split;

#[derive(Component, Debug)]
pub struct Merge;
