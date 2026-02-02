use std::collections::HashMap;

#[cfg(feature = "bevy")]
use bevy::prelude::*;

#[derive(Debug)]
#[cfg_attr(feature = "bevy", derive(Component))]
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

#[derive(Debug)]
#[cfg_attr(feature = "bevy", derive(Component))]
pub struct Crop;

#[derive(Debug)]
#[cfg_attr(feature = "bevy", derive(Component))]
pub struct Cast;

#[derive(Debug)]
#[cfg_attr(feature = "bevy", derive(Component))]
pub struct Split;

#[derive(Debug)]
#[cfg_attr(feature = "bevy", derive(Component))]
pub struct Merge;
