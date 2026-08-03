use std::collections::HashMap;

#[cfg(feature = "bevy")]
use bevy::prelude::*;

/// Legacy ECS scaffold for the Bevy editor experiments (World B). The
/// runtime's frame plumbing now lives on [`crate::exec::EdgeBuffer`]
/// (`push`/`get_last`) rather than on this type; the earlier `get_last_frame`
/// / `push_frame` stubs have moved there. Retained until the Bevy layer is
/// reworked to be a view over the core graph (Phase 8).
#[derive(Debug)]
#[cfg_attr(feature = "bevy", derive(Component))]
pub struct Stage {
    pub parameters: HashMap<String, String>,
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
