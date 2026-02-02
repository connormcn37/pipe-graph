#[cfg(feature = "bevy")]
use bevy::prelude::*;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(Component))]
pub struct Object {
    pub id: String,
}
