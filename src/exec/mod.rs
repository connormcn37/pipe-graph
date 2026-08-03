//! Execution layer: turns the topology in [`crate::graph`] into something that
//! runs. This module owns the node abstraction (ports + evaluation), the
//! registry that maps `kind` strings to nodes, and graph validation; later
//! phases add buffers and the scheduler.

mod buffer;
pub use self::buffer::*;

mod node;
pub use self::node::*;

mod registry;
pub use self::registry::*;

mod schedule;
pub use self::schedule::*;

mod tap;
pub use self::tap::*;

mod validate;
pub use self::validate::*;
