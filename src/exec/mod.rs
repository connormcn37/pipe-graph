//! Execution layer: turns the topology in [`crate::graph`] into something that
//! runs. This module owns the node abstraction (ports + evaluation) and, in
//! later phases, the registry, buffers, and scheduler.

mod node;
pub use self::node::*;
