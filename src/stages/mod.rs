//! Concrete pipeline stages, discovered automatically.
//!
//! **Adding a stage:** create `src/stages/<name>.rs` (or `<name>/mod.rs` for a
//! multi-file stage) and give it a `register` fn. There is nothing to list
//! here or in the registry — `build.rs` finds the file on the next build:
//!
//! ```ignore
//! use crate::exec::Registry;
//!
//! pub fn register(reg: &mut Registry) {
//!     // A `Node` that implements `TryFrom<&Params>`:
//!     reg.register_stage::<MyStage>("my_stage");
//!     // Or a single-in/single-out `Processor`:
//!     // reg.register_processor("my_filter", |_params| Ok(MyFilter));
//! }
//! ```
//!
//! The module name must be a snake_case Rust identifier. A stage with
//! submodules must be a `<name>/mod.rs` folder: a `<name>.rs` file is loaded
//! via `#[path]`, so its `mod x;` would resolve to `src/stages/x.rs` (the
//! build rejects `<name>.rs` beside a `<name>/` folder). Registering a kind
//! that another stage already registered panics at startup (and in `cargo
//! test`). Stage files are not reached by `cargo fmt`; format them with the
//! bash or PowerShell command in the README (CI checks them).
//!
//! Split/Merge declare a param-dependent number of ports (`out0..`, `in0..`),
//! which is exactly why [`crate::exec::Node::ports`] takes `&self`.

include!(concat!(env!("OUT_DIR"), "/stages_gen.rs"));
