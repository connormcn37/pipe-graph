//! Stage discovery.
//!
//! Every module in `src/stages/` — a `name.rs` file, or a `name/` directory
//! holding a `mod.rs` — is compiled and registered without being listed
//! anywhere. This script only *lists* files (see `build/discover.rs`); it
//! writes `$OUT_DIR/stages_gen.rs`, which `src/stages/mod.rs` includes:
//!
//! ```ignore
//! #[path = "/abs/path/src/stages/blend.rs"]
//! pub mod blend;
//! pub fn register_all(reg: &mut crate::exec::Registry) { blend::register(reg); }
//! ```
//!
//! Each stage module must define `pub fn register(reg: &mut Registry)`; the
//! compiler reports a module that does not.

#[path = "build/discover.rs"]
mod discover;

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let stages_dir = manifest.join("src").join("stages");
    // A directory path makes Cargo rescan everything under it, so adding or
    // deleting a stage file reruns this script. Listing any path disables
    // Cargo's default "rerun on any change", so every input is named here.
    println!("cargo:rerun-if-changed=src/stages");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build/discover.rs");

    let modules = discover::discover(&stages_dir).unwrap_or_else(|e| panic!("{e}"));
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("stages_gen.rs");
    fs::write(&out, discover::render(&modules))
        .unwrap_or_else(|e| panic!("writing {}: {e}", out.display()));
}
