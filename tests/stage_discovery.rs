//! Tests for the build-time stage discovery in `build/discover.rs`, driven
//! against throwaway directories instead of the real `src/stages/`.

#[path = "../build/discover.rs"]
mod discover;

use std::fs;
use std::path::{Path, PathBuf};

/// A fresh, empty directory under the system temp dir, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("pipe_graph_discover_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn file(&self, rel: &str) -> &Self {
        let path = self.0.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "").unwrap();
        self
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn names(dir: &Path) -> Vec<String> {
    discover::discover(dir)
        .unwrap()
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

#[test]
fn finds_files_and_mod_rs_folders_sorted() {
    let s = Scratch::new("finds");
    s.file("mod.rs")
        .file("zoom.rs")
        .file("blur/mod.rs")
        .file("blur/kernel.rs");
    assert_eq!(names(s.path()), ["blur", "zoom"]);
}

#[test]
fn ignores_non_stage_files() {
    let s = Scratch::new("ignores");
    s.file("crop.rs")
        .file("NOTES.md")
        .file("scratch.rs.bk")
        .file("kernels/blur.cu");
    assert_eq!(names(s.path()), ["crop"]);
}

#[test]
fn rejects_invalid_module_names() {
    for bad in ["my-stage.rs", "2d_blur.rs", "type.rs", "register_all.rs"] {
        let s = Scratch::new("invalid");
        s.file(bad);
        let err = discover::discover(s.path()).unwrap_err();
        assert!(err.contains(&format!("src/stages/{bad}")), "{err}");
    }
}

#[test]
fn render_declares_and_registers_every_module() {
    let s = Scratch::new("render");
    s.file("blend.rs").file("split.rs");
    let generated = discover::render(&discover::discover(s.path()).unwrap());
    assert!(generated.contains("pub mod blend;"));
    assert!(generated.contains("    blend::register(reg);\n    split::register(reg);"));
}

/// `foo.rs` is loaded via `#[path]`, so its `mod helper;` would resolve to
/// `src/stages/helper.rs`, never `foo/helper.rs`. Say so instead of letting
/// the compiler suggest a file that would itself be taken for a stage.
#[test]
fn rejects_stage_file_beside_folder_of_same_name() {
    let s = Scratch::new("beside");
    s.file("foo.rs").file("foo/helper.rs");
    let err = discover::discover(s.path()).unwrap_err();
    assert!(err.contains("src/stages/foo.rs"), "{err}");
    assert!(err.contains("foo/mod.rs"), "{err}");
}

/// `foo.rs` and `foo/mod.rs` would both generate `pub mod foo;`.
#[test]
fn rejects_stage_file_and_stage_folder_with_same_name() {
    let s = Scratch::new("twice");
    s.file("foo.rs").file("foo/mod.rs");
    let err = discover::discover(s.path()).unwrap_err();
    assert!(err.contains("src/stages/foo.rs"), "{err}");
}
