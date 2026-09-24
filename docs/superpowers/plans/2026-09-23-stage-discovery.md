# Stage Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adding a node kind means creating one file in `src/stages/` and rebuilding — no edits to `stages/mod.rs`, `exec/registry.rs`, or any other list.

**Architecture:** A std-only `build.rs` lists `src/stages/` and writes `$OUT_DIR/stages_gen.rs` containing one `#[path] pub mod` per stage file plus a `register_all(&mut Registry)` that calls each module's `pub fn register(reg: &mut Registry)`. `stages/mod.rs` is a static `include!` of that file. Two `Registry` helpers (`register_stage`, `register_processor`) make each file's `register` a one-liner and panic on duplicate kinds.

**Tech Stack:** Rust 1.94 (edition 2024), Cargo build scripts, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-stage-discovery-design.md`

## Global Constraints

- No new crates in `[dependencies]` or `[build-dependencies]`; `build.rs` uses `std` only.
- Every existing kind keeps its exact name and params: `blend`, `cast`, `clear_channel`, `crop`, `grayscale`, `image_read`, `image_write`, `invert`, `merge`, `split`.
- Duplicate kinds registered through `register_stage` / `register_processor` panic with the message `node kind '<kind>' registered twice`. `Registry::register` keeps replace-on-duplicate.
- `build.rs` never parses Rust; it only lists files.
- Both CI legs pass: `cargo fmt --all --check`, `cargo clippy --all-targets <features> -- -D warnings`, `cargo test <features>` for `--no-default-features` and `--features bevy`.
- Work in the worktree `C:\code\pipe-graph\.claude\worktrees\cleanup` on branch `cleanup`. Set `CARGO_TARGET_DIR=C:/code/pipe-graph/target` in every shell to reuse the build cache (the bevy leg is otherwise a full rebuild).
- Commit message trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **`cargo fmt` stops seeing stage files.** rustfmt only follows out-of-line `mod` declarations, and stage modules are now declared inside an `include!`d generated file — so `cargo fmt` and CI's fmt check would silently skip `src/stages/`. Expected: CI still fails on a misformatted stage file. Pinned in Task 2 (Steps 8–9).
2. **A file name that is not a Rust identifier** (`my-stage.rs`, `2d_blur.rs`, `type.rs`). Expected: the build fails with a message naming the file, not a cryptic error inside generated code. Pinned in Task 4 (Step 3).
3. **Non-stage files in the folder** (a `README.md`, rustfmt's `foo.rs.bk` backups, a future `kernels/` folder of GPU sources with no `mod.rs`). Expected: ignored, build succeeds. Pinned in Task 4 (Step 4).
4. **Adding or deleting a stage file without touching anything else.** Expected: the next `cargo build` picks up / drops the kind (the build script reruns because `src/stages` is watched). Pinned in Task 4 (Step 2).
5. **Two stage files claiming the same kind.** Expected: startup panic naming the kind, caught by `cargo test`. Pinned in Task 1 (unit tests) and Task 3 (Step 3, which deliberately trips it during the move).

---

## File map

| File | Change | Responsibility |
|---|---|---|
| `src/exec/registry.rs` | modify | `register_stage` / `register_processor` helpers; `builtin_registry()` delegates to `stages::register_all` |
| `build.rs` | create | discover `src/stages/` modules, write `$OUT_DIR/stages_gen.rs` |
| `src/stages/mod.rs` | rewrite | docs + `include!` of the generated file; nothing else |
| `src/stages/{crop,cast,split,merge,blend,image_io}.rs` | modify | each gains `pub fn register(reg: &mut Registry)` |
| `src/stages/{clear_channel,grayscale,invert}.rs` | move from `src/processors/` | same, via `register_processor` |
| `src/processors/mod.rs` | modify | keeps only `ProcessList` |
| `src/processors/process_list.rs`, `src/exec/node.rs` | modify | test imports follow `ClearChannel`'s move |
| `tests/pipeline_execution.rs` | modify | receives the split/merge unit round-trip test |
| `.github/workflows/ci.yml` | modify | extra rustfmt step over `src/stages/` |
| `README.md` | modify | "Adding a stage" section |

---

### Task 1: Registry helpers with duplicate detection

**Files:**
- Modify: `src/exec/registry.rs` (imports at lines 8–13; `impl Registry` block; `mod tests`)

**Interfaces:**
- Consumes: existing `Registry::register`, `Registry::contains`, `ProcessorNode::new`, trait `crate::traits::Processor`.
- Produces:
  - `pub fn register_stage<T>(&mut self, kind: &str) where T: Node + for<'a> TryFrom<&'a Params, Error = BuildError> + 'static`
  - `pub fn register_processor<P, F>(&mut self, kind: &str, ctor: F) where P: Processor + 'static, F: Fn(&Params) -> Result<P, BuildError> + 'static`
  - Both panic with `node kind '{kind}' registered twice` if `kind` is already present.

- [ ] **Step 1: Write the failing tests**

In `src/exec/registry.rs`, replace the test module's import lines

```rust
    use super::*;
    use crate::data::{Frame, Payload};
    use crate::exec::{Inputs, Outputs};
    use crate::graph::{NodeId, PortId};
```

with

```rust
    use super::*;
    use crate::data::{Frame, FrameData, Payload, PayloadKind};
    use crate::exec::{Inputs, NodeError, Outputs, PortSpec};
    use crate::graph::{NodeId, PortId};
    use crate::traits::Processor;

    /// Minimal param-driven stage: `n` output ports, no behaviour.
    struct Probe(u32);

    impl TryFrom<&Params> for Probe {
        type Error = BuildError;
        fn try_from(p: &Params) -> Result<Self, BuildError> {
            Ok(Probe(p.get_u32_or("n", 0)?))
        }
    }

    impl Node for Probe {
        fn ports(&self) -> PortSet {
            let outs = (0..self.0)
                .map(|i| PortSpec::new(format!("out{i}"), PayloadKind::Frame))
                .collect();
            PortSet::new(vec![], outs)
        }
        fn eval(&mut self, _: &Inputs, _: &mut Outputs) -> Result<(), NodeError> {
            Ok(())
        }
    }

    /// Minimal processor: adds 1 to every u8 sample.
    struct AddOne;

    impl Processor for AddOne {
        fn process(&self, f: &mut Frame) {
            if let FrameData::U8(buf) = f.data_mut() {
                for v in buf {
                    *v += 1;
                }
            }
        }
    }
```

and append these tests at the end of `mod tests`:

```rust
    #[test]
    fn register_stage_builds_via_try_from() {
        let mut reg = Registry::new();
        reg.register_stage::<Probe>("probe");
        let ports = reg.ports_of(&spec("probe", &[("n", "2")])).unwrap();
        assert_eq!(ports.outputs.len(), 2);
        let err = reg.build(&spec("probe", &[("n", "x")])).err().unwrap();
        assert!(matches!(err, BuildError::BadParam { .. }));
    }

    #[test]
    fn register_processor_wraps_in_processor_node() {
        let mut reg = Registry::new();
        reg.register_processor("add_one", |_| Ok(AddOne));
        let mut node = reg.build(&spec("add_one", &[])).unwrap();

        let mut m = HashMap::new();
        m.insert(
            PortId("in".to_string()),
            Payload::Frame(Frame::from_rgb8(1, 1, vec![(1, 2, 3)])),
        );
        let mut out = Outputs::new();
        node.eval(&Inputs::new(m), &mut out).unwrap();
        assert_eq!(
            out.get("out").unwrap().as_frame().unwrap().to_rgb8(),
            vec![(2, 3, 4)]
        );
    }

    #[test]
    #[should_panic(expected = "node kind 'probe' registered twice")]
    fn register_stage_rejects_duplicate_kind() {
        let mut reg = Registry::new();
        reg.register_stage::<Probe>("probe");
        reg.register_stage::<Probe>("probe");
    }

    #[test]
    #[should_panic(expected = "node kind 'probe' registered twice")]
    fn register_processor_rejects_kind_taken_by_a_stage() {
        let mut reg = Registry::new();
        reg.register_stage::<Probe>("probe");
        reg.register_processor("probe", |_| Ok(AddOne));
    }

    #[test]
    fn plain_register_still_replaces() {
        let mut reg = Registry::new();
        reg.register_stage::<Probe>("probe");
        reg.register("probe", |_| Ok(Box::new(Probe(5)) as Box<dyn Node>));
        let ports = reg.ports_of(&spec("probe", &[])).unwrap();
        assert_eq!(ports.outputs.len(), 5);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib exec::registry`
Expected: compile error `no method named register_stage found for struct Registry` (and `register_processor`).

- [ ] **Step 3: Implement the helpers**

Add to the imports at the top of `src/exec/registry.rs`:

```rust
use crate::traits::Processor;
```

Add inside `impl Registry`, directly after `fn register`:

```rust
    /// Register a stage built from its params via `TryFrom<&Params>`.
    ///
    /// This is what a stage file's `register` fn normally calls. Unlike
    /// [`Registry::register`], registering an existing kind panics: two stage
    /// files claiming one kind is a bug, not an override.
    pub fn register_stage<T>(&mut self, kind: &str)
    where
        T: Node + for<'a> TryFrom<&'a Params, Error = BuildError> + 'static,
    {
        self.register_new(kind, |p| Ok(Box::new(T::try_from(p)?) as Box<dyn Node>));
    }

    /// Register a [`Processor`] as a 1-in/1-out node (ports `in` → `out`).
    ///
    /// `ctor` parses params into the processor. Panics on a duplicate kind,
    /// like [`Registry::register_stage`].
    pub fn register_processor<P, F>(&mut self, kind: &str, ctor: F)
    where
        P: Processor + 'static,
        F: Fn(&Params) -> Result<P, BuildError> + 'static,
    {
        self.register_new(kind, move |p| {
            Ok(Box::new(ProcessorNode::new(ctor(p)?)) as Box<dyn Node>)
        });
    }

    fn register_new<F>(&mut self, kind: &str, ctor: F)
    where
        F: Fn(&Params) -> Result<Box<dyn Node>, BuildError> + 'static,
    {
        assert!(!self.contains(kind), "node kind '{kind}' registered twice");
        self.register(kind, ctor);
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo fmt --all && cargo test --lib exec::registry`
Expected: all registry tests PASS (the 5 existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add src/exec/registry.rs
git commit -m "registry: add register_stage/register_processor with duplicate-kind panic"
```

---

### Task 2: build.rs discovery for the existing stage files

**Files:**
- Create: `build.rs`
- Rewrite: `src/stages/mod.rs`
- Modify: `src/stages/{crop,cast,split,merge,blend,image_io}.rs` (exec import line + new `register` fn)
- Modify: `src/exec/registry.rs` (`builtin_registry`, imports, tests)
- Modify: `tests/pipeline_execution.rs` (receives moved test)
- Modify: `.github/workflows/ci.yml`, spec Risks section

**Interfaces:**
- Consumes: `Registry::register_stage`, `Registry::register_processor` (Task 1).
- Produces:
  - `crate::stages::register_all(reg: &mut crate::exec::Registry)` — generated.
  - Public modules `crate::stages::{blend, cast, crop, image_io, merge, split}`; types are reached as `stages::split::SplitStage` etc. The flat re-exports (`stages::SplitStage`) are removed.
  - Each stage module: `pub fn register(reg: &mut Registry)`.

- [ ] **Step 1: Write the failing tests**

In `src/exec/registry.rs` `mod tests`, append:

```rust
    #[test]
    fn builtin_registry_has_every_builtin_kind() {
        let reg = builtin_registry();
        let mut kinds: Vec<&str> = reg.registered_kinds().collect();
        kinds.sort_unstable();
        assert_eq!(
            kinds,
            [
                "blend",
                "cast",
                "clear_channel",
                "crop",
                "grayscale",
                "image_read",
                "image_write",
                "invert",
                "merge",
                "split",
            ]
        );
    }

    #[test]
    fn register_all_covers_the_stage_folder() {
        let mut reg = Registry::new();
        crate::stages::register_all(&mut reg);
        let mut kinds: Vec<&str> = reg.registered_kinds().collect();
        kinds.sort_unstable();
        assert_eq!(
            kinds,
            ["blend", "cast", "crop", "image_read", "image_write", "merge", "split"]
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib exec::registry`
Expected: compile error `cannot find function register_all in module crate::stages`.

- [ ] **Step 3: Create `build.rs` at the crate root**

```rust
//! Stage discovery.
//!
//! Every module in `src/stages/` — a `name.rs` file, or a `name/` directory
//! holding a `mod.rs` — is compiled and registered without being listed
//! anywhere. This script only *lists* files; it writes
//! `$OUT_DIR/stages_gen.rs`, which `src/stages/mod.rs` includes:
//!
//! ```ignore
//! #[path = "/abs/path/src/stages/blend.rs"]
//! pub mod blend;
//! pub fn register_all(reg: &mut crate::exec::Registry) { blend::register(reg); }
//! ```
//!
//! Each stage module must define `pub fn register(reg: &mut Registry)`; the
//! compiler reports a module that does not.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::{env, fs};

/// Names a stage module cannot take: Rust keywords (strict, reserved and
/// edition-2024 weak ones that break `mod x;`), plus the item the generated
/// file defines itself.
const RESERVED: &[&str] = &[
    "abstract", "as", "async", "await", "become", "box", "break", "const", "continue", "crate",
    "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if", "impl",
    "in", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub",
    "ref", "return", "self", "static", "struct", "super", "trait", "true", "try", "type",
    "typeof", "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
    "register_all",
];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let stages_dir = manifest.join("src").join("stages");
    // A directory path makes Cargo rescan everything under it, so adding or
    // deleting a stage file reruns this script.
    println!("cargo:rerun-if-changed=src/stages");
    println!("cargo:rerun-if-changed=build.rs");

    let modules = discover(&stages_dir);
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("stages_gen.rs");
    fs::write(&out, render(&modules))
        .unwrap_or_else(|e| panic!("writing {}: {e}", out.display()));
}

/// `(module name, source file)` for every stage module, sorted by name.
fn discover(dir: &Path) -> Vec<(String, PathBuf)> {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()));
    let mut modules = Vec::new();
    for entry in entries {
        let path = entry.unwrap().path();
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        let (name, source) = if path.is_dir() {
            let source = path.join("mod.rs");
            if !source.is_file() {
                continue; // not a module, e.g. a folder of shaders or assets
            }
            (file_name.clone(), source)
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            if stem == "mod" {
                continue;
            }
            (stem, path)
        } else {
            continue; // README, rustfmt `.rs.bk` backups, ...
        };
        if let Err(why) = check_name(&name) {
            panic!("src/stages/{file_name}: {why}; rename it to a snake_case Rust identifier");
        }
        modules.push((name, source));
    }
    modules.sort();
    modules
}

fn check_name(name: &str) -> Result<(), String> {
    let mut chars = name.chars();
    let first_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    if !first_ok || name == "_" || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(format!("'{name}' is not a valid module name"));
    }
    if RESERVED.contains(&name) {
        return Err(format!("'{name}' is a reserved name"));
    }
    Ok(())
}

fn render(modules: &[(String, PathBuf)]) -> String {
    let mut s = String::from("// @generated by build.rs from the contents of src/stages/. Do not edit.\n\n");
    for (name, source) in modules {
        // `{:?}` emits a valid Rust string literal, escaping Windows backslashes.
        writeln!(s, "#[path = {:?}]\npub mod {name};", source.display().to_string()).unwrap();
    }
    s.push_str("\n/// Register every stage module's node kinds. Generated by `build.rs`.\n");
    s.push_str("pub fn register_all(reg: &mut crate::exec::Registry) {\n");
    for (name, _) in modules {
        writeln!(s, "    {name}::register(reg);").unwrap();
    }
    s.push_str("}\n");
    s
}
```

- [ ] **Step 4: Rewrite `src/stages/mod.rs`**

Replace the whole file (this deletes the flat `pub use` re-exports, the trailing `mod blend; pub use self::blend::*;`, and the split/merge test, which moves in Step 6):

```rust
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
//! The module name must be a snake_case Rust identifier. Registering a kind
//! that another stage already registered panics at startup (and in `cargo
//! test`). Stage files are not reached by `cargo fmt`; run
//! `rustfmt --edition 2024 src/stages/*.rs` (CI checks it).
//!
//! Split/Merge declare a param-dependent number of ports (`out0..`, `in0..`),
//! which is exactly why [`crate::exec::Node::ports`] takes `&self`.

include!(concat!(env!("OUT_DIR"), "/stages_gen.rs"));
```

- [ ] **Step 5: Add a `register` fn to each existing stage file**

In each file, add `Registry` to the `use crate::exec::{...}` line (keep the rest of the list), and insert the `register` fn immediately above the file's `#[cfg(test)]` line.

`src/stages/crop.rs` — import becomes `use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec, Registry};`

```rust
pub fn register(reg: &mut Registry) {
    reg.register_stage::<CropStage>("crop");
}
```

`src/stages/cast.rs` — same import change.

```rust
pub fn register(reg: &mut Registry) {
    reg.register_stage::<CastStage>("cast");
}
```

`src/stages/split.rs` — same import change.

```rust
pub fn register(reg: &mut Registry) {
    reg.register_stage::<SplitStage>("split");
}
```

`src/stages/merge.rs` — same import change.

```rust
pub fn register(reg: &mut Registry) {
    reg.register_stage::<MergeStage>("merge");
}
```

`src/stages/blend.rs` — import becomes `use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, PortSet, PortSpec, Registry};`

```rust
pub fn register(reg: &mut Registry) {
    reg.register_stage::<BlendStage>("blend");
}
```

`src/stages/image_io.rs` — import becomes `use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec, Registry};`

```rust
pub fn register(reg: &mut Registry) {
    reg.register_stage::<ImageReadStage>("image_read");
    reg.register_stage::<ImageWriteStage>("image_write");
}
```

- [ ] **Step 6: Point `builtin_registry` at `register_all`; move the round-trip test**

In `src/exec/registry.rs`, replace the two import lines

```rust
use crate::processors::{Channel, ClearChannel, Grayscale, Invert};
use crate::stages::{BlendStage, CastStage, CropStage, MergeStage, SplitStage, ImageReadStage, ImageWriteStage};
```

(rustfmt may have wrapped the second one) with

```rust
use crate::processors::{Channel, ClearChannel, Grayscale, Invert};
```

and replace the whole `builtin_registry` fn (doc comment included) with:

```rust
/// A registry preloaded with the built-in stage kinds: everything in
/// `src/stages/` (discovered by `build.rs`) plus the processor kinds.
pub fn builtin_registry() -> Registry {
    let mut reg = Registry::new();
    crate::stages::register_all(&mut reg);

    reg.register_processor("clear_channel", |p| {
        Ok(ClearChannel(match p.get_str("channel")? {
            "red" => Channel::Red,
            "green" => Channel::Green,
            "blue" => Channel::Blue,
            other => {
                return Err(BuildError::BadParam {
                    key: "channel".to_string(),
                    value: other.to_string(),
                    expected: "red|green|blue",
                });
            }
        }))
    });
    reg.register_processor("grayscale", |_| Ok(Grayscale));
    reg.register_processor("invert", |_| Ok(Invert));

    reg
}
```

Leave `use crate::exec::{Node, PortSet, ProcessorNode};` as is: `NodeCtor`, `ports_of` and `register_processor` still use all three.

Append to `tests/pipeline_execution.rs`:

```rust
/// Stage-level (no runtime) check that split and merge are inverses.
#[test]
fn split_and_merge_stages_invert_each_other() {
    use pipe_graph::stages::{merge::MergeStage, split::SplitStage};
    use std::collections::HashMap;

    let src = Frame::from_data(
        2,
        2,
        3,
        FrameData::U8(vec![1, 10, 100, 2, 20, 101, 3, 30, 102, 4, 40, 103]),
    );

    let mut split = SplitStage::new(3);
    let mut m = HashMap::new();
    m.insert(PortId("in".to_string()), Payload::Frame(src.clone()));
    let mut split_out = Outputs::new();
    split.eval(&Inputs::new(m), &mut split_out).unwrap();
    let ch0 = split_out.get("out0").unwrap().as_frame().unwrap().clone();
    assert_eq!(ch0.channels, 1);
    assert_eq!(ch0.as_u8().unwrap(), &[1, 2, 3, 4]);

    let mut merge = MergeStage::new(3);
    let mut m = HashMap::new();
    for i in 0..3 {
        let ch = split_out.get(&format!("out{i}")).unwrap().clone();
        m.insert(PortId(format!("in{i}")), ch);
    }
    let mut merged = Outputs::new();
    merge.eval(&Inputs::new(m), &mut merged).unwrap();
    assert_eq!(merged.get("out").unwrap().as_frame().unwrap(), &src);
}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test --no-default-features`
Expected: PASS, including `builtin_registry_has_every_builtin_kind`, `register_all_covers_the_stage_folder`, and `split_and_merge_stages_invert_each_other`.

- [ ] **Step 8: Confirm the rustfmt blind spot (Review Focus 1)**

Temporarily add a line `fn   badly_formatted( ){}` to the end of `src/stages/blend.rs`, then run `cargo fmt --all --check`.
Expected: exits 0 (the stage file is not seen). Then run
`rustfmt --edition 2024 --check src/stages/blend.rs` — expected: exits non-zero showing the diff. Remove the line. If `cargo fmt --all --check` *does* catch it, skip Step 9's CI change and delete the rustfmt sentence from the `stages/mod.rs` docs.

- [ ] **Step 9: Cover `src/stages/` in CI and the spec**

In `.github/workflows/ci.yml`, directly after the `Format` step, add:

```yaml
      # Stage modules are declared inside a build.rs-generated file, which
      # `cargo fmt` does not follow; check them directly.
      - name: Format (stages)
        run: find src/stages -name '*.rs' -print0 | xargs -0 rustfmt --edition 2024 --check
```

In `docs/superpowers/specs/2026-09-23-stage-discovery-design.md`, append to the `## Risks` list:

```markdown
- **rustfmt coverage:** `cargo fmt` only follows out-of-line `mod`
  declarations, so it skips stage files declared in the generated file. CI
  runs `rustfmt --check` over `src/stages/` directly; locally run
  `rustfmt --edition 2024 src/stages/*.rs` after `cargo fmt`.
```

- [ ] **Step 10: Format, lint, test both legs**

```bash
cargo fmt --all && rustfmt --edition 2024 src/stages/*.rs
cargo fmt --all --check && rustfmt --edition 2024 --check src/stages/*.rs
cargo clippy --all-targets --no-default-features -- -D warnings
cargo clippy --all-targets --features bevy -- -D warnings
cargo test --no-default-features && cargo test --features bevy
cargo run -q -- run sample.yaml
```

Expected: all exit 0; last command prints `Output data: U8([1, 10, 100, 2, 20, 101, 3, 30, 102, 4, 40, 103])`.

- [ ] **Step 11: Commit**

```bash
git add build.rs src/stages src/exec/registry.rs tests/pipeline_execution.rs .github/workflows/ci.yml docs/superpowers/specs
git commit -m "stages: discover stage modules with build.rs instead of listing them"
```

---

### Task 3: Move the processor kinds into `src/stages/`

**Files:**
- Move: `src/processors/{clear_channel,grayscale,invert}.rs` → `src/stages/`
- Modify: the three moved files (imports + `register` fn), `src/processors/mod.rs`, `src/processors/process_list.rs` (test import), `src/exec/node.rs:230` (test import), `src/exec/registry.rs` (`builtin_registry`, imports, `register_all` test)

**Interfaces:**
- Consumes: `Registry::register_processor` (Task 1), `register_all` generation (Task 2).
- Produces: modules `crate::stages::{clear_channel, grayscale, invert}`; `Channel`/`ClearChannel` are now `crate::stages::clear_channel::{Channel, ClearChannel}`; `crate::processors` exports only `ProcessList`. `builtin_registry()` is exactly `register_all`.

- [ ] **Step 1: Update the test to expect all ten kinds from the folder**

In `src/exec/registry.rs`, change the expected list in `register_all_covers_the_stage_folder` to:

```rust
            [
                "blend",
                "cast",
                "clear_channel",
                "crop",
                "grayscale",
                "image_read",
                "image_write",
                "invert",
                "merge",
                "split",
            ]
```

Run: `cargo test --lib register_all_covers`
Expected: FAIL (left has 7 kinds, right has 10).

- [ ] **Step 2: Move the files and fix imports**

```bash
git mv src/processors/clear_channel.rs src/processors/grayscale.rs src/processors/invert.rs src/stages/
```

Replace `src/processors/mod.rs` with:

```rust
//! Processor combinators. Individual processors that are node kinds live in
//! [`crate::stages`].

mod process_list;
pub use self::process_list::*;
```

In `src/processors/process_list.rs` tests: `use crate::processors::{Channel, ClearChannel};` → `use crate::stages::clear_channel::{Channel, ClearChannel};`

In `src/exec/node.rs` tests (line ~230): same replacement.

- [ ] **Step 3: Add `register` fns and watch the duplicate check fire**

`src/stages/grayscale.rs` — add `use crate::exec::Registry;` to the imports and, above `#[cfg(test)]`:

```rust
pub fn register(reg: &mut Registry) {
    reg.register_processor("grayscale", |_| Ok(Grayscale));
}
```

`src/stages/invert.rs` — add `use crate::exec::Registry;` and:

```rust
pub fn register(reg: &mut Registry) {
    reg.register_processor("invert", |_| Ok(Invert));
}
```

`src/stages/clear_channel.rs` — add `use crate::exec::{BuildError, ParamsExt, Registry};` and:

```rust
pub fn register(reg: &mut Registry) {
    reg.register_processor("clear_channel", |p| {
        Ok(ClearChannel(match p.get_str("channel")? {
            "red" => Channel::Red,
            "green" => Channel::Green,
            "blue" => Channel::Blue,
            other => {
                return Err(BuildError::BadParam {
                    key: "channel".to_string(),
                    value: other.to_string(),
                    expected: "red|green|blue",
                });
            }
        }))
    });
}
```

Run: `cargo test --lib builtin_registry_has_every`
Expected: FAIL with panic `node kind 'clear_channel' registered twice` — `builtin_registry` still registers them by hand. This is Review Focus 5 working.

- [ ] **Step 4: Remove the hand registrations**

In `src/exec/registry.rs`, delete the `use crate::processors::{...};` import and replace `builtin_registry` with:

```rust
/// A registry preloaded with every stage in `src/stages/` (discovered by
/// `build.rs`; see [`crate::stages`]).
pub fn builtin_registry() -> Registry {
    let mut reg = Registry::new();
    crate::stages::register_all(&mut reg);
    reg
}
```

- [ ] **Step 5: Format, lint, test both legs**

Same commands as Task 2 Step 10. Expected: all exit 0; both kind-list tests pass; `clear_channel`'s `bad_param_errors` / `missing_param_errors` registry tests still pass.

- [ ] **Step 6: Commit**

```bash
git add -A src
git commit -m "stages: move clear_channel/grayscale/invert into the discovered stage folder"
```

---

### Task 4: Prove the drop-in workflow; document it

**Files:**
- Modify: `README.md`
- Temporary only (created and deleted in this task): `src/stages/passthrough.rs`, `src/stages/my-stage.rs`, `src/stages/NOTES.md`, `src/stages/scratch.rs.bk`, `src/stages/kernels/`, `$CLAUDE_JOB_DIR/tmp/passthrough.yaml`

**Interfaces:**
- Consumes: everything above. Produces: docs only.

- [ ] **Step 1: Write a throwaway stage and a pipeline that uses it**

`src/stages/passthrough.rs`:

```rust
use crate::exec::Registry;
use crate::traits::Processor;

pub struct Passthrough;

impl Processor for Passthrough {
    fn process(&self, _input: &mut crate::data::Frame) {}
}

pub fn register(reg: &mut Registry) {
    reg.register_processor("passthrough", |_| Ok(Passthrough));
}
```

`$CLAUDE_JOB_DIR/tmp/passthrough.yaml`:

```yaml
nodes:
  - id: p
    kind: passthrough
edges: []
```

- [ ] **Step 2: Add → available, delete → gone (Review Focus 4)**

Run: `cargo run -q -- check "$CLAUDE_JOB_DIR/tmp/passthrough.yaml"`
Expected: `Graph is valid and compiled successfully!` — no file other than `passthrough.rs` was touched (`git status --short` shows only `?? src/stages/passthrough.rs`).

Run: `rm src/stages/passthrough.rs && cargo run -q -- check "$CLAUDE_JOB_DIR/tmp/passthrough.yaml"`
Expected: exit 1, `Graph compilation failed: ... UnknownKind("passthrough")`.

- [ ] **Step 3: Bad module name fails clearly (Review Focus 2)**

Run: `printf 'pub fn register(_: &mut crate::exec::Registry) {}\n' > src/stages/my-stage.rs && cargo build 2>&1 | grep 'src/stages/my-stage.rs'`
Expected: build fails; output contains `src/stages/my-stage.rs: 'my-stage' is not a valid module name; rename it to a snake_case Rust identifier`. Then `rm src/stages/my-stage.rs`.

- [ ] **Step 4: Non-stage files are ignored (Review Focus 3)**

```bash
echo notes > src/stages/NOTES.md
echo 'garbage(' > src/stages/scratch.rs.bk
mkdir src/stages/kernels && echo '__global__ void k() {}' > src/stages/kernels/blur.cu
cargo build
```

Expected: build succeeds. Then `rm -r src/stages/NOTES.md src/stages/scratch.rs.bk src/stages/kernels` and confirm `git status --short` is clean.

- [ ] **Step 5: Document "Adding a stage" in the README**

In `README.md`, replace the `- **`stages`** — ...` bullet with:

```markdown
- **`stages`** — every node kind, one module per file, **discovered
  automatically** by `build.rs` (see *Adding a stage*): `crop`, `cast`,
  `split`, `merge`, `blend`, `image_read`, `image_write`, and the
  `Processor`-backed `clear_channel`, `grayscale`, `invert`.
- **`processors`** — `ProcessList`, which chains `Processor`s.
```

and insert this section immediately before `### Pipeline files`:

````markdown
### Adding a stage

Create `src/stages/<name>.rs` and rebuild — nothing else to edit. The file
needs one `register` fn; `build.rs` declares the module and calls it:

```rust
use crate::exec::Registry;

pub fn register(reg: &mut Registry) {
    // A `Node` that implements `TryFrom<&Params>`:
    reg.register_stage::<MyStage>("my_stage");
    // Or a single-in/single-out `Processor`:
    // reg.register_processor("my_filter", |_params| Ok(MyFilter));
}
```

`<name>` must be a snake_case Rust identifier; a multi-file stage can be a
`<name>/mod.rs` folder instead. Two stages registering the same kind panic at
startup. `cargo fmt` does not reach stage files, so also run
`rustfmt --edition 2024 src/stages/*.rs`.
````

- [ ] **Step 6: Final verification of both CI legs**

Same commands as Task 2 Step 10, plus `git status --short` (expect only `README.md` modified).

- [ ] **Step 7: Commit and push**

```bash
git add README.md
git commit -m "docs: explain adding a stage by dropping a file in src/stages"
git push
```
