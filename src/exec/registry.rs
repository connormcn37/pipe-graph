//! Registry: the bridge from a node's `kind` string to an executable [`Node`].
//!
//! A `NodeSpec` produced by an editor/UI carries a `kind` (e.g. `"clear_channel"`)
//! and a string→string `Params` map. The `Registry` maps each `kind` to a
//! constructor that parses those params into a typed stage and returns a boxed
//! `Node`. This is what lets a serialized graph become a runnable one.

use std::collections::HashMap;

use crate::exec::{KindInfo, Node, ParamIssue, ParamKind, ParamSpec, PortSet, ProcessorNode};
use crate::graph::{NodeSpec, Params};
use crate::processors::{Channel, ClearChannel};
use crate::stages::{
    BoxBlurStage, CastStage, CropStage, GainStage, GrayscaleStage, InvertStage, MergeStage,
    SplitStage, ThresholdStage,
};

/// Errors raised while constructing a node from its spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// No constructor is registered for this `kind`.
    UnknownKind(String),
    /// A required parameter was absent.
    MissingParam(String),
    /// A parameter was present but could not be parsed as `expected`.
    BadParam {
        key: String,
        value: String,
        expected: &'static str,
    },
    /// The node is well-specified but cannot run in this environment (e.g. a
    /// required external program such as `ffmpeg` is not installed, or an
    /// input file does not exist).
    Unavailable(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::UnknownKind(k) => write!(f, "unknown node kind '{k}'"),
            BuildError::MissingParam(k) => write!(f, "missing required parameter '{k}'"),
            BuildError::BadParam {
                key,
                value,
                expected,
            } => write!(f, "parameter '{key}' = '{value}' is not a valid {expected}"),
            BuildError::Unavailable(m) => write!(f, "unavailable: {m}"),
        }
    }
}

impl std::error::Error for BuildError {}

/// Typed accessors over the string-keyed [`Params`] map, so stage constructors
/// don't hand-roll string parsing.
pub trait ParamsExt {
    fn get_str(&self, key: &str) -> Result<&str, BuildError>;
    fn get_u32(&self, key: &str) -> Result<u32, BuildError>;
    fn get_f32(&self, key: &str) -> Result<f32, BuildError>;
    /// Like [`ParamsExt::get_u32`] but returns `default` when the key is absent.
    fn get_u32_or(&self, key: &str, default: u32) -> Result<u32, BuildError>;
}

impl ParamsExt for Params {
    fn get_str(&self, key: &str) -> Result<&str, BuildError> {
        self.get(key)
            .map(String::as_str)
            .ok_or_else(|| BuildError::MissingParam(key.to_string()))
    }

    fn get_u32(&self, key: &str) -> Result<u32, BuildError> {
        let v = self.get_str(key)?;
        v.parse().map_err(|_| BuildError::BadParam {
            key: key.to_string(),
            value: v.to_string(),
            expected: "u32",
        })
    }

    fn get_f32(&self, key: &str) -> Result<f32, BuildError> {
        let v = self.get_str(key)?;
        v.parse().map_err(|_| BuildError::BadParam {
            key: key.to_string(),
            value: v.to_string(),
            expected: "f32",
        })
    }

    fn get_u32_or(&self, key: &str, default: u32) -> Result<u32, BuildError> {
        match self.get(key) {
            None => Ok(default),
            Some(_) => self.get_u32(key),
        }
    }
}

/// A constructor: parse params and produce a boxed node.
pub type NodeCtor = Box<dyn Fn(&Params) -> Result<Box<dyn Node>, BuildError>>;

/// One registered kind: its schema plus its constructor.
struct Entry {
    info: KindInfo,
    ctor: NodeCtor,
}

/// Maps `kind` strings to node constructors and their [`KindInfo`] schemas.
#[derive(Default)]
pub struct Registry {
    entries: HashMap<String, Entry>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a kind together with its parameter schema, replacing any prior
    /// registration of `info.kind`. This is how every builtin kind is added,
    /// so UIs can list it and validation can check its params up front.
    pub fn register_kind<F>(&mut self, info: KindInfo, ctor: F)
    where
        F: Fn(&Params) -> Result<Box<dyn Node>, BuildError> + 'static,
    {
        self.entries.insert(
            info.kind.clone(),
            Entry {
                info,
                ctor: Box::new(ctor),
            },
        );
    }

    /// Register a constructor for `kind` with no declared schema
    /// ([`KindInfo::unknown`]), replacing any prior registration. Its params
    /// are left entirely to the constructor; kept for ad-hoc kinds in tests.
    pub fn register<F>(&mut self, kind: &str, ctor: F)
    where
        F: Fn(&Params) -> Result<Box<dyn Node>, BuildError> + 'static,
    {
        self.register_kind(KindInfo::unknown(kind), ctor);
    }

    pub fn contains(&self, kind: &str) -> bool {
        self.entries.contains_key(kind)
    }

    /// The schema registered for `kind`.
    pub fn info(&self, kind: &str) -> Option<&KindInfo> {
        self.entries.get(kind).map(|e| &e.info)
    }

    /// Every registered kind name, sorted (stable order for palettes/listings).
    pub fn kinds(&self) -> Vec<&str> {
        let mut kinds: Vec<&str> = self.entries.keys().map(String::as_str).collect();
        kinds.sort_unstable();
        kinds
    }

    /// Check `spec.params` against its kind's schema, reporting every issue
    /// (unknown keys, missing required params, unparsable, out-of-range or
    /// not-in-enum values). Empty when the params are valid, when the kind has
    /// no declared schema, or when the kind is not registered at all (that is
    /// [`BuildError::UnknownKind`]'s job).
    pub fn check_params(&self, spec: &NodeSpec) -> Vec<ParamIssue> {
        self.info(&spec.kind)
            .map(|info| info.check(&spec.params))
            .unwrap_or_default()
    }

    /// The params a freshly added node of `kind` should start with: every
    /// declared param that has a default. Empty for unknown kinds/schemas.
    pub fn defaults(&self, kind: &str) -> Params {
        self.info(kind).map(KindInfo::defaults).unwrap_or_default()
    }

    /// Instantiate the node described by `spec`.
    ///
    /// This runs the constructor only; it does not apply
    /// [`Registry::check_params`] (graph validation does), so a direct build
    /// keeps the stage's own [`BuildError`]s.
    pub fn build(&self, spec: &NodeSpec) -> Result<Box<dyn Node>, BuildError> {
        let entry = self
            .entries
            .get(&spec.kind)
            .ok_or_else(|| BuildError::UnknownKind(spec.kind.clone()))?;
        (entry.ctor)(&spec.params)
    }

    /// The port set a node of this spec would declare. Builds a throwaway node,
    /// because port shape can depend on params (e.g. Split/Merge port counts).
    pub fn ports_of(&self, spec: &NodeSpec) -> Result<PortSet, BuildError> {
        Ok(self.build(spec)?.ports())
    }
}

/// Schema for `clear_channel`, which (unlike the `stages`) is a legacy
/// [`crate::traits::Processor`] wrapped in a `ProcessorNode`, so its parser
/// lives in [`builtin_registry`] and its schema sits here beside it.
fn clear_channel_info() -> KindInfo {
    KindInfo::new(
        "clear_channel",
        "Zero one channel of an RGB frame.",
        vec![
            ParamSpec::required(
                "channel",
                ParamKind::Enum(&["red", "green", "blue"]),
                "Which channel to set to 0.",
            )
            .with_default("red"),
        ],
    )
}

/// A registry preloaded with every built-in kind and its schema.
pub fn builtin_registry() -> Registry {
    let mut reg = Registry::new();

    reg.register_kind(clear_channel_info(), |p| {
        let ch = match p.get_str("channel")? {
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
        };
        Ok(Box::new(ProcessorNode::new(ClearChannel(ch))) as Box<dyn Node>)
    });

    reg.register_kind(CropStage::info(), |p| {
        Ok(Box::new(CropStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register_kind(CastStage::info(), |p| {
        Ok(Box::new(CastStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register_kind(SplitStage::info(), |p| {
        Ok(Box::new(SplitStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register_kind(MergeStage::info(), |p| {
        Ok(Box::new(MergeStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register_kind(GainStage::info(), |p| {
        Ok(Box::new(GainStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register_kind(GrayscaleStage::info(), |p| {
        Ok(Box::new(GrayscaleStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register_kind(InvertStage::info(), |p| {
        Ok(Box::new(InvertStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register_kind(ThresholdStage::info(), |p| {
        Ok(Box::new(ThresholdStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register_kind(BoxBlurStage::info(), |p| {
        Ok(Box::new(BoxBlurStage::try_from(p)?) as Box<dyn Node>)
    });
    crate::io::register_io(&mut reg);

    reg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Frame, Payload};
    use crate::exec::{Inputs, Outputs};
    use crate::graph::{NodeId, PortId};

    fn spec(kind: &str, params: &[(&str, &str)]) -> NodeSpec {
        let mut p = Params::new();
        for (k, v) in params {
            p.insert(k.to_string(), v.to_string());
        }
        NodeSpec {
            id: NodeId("n".to_string()),
            kind: kind.to_string(),
            params: p,
        }
    }

    #[test]
    fn builds_and_evaluates_clear_channel() {
        let reg = builtin_registry();
        let mut node = reg
            .build(&spec("clear_channel", &[("channel", "red")]))
            .unwrap();

        let mut inputs_map = HashMap::new();
        inputs_map.insert(
            PortId("in".to_string()),
            Payload::Frame(Frame::from_rgb8(1, 1, vec![(255, 9, 9)])),
        );
        let mut outputs = Outputs::new();
        node.eval(&Inputs::new(inputs_map), &mut outputs).unwrap();

        assert_eq!(
            outputs.get("out").unwrap().as_frame().unwrap().to_rgb8(),
            vec![(0, 9, 9)]
        );
    }

    #[test]
    fn unknown_kind_errors() {
        let reg = builtin_registry();
        // `.err().unwrap()` rather than `.unwrap_err()`: the Ok payload is a
        // `Box<dyn Node>`, which is not `Debug`.
        let err = reg.build(&spec("nope", &[])).err().unwrap();
        assert_eq!(err, BuildError::UnknownKind("nope".to_string()));
    }

    #[test]
    fn missing_param_errors() {
        let reg = builtin_registry();
        let err = reg.build(&spec("clear_channel", &[])).err().unwrap();
        assert_eq!(err, BuildError::MissingParam("channel".to_string()));
    }

    #[test]
    fn bad_param_errors() {
        let reg = builtin_registry();
        let err = reg
            .build(&spec("clear_channel", &[("channel", "purple")]))
            .err()
            .unwrap();
        assert_eq!(
            err,
            BuildError::BadParam {
                key: "channel".to_string(),
                value: "purple".to_string(),
                expected: "red|green|blue",
            }
        );
    }

    #[test]
    fn typed_param_accessors() {
        let mut p = Params::new();
        p.insert("w".to_string(), "640".to_string());
        p.insert("scale".to_string(), "1.5".to_string());
        assert_eq!(p.get_u32("w").unwrap(), 640);
        assert_eq!(p.get_f32("scale").unwrap(), 1.5);
        assert_eq!(p.get_u32_or("missing", 7).unwrap(), 7);
        assert!(matches!(
            p.get_u32("scale"),
            Err(BuildError::BadParam { .. })
        ));
    }

    /// Every kind `builtin_registry` provides; a new builtin must be added
    /// here (and so get a schema) or `every_builtin_kind_has_a_schema` fails.
    const BUILTIN_KINDS: &[&str] = &[
        "box_blur",
        "cast",
        "clear_channel",
        "crop",
        "ffmpeg_sink",
        "ffmpeg_source",
        "gain",
        "grayscale",
        "invert",
        "merge",
        "split",
        "threshold",
        "y4m_sink",
        "y4m_source",
    ];

    /// Build `spec`, treating "ffmpeg is not installed" as success: the
    /// params were accepted, only the environment is missing.
    fn builds_or_unavailable(reg: &Registry, spec: &NodeSpec) -> Result<(), BuildError> {
        match reg.build(spec) {
            Ok(_) | Err(BuildError::Unavailable(_)) if spec.kind.starts_with("ffmpeg") => Ok(()),
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// A complete, valid param set for `kind`: its defaults, plus a sample
    /// value for each required param that has no default.
    fn complete_params(reg: &Registry, kind: &str) -> Params {
        let mut p = reg.defaults(kind);
        for s in reg.info(kind).unwrap().params.as_ref().unwrap() {
            if s.required && s.default.is_none() {
                let v = match (kind, s.name) {
                    // Must exist: y4m_source checks at build time.
                    ("y4m_source", "path") => env!("CARGO_MANIFEST_PATH"),
                    (_, "path") => "/nonexistent/never-written.out",
                    ("crop", _) => "1",
                    _ => panic!("no sample value for {kind}.{}", s.name),
                };
                p.insert(s.name.to_string(), v.to_string());
            }
        }
        p
    }

    fn spec_with(kind: &str, params: Params) -> NodeSpec {
        NodeSpec {
            id: NodeId("n".to_string()),
            kind: kind.to_string(),
            params,
        }
    }

    #[test]
    fn every_builtin_kind_has_a_schema() {
        let reg = builtin_registry();
        assert_eq!(
            reg.kinds(),
            BUILTIN_KINDS,
            "kinds() must be sorted and complete"
        );
        for kind in BUILTIN_KINDS {
            let info = reg.info(kind).unwrap();
            assert_eq!(info.kind, *kind);
            assert!(info.params.is_some(), "{kind} has no declared schema");
            assert!(!info.summary.is_empty());
            for s in info.params.as_ref().unwrap() {
                assert!(!s.doc.is_empty(), "{kind}.{} has no doc", s.name);
            }
        }
    }

    #[test]
    fn print_every_schema() {
        // Run with `--nocapture` to eyeball the catalog.
        let reg = builtin_registry();
        for kind in reg.kinds() {
            println!("{}", reg.info(kind).unwrap());
        }
    }

    #[test]
    fn every_declared_default_passes_its_schema() {
        let reg = builtin_registry();
        for kind in reg.kinds() {
            let info = reg.info(kind).unwrap();
            for s in info.params.as_ref().unwrap() {
                if let Some(d) = s.default {
                    assert_eq!(s.kind.check(s.name, d), None, "{kind}.{} default", s.name);
                }
            }
        }
    }

    #[test]
    fn kinds_whose_required_params_have_defaults_build_from_defaults() {
        let reg = builtin_registry();
        let mut built = 0;
        for kind in reg.kinds() {
            let info = reg.info(kind).unwrap();
            let specs = info.params.as_ref().unwrap();
            if specs.iter().any(|s| s.required && s.default.is_none()) {
                continue;
            }
            let spec = spec_with(kind, reg.defaults(kind));
            assert_eq!(reg.check_params(&spec), vec![], "{kind}");
            builds_or_unavailable(&reg, &spec).unwrap_or_else(|e| panic!("{kind}: {e}"));
            built += 1;
        }
        // Guard against the filter silently skipping everything.
        assert!(built >= 9, "only {built} kinds built from defaults");
    }

    /// The schema and `TryFrom` must describe the same params:
    /// - a complete param set passes `check_params` and builds;
    /// - dropping a *required* param makes the stage itself report exactly
    ///   that param missing, and dropping an optional one still builds;
    /// - a garbage value in any typed param is rejected by both the schema
    ///   and the stage, naming that param (so the stage really reads it);
    /// - any undeclared key is flagged by `check_params`.
    #[test]
    fn schemas_match_what_try_from_accepts() {
        let reg = builtin_registry();
        for kind in reg.kinds() {
            let info = reg.info(kind).unwrap();
            let full = complete_params(&reg, kind);
            let spec = spec_with(kind, full.clone());
            assert_eq!(reg.check_params(&spec), vec![], "{kind}");
            builds_or_unavailable(&reg, &spec).unwrap_or_else(|e| panic!("{kind}: {e}"));

            for s in info.params.as_ref().unwrap() {
                let mut without = full.clone();
                without.remove(s.name);
                let spec = spec_with(kind, without);
                if s.required {
                    assert_eq!(
                        reg.check_params(&spec),
                        vec![ParamIssue::Missing {
                            name: s.name.to_string()
                        }],
                        "{kind}.{}",
                        s.name
                    );
                    assert_eq!(
                        reg.build(&spec).err(),
                        Some(BuildError::MissingParam(s.name.to_string())),
                        "{kind}.{} is declared required but try_from does not require it",
                        s.name
                    );
                } else {
                    assert_eq!(reg.check_params(&spec), vec![], "{kind}.{}", s.name);
                    builds_or_unavailable(&reg, &spec)
                        .unwrap_or_else(|e| panic!("{kind}.{} is declared optional: {e}", s.name));
                }

                if matches!(s.kind, ParamKind::Str | ParamKind::Path) {
                    continue; // any non-empty string is legal
                }
                let mut garbage = full.clone();
                garbage.insert(s.name.to_string(), "%garbage%".to_string());
                let spec = spec_with(kind, garbage);
                let issues = reg.check_params(&spec);
                assert_eq!(issues.len(), 1, "{kind}.{}: {issues:?}", s.name);
                assert_eq!(issues[0].name(), s.name);
                match reg.build(&spec).err() {
                    Some(BuildError::BadParam { key, .. }) => assert_eq!(key, s.name),
                    other => panic!("{kind}.{}: try_from accepted garbage ({other:?})", s.name),
                }
            }

            let mut extra = full.clone();
            extra.insert("definitely_not_a_param".to_string(), "1".to_string());
            assert_eq!(
                reg.check_params(&spec_with(kind, extra)),
                vec![ParamIssue::Unknown {
                    name: "definitely_not_a_param".to_string()
                }],
                "{kind}"
            );
        }
    }

    #[test]
    fn check_params_catches_ranges_and_enums_of_builtins() {
        let reg = builtin_registry();
        let check = |kind: &str, kv: &[(&str, &str)]| reg.check_params(&spec(kind, kv));
        assert!(matches!(
            check("box_blur", &[("radius", "4097")])[..],
            [ParamIssue::OutOfRange { .. }]
        ));
        assert!(check("box_blur", &[("radius", "4096")]).is_empty());
        assert!(matches!(
            check("threshold", &[("level", "NaN")])[..],
            [ParamIssue::OutOfRange { .. }]
        ));
        assert!(matches!(
            check("split", &[("channels", "4000000000")])[..],
            [ParamIssue::OutOfRange { .. }]
        ));
        assert!(matches!(
            check("crop", &[("x", "0"), ("y", "0"), ("w", "0"), ("h", "1")])[..],
            [ParamIssue::OutOfRange { .. }]
        ));
        assert!(matches!(
            check("split", &[("channels", "0")])[..],
            [ParamIssue::OutOfRange { .. }]
        ));
        assert!(matches!(
            check("cast", &[("dtype", "f64")])[..],
            [ParamIssue::NotInEnum { .. }]
        ));
        assert!(matches!(
            check("y4m_sink", &[("path", "a.y4m"), ("fps", "0")])[..],
            [ParamIssue::Unparsable { .. }]
        ));
        assert!(matches!(
            check("gain", &[("factor", "inf")])[..],
            [ParamIssue::OutOfRange { .. }]
        ));
    }

    #[test]
    fn untyped_register_has_unknown_schema() {
        let mut reg = Registry::new();
        reg.register("custom", |p| {
            Ok(Box::new(GainStage::try_from(p)?) as Box<dyn Node>)
        });
        let info = reg.info("custom").unwrap();
        assert_eq!(info.params, None);
        assert!(reg.check_params(&spec("custom", &[("x", "y")])).is_empty());
        assert!(reg.defaults("custom").is_empty());
        assert!(reg.info("nope").is_none());
        assert!(reg.check_params(&spec("nope", &[("x", "y")])).is_empty());
    }
}
