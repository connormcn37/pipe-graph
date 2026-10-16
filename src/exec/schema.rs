//! Declarative parameter schemas for node kinds.
//!
//! Node params travel as an untyped `String -> String` map ([`Params`]) so a
//! graph can be serialized, edited as text, and typed into by a UI. Each stage
//! parses that map in its own `TryFrom<&Params>`, which is the source of truth
//! for *building* a node, but it is opaque: nothing outside the stage could tell
//! which keys exist, what they mean, or which values are legal. A [`KindInfo`]
//! states that contract as data, so that:
//!
//! - a UI or CLI can list kinds, render a typed editor for each param, and seed
//!   a new node with [`crate::exec::Registry::defaults`];
//! - validation can report *every* problem with a node's params at once, by
//!   name, before anything is constructed ([`crate::exec::Registry::check_params`]),
//!   instead of whichever error the stage's parser happened to hit first.
//!
//! Each stage declares its schema next to its `TryFrom` (an associated
//! `fn info() -> KindInfo`) so the two are edited together, and a test in
//! `registry.rs` cross-checks them for every builtin kind.
//!
//! A schema may be *stricter* than the stage's parser, never looser: e.g.
//! `split`'s `channels` is capped at `SplitStage::MAX_CHANNELS` and `crop`
//! rejects an empty rectangle, while their `try_from`s accept any `u32`. Anything
//! the schema accepts must build.

use std::fmt;

use crate::graph::Params;

/// The type (and legal range) of one parameter's string value.
///
/// Bounds are inclusive. `None` means unbounded on that side.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    /// A decimal `u32`.
    U32 { min: Option<u32>, max: Option<u32> },
    /// A finite `f32` (NaN and infinities are always rejected: no stage gives
    /// them a useful meaning, and NaN would slip past any range check).
    F32 { min: Option<f32>, max: Option<f32> },
    /// `true` or `false`.
    Bool,
    /// One of a fixed set of literal strings.
    Enum(&'static [&'static str]),
    /// Free-form text, passed through to the stage (or an external program)
    /// without interpretation.
    Str,
    /// A filesystem path (or, for ffmpeg sources, a URL/device). Must be
    /// non-empty; existence is *not* checked here because that is an
    /// environment question the stage answers at build time.
    Path,
    /// A frame rate: an integer (`30`), a ratio (`30000/1001`) or a decimal
    /// (`29.97`), positive. Parsed exactly as the video I/O nodes do.
    FrameRate,
}

impl ParamKind {
    /// Short human-readable description of the accepted values, used in
    /// [`ParamIssue`] messages and schema listings.
    pub fn expected(&self) -> String {
        fn range<T: fmt::Display>(ty: &str, min: &Option<T>, max: &Option<T>) -> String {
            match (min, max) {
                (None, None) => ty.to_string(),
                (Some(lo), None) => format!("{ty} >= {lo}"),
                (None, Some(hi)) => format!("{ty} <= {hi}"),
                (Some(lo), Some(hi)) => format!("{ty} in {lo}..={hi}"),
            }
        }
        match self {
            ParamKind::U32 { min, max } => range("u32", min, max),
            ParamKind::F32 { min, max } => range("finite f32", min, max),
            ParamKind::Bool => "true|false".to_string(),
            ParamKind::Enum(values) => values.join("|"),
            ParamKind::Str => "string".to_string(),
            ParamKind::Path => "non-empty path".to_string(),
            ParamKind::FrameRate => "frame rate (e.g. 30, 30000/1001, 29.97)".to_string(),
        }
    }

    /// Check one value against this kind. `name` is only used to label the
    /// returned issue.
    pub fn check(&self, name: &str, value: &str) -> Option<ParamIssue> {
        let unparsable = || {
            Some(ParamIssue::Unparsable {
                name: name.to_string(),
                value: value.to_string(),
                expected: self.expected(),
            })
        };
        let out_of_range = || {
            Some(ParamIssue::OutOfRange {
                name: name.to_string(),
                value: value.to_string(),
                expected: self.expected(),
            })
        };
        match self {
            ParamKind::U32 { min, max } => {
                // Same parser the stages use (`ParamsExt::get_u32`), so the
                // schema and the build path agree on e.g. leading `+`.
                let Ok(v) = value.parse::<u32>() else {
                    return unparsable();
                };
                if min.is_some_and(|lo| v < lo) || max.is_some_and(|hi| v > hi) {
                    return out_of_range();
                }
                None
            }
            ParamKind::F32 { min, max } => {
                let Ok(v) = value.parse::<f32>() else {
                    return unparsable();
                };
                if !v.is_finite() || min.is_some_and(|lo| v < lo) || max.is_some_and(|hi| v > hi) {
                    return out_of_range();
                }
                None
            }
            ParamKind::Bool => match value {
                "true" | "false" => None,
                _ => unparsable(),
            },
            ParamKind::Enum(values) => {
                if values.contains(&value) {
                    None
                } else {
                    Some(ParamIssue::NotInEnum {
                        name: name.to_string(),
                        value: value.to_string(),
                        allowed: values,
                    })
                }
            }
            ParamKind::Str => None,
            ParamKind::Path => {
                if value.is_empty() {
                    unparsable()
                } else {
                    None
                }
            }
            ParamKind::FrameRate => {
                let mut p = Params::new();
                p.insert(name.to_string(), value.to_string());
                match crate::io::parse_fps(&p, name, (1, 1)) {
                    Ok(_) => None,
                    Err(_) => unparsable(),
                }
            }
        }
    }
}

/// One declared parameter of a node kind.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamSpec {
    /// The key in [`Params`].
    pub name: &'static str,
    pub kind: ParamKind,
    /// Whether the key must be present for the node to build.
    pub required: bool,
    /// For an optional param: the value the stage assumes when the key is
    /// absent. For a required param: a sensible starting value for a freshly
    /// added node (the stage itself has no fallback). Either way it must pass
    /// [`ParamKind::check`]; `None` when no single value makes sense (an
    /// output path, or an optional param whose absence means "automatic").
    pub default: Option<&'static str>,
    /// One-line explanation for UIs and `--help`-style listings.
    pub doc: &'static str,
}

impl ParamSpec {
    /// A required parameter.
    pub fn required(name: &'static str, kind: ParamKind, doc: &'static str) -> Self {
        Self {
            name,
            kind,
            required: true,
            default: None,
            doc,
        }
    }

    /// An optional parameter.
    pub fn optional(name: &'static str, kind: ParamKind, doc: &'static str) -> Self {
        Self {
            required: false,
            ..Self::required(name, kind, doc)
        }
    }

    /// Set [`ParamSpec::default`].
    pub fn with_default(mut self, default: &'static str) -> Self {
        self.default = Some(default);
        self
    }
}

/// Everything a UI needs to know about a node kind without building one.
#[derive(Debug, Clone, PartialEq)]
pub struct KindInfo {
    /// The registry key (`NodeSpec::kind`).
    pub kind: String,
    /// One-line description of what the node does.
    pub summary: &'static str,
    /// Declared params, in display order. `None` means the schema is unknown
    /// (a kind registered through the untyped [`crate::exec::Registry::register`],
    /// typically in tests): its params are not checked and no defaults exist.
    pub params: Option<Vec<ParamSpec>>,
}

impl KindInfo {
    pub fn new(kind: &str, summary: &'static str, params: Vec<ParamSpec>) -> Self {
        Self {
            kind: kind.to_string(),
            summary,
            params: Some(params),
        }
    }

    /// Info for a kind registered without a schema.
    pub fn unknown(kind: &str) -> Self {
        Self {
            kind: kind.to_string(),
            summary: "(no schema declared)",
            params: None,
        }
    }

    /// The declared param named `name`, if any.
    pub fn param(&self, name: &str) -> Option<&ParamSpec> {
        self.params.as_deref()?.iter().find(|p| p.name == name)
    }

    /// Check `params` against this schema, returning every issue found (empty
    /// when valid, or when the schema is unknown). Issues are ordered: unknown
    /// keys (sorted), then declared params in declaration order.
    pub fn check(&self, params: &Params) -> Vec<ParamIssue> {
        let Some(specs) = &self.params else {
            return Vec::new();
        };
        let mut issues = Vec::new();

        let mut unknown: Vec<&String> = params
            .keys()
            .filter(|k| !specs.iter().any(|s| s.name == k.as_str()))
            .collect();
        unknown.sort();
        issues.extend(
            unknown
                .into_iter()
                .map(|k| ParamIssue::Unknown { name: k.clone() }),
        );

        for spec in specs {
            match params.get(spec.name) {
                None if spec.required => issues.push(ParamIssue::Missing {
                    name: spec.name.to_string(),
                }),
                None => {}
                Some(value) => issues.extend(spec.kind.check(spec.name, value)),
            }
        }
        issues
    }

    /// Every declared param that has a default, as a [`Params`] map.
    pub fn defaults(&self) -> Params {
        self.params
            .iter()
            .flatten()
            .filter_map(|s| Some((s.name.to_string(), s.default?.to_string())))
            .collect()
    }
}

impl fmt::Display for KindInfo {
    /// Multi-line listing: `kind — summary`, then one indented line per param.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} — {}", self.kind, self.summary)?;
        let Some(specs) = &self.params else {
            return Ok(());
        };
        if specs.is_empty() {
            write!(f, "\n    (no params)")?;
        }
        for s in specs {
            write!(
                f,
                "\n    {}{}: {}",
                s.name,
                if s.required { "" } else { "?" },
                s.kind.expected()
            )?;
            if let Some(d) = s.default {
                write!(f, " = {d}")?;
            }
            write!(f, " — {}", s.doc)?;
        }
        Ok(())
    }
}

/// One problem with a node's params, found by [`KindInfo::check`]. Every
/// variant names the offending param.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamIssue {
    /// The key is not a declared param of this kind (often a typo).
    Unknown { name: String },
    /// A required param is absent.
    Missing { name: String },
    /// The value does not parse as the declared type.
    Unparsable {
        name: String,
        value: String,
        expected: String,
    },
    /// The value parses but lies outside the declared range.
    OutOfRange {
        name: String,
        value: String,
        expected: String,
    },
    /// The value is not one of the declared enum choices.
    NotInEnum {
        name: String,
        value: String,
        allowed: &'static [&'static str],
    },
}

impl ParamIssue {
    /// The param this issue is about.
    pub fn name(&self) -> &str {
        match self {
            ParamIssue::Unknown { name }
            | ParamIssue::Missing { name }
            | ParamIssue::Unparsable { name, .. }
            | ParamIssue::OutOfRange { name, .. }
            | ParamIssue::NotInEnum { name, .. } => name,
        }
    }
}

impl fmt::Display for ParamIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamIssue::Unknown { name } => write!(f, "unknown parameter '{name}'"),
            ParamIssue::Missing { name } => write!(f, "missing required parameter '{name}'"),
            ParamIssue::Unparsable {
                name,
                value,
                expected,
            } => write!(
                f,
                "parameter '{name}' = '{value}' is not a valid {expected}"
            ),
            ParamIssue::OutOfRange {
                name,
                value,
                expected,
            } => write!(
                f,
                "parameter '{name}' = '{value}' is out of range ({expected})"
            ),
            ParamIssue::NotInEnum {
                name,
                value,
                allowed,
            } => write!(
                f,
                "parameter '{name}' = '{value}' is not one of {}",
                allowed.join("|")
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(kv: &[(&str, &str)]) -> Params {
        kv.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn sample() -> KindInfo {
        KindInfo::new(
            "sample",
            "test kind",
            vec![
                ParamSpec::required(
                    "n",
                    ParamKind::U32 {
                        min: Some(1),
                        max: Some(10),
                    },
                    "n",
                ),
                ParamSpec::optional(
                    "x",
                    ParamKind::F32 {
                        min: Some(0.0),
                        max: Some(1.0),
                    },
                    "x",
                )
                .with_default("0.5"),
                ParamSpec::optional("mode", ParamKind::Enum(&["a", "b"]), "mode"),
                ParamSpec::optional("on", ParamKind::Bool, "on"),
                ParamSpec::optional("p", ParamKind::Path, "p"),
                ParamSpec::optional("fps", ParamKind::FrameRate, "fps"),
            ],
        )
    }

    #[test]
    fn valid_params_have_no_issues() {
        let p = params(&[
            ("n", "3"),
            ("x", "1"),
            ("mode", "b"),
            ("on", "true"),
            ("p", "a.y4m"),
            ("fps", "30000/1001"),
        ]);
        assert_eq!(sample().check(&p), vec![]);
    }

    #[test]
    fn catches_each_issue_kind() {
        let issues = sample().check(&params(&[
            ("zz", "1"),
            ("x", "abc"),
            ("mode", "c"),
            ("on", "yes"),
            ("p", ""),
            ("fps", "0"),
        ]));
        let names: Vec<&str> = issues.iter().map(ParamIssue::name).collect();
        assert_eq!(names, ["zz", "n", "x", "mode", "on", "p", "fps"]);
        assert!(matches!(issues[0], ParamIssue::Unknown { .. }));
        assert!(matches!(issues[1], ParamIssue::Missing { .. }));
        assert!(matches!(issues[2], ParamIssue::Unparsable { .. }));
        assert!(matches!(issues[3], ParamIssue::NotInEnum { .. }));
        assert!(matches!(issues[4], ParamIssue::Unparsable { .. }));
        assert!(matches!(issues[5], ParamIssue::Unparsable { .. }));
        assert!(matches!(issues[6], ParamIssue::Unparsable { .. }));
    }

    #[test]
    fn range_bounds_are_inclusive_and_enforced() {
        let info = sample();
        for (n, x, ok) in [
            ("1", "0", true),
            ("10", "1", true),
            ("0", "0.5", false),
            ("11", "0.5", false),
            ("5", "-0.1", false),
            ("5", "1.01", false),
            ("5", "NaN", false),
            ("5", "inf", false),
        ] {
            let issues = info.check(&params(&[("n", n), ("x", x)]));
            assert_eq!(issues.is_empty(), ok, "n={n} x={x}: {issues:?}");
            if !ok {
                assert!(
                    issues
                        .iter()
                        .all(|i| matches!(i, ParamIssue::OutOfRange { .. })),
                    "{issues:?}"
                );
            }
        }
    }

    #[test]
    fn unknown_schema_checks_nothing() {
        let info = KindInfo::unknown("mystery");
        assert!(info.check(&params(&[("anything", "goes")])).is_empty());
        assert!(info.defaults().is_empty());
    }

    #[test]
    fn defaults_collects_declared_defaults() {
        assert_eq!(sample().defaults(), params(&[("x", "0.5")]));
    }
}
