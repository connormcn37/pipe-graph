//! Registry: the bridge from a node's `kind` string to an executable [`Node`].
//!
//! A `NodeSpec` produced by an editor/UI carries a `kind` (e.g. `"clear_channel"`)
//! and a string→string `Params` map. The `Registry` maps each `kind` to a
//! constructor that parses those params into a typed stage and returns a boxed
//! `Node`. This is what lets a serialized graph become a runnable one.

use std::collections::HashMap;

use crate::exec::{Node, PortSet, ProcessorNode};
use crate::graph::{NodeSpec, Params};
use crate::processors::{Channel, ClearChannel, Grayscale, Invert};
use crate::stages::{
    BlendStage, CastStage, CropStage, ImageReadStage, ImageWriteStage, MergeStage, SplitStage,
};
use crate::traits::Processor;

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

/// Maps `kind` strings to node constructors.
#[derive(Default)]
pub struct Registry {
    ctors: HashMap<String, NodeCtor>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a constructor for `kind`, replacing any prior registration.
    pub fn register<F>(&mut self, kind: &str, ctor: F)
    where
        F: Fn(&Params) -> Result<Box<dyn Node>, BuildError> + 'static,
    {
        self.ctors.insert(kind.to_string(), Box::new(ctor));
    }

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

    pub fn contains(&self, kind: &str) -> bool {
        self.ctors.contains_key(kind)
    }

    /// Returns an iterator over all registered node kinds.
    pub fn registered_kinds(&self) -> impl Iterator<Item = &str> {
        self.ctors.keys().map(|k| k.as_str())
    }

    /// Instantiate the node described by `spec`.
    pub fn build(&self, spec: &NodeSpec) -> Result<Box<dyn Node>, BuildError> {
        let ctor = self
            .ctors
            .get(&spec.kind)
            .ok_or_else(|| BuildError::UnknownKind(spec.kind.clone()))?;
        ctor(&spec.params)
    }

    /// The port set a node of this spec would declare. Builds a throwaway node,
    /// because port shape can depend on params (e.g. Split/Merge port counts).
    pub fn ports_of(&self, spec: &NodeSpec) -> Result<PortSet, BuildError> {
        Ok(self.build(spec)?.ports())
    }
}

/// A registry preloaded with the built-in stage kinds.
///
/// More kinds (crop/cast/split/merge) are registered as those stages land.
pub fn builtin_registry() -> Registry {
    let mut reg = Registry::new();

    reg.register("clear_channel", |p| {
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

    reg.register("crop", |p| {
        Ok(Box::new(CropStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register("cast", |p| {
        Ok(Box::new(CastStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register("split", |p| {
        Ok(Box::new(SplitStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register("merge", |p| {
        Ok(Box::new(MergeStage::try_from(p)?) as Box<dyn Node>)
    });

    reg.register("grayscale", |_| {
        Ok(Box::new(ProcessorNode::new(Grayscale)) as Box<dyn Node>)
    });
    reg.register("invert", |_| {
        Ok(Box::new(ProcessorNode::new(Invert)) as Box<dyn Node>)
    });
    reg.register("blend", |p| {
        Ok(Box::new(BlendStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register("image_read", |p| {
        Ok(Box::new(ImageReadStage::try_from(p)?) as Box<dyn Node>)
    });
    reg.register("image_write", |p| {
        Ok(Box::new(ImageWriteStage::try_from(p)?) as Box<dyn Node>)
    });

    reg
}

#[cfg(test)]
mod tests {
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
}
