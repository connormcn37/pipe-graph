//! The `Node` execution unit and its port model.
//!
//! `graph::Graph` describes *topology* (which node's port connects to which).
//! A `Node` is the *behavior* behind a node: it declares named input/output
//! ports (with payload kinds) via [`Node::ports`] and computes outputs from
//! inputs via [`Node::eval`]. Unlike [`crate::traits::Processor`] (a single
//! in-place `&mut Frame`), a `Node` is N-in / M-out over named ports, which is
//! what Split/Merge and arbitrary graphs require.
//!
//! Any existing `Processor` becomes a 1-in/1-out `Node` for free via
//! [`ProcessorNode`], so the working processor stack is reused unchanged.

use std::collections::HashMap;

use crate::data::{Frame, Payload, PayloadKind};
use crate::graph::PortId;
use crate::traits::Processor;

/// Declaration of a single named port and the payload kind it carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortSpec {
    pub id: PortId,
    pub kind: PayloadKind,
}

impl PortSpec {
    pub fn new(id: impl Into<String>, kind: PayloadKind) -> Self {
        Self {
            id: PortId(id.into()),
            kind,
        }
    }
}

/// A node's static (per-instance) declaration of its inputs and outputs.
///
/// It is computed from the built node (`&self`) rather than being a constant,
/// because stages like Split/Merge have a port count that depends on their
/// parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PortSet {
    pub inputs: Vec<PortSpec>,
    pub outputs: Vec<PortSpec>,
}

impl PortSet {
    pub fn new(inputs: Vec<PortSpec>, outputs: Vec<PortSpec>) -> Self {
        Self { inputs, outputs }
    }

    pub fn find_input(&self, id: &str) -> Option<&PortSpec> {
        self.inputs.iter().find(|p| p.id.0 == id)
    }

    pub fn find_output(&self, id: &str) -> Option<&PortSpec> {
        self.outputs.iter().find(|p| p.id.0 == id)
    }
}

/// Errors a node can raise while evaluating.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeError {
    /// A required input port had no value available this evaluation.
    MissingInput(String),
    /// An input carried the wrong payload kind.
    WrongPayload {
        port: String,
        expected: PayloadKind,
        got: PayloadKind,
    },
    /// A stage-specific failure (bad shape, mismatched inputs, etc.).
    Message(String),
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeError::MissingInput(p) => write!(f, "missing input on port '{p}'"),
            NodeError::WrongPayload {
                port,
                expected,
                got,
            } => write!(f, "port '{port}' expected {expected:?} but got {got:?}"),
            NodeError::Message(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for NodeError {}

/// The values presented to a node's input ports for one evaluation.
///
/// A port may be absent (e.g. a feedback edge on the first tick), so lookups
/// return `Option`; helpers like [`Inputs::frame`] turn "absent" or
/// "wrong kind" into a [`NodeError`].
#[derive(Debug, Default)]
pub struct Inputs {
    values: HashMap<PortId, Payload>,
}

impl Inputs {
    pub fn new(values: HashMap<PortId, Payload>) -> Self {
        Self { values }
    }

    /// Raw lookup; `None` if the port has no value this evaluation.
    pub fn get(&self, port: &str) -> Option<&Payload> {
        self.values.get(port)
    }

    /// Require a `Frame` on `port`, erroring if absent or the wrong kind.
    pub fn frame(&self, port: &str) -> Result<&Frame, NodeError> {
        match self.get(port) {
            None => Err(NodeError::MissingInput(port.to_string())),
            Some(p) => p.as_frame().ok_or_else(|| NodeError::WrongPayload {
                port: port.to_string(),
                expected: PayloadKind::Frame,
                got: p.kind(),
            }),
        }
    }
}

/// The values a node produces on its output ports for one evaluation.
#[derive(Debug, Default)]
pub struct Outputs {
    values: HashMap<PortId, Payload>,
}

impl Outputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, port: &str, payload: Payload) {
        self.values.insert(PortId(port.to_string()), payload);
    }

    pub fn get(&self, port: &str) -> Option<&Payload> {
        self.values.get(port)
    }

    /// Consume the outputs as a raw port→payload map (used by the scheduler).
    pub fn into_map(self) -> HashMap<PortId, Payload> {
        self.values
    }
}

/// The unit of execution behind a graph node.
///
/// Note: no `Send` bound yet — the initial scheduler is single-threaded. It can
/// be added when parallel component execution lands, at which point the
/// `Processor` stack it wraps would gain the same bound.
pub trait Node {
    /// Declare this node's input/output ports (may depend on `self`/params).
    fn ports(&self) -> PortSet;

    /// Compute outputs from inputs. Called once per run (acyclic) or once per
    /// tick (cyclic).
    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError>;

    /// Reset internal state (e.g. seed feedback buffers before a fresh run).
    fn reset(&mut self) {}
}

/// Adapts any single-in/single-out [`Processor`] into a [`Node`] with ports
/// `"in"` and `"out"`, both carrying frames. Keeps `ClearChannel`,
/// `ProcessList`, etc. usable in a graph with zero changes.
pub struct ProcessorNode<P: Processor> {
    inner: P,
}

impl<P: Processor> ProcessorNode<P> {
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

impl<P: Processor> Node for ProcessorNode<P> {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Frame)],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let mut frame = inputs.frame("in")?.clone();
        self.inner.process(&mut frame);
        outputs.set("out", Payload::Frame(frame));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::{Channel, ClearChannel};

    fn inputs_with(port: &str, payload: Payload) -> Inputs {
        let mut m = HashMap::new();
        m.insert(PortId(port.to_string()), payload);
        Inputs::new(m)
    }

    #[test]
    fn processor_node_declares_in_out_frame_ports() {
        let node = ProcessorNode::new(ClearChannel(Channel::Red));
        let ports = node.ports();
        assert_eq!(ports.find_input("in").unwrap().kind, PayloadKind::Frame);
        assert_eq!(ports.find_output("out").unwrap().kind, PayloadKind::Frame);
    }

    #[test]
    fn processor_node_eval_matches_direct_process() {
        let frame = Frame::from_rgb8(1, 2, vec![(255, 10, 20), (30, 40, 50)]);

        // Through the node adapter.
        let mut node = ProcessorNode::new(ClearChannel(Channel::Red));
        let inputs = inputs_with("in", Payload::Frame(frame.clone()));
        let mut outputs = Outputs::new();
        node.eval(&inputs, &mut outputs).unwrap();
        let via_node = outputs.get("out").unwrap().as_frame().unwrap().to_rgb8();

        // Directly.
        let mut direct = frame;
        ClearChannel(Channel::Red).process(&mut direct);

        assert_eq!(via_node, direct.to_rgb8());
        assert_eq!(via_node, vec![(0, 10, 20), (0, 40, 50)]);
    }

    #[test]
    fn missing_input_is_an_error() {
        let mut node = ProcessorNode::new(ClearChannel(Channel::Red));
        let inputs = Inputs::default();
        let mut outputs = Outputs::new();
        let err = node.eval(&inputs, &mut outputs).unwrap_err();
        assert_eq!(err, NodeError::MissingInput("in".to_string()));
    }

    #[test]
    fn wrong_payload_kind_is_an_error() {
        let mut node = ProcessorNode::new(ClearChannel(Channel::Red));
        let inputs = inputs_with("in", Payload::Scalar(1.0));
        let mut outputs = Outputs::new();
        let err = node.eval(&inputs, &mut outputs).unwrap_err();
        assert_eq!(
            err,
            NodeError::WrongPayload {
                port: "in".to_string(),
                expected: PayloadKind::Frame,
                got: PayloadKind::Scalar,
            }
        );
    }
}
