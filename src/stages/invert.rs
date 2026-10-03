//! Invert: photographic negative of every sample.
//!
//! Semantics per dtype:
//! - `u8`: `255 - v` (exact, never overflows).
//! - `f32`: `1.0 - v`. This assumes the conventional normalized `[0, 1]`
//!   range produced by `cast` (`u8 -> f32`); values outside it are mirrored
//!   about the same axis and are **not** clamped.
//!
//! Every channel is inverted, including alpha if present — split it off first
//! if it should be preserved.

use crate::data::{Frame, FrameData, Payload, PayloadKind};
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, PortSet, PortSpec};
use crate::graph::Params;

/// 1→1 stage that inverts every sample. Takes no parameters.
#[derive(Default)]
pub struct InvertStage;

impl InvertStage {
    pub fn new() -> Self {
        Self
    }
}

impl TryFrom<&Params> for InvertStage {
    type Error = BuildError;

    fn try_from(_: &Params) -> Result<Self, BuildError> {
        Ok(Self)
    }
}

impl Node for InvertStage {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Frame)],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;
        let data = match frame.data() {
            FrameData::U8(buf) => FrameData::U8(buf.iter().map(|&v| 255 - v).collect()),
            FrameData::F32(buf) => FrameData::F32(buf.iter().map(|&v| 1.0 - v).collect()),
        };
        let out = Frame::from_data(frame.width, frame.height, frame.channels, data);
        outputs.set("out", Payload::Frame(out));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PortId;
    use std::collections::HashMap;

    fn run(frame: Frame) -> Frame {
        let mut m = HashMap::new();
        m.insert(PortId("in".to_string()), Payload::Frame(frame));
        let mut out = Outputs::new();
        InvertStage::new().eval(&Inputs::new(m), &mut out).unwrap();
        out.get("out").unwrap().as_frame().unwrap().clone()
    }

    #[test]
    fn inverts_u8() {
        let f = Frame::from_rgb8(1, 1, vec![(0, 100, 255)]);
        assert_eq!(run(f).to_rgb8(), vec![(255, 155, 0)]);
    }

    #[test]
    fn inverts_f32_about_one() {
        let f = Frame::from_data(3, 1, 1, FrameData::F32(vec![0.0, 0.25, 1.0]));
        assert_eq!(run(f).as_f32().unwrap(), &[1.0, 0.75, 0.0]);
    }
}
