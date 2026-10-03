//! Gain: multiply every sample by a constant `factor`.
//!
//! Semantics per dtype:
//! - `u8`: `round(v * factor)` saturated to `[0, 255]`. Because there are only
//!   256 possible inputs, the stage precomputes a lookup table once at build
//!   time, so evaluation is a single table load per sample.
//! - `f32`: plain `v * factor`, **unclamped**. Float frames are allowed to
//!   carry out-of-range intermediates; clamping is the job of a later `cast`
//!   back to `u8`.
//!
//! Channels and dtype are preserved; every channel (including alpha) is scaled.

use crate::data::{Frame, FrameData, Payload, PayloadKind};
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec};
use crate::graph::Params;

/// 1→1 stage that scales every sample by `factor`.
pub struct GainStage {
    factor: f32,
    /// `lut[v]` = gained, saturated `u8` value for input `v`.
    lut: [u8; 256],
}

impl GainStage {
    /// Panics if `factor` is not finite (NaN/inf would make both the `u8` LUT
    /// and the `f32` output meaningless); `TryFrom<&Params>` is the fallible
    /// build path.
    pub fn new(factor: f32) -> Self {
        assert!(factor.is_finite(), "gain factor must be finite");
        let mut lut = [0u8; 256];
        for (v, slot) in lut.iter_mut().enumerate() {
            *slot = (v as f32 * factor).round().clamp(0.0, 255.0) as u8;
        }
        Self { factor, lut }
    }
}

impl TryFrom<&Params> for GainStage {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        let factor = p.get_f32("factor")?;
        if !factor.is_finite() {
            return Err(BuildError::BadParam {
                key: "factor".to_string(),
                value: p.get_str("factor")?.to_string(),
                expected: "finite f32",
            });
        }
        Ok(Self::new(factor))
    }
}

impl Node for GainStage {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Frame)],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;
        let data = match frame.data() {
            FrameData::U8(buf) => {
                FrameData::U8(buf.iter().map(|&v| self.lut[v as usize]).collect())
            }
            FrameData::F32(buf) => FrameData::F32(buf.iter().map(|&v| v * self.factor).collect()),
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

    fn run(stage: &mut GainStage, frame: Frame) -> Frame {
        let mut m = HashMap::new();
        m.insert(PortId("in".to_string()), Payload::Frame(frame));
        let mut out = Outputs::new();
        stage.eval(&Inputs::new(m), &mut out).unwrap();
        out.get("out").unwrap().as_frame().unwrap().clone()
    }

    #[test]
    fn u8_rounds_and_saturates() {
        let f = Frame::from_data(4, 1, 1, FrameData::U8(vec![0, 10, 101, 200]));
        let out = run(&mut GainStage::new(1.5), f);
        // 101 * 1.5 = 151.5 rounds to 152; 200 * 1.5 = 300 saturates.
        assert_eq!(out.as_u8().unwrap(), &[0, 15, 152, 255]);
    }

    #[test]
    fn u8_negative_factor_clamps_to_zero() {
        let f = Frame::from_data(2, 1, 1, FrameData::U8(vec![0, 200]));
        let out = run(&mut GainStage::new(-1.0), f);
        assert_eq!(out.as_u8().unwrap(), &[0, 0]);
    }

    #[test]
    fn f32_is_unclamped() {
        let f = Frame::from_data(1, 1, 2, FrameData::F32(vec![0.5, 0.75]));
        let out = run(&mut GainStage::new(2.0), f);
        assert_eq!(out.as_f32().unwrap(), &[1.0, 1.5]);
    }

    #[test]
    fn rejects_non_finite_factor() {
        let mut p = Params::new();
        p.insert("factor".to_string(), "NaN".to_string());
        assert!(matches!(
            GainStage::try_from(&p),
            Err(BuildError::BadParam { .. })
        ));
    }
}
