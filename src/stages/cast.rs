//! Cast: convert a frame between `u8` and `f32`.
//!
//! Defaults follow the usual image convention: `u8 -> f32` maps `[0, 255]` to
//! `[0.0, 1.0]` (scale `1/255`), and `f32 -> u8` maps `[0.0, 1.0]` back to
//! `[0, 255]` (scale `255`, rounded and clamped). An explicit `scale` param
//! overrides the default. Casting to the same dtype is an identity copy.

use crate::data::{DType, Frame, FrameData, Payload, PayloadKind};
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec};
use crate::graph::Params;

/// 1→1 stage that converts the input frame's dtype.
pub struct CastStage {
    dtype: DType,
    scale: Option<f32>,
}

impl CastStage {
    pub fn new(dtype: DType, scale: Option<f32>) -> Self {
        Self { dtype, scale }
    }
}

impl TryFrom<&Params> for CastStage {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        let dtype = match p.get_str("dtype")? {
            "u8" => DType::U8,
            "f32" => DType::F32,
            other => {
                return Err(BuildError::BadParam {
                    key: "dtype".to_string(),
                    value: other.to_string(),
                    expected: "u8|f32",
                });
            }
        };
        let scale = match p.get("scale") {
            Some(_) => Some(p.get_f32("scale")?),
            None => None,
        };
        Ok(Self { dtype, scale })
    }
}

impl Node for CastStage {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Frame)],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;
        let (w, h, ch) = (frame.width, frame.height, frame.channels);

        let out = match (frame.dtype(), self.dtype) {
            (DType::U8, DType::F32) => {
                let scale = self.scale.unwrap_or(1.0 / 255.0);
                let buf: Vec<f32> = frame
                    .as_u8()
                    .unwrap()
                    .iter()
                    .map(|&v| v as f32 * scale)
                    .collect();
                Frame::from_data(w, h, ch, FrameData::F32(buf))
            }
            (DType::F32, DType::U8) => {
                let scale = self.scale.unwrap_or(255.0);
                let buf: Vec<u8> = frame
                    .as_f32()
                    .unwrap()
                    .iter()
                    .map(|&v| (v * scale).round().clamp(0.0, 255.0) as u8)
                    .collect();
                Frame::from_data(w, h, ch, FrameData::U8(buf))
            }
            // Same dtype: identity copy (scale ignored).
            _ => frame.clone(),
        };

        outputs.set("out", Payload::Frame(out));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PortId;
    use std::collections::HashMap;

    fn run(stage: &mut CastStage, frame: Frame) -> Frame {
        let mut m = HashMap::new();
        m.insert(PortId("in".to_string()), Payload::Frame(frame));
        let mut out = Outputs::new();
        stage.eval(&Inputs::new(m), &mut out).unwrap();
        out.get("out").unwrap().as_frame().unwrap().clone()
    }

    #[test]
    fn u8_to_f32_normalizes() {
        let f = Frame::from_rgb8(1, 1, vec![(255, 0, 128)]);
        let out = run(&mut CastStage::new(DType::F32, None), f);
        let d = out.as_f32().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6);
        assert_eq!(d[1], 0.0);
        assert!((d[2] - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn f32_to_u8_denormalizes_and_clamps() {
        let f = Frame::from_data(1, 1, 3, FrameData::F32(vec![1.0, 0.0, 2.0]));
        let out = run(&mut CastStage::new(DType::U8, None), f);
        // 2.0 * 255 clamps to 255.
        assert_eq!(out.as_u8().unwrap(), &[255, 0, 255]);
    }

    #[test]
    fn same_dtype_is_identity() {
        let f = Frame::from_rgb8(1, 1, vec![(3, 4, 5)]);
        let out = run(&mut CastStage::new(DType::U8, None), f.clone());
        assert_eq!(out, f);
    }
}
