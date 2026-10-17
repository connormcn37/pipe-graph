//! Threshold: binarize every sample against a `level`.
//!
//! `level` is expressed in **normalized** units (`[0, 1]`, the scale the `f32`
//! pipeline uses), so one graph param means the same thing regardless of the
//! dtype flowing through it:
//! - `u8`: `v >= level * 255` → `255`, else `0`.
//! - `f32`: `v >= level` → `1.0`, else `0.0` (input assumed in `[0, 1]`).
//!
//! The comparison is inclusive (a sample exactly at the level is "on"). Every
//! channel is thresholded independently; dtype and channel count are kept.

use crate::data::{Frame, FrameData, Payload, PayloadKind};
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec};
use crate::exec::{KindInfo, ParamKind, ParamSpec};
use crate::graph::Params;

/// 1→1 stage that maps each sample to "off" or "on".
pub struct ThresholdStage {
    level: f32,
    /// Precomputed `u8` result for each of the 256 possible inputs.
    lut: [u8; 256],
}

impl ThresholdStage {
    /// Panics if `level` is NaN (every comparison would be false);
    /// `TryFrom<&Params>` is the fallible build path.
    pub fn new(level: f32) -> Self {
        assert!(!level.is_nan(), "threshold level must not be NaN");
        let cut = level * 255.0;
        let mut lut = [0u8; 256];
        for (v, slot) in lut.iter_mut().enumerate() {
            *slot = if v as f32 >= cut { 255 } else { 0 };
        }
        Self { level, lut }
    }
}

impl ThresholdStage {
    /// Parameter schema for this kind; kept beside `try_from` so the two
    /// change together (cross-checked in `exec::registry` tests).
    pub fn info() -> KindInfo {
        KindInfo::new(
            "threshold",
            "Binarize every sample against a normalized level (inclusive).",
            vec![
                // Normalized, but deliberately not clamped to [0, 1]: f32
                // frames may carry out-of-range values (e.g. after `gain`),
                // where a level above 1 is meaningful. (For u8 frames such a
                // level just yields a constant output.)
                ParamSpec::required(
                    "level",
                    ParamKind::F32 {
                        min: None,
                        max: None,
                    },
                    "Cut-off in normalized units (typically 0..1); u8 compares against level * 255.",
                )
                .with_default("0.5"),
            ],
        )
    }
}

impl TryFrom<&Params> for ThresholdStage {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        let level = p.get_f32("level")?;
        if level.is_nan() {
            return Err(BuildError::BadParam {
                key: "level".to_string(),
                value: p.get_str("level")?.to_string(),
                expected: "non-NaN f32",
            });
        }
        Ok(Self::new(level))
    }
}

impl Node for ThresholdStage {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Frame)],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    #[cfg(feature = "gpu")]
    fn supports_gpu(&self) -> bool {
        true
    }

    /// Bit-identical to `eval`: `u8` goes through the same table.
    #[cfg(feature = "gpu")]
    fn eval_gpu(
        &mut self,
        ctx: &crate::gpu::GpuContext,
        inputs: &Inputs,
        outputs: &mut Outputs,
    ) -> Option<Result<(), NodeError>> {
        use super::gpu_kernels as k;
        use crate::data::DType;
        use crate::gpu::GpuBuffer;
        Some((|| {
            let frame = inputs.frame("in")?;
            let src = frame.to_gpu(ctx)?;
            let dst = GpuBuffer::zeroed(ctx, src.dtype(), src.len());
            let bufs = [src.buffer(), dst.buffer()];
            match src.dtype() {
                DType::U8 => k::dispatch(
                    ctx,
                    k::LUT_U8,
                    &bufs,
                    &k::lut_params(&self.lut),
                    k::threads(&dst),
                ),
                DType::F32 => k::dispatch(
                    ctx,
                    k::THRESHOLD_F32,
                    &bufs,
                    &[self.level.to_bits()],
                    k::threads(&dst),
                ),
            }
            let out = Frame::from_gpu(frame.width, frame.height, frame.channels, dst);
            outputs.set("out", Payload::Frame(out));
            Ok(())
        })())
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;
        let level = self.level;
        let data = match frame.data() {
            FrameData::U8(buf) => {
                FrameData::U8(buf.iter().map(|&v| self.lut[v as usize]).collect())
            }
            FrameData::F32(buf) => FrameData::F32(
                buf.iter()
                    .map(|&v| if v >= level { 1.0 } else { 0.0 })
                    .collect(),
            ),
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

    fn run(stage: &mut ThresholdStage, frame: Frame) -> Frame {
        let mut m = HashMap::new();
        m.insert(PortId("in".to_string()), Payload::Frame(frame));
        let mut out = Outputs::new();
        stage.eval(&Inputs::new(m), &mut out).unwrap();
        out.get("out").unwrap().as_frame().unwrap().clone()
    }

    #[test]
    fn u8_level_is_normalized_and_inclusive() {
        // 0.5 * 255 = 127.5: 127 is below, 128 is above.
        let f = Frame::from_data(4, 1, 1, FrameData::U8(vec![0, 127, 128, 255]));
        let out = run(&mut ThresholdStage::new(0.5), f);
        assert_eq!(out.as_u8().unwrap(), &[0, 0, 255, 255]);
    }

    #[test]
    fn f32_compares_directly_and_inclusively() {
        let f = Frame::from_data(3, 1, 1, FrameData::F32(vec![0.2, 0.5, 0.9]));
        let out = run(&mut ThresholdStage::new(0.5), f);
        assert_eq!(out.as_f32().unwrap(), &[0.0, 1.0, 1.0]);
    }

    #[test]
    fn missing_level_errors() {
        assert_eq!(
            ThresholdStage::try_from(&Params::new()).err(),
            Some(BuildError::MissingParam("level".to_string()))
        );
    }
}
