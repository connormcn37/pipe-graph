//! Grayscale: collapse RGB(A) into a single BT.601 luma channel.
//!
//! `Y = 0.299 R + 0.587 G + 0.114 B` (ITU-R BT.601, the classic luma used by
//! JPEG and SD video). Input must be 3-channel (RGB) or 4-channel (RGBA) and
//! the output is always **1 channel**. For RGBA the alpha channel is
//! **dropped**: a 1-channel frame has nowhere to keep it, so split alpha off
//! beforehand and merge it back if it is needed. Any other channel count is a
//! [`NodeError::Message`], since there is no unambiguous luma for it.
//!
//! Per dtype:
//! - `u8`: exact integer arithmetic with the weights scaled by 1000 and the
//!   result rounded half-up: `(299 R + 587 G + 114 B + 500) / 1000`. The
//!   weights sum to 1000, so the result always fits in `[0, 255]`.
//! - `f32`: the same weighted sum in float, unclamped (inputs assumed `[0, 1]`).

use crate::data::{Frame, FrameData, Payload, PayloadKind};
use crate::exec::KindInfo;
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, PortSet, PortSpec};
use crate::graph::Params;

/// 1→1 stage: `(w, h, 3|4)` → `(w, h, 1)` luma. Takes no parameters.
#[derive(Default)]
pub struct GrayscaleStage;

impl GrayscaleStage {
    pub fn new() -> Self {
        Self
    }
}

impl GrayscaleStage {
    /// Parameter schema for this kind; kept beside `try_from` so the two
    /// change together (cross-checked in `exec::registry` tests).
    pub fn info() -> KindInfo {
        KindInfo::new(
            "grayscale",
            "RGB(A) to 1-channel BT.601 luma (alpha dropped).",
            vec![],
        )
    }
}

impl TryFrom<&Params> for GrayscaleStage {
    type Error = BuildError;

    fn try_from(_: &Params) -> Result<Self, BuildError> {
        Ok(Self)
    }
}

/// The input's channel count, if grayscale has a luma for it (3 or 4).
fn check_channels(frame: &Frame) -> Result<usize, NodeError> {
    let k = frame.channels as usize;
    if k != 3 && k != 4 {
        return Err(NodeError::Message(format!(
            "grayscale needs a 3- or 4-channel frame, got {k} channels"
        )));
    }
    Ok(k)
}

impl Node for GrayscaleStage {
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

    /// Bit-identical to `eval` for `u8` (integer arithmetic); `f32` may differ
    /// in the last bit where the device fuses multiply-adds.
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
            let k = check_channels(frame)? as u32;
            let pixels = frame.pixel_count();
            let src = frame.to_gpu(ctx)?;
            let dst = GpuBuffer::zeroed(ctx, src.dtype(), pixels);
            let bufs = [src.buffer(), dst.buffer()];
            match src.dtype() {
                DType::U8 => k::dispatch(
                    ctx,
                    k::GRAYSCALE_U8,
                    &bufs,
                    &[k, pixels as u32],
                    k::threads(&dst),
                ),
                DType::F32 => k::dispatch(ctx, k::GRAYSCALE_F32, &bufs, &[k], k::threads(&dst)),
            }
            let out = Frame::from_gpu(frame.width, frame.height, 1, dst);
            outputs.set("out", Payload::Frame(out));
            Ok(())
        })())
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;
        let k = check_channels(frame)?;

        let data = match frame.data() {
            FrameData::U8(buf) => FrameData::U8(
                buf.chunks_exact(k)
                    .map(|px| {
                        let y = 299 * px[0] as u32 + 587 * px[1] as u32 + 114 * px[2] as u32;
                        ((y + 500) / 1000) as u8
                    })
                    .collect(),
            ),
            FrameData::F32(buf) => FrameData::F32(
                buf.chunks_exact(k)
                    .map(|px| 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2])
                    .collect(),
            ),
        };

        let out = Frame::from_data(frame.width, frame.height, 1, data);
        outputs.set("out", Payload::Frame(out));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PortId;
    use std::collections::HashMap;

    fn run(frame: Frame) -> Result<Frame, NodeError> {
        let mut m = HashMap::new();
        m.insert(PortId("in".to_string()), Payload::Frame(frame));
        let mut out = Outputs::new();
        GrayscaleStage::new().eval(&Inputs::new(m), &mut out)?;
        Ok(out.get("out").unwrap().as_frame().unwrap().clone())
    }

    #[test]
    fn rgb_u8_luma() {
        let f = Frame::from_rgb8(
            4,
            1,
            vec![(255, 255, 255), (255, 0, 0), (0, 255, 0), (10, 20, 30)],
        );
        let out = run(f).unwrap();
        assert_eq!(out.channels, 1);
        // red: 76.245 -> 76; green: 149.685 -> 150;
        // (10,20,30): (2990 + 11740 + 3420 + 500) / 1000 = 18.
        assert_eq!(out.as_u8().unwrap(), &[255, 76, 150, 18]);
    }

    #[test]
    fn rgba_drops_alpha() {
        let f = Frame::from_data(1, 1, 4, FrameData::U8(vec![0, 0, 255, 7]));
        let out = run(f).unwrap();
        assert_eq!(out.channels, 1);
        // 114 * 255 / 1000 = 29.07 -> 29; alpha ignored.
        assert_eq!(out.as_u8().unwrap(), &[29]);
    }

    #[test]
    fn rgb_f32_luma() {
        let f = Frame::from_data(1, 1, 3, FrameData::F32(vec![1.0, 0.5, 0.0]));
        let out = run(f).unwrap();
        assert!((out.as_f32().unwrap()[0] - (0.299 + 0.2935)).abs() < 1e-6);
    }

    #[test]
    fn wrong_channel_count_is_an_error() {
        let f = Frame::from_data(1, 1, 2, FrameData::U8(vec![1, 2]));
        assert!(matches!(run(f), Err(NodeError::Message(_))));
    }
}
