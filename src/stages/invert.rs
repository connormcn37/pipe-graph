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

#[cfg(feature = "gpu")]
use crate::data::DType;
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

    /// `u8` frames invert a whole packed word at a time (`!w` is `255 - v`
    /// for each of its four bytes); `f32` is `1.0 - v`, as on the CPU.
    #[cfg(feature = "gpu")]
    fn eval_gpu(
        &mut self,
        ctx: &crate::gpu::GpuContext,
        inputs: &Inputs,
        outputs: &mut Outputs,
    ) -> Option<Result<(), NodeError>> {
        Some((|| {
            let frame = inputs.frame("in")?;
            let src = frame.to_gpu(ctx)?;
            let kernel = match frame.dtype() {
                DType::U8 => ctx.kernel("invert_u8", gpu::INVERT_U8),
                DType::F32 => ctx.kernel("invert_f32", gpu::INVERT_F32),
            };
            let dst = crate::gpu::GpuBuffer::zeroed(ctx, src.dtype(), src.len());
            kernel.run(ctx, &[src.buffer(), dst.buffer()], dst.words());
            let out = Frame::from_gpu(frame.width, frame.height, frame.channels, dst);
            outputs.set("out", Payload::Frame(out));
            Ok(())
        })())
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    pub(super) const INVERT_U8: &str = "
        @group(0) @binding(0) var<storage, read> src: array<u32>;
        @group(0) @binding(1) var<storage, read_write> dst: array<u32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(num_workgroups) nwg: vec3<u32>) {
            let i = gid.x + gid.y * nwg.x * 64u;
            if (i < arrayLength(&dst)) { dst[i] = ~src[i]; }
        }";

    pub(super) const INVERT_F32: &str = "
        @group(0) @binding(0) var<storage, read> src: array<f32>;
        @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(num_workgroups) nwg: vec3<u32>) {
            let i = gid.x + gid.y * nwg.x * 64u;
            if (i < arrayLength(&dst)) { dst[i] = 1.0 - src[i]; }
        }";
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

    /// The GPU path agrees with the CPU path, from CPU and from GPU inputs,
    /// on sizes that aren't a whole number of words or workgroups.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_matches_cpu() {
        use crate::data::Location;
        let Some(ctx) = crate::gpu::tests::context() else {
            return;
        };
        let frames = [
            Frame::from_data(7, 3, 3, FrameData::U8((0..63).map(|v| v * 4).collect())),
            Frame::from_data(
                5,
                13,
                1,
                FrameData::F32((0..65).map(|v| v as f32 / 64.0).collect()),
            ),
        ];
        for frame in frames {
            let expected = run(frame.clone());
            for input in [frame.clone(), frame.clone().into_gpu(&ctx).unwrap()] {
                let mut m = HashMap::new();
                m.insert(PortId("in".to_string()), Payload::Frame(input));
                let mut out = Outputs::new();
                InvertStage::new()
                    .eval_gpu(&ctx, &Inputs::new(m), &mut out)
                    .expect("invert has a GPU path")
                    .unwrap();
                let got = out.get("out").unwrap().as_frame().unwrap().clone();
                assert_eq!(got.location(), Location::Gpu);
                assert_eq!(got.into_cpu().unwrap(), expected);
            }
        }
    }
}
