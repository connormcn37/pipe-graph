//! Merge: k single-channel frames `(w, h, 1)` into one k-channel frame `(w, h, k)`.

use crate::data::{DType, Frame, FrameData, Payload, PayloadKind};
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec};
use crate::graph::Params;

/// k→1 stage. Declares input ports `in0 .. in{k-1}`.
pub struct MergeStage {
    channels: u32,
}

impl MergeStage {
    pub fn new(channels: u32) -> Self {
        Self { channels }
    }
}

impl TryFrom<&Params> for MergeStage {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        Ok(Self {
            channels: p.get_u32("channels")?,
        })
    }
}

impl Node for MergeStage {
    fn ports(&self) -> PortSet {
        let inputs = (0..self.channels)
            .map(|c| PortSpec::new(format!("in{c}"), PayloadKind::Frame))
            .collect();
        PortSet::new(inputs, vec![PortSpec::new("out", PayloadKind::Frame)])
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let k = self.channels as usize;

        // Collect and validate the k single-channel inputs.
        let first = inputs.frame("in0")?;
        let (w, h, dtype) = (first.width, first.height, first.dtype());
        let pixels = first.pixel_count();

        let mut frames = Vec::with_capacity(k);
        for c in 0..k {
            let fr = inputs.frame(&format!("in{c}"))?;
            if fr.width != w || fr.height != h {
                return Err(NodeError::Message(
                    "merge inputs differ in dimensions".to_string(),
                ));
            }
            if fr.channels != 1 {
                return Err(NodeError::Message(
                    "merge inputs must be single-channel".to_string(),
                ));
            }
            if fr.dtype() != dtype {
                return Err(NodeError::Message(
                    "merge inputs differ in dtype".to_string(),
                ));
            }
            frames.push(fr);
        }

        let data = match dtype {
            DType::U8 => {
                let mut out = vec![0u8; pixels * k];
                for (c, fr) in frames.iter().enumerate() {
                    let buf = fr.as_u8().unwrap();
                    for p in 0..pixels {
                        out[p * k + c] = buf[p];
                    }
                }
                FrameData::U8(out)
            }
            DType::F32 => {
                let mut out = vec![0.0f32; pixels * k];
                for (c, fr) in frames.iter().enumerate() {
                    let buf = fr.as_f32().unwrap();
                    for p in 0..pixels {
                        out[p * k + c] = buf[p];
                    }
                }
                FrameData::F32(out)
            }
        };

        outputs.set(
            "out",
            Payload::Frame(Frame::from_data(w, h, k as u32, data)),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PortId;
    use std::collections::HashMap;

    #[test]
    fn ports_scale_with_channel_count() {
        let ports = MergeStage::new(3).ports();
        assert_eq!(ports.inputs.len(), 3);
        assert_eq!(ports.outputs.len(), 1);
        assert!(ports.find_input("in2").is_some());
    }

    #[test]
    fn mismatched_dtype_is_an_error() {
        let mut stage = MergeStage::new(2);
        let mut m = HashMap::new();
        m.insert(
            PortId("in0".to_string()),
            Payload::Frame(Frame::from_data(1, 1, 1, FrameData::U8(vec![1]))),
        );
        m.insert(
            PortId("in1".to_string()),
            Payload::Frame(Frame::from_data(1, 1, 1, FrameData::F32(vec![1.0]))),
        );
        let mut out = Outputs::new();
        assert!(matches!(
            stage.eval(&Inputs::new(m), &mut out),
            Err(NodeError::Message(_))
        ));
    }
}
