//! Split: a k-channel frame `(w, h, k)` into k single-channel frames `(w, h, 1)`.

use crate::data::{Frame, FrameData, Payload, PayloadKind};
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec};
use crate::graph::Params;

/// 1→k stage. Declares output ports `out0 .. out{k-1}`.
pub struct SplitStage {
    channels: u32,
}

impl SplitStage {
    pub fn new(channels: u32) -> Self {
        Self { channels }
    }
}

impl TryFrom<&Params> for SplitStage {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        Ok(Self {
            channels: p.get_u32("channels")?,
        })
    }
}

fn extract_channel<T: Copy>(src: &[T], k: usize, c: usize, pixels: usize) -> Vec<T> {
    (0..pixels).map(|p| src[p * k + c]).collect()
}

impl Node for SplitStage {
    fn ports(&self) -> PortSet {
        let outputs = (0..self.channels)
            .map(|c| PortSpec::new(format!("out{c}"), PayloadKind::Frame))
            .collect();
        PortSet::new(vec![PortSpec::new("in", PayloadKind::Frame)], outputs)
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;
        let k = self.channels;

        if frame.channels != k {
            return Err(NodeError::Message(format!(
                "split expected {k} channels, got {}",
                frame.channels
            )));
        }

        let (w, h) = (frame.width, frame.height);
        let pixels = frame.pixel_count();

        for c in 0..k as usize {
            let data = match frame.data() {
                FrameData::U8(buf) => FrameData::U8(extract_channel(buf, k as usize, c, pixels)),
                FrameData::F32(buf) => FrameData::F32(extract_channel(buf, k as usize, c, pixels)),
            };
            let channel_frame = Frame::from_data(w, h, 1, data);
            outputs.set(&format!("out{c}"), Payload::Frame(channel_frame));
        }

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
        let ports = SplitStage::new(4).ports();
        assert_eq!(ports.inputs.len(), 1);
        assert_eq!(ports.outputs.len(), 4);
        assert!(ports.find_output("out3").is_some());
        assert!(ports.find_output("out4").is_none());
    }

    #[test]
    fn wrong_channel_count_is_an_error() {
        let mut stage = SplitStage::new(3);
        let mut m = HashMap::new();
        m.insert(
            PortId("in".to_string()),
            Payload::Frame(Frame::from_data(1, 1, 2, FrameData::U8(vec![1, 2]))),
        );
        let mut out = Outputs::new();
        assert!(matches!(
            stage.eval(&Inputs::new(m), &mut out),
            Err(NodeError::Message(_))
        ));
    }
}
