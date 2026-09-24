use crate::data::{DType, Frame, FrameData, Payload, PayloadKind};
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, PortSet, PortSpec, Registry};
use crate::graph::Params;

/// Averages two same-shaped frames (`in0`, `in1`) into `out`.
#[derive(Default)]
pub struct BlendStage;

impl BlendStage {
    pub fn new() -> Self {
        Self
    }
}

impl TryFrom<&Params> for BlendStage {
    type Error = BuildError;

    fn try_from(_: &Params) -> Result<Self, BuildError> {
        Ok(Self)
    }
}

impl Node for BlendStage {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![
                PortSpec::new("in0", PayloadKind::Frame),
                PortSpec::new("in1", PayloadKind::Frame),
            ],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let f0 = inputs.frame("in0")?;
        let f1 = inputs.frame("in1")?;

        if f0.width != f1.width || f0.height != f1.height || f0.channels != f1.channels {
            return Err(NodeError::Message(
                "blend inputs must have same dimensions and channels".to_string(),
            ));
        }
        if f0.dtype() != f1.dtype() {
            return Err(NodeError::Message(
                "blend inputs must have same dtype".to_string(),
            ));
        }

        let data = match f0.dtype() {
            DType::U8 => {
                let buf0 = f0.as_u8().unwrap();
                let buf1 = f1.as_u8().unwrap();
                FrameData::U8(
                    buf0.iter()
                        .zip(buf1)
                        .map(|(&a, &b)| ((a as u16 + b as u16) / 2) as u8)
                        .collect(),
                )
            }
            DType::F32 => {
                let buf0 = f0.as_f32().unwrap();
                let buf1 = f1.as_f32().unwrap();
                FrameData::F32(buf0.iter().zip(buf1).map(|(a, b)| (a + b) * 0.5).collect())
            }
        };

        outputs.set(
            "out",
            Payload::Frame(Frame::from_data(f0.width, f0.height, f0.channels, data)),
        );
        Ok(())
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_stage::<BlendStage>("blend");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PortId;
    use std::collections::HashMap;

    #[test]
    fn blend_u8() {
        let mut stage = BlendStage::new();
        let mut m = HashMap::new();
        m.insert(
            PortId("in0".to_string()),
            Payload::Frame(Frame::from_rgb8(1, 1, vec![(10, 20, 30)])),
        );
        m.insert(
            PortId("in1".to_string()),
            Payload::Frame(Frame::from_rgb8(1, 1, vec![(30, 40, 50)])),
        );
        let mut out = Outputs::new();
        stage.eval(&Inputs::new(m), &mut out).unwrap();
        let f = out.get("out").unwrap().as_frame().unwrap();
        assert_eq!(f.to_rgb8(), vec![(20, 30, 40)]);
    }
}
