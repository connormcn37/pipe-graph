//! Crop: extract an axis-aligned sub-rectangle, preserving channels and dtype.

use crate::data::{Frame, FrameData, Payload, PayloadKind};
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec};
use crate::graph::Params;

/// 1→1 stage that crops the input frame to `[x, x+w) x [y, y+h)`.
pub struct CropStage {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl CropStage {
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
}

impl TryFrom<&Params> for CropStage {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        Ok(Self {
            x: p.get_u32("x")?,
            y: p.get_u32("y")?,
            w: p.get_u32("w")?,
            h: p.get_u32("h")?,
        })
    }
}

fn crop_buf<T: Copy>(
    src: &[T],
    width: u32,
    channels: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> Vec<T> {
    let channels = channels as usize;
    let width = width as usize;
    let mut out = Vec::with_capacity(w as usize * h as usize * channels);
    for row in y..y + h {
        for col in x..x + w {
            let base = (row as usize * width + col as usize) * channels;
            out.extend_from_slice(&src[base..base + channels]);
        }
    }
    out
}

impl Node for CropStage {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Frame)],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;

        if self.x + self.w > frame.width || self.y + self.h > frame.height {
            return Err(NodeError::Message(format!(
                "crop rect {}x{}+{}+{} exceeds {}x{} frame",
                self.w, self.h, self.x, self.y, frame.width, frame.height
            )));
        }

        let data = match frame.data() {
            FrameData::U8(buf) => FrameData::U8(crop_buf(
                buf,
                frame.width,
                frame.channels,
                self.x,
                self.y,
                self.w,
                self.h,
            )),
            FrameData::F32(buf) => FrameData::F32(crop_buf(
                buf,
                frame.width,
                frame.channels,
                self.x,
                self.y,
                self.w,
                self.h,
            )),
        };

        let cropped = Frame::from_data(self.w, self.h, frame.channels, data);
        outputs.set("out", Payload::Frame(cropped));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crops_center_rectangle() {
        // 3x3 RGB, crop the middle 1x1 at (1,1).
        let f = Frame::from_rgb8(
            3,
            3,
            vec![
                (0, 0, 0),
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                (9, 8, 7),
                (5, 5, 5),
                (6, 6, 6),
                (7, 7, 7),
                (8, 8, 8),
            ],
        );
        let mut stage = CropStage::new(1, 1, 1, 1);
        let mut out = Outputs::new();
        let mut m = std::collections::HashMap::new();
        m.insert(crate::graph::PortId("in".to_string()), Payload::Frame(f));
        stage.eval(&Inputs::new(m), &mut out).unwrap();
        assert_eq!(
            out.get("out").unwrap().as_frame().unwrap().to_rgb8(),
            vec![(9, 8, 7)]
        );
    }

    #[test]
    fn out_of_bounds_is_an_error() {
        let f = Frame::from_rgb8(2, 2, vec![(0, 0, 0); 4]);
        let mut stage = CropStage::new(1, 1, 2, 2);
        let mut out = Outputs::new();
        let mut m = std::collections::HashMap::new();
        m.insert(crate::graph::PortId("in".to_string()), Payload::Frame(f));
        assert!(matches!(
            stage.eval(&Inputs::new(m), &mut out),
            Err(NodeError::Message(_))
        ));
    }
}
