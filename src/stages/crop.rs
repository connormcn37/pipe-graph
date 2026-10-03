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

/// In-place counterpart of [`crop_buf`]: compact the cropped rows to the front
/// of `buf` and truncate. Each destination row starts at or before its source
/// row (the cropped width never exceeds the frame width), so copying rows in
/// order never overwrites data still to be read.
fn crop_buf_in_place<T: Copy>(
    buf: &mut Vec<T>,
    width: u32,
    channels: u32,
    (x, y, w, h): (u32, u32, u32, u32),
) {
    let (ch, width) = (channels as usize, width as usize);
    let row_len = w as usize * ch;
    for (dst_row, src_row) in (y..y + h).enumerate() {
        let src = (src_row as usize * width + x as usize) * ch;
        buf.copy_within(src..src + row_len, dst_row * row_len);
    }
    buf.truncate(row_len * h as usize);
}

impl CropStage {
    fn check_bounds(&self, frame: &Frame) -> Result<(), NodeError> {
        if self.x + self.w > frame.width || self.y + self.h > frame.height {
            return Err(NodeError::Message(format!(
                "crop rect {}x{}+{}+{} exceeds {}x{} frame",
                self.w, self.h, self.x, self.y, frame.width, frame.height
            )));
        }
        Ok(())
    }
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
        self.check_bounds(frame)?;

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

    /// Crop by compacting the kept pixels within the input's own buffer, so a
    /// fused chain allocates nothing for this stage.
    ///
    /// Compaction keeps the input's whole allocation, so a small crop of a
    /// large frame would pin far more memory than its result needs for as long
    /// as the result lives. When less than half the frame survives, this
    /// declines and `eval` allocates an exact-size buffer instead.
    fn eval_in_place(&mut self, payload: &mut Payload) -> Option<Result<(), NodeError>> {
        // A non-frame input is reported by `eval`, with its usual error.
        let Payload::Frame(frame) = payload else {
            return None;
        };
        if let Err(e) = self.check_bounds(frame) {
            return Some(Err(e));
        }
        let kept = self.w as u64 * self.h as u64;
        if kept * 2 < frame.width as u64 * frame.height as u64 {
            return None;
        }
        let rect = (self.x, self.y, self.w, self.h);
        let (width, channels) = (frame.width, frame.channels);
        let mut data = std::mem::replace(frame.data_mut(), FrameData::U8(Vec::new()));
        match &mut data {
            FrameData::U8(buf) => crop_buf_in_place(buf, width, channels, rect),
            FrameData::F32(buf) => crop_buf_in_place(buf, width, channels, rect),
        }
        *frame = Frame::from_data(self.w, self.h, channels, data);
        Some(Ok(()))
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
    fn in_place_matches_eval_without_reallocating() {
        // 4x3 single-channel f32 ramp; crop a 3x2 window off the origin.
        let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let f = Frame::from_data(4, 3, 1, FrameData::F32(data));
        let mut stage = CropStage::new(1, 1, 3, 2);

        let mut out = Outputs::new();
        let mut m = std::collections::HashMap::new();
        m.insert(
            crate::graph::PortId("in".to_string()),
            Payload::Frame(f.clone()),
        );
        stage.eval(&Inputs::new(m), &mut out).unwrap();
        let via_eval = out.get("out").unwrap().as_frame().unwrap();

        let mut payload = Payload::Frame(f);
        let before = payload.as_frame().unwrap().as_f32().unwrap().as_ptr();
        assert_eq!(stage.eval_in_place(&mut payload), Some(Ok(())));
        let in_place = payload.as_frame().unwrap();
        assert_eq!(in_place, via_eval);
        assert_eq!(
            in_place.as_f32().unwrap(),
            &[5.0, 6.0, 7.0, 9.0, 10.0, 11.0]
        );
        assert_eq!(before, in_place.as_f32().unwrap().as_ptr());
    }

    #[test]
    fn in_place_declines_small_crops_of_large_frames() {
        // Keeping 1 of 9 pixels in place would pin the whole 3x3 buffer.
        let f = Frame::from_rgb8(3, 3, vec![(1, 2, 3); 9]);
        let mut payload = Payload::Frame(f.clone());
        assert_eq!(CropStage::new(1, 1, 1, 1).eval_in_place(&mut payload), None);
        assert_eq!(payload.as_frame().unwrap(), &f);
    }

    #[test]
    fn in_place_out_of_bounds_is_an_error() {
        let mut payload = Payload::Frame(Frame::from_rgb8(2, 2, vec![(0, 0, 0); 4]));
        let mut stage = CropStage::new(1, 1, 2, 2);
        assert!(matches!(
            stage.eval_in_place(&mut payload),
            Some(Err(NodeError::Message(_)))
        ));
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
