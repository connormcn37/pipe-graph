//! Concrete pipeline stages implemented as [`crate::exec::Node`]s.
//!
//! These realize the README's stage catalog on the generalized [`Frame`]:
//! - [`CropStage`] — 1→1, extract a sub-rectangle.
//! - [`CastStage`] — 1→1, convert between `u8` and `f32`.
//! - [`SplitStage`] — 1→k, split a k-channel frame into k single-channel frames.
//! - [`MergeStage`] — k→1, interleave k single-channel frames into one.
//!
//! Per-pixel workloads (u8 and f32; semantics documented on each type):
//! - [`GainStage`] — multiply by `factor` (u8 saturates).
//! - [`GrayscaleStage`] — RGB(A) → 1-channel BT.601 luma (alpha dropped).
//! - [`InvertStage`] — `255 - v` / `1.0 - v`.
//! - [`ThresholdStage`] — binarize against a normalized `level`.
//! - [`BoxBlurStage`] — per-channel box blur, edge-clamped, O(1) per pixel.
//!
//! Split/Merge declare a param-dependent number of ports (`out0..`, `in0..`),
//! which is exactly why [`crate::exec::Node::ports`] takes `&self`.
//!
//! [`Frame`]: crate::data::Frame

mod box_blur;
mod cast;
mod crop;
mod gain;
mod grayscale;
mod invert;
mod merge;
mod split;
mod threshold;

pub use self::box_blur::BoxBlurStage;
pub use self::cast::CastStage;
pub use self::crop::CropStage;
pub use self::gain::GainStage;
pub use self::grayscale::GrayscaleStage;
pub use self::invert::InvertStage;
pub use self::merge::MergeStage;
pub use self::split::SplitStage;
pub use self::threshold::ThresholdStage;

#[cfg(test)]
mod tests {
    use crate::data::{Frame, FrameData, Payload};
    use crate::exec::{Inputs, Node, Outputs};
    use crate::graph::PortId;
    use std::collections::HashMap;

    use super::{MergeStage, SplitStage};

    fn eval1(node: &mut dyn Node, port: &str, frame: Frame) -> Outputs {
        let mut m = HashMap::new();
        m.insert(PortId(port.to_string()), Payload::Frame(frame));
        let mut out = Outputs::new();
        node.eval(&Inputs::new(m), &mut out).unwrap();
        out
    }

    #[test]
    fn split_then_merge_round_trips() {
        // A 2x2x3 frame with distinct per-channel values.
        let src = Frame::from_data(
            2,
            2,
            3,
            FrameData::U8(vec![
                1, 10, 100, // px0
                2, 20, 101, // px1
                3, 30, 102, // px2
                4, 40, 103, // px3
            ]),
        );

        // Split into three single-channel frames.
        let mut split = SplitStage::new(3);
        let split_out = eval1(&mut split, "in", src.clone());
        let ch0 = split_out.get("out0").unwrap().as_frame().unwrap().clone();
        let ch1 = split_out.get("out1").unwrap().as_frame().unwrap().clone();
        let ch2 = split_out.get("out2").unwrap().as_frame().unwrap().clone();
        assert_eq!(ch0.channels, 1);
        assert_eq!(ch0.as_u8().unwrap(), &[1, 2, 3, 4]);
        assert_eq!(ch1.as_u8().unwrap(), &[10, 20, 30, 40]);
        assert_eq!(ch2.as_u8().unwrap(), &[100, 101, 102, 103]);

        // Merge them back.
        let mut merge = MergeStage::new(3);
        let mut m = HashMap::new();
        m.insert(PortId("in0".to_string()), Payload::Frame(ch0));
        m.insert(PortId("in1".to_string()), Payload::Frame(ch1));
        m.insert(PortId("in2".to_string()), Payload::Frame(ch2));
        let mut merged = Outputs::new();
        merge.eval(&Inputs::new(m), &mut merged).unwrap();

        assert_eq!(merged.get("out").unwrap().as_frame().unwrap(), &src);
    }
}
