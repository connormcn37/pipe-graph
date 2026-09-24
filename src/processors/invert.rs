use crate::data::{Frame, FrameData};
use crate::traits::Processor;

pub struct Invert;

impl Processor for Invert {
    fn process(&self, input: &mut Frame) {
        match input.data_mut() {
            FrameData::U8(buf) => {
                for v in buf.iter_mut() {
                    *v = 255 - *v;
                }
            }
            FrameData::F32(buf) => {
                for v in buf.iter_mut() {
                    *v = 1.0 - *v;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Frame;
    use crate::traits::Processor;

    #[test]
    fn invert_u8() {
        let mut f = Frame::from_rgb8(1, 2, vec![(0, 100, 255), (10, 20, 30)]);
        Invert.process(&mut f);
        assert_eq!(f.to_rgb8(), vec![(255, 155, 0), (245, 235, 225)]);
    }
}
