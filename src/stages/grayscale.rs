use crate::data::{Frame, FrameData};
use crate::exec::Registry;
use crate::traits::Processor;

pub struct Grayscale;

impl Processor for Grayscale {
    fn process(&self, input: &mut Frame) {
        let channels = input.channels as usize;
        if channels == 0 {
            return;
        }
        match input.data_mut() {
            FrameData::U8(buf) => {
                for pixel in buf.chunks_exact_mut(channels) {
                    let sum: u32 = pixel.iter().map(|&v| v as u32).sum();
                    let avg = (sum / channels as u32) as u8;
                    for c in pixel.iter_mut() {
                        *c = avg;
                    }
                }
            }
            FrameData::F32(buf) => {
                for pixel in buf.chunks_exact_mut(channels) {
                    let sum: f32 = pixel.iter().sum();
                    let avg = sum / channels as f32;
                    for c in pixel.iter_mut() {
                        *c = avg;
                    }
                }
            }
        }
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_processor("grayscale", |_| Ok(Grayscale));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Frame;
    use crate::traits::Processor;

    #[test]
    fn grayscale_u8() {
        let mut f = Frame::from_rgb8(1, 2, vec![(10, 20, 30), (255, 255, 255)]);
        Grayscale.process(&mut f);
        assert_eq!(f.to_rgb8(), vec![(20, 20, 20), (255, 255, 255)]);
    }
}
