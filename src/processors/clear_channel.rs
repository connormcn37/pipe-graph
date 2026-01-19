use crate::traits::Processor;

pub enum Channel {
    Red,
    Green,
    Blue,
}

pub struct ClearChannel(pub Channel);

impl Processor for ClearChannel {
    fn process(&self, input: &mut crate::data::Frame) {
        match self.0 {
            Channel::Red => {
                for pixel in input.pixels.iter_mut() {
                    pixel.0 = 0;
                }
            }
            Channel::Green => {
                for pixel in input.pixels.iter_mut() {
                    pixel.1 = 0;
                }
            }
            Channel::Blue => {
                for pixel in input.pixels.iter_mut() {
                    pixel.2 = 0;
                }
            }
        }
    }
}