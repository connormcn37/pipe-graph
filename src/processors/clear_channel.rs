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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Frame;
    use crate::traits::Processor;

    fn demo_frame() -> Frame {
        Frame {
            width: 3,
            height: 3,
            pixels: vec![
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (0, 255, 255),
                (255, 0, 255),
                (192, 192, 192),
                (128, 128, 128),
                (64, 64, 64),
            ],
        }
    }

    #[test]
    fn clear_red_zeroes_only_red() {
        let mut f = demo_frame();
        ClearChannel(Channel::Red).process(&mut f);
        assert!(f.pixels.iter().all(|p| p.0 == 0));
        // green/blue are untouched
        assert_eq!(f.pixels[1], (0, 255, 0));
        assert_eq!(f.pixels[2], (0, 0, 255));
        assert_eq!(f.pixels[6], (0, 192, 192));
    }

    #[test]
    fn clear_green_zeroes_only_green() {
        let mut f = demo_frame();
        ClearChannel(Channel::Green).process(&mut f);
        assert!(f.pixels.iter().all(|p| p.1 == 0));
        assert_eq!(f.pixels[0], (255, 0, 0));
    }

    #[test]
    fn clear_blue_zeroes_only_blue() {
        let mut f = demo_frame();
        ClearChannel(Channel::Blue).process(&mut f);
        assert!(f.pixels.iter().all(|p| p.2 == 0));
        assert_eq!(f.pixels[8], (64, 64, 0));
    }
}
