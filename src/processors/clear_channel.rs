use crate::data::{Frame, FrameData};
use crate::traits::Processor;

/// Named RGB channels, kept for demo ergonomics. Each maps to an interleaved
/// channel index (R=0, G=1, B=2).
pub enum Channel {
    Red,
    Green,
    Blue,
}

impl Channel {
    pub fn index(&self) -> usize {
        match self {
            Channel::Red => 0,
            Channel::Green => 1,
            Channel::Blue => 2,
        }
    }
}

/// Zeroes a single channel across every pixel of the frame.
///
/// Generalized over the frame's dtype and channel count: it clears channel
/// index `self.0.index()`, and is a no-op if that index is out of range.
pub struct ClearChannel(pub Channel);

impl Processor for ClearChannel {
    fn process(&self, input: &mut Frame) {
        let c = self.0.index();
        let channels = input.channels as usize;
        if c >= channels {
            return;
        }
        match input.data_mut() {
            FrameData::U8(buf) => {
                let mut i = c;
                while i < buf.len() {
                    buf[i] = 0;
                    i += channels;
                }
            }
            FrameData::F32(buf) => {
                let mut i = c;
                while i < buf.len() {
                    buf[i] = 0.0;
                    i += channels;
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
        Frame::from_rgb8(
            3,
            3,
            vec![
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
        )
    }

    #[test]
    fn clear_red_zeroes_only_red() {
        let mut f = demo_frame();
        ClearChannel(Channel::Red).process(&mut f);
        let px = f.to_rgb8();
        assert!(px.iter().all(|p| p.0 == 0));
        // green/blue are untouched
        assert_eq!(px[1], (0, 255, 0));
        assert_eq!(px[2], (0, 0, 255));
        assert_eq!(px[6], (0, 192, 192));
    }

    #[test]
    fn clear_green_zeroes_only_green() {
        let mut f = demo_frame();
        ClearChannel(Channel::Green).process(&mut f);
        let px = f.to_rgb8();
        assert!(px.iter().all(|p| p.1 == 0));
        assert_eq!(px[0], (255, 0, 0));
    }

    #[test]
    fn clear_blue_zeroes_only_blue() {
        let mut f = demo_frame();
        ClearChannel(Channel::Blue).process(&mut f);
        let px = f.to_rgb8();
        assert!(px.iter().all(|p| p.2 == 0));
        assert_eq!(px[8], (64, 64, 0));
    }
}
