//! Turning a pipeline [`Frame`] into pixels a texture can show.
//!
//! The pipeline is dtype- and channel-generic; a preview texture is always
//! 8-bit RGBA. [`frame_to_rgba8`] is the one conversion between them, kept as
//! a plain function (no Bevy types) so its edge cases are unit-tested here and
//! the renderer only has to wrap the bytes in an `Image`.

use crate::data::{Frame, FrameData};

/// Convert `frame` to tightly packed RGBA8, row-major, top row first.
///
/// - 1 channel → gray (`v, v, v, 255`); 2 channels → gray + alpha;
///   3 → RGB with opaque alpha; 4 → RGBA as is.
/// - `u8` samples are copied; `f32` samples are treated as `0.0..=1.0`,
///   clamped and rounded (NaN shows as 0).
///
/// Returns `None` for a frame with more than 4 channels or no pixels, which
/// has no obvious picture.
pub fn frame_to_rgba8(frame: &Frame) -> Option<(u32, u32, Vec<u8>)> {
    let channels = frame.channels as usize;
    if !(1..=4).contains(&channels) || frame.width == 0 || frame.height == 0 {
        return None;
    }
    let samples: Vec<u8> = match frame.data() {
        FrameData::U8(v) => v.clone(),
        FrameData::F32(v) => v
            .iter()
            .map(|&x| {
                // `as` saturates and maps NaN to 0, which is what we want.
                (x.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect(),
    };

    let pixels = frame.width as usize * frame.height as usize;
    let mut out = Vec::with_capacity(pixels * 4);
    for px in samples.chunks_exact(channels) {
        match *px {
            [v] => out.extend_from_slice(&[v, v, v, 255]),
            [v, a] => out.extend_from_slice(&[v, v, v, a]),
            [r, g, b] => out.extend_from_slice(&[r, g, b, 255]),
            [r, g, b, a] => out.extend_from_slice(&[r, g, b, a]),
            _ => unreachable!("channel count checked above"),
        }
    }
    Some((frame.width, frame.height, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_u8_gets_opaque_alpha() {
        let f = Frame::from_rgb8(2, 1, vec![(1, 2, 3), (4, 5, 6)]);
        assert_eq!(
            frame_to_rgba8(&f),
            Some((2, 1, vec![1, 2, 3, 255, 4, 5, 6, 255]))
        );
    }

    #[test]
    fn gray_and_gray_alpha_expand_to_rgba() {
        let gray = Frame::from_data(2, 1, 1, FrameData::U8(vec![7, 9]));
        assert_eq!(
            frame_to_rgba8(&gray).unwrap().2,
            vec![7, 7, 7, 255, 9, 9, 9, 255]
        );
        let ga = Frame::from_data(1, 1, 2, FrameData::U8(vec![7, 128]));
        assert_eq!(frame_to_rgba8(&ga).unwrap().2, vec![7, 7, 7, 128]);
    }

    #[test]
    fn rgba_passes_through() {
        let f = Frame::from_data(1, 1, 4, FrameData::U8(vec![10, 20, 30, 40]));
        assert_eq!(frame_to_rgba8(&f).unwrap().2, vec![10, 20, 30, 40]);
    }

    #[test]
    fn f32_is_clamped_and_scaled() {
        let f = Frame::from_data(1, 1, 3, FrameData::F32(vec![-0.5, 0.5, 2.0]));
        assert_eq!(frame_to_rgba8(&f).unwrap().2, vec![0, 128, 255, 255]);
        let nan = Frame::from_data(1, 1, 1, FrameData::F32(vec![f32::NAN]));
        assert_eq!(frame_to_rgba8(&nan).unwrap().2, vec![0, 0, 0, 255]);
    }

    #[test]
    fn unshowable_frames_are_none() {
        let five = Frame::from_data(1, 1, 5, FrameData::U8(vec![0; 5]));
        assert_eq!(frame_to_rgba8(&five), None);
        let empty = Frame::from_data(0, 3, 3, FrameData::U8(vec![]));
        assert_eq!(frame_to_rgba8(&empty), None);
    }
}
