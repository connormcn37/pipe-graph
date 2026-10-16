//! Metadata that travels with a frame (or audio buffer) through the graph.
//!
//! Pixels say what a frame looks like; `Meta` says *where it belongs*: its
//! position in the stream (timestamp, frame number, duration) and how to
//! interpret its values (colour space). Sources fill it in; most stages never
//! touch it, and the runtime carries it from a node's input to its output
//! for them (see `exec::schedule`'s `inherit_meta`).

use std::time::Duration;

/// How a frame's sample values are to be interpreted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// Nothing is known (hand-built frames, generic tensors).
    #[default]
    Unspecified,
    /// Gamma-encoded sRGB / BT.709 primaries — what decoded video RGB is.
    Srgb,
    /// Linear-light RGB.
    Linear,
}

/// Timing and interpretation of a frame or audio buffer. Every field is
/// optional: a value nobody stamped stays `None`/`Unspecified` rather than
/// being guessed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Meta {
    /// Presentation time from the start of the stream.
    pub pts: Option<Duration>,
    /// How long this frame is shown (`1 / fps` for constant-rate video).
    pub duration: Option<Duration>,
    /// Zero-based index of this frame in its stream.
    pub frame_index: Option<u64>,
    pub color: ColorSpace,
}

impl Meta {
    /// True when nothing has been stamped, so an inherited value may fill it.
    pub fn is_unset(&self) -> bool {
        *self == Meta::default()
    }

    /// Timing for frame `index` of a constant-rate stream at `fps_num /
    /// fps_den` frames per second. Computed exactly in integer nanoseconds,
    /// so frame `n` of 30000/1001 video lands where `n * 1001/30000 s` does,
    /// without drift accumulating across a long stream.
    pub fn for_frame(index: u64, fps_num: u32, fps_den: u32) -> Self {
        let at = |n: u64| {
            if fps_num == 0 {
                return None;
            }
            let nanos = n as u128 * fps_den as u128 * 1_000_000_000 / fps_num as u128;
            Some(Duration::from_nanos(u64::try_from(nanos).ok()?))
        };
        Meta {
            pts: at(index),
            duration: at(index + 1).zip(at(index)).map(|(next, this)| next - this),
            frame_index: Some(index),
            color: ColorSpace::Unspecified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_unset() {
        assert!(Meta::default().is_unset());
        let stamped = Meta {
            frame_index: Some(0),
            ..Meta::default()
        };
        assert!(!stamped.is_unset());
    }

    #[test]
    fn ntsc_timing_is_exact_and_does_not_drift() {
        // 30000/1001 fps: frame 30000 starts at exactly 1001 s.
        let m = Meta::for_frame(30_000, 30_000, 1001);
        assert_eq!(m.pts, Some(Duration::from_secs(1001)));
        assert_eq!(m.frame_index, Some(30_000));
        // Durations alternate by a nanosecond at most, summing exactly.
        let d = m.duration.unwrap().as_nanos();
        assert!((33_366_666..=33_366_667).contains(&d), "{d}");
    }

    #[test]
    fn integer_rate() {
        let m = Meta::for_frame(3, 25, 1);
        assert_eq!(m.pts, Some(Duration::from_millis(120)));
        assert_eq!(m.duration, Some(Duration::from_millis(40)));
    }

    #[test]
    fn zero_rate_has_no_timing() {
        let m = Meta::for_frame(1, 0, 1);
        assert_eq!(m.pts, None);
        assert_eq!(m.duration, None);
        assert_eq!(m.frame_index, Some(1));
    }
}
