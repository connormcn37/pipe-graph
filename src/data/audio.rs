//! Audio buffers: a stub payload so graphs can carry audio alongside video.
//!
//! The runtime is video-first; audio has a type, a port kind and metadata so
//! that sources, sinks and graph structure can account for it now, but no
//! audio processing exists yet. Samples are interleaved `f32` in `[-1, 1]`,
//! shared behind an `Arc` so fan-out never copies them.

use std::sync::Arc;

use crate::data::Meta;

/// A block of interleaved audio samples.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved: `[l0, r0, l1, r1, ...]` for stereo.
    pub samples: Arc<[f32]>,
    pub meta: Meta,
}

impl AudioBuffer {
    /// Panics if `samples.len()` is not a multiple of `channels`, or
    /// `channels` is zero.
    pub fn new(sample_rate: u32, channels: u16, samples: impl Into<Arc<[f32]>>) -> Self {
        let samples = samples.into();
        assert!(channels > 0, "audio needs at least one channel");
        assert_eq!(
            samples.len() % channels as usize,
            0,
            "{} samples do not divide into {channels} channels",
            samples.len()
        );
        Self {
            sample_rate,
            channels,
            samples,
            meta: Meta::default(),
        }
    }

    /// Samples per channel.
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Length in seconds (`None` at a zero sample rate).
    pub fn seconds(&self) -> Option<f64> {
        (self.sample_rate > 0).then(|| self.frames() as f64 / self.sample_rate as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_frames_and_length() {
        let a = AudioBuffer::new(48_000, 2, vec![0.0f32; 96_000]);
        assert_eq!(a.frames(), 48_000);
        assert_eq!(a.seconds(), Some(1.0));
    }

    #[test]
    #[should_panic]
    fn ragged_samples_are_rejected() {
        AudioBuffer::new(48_000, 2, vec![0.0f32; 3]);
    }
}
