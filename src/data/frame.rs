//! Frame: a generic, interleaved image/tensor buffer.
//!
//! A `Frame` is `width * height` pixels, each with `channels` components,
//! stored as a single flat, interleaved buffer (`[c0, c1, .., c{k-1}]` per
//! pixel). The element type is chosen by `dtype` (`u8` or `f32`). This makes
//! per-channel ops, cropping, and split/merge index math straightforward, and
//! lets the pipeline carry more than just 8-bit RGB.

/// Element type of a frame's buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    F32,
}

/// The backing buffer for a frame, one variant per [`DType`].
#[derive(Debug, Clone, PartialEq)]
pub enum FrameData {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

impl FrameData {
    /// Number of scalar elements in the buffer (`width * height * channels`).
    pub fn len(&self) -> usize {
        match self {
            FrameData::U8(v) => v.len(),
            FrameData::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> DType {
        match self {
            FrameData::U8(_) => DType::U8,
            FrameData::F32(_) => DType::F32,
        }
    }
}

/// A generic interleaved frame buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    dtype: DType,
    data: FrameData,
}

impl Frame {
    /// Construct from an explicit buffer, validating the length invariant.
    ///
    /// Panics if `data.len() != width * height * channels`.
    pub fn from_data(width: u32, height: u32, channels: u32, data: FrameData) -> Self {
        let expected = width as usize * height as usize * channels as usize;
        assert_eq!(
            data.len(),
            expected,
            "frame buffer length {} does not match {}x{}x{} = {}",
            data.len(),
            width,
            height,
            channels,
            expected
        );
        let dtype = data.dtype();
        Self {
            width,
            height,
            channels,
            dtype,
            data,
        }
    }

    /// Allocate a zero-filled frame with the given shape and dtype.
    pub fn zeros(width: u32, height: u32, channels: u32, dtype: DType) -> Self {
        let n = width as usize * height as usize * channels as usize;
        let data = match dtype {
            DType::U8 => FrameData::U8(vec![0; n]),
            DType::F32 => FrameData::F32(vec![0.0; n]),
        };
        Self {
            width,
            height,
            channels,
            dtype,
            data,
        }
    }

    /// Build a 3-channel `u8` frame from a list of RGB pixels (row-major).
    ///
    /// Panics if `pixels.len() != width * height`.
    pub fn from_rgb8(width: u32, height: u32, pixels: Vec<(u8, u8, u8)>) -> Self {
        assert_eq!(
            pixels.len(),
            width as usize * height as usize,
            "pixel count does not match {width}x{height}"
        );
        let mut buf = Vec::with_capacity(pixels.len() * 3);
        for (r, g, b) in pixels {
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
        Self::from_data(width, height, 3, FrameData::U8(buf))
    }

    /// View this frame as RGB pixel tuples.
    ///
    /// Panics unless the frame is 3-channel `u8`.
    pub fn to_rgb8(&self) -> Vec<(u8, u8, u8)> {
        assert_eq!(self.channels, 3, "to_rgb8 requires 3 channels");
        let buf = self.as_u8().expect("to_rgb8 requires a u8 frame");
        buf.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of pixels (`width * height`).
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Number of scalar elements (`width * height * channels`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &FrameData {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut FrameData {
        &mut self.data
    }

    pub fn as_u8(&self) -> Option<&[u8]> {
        match &self.data {
            FrameData::U8(v) => Some(v),
            FrameData::F32(_) => None,
        }
    }

    pub fn as_u8_mut(&mut self) -> Option<&mut [u8]> {
        match &mut self.data {
            FrameData::U8(v) => Some(v),
            FrameData::F32(_) => None,
        }
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            FrameData::F32(v) => Some(v),
            FrameData::U8(_) => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match &mut self.data {
            FrameData::F32(v) => Some(v),
            FrameData::U8(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb8_round_trips() {
        let pixels = vec![(255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 20, 30)];
        let f = Frame::from_rgb8(2, 2, pixels.clone());
        assert_eq!(f.channels, 3);
        assert_eq!(f.dtype(), DType::U8);
        assert_eq!(f.len(), 2 * 2 * 3);
        assert_eq!(f.to_rgb8(), pixels);
    }

    #[test]
    fn interleave_layout_is_channel_minor() {
        let f = Frame::from_rgb8(1, 2, vec![(1, 2, 3), (4, 5, 6)]);
        assert_eq!(f.as_u8().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn zeros_has_correct_shape() {
        let f = Frame::zeros(4, 3, 2, DType::F32);
        assert_eq!(f.len(), 4 * 3 * 2);
        assert!(f.as_f32().unwrap().iter().all(|&x| x == 0.0));
        assert!(f.as_u8().is_none());
    }

    #[test]
    #[should_panic]
    fn from_data_rejects_bad_length() {
        Frame::from_data(2, 2, 3, FrameData::U8(vec![0; 5]));
    }
}
