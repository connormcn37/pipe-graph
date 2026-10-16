//! Frame: a generic, interleaved image/tensor buffer.
//!
//! A `Frame` is `width * height` pixels, each with `channels` components,
//! stored as a single flat, interleaved buffer (`[c0, c1, .., c{k-1}]` per
//! pixel). The element type is chosen by `dtype` (`u8` or `f32`). This makes
//! per-channel ops, cropping, and split/merge index math straightforward, and
//! lets the pipeline carry more than just 8-bit RGB.
//!
//! A frame also carries [`Meta`] (timestamp, frame number, colour space).
//! Metadata is an annotation, not content: two frames are equal when their
//! shape and samples are, whatever their metadata.
//!
//! The samples themselves can live on the CPU, on the GPU (`gpu` feature) or
//! in storage this crate does not manage; see [`Storage`].

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use crate::data::Meta;

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

/// Where a frame's samples currently live. See [`Storage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Location {
    Cpu,
    Gpu,
    External,
}

/// A frame's samples.
///
/// A closed enum for the storage this crate owns, so stage code can match on
/// it and get the concrete buffer it works on, plus one open variant for
/// storage it doesn't (hardware decoder surfaces, another API's textures…).
#[derive(Debug, Clone)]
pub enum Storage {
    /// In main memory, readable directly.
    Cpu(FrameData),
    /// In a GPU buffer; see [`crate::gpu::GpuBuffer`] for its layout.
    #[cfg(feature = "gpu")]
    Gpu(crate::gpu::GpuBuffer),
    /// Somewhere this crate does not manage. It can only be read by
    /// downloading it ([`ExternalStorage::download`]).
    External(Arc<dyn ExternalStorage>),
}

impl Storage {
    pub fn location(&self) -> Location {
        match self {
            Storage::Cpu(_) => Location::Cpu,
            #[cfg(feature = "gpu")]
            Storage::Gpu(_) => Location::Gpu,
            Storage::External(_) => Location::External,
        }
    }
}

/// Frame storage owned by something outside this crate.
///
/// The frame keeps the shape and dtype; the implementation only has to
/// produce the samples, interleaved as [`FrameData`] lays them out, with
/// exactly `width * height * channels` elements of the frame's dtype.
pub trait ExternalStorage: Send + Sync + fmt::Debug {
    fn download(&self) -> Result<FrameData, TransferError>;
}

/// Moving a frame between locations failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferError(pub String);

impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "frame transfer failed: {}", self.0)
    }
}

impl std::error::Error for TransferError {}

/// A generic interleaved frame buffer.
///
/// Shape, dtype and metadata are always available; the samples may live on
/// the CPU, the GPU, or elsewhere ([`Frame::location`]). The CPU accessors
/// (`data`, `as_u8`, …) are for frames on the CPU: CPU stages only ever see
/// those, since the scheduler downloads a frame before handing it to one.
/// Code that may meet any frame uses [`Frame::to_cpu`].
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    dtype: DType,
    storage: Storage,
    meta: Meta,
}

/// Equality is shape and samples only; see the module docs. Samples are
/// compared only when both frames are on the CPU (comparing GPU frames would
/// mean downloading them); otherwise frames are equal only if they share the
/// same external storage.
impl PartialEq for Frame {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.channels == other.channels
            && self.dtype == other.dtype
            && match (&self.storage, &other.storage) {
                (Storage::Cpu(a), Storage::Cpu(b)) => a == b,
                (Storage::External(a), Storage::External(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

const OFF_CPU: &str = "frame samples are not on the CPU; use Frame::to_cpu";

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
            storage: Storage::Cpu(data),
            meta: Meta::default(),
        }
    }

    /// A frame whose samples live in external storage. The storage must
    /// download to `width * height * channels` elements of `dtype`.
    pub fn from_external(
        width: u32,
        height: u32,
        channels: u32,
        dtype: DType,
        storage: Arc<dyn ExternalStorage>,
    ) -> Self {
        Self {
            width,
            height,
            channels,
            dtype,
            storage: Storage::External(storage),
            meta: Meta::default(),
        }
    }

    /// A frame whose samples are already on the GPU.
    ///
    /// Panics if the buffer's element count or dtype does not match.
    #[cfg(feature = "gpu")]
    pub fn from_gpu(width: u32, height: u32, channels: u32, buffer: crate::gpu::GpuBuffer) -> Self {
        let expected = width as usize * height as usize * channels as usize;
        assert_eq!(
            buffer.len(),
            expected,
            "GPU buffer length {} does not match {width}x{height}x{channels}",
            buffer.len()
        );
        Self {
            width,
            height,
            channels,
            dtype: buffer.dtype(),
            storage: Storage::Gpu(buffer),
            meta: Meta::default(),
        }
    }

    /// Allocate a zero-filled frame with the given shape and dtype.
    pub fn zeros(width: u32, height: u32, channels: u32, dtype: DType) -> Self {
        let n = width as usize * height as usize * channels as usize;
        let data = match dtype {
            DType::U8 => FrameData::U8(vec![0; n]),
            DType::F32 => FrameData::F32(vec![0.0; n]),
        };
        Self::from_data(width, height, channels, data)
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
    /// Panics unless the frame is 3-channel `u8` on the CPU.
    pub fn to_rgb8(&self) -> Vec<(u8, u8, u8)> {
        assert_eq!(self.channels, 3, "to_rgb8 requires 3 channels");
        let buf = self
            .as_u8()
            .expect("to_rgb8 requires a u8 frame on the CPU");
        buf.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    pub fn meta_mut(&mut self) -> &mut Meta {
        &mut self.meta
    }

    /// This frame with `meta` in place of its current metadata.
    pub fn with_meta(mut self, meta: Meta) -> Self {
        self.meta = meta;
        self
    }

    /// Number of pixels (`width * height`).
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Number of scalar elements (`width * height * channels`).
    pub fn len(&self) -> usize {
        self.pixel_count() * self.channels as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn location(&self) -> Location {
        self.storage.location()
    }

    /// The samples, if they are on the CPU.
    pub fn try_data(&self) -> Option<&FrameData> {
        match &self.storage {
            Storage::Cpu(data) => Some(data),
            _ => None,
        }
    }

    /// The samples, if they are on the CPU.
    pub fn try_data_mut(&mut self) -> Option<&mut FrameData> {
        match &mut self.storage {
            Storage::Cpu(data) => Some(data),
            _ => None,
        }
    }

    /// The samples of a frame on the CPU.
    ///
    /// Panics if they are elsewhere: see [`Frame::try_data`] and
    /// [`Frame::to_cpu`].
    pub fn data(&self) -> &FrameData {
        self.try_data().expect(OFF_CPU)
    }

    /// The samples of a frame on the CPU. Panics if they are elsewhere.
    pub fn data_mut(&mut self) -> &mut FrameData {
        self.try_data_mut().expect(OFF_CPU)
    }

    /// `u8` samples, if the frame is `u8` and on the CPU.
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self.try_data()? {
            FrameData::U8(v) => Some(v),
            FrameData::F32(_) => None,
        }
    }

    pub fn as_u8_mut(&mut self) -> Option<&mut [u8]> {
        match self.try_data_mut()? {
            FrameData::U8(v) => Some(v),
            FrameData::F32(_) => None,
        }
    }

    /// `f32` samples, if the frame is `f32` and on the CPU.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self.try_data()? {
            FrameData::F32(v) => Some(v),
            FrameData::U8(_) => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match self.try_data_mut()? {
            FrameData::F32(v) => Some(v),
            FrameData::U8(_) => None,
        }
    }

    /// The samples on the CPU: borrowed if they are already there, otherwise
    /// downloaded.
    pub fn to_cpu(&self) -> Result<Cow<'_, FrameData>, TransferError> {
        let data = match &self.storage {
            Storage::Cpu(data) => return Ok(Cow::Borrowed(data)),
            #[cfg(feature = "gpu")]
            Storage::Gpu(buffer) => buffer.download()?,
            Storage::External(ext) => ext.download()?,
        };
        if data.len() != self.len() || data.dtype() != self.dtype {
            return Err(TransferError(format!(
                "downloaded {} {:?} samples for a {}x{}x{} {:?} frame",
                data.len(),
                data.dtype(),
                self.width,
                self.height,
                self.channels,
                self.dtype
            )));
        }
        Ok(Cow::Owned(data))
    }

    /// This frame, with its samples on the CPU (metadata kept).
    pub fn into_cpu(self) -> Result<Frame, TransferError> {
        if self.location() == Location::Cpu {
            return Ok(self);
        }
        let data = self.to_cpu()?.into_owned();
        Ok(Frame {
            storage: Storage::Cpu(data),
            ..self
        })
    }

    /// The samples on the GPU of `ctx`: borrowed if they are already there,
    /// otherwise uploaded (downloading external storage first).
    ///
    /// A frame already on the GPU is assumed to be on `ctx`'s device.
    #[cfg(feature = "gpu")]
    pub fn to_gpu(
        &self,
        ctx: &crate::gpu::GpuContext,
    ) -> Result<Cow<'_, crate::gpu::GpuBuffer>, TransferError> {
        match &self.storage {
            Storage::Gpu(buffer) => Ok(Cow::Borrowed(buffer)),
            _ => Ok(Cow::Owned(crate::gpu::GpuBuffer::upload(
                ctx,
                &*self.to_cpu()?,
            ))),
        }
    }

    /// This frame, with its samples on the GPU of `ctx` (metadata kept).
    #[cfg(feature = "gpu")]
    pub fn into_gpu(self, ctx: &crate::gpu::GpuContext) -> Result<Frame, TransferError> {
        if self.location() == Location::Gpu {
            return Ok(self);
        }
        let buffer = self.to_gpu(ctx)?.into_owned();
        Ok(Frame {
            storage: Storage::Gpu(buffer),
            ..self
        })
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
    fn equality_ignores_metadata() {
        let a = Frame::from_rgb8(1, 1, vec![(1, 2, 3)]);
        let b = a.clone().with_meta(Meta {
            frame_index: Some(7),
            ..Meta::default()
        });
        assert_eq!(a, b);
        assert_eq!(b.meta().frame_index, Some(7));
        assert!(a.meta().is_unset());
    }

    /// Storage outside this crate: a fixed buffer, or a failing download.
    #[derive(Debug)]
    struct Remote(Result<FrameData, TransferError>);

    impl ExternalStorage for Remote {
        fn download(&self) -> Result<FrameData, TransferError> {
            self.0.clone()
        }
    }

    #[test]
    fn external_frames_download_on_demand() {
        let remote = Arc::new(Remote(Ok(FrameData::U8(vec![1, 2, 3, 4]))));
        let f = Frame::from_external(2, 1, 2, DType::U8, remote.clone()).with_meta(Meta {
            frame_index: Some(3),
            ..Meta::default()
        });
        assert_eq!(f.location(), Location::External);
        assert_eq!((f.len(), f.dtype()), (4, DType::U8));
        assert!(f.as_u8().is_none() && f.try_data().is_none());
        assert!(matches!(f.to_cpu().unwrap(), Cow::Owned(_)));
        // Same storage compares equal without downloading.
        assert_eq!(f, Frame::from_external(2, 1, 2, DType::U8, remote));

        let cpu = f.into_cpu().unwrap();
        assert_eq!(cpu.location(), Location::Cpu);
        assert_eq!(cpu.as_u8().unwrap(), &[1, 2, 3, 4]);
        assert_eq!(cpu.meta().frame_index, Some(3));
        assert!(matches!(cpu.to_cpu().unwrap(), Cow::Borrowed(_)));
    }

    #[test]
    fn external_download_errors_and_mismatches_are_reported() {
        let failing = Remote(Err(TransferError("device lost".into())));
        let f = Frame::from_external(1, 1, 1, DType::U8, Arc::new(failing));
        assert_eq!(f.to_cpu().unwrap_err(), TransferError("device lost".into()));

        let short = Remote(Ok(FrameData::U8(vec![0; 3])));
        let f = Frame::from_external(2, 2, 1, DType::U8, Arc::new(short));
        assert!(f.into_cpu().is_err());
    }

    #[test]
    #[should_panic(expected = "not on the CPU")]
    fn cpu_accessor_panics_off_cpu() {
        let remote = Remote(Ok(FrameData::U8(vec![0])));
        Frame::from_external(1, 1, 1, DType::U8, Arc::new(remote)).data();
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_round_trip_keeps_shape_and_meta() {
        let Some(ctx) = crate::gpu::tests::context() else {
            return;
        };
        let f = Frame::from_rgb8(3, 1, vec![(1, 2, 3), (4, 5, 6), (7, 8, 9)]).with_meta(Meta {
            frame_index: Some(1),
            ..Meta::default()
        });
        assert!(matches!(f.to_gpu(&ctx).unwrap(), Cow::Owned(_)));
        let g = f.clone().into_gpu(&ctx).unwrap();
        assert_eq!(g.location(), Location::Gpu);
        assert!(matches!(g.to_gpu(&ctx).unwrap(), Cow::Borrowed(_)));
        assert!(g.as_u8().is_none());
        assert_ne!(g, f, "GPU frames are not compared by value");
        let back = g.into_cpu().unwrap();
        assert_eq!(back, f);
        assert_eq!(back.meta().frame_index, Some(1));
    }

    #[test]
    #[should_panic]
    fn from_data_rejects_bad_length() {
        Frame::from_data(2, 2, 3, FrameData::U8(vec![0; 5]));
    }
}
