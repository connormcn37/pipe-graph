//! GPU foundation (`gpu` feature): a wgpu device, frame buffers on it, and
//! compute kernels.
//!
//! - [`GpuContext`] is a device + queue. Make one with [`GpuContext::new`],
//!   or wrap a device you already have ([`GpuContext::from_device`]) so the
//!   runtime shares it with, say, Bevy's renderer.
//! - [`GpuBuffer`] holds a frame's samples on that device. It remembers its
//!   context, so downloading needs nothing else; see
//!   [`crate::data::Frame::to_cpu`] and [`crate::data::Frame::to_gpu`].
//! - [`Kernel`] is a compiled WGSL compute shader that maps storage buffers
//!   bound in order at `@group(0)`, cached per device by
//!   [`GpuContext::kernel`], so stages stay stateless.
//!
//! ## Buffer layout
//!
//! Samples are interleaved exactly as in [`FrameData`]. WGSL has no 8-bit
//! type, so a `u8` frame is packed four samples per `u32` word, little-endian
//! (sample `i` is byte `i % 4` of word `i / 4`), and `f32` frames are plain
//! `array<f32>`. Buffers are padded to a whole number of words (at least one);
//! padding bytes are unspecified.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::data::{DType, FrameData, TransferError};

/// Creating a GPU context failed (no adapter, device refused…).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuError(pub String);

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU unavailable: {}", self.0)
    }
}

impl std::error::Error for GpuError {}

/// A wgpu device and queue. Cheap to clone; clones share the device.
#[derive(Clone)]
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: Option<Arc<wgpu::AdapterInfo>>,
    kernels: Arc<Mutex<HashMap<&'static str, Arc<Kernel>>>>,
}

impl fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter", &self.info.as_ref().map(|i| &i.name))
            .finish_non_exhaustive()
    }
}

impl GpuContext {
    /// Open the default adapter, honouring wgpu's environment variables
    /// (`WGPU_BACKEND`, `WGPU_ADAPTER_NAME`, …), with the adapter's own limits
    /// so frame-sized buffers fit.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter =
            pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::from_env()
                        .unwrap_or(wgpu::PowerPreference::HighPerformance),
                    force_fallback_adapter: false,
                    compatible_surface: None,
                }),
            )
            .map_err(|e| GpuError(e.to_string()))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pipe-graph"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| GpuError(e.to_string()))?;
        Ok(Self {
            info: Some(Arc::new(adapter.get_info())),
            ..Self::from_device(device, queue)
        })
    }

    /// Use an existing device, e.g. the one a renderer already opened.
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            info: None,
            kernels: Arc::default(),
        }
    }

    /// The adapter this context opened, when it opened one itself
    /// ([`GpuContext::new`]); `None` for a borrowed device.
    pub fn adapter_info(&self) -> Option<&wgpu::AdapterInfo> {
        self.info.as_deref()
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The kernel compiled from `wgsl` (entry point `main`), compiled on first
    /// use and cached under `key` for the life of this device. `key` must
    /// identify the source: one key, one shader.
    pub fn kernel(&self, key: &'static str, wgsl: &'static str) -> Arc<Kernel> {
        let mut kernels = self.kernels.lock().expect("kernel cache poisoned");
        kernels
            .entry(key)
            .or_insert_with(|| Arc::new(Kernel::new(self, key, wgsl)))
            .clone()
    }
}

/// A frame's samples in a GPU storage buffer; see the module docs for the
/// layout. Clones share the buffer.
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    ctx: GpuContext,
    buffer: wgpu::Buffer,
    dtype: DType,
    len: usize,
}

/// Bytes `len` samples of `dtype` occupy, padded to whole (and at least one)
/// `u32` words.
fn padded_size(dtype: DType, len: usize) -> u64 {
    let bytes = match dtype {
        DType::U8 => len,
        DType::F32 => len * 4,
    } as u64;
    bytes.div_ceil(4).max(1) * 4
}

const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

impl GpuBuffer {
    /// Copy `data` to a new buffer on `ctx`'s device.
    pub fn upload(ctx: &GpuContext, data: &FrameData) -> Self {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: padded_size(data.dtype(), data.len()),
            usage: USAGE,
            mapped_at_creation: true,
        });
        {
            let mut view = buffer.slice(..).get_mapped_range_mut();
            match data {
                FrameData::U8(v) => view[..v.len()].copy_from_slice(v),
                FrameData::F32(v) => {
                    for (dst, x) in view.chunks_exact_mut(4).zip(v) {
                        dst.copy_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
        buffer.unmap();
        Self {
            ctx: ctx.clone(),
            buffer,
            dtype: data.dtype(),
            len: data.len(),
        }
    }

    /// A zero-filled buffer for `len` samples of `dtype`: a kernel's output.
    pub fn zeroed(ctx: &GpuContext, dtype: DType, len: usize) -> Self {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: padded_size(dtype, len),
            usage: USAGE,
            mapped_at_creation: false,
        });
        Self {
            ctx: ctx.clone(),
            buffer,
            dtype,
            len,
        }
    }

    /// Copy the samples back to main memory. Blocks until the GPU is done
    /// with everything submitted before.
    pub fn download(&self) -> Result<FrameData, TransferError> {
        let size = self.buffer.size();
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame download"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
        self.ctx.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| drop(tx.send(r)));
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| TransferError(e.to_string()))?;
        rx.recv()
            .map_err(|e| TransferError(e.to_string()))?
            .map_err(|e| TransferError(e.to_string()))?;

        let view = staging.slice(..).get_mapped_range();
        let data = match self.dtype {
            DType::U8 => FrameData::U8(view[..self.len].to_vec()),
            DType::F32 => FrameData::F32(
                view.chunks_exact(4)
                    .take(self.len)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
            ),
        };
        drop(view);
        staging.unmap();
        Ok(data)
    }

    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of samples (not bytes).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of `u32`/`f32` words the buffer holds, padding included: what
    /// a kernel indexes.
    pub fn words(&self) -> u32 {
        (self.buffer.size() / 4) as u32
    }
}

/// Threads per workgroup every [`Kernel`] shader must declare:
/// `@compute @workgroup_size(64)`.
pub const WORKGROUP_SIZE: u32 = 64;

/// Most workgroups one dispatch dimension may have (WebGPU's minimum limit).
const MAX_GROUPS_PER_DIM: u32 = 65_535;

/// A compiled compute shader over storage buffers.
///
/// The shader's entry point is `main`, with `@workgroup_size(64)` and its
/// buffers at `@group(0) @binding(0..n)`, in the order passed to
/// [`Kernel::run`]. Large dispatches are split over two dimensions, so the
/// shader computes its element index as
/// `gid.x + gid.y * nwg.x * 64u` (from `@builtin(global_invocation_id) gid` and
/// `@builtin(num_workgroups) nwg`) and must ignore indices past the end.
#[derive(Debug)]
pub struct Kernel {
    pipeline: wgpu::ComputePipeline,
}

impl Kernel {
    fn new(ctx: &GpuContext, label: &str, wgsl: &str) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self { pipeline }
    }

    /// Run over `threads` elements (one invocation each) with `buffers` bound
    /// in order, and submit. Does not wait for the GPU.
    pub fn run(&self, ctx: &GpuContext, buffers: &[&wgpu::Buffer], threads: u32) {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let groups = threads.div_ceil(WORKGROUP_SIZE).max(1);
        let x = groups.min(MAX_GROUPS_PER_DIM);
        let y = groups.div_ceil(x);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A context for tests, or `None` (test skipped) when there is no
    /// adapter. With `PIPE_GRAPH_REQUIRE_GPU` set — as in CI — a missing
    /// adapter fails the test instead.
    pub(crate) fn context() -> Option<GpuContext> {
        match GpuContext::new() {
            Ok(ctx) => {
                if let Some(info) = ctx.adapter_info() {
                    eprintln!("GPU test on {} ({:?})", info.name, info.backend);
                }
                Some(ctx)
            }
            Err(e) if std::env::var_os("PIPE_GRAPH_REQUIRE_GPU").is_some() => {
                panic!("PIPE_GRAPH_REQUIRE_GPU is set but {e}")
            }
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    #[test]
    fn padding_rounds_up_to_whole_words() {
        assert_eq!(padded_size(DType::U8, 0), 4);
        assert_eq!(padded_size(DType::U8, 5), 8);
        assert_eq!(padded_size(DType::U8, 8), 8);
        assert_eq!(padded_size(DType::F32, 3), 12);
    }

    #[test]
    fn round_trips_every_dtype_and_odd_length() {
        let Some(ctx) = context() else { return };
        for data in [
            FrameData::U8((0..=254).collect()),
            FrameData::U8(vec![]),
            FrameData::F32(vec![0.0, -1.5, f32::MAX, 1e-30, 0.25]),
        ] {
            let gpu = GpuBuffer::upload(&ctx, &data);
            assert_eq!((gpu.dtype(), gpu.len()), (data.dtype(), data.len()));
            assert_eq!(gpu.download().unwrap(), data);
        }
    }

    #[test]
    fn kernels_are_cached_per_key() {
        let Some(ctx) = context() else { return };
        const COPY: &str = "
            @group(0) @binding(0) var<storage, read> src: array<u32>;
            @group(0) @binding(1) var<storage, read_write> dst: array<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>,
                    @builtin(num_workgroups) nwg: vec3<u32>) {
                let i = gid.x + gid.y * nwg.x * 64u;
                if (i < arrayLength(&dst)) { dst[i] = src[i]; }
            }";
        let a = ctx.kernel("test_copy", COPY);
        assert!(Arc::ptr_eq(&a, &ctx.kernel("test_copy", COPY)));

        let src = GpuBuffer::upload(&ctx, &FrameData::U8((0..200).collect()));
        let dst = GpuBuffer::zeroed(&ctx, DType::U8, 200);
        a.run(&ctx, &[src.buffer(), dst.buffer()], dst.words());
        assert_eq!(dst.download().unwrap(), src.download().unwrap());
    }
}
