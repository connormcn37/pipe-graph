//! WGSL kernels behind the stages' GPU paths (`gpu` feature), and the one
//! helper that dispatches them.
//!
//! Every kernel follows [`crate::gpu::Kernel`]'s conventions: `main`,
//! `@workgroup_size(64)`, element index `gid.x + gid.y * nwg.x * 64u`, and
//! bindings in the order the stage passes buffers — inputs, then the output,
//! then (if any) a `params: array<u32>` (floats travel as their bits).
//!
//! `u8` frames are packed four samples per word (see [`crate::gpu`]), so a
//! kernel writing `u8` runs one invocation per *output word* and fills its four
//! bytes; nothing else writes that word, so there are no races. Kernels
//! writing `f32` run one invocation per sample.
//!
//! Results are meant to be bit-identical to the CPU stages: `u8` maps use the
//! stages' own lookup tables, integer arithmetic is integer, rounding follows
//! Rust's `round` (half away from zero, not WGSL's half-to-even), and moves
//! copy `f32` samples as raw bits. The exception is `box_blur` on `f32`, which
//! sums in `f32` where the CPU uses `f64` (see the conformance tolerances).

use crate::gpu::{GpuBuffer, GpuContext};

/// Run kernel `key` (compiled from `wgsl` on first use) over `threads`
/// invocations with `buffers` bound in order, followed by `params` (if
/// non-empty) as a storage buffer of `u32`s.
pub(crate) fn dispatch(
    ctx: &GpuContext,
    (key, wgsl): (&'static str, &'static str),
    buffers: &[&wgpu::Buffer],
    params: &[u32],
    threads: u32,
) {
    let kernel = ctx.kernel(key, wgsl);
    let params = (!params.is_empty()).then(|| ctx.params(params));
    let mut bound: Vec<&wgpu::Buffer> = buffers.to_vec();
    bound.extend(params.as_ref());
    kernel.run(ctx, &bound, threads);
}

/// Invocations for a kernel writing `dst`: one per word, which is one per
/// `u8` word and one per `f32` sample.
pub(crate) fn threads(dst: &GpuBuffer) -> u32 {
    dst.words()
}

/// The 256-entry table as the `u32`s a kernel indexes.
pub(crate) fn lut_params(lut: &[u8; 256]) -> Vec<u32> {
    lut.iter().map(|&v| v as u32).collect()
}

/// `u8 -> u8` through a 256-entry table: `gain` and `threshold`.
pub(crate) const LUT_U8: (&str, &str) = (
    "lut_u8",
    "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= arrayLength(&dst)) { return; }
    let w = src[i];
    var o = 0u;
    for (var j = 0u; j < 4u; j++) {
        o |= lut[(w >> (j * 8u)) & 255u] << (j * 8u);
    }
    dst[i] = o;
}",
);

/// `f32 -> f32`: `v * params[0]` (gain).
pub(crate) const GAIN_F32: (&str, &str) = (
    "gain_f32",
    "
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= arrayLength(&dst)) { return; }
    dst[i] = src[i] * bitcast<f32>(p[0]);
}",
);

/// `f32 -> f32`: `1.0` if `v >= params[0]`, else `0.0` (threshold).
pub(crate) const THRESHOLD_F32: (&str, &str) = (
    "threshold_f32",
    "
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= arrayLength(&dst)) { return; }
    dst[i] = select(0.0, 1.0, src[i] >= bitcast<f32>(p[0]));
}",
);

/// `u8 -> f32`: `v as f32 * params[0]` (cast).
pub(crate) const CAST_U8_F32: (&str, &str) = (
    "cast_u8_f32",
    "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= arrayLength(&dst)) { return; }
    let v = (src[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
    dst[i] = f32(v) * bitcast<f32>(p[0]);
}",
);

/// `f32 -> u8`: `round(v * params[0])` clamped to `0..=255`, rounding half
/// away from zero like Rust (WGSL's `round` is half-to-even), NaN to 0.
/// `params[1]` is the sample count.
pub(crate) const CAST_F32_U8: (&str, &str) = (
    "cast_f32_u8",
    "
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
fn to_u8(x: f32) -> u32 {
    // NaN is 0, as in Rust. Tested on the bits: compilers may assume floats
    // are never NaN and fold away `x != x`.
    let b = bitcast<u32>(x);
    if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u) { return 0u; }
    let t = trunc(x);
    var r = t;
    if (abs(x - t) >= 0.5) { r = t + sign(x); }
    return u32(clamp(r, 0.0, 255.0));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = gid.x + gid.y * nwg.x * 64u;
    if (w >= arrayLength(&dst)) { return; }
    let scale = bitcast<f32>(p[0]);
    var o = 0u;
    for (var j = 0u; j < 4u; j++) {
        let i = w * 4u + j;
        if (i < p[1]) { o |= to_u8(src[i] * scale) << (j * 8u); }
    }
    dst[w] = o;
}",
);

/// `u8` RGB(A) → 1-channel BT.601 luma, integer and rounded half-up.
/// `params = [channels, pixels]`.
pub(crate) const GRAYSCALE_U8: (&str, &str) = (
    "grayscale_u8",
    "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
fn ld(i: u32) -> u32 { return (src[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = gid.x + gid.y * nwg.x * 64u;
    if (w >= arrayLength(&dst)) { return; }
    let k = p[0];
    var o = 0u;
    for (var j = 0u; j < 4u; j++) {
        let px = w * 4u + j;
        if (px < p[1]) {
            let y = 299u * ld(px * k) + 587u * ld(px * k + 1u) + 114u * ld(px * k + 2u);
            o |= ((y + 500u) / 1000u) << (j * 8u);
        }
    }
    dst[w] = o;
}",
);

/// `f32` RGB(A) → 1-channel BT.601 luma. `params = [channels]`.
pub(crate) const GRAYSCALE_F32: (&str, &str) = (
    "grayscale_f32",
    "
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let px = gid.x + gid.y * nwg.x * 64u;
    if (px >= arrayLength(&dst)) { return; }
    let b = px * p[0];
    dst[px] = 0.299 * src[b] + 0.587 * src[b + 1u] + 0.114 * src[b + 2u];
}",
);

// Gathers: output sample `i` is input sample `src_index(i)`, by mode `p[0]`:
// - `0` crop: `p = [0, n_out, src_width, channels, x, y, w]`
// - `1` split: `p = [1, n_out, channels, channel]`

/// Gather `u8` samples: output sample `i` is input sample `src_index(i)`.
pub(crate) const GATHER_U8: (&str, &str) = (
    "gather_u8",
    concat!(
        "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
fn ld(i: u32) -> u32 { return (src[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = gid.x + gid.y * nwg.x * 64u;
    if (w >= arrayLength(&dst)) { return; }
    var o = 0u;
    for (var j = 0u; j < 4u; j++) {
        let i = w * 4u + j;
        if (i < p[1]) { o |= ld(src_index(i)) << (j * 8u); }
    }
    dst[w] = o;
}",
        "
fn src_index(i: u32) -> u32 {
    if (p[0] == 0u) {
        let ch = p[3];
        let px = i / ch;
        return ((p[5] + px / p[6]) * p[2] + p[4] + px % p[6]) * ch + i % ch;
    }
    return i * p[2] + p[3];
}"
    ),
);

/// Gather 4-byte samples (`f32`, moved as raw bits so every value — NaN
/// payloads included — survives exactly).
pub(crate) const GATHER_W: (&str, &str) = (
    "gather_w",
    concat!(
        "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= p[1]) { return; }
    dst[i] = src[src_index(i)];
}",
        "
fn src_index(i: u32) -> u32 {
    if (p[0] == 0u) {
        let ch = p[3];
        let px = i / ch;
        return ((p[5] + px / p[6]) * p[2] + p[4] + px % p[6]) * ch + i % ch;
    }
    return i * p[2] + p[3];
}"
    ),
);

/// Write single-channel `u8` input into channel `p[1]` of a `p[0]`-channel
/// output of `p[2]` samples, keeping the other channels' bytes: one pass of
/// `merge` per input.
pub(crate) const MERGE_U8: (&str, &str) = (
    "merge_u8",
    "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
fn ld(i: u32) -> u32 { return (src[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = gid.x + gid.y * nwg.x * 64u;
    if (w >= arrayLength(&dst)) { return; }
    var o = dst[w];
    for (var j = 0u; j < 4u; j++) {
        let i = w * 4u + j;
        if (i < p[2] && i % p[0] == p[1]) {
            o = (o & ~(255u << (j * 8u))) | (ld(i / p[0]) << (j * 8u));
        }
    }
    dst[w] = o;
}",
);

/// Same as [`MERGE_U8`] for 4-byte samples: `p = [channels, channel,
/// pixels]`, one invocation per input sample.
pub(crate) const MERGE_W: (&str, &str) = (
    "merge_w",
    "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let px = gid.x + gid.y * nwg.x * 64u;
    if (px >= p[2]) { return; }
    dst[px * p[0] + p[1]] = src[px];
}",
);

/// Box blur, horizontal pass over `u8`: each sample becomes the `u32` sum of
/// its row window (edges clamped). `p = [width, height, channels, radius,
/// samples]`.
pub(crate) const BLUR_H_U8: (&str, &str) = (
    "blur_h_u8",
    "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
fn ld(i: u32) -> u32 { return (src[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= p[4]) { return; }
    let ch = p[2];
    let px = i / ch;
    let x = i32(px % p[0]);
    let row = (px / p[0]) * p[0];
    let r = i32(p[3]);
    var sum = 0u;
    for (var dx = -r; dx <= r; dx++) {
        let xx = u32(clamp(x + dx, 0, i32(p[0]) - 1));
        sum += ld((row + xx) * ch + i % ch);
    }
    dst[i] = sum;
}",
);

/// Box blur, vertical pass back to `u8`: sum the row sums of the column
/// window and take the mean rounded half-up, as the CPU does.
pub(crate) const BLUR_V_U8: (&str, &str) = (
    "blur_v_u8",
    "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = gid.x + gid.y * nwg.x * 64u;
    if (w >= arrayLength(&dst)) { return; }
    let ch = p[2];
    let r = i32(p[3]);
    let count = u32((2 * r + 1) * (2 * r + 1));
    var o = 0u;
    for (var j = 0u; j < 4u; j++) {
        let i = w * 4u + j;
        if (i < p[4]) {
            let px = i / ch;
            let x = px % p[0];
            let y = i32(px / p[0]);
            var sum = 0u;
            for (var dy = -r; dy <= r; dy++) {
                let yy = u32(clamp(y + dy, 0, i32(p[1]) - 1));
                sum += src[(yy * p[0] + x) * ch + i % ch];
            }
            o |= ((sum + count / 2u) / count) << (j * 8u);
        }
    }
    dst[w] = o;
}",
);

/// Box blur, horizontal pass over `f32` (row-window sums).
pub(crate) const BLUR_H_F32: (&str, &str) = (
    "blur_h_f32",
    "
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= p[4]) { return; }
    let ch = p[2];
    let px = i / ch;
    let x = i32(px % p[0]);
    let row = (px / p[0]) * p[0];
    let r = i32(p[3]);
    var sum = 0.0;
    for (var dx = -r; dx <= r; dx++) {
        let xx = u32(clamp(x + dx, 0, i32(p[0]) - 1));
        sum += src[(row + xx) * ch + i % ch];
    }
    dst[i] = sum;
}",
);

/// Box blur, vertical pass over `f32` row sums, divided by the window size.
pub(crate) const BLUR_V_F32: (&str, &str) = (
    "blur_v_f32",
    "
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= p[4]) { return; }
    let ch = p[2];
    let px = i / ch;
    let x = px % p[0];
    let y = i32(px / p[0]);
    let r = i32(p[3]);
    var sum = 0.0;
    for (var dy = -r; dy <= r; dy++) {
        let yy = u32(clamp(y + dy, 0, i32(p[1]) - 1));
        sum += src[(yy * p[0] + x) * ch + i % ch];
    }
    dst[i] = sum / f32((2 * r + 1) * (2 * r + 1));
}",
);
