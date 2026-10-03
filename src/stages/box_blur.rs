//! Box blur: mean over a `(2r+1) x (2r+1)` window, per channel.
//!
//! Semantics:
//! - Each output sample is the unweighted mean of the `(2r+1)^2` samples of
//!   the *same channel* centred on it. Channels never mix.
//! - Edges are handled by **clamping** coordinates into the frame (i.e. the
//!   border pixel is replicated outward), so the divisor is always
//!   `(2r+1)^2` and a flat frame stays exactly flat.
//! - `radius = 0` is an identity copy; `radius` is capped at
//!   [`BoxBlurStage::MAX_RADIUS`].
//! - `u8`: sums are exact integers; the mean is rounded half-up.
//! - `f32`: sums are accumulated in `f64` (so the running sum does not drift
//!   over a 1920-wide row) and the mean is narrowed back to `f32`. A NaN/inf
//!   sample only affects the windows containing it (such frames take a slower
//!   non-sliding path; see `box_blur`).
//!
//! Implementation: the box kernel is separable, so we do a horizontal pass
//! (row sums) followed by a vertical pass (sums of row sums). Each pass is a
//! *running* sum — slide the window by adding the sample entering and
//! subtracting the one leaving — so the cost is O(1) per sample regardless of
//! `radius` (plus O(r) per row/column to prime the window).

use std::ops::{AddAssign, SubAssign};

use crate::data::{Frame, FrameData, Payload, PayloadKind};
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec};
use crate::graph::Params;

/// 1→1 stage that box-blurs every channel with the given `radius`.
pub struct BoxBlurStage {
    radius: u32,
}

impl BoxBlurStage {
    /// Largest accepted radius (a 8193-wide window — wider than a 4K frame).
    /// Priming a window costs O(r), so an unbounded radius from a serialized
    /// graph could stall `run_once` for minutes.
    pub const MAX_RADIUS: u32 = 4096;

    /// Panics if `radius > MAX_RADIUS`; `TryFrom<&Params>` is the fallible
    /// build path.
    pub fn new(radius: u32) -> Self {
        assert!(
            radius <= Self::MAX_RADIUS,
            "box_blur radius {radius} exceeds {}",
            Self::MAX_RADIUS
        );
        Self { radius }
    }
}

impl TryFrom<&Params> for BoxBlurStage {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        let radius = p.get_u32("radius")?;
        if radius > Self::MAX_RADIUS {
            return Err(BuildError::BadParam {
                key: "radius".to_string(),
                value: radius.to_string(),
                expected: "u32 <= 4096",
            });
        }
        Ok(Self { radius })
    }
}

/// A sample type that can be summed exactly enough and averaged back.
trait BlurSample: Copy {
    /// Wide accumulator: big enough that `(2r+1)^2` samples never overflow
    /// (`u64` for `u8`) or lose meaningful precision (`f64` for `f32`).
    type Acc: Copy + Default + AddAssign + SubAssign;
    fn widen(self) -> Self::Acc;
    fn mean(sum: Self::Acc, count: u64) -> Self;
}

impl BlurSample for u8 {
    type Acc = u64;
    fn widen(self) -> u64 {
        self as u64
    }
    fn mean(sum: u64, count: u64) -> u8 {
        // Round half-up; sum <= 255 * count, so the result fits in u8.
        ((sum + count / 2) / count) as u8
    }
}

impl BlurSample for f32 {
    type Acc = f64;
    fn widen(self) -> f64 {
        self as f64
    }
    fn mean(sum: f64, count: u64) -> f32 {
        (sum / count as f64) as f32
    }
}

/// Separable box blur over an interleaved `(w, h, ch)` buffer.
///
/// With `running = true` each pass slides its window (O(1) per sample). With
/// `running = false` every window is summed afresh (O(r) per sample): slower,
/// but a non-finite sample only affects the windows that actually contain it,
/// whereas in a running sum `NaN - NaN` / `inf - inf` would poison the rest
/// of the row and column.
fn box_blur<T: BlurSample>(
    src: &[T],
    w: usize,
    h: usize,
    ch: usize,
    r: usize,
    running: bool,
) -> Vec<T> {
    if r == 0 || src.is_empty() {
        return src.to_vec();
    }
    let row_len = w * ch;
    // `i` may be negative or past the end; clamp it to a valid index.
    let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
    let ri = r as isize;

    // Horizontal pass: horiz[y][x][c] = sum of src[y][x-r..=x+r][c].
    let mut horiz = vec![T::Acc::default(); src.len()];
    let mut sums = vec![T::Acc::default(); ch];
    for (row, out) in src
        .chunks_exact(row_len)
        .zip(horiz.chunks_exact_mut(row_len))
    {
        for x in 0..w as isize {
            if x == 0 || !running {
                for (c, s) in sums.iter_mut().enumerate() {
                    *s = T::Acc::default();
                    for i in x - ri..=x + ri {
                        *s += row[clamp(i, w) * ch + c].widen();
                    }
                }
            }
            let enter = clamp(x + ri + 1, w) * ch;
            let leave = clamp(x - ri, w) * ch;
            for (c, s) in sums.iter_mut().enumerate() {
                out[x as usize * ch + c] = *s;
                if running {
                    // Add before subtracting so unsigned sums never underflow.
                    *s += row[enter + c].widen();
                    *s -= row[leave + c].widen();
                }
            }
        }
    }

    // Vertical pass: slide a whole row of accumulators down the frame, which
    // keeps memory access sequential instead of striding down columns.
    let count = (2 * r as u64 + 1) * (2 * r as u64 + 1);
    let hrow = |y: isize| {
        let y = clamp(y, h);
        &horiz[y * row_len..(y + 1) * row_len]
    };
    let mut acc = vec![T::Acc::default(); row_len];
    let mut out = Vec::with_capacity(src.len());
    for y in 0..h as isize {
        if y == 0 || !running {
            acc.fill(T::Acc::default());
            for j in y - ri..=y + ri {
                for (a, &v) in acc.iter_mut().zip(hrow(j)) {
                    *a += v;
                }
            }
        }
        out.extend(acc.iter().map(|&a| T::mean(a, count)));
        if running {
            let (enter, leave) = (hrow(y + ri + 1), hrow(y - ri));
            for ((a, &e), &l) in acc.iter_mut().zip(enter).zip(leave) {
                *a += e;
                *a -= l;
            }
        }
    }
    out
}

impl Node for BoxBlurStage {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Frame)],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;
        let (w, h, ch) = (
            frame.width as usize,
            frame.height as usize,
            frame.channels as usize,
        );
        let r = self.radius as usize;
        let data = match frame.data() {
            FrameData::U8(buf) => FrameData::U8(box_blur(buf, w, h, ch, r, true)),
            FrameData::F32(buf) => {
                // Integer sums are exact; float ones are only safe to slide
                // when every sample is finite (see `box_blur`).
                let running = buf.iter().all(|v| v.is_finite());
                FrameData::F32(box_blur(buf, w, h, ch, r, running))
            }
        };
        let out = Frame::from_data(frame.width, frame.height, frame.channels, data);
        outputs.set("out", Payload::Frame(out));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PortId;
    use std::collections::HashMap;

    fn run(radius: u32, frame: Frame) -> Frame {
        let mut m = HashMap::new();
        m.insert(PortId("in".to_string()), Payload::Frame(frame));
        let mut out = Outputs::new();
        BoxBlurStage::new(radius)
            .eval(&Inputs::new(m), &mut out)
            .unwrap();
        out.get("out").unwrap().as_frame().unwrap().clone()
    }

    /// Direct O(r^2) reference with the same clamping + rounding rules.
    fn naive_u8(src: &[u8], w: usize, h: usize, ch: usize, r: usize) -> Vec<u8> {
        let ri = r as isize;
        let cl = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
        let count = ((2 * r + 1) * (2 * r + 1)) as u64;
        let mut out = Vec::new();
        for y in 0..h as isize {
            for x in 0..w as isize {
                for c in 0..ch {
                    let mut s = 0u64;
                    for dy in -ri..=ri {
                        for dx in -ri..=ri {
                            s += src[(cl(y + dy, h) * w + cl(x + dx, w)) * ch + c] as u64;
                        }
                    }
                    out.push(((s + count / 2) / count) as u8);
                }
            }
        }
        out
    }

    #[test]
    fn radius_zero_is_identity() {
        let f = Frame::from_rgb8(2, 1, vec![(1, 2, 3), (4, 5, 6)]);
        assert_eq!(run(0, f.clone()), f);
    }

    #[test]
    fn hand_computed_3x1_with_edge_clamping() {
        // Row [0, 90, 180], r=1, single row so vertical clamping triples each
        // row sum and the divisor is 9:
        //   x0: (0+0+90)*3/9 = 30
        //   x1: (0+90+180)*3/9 = 90
        //   x2: (90+180+180)*3/9 = 150
        let f = Frame::from_data(3, 1, 1, FrameData::U8(vec![0, 90, 180]));
        assert_eq!(run(1, f).as_u8().unwrap(), &[30, 90, 150]);
    }

    #[test]
    fn channels_do_not_mix() {
        // Channel 0 is flat 10, channel 1 is flat 200: blur must keep both.
        let buf: Vec<u8> = (0..4 * 3).flat_map(|_| [10u8, 200u8]).collect();
        let f = Frame::from_data(4, 3, 2, FrameData::U8(buf.clone()));
        assert_eq!(run(2, f).as_u8().unwrap(), buf.as_slice());
    }

    #[test]
    fn matches_naive_reference() {
        let (w, h, ch) = (7usize, 5usize, 3usize);
        let src: Vec<u8> = (0..w * h * ch).map(|i| (i * 37 % 251) as u8).collect();
        for r in [1usize, 2, 3, 9] {
            let f = Frame::from_data(w as u32, h as u32, ch as u32, FrameData::U8(src.clone()));
            assert_eq!(
                run(r as u32, f).as_u8().unwrap(),
                naive_u8(&src, w, h, ch, r).as_slice(),
                "radius {r}"
            );
        }
    }

    #[test]
    fn f32_non_finite_stays_local() {
        // NaN at x=0 of a 5x1 row, r=1: only x=0 and x=1 see it.
        let f = Frame::from_data(5, 1, 1, FrameData::F32(vec![f32::NAN, 0.3, 0.3, 0.3, 0.3]));
        let out = run(1, f);
        let d = out.as_f32().unwrap();
        assert!(d[0].is_nan() && d[1].is_nan());
        for v in &d[2..] {
            assert!((v - 0.3).abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn radius_above_cap_is_rejected() {
        let mut p = Params::new();
        p.insert("radius".to_string(), "4000000000".to_string());
        assert!(matches!(
            BoxBlurStage::try_from(&p),
            Err(BuildError::BadParam { .. })
        ));
    }

    #[test]
    fn f32_mean() {
        let f = Frame::from_data(3, 1, 1, FrameData::F32(vec![0.0, 0.3, 0.6]));
        let out = run(1, f);
        let d = out.as_f32().unwrap();
        for (got, want) in d.iter().zip([0.1f32, 0.3, 0.5]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }
}
