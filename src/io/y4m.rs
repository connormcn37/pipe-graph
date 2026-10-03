//! A small, pure-Rust YUV4MPEG2 (`.y4m`) reader and writer.
//!
//! Y4M is the simplest container that every video tool speaks: a one-line
//! text header (`YUV4MPEG2 W.. H.. F.. C..`) followed by frames, each a
//! `FRAME` line and then raw planar YUV. That makes it the natural interchange
//! format here: files can be read and written with no dependencies, and the
//! *same* code drives `ffmpeg` through a pipe (`-f yuv4mpegpipe`), so ffmpeg
//! support is "spawn a process" rather than a second codec path.
//!
//! Both halves are generic over [`Read`]/[`Write`] for exactly that reason.
//!
//! Supported colour spaces (`C` tag): `420jpeg`, `420mpeg2`, `420paldv` and
//! plain `420` (all 4:2:0; chroma *siting* differs between them, which we treat
//! loosely — nearest-neighbour upsampling and 2x2 box downsampling), `444`,
//! and `mono`. Anything else (4:2:2, alpha, high bit depth) is rejected with a
//! [`Y4mError::Unsupported`] rather than being misread as garbage.
//!
//! Colour conversion is BT.601. Y4M carries no matrix information, and 601 is
//! what ffmpeg assumes for it. Range is *limited* (studio swing, Y in 16..=235)
//! unless the header carries ffmpeg's `XCOLORRANGE=FULL` extension.

use std::io::{self, BufRead, BufReader, Read, Write};

use crate::data::{Frame, FrameData};

/// Longest header / frame line we accept. Real headers are ~60 bytes; the cap
/// only exists so a non-Y4M input cannot make us buffer without bound.
const MAX_LINE: u64 = 4096;

/// Largest width or height we accept from a header (well beyond 8K/16K video).
const MAX_DIM: u32 = 32768;

/// Errors from reading or writing a Y4M stream.
#[derive(Debug)]
pub enum Y4mError {
    /// The underlying reader/writer failed.
    Io(io::Error),
    /// The bytes are not a well-formed Y4M stream.
    Format(String),
    /// Well-formed, but uses a feature this implementation does not handle.
    Unsupported(String),
}

impl std::fmt::Display for Y4mError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Y4mError::Io(e) => write!(f, "y4m i/o error: {e}"),
            Y4mError::Format(m) => write!(f, "malformed y4m: {m}"),
            Y4mError::Unsupported(m) => write!(f, "unsupported y4m: {m}"),
        }
    }
}

impl std::error::Error for Y4mError {}

impl From<io::Error> for Y4mError {
    fn from(e: io::Error) -> Self {
        Y4mError::Io(e)
    }
}

/// Chroma layout of a Y4M stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chroma {
    /// 4:2:0 — one chroma sample per 2x2 luma block.
    C420,
    /// 4:4:4 — full-resolution chroma.
    C444,
    /// Luma only (greyscale).
    Mono,
}

impl Chroma {
    /// Parse the value of a `C` header tag.
    pub fn parse(tag: &str) -> Result<Self, Y4mError> {
        match tag {
            "420" | "420jpeg" | "420mpeg2" | "420paldv" => Ok(Chroma::C420),
            "444" => Ok(Chroma::C444),
            "mono" => Ok(Chroma::Mono),
            other => Err(Y4mError::Unsupported(format!(
                "colour space 'C{other}' (supported: C420, C420jpeg, C420mpeg2, C420paldv, C444, Cmono)"
            ))),
        }
    }

    /// The tag we write. `420jpeg` (centred chroma) is what our box filter
    /// actually produces, and it is ffmpeg's default for 4:2:0 output.
    fn tag(self) -> &'static str {
        match self {
            Chroma::C420 => "420jpeg",
            Chroma::C444 => "444",
            Chroma::Mono => "mono",
        }
    }

    /// Size of one chroma plane for a `w x h` frame (0 for mono).
    fn chroma_dims(self, w: usize, h: usize) -> (usize, usize) {
        match self {
            Chroma::C420 => (w.div_ceil(2), h.div_ceil(2)),
            Chroma::C444 => (w, h),
            Chroma::Mono => (0, 0),
        }
    }
}

/// The stream-level parameters from a Y4M header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Y4mHeader {
    pub width: u32,
    pub height: u32,
    /// Frame rate as a rational `num / den`.
    pub fps_num: u32,
    pub fps_den: u32,
    pub chroma: Chroma,
    /// Full-range (`0..=255`) rather than limited-range YUV.
    pub full_range: bool,
}

impl Y4mHeader {
    /// A header with the defaults we write: limited range, given rate.
    pub fn new(width: u32, height: u32, fps: (u32, u32), chroma: Chroma) -> Self {
        Self {
            width,
            height,
            fps_num: fps.0,
            fps_den: fps.1,
            chroma,
            full_range: false,
        }
    }

    /// Bytes of planar data in one frame.
    pub fn frame_len(&self) -> usize {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = self.chroma.chroma_dims(w, h);
        w * h + 2 * cw * ch
    }

    fn parse(line: &str) -> Result<Self, Y4mError> {
        let mut tokens = line.split(' ').filter(|t| !t.is_empty());
        if tokens.next() != Some("YUV4MPEG2") {
            return Err(Y4mError::Format("missing 'YUV4MPEG2' signature".into()));
        }
        let (mut width, mut height) = (None, None);
        let (mut fps_num, mut fps_den) = (25, 1);
        // The spec's default when `C` is absent.
        let mut chroma = Chroma::C420;
        let mut full_range = false;
        for tok in tokens {
            // Split off the one-letter tag by `char`, not byte: a malformed
            // header may start a token with a multi-byte character.
            let mut chars = tok.chars();
            let tag = chars.next();
            let val = chars.as_str();
            match tag {
                Some('W') => width = Some(parse_num(val, "width")?),
                Some('H') => height = Some(parse_num(val, "height")?),
                Some('F') => {
                    let (n, d) = val
                        .split_once(':')
                        .ok_or_else(|| Y4mError::Format(format!("bad frame rate '{val}'")))?;
                    fps_num = parse_num(n, "frame rate")?;
                    fps_den = parse_num(d, "frame rate")?;
                }
                Some('C') => chroma = Chroma::parse(val)?,
                Some('X') if val == "COLORRANGE=FULL" => full_range = true,
                // Interlacing, aspect ratio and other extensions do not change
                // how the bytes are laid out, so they are safe to ignore.
                _ => {}
            }
        }
        let width = width.ok_or_else(|| Y4mError::Format("missing W (width)".into()))?;
        let height = height.ok_or_else(|| Y4mError::Format("missing H (height)".into()))?;
        if width == 0 || height == 0 || width > MAX_DIM || height > MAX_DIM {
            // The upper bound keeps a corrupt header from overflowing the
            // frame-size arithmetic or requesting an absurd allocation.
            return Err(Y4mError::Format(format!(
                "frame size {width}x{height} out of range (1..={MAX_DIM} per side)"
            )));
        }
        Ok(Self {
            width,
            height,
            fps_num,
            fps_den,
            chroma,
            full_range,
        })
    }

    fn to_line(&self) -> String {
        let mut s = format!(
            "YUV4MPEG2 W{} H{} F{}:{} Ip A1:1 C{}",
            self.width,
            self.height,
            self.fps_num,
            self.fps_den,
            self.chroma.tag()
        );
        if self.full_range {
            s.push_str(" XCOLORRANGE=FULL");
        }
        s.push('\n');
        s
    }
}

fn parse_num(s: &str, what: &str) -> Result<u32, Y4mError> {
    s.parse()
        .map_err(|_| Y4mError::Format(format!("bad {what} '{s}'")))
}

/// Read one `\n`-terminated line (without the newline), bounded by
/// [`MAX_LINE`]. Returns `Ok(None)` on a clean EOF before any byte.
fn read_line<R: BufRead>(r: &mut R) -> Result<Option<Vec<u8>>, Y4mError> {
    let mut line = Vec::new();
    r.by_ref().take(MAX_LINE).read_until(b'\n', &mut line)?;
    if line.is_empty() {
        return Ok(None);
    }
    if line.pop() != Some(b'\n') {
        return Err(Y4mError::Format(if line.len() as u64 >= MAX_LINE - 1 {
            "header line too long (is this a y4m stream?)".into()
        } else {
            "stream truncated inside a header line".into()
        }));
    }
    Ok(Some(line))
}

/// Streaming Y4M decoder over any [`Read`] (a file, a pipe from ffmpeg, ...).
pub struct Y4mReader<R: Read> {
    inner: BufReader<R>,
    header: Y4mHeader,
    /// Reused planar buffer, so steady-state reading does not allocate it.
    planes: Vec<u8>,
}

impl<R: Read> Y4mReader<R> {
    /// Read and parse the stream header.
    pub fn new(reader: R) -> Result<Self, Y4mError> {
        let mut inner = BufReader::new(reader);
        let line = read_line(&mut inner)?
            .ok_or_else(|| Y4mError::Format("empty stream (no header)".into()))?;
        let line =
            String::from_utf8(line).map_err(|_| Y4mError::Format("header is not ASCII".into()))?;
        let header = Y4mHeader::parse(&line)?;
        let planes = vec![0; header.frame_len()];
        Ok(Self {
            inner,
            header,
            planes,
        })
    }

    pub fn header(&self) -> &Y4mHeader {
        &self.header
    }

    /// Read the next frame's raw planes (Y, then U, then V). Returns `None` at
    /// a clean end of stream; a stream cut off mid-frame is an error.
    pub fn read_planes(&mut self) -> Result<Option<&[u8]>, Y4mError> {
        let Some(line) = read_line(&mut self.inner)? else {
            return Ok(None);
        };
        // `FRAME` may carry per-frame parameters after a space; none of them
        // change the layout, so only the keyword is checked.
        if !(line == b"FRAME" || line.starts_with(b"FRAME ")) {
            return Err(Y4mError::Format("expected a FRAME marker".into()));
        }
        self.inner.read_exact(&mut self.planes).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                Y4mError::Format("stream truncated inside frame data".into())
            } else {
                Y4mError::Io(e)
            }
        })?;
        Ok(Some(&self.planes))
    }

    /// Read the next frame as a 3-channel `u8` RGB [`Frame`] (mono streams are
    /// expanded to grey RGB, so downstream stages see one shape).
    pub fn read_rgb(&mut self) -> Result<Option<Frame>, Y4mError> {
        let header = self.header.clone();
        Ok(self.read_planes()?.map(|p| planes_to_rgb(&header, p)))
    }
}

/// Streaming Y4M encoder over any [`Write`].
///
/// The header is written lazily with the first frame, so a writer that never
/// receives a frame leaves its output empty rather than holding a header for
/// a zero-length video.
pub struct Y4mWriter<W: Write> {
    inner: W,
    header: Y4mHeader,
    wrote_header: bool,
    planes: Vec<u8>,
}

impl<W: Write> Y4mWriter<W> {
    pub fn new(writer: W, header: Y4mHeader) -> Self {
        Self {
            inner: writer,
            header,
            wrote_header: false,
            planes: Vec::new(),
        }
    }

    pub fn header(&self) -> &Y4mHeader {
        &self.header
    }

    /// Encode one frame. Accepts 3-channel `u8` RGB, or 1-channel `u8` grey
    /// (a grey frame in a colour stream is written with neutral chroma; an RGB
    /// frame in a mono stream keeps only its luma).
    pub fn write_frame(&mut self, frame: &Frame) -> Result<(), Y4mError> {
        if frame.width != self.header.width || frame.height != self.header.height {
            return Err(Y4mError::Format(format!(
                "frame is {}x{} but the stream is {}x{}",
                frame.width, frame.height, self.header.width, self.header.height
            )));
        }
        rgb_to_planes(&self.header, frame, &mut self.planes)?;
        if !self.wrote_header {
            self.inner.write_all(self.header.to_line().as_bytes())?;
            self.wrote_header = true;
        }
        self.inner.write_all(b"FRAME\n")?;
        self.inner.write_all(&self.planes)?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), Y4mError> {
        self.inner.flush()?;
        Ok(())
    }

    /// Flush and hand back the underlying writer (e.g. to close a pipe).
    pub fn into_inner(mut self) -> Result<W, Y4mError> {
        self.inner.flush()?;
        Ok(self.inner)
    }
}

// --- BT.601 colour conversion ------------------------------------------------
//
// Fixed-point (8 fractional bits) versions of the standard BT.601 equations.
// Integer maths keeps encode/decode bit-exact across platforms, which the
// round-trip tests rely on.

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// RGB -> (Y, U, V).
fn rgb_to_yuv(r: u8, g: u8, b: u8, full: bool) -> (u8, u8, u8) {
    let (r, g, b) = (r as i32, g as i32, b as i32);
    if full {
        let y = (77 * r + 150 * g + 29 * b + 128) >> 8;
        let u = ((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128;
        let v = ((128 * r - 107 * g - 21 * b + 128) >> 8) + 128;
        (clamp_u8(y), clamp_u8(u), clamp_u8(v))
    } else {
        let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        (clamp_u8(y), clamp_u8(u), clamp_u8(v))
    }
}

/// (Y, U, V) -> RGB.
fn yuv_to_rgb(y: u8, u: u8, v: u8, full: bool) -> (u8, u8, u8) {
    let (d, e) = (u as i32 - 128, v as i32 - 128);
    if full {
        let c = (y as i32) << 8;
        (
            clamp_u8((c + 359 * e + 128) >> 8),
            clamp_u8((c - 88 * d - 183 * e + 128) >> 8),
            clamp_u8((c + 454 * d + 128) >> 8),
        )
    } else {
        let c = 298 * (y as i32 - 16);
        (
            clamp_u8((c + 409 * e + 128) >> 8),
            clamp_u8((c - 100 * d - 208 * e + 128) >> 8),
            clamp_u8((c + 516 * d + 128) >> 8),
        )
    }
}

/// Decode planar YUV into a 3-channel RGB frame.
fn planes_to_rgb(h: &Y4mHeader, planes: &[u8]) -> Frame {
    let (w, ht) = (h.width as usize, h.height as usize);
    let (cw, ch) = h.chroma.chroma_dims(w, ht);
    let (yp, rest) = planes.split_at(w * ht);
    let (up, vp) = rest.split_at(cw * ch);
    let mut out = Vec::with_capacity(w * ht * 3);
    for row in 0..ht {
        for col in 0..w {
            let y = yp[row * w + col];
            let (u, v) = match h.chroma {
                Chroma::Mono => (128, 128),
                Chroma::C444 => (up[row * cw + col], vp[row * cw + col]),
                // Nearest-neighbour: each chroma sample covers its 2x2 block.
                Chroma::C420 => {
                    let i = (row / 2) * cw + col / 2;
                    (up[i], vp[i])
                }
            };
            let (r, g, b) = yuv_to_rgb(y, u, v, h.full_range);
            out.extend_from_slice(&[r, g, b]);
        }
    }
    Frame::from_data(h.width, h.height, 3, FrameData::U8(out))
}

/// Encode a 1- or 3-channel `u8` frame into planar YUV for `h`.
fn rgb_to_planes(h: &Y4mHeader, frame: &Frame, planes: &mut Vec<u8>) -> Result<(), Y4mError> {
    let px = frame.as_u8().ok_or_else(|| {
        Y4mError::Unsupported("frame is f32; cast it to u8 before writing y4m".into())
    })?;
    if !matches!(frame.channels, 1 | 3) {
        return Err(Y4mError::Unsupported(format!(
            "{}-channel frame (y4m output needs 1-channel grey or 3-channel RGB)",
            frame.channels
        )));
    }
    let rgb = |i: usize| -> (u8, u8, u8) {
        match frame.channels {
            1 => (px[i], px[i], px[i]),
            _ => (px[i * 3], px[i * 3 + 1], px[i * 3 + 2]),
        }
    };

    let (w, ht) = (h.width as usize, h.height as usize);
    let (cw, ch) = h.chroma.chroma_dims(w, ht);
    planes.clear();
    planes.resize(h.frame_len(), 0);
    let (yp, rest) = planes.split_at_mut(w * ht);
    let (up, vp) = rest.split_at_mut(cw * ch);

    match h.chroma {
        Chroma::Mono => {
            for (i, y) in yp.iter_mut().enumerate() {
                let (r, g, b) = rgb(i);
                *y = rgb_to_yuv(r, g, b, h.full_range).0;
            }
        }
        Chroma::C444 => {
            for i in 0..w * ht {
                let (r, g, b) = rgb(i);
                let (y, u, v) = rgb_to_yuv(r, g, b, h.full_range);
                (yp[i], up[i], vp[i]) = (y, u, v);
            }
        }
        Chroma::C420 => {
            // Accumulate chroma per 2x2 block, then average (a box filter).
            // Blocks on an odd right/bottom edge average fewer samples.
            let mut acc = vec![(0u32, 0u32, 0u32); cw * ch];
            for row in 0..ht {
                for col in 0..w {
                    let i = row * w + col;
                    let (r, g, b) = rgb(i);
                    let (y, u, v) = rgb_to_yuv(r, g, b, h.full_range);
                    yp[i] = y;
                    let a = &mut acc[(row / 2) * cw + col / 2];
                    a.0 += u as u32;
                    a.1 += v as u32;
                    a.2 += 1;
                }
            }
            for (i, (su, sv, n)) in acc.into_iter().enumerate() {
                up[i] = ((su + n / 2) / n) as u8;
                vp[i] = ((sv + n / 2) / n) as u8;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32) -> Frame {
        let px = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                ((x * 255 / w) as u8, (y * 255 / h) as u8, 128u8)
            })
            .collect();
        Frame::from_rgb8(w, h, px)
    }

    fn max_diff(a: &Frame, b: &Frame) -> u8 {
        a.as_u8()
            .unwrap()
            .iter()
            .zip(b.as_u8().unwrap())
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap()
    }

    #[test]
    fn primaries_convert_close_to_reference() {
        // Limited-range BT.601 reference values for pure red/green/blue/white.
        assert_eq!(rgb_to_yuv(255, 0, 0, false), (82, 90, 240));
        // (Exact green luma is 144.6; the standard 8-bit approximation floors it.)
        assert_eq!(rgb_to_yuv(0, 255, 0, false), (144, 54, 34));
        assert_eq!(rgb_to_yuv(0, 0, 255, false), (41, 240, 110));
        assert_eq!(rgb_to_yuv(255, 255, 255, false), (235, 128, 128));
        assert_eq!(rgb_to_yuv(0, 0, 0, false), (16, 128, 128));
        assert_eq!(yuv_to_rgb(235, 128, 128, false), (255, 255, 255));
        assert_eq!(yuv_to_rgb(16, 128, 128, false), (0, 0, 0));
    }

    #[test]
    fn colour_round_trip_is_near_lossless() {
        for full in [false, true] {
            for &(r, g, b) in &[(255, 0, 0), (0, 255, 0), (12, 200, 99), (128, 128, 128)] {
                let (y, u, v) = rgb_to_yuv(r, g, b, full);
                let (r2, g2, b2) = yuv_to_rgb(y, u, v, full);
                for (a, b) in [(r, r2), (g, g2), (b, b2)] {
                    assert!(a.abs_diff(b) <= 3, "{full} {:?}", (r, g, b));
                }
            }
        }
    }

    #[test]
    fn write_then_read_444_round_trips() {
        let f = gradient(7, 5);
        let mut buf = Vec::new();
        let mut w = Y4mWriter::new(&mut buf, Y4mHeader::new(7, 5, (30, 1), Chroma::C444));
        w.write_frame(&f).unwrap();
        w.write_frame(&f).unwrap();
        drop(w);

        let mut r = Y4mReader::new(&buf[..]).unwrap();
        assert_eq!(r.header().fps_num, 30);
        assert_eq!(r.header().chroma, Chroma::C444);
        let a = r.read_rgb().unwrap().unwrap();
        let b = r.read_rgb().unwrap().unwrap();
        assert!(r.read_rgb().unwrap().is_none());
        assert!(max_diff(&a, &f) <= 3);
        assert_eq!(a, b);
    }

    #[test]
    fn write_then_read_420_with_odd_size() {
        // Flat colour: 4:2:0 subsampling is then lossless apart from rounding,
        // including the half-covered blocks on the odd right/bottom edge.
        let f = Frame::from_rgb8(5, 3, vec![(200, 30, 90); 15]);
        let mut buf = Vec::new();
        let mut w = Y4mWriter::new(&mut buf, Y4mHeader::new(5, 3, (25, 1), Chroma::C420));
        w.write_frame(&f).unwrap();
        drop(w);
        // 15 luma + 2 * (3x2) chroma bytes.
        assert_eq!(Y4mHeader::new(5, 3, (25, 1), Chroma::C420).frame_len(), 27);

        let mut r = Y4mReader::new(&buf[..]).unwrap();
        let back = r.read_rgb().unwrap().unwrap();
        assert!(max_diff(&back, &f) <= 3);
    }

    #[test]
    fn mono_writes_luma_only_and_reads_back_grey() {
        let f = Frame::from_data(2, 2, 1, FrameData::U8(vec![0, 64, 128, 255]));
        let mut buf = Vec::new();
        let mut w = Y4mWriter::new(&mut buf, Y4mHeader::new(2, 2, (25, 1), Chroma::Mono));
        w.write_frame(&f).unwrap();
        drop(w);
        let header_len = buf.iter().position(|&b| b == b'\n').unwrap() + 1;
        assert_eq!(buf.len(), header_len + "FRAME\n".len() + 4);

        let mut r = Y4mReader::new(&buf[..]).unwrap();
        let back = r.read_rgb().unwrap().unwrap();
        assert_eq!(back.channels, 3);
        for (px, want) in back.to_rgb8().into_iter().zip([0u8, 64, 128, 255]) {
            assert_eq!(px.0, px.1);
            assert_eq!(px.1, px.2);
            assert!(px.0.abs_diff(want) <= 2);
        }
    }

    #[test]
    fn parses_ffmpeg_style_header() {
        let src = b"YUV4MPEG2 W2 H2 F30000:1001 Ip A1:1 C420mpeg2 XYSCSS=420MPEG2 XCOLORRANGE=FULL\nFRAME\n\x10\x10\x10\x10\x80\x80";
        let mut r = Y4mReader::new(&src[..]).unwrap();
        let h = r.header().clone();
        assert_eq!((h.width, h.height), (2, 2));
        assert_eq!((h.fps_num, h.fps_den), (30000, 1001));
        assert_eq!(h.chroma, Chroma::C420);
        assert!(h.full_range);
        assert_eq!(r.read_rgb().unwrap().unwrap().to_rgb8()[0], (16, 16, 16));
    }

    #[test]
    fn rejects_unsupported_and_malformed_streams() {
        let bad = |s: &[u8]| Y4mReader::new(s).err().unwrap();
        assert!(matches!(
            bad(b"YUV4MPEG2 W2 H2 C422\n"),
            Y4mError::Unsupported(_)
        ));
        assert!(matches!(
            bad(b"YUV4MPEG2 W2 H2 C420p10\n"),
            Y4mError::Unsupported(_)
        ));
        assert!(matches!(bad(b"RIFF....\n"), Y4mError::Format(_)));
        assert!(matches!(bad(b"YUV4MPEG2 H2\n"), Y4mError::Format(_)));
        assert!(matches!(bad(b""), Y4mError::Format(_)));
        assert!(matches!(bad(&[b'x'; 10_000]), Y4mError::Format(_)));
        // An unknown tag starting with a multi-byte character is ignored like
        // any other unknown tag (and, crucially, does not panic).
        assert!(Y4mReader::new("YUV4MPEG2 W2 H2 \u{e9}x\n".as_bytes()).is_ok());
        assert!(matches!(
            bad(b"YUV4MPEG2 W4000000000 H4000000000\n"),
            Y4mError::Format(_)
        ));
    }

    #[test]
    fn truncated_frame_is_an_error_not_eos() {
        let src = b"YUV4MPEG2 W2 H2 Cmono\nFRAME\n\x10\x10";
        let mut r = Y4mReader::new(&src[..]).unwrap();
        assert!(matches!(r.read_rgb(), Err(Y4mError::Format(_))));
    }

    #[test]
    fn writer_rejects_wrong_shapes() {
        let mut buf = Vec::new();
        let mut w = Y4mWriter::new(&mut buf, Y4mHeader::new(2, 2, (25, 1), Chroma::C420));
        let wrong_size = Frame::from_rgb8(1, 1, vec![(0, 0, 0)]);
        assert!(w.write_frame(&wrong_size).is_err());
        let f32_frame = Frame::zeros(2, 2, 3, crate::data::DType::F32);
        assert!(w.write_frame(&f32_frame).is_err());
        let two_ch = Frame::zeros(2, 2, 2, crate::data::DType::U8);
        assert!(w.write_frame(&two_ch).is_err());
        drop(w);
        // Nothing valid was written, so not even a header.
        assert!(buf.is_empty());
    }
}
