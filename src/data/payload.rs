//! Payload: the data-agnostic value that flows along graph edges.
//!
//! The core runtime is not video-specific. A port carries a `Payload`, and a
//! node downcasts to the variant it expects. `Frame` is the image/tensor case;
//! `Scalar` covers control/feedback values (e.g. an auto-exposure gain);
//! `Bytes` is an opaque escape hatch. `Arc` on `Bytes` (and cheap `Clone` on
//! the whole enum) keeps fan-out and taps inexpensive.

use std::sync::Arc;

use crate::data::{AudioBuffer, Frame, Meta};

#[derive(Debug, Clone)]
pub enum Payload {
    Frame(Frame),
    Scalar(f64),
    Bytes(Arc<[u8]>),
    /// Audio is carried but not yet processed; see [`AudioBuffer`].
    Audio(AudioBuffer),
}

/// The type tag of a payload, used to declare and validate port compatibility.
///
/// `Any` is only meaningful on a *port declaration* (a port that accepts any
/// payload); a concrete [`Payload`] never reports `Any` from [`Payload::kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PayloadKind {
    Frame,
    Scalar,
    Bytes,
    Audio,
    Any,
}

impl PayloadKind {
    /// Whether a value of kind `produced` may be delivered to a port declared
    /// to accept `self`. `Any` accepts everything; otherwise kinds must match.
    pub fn accepts(self, produced: PayloadKind) -> bool {
        self == PayloadKind::Any || produced == PayloadKind::Any || self == produced
    }
}

impl Payload {
    pub fn kind(&self) -> PayloadKind {
        match self {
            Payload::Frame(_) => PayloadKind::Frame,
            Payload::Scalar(_) => PayloadKind::Scalar,
            Payload::Bytes(_) => PayloadKind::Bytes,
            Payload::Audio(_) => PayloadKind::Audio,
        }
    }

    /// The metadata of a frame or audio buffer; `None` for payloads that
    /// carry none (scalars, raw bytes).
    pub fn meta(&self) -> Option<&Meta> {
        match self {
            Payload::Frame(f) => Some(f.meta()),
            Payload::Audio(a) => Some(&a.meta),
            Payload::Scalar(_) | Payload::Bytes(_) => None,
        }
    }

    /// Mutable access to [`Payload::meta`].
    pub fn meta_mut(&mut self) -> Option<&mut Meta> {
        match self {
            Payload::Frame(f) => Some(f.meta_mut()),
            Payload::Audio(a) => Some(&mut a.meta),
            Payload::Scalar(_) | Payload::Bytes(_) => None,
        }
    }

    pub fn as_audio(&self) -> Option<&AudioBuffer> {
        match self {
            Payload::Audio(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_frame(&self) -> Option<&Frame> {
        match self {
            Payload::Frame(f) => Some(f),
            _ => None,
        }
    }

    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Payload::Scalar(s) => Some(*s),
            _ => None,
        }
    }

    pub fn into_frame(self) -> Option<Frame> {
        match self {
            Payload::Frame(f) => Some(f),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_reflects_variant() {
        assert_eq!(Payload::Scalar(1.0).kind(), PayloadKind::Scalar);
        let f = Frame::from_rgb8(1, 1, vec![(0, 0, 0)]);
        assert_eq!(Payload::Frame(f).kind(), PayloadKind::Frame);
    }

    #[test]
    fn any_accepts_all_and_kinds_match() {
        assert!(PayloadKind::Any.accepts(PayloadKind::Frame));
        assert!(PayloadKind::Frame.accepts(PayloadKind::Any));
        assert!(PayloadKind::Frame.accepts(PayloadKind::Frame));
        assert!(!PayloadKind::Frame.accepts(PayloadKind::Scalar));
    }
}
