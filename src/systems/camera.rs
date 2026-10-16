//! Pure 2D camera math for editor navigation: pan, zoom about a point, and
//! fit-to-view.
//!
//! The editor's view is a `Camera2d` with Bevy's default orthographic
//! projection (`ScalingMode::WindowSize`, viewport origin at the center). Its
//! whole state is two numbers: the world point at the middle of the viewport
//! (the camera's translation) and the projection `scale` — world units per
//! logical screen pixel, so `scale < 1` is zoomed in and `scale > 1` zoomed
//! out. Every function here maps that `(center, scale)` pair plus plain input
//! values to a new pair, with no ECS or device access, so the fiddly
//! invariants ("the point under the cursor stays under the cursor") are pinned
//! by ordinary unit tests. [`super::render`] reads the mouse, calls these, and
//! writes the result back to the camera.
//!
//! Screen coordinates follow Bevy's window convention: logical pixels, origin
//! at the top-left, +Y *down*. World space is +Y *up*.
//!
//! [`FitView`] is the one piece of ECS state: a request flag raised by the
//! keyboard router ([`super::route_keys`], on `F`) and consumed by the
//! renderer, so the key goes through the same routing as every other shortcut
//! and typing an `f` into a parameter value never re-frames the view.

use bevy::math::{Rect, Vec2};
use bevy::prelude::Resource;

/// Smallest projection scale (most zoomed in): 10× magnification.
pub const MIN_SCALE: f32 = 0.1;
/// Largest projection scale (most zoomed out): 0.1× magnification.
pub const MAX_SCALE: f32 = 10.0;
/// Magnification change per wheel "line" (one notch on most mice).
pub const ZOOM_STEP: f32 = 1.1;
/// Pixel-unit wheel deltas (touchpads, some mice) counted as one line.
pub const PIXELS_PER_LINE: f32 = 40.0;
/// Screen-space margin, in logical pixels, kept around the nodes on each
/// side when fitting them to the view.
pub const FIT_MARGIN: f32 = 60.0;
/// Fit never magnifies beyond this scale. Framing one or two small boxes at
/// 7× is rarely what anyone wants; 1:1 keeps text crisp and leaves room for
/// the previews drawn below the selected node.
pub const FIT_MIN_SCALE: f32 = 1.0;

/// Request to frame every node in the view. Raised by
/// [`super::route_keys`] when `F` is pressed outside a value edit; the render
/// plugin performs the fit and clears it. Headless apps (no camera) just
/// leave it raised, which is harmless.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FitView {
    pub requested: bool,
}

/// Keep a scale within [`MIN_SCALE`, `MAX_SCALE`]. A non-finite or
/// non-positive scale (which would invert or collapse the view) maps to 1.
pub fn clamp_scale(scale: f32) -> f32 {
    if scale.is_finite() && scale > 0.0 {
        scale.clamp(MIN_SCALE, MAX_SCALE)
    } else {
        1.0
    }
}

/// The world point shown at `screen` (top-left origin, +Y down) for a camera
/// centered on `center` at `scale`, with a viewport of `viewport` logical
/// pixels. Same mapping as `Camera::viewport_to_world_2d` for the default 2D
/// projection, but computed from the camera's *current* transform rather
/// than last frame's propagated `GlobalTransform`, so pan and zoom never act
/// on a stale view.
pub fn screen_to_world(center: Vec2, scale: f32, viewport: Vec2, screen: Vec2) -> Vec2 {
    let from_middle = screen - viewport / 2.0;
    center + Vec2::new(from_middle.x, -from_middle.y) * scale
}

/// The camera center that puts world point `world` under screen point
/// `screen` at `scale` — the inverse of [`screen_to_world`] solved for the
/// center.
///
/// Panning uses this directly: remember the world point grabbed when the
/// drag started and re-solve every frame, so the grabbed point stays exactly
/// under the cursor with no accumulated drift.
pub fn center_keeping(world: Vec2, screen: Vec2, scale: f32, viewport: Vec2) -> Vec2 {
    let from_middle = screen - viewport / 2.0;
    world - Vec2::new(from_middle.x, -from_middle.y) * scale
}

/// Zoom by `factor` (magnification; `> 1` zooms in) about the world point
/// `anchor`, which stays at the same screen position. The resulting scale is
/// clamped, and the center is solved from the *clamped* scale, so zooming
/// against a limit leaves the view where it is instead of sliding it.
pub fn zoom_about(center: Vec2, scale: f32, anchor: Vec2, factor: f32) -> (Vec2, f32) {
    let scale = clamp_scale(scale);
    let new_scale = if factor.is_finite() && factor > 0.0 {
        clamp_scale(scale / factor)
    } else {
        scale
    };
    // `anchor`'s offset from the center, in screen pixels, is
    // (anchor - center) / scale; keep that offset fixed at the new scale.
    let new_center = anchor + (center - anchor) * (new_scale / scale);
    (new_center, new_scale)
}

/// Magnification factor for a wheel delta of `lines` notches (positive =
/// away from the user = zoom in, as in most editors).
pub fn wheel_zoom_factor(lines: f32) -> f32 {
    ZOOM_STEP.powf(lines)
}

/// Bounding rectangle of node boxes given as `(center, size)`, or `None` for
/// no boxes.
pub fn boxes_bounds(boxes: impl IntoIterator<Item = (Vec2, Vec2)>) -> Option<Rect> {
    boxes
        .into_iter()
        .map(|(center, size)| Rect::from_center_size(center, size))
        .reduce(|a, b| a.union(b))
}

/// The `(center, scale)` that frames `bounds` in a viewport of `viewport`
/// logical pixels, leaving [`FIT_MARGIN`] pixels clear on every side.
///
/// The scale is whichever axis needs more room, never below
/// [`FIT_MIN_SCALE`] (fit does not magnify) and within the zoom clamp.
/// Returns `None` for a viewport too small to hold the margins, where no
/// sensible framing exists (e.g. a minimized window).
pub fn fit_rect(bounds: Rect, viewport: Vec2) -> Option<(Vec2, f32)> {
    let usable = viewport - Vec2::splat(2.0 * FIT_MARGIN);
    if usable.x <= 0.0 || usable.y <= 0.0 {
        return None;
    }
    let needed = bounds.size() / usable;
    let scale = clamp_scale(needed.x.max(needed.y).max(FIT_MIN_SCALE));
    Some((bounds.center(), scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    const VIEW: Vec2 = Vec2::new(1280.0, 800.0);

    fn close(a: Vec2, b: Vec2) -> bool {
        (a - b).length() < 1e-3
    }

    #[test]
    fn screen_middle_is_the_camera_center_and_y_flips() {
        let c = Vec2::new(100.0, -50.0);
        assert!(close(screen_to_world(c, 2.0, VIEW, VIEW / 2.0), c));
        // 10 px right and 10 px down on screen = +x, -y in world, times scale.
        let p = screen_to_world(c, 2.0, VIEW, VIEW / 2.0 + Vec2::new(10.0, 10.0));
        assert!(close(p, c + Vec2::new(20.0, -20.0)));
    }

    #[test]
    fn center_keeping_inverts_screen_to_world() {
        let world = Vec2::new(-300.0, 42.0);
        let screen = Vec2::new(17.0, 640.0);
        for scale in [0.1, 0.7, 1.0, 3.5] {
            let c = center_keeping(world, screen, scale, VIEW);
            assert!(close(screen_to_world(c, scale, VIEW, screen), world));
        }
    }

    #[test]
    fn zoom_keeps_the_anchor_under_the_same_screen_point() {
        let (center, scale) = (Vec2::new(220.0, -180.0), 1.0);
        let cursor = Vec2::new(900.0, 150.0);
        let anchor = screen_to_world(center, scale, VIEW, cursor);
        for factor in [1.1, 2.0, 0.5, 1.0 / 1.1] {
            let (c2, s2) = zoom_about(center, scale, anchor, factor);
            assert!((s2 - scale / factor).abs() < 1e-5);
            assert!(close(screen_to_world(c2, s2, VIEW, cursor), anchor));
        }
    }

    #[test]
    fn zoom_is_clamped_and_a_clamped_zoom_does_not_slide() {
        let center = Vec2::new(5.0, 5.0);
        let anchor = Vec2::new(500.0, 0.0);
        let (_, s) = zoom_about(center, 1.0, anchor, 1e6);
        assert_eq!(s, MIN_SCALE);
        let (_, s) = zoom_about(center, 1.0, anchor, 1e-6);
        assert_eq!(s, MAX_SCALE);
        // Already at the limit: zooming further changes nothing.
        let (c, s) = zoom_about(center, MIN_SCALE, anchor, 2.0);
        assert_eq!(s, MIN_SCALE);
        assert!(close(c, center));
        // Garbage factors are ignored.
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(zoom_about(center, 2.0, anchor, bad), (center, 2.0));
        }
    }

    #[test]
    fn clamp_scale_rejects_degenerate_values() {
        assert_eq!(clamp_scale(0.0), 1.0);
        assert_eq!(clamp_scale(-3.0), 1.0);
        assert_eq!(clamp_scale(f32::NAN), 1.0);
        assert_eq!(clamp_scale(0.01), MIN_SCALE);
        assert_eq!(clamp_scale(100.0), MAX_SCALE);
        assert_eq!(clamp_scale(2.5), 2.5);
    }

    #[test]
    fn wheel_up_zooms_in_and_opposite_notches_cancel() {
        assert!(wheel_zoom_factor(1.0) > 1.0);
        assert!(wheel_zoom_factor(-1.0) < 1.0);
        assert!((wheel_zoom_factor(3.0) * wheel_zoom_factor(-3.0) - 1.0).abs() < 1e-5);
        assert_eq!(wheel_zoom_factor(0.0), 1.0);
    }

    #[test]
    fn bounds_cover_every_box() {
        assert_eq!(boxes_bounds([]), None);
        let r = boxes_bounds([
            (Vec2::ZERO, Vec2::new(10.0, 20.0)),
            (Vec2::new(100.0, -50.0), Vec2::new(20.0, 10.0)),
        ])
        .unwrap();
        assert_eq!(r.min, Vec2::new(-5.0, -55.0));
        assert_eq!(r.max, Vec2::new(110.0, 10.0));
    }

    #[test]
    fn fit_frames_large_content_with_margin_on_the_tighter_axis() {
        // 2320 x 680 world: width is the constraint ((1280 - 120) px usable).
        let bounds = Rect::new(-1000.0, -400.0, 1320.0, 280.0);
        let (center, scale) = fit_rect(bounds, VIEW).unwrap();
        assert!(close(center, Vec2::new(160.0, -60.0)));
        assert!((scale - 2.0).abs() < 1e-5);
        // Every corner lands inside the margin-inset viewport.
        let usable = Rect::new(
            FIT_MARGIN - 1e-3,
            FIT_MARGIN - 1e-3,
            VIEW.x - FIT_MARGIN + 1e-3,
            VIEW.y - FIT_MARGIN + 1e-3,
        );
        for corner in [bounds.min, bounds.max] {
            let screen = (corner - center) / scale * Vec2::new(1.0, -1.0) + VIEW / 2.0;
            assert!(usable.contains(screen), "{corner} -> {screen}");
        }
    }

    #[test]
    fn fit_does_not_magnify_small_content_and_rejects_tiny_viewports() {
        let small = Rect::from_center_size(Vec2::new(7.0, 9.0), Vec2::splat(50.0));
        assert_eq!(
            fit_rect(small, VIEW),
            Some((Vec2::new(7.0, 9.0), FIT_MIN_SCALE))
        );
        assert_eq!(fit_rect(small, Vec2::new(100.0, 800.0)), None);
        // Huge content is clamped to the most zoomed-out scale.
        let huge = Rect::from_center_size(Vec2::ZERO, Vec2::splat(1e7));
        assert_eq!(fit_rect(huge, VIEW).unwrap().1, MAX_SCALE);
    }
}
