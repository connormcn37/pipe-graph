//! The editor's bundled font.
//!
//! Bevy's built-in default font is `FiraMono-subset`, an ASCII-only subset:
//! any other character — an accented node id (`résumé`), a Greek or Cyrillic
//! parameter value, a non-English file path, even `…` — renders as a
//! missing-glyph box. Node ids, parameter values and paths are user data, so
//! the editor can't assume they are ASCII.
//!
//! Instead it ships DejaVu Sans Mono (`assets/fonts/`, Bitstream Vera licence;
//! see `assets/fonts/README.md`): monospaced like the default — which
//! [`super::layout::fit_label`]'s width estimate relies on — with Latin,
//! Greek, Cyrillic and common symbols. The bytes are compiled into the binary
//! with `include_bytes!`, so the editor needs no asset folder at runtime and
//! works from any working directory.
//!
//! [`setup_editor_font`] registers the font in `Assets<Font>` during
//! `PreStartup` and stores its handle in the [`EditorFont`] resource;
//! everything that spawns editor text builds its [`TextFont`] with
//! [`EditorFont::text_font`] so no label falls back to the default font.

use bevy::prelude::*;

/// The bundled font file (DejaVu Sans Mono 2.37, regular weight).
pub const EDITOR_FONT_BYTES: &[u8] = include_bytes!("../../assets/fonts/DejaVuSansMono.ttf");

/// Handle to the bundled editor font, inserted by [`setup_editor_font`].
///
/// Systems that spawn text take `Res<EditorFont>` and call
/// [`EditorFont::text_font`] rather than `TextFont::from_font_size`, which
/// would silently use Bevy's ASCII-only default.
#[derive(Resource, Debug, Clone)]
pub struct EditorFont(pub Handle<Font>);

impl EditorFont {
    /// A [`TextFont`] using the editor font at `size`.
    pub fn text_font(&self, size: f32) -> TextFont {
        TextFont {
            font: self.0.clone(),
            font_size: size,
            ..default()
        }
    }
}

/// Parse [`EDITOR_FONT_BYTES`] into a [`Font`].
///
/// # Panics
/// If the bundled bytes are not a valid font. They are a fixed file checked in
/// next to the source (and parsed by this module's tests), so this can only
/// fail if the repository itself is broken.
pub fn editor_font() -> Font {
    Font::try_from_bytes(EDITOR_FONT_BYTES.to_vec()).expect("bundled editor font is a valid font")
}

/// Add the bundled font to `Assets<Font>` and insert [`EditorFont`].
///
/// Runs in `PreStartup` (rather than in `Plugin::build`) so it doesn't depend
/// on the order plugins were added in — `Assets<Font>` only exists once Bevy's
/// `TextPlugin` has been built — while still being in place before the
/// `Startup` systems that spawn the status line and inspector panel.
pub fn setup_editor_font(mut commands: Commands, mut fonts: ResMut<Assets<Font>>) {
    commands.insert_resource(EditorFont(fonts.add(editor_font())));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face() -> ttf_parser::Face<'static> {
        ttf_parser::Face::parse(EDITOR_FONT_BYTES, 0).expect("bundled font parses")
    }

    #[test]
    fn bundled_font_loads_as_a_bevy_font() {
        let font = editor_font();
        assert_eq!(font.data.len(), EDITOR_FONT_BYTES.len());
    }

    #[test]
    fn bundled_font_covers_non_ascii_text() {
        let face = face();
        // Latin-1, the ellipsis `fit_label` appends, Cyrillic, Greek, plus a
        // few symbols likely in parameter values and paths.
        for c in ['é', 'ß', 'ñ', '…', 'Ж', 'я', 'λ', 'Ω', '°', '×', '→', '€'] {
            assert!(face.glyph_index(c).is_some(), "no glyph for {c:?}");
        }
    }

    #[test]
    fn bundled_font_is_monospaced_at_the_layout_advance() {
        // `fit_label` estimates a label's width as chars × GLYPH_ADVANCE_EM;
        // that is only exact if every glyph it may draw has that advance.
        let face = face();
        let em = f32::from(face.units_per_em());
        for c in ['i', 'W', 'é', '…', 'Ж', 'λ', '_', '.'] {
            let glyph = face.glyph_index(c).unwrap();
            let advance = f32::from(face.glyph_hor_advance(glyph).unwrap()) / em;
            assert!(
                (advance - super::super::layout::GLYPH_ADVANCE_EM).abs() < 0.005,
                "{c:?} advances {advance} em"
            );
        }
    }
}
