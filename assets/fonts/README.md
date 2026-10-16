# Bundled fonts

| File | Font | Version | Licence |
|---|---|---|---|
| `DejaVuSansMono.ttf` | DejaVu Sans Mono (regular) | 2.37 | Bitstream Vera Fonts licence, DejaVu changes public domain; Arev glyphs under the Arev licence — full text in `DejaVuSansMono-LICENSE.txt` |

Source: <https://dejavu-fonts.github.io/> (licence text from
<https://github.com/dejavu-fonts/dejavu-fonts/blob/version_2_37/LICENSE>).
The file is the 2.37 release as packaged by Debian (`fonts-dejavu-mono`,
unmodified; SHA-256 `c805f943…8d05`).

The editor (`--features bevy`) embeds it with `include_bytes!` in
`src/systems/font.rs` because Bevy's default font is an ASCII-only subset.
It was picked for being monospaced (0.6 em advance, which `fit_label` in
`src/systems/layout.rs` relies on) with Latin, Greek, Cyrillic and common
symbol coverage, under a permissive licence, at ~340 KB.
