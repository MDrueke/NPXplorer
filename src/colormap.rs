//! Every heatmap colormap and the colors that go with it, in one place.
//!
//! To add a colormap, add one entry to the `colormaps!` list at the bottom: a variant
//! name (stored in the settings file, so don't rename existing ones) and its
//! `ColorMap`. The enum, the Settings dropdown and all lookups follow from that.

/// App background, also the zero color of the dark-background maps (#171b21).
pub const C_ZERO: [u8; 3] = rgb(0x171b21);

const WHITE: [u8; 3] = rgb(0xffffff);
const BLACK: [u8; 3] = rgb(0x000000);

/// `0xRRGGBB` -> `[r, g, b]`, so colors can be written as hex codes.
pub const fn rgb(hex: u32) -> [u8; 3] {
    [(hex >> 16) as u8, (hex >> 8) as u8, hex as u8]
}

/// One colormap: the heatmap gradient plus the overlay / marker colors chosen to read
/// well on top of it.
pub struct ColorMap {
    /// name shown in the Settings dropdown
    pub name: &'static str,

    // --- heatmap gradient: color stops, evenly spaced, linearly interpolated ---
    /// color at 0 µV
    pub zero: [u8; 3],
    /// stops from just above 0 up to +vmax (`zero` itself not repeated)
    pub positive: &'static [[u8; 3]],
    /// stops from just below 0 down to -vmax, i.e. spikes (`zero` not repeated)
    pub negative: &'static [[u8; 3]],

    // --- colors that track the colormap outside the heatmap itself ---
    /// representative accent: nav bar view/buffer markers, TTL overlay, PSTH traces,
    /// firing-rate overlay
    pub accent: [u8; 3],
    /// opacity of the firing-rate overlay; tuned for its many-triangle accumulation
    pub overlay_alpha: u8,
    /// atlas region borders, region labels on the heatmap, region names in the Atlas
    /// Registration table — must stand out against the map and the other heatmap lines
    /// (white shank boundaries, grey channel gaps, white/orange selections)
    pub atlas: [u8; 3],
    /// boxes behind region labels / table names / the zoom notice (drawn
    /// semi-transparent) — must contrast with `atlas` and `heatmap_fg`
    pub label_bg: [u8; 3],
    /// markers drawn directly on the heatmap: shank boundaries, "shank N" labels,
    /// first selected channel, scale bar, zoom notice text
    pub heatmap_fg: [u8; 3],
}

impl ColorMap {
    /// Color for `t` in -1..=1 (value / vmax, already clamped).
    #[inline]
    pub fn color(&self, t: f32) -> [u8; 3] {
        if t >= 0.0 {
            interpolate(self.zero, self.positive, t)
        } else {
            interpolate(self.zero, self.negative, -t)
        }
    }
}

#[inline]
fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t) as u8,
    ]
}

/// Interpolate `t` in 0..=1 over the evenly spaced stops `[zero, rest...]`.
#[inline]
fn interpolate(zero: [u8; 3], rest: &[[u8; 3]], t: f32) -> [u8; 3] {
    let n = rest.len();
    if n == 0 {
        return zero;
    }
    let scaled_t = t * n as f32;
    let idx = scaled_t.floor() as usize;
    if idx >= n {
        return rest[n - 1];
    }
    let stop = |i: usize| if i == 0 { zero } else { rest[i - 1] };
    lerp_rgb(stop(idx), stop(idx + 1), scaled_t - idx as f32)
}

/// Generates `ColorMapChoice` (one variant per entry, in dropdown order), its `ALL`
/// list and `spec()` lookup from the list below.
macro_rules! colormaps {
    ($($(#[$doc:meta])* $variant:ident => $map:expr,)+) => {
        #[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        pub enum ColorMapChoice {
            $($(#[$doc])* $variant,)+
        }

        impl ColorMapChoice {
            /// every colormap, in dropdown order
            pub const ALL: &'static [ColorMapChoice] = &[$(ColorMapChoice::$variant,)+];

            /// This map's gradient and associated colors.
            #[inline]
            pub fn spec(&self) -> &'static ColorMap {
                match self {
                    $(ColorMapChoice::$variant => {
                        static MAP: ColorMap = $map;
                        &MAP
                    })+
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// The colormaps. Negative values (spikes) are drawn in the `negative` stops.
// ---------------------------------------------------------------------------

colormaps! {
    SunFire => ColorMap {
        name: "SunFire",
        zero: C_ZERO,
        positive: &[rgb(0x402626), rgb(0x581f1f), rgb(0xb82424), rgb(0xdf0404)],
        negative: &[rgb(0x3e3528), rgb(0x4d3d25), rgb(0x564422), rgb(0x816124), rgb(0xd19b20), rgb(0xffbd00)],
        accent: [248, 230, 5],
        overlay_alpha: 5,
        atlas: WHITE,
        label_bg: C_ZERO,
        heatmap_fg: WHITE,
    },
    YellowMagenta => ColorMap {
        name: "Yellow-Magenta",
        zero: C_ZERO,
        positive: &[rgb(0x442a4a), rgb(0x5d3366), rgb(0x7b268c), rgb(0x9304b0)],
        negative: &[rgb(0x333126), rgb(0x3d391f), rgb(0x524b1e), rgb(0x756a1e), rgb(0xa39012), rgb(0xffdf12)],
        accent: [250, 234, 130],
        overlay_alpha: 5,
        atlas: WHITE,
        label_bg: C_ZERO,
        heatmap_fg: WHITE,
    },
    RedBlue => ColorMap {
        name: "Red-Blue",
        zero: C_ZERO,
        positive: &[rgb(0x2e3042), rgb(0x252c61), rgb(0x2434b3), rgb(0x2c43f5)],
        negative: &[rgb(0x402c2b), rgb(0x612f2c), rgb(0x9e322b), rgb(0xf54336)],
        accent: [248, 5, 5],
        overlay_alpha: 8,
        atlas: WHITE,
        label_bg: C_ZERO,
        heatmap_fg: WHITE,
    },
    OrangeBlue => ColorMap {
        name: "Orange-Blue",
        zero: C_ZERO,
        positive: &[rgb(0x293b54), rgb(0x315485), rgb(0x2d6fc4)],
        negative: &[rgb(0x4a2922), rgb(0x753628), rgb(0xd14221)],
        accent: [242, 171, 126],
        overlay_alpha: 5,
        atlas: WHITE,
        label_bg: C_ZERO,
        heatmap_fg: WHITE,
    },
    IceFire => ColorMap {
        name: "Ice-Fire",
        zero: C_ZERO,
        positive: &[rgb(0x393247), rgb(0x39295c), rgb(0x46278a), rgb(0x205f9e), rgb(0x71b5bd), rgb(0x93cfc9)],
        negative: &[rgb(0x403130), rgb(0x4d2f2d), rgb(0x5e2925), rgb(0x8a241d), rgb(0xba4f22), rgb(0xd9a273)],
        accent: [57, 5, 248],
        overlay_alpha: 8,
        atlas: WHITE,
        label_bg: C_ZERO,
        heatmap_fg: WHITE,
    },
    Vanimo => ColorMap {
        name: "Vanimo",
        zero: C_ZERO,
        positive: &[rgb(0x2e3627), rgb(0x3c5227), rgb(0x568a22), rgb(0x8ded2d)],
        negative: &[rgb(0x433147), rgb(0x663573), rgb(0xb94ed4)],
        accent: [202, 237, 166],
        overlay_alpha: 5,
        atlas: WHITE,
        label_bg: C_ZERO,
        heatmap_fg: WHITE,
    },
    GreyScale => ColorMap {
        name: "Greyscale",
        zero: C_ZERO,
        positive: &[BLACK],
        negative: &[rgb(0x303030), rgb(0x505050), rgb(0x606060), rgb(0xd0d0d0)],
        accent: [255, 255, 255],
        overlay_alpha: 2,
        atlas: WHITE,
        label_bg: C_ZERO,
        heatmap_fg: WHITE,
    },
    /// matplotlib's "coolwarm" (Moreland) colors: blue - light grey - red. Oriented
    /// like the other maps, negative (spikes) in the warm color; unlike them, zero is
    /// light grey rather than the background, so the overlay colors are inverted
    CoolWarm => ColorMap {
        name: "Cool-Warm",
        zero: rgb(0xdddddd),
        positive: &[rgb(0xb8d0f9), rgb(0x8db0fe), rgb(0x6282ea), rgb(0x3b4cc0)],
        negative: &[rgb(0xf5c4ad), rgb(0xf49a7b), rgb(0xde604d), rgb(0xb40426)],
        accent: [120, 150, 240],
        overlay_alpha: 10,
        atlas: BLACK,
        label_bg: WHITE,
        heatmap_fg: rgb(0x191919),
    },
}

// ---------------------------------------------------------------------------
// Fixed one-sided ramps (not selectable in Settings)
// ---------------------------------------------------------------------------

/// Brewer/matplotlib 11-stop diverging "Spectral" ramp, low (quiet) to high (loud);
/// colors the power spectrum heatmap.
const SPECTRAL: &[[u8; 3]] = &[
    rgb(0x5e4fa2), rgb(0x3288bd), rgb(0x66c2a5), rgb(0xabdda4), rgb(0xe6f598), rgb(0xffffbf),
    rgb(0xfee08b), rgb(0xfdae61), rgb(0xf46d43), rgb(0xd53e4f), rgb(0x9e0142),
];

/// Spectral color for `t` in 0..=1.
#[inline]
pub fn spectrum_color(t: f32) -> [u8; 3] {
    interpolate(SPECTRAL[0], &SPECTRAL[1..], t.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolate_hits_stops_and_ends() {
        let m = ColorMapChoice::RedBlue.spec();
        assert_eq!(m.color(0.0), m.zero);
        assert_eq!(m.color(1.0), *m.positive.last().unwrap());
        assert_eq!(m.color(-1.0), *m.negative.last().unwrap());
        // t = 1/n lands exactly on the first stop after zero
        let n = m.positive.len() as f32;
        assert_eq!(m.color(1.0 / n), m.positive[0]);
    }

    #[test]
    fn every_map_has_stops() {
        for c in ColorMapChoice::ALL {
            let m = c.spec();
            assert!(!m.positive.is_empty() && !m.negative.is_empty(), "{}", m.name);
        }
    }
}
