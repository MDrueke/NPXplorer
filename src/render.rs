use crate::data::DisplayRow;

pub const C_ZERO: [u8; 3] = [0x17, 0x1b, 0x21]; // #171b21 (was #262930 grey)

/// Single representative accent color per colormap. This is the one place to edit
/// when tuning colors that need to track the active colormap outside the heatmap
/// itself — e.g. the nav bar's view/buffer markers and the spike projection overlay.
pub fn colormap_accent(cmap: &crate::app::ColorMapChoice) -> [u8; 3] {
    match cmap {
        crate::app::ColorMapChoice::YellowMagenta => [250, 234, 130],
        crate::app::ColorMapChoice::RedBlue => [204, 103, 230],
        crate::app::ColorMapChoice::OrangeBlue => [242, 171, 126],
        crate::app::ColorMapChoice::IceFire => [166, 217, 237],
        crate::app::ColorMapChoice::Vanimo => [202, 237, 166],
        crate::app::ColorMapChoice::GreyScale => [255, 255, 255],
        crate::app::ColorMapChoice::CoolWarm => [120, 150, 240],
    }
}

/// Color of the atlas region borders, region labels on the heatmap and region names in
/// the Atlas Registration table — one per colormap, picked to stand out against the map
/// and against the other heatmap lines (white shank boundaries, grey channel gaps,
/// white/orange selections). This is the one place to edit to change them.
pub fn atlas_color(cmap: &crate::app::ColorMapChoice) -> [u8; 3] {
    match cmap {
        crate::app::ColorMapChoice::YellowMagenta => [0, 220, 255],
        crate::app::ColorMapChoice::RedBlue => [0, 235, 170],
        crate::app::ColorMapChoice::OrangeBlue => [140, 255, 90],
        crate::app::ColorMapChoice::IceFire => [150, 255, 60],
        crate::app::ColorMapChoice::Vanimo => [0, 210, 255],
        crate::app::ColorMapChoice::GreyScale => [0, 220, 255],
        crate::app::ColorMapChoice::CoolWarm => [0, 0, 0],
    }
}

/// Background of the boxes behind the region labels on the heatmap and behind the
/// region names in the Atlas Registration table (drawn semi-transparent) — must
/// contrast with `atlas_color`: dark for the light region colors, white for Cool-Warm's
/// black ones.
pub fn atlas_label_bg(cmap: &crate::app::ColorMapChoice) -> [u8; 3] {
    match cmap {
        crate::app::ColorMapChoice::CoolWarm => [255, 255, 255],
        _ => C_ZERO,
    }
}

/// Color of the markers drawn directly on top of the heatmap (shank boundaries,
/// "shank N" labels, first selected channel, scale bar): white on the maps whose
/// zero is the dark background, near-black on Cool-Warm, whose zero is light grey.
pub fn heatmap_fg(cmap: &crate::app::ColorMapChoice) -> [u8; 3] {
    match cmap {
        crate::app::ColorMapChoice::CoolWarm => [25, 25, 25],
        _ => [255, 255, 255],
    }
}

/// Opacity (0-255) of the nav bar's "preprocessed buffer extent" shading.
/// 26 ≈ 10% — the one place to tune this.
pub const BUFFER_EXTENT_ALPHA: u8 = 10;

/// Opacity (0-255) of the nav bar's "currently displayed view" marker.
pub const VIEW_MARKER_ALPHA: u8 = 180;

#[inline]
fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t) as u8,
    ]
}

#[inline]
fn interpolate_stops(t: f32, stops: &[[u8; 3]]) -> [u8; 3] {
    let n = stops.len() - 1;
    let scaled_t = t * n as f32;
    let idx = scaled_t.floor() as usize;
    if idx >= n {
        return stops[n];
    }
    let local_t = scaled_t - idx as f32;
    lerp_rgb(stops[idx], stops[idx + 1], local_t)
}

#[inline]
pub fn voltage_to_rgba(v: f32, vmax: f32, cmap: &crate::app::ColorMapChoice) -> [u8; 4] {
    let t = (v / vmax).clamp(-1.0, 1.0); // -1..1

    let [r, g, b] = match cmap {
        crate::app::ColorMapChoice::YellowMagenta => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        C_ZERO,
                        [0x44, 0x2a, 0x4a],
                        [0x5d, 0x33, 0x66],
                        [0x7b, 0x26, 0x8c],
                        [0x93, 0x04, 0xb0],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x33, 0x31, 0x26],
                        [0x3d, 0x39, 0x1f],
                        [0x52, 0x4b, 0x1e],
                        [0x75, 0x6a, 0x1e],
                        [0xa3, 0x90, 0x12],
                        [0xff, 0xdf, 0x12],
                    ],
                )
            }
        }
        crate::app::ColorMapChoice::RedBlue => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        C_ZERO,
                        [0x2e, 0x30, 0x42],
                        [0x25, 0x2c, 0x61],
                        [0x24, 0x34, 0xb3],
                        [0x2c, 0x43, 0xf5],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x40, 0x2c, 0x2b],
                        [0x61, 0x2f, 0x2c],
                        [0x9e, 0x32, 0x2b],
                        [0xf5, 0x43, 0x36],
                    ],
                )
            }
        }
        crate::app::ColorMapChoice::OrangeBlue => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        C_ZERO,
                        [0x29, 0x3b, 0x54],
                        [0x31, 0x54, 0x85],
                        [0x2d, 0x6f, 0xc4],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x4a, 0x29, 0x22],
                        [0x75, 0x36, 0x28],
                        [0xd1, 0x42, 0x21],
                    ],
                )
            }
        }
        crate::app::ColorMapChoice::IceFire => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        C_ZERO,
                        [0x39, 0x32, 0x47],
                        [0x39, 0x29, 0x5c],
                        [0x46, 0x27, 0x8a],
                        [0x20, 0x5f, 0x9e],
                        [0x71, 0xb5, 0xbd],
                        [0x93, 0xcf, 0xc9],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x40, 0x31, 0x30],
                        [0x4d, 0x2f, 0x2d],
                        [0x5e, 0x29, 0x25],
                        [0x8a, 0x24, 0x1d],
                        [0xba, 0x4f, 0x22],
                        [0xd9, 0xa2, 0x73],
                    ],
                )
            }
        }
        crate::app::ColorMapChoice::Vanimo => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        C_ZERO,
                        [0x2e, 0x36, 0x27],
                        [0x3c, 0x52, 0x27],
                        [0x56, 0x8a, 0x22],
                        [0x8d, 0xed, 0x2d],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x43, 0x31, 0x47],
                        [0x66, 0x35, 0x73],
                        [0xb9, 0x4e, 0xd4],
                    ],
                )
            }
        }
        crate::app::ColorMapChoice::GreyScale => {
            if t >= 0.0 {
                interpolate_stops(t, &[C_ZERO, [0x00, 0x00, 0x00]])
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x30, 0x30, 0x30],
                        [0x50, 0x50, 0x50],
                        [0x60, 0x60, 0x60],
                        [0xd0, 0xd0, 0xd0],
                    ],
                )
            }
        }
        // matplotlib's "coolwarm" (Moreland) colors: blue - light grey - red. Oriented
        // like the other maps, negative (spikes) in the warm color; unlike them, zero is
        // light grey rather than the background
        crate::app::ColorMapChoice::CoolWarm => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        [0xdd, 0xdd, 0xdd],
                        [0xb8, 0xd0, 0xf9],
                        [0x8d, 0xb0, 0xfe],
                        [0x62, 0x82, 0xea],
                        [0x3b, 0x4c, 0xc0],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        [0xdd, 0xdd, 0xdd],
                        [0xf5, 0xc4, 0xad],
                        [0xf4, 0x9a, 0x7b],
                        [0xde, 0x60, 0x4d],
                        [0xb4, 0x04, 0x26],
                    ],
                )
            }
        }
    };
    [r, g, b, 255]
}

// ---------------------------------------------------------------------------
// Row -> pixel-row mapping shared by both heatmaps
// ---------------------------------------------------------------------------

/// Display row shown at pixel row `py` (last row at the top, first at the bottom).
#[inline]
fn disp_idx_of_pixel_row(py: usize, n_rows: usize, pixel_h: usize) -> usize {
    n_rows
        .saturating_sub(1)
        .saturating_sub((py * n_rows) / pixel_h)
        .min(n_rows - 1)
}

/// Fill `out` (pixel_w × pixel_h RGBA) from a per-display-row painter. Each display
/// row is rendered once and copied to every pixel row that shows it, instead of
/// re-rendering it per pixel row; when there are more rows than pixels, only the
/// rows actually shown are rendered.
fn paint_rows(
    out: &mut [u8],
    n_rows: usize,
    pixel_w: usize,
    pixel_h: usize,
    fill_row: impl Fn(usize, &mut [u8]) + Sync,
) {
    use rayon::prelude::*;
    let row_bytes = pixel_w * 4;
    if n_rows >= pixel_h {
        // at most one pixel row per display row: paint directly
        out.par_chunks_mut(row_bytes).enumerate().for_each(|(py, row)| {
            fill_row(disp_idx_of_pixel_row(py, n_rows, pixel_h), row);
        });
        return;
    }
    let mut row_px = vec![0u8; n_rows * row_bytes];
    row_px.par_chunks_mut(row_bytes).enumerate().for_each(|(disp_idx, row)| fill_row(disp_idx, row));
    out.par_chunks_mut(row_bytes).enumerate().for_each(|(py, row)| {
        let d = disp_idx_of_pixel_row(py, n_rows, pixel_h);
        row.copy_from_slice(&row_px[d * row_bytes..(d + 1) * row_bytes]);
    });
}

#[inline]
fn fill_solid(row: &mut [u8], rgb: [u8; 3]) {
    for px in row.chunks_exact_mut(4) {
        px[0] = rgb[0];
        px[1] = rgb[1];
        px[2] = rgb[2];
        px[3] = 255;
    }
}

/// Background with a dotted grey line: the marker of a gap between electrode rows.
#[inline]
fn fill_gap(row: &mut [u8]) {
    for (px_idx, px) in row.chunks_exact_mut(4).enumerate() {
        let (r, g, b) = if (px_idx / 4) % 2 == 0 {
            (0x60, 0x60, 0x60)
        } else {
            (C_ZERO[0], C_ZERO[1], C_ZERO[2])
        };
        px[0] = r; px[1] = g; px[2] = b; px[3] = 255;
    }
}

// ---------------------------------------------------------------------------
// Pooling: several samples per pixel column -> one value
// ---------------------------------------------------------------------------

/// Reduce the samples of one pixel column to one value: their mean, or with `peak`
/// the sample of largest magnitude (sign kept), so short events such as spikes keep
/// their amplitude however many samples share a column.
#[inline]
fn pool(samples: &[f32], peak: bool) -> f32 {
    if peak {
        // min and max in separate branch-free reductions (vectorizable), then the
        // one of larger magnitude
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in samples {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if -lo > hi { lo } else { hi }
    } else {
        samples.iter().sum::<f32>() / samples.len() as f32
    }
}

/// Sample range `[t0, t1)` (within `n` samples) shown by pixel column `px` of `w`.
#[inline]
fn column_range(px: usize, n: usize, w: usize) -> (usize, usize) {
    let t0 = (px * n) / w;
    let t1 = (((px + 1) * n) / w).min(n).max(t0 + 1);
    (t0, t1)
}

/// Value at percentile `pct` (0..=100) of |values|, ignoring NaN; `None` if empty.
fn abs_percentile(values: impl Iterator<Item = f32>, pct: f32) -> Option<f32> {
    let mut v: Vec<f32> = values.filter(|x| !x.is_nan()).map(f32::abs).collect();
    if v.is_empty() {
        return None;
    }
    let k = (((v.len() - 1) as f32) * (pct / 100.0).clamp(0.0, 1.0)).round() as usize;
    let (_, kth, _) = v.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(*kth)
}

/// Colour a pixel row from pooled values (NaN = no data -> background).
fn colour_row(row: &mut [u8], pooled: &[f32], vmax: f32, cmap: &crate::app::ColorMapChoice) {
    for (px, &v) in row.chunks_exact_mut(4).zip(pooled) {
        let rgba = if v.is_nan() {
            [C_ZERO[0], C_ZERO[1], C_ZERO[2], 255]
        } else {
            voltage_to_rgba(v, vmax, cmap)
        };
        px.copy_from_slice(&rgba);
    }
}

/// How a heatmap's colour range is chosen.
#[derive(Clone, Copy, Debug)]
pub enum ColorScale {
    /// fixed ±vmax (µV)
    Fixed(f32),
    /// this percentile (0..=100) of the |values| actually drawn
    ViewPercentile(f32),
}

// ---------------------------------------------------------------------------
// Heatmap renderer
//
// `display_rows` — the full ordered list of rows to render (Data + Gap variants).
//   Data rows carry a `data_idx` into the flat `data` buffer.
//   Gap rows are rendered as a dotted grey line.
// ---------------------------------------------------------------------------

/// Render the main heatmap; returns the colour range (vmax, µV) used.
pub fn build_heatmap_into(
    out: &mut Vec<u8>,
    data: &[f32],
    display_rows: &[DisplayRow],
    data_stride: usize,   // n_samp in the buffer
    buf_first: usize,     // absolute first sample covered by the buffer
    buf_n_samp: usize,    // number of samples covered by the buffer
    view_first: usize,    // absolute first sample of the requested view
    n_view: usize,        // number of samples in the requested view
    pixel_w: usize,
    pixel_h: usize,
    scale: ColorScale,
    peak_pooling: bool,
    cmap: &crate::app::ColorMapChoice,
) -> f32 {
    use rayon::prelude::*;
    let total = pixel_w * pixel_h * 4;
    out.resize(total, 0);
    let fallback_vmax = match scale { ColorScale::Fixed(v) => v, ColorScale::ViewPercentile(_) => 1.0 };

    if pixel_w == 0 || pixel_h == 0 || n_view == 0 || display_rows.is_empty() {
        return fallback_vmax;
    }
    let n_rows = display_rows.len();
    let buf_end = buf_first + buf_n_samp;

    // pool every data row once: pooled[r] holds one value per pixel column
    let pooled: Vec<Option<Vec<f32>>> = display_rows
        .par_iter()
        .map(|row| {
            let DisplayRow::Data { data_idx, .. } = row else { return None };
            let row_base = data_idx * data_stride;
            let mut vals = vec![f32::NAN; pixel_w];
            if row_base + data_stride > data.len() {
                return Some(vals); // row not in this buffer: background
            }
            let ch_data = &data[row_base..row_base + data_stride];
            for (px, v) in vals.iter_mut().enumerate() {
                let (t0, t1) = column_range(px, n_view, pixel_w);
                let (abs_lo, abs_hi) = (view_first + t0, view_first + t1);
                // background wherever the buffer hasn't been preprocessed yet
                if abs_lo >= buf_first && abs_hi <= buf_end {
                    *v = pool(&ch_data[abs_lo - buf_first..abs_hi - buf_first], peak_pooling);
                }
            }
            Some(vals)
        })
        .collect();

    let vmax = match scale {
        ColorScale::Fixed(v) => v,
        ColorScale::ViewPercentile(p) => abs_percentile(pooled.iter().flatten().flatten().copied(), p)
            .unwrap_or(1.0)
            .max(1.0),
    };

    paint_rows(out, n_rows, pixel_w, pixel_h, |disp_idx, row| match &display_rows[disp_idx] {
        DisplayRow::IntraShankGap => fill_gap(row),
        DisplayRow::ShankBoundary => fill_solid(row, heatmap_fg(cmap)),
        DisplayRow::Data { .. } => colour_row(row, pooled[disp_idx].as_deref().unwrap_or(&[]), vmax, cmap),
    });
    vmax
}

/// Render a PSTH result (a dense, already-aligned rows×time matrix) into an RGBA
/// buffer of size `pixel_w × pixel_h`. Rows are drawn last at top / first at
/// bottom, matching the main heatmap; gaps and shank boundaries render identically.
pub fn build_psth_heatmap_into(
    out: &mut Vec<u8>,
    result: &crate::psth::PsthResult,
    pixel_w: usize,
    pixel_h: usize,
    vmax: f32,
    peak_pooling: bool,
    cmap: &crate::app::ColorMapChoice,
) {
    out.resize(pixel_w * pixel_h * 4, 0);
    if pixel_w == 0 || pixel_h == 0 {
        return;
    }

    let display_rows = &result.display_rows;
    let n_rows = display_rows.len();
    let n_win = result.n_win;
    let data = &result.data;
    if n_rows == 0 || n_win == 0 {
        return;
    }

    paint_rows(out, n_rows, pixel_w, pixel_h, |disp_idx, row| match &display_rows[disp_idx] {
        DisplayRow::IntraShankGap => fill_gap(row),
        DisplayRow::ShankBoundary => fill_solid(row, heatmap_fg(cmap)),
        DisplayRow::Data { data_idx, .. } => {
            let ch_data = &data[data_idx * n_win..(data_idx + 1) * n_win];
            let pooled: Vec<f32> = (0..pixel_w)
                .map(|px| {
                    let (t0, t1) = column_range(px, n_win, pixel_w);
                    pool(&ch_data[t0..t1], peak_pooling)
                })
                .collect();
            colour_row(row, &pooled, vmax, cmap);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_pooling_keeps_spike_amplitude() {
        // a -100 µV, 8-sample trough in 214 samples of zero (one 10 s-window column)
        let mut x = vec![0.0f32; 214];
        for v in &mut x[100..108] {
            *v = -100.0;
        }
        x[50] = 20.0;
        assert_eq!(pool(&x, true), -100.0);
        assert!((pool(&x, false) - (-800.0 + 20.0) / 214.0).abs() < 1e-4);
        assert_eq!(column_range(0, 10, 20), (0, 1)); // fewer samples than pixels
        assert_eq!(abs_percentile([1.0, -3.0, f32::NAN, 2.0].into_iter(), 100.0), Some(3.0));
    }
}
