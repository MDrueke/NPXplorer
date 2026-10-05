use crate::colormap::ColorMapChoice;
use crate::data::DisplayRow;

pub use crate::colormap::C_ZERO;

/// Opacity (0-255) of the nav bar's "preprocessed buffer extent" shading.
/// 26 ≈ 10% — the one place to tune this.
pub const BUFFER_EXTENT_ALPHA: u8 = 10;

/// Opacity (0-255) of the nav bar's "currently displayed view" marker.
pub const VIEW_MARKER_ALPHA: u8 = 180;

#[inline]
pub fn voltage_to_rgba(v: f32, vmax: f32, cmap: &ColorMapChoice) -> [u8; 4] {
    let [r, g, b] = cmap.spec().color((v / vmax).clamp(-1.0, 1.0));
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
        out.par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(py, row)| {
                fill_row(disp_idx_of_pixel_row(py, n_rows, pixel_h), row);
            });
        return;
    }
    let mut row_px = vec![0u8; n_rows * row_bytes];
    row_px
        .par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(disp_idx, row)| fill_row(disp_idx, row));
    out.par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(py, row)| {
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
        px[0] = r;
        px[1] = g;
        px[2] = b;
        px[3] = 255;
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
        if -lo > hi {
            lo
        } else {
            hi
        }
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
    let (_, kth, _) = v.select_nth_unstable_by(k, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(*kth)
}

/// Colour a pixel row from pooled values (NaN = no data -> background).
fn colour_row(row: &mut [u8], pooled: &[f32], vmax: f32, cmap: &ColorMapChoice) {
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
    data_stride: usize, // n_samp in the buffer
    buf_first: usize,   // absolute first sample covered by the buffer
    buf_n_samp: usize,  // number of samples covered by the buffer
    view_first: usize,  // absolute first sample of the requested view
    n_view: usize,      // number of samples in the requested view
    pixel_w: usize,
    pixel_h: usize,
    scale: ColorScale,
    peak_pooling: bool,
    cmap: &ColorMapChoice,
) -> f32 {
    use rayon::prelude::*;
    let total = pixel_w * pixel_h * 4;
    out.resize(total, 0);
    let fallback_vmax = match scale {
        ColorScale::Fixed(v) => v,
        ColorScale::ViewPercentile(_) => 1.0,
    };

    if pixel_w == 0 || pixel_h == 0 || n_view == 0 || display_rows.is_empty() {
        return fallback_vmax;
    }
    let n_rows = display_rows.len();
    let buf_end = buf_first + buf_n_samp;

    // pool every data row once: pooled[r] holds one value per pixel column
    let pooled: Vec<Option<Vec<f32>>> = display_rows
        .par_iter()
        .map(|row| {
            let DisplayRow::Data { data_idx, .. } = row else {
                return None;
            };
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
                    *v = pool(
                        &ch_data[abs_lo - buf_first..abs_hi - buf_first],
                        peak_pooling,
                    );
                }
            }
            Some(vals)
        })
        .collect();

    let vmax = match scale {
        ColorScale::Fixed(v) => v,
        ColorScale::ViewPercentile(p) => {
            abs_percentile(pooled.iter().flatten().flatten().copied(), p)
                .unwrap_or(1.0)
                .max(1.0)
        }
    };

    paint_rows(
        out,
        n_rows,
        pixel_w,
        pixel_h,
        |disp_idx, row| match &display_rows[disp_idx] {
            DisplayRow::IntraShankGap => fill_gap(row),
            DisplayRow::ShankBoundary => fill_solid(row, cmap.spec().heatmap_fg),
            DisplayRow::Data { .. } => {
                colour_row(row, pooled[disp_idx].as_deref().unwrap_or(&[]), vmax, cmap)
            }
        },
    );
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
    cmap: &ColorMapChoice,
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

    paint_rows(
        out,
        n_rows,
        pixel_w,
        pixel_h,
        |disp_idx, row| match &display_rows[disp_idx] {
            DisplayRow::IntraShankGap => fill_gap(row),
            DisplayRow::ShankBoundary => fill_solid(row, cmap.spec().heatmap_fg),
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
        },
    );
}

// ---------------------------------------------------------------------------
// Power spectrum heatmap
// ---------------------------------------------------------------------------

pub use crate::colormap::spectrum_color;

/// Dynamic range (dB below the peak) mapped into the colour scale, dB scaling only.
const SPECTRUM_DB_RANGE: f32 = 60.0;

#[inline]
fn spectrum_scaled(v: f32, scaling: crate::spectrum::SpectrumScaling) -> f32 {
    match scaling {
        crate::spectrum::SpectrumScaling::Linear => v,
        crate::spectrum::SpectrumScaling::Db => 10.0 * v.max(1e-20).log10(),
    }
}

/// Resolves a requested display frequency range (Hz) against the data's actual
/// bins (first non-DC bin .. last bin): `None` shows the full band; `Some((lo, hi))`
/// is clamped to it. Shared by the heatmap builder and its axis tick labels so both
/// agree on exactly what's displayed.
pub fn spectrum_freq_bounds(freqs: &[f32], range: Option<(f32, f32)>) -> (f32, f32) {
    let n = freqs.len();
    let data_lo = freqs.get(1).copied().unwrap_or(0.0).max(1e-6);
    let data_hi = freqs.get(n.wrapping_sub(1)).copied().unwrap_or(data_lo * 2.0).max(data_lo * 1.0001);
    let Some((a, b)) = range else {
        return (data_lo, data_hi);
    };
    // order-independent; data_lo < data_hi, so these clamps can't panic
    let lo = a.min(b).clamp(data_lo, data_hi);
    let hi = a.max(b).clamp(data_lo, data_hi);
    if hi > lo * 1.0001 {
        return (lo, hi);
    }
    // zero-width request (e.g. both ends at Nyquist): the narrowest band at that end
    let hi = (lo * 1.0001).min(data_hi);
    (hi / 1.0001, hi)
}

/// Frequencies (Hz) to label on the spectrum panel's axis, ascending. The full band
/// gets fixed anchors (those that fall inside it, if at least 3 do); otherwise up to 4
/// round values spread evenly along the log axis.
pub fn spectrum_ticks(f_lo: f32, f_hi: f32, restricted: bool) -> Vec<f32> {
    let inside = |f: f32| f >= f_lo * 0.999 && f <= f_hi * 1.001;
    if !restricted {
        let anchors: Vec<f32> = [50.0, 250.0, 2000.0, 15000.0].into_iter().filter(|&f| inside(f)).collect();
        // e.g. an LFP band (Nyquist ~1.25 kHz) holds only 50 and 250: too sparse
        if anchors.len() >= 3 {
            return anchors;
        }
    }
    // round candidates per decade: 1-2-5 first, every multiple if that's too sparse
    let candidates = |mults: &[f32]| -> Vec<f32> {
        let mut out = Vec::new();
        let mut decade = 10f32.powi(f_lo.max(1.0).log10().floor() as i32);
        while decade <= f_hi * 1.001 {
            out.extend(mults.iter().map(|m| m * decade).filter(|&f| inside(f)));
            decade *= 10.0;
        }
        out
    };
    let coarse = candidates(&[1.0, 2.0, 5.0]);
    let fine = candidates(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let pick = if coarse.len() >= 3 {
        coarse
    } else if fine.len() >= 2 {
        fine
    } else {
        // band too narrow for any of those: round linear steps
        let raw = ((f_hi - f_lo) / 3.0).max(1e-6);
        let mag = 10f32.powf(raw.log10().floor());
        let m = [1.0, 2.0, 5.0, 10.0].into_iter().find(|&m| m * mag >= raw).unwrap_or(10.0);
        let step = (m * mag).max(1.0);
        let mut out = Vec::new();
        let mut f = (f_lo / step).ceil() * step;
        while f <= f_hi * 1.001 && out.len() < 4 {
            out.push(f);
            f += step;
        }
        return out;
    };
    if pick.len() <= 4 {
        return pick;
    }
    // the candidate nearest (in log distance) to each of 4 evenly log-spaced targets
    let (log_lo, log_hi) = (f_lo.ln(), f_hi.ln());
    let mut out: Vec<f32> = Vec::new();
    for i in 0..4 {
        let target = log_lo + (log_hi - log_lo) * i as f32 / 3.0;
        let best = pick
            .iter()
            .copied()
            .min_by(|a, b| (a.ln() - target).abs().total_cmp(&(b.ln() - target).abs()))
            .unwrap();
        if out.last() != Some(&best) {
            out.push(best);
        }
    }
    out
}

/// "50", "250", "2k", "15k": thousands get a k suffix when they're whole.
pub fn spectrum_tick_label(f: f32) -> String {
    let r = f.round() as i64;
    if r >= 1000 && r % 1000 == 0 {
        format!("{}k", r / 1000)
    } else {
        r.to_string()
    }
}

/// Render a per-channel power spectrum as an opaque heatmap: one row per channel
/// (same row layout/order as the main heatmap), x = frequency on a log scale,
/// colour = power. `freq_range` restricts the displayed band (see
/// `spectrum_freq_bounds`); `None` shows the full band up to Nyquist.
///
/// `power[r]` is the full-bandwidth PSD of the `r`-th *data* row of the full
/// (unzoomed) display-row list the spectrum was computed from. `shown_rows` may be a
/// zoomed sub-slice of that same list; `row_offset` is how many data rows precede
/// `shown_rows[0]` in the full list, so indexing into `power` still lines up.
pub fn build_spectrum_heatmap_into(
    out: &mut Vec<u8>,
    power: &[Vec<f32>],
    freqs: &[f32],
    shown_rows: &[DisplayRow],
    row_offset: usize,
    pixel_w: usize,
    pixel_h: usize,
    scaling: crate::spectrum::SpectrumScaling,
    normalization: crate::spectrum::SpectrumNormalization,
    freq_range: Option<(f32, f32)>,
) {
    let total = pixel_w * pixel_h * 4;
    out.resize(total, 0);
    let n_rows = shown_rows.len();
    if pixel_w == 0 || pixel_h == 0 || n_rows == 0 || freqs.len() < 2 {
        return;
    }
    let n_bins = freqs.len();
    let df = (freqs[1] - freqs[0]).max(1e-9);
    let (f_min, f_max) = spectrum_freq_bounds(freqs, freq_range);
    let log_min = f_min.ln();
    let log_span = f_max.ln() - log_min;
    let bin_lo = ((f_min / df).round() as usize).clamp(1, n_bins - 1);
    let bin_hi = ((f_max / df).round() as usize).clamp(bin_lo, n_bins - 1);

    // frequency bin shown by each pixel column (same for every row)
    let col_bin: Vec<usize> = (0..pixel_w)
        .map(|px| {
            let frac = px as f32 / (pixel_w.max(2) - 1) as f32;
            let f = (log_min + log_span * frac).exp();
            ((f / df).round() as usize).clamp(1, n_bins - 1)
        })
        .collect();

    // linear-power maximum of a row's bins, restricted to the displayed band
    let row_max = |p: &[f32]| -> f32 {
        p.get(bin_lo..=bin_hi)
            .unwrap_or(&[])
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0f32, f32::max)
            .max(1e-12)
    };

    // position of each shown row among shown_rows' Data rows (for indexing into `power`)
    let data_positions: Vec<Option<usize>> = {
        let mut k = 0usize;
        shown_rows
            .iter()
            .map(|r| match r {
                DisplayRow::Data { .. } => {
                    let idx = row_offset + k;
                    k += 1;
                    Some(idx)
                }
                _ => None,
            })
            .collect()
    };

    let global_max = if matches!(normalization, crate::spectrum::SpectrumNormalization::Global) {
        data_positions
            .iter()
            .filter_map(|&i| i.and_then(|i| power.get(i)))
            .map(|p| row_max(p))
            .fold(1e-12f32, f32::max)
    } else {
        1.0
    };

    paint_rows(out, n_rows, pixel_w, pixel_h, |disp_idx, row| {
        match &shown_rows[disp_idx] {
            DisplayRow::IntraShankGap => fill_gap(row),
            DisplayRow::ShankBoundary => fill_solid(row, [255, 255, 255]),
            DisplayRow::Data { .. } => {
                let Some(p) = data_positions[disp_idx].and_then(|i| power.get(i)) else {
                    fill_solid(row, C_ZERO);
                    return;
                };
                let rmax = match normalization {
                    crate::spectrum::SpectrumNormalization::PerChannel => row_max(p),
                    crate::spectrum::SpectrumNormalization::Global => global_max,
                };
                let max_scaled = spectrum_scaled(rmax, scaling);
                for (px, &bin) in col_bin.iter().enumerate() {
                    let v = spectrum_scaled(p[bin], scaling);
                    let t = match scaling {
                        crate::spectrum::SpectrumScaling::Linear => (v / max_scaled).clamp(0.0, 1.0),
                        crate::spectrum::SpectrumScaling::Db => {
                            ((v - (max_scaled - SPECTRUM_DB_RANGE)) / SPECTRUM_DB_RANGE).clamp(0.0, 1.0)
                        }
                    };
                    let [r, g, b] = spectrum_color(t);
                    let o = px * 4;
                    row[o] = r;
                    row[o + 1] = g;
                    row[o + 2] = b;
                    row[o + 3] = 255;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectrum_tick_choice() {
        // full AP band: fixed anchors
        assert_eq!(spectrum_ticks(29.3, 15000.0, false), vec![50.0, 250.0, 2000.0, 15000.0]);
        // full LFP band: only the anchors inside it would be too few, so round values
        let lfp = spectrum_ticks(2.4, 1250.0, false);
        assert!(lfp.len() >= 3 && lfp.iter().all(|&f| (2.4..=1250.0).contains(&f)));
        // restricted to 10..100
        assert_eq!(spectrum_ticks(10.0, 100.0, true), vec![10.0, 20.0, 50.0, 100.0]);
        // narrow restricted band still gets round, in-range ticks
        let narrow = spectrum_ticks(300.0, 340.0, true);
        assert!(narrow.len() >= 2 && narrow.iter().all(|&f| (300.0..=340.0).contains(&f) && f.fract() == 0.0));
        assert_eq!(spectrum_tick_label(2000.0), "2k");
        // band bounds never panic and always give lo < hi, whatever the request
        let freqs: Vec<f32> = (0..=512).map(|k| k as f32 * 30000.0 / 1024.0).collect();
        for req in [(15000.0, 15000.0), (20000.0, 30000.0), (500.0, 100.0), (0.0, 0.0), (29.3, 29.3)] {
            let (lo, hi) = spectrum_freq_bounds(&freqs, Some(req));
            assert!(lo < hi && lo >= freqs[1] && hi <= 15000.0, "{req:?} -> {lo} {hi}");
        }
        assert_eq!(spectrum_freq_bounds(&freqs, Some((500.0, 100.0))), (100.0, 500.0));
        assert_eq!(spectrum_tick_label(250.0), "250");
        assert_eq!(spectrum_tick_label(1250.0), "1250");
    }

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
        assert_eq!(
            abs_percentile([1.0, -3.0, f32::NAN, 2.0].into_iter(), 100.0),
            Some(3.0)
        );
    }
}
