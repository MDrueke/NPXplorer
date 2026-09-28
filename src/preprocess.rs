use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::data::{DisplayRow, GAP_PITCH_FACTOR};

// ---------------------------------------------------------------------------
// SOS biquad — f32 arithmetic
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Sos {
    pub b0: f32, pub b1: f32, pub b2: f32,
    pub a1: f32, pub a2: f32,
}

/// Cascade of second-order sections with scipy-compatible zero-phase filtering:
/// `filtfilt` extends the signal by odd reflection (`padlen` samples) and starts each
/// pass from the steady-state initial conditions for the first sample, so the ends of
/// a segment carry no start-up transient (`scipy.signal.sosfiltfilt` semantics).
#[derive(Clone, Debug)]
pub struct SosFilter {
    pub sos: Vec<Sos>,
    /// steady-state section states for a unit step input (`sosfilt_zi`)
    zi: Vec<[f32; 2]>,
    padlen: usize,
}

impl SosFilter {
    pub fn new(sos: Vec<Sos>) -> Self {
        // per section: state after infinitely many unit-step inputs, scaled by the DC
        // gain of the sections before it (that is the input the section actually sees)
        let mut zi = Vec::with_capacity(sos.len());
        let mut scale = 1.0f64;
        for s in &sos {
            let (b0, b1, b2, a1, a2) = (s.b0 as f64, s.b1 as f64, s.b2 as f64, s.a1 as f64, s.a2 as f64);
            let g = (b0 + b1 + b2) / (1.0 + a1 + a2);
            let z1 = b2 - a2 * g;
            let z0 = b1 - a1 * g + z1;
            zi.push([(z0 * scale) as f32, (z1 * scale) as f32]);
            scale *= g;
        }
        let n_b2_zero = sos.iter().filter(|s| s.b2 == 0.0).count();
        let n_a2_zero = sos.iter().filter(|s| s.a2 == 0.0).count();
        let ntaps = 2 * sos.len() + 1 - n_b2_zero.min(n_a2_zero);
        Self { sos, zi, padlen: 3 * ntaps }
    }

    /// Single forward pass starting from `zi * x0` (transposed direct form II).
    fn filt_with_zi(&self, x: &mut [f32], x0: f32) {
        for (s, z) in self.sos.iter().zip(&self.zi) {
            let (b0, b1, b2, a1, a2) = (s.b0, s.b1, s.b2, s.a1, s.a2);
            let (mut z0, mut z1) = (z[0] * x0, z[1] * x0);
            for v in x.iter_mut() {
                let xi = *v;
                let yi = b0 * xi + z0;
                z0 = b1 * xi - a1 * yi + z1;
                z1 = b2 * xi - a2 * yi;
                *v = yi;
            }
        }
    }

    /// Zero-phase filter `x` in place. `scratch` is reused between calls to avoid
    /// allocating the extended signal each time.
    pub fn filtfilt(&self, x: &mut [f32], scratch: &mut Vec<f32>) {
        let n = x.len();
        if n < 2 {
            return;
        }
        let padlen = self.padlen.min(n - 1);
        scratch.clear();
        scratch.reserve(n + 2 * padlen);
        // odd extension: mirror the signal around its end points
        for i in 0..padlen {
            scratch.push(2.0 * x[0] - x[padlen - i]);
        }
        scratch.extend_from_slice(x);
        for i in 0..padlen {
            scratch.push(2.0 * x[n - 1] - x[n - 2 - i]);
        }
        let ext = scratch.as_mut_slice();
        let x0 = ext[0];
        self.filt_with_zi(ext, x0);
        ext.reverse();
        let y0 = ext[0];
        self.filt_with_zi(ext, y0);
        ext.reverse();
        x.copy_from_slice(&ext[padlen..padlen + n]);
    }
}

// ---------------------------------------------------------------------------
// Butterworth highpass
// ---------------------------------------------------------------------------

pub fn butter_highpass_sos(n: usize, wn: f64) -> Vec<Sos> {
    use std::f64::consts::PI;
    let wd = 2.0 * (PI * wn / 2.0).tan();
    let ni = n as i32;
    let mut sections = Vec::new();

    for k in 0..n {
        let angle = PI * (2 * k as i32 + ni + 1) as f64 / (2.0 * ni as f64);
        let lp_re = angle.cos();
        let lp_im = angle.sin();

        if lp_im.abs() < 1e-10 {
            let pr = wd * lp_re;
            let d = 2.0 - pr;
            sections.push(Sos {
                b0: (2.0 / d) as f32,
                b1: (-2.0 / d) as f32,
                b2: 0.0,
                a1: ((-2.0 - pr) / d) as f32,
                a2: 0.0,
            });
        } else if lp_im > 0.0 {
            let pr = wd * lp_re;
            let pi_v = wd * lp_im;
            let a_c = -2.0 * pr;
            let b_c = pr * pr + pi_v * pi_v;
            let d = 4.0 + 2.0 * a_c + b_c;
            sections.push(Sos {
                b0: ( 4.0 / d) as f32,
                b1: (-8.0 / d) as f32,
                b2: ( 4.0 / d) as f32,
                a1: ((-8.0 + 2.0 * b_c) / d) as f32,
                a2: (( 4.0 - 2.0 * a_c + b_c) / d) as f32,
            });
        }
    }
    sections
}

pub fn butter_highpass(n: usize, wn: f64) -> SosFilter {
    SosFilter::new(butter_highpass_sos(n, wn))
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum SpatialFilter {
    Off,
    GlobalCmr,
    LocalCmr,
    Destripe,
}

#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct PreprocConfig {
    pub dc_removal: bool,
    /// correct each channel's ADC sampling delay (applied while reading, before
    /// depth averaging — see `RawData::read_rows`)
    pub phase_shift: bool,
    pub highpass: bool,
    pub spatial_filter: SpatialFilter,
    pub avg_depths: bool,
    pub sample_rate: f64,
    /// 0-based channel indices excluded from display and from every computation
    /// (CMR/destripe reference, depth averaging). Recording-specific, so it is
    /// never persisted to the saved preferences.
    #[serde(default, skip_serializing)]
    pub removed_channels: std::collections::BTreeSet<usize>,
    #[serde(default)]
    pub channel_order: crate::data::ChannelOrder,
    #[serde(default)]
    pub shank_order: crate::data::ShankOrder,
}

#[derive(Clone)]
pub struct Filters {
    pub hp: SosFilter,
    pub kfilt: SosFilter,
    pub kfilt_lagc: usize,
}

impl Filters {
    pub fn new(cfg: &PreprocConfig) -> Self {
        let fs = cfg.sample_rate;
        Filters {
            hp: butter_highpass(3, 300.0 / fs * 2.0),
            kfilt: butter_highpass(3, 0.01),
            kfilt_lagc: (fs / 10.0).round() as usize,
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level entry point
// Order: depth-averaging (and the ADC delay correction) happen while reading.
// Here: DC offset -> Temporal HP -> Spatial filter, per shank.
// ---------------------------------------------------------------------------

/// Preprocess `data` (`[n_data_rows][n_samp]`) in place.
///
/// Destripe's AGC floor (`epsilon`, one per block of neighbouring rows) normally comes
/// from the data itself. `agc_eps` lets a caller reuse the values of an earlier run —
/// used when extending a buffer, so a chunk that is stitched onto it gets the floor
/// of the buffer it joins instead of its own, and no seam appears at the join.
/// Returns the floors actually used (empty unless the spatial filter is Destripe).
pub fn preprocess(
    data: &mut [f32],
    n_samp: usize,
    cfg: &PreprocConfig,
    filt: &Filters,
    cancel: &AtomicBool,
    display_rows: &[DisplayRow],
    agc_eps: Option<&[f32]>,
) -> Vec<f32> {
    let mut eps_used = Vec::new();
    if display_rows.is_empty() || n_samp == 0 { return eps_used; }

    // Identify contiguous chunks of DisplayRow::Data that share the same shank.
    let mut shank_blocks = Vec::new();
    let mut current_shank = None;
    let mut start_idx = 0;

    // We only care about Data rows for partitioning the data array
    let data_rows: Vec<&DisplayRow> = display_rows.iter()
        .filter(|r| matches!(r, DisplayRow::Data { .. }))
        .collect();

    for (i, row) in data_rows.iter().enumerate() {
        if let DisplayRow::Data { shank, .. } = row {
            if current_shank.is_none() {
                current_shank = Some(*shank);
            } else if Some(*shank) != current_shank {
                shank_blocks.push((start_idx, i));
                start_idx = i;
                current_shank = Some(*shank);
            }
        }
    }
    if start_idx < data_rows.len() {
        shank_blocks.push((start_idx, data_rows.len()));
    }

    // Split data into independent, isolated slices per shank
    // Since `data` is completely ordered exactly as `data_rows`, we can chunk it cleanly.
    let mut current_data_offset = 0;

    for (start_row, end_row) in shank_blocks {
        if cancel.load(Ordering::Relaxed) { return eps_used; }

        let n_shank_rows = end_row - start_row;
        let n_shank_samples = n_shank_rows * n_samp;

        let shank_data = &mut data[current_data_offset .. current_data_offset + n_shank_samples];
        let shank_data_rows = &data_rows[start_row .. end_row];

        // 1. DC Offset Correction
        if cfg.dc_removal {
            apply_dc_removal(shank_data, n_samp);
        }
        if cancel.load(Ordering::Relaxed) { return eps_used; }

        // 2. Highpass Filter
        if cfg.highpass {
            apply_temporal_hp(shank_data, n_samp, &filt.hp);
        }
        if cancel.load(Ordering::Relaxed) { return eps_used; }

        // 3. Spatial Filter
        match cfg.spatial_filter {
            SpatialFilter::Off => {}
            SpatialFilter::GlobalCmr => apply_global_cmr(shank_data, n_shank_rows, n_samp),
            SpatialFilter::LocalCmr => apply_local_cmr(shank_data, n_samp, shank_data_rows),
            SpatialFilter::Destripe => {
                // the spatial highpass runs along physical depth and only across rows
                // that are actually neighbours on the shank — never across a gap
                for block in contiguous_depth_blocks(shank_data_rows) {
                    if cancel.load(Ordering::Relaxed) { return eps_used; }
                    let reuse = agc_eps.and_then(|e| e.get(eps_used.len())).copied();
                    eps_used.push(apply_kfilt(shank_data, n_samp, &block, filt, reuse));
                }
            }
        }

        current_data_offset += n_shank_samples;
    }
    eps_used
}

/// Row indices (into one shank's rows) grouped into runs of physically adjacent rows,
/// each run sorted by depth then x. A run ends where the next row is further away
/// than `GAP_PITCH_FACTOR` × the shank's electrode pitch (the same rule that draws a
/// gap in the display), so a spatial filter never mixes rows across a gap.
pub(crate) fn contiguous_depth_blocks(rows: &[&DisplayRow]) -> Vec<Vec<usize>> {
    let pos = |r: &DisplayRow| match r {
        DisplayRow::Data { y_um, x_um, .. } => (*y_um, *x_um),
        _ => (0.0, 0.0),
    };
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| {
        let (ya, xa) = pos(rows[a]);
        let (yb, xb) = pos(rows[b]);
        ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal)
            .then(xa.partial_cmp(&xb).unwrap_or(std::cmp::Ordering::Equal))
    });
    let pitch = order
        .windows(2)
        .map(|w| pos(rows[w[1]]).0 - pos(rows[w[0]]).0)
        .filter(|&d| d > 0.1)
        .fold(f32::INFINITY, f32::min);
    let pitch = if pitch.is_finite() { pitch } else { 20.0 };

    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let mut prev_y: Option<f32> = None;
    for r in order {
        let y = pos(rows[r]).0;
        match (prev_y, blocks.last_mut()) {
            (Some(py), Some(b)) if y - py <= pitch * GAP_PITCH_FACTOR => b.push(r),
            _ => blocks.push(vec![r]),
        }
        prev_y = Some(y);
    }
    blocks
}

// ---------------------------------------------------------------------------
// DC offset removal
// ---------------------------------------------------------------------------

fn apply_dc_removal(data: &mut [f32], n_samp: usize) {
    data.par_chunks_mut(n_samp).for_each(|row| {
        let mean = row.iter().sum::<f32>() / n_samp as f32;
        row.iter_mut().for_each(|v| *v -= mean);
    });
}

use std::cell::RefCell;

thread_local! {
    static LOCAL_CMR_BUF: RefCell<(Vec<f32>, Vec<f32>)> = RefCell::new((Vec::new(), Vec::new()));
    static SCRATCH: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

// ---------------------------------------------------------------------------
// Local CMR
// ---------------------------------------------------------------------------

fn apply_local_cmr(data: &mut [f32], n_samp: usize, data_rows: &[&DisplayRow]) {
    let n_rows = data_rows.len();
    let mut neighborhoods = vec![Vec::new(); n_rows];

    // Pre-calculate neighborhoods
    for i in 0..n_rows {
        if let DisplayRow::Data { x_um: x1, y_um: y1, .. } = data_rows[i] {
            for j in 0..n_rows {
                if let DisplayRow::Data { x_um: x2, y_um: y2, .. } = data_rows[j] {
                    let dx = x1 - x2;
                    let dy = y1 - y2;
                    let d = (dx*dx + dy*dy).sqrt();
                    if d >= 100.0 && d <= 400.0 {
                        neighborhoods[i].push(j);
                    }
                }
            }
        }
    }

    let data_ptr = SendPtr(data.as_mut_ptr());
    (0..n_samp).into_par_iter().for_each(|t| {
        let dp = data_ptr.0;
        let _ = &data_ptr;
        LOCAL_CMR_BUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.0.resize(n_rows, 0.0);
            for ch in 0..n_rows {
                buf.0[ch] = unsafe { *dp.add(ch * n_samp + t) };
            }

            for (ch, neighbors) in neighborhoods.iter().enumerate() {
                if neighbors.is_empty() { continue; }
                buf.1.resize(neighbors.len(), 0.0);
                for (i, &n) in neighbors.iter().enumerate() {
                    buf.1[i] = buf.0[n];
                }
                let half = buf.1.len() / 2;
                buf.1.select_nth_unstable_by(half, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = if buf.1.len() % 2 == 0 {
                    let upper = buf.1[half];
                    buf.1[..half].select_nth_unstable_by(half - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    (upper + buf.1[half - 1]) / 2.0
                } else {
                    buf.1[half]
                };
                unsafe { *dp.add(ch * n_samp + t) = buf.0[ch] - median; }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Temporal HP
// ---------------------------------------------------------------------------

fn apply_temporal_hp(data: &mut [f32], n_samp: usize, filt: &SosFilter) {
    data.par_chunks_mut(n_samp).for_each(|ch| {
        SCRATCH.with(|s| filt.filtfilt(ch, &mut s.borrow_mut()));
    });
}

// ---------------------------------------------------------------------------
// Global CMR
// ---------------------------------------------------------------------------

thread_local! {
    static CMR_COL: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

fn apply_global_cmr(data: &mut [f32], n_rows: usize, n_samp: usize) {
    if n_rows == 0 {
        return;
    }
    let data_ptr = SendPtr(data.as_mut_ptr());
    let half = n_rows / 2;

    (0..n_samp).into_par_iter().for_each(|t| {
        let dp = data_ptr.0;
        let _ = &data_ptr;
        CMR_COL.with(|cell| {
            let mut col = cell.borrow_mut();
            col.resize(n_rows, 0.0);
            for ch in 0..n_rows {
                col[ch] = unsafe { *dp.add(ch * n_samp + t) };
            }
            col.select_nth_unstable_by(half, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let median = if n_rows % 2 == 0 {
                let upper = col[half];
                col[..half].select_nth_unstable_by(half - 1, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                (upper + col[half - 1]) / 2.0
            } else {
                col[half]
            };
            for ch in 0..n_rows {
                unsafe { *dp.add(ch * n_samp + t) -= median; }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// AGC — sliding mean absolute value via prefix sums (O(n) per channel)
// epsilon = std(data) * 0.003, matching IBL Python reference
// ---------------------------------------------------------------------------

/// AGC floor of a block of rows: 0.3 % of the standard deviation of all its samples.
fn agc_epsilon(data: &[f32], n_samp: usize, rows: &[usize]) -> f32 {
    let n_total = (rows.len() * n_samp) as f64;
    let sum: f64 = rows
        .par_iter()
        .map(|&r| data[r * n_samp..(r + 1) * n_samp].iter().map(|&v| v as f64).sum::<f64>())
        .sum();
    let mean = sum / n_total;
    let var: f64 = rows
        .par_iter()
        .map(|&r| {
            data[r * n_samp..(r + 1) * n_samp]
                .iter()
                .map(|&v| (v as f64 - mean) * (v as f64 - mean))
                .sum::<f64>()
        })
        .sum::<f64>()
        / n_total;
    ((var.sqrt() * 0.003) as f32).max(1e-8f32)
}

/// Per-row AGC gain for the rows listed in `rows` (indices into `data`'s rows);
/// returns `[rows.len()][n_samp]`. `epsilon` is the gain's floor.
fn compute_agc_gain(data: &[f32], n_samp: usize, rows: &[usize], win: usize, epsilon: f32) -> Vec<f32> {
    let half = win / 2;
    let mut gain = vec![epsilon; rows.len() * n_samp];

    gain.par_chunks_mut(n_samp).enumerate().for_each(|(i, g)| {
        let src = &data[rows[i] * n_samp..(rows[i] + 1) * n_samp];
        // prefix sum of |x|
        let mut prefix = vec![0.0f64; n_samp + 1];
        for t in 0..n_samp {
            prefix[t + 1] = prefix[t] + src[t].abs() as f64;
        }
        for t in 0..n_samp {
            let lo = t.saturating_sub(half);
            let hi = (t + half + 1).min(n_samp);
            let count = hi - lo;
            let mean_abs = (prefix[hi] - prefix[lo]) / count as f64;
            g[t] = (mean_abs as f32).max(epsilon);
        }
    });
    gain
}

// ---------------------------------------------------------------------------
// Spatial kfilt (IBL destripe)
// ---------------------------------------------------------------------------

thread_local! {
    static KFILT_COL: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

/// Raw mutable pointer wrapper for cross-thread access in rayon.
/// SAFETY: this is only safe when each parallel iteration accesses
/// disjoint memory. All uses iterate over time samples `t`, where
/// each `t` touches `ch * n_samp + t` — guaranteed disjoint for distinct `t`.
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Spatial highpass along `rows` (indices into `data`'s rows, ordered by depth),
/// with AGC before and its inverse after, like IBL's `kfilt`. `epsilon` is the AGC
/// floor to use (`None` = compute it from this data); returns the floor used.
fn apply_kfilt(data: &mut [f32], n_samp: usize, rows: &[usize], filt: &Filters, epsilon: Option<f32>) -> f32 {
    let n_rows = rows.len();
    if n_rows == 0 {
        return epsilon.unwrap_or(1e-8);
    }
    let epsilon = epsilon.unwrap_or_else(|| agc_epsilon(data, n_samp, rows));
    let pad = 60usize.min(n_rows);
    let n_padded = n_rows + 2 * pad;
    let gain = compute_agc_gain(data, n_samp, rows, filt.kfilt_lagc, epsilon);

    // divide by gain
    let data_ptr = SendPtr(data.as_mut_ptr());
    gain.par_chunks(n_samp).enumerate().for_each(|(i, g)| {
        let dp = data_ptr.0;
        let _ = &data_ptr;
        let row = unsafe { std::slice::from_raw_parts_mut(dp.add(rows[i] * n_samp), n_samp) };
        for (v, &gv) in row.iter_mut().zip(g.iter()) {
            *v /= gv;
        }
    });

    let gain_ptr = SendPtr(gain.as_ptr() as *mut f32);
    let sos = &filt.kfilt;

    let chunk_size = 512;
    (0..n_samp).into_par_iter().step_by(chunk_size).for_each(|t_start| {
        let t_end = (t_start + chunk_size).min(n_samp);
        let n_t = t_end - t_start;
        let dp = data_ptr.0;
        let gp = gain_ptr.0;
        let _ = (&data_ptr, &gain_ptr);

        KFILT_COL.with(|cell| {
            let mut buf = cell.borrow_mut();
            // We use buf to store a 2D block: [n_padded][n_t] in row-major
            buf.resize(n_padded * n_t, 0.0);

            // Read row by row (in depth order) for contiguous memory access
            for (i, &r) in rows.iter().enumerate() {
                let src_row = unsafe { std::slice::from_raw_parts(dp.add(r * n_samp + t_start), n_t) };
                let dst_row = &mut buf[(pad + i) * n_t .. (pad + i + 1) * n_t];
                dst_row.copy_from_slice(src_row);
            }

            // mirror-pad at top
            for p in 0..pad {
                let src_ch = (pad - 1 - p).min(n_rows.saturating_sub(1));
                let src_idx = (pad + src_ch) * n_t;
                let dst_idx = p * n_t;
                for i in 0..n_t {
                    buf[dst_idx + i] = buf[src_idx + i];
                }
            }

            // mirror-pad at bottom
            for p in 0..pad {
                let src_ch = n_rows.saturating_sub(1 + p);
                let src_idx = (pad + src_ch) * n_t;
                let dst_idx = (pad + n_rows + p) * n_t;
                for i in 0..n_t {
                    buf[dst_idx + i] = buf[src_idx + i];
                }
            }

            // Filter each column
            let mut col = vec![0.0f32; n_padded];
            SCRATCH.with(|s| {
                let mut scratch = s.borrow_mut();
                for i in 0..n_t {
                    for r in 0..n_padded {
                        col[r] = buf[r * n_t + i];
                    }
                    sos.filtfilt(&mut col, &mut scratch);
                    for r in 0..n_padded {
                        buf[r * n_t + i] = col[r];
                    }
                }
            });

            // Write back row by row, restoring the gain
            for (i, &r) in rows.iter().enumerate() {
                let dst_row = unsafe { std::slice::from_raw_parts_mut(dp.add(r * n_samp + t_start), n_t) };
                let src_row = &buf[(pad + i) * n_t .. (pad + i + 1) * n_t];
                let g_row = unsafe { std::slice::from_raw_parts(gp.add(i * n_samp + t_start), n_t) };
                for k in 0..n_t {
                    dst_row[k] = src_row[k] * g_row[k];
                }
            }
        });
    });
    epsilon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtfilt_has_no_edge_transient_on_a_constant() {
        // a highpass with scipy-style edge handling turns a constant into (nearly) zero
        // everywhere, including the first samples where a zero-state filter rings
        let f = butter_highpass(3, 0.02);
        let mut x = vec![100.0f32; 500];
        let mut scratch = Vec::new();
        f.filtfilt(&mut x, &mut scratch);
        assert!(x.iter().all(|v| v.abs() < 1e-2), "max |y| = {}", x.iter().fold(0.0f32, |m, v| m.max(v.abs())));
        // a pure high-frequency tone passes with unit gain, away from the ends
        let mut y: Vec<f32> = (0..500).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        f.filtfilt(&mut y, &mut scratch);
        assert!((y[250].abs() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn depth_blocks_split_at_gaps_and_ignore_display_order() {
        let row = |y: f32| DisplayRow::Data { data_idx: 0, channels: vec![], first_ch: 0, x_um: 0.0, y_um: y, shank: 0 };
        // rows given out of depth order, with a 100 µm gap between 40 and 140
        let rows = [row(140.0), row(0.0), row(40.0), row(20.0), row(160.0)];
        let refs: Vec<&DisplayRow> = rows.iter().collect();
        let blocks = contiguous_depth_blocks(&refs);
        assert_eq!(blocks, vec![vec![1, 3, 2], vec![0, 4]]);
    }
}
