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

    /// `filtfilt` of every column of `x` (`[n_rows][n_cols]`, row-major), along the
    /// rows. Same arithmetic per column, in the same order (bit-identical results), but
    /// all columns advance together, so the inner loops run over contiguous memory and
    /// vectorize.
    pub fn filtfilt_columns(&self, x: &mut [f32], n_rows: usize, n_cols: usize, scratch: &mut Vec<f32>) {
        let n = n_rows;
        if n < 2 || n_cols == 0 {
            return;
        }
        debug_assert_eq!(x.len(), n_rows * n_cols);
        let padlen = self.padlen.min(n - 1);
        let n_ext = n + 2 * padlen;
        scratch.clear();
        scratch.resize(n_ext * n_cols + 2 * n_cols, 0.0);
        let (ext, z) = scratch.split_at_mut(n_ext * n_cols);
        let (z0, z1) = z.split_at_mut(n_cols);
        let row = |r: usize| r * n_cols..(r + 1) * n_cols;
        // odd extension at both ends, as in `filtfilt`
        for i in 0..padlen {
            let (a, b) = (&x[row(0)], &x[row(padlen - i)]);
            for ((e, &a), &b) in ext[row(i)].iter_mut().zip(a).zip(b) {
                *e = 2.0 * a - b;
            }
        }
        ext[padlen * n_cols..(padlen + n) * n_cols].copy_from_slice(x);
        for i in 0..padlen {
            let (a, b) = (&x[row(n - 1)], &x[row(n - 2 - i)]);
            for ((e, &a), &b) in ext[row(padlen + n + i)].iter_mut().zip(a).zip(b) {
                *e = 2.0 * a - b;
            }
        }
        // forward pass from the first row, then backward from the last
        for backward in [false, true] {
            for (s, zi) in self.sos.iter().zip(&self.zi) {
                let (b0, b1, b2, a1, a2) = (s.b0, s.b1, s.b2, s.a1, s.a2);
                let start = if backward { n_ext - 1 } else { 0 };
                for c in 0..n_cols {
                    let x0 = ext[start * n_cols + c];
                    z0[c] = zi[0] * x0;
                    z1[c] = zi[1] * x0;
                }
                for k in 0..n_ext {
                    let r = if backward { n_ext - 1 - k } else { k };
                    for ((v, z0), z1) in ext[row(r)].iter_mut().zip(z0.iter_mut()).zip(z1.iter_mut()) {
                        let xi = *v;
                        let yi = b0 * xi + *z0;
                        *z0 = b1 * xi - a1 * yi + *z1;
                        *z1 = b2 * xi - a2 * yi;
                        *v = yi;
                    }
                }
            }
        }
        x.copy_from_slice(&ext[padlen * n_cols..(padlen + n) * n_cols]);
    }
}

// ---------------------------------------------------------------------------
// Butterworth highpass
// ---------------------------------------------------------------------------

pub fn butter_highpass_sos(n: usize, wn: f64) -> Vec<Sos> {
    butter_sos_f64(n, wn, false)
        .into_iter()
        .map(|[b0, b1, b2, a1, a2]| Sos { b0: b0 as f32, b1: b1 as f32, b2: b2 as f32, a1: a1 as f32, a2: a2 as f32 })
        .collect()
}

/// Butterworth highpass or lowpass of order `n`, cutoff `wn` (fraction of Nyquist), as
/// `[b0, b1, b2, a1, a2]` sections (bilinear transform with prewarping).
fn butter_sos_f64(n: usize, wn: f64, lowpass: bool) -> Vec<[f64; 5]> {
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
            let (b0, b1) = if lowpass { (-pr / d, -pr / d) } else { (2.0 / d, -2.0 / d) };
            sections.push([b0, b1, 0.0, (-2.0 - pr) / d, 0.0]);
        } else if lp_im > 0.0 {
            let pr = wd * lp_re;
            let pi_v = wd * lp_im;
            let a_c = -2.0 * pr;
            let b_c = pr * pr + pi_v * pi_v;
            let d = 4.0 + 2.0 * a_c + b_c;
            let (b0, b1, b2) = if lowpass { (b_c / d, 2.0 * b_c / d, b_c / d) } else { (4.0 / d, -8.0 / d, 4.0 / d) };
            sections.push([b0, b1, b2, (-8.0 + 2.0 * b_c) / d, (4.0 - 2.0 * a_c + b_c) / d]);
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
    /// current source density along depth (`CsdParams`)
    Csd,
}

#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CsdParams {
    /// Gaussian smoothing along depth before the second derivative, with σ (µm)
    pub smoothing: bool,
    pub sigma_um: f32,
    /// zero-phase Butterworth bandpass before the CSD; replaces the 300 Hz highpass
    pub bandpass: bool,
    pub low_hz: f64,
    pub high_hz: f64,
}

impl Default for CsdParams {
    fn default() -> Self {
        CsdParams { smoothing: true, sigma_um: 40.0, bandpass: true, low_hz: 5.0, high_hz: 150.0 }
    }
}

/// CSD output is −∂²φ/∂z² · (CSD_REF_UM² + σ²): µV-like values whatever the smoothing,
/// so they keep the resolution of the cache's i16 steps. `csd_to_mv_per_mm2` converts.
const CSD_REF_UM: f64 = 20.0;

/// Factor from preprocessed values to display units: mV/mm² for CSD, 1 (µV) otherwise.
pub fn csd_to_mv_per_mm2(cfg: &PreprocConfig) -> f32 {
    if cfg.spatial_filter != SpatialFilter::Csd {
        return 1.0;
    }
    let sigma = if cfg.csd.smoothing { cfg.csd.sigma_um as f64 } else { 0.0 };
    // µV/µm² → mV/mm²: × 1e6 / 1e3
    (1000.0 / (CSD_REF_UM * CSD_REF_UM + sigma * sigma)) as f32
}

/// Order of each half (highpass, lowpass) of the CSD bandpass.
const CSD_BANDPASS_ORDER: usize = 3;

/// Seconds the CSD bandpass needs to settle at the edge of a chunk (about 6 time
/// constants of its slowest pole, at the low cutoff); 0 when it is off.
pub fn bandpass_settle_s(cfg: &PreprocConfig) -> f64 {
    if cfg.spatial_filter == SpatialFilter::Csd && cfg.csd.bandpass {
        2.0 / cfg.csd.low_hz.max(0.01)
    } else {
        0.0
    }
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
    /// (CMR/destripe reference, depth averaging). Recording-specific: saved with the
    /// recording's settings (settings.rs), never in the preferences.
    #[serde(default, skip_serializing)]
    pub removed_channels: std::collections::BTreeSet<usize>,
    #[serde(default)]
    pub channel_order: crate::data::ChannelOrder,
    #[serde(default)]
    pub shank_order: crate::data::ShankOrder,
    /// notch filters; recording-specific (saved with the recording's settings), so
    /// never persisted to the saved preferences
    #[serde(default, skip_serializing)]
    pub notches: Vec<crate::notch::Notch>,
    #[serde(default, skip_serializing)]
    pub notch_enabled: bool,
    #[serde(default)]
    pub csd: CsdParams,
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
// Here: DC offset -> Notch -> Temporal HP -> Spatial filter, per shank.
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

        // 1b. Notch filters
        if cfg.notch_enabled && !cfg.notches.is_empty() {
            crate::notch::apply_notches(shank_data, n_samp, &cfg.notches, cfg.sample_rate);
        }
        if cancel.load(Ordering::Relaxed) { return eps_used; }

        // 2. Highpass Filter (or the CSD bandpass, which replaces it)
        if cfg.highpass {
            apply_temporal_hp(shank_data, n_samp, &filt.hp);
        }
        if cfg.spatial_filter == SpatialFilter::Csd && cfg.csd.bandpass {
            apply_bandpass(shank_data, n_samp, &cfg.csd, cfg.sample_rate);
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
            SpatialFilter::Csd => {
                // one depth profile per run of neighbouring rows, never across a gap
                for block in contiguous_depth_blocks(shank_data_rows) {
                    let sigma = if cfg.csd.smoothing { cfg.csd.sigma_um } else { 0.0 };
                    apply_csd(shank_data, n_samp, &block, shank_data_rows, sigma);
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
        ya.total_cmp(&yb)
            .then(xa.total_cmp(&xb))
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

use crate::rows::{select_rows_mut, time_blocks_mut};

/// Time samples per parallel work item in the spatial filters, which combine all rows
/// at each sample: small enough that a block of every row stays in cache.
const SPATIAL_BLOCK_T: usize = 512;

thread_local! {
    /// filtfilt work buffer, reused across calls (it is as long as a row)
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------
// Local CMR
// ---------------------------------------------------------------------------

/// Local CMR reference: the median of the channels in this distance range (µm) from
/// each channel, an annulus that leaves out the channel's own spikes (as in IBL's and
/// Kilosort's local referencing).
const LOCAL_CMR_INNER_UM: f32 = 100.0;
const LOCAL_CMR_OUTER_UM: f32 = 400.0;
/// Time samples per parallel work item of local CMR. Smaller than `SPATIAL_BLOCK_T`:
/// each sample costs one median per row, so a second of data in 512-sample blocks
/// gives too few items to keep every thread busy until the end.
const LOCAL_CMR_BLOCK_T: usize = 64;

fn apply_local_cmr(data: &mut [f32], n_samp: usize, data_rows: &[&DisplayRow]) {
    let n_rows = data_rows.len();
    let position = |r: &DisplayRow| match r {
        DisplayRow::Data { x_um, y_um, .. } => Some((*x_um, *y_um)),
        _ => None,
    };
    // per row, the rows in the annulus around it
    let neighborhoods: Vec<Vec<usize>> = data_rows
        .iter()
        .map(|a| {
            let Some((x1, y1)) = position(a) else { return Vec::new() };
            data_rows
                .iter()
                .enumerate()
                .filter_map(|(j, b)| {
                    let (x2, y2) = position(b)?;
                    let d = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)).sqrt();
                    (LOCAL_CMR_INNER_UM..=LOCAL_CMR_OUTER_UM).contains(&d).then_some(j)
                })
                .collect()
        })
        .collect();

    if n_rows == 0 || n_samp == 0 {
        return;
    }
    assert_eq!(data.len(), n_rows * n_samp, "data does not match the rows");
    let blocks = time_blocks_mut(data.chunks_exact_mut(n_samp), LOCAL_CMR_BLOCK_T);
    blocks.into_par_iter().for_each_init(
        || (Vec::with_capacity(n_rows), Vec::new()),
        |(col, ring), mut rows| {
            for t in 0..rows[0].len() {
                // every row's sample before any is changed: the references use the input
                col.clear();
                col.extend(rows.iter().map(|r| r[t]));
                for (ch, neighbors) in neighborhoods.iter().enumerate() {
                    if neighbors.is_empty() {
                        continue;
                    }
                    ring.clear();
                    ring.extend(neighbors.iter().map(|&n| col[n]));
                    let median = crate::stats::median(ring).expect("neighbourhood is not empty");
                    rows[ch][t] = col[ch] - median;
                }
            }
        },
    );
}

// ---------------------------------------------------------------------------
// CSD
// ---------------------------------------------------------------------------

fn apply_bandpass(data: &mut [f32], n_samp: usize, p: &CsdParams, fs: f64) {
    let nyq = fs / 2.0;
    let mut sos = Vec::new();
    if p.low_hz > 0.0 && p.low_hz < nyq {
        sos.extend(butter_sos_f64(CSD_BANDPASS_ORDER, p.low_hz / nyq, false));
    }
    if p.high_hz > p.low_hz && p.high_hz < nyq {
        sos.extend(butter_sos_f64(CSD_BANDPASS_ORDER, p.high_hz / nyq, true));
    }
    crate::notch::filtfilt_rows_f64(data, n_samp, &sos);
}

/// Current source density of one run of neighbouring rows (`rows`, indices into
/// `data_rows`, sorted by depth): Gaussian smoothing along depth (σ µm), then the
/// second derivative with the rows' actual spacing (removed channels leave uneven
/// gaps), the end rows mirrored (Vaknin et al. 1988). Output −∂²φ/∂z² · (CSD_REF_UM² +
/// σ²); sinks are negative.
fn apply_csd(data: &mut [f32], n_samp: usize, rows: &[usize], data_rows: &[&DisplayRow], sigma_um: f32) {
    let n = rows.len();
    let y: Vec<f64> = rows
        .iter()
        .map(|&r| match data_rows[r] {
            DisplayRow::Data { y_um, .. } => *y_um as f64,
            _ => 0.0,
        })
        .collect();
    let h: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    let mut h_sorted = h.clone();
    h_sorted.sort_by(|a, b| a.total_cmp(b));
    let h_med = h_sorted.get(h_sorted.len() / 2).copied().unwrap_or(0.0);
    if n < 2 || h_med < 0.1 {
        // a single row (or rows at one depth) has no depth profile
        for &r in rows {
            data[r * n_samp..(r + 1) * n_samp].fill(0.0);
        }
        return;
    }

    let sigma = sigma_um as f64;
    let mut g = vec![0.0f64; n * n];
    for i in 0..n {
        if sigma <= 0.0 {
            g[i * n + i] = 1.0;
            continue;
        }
        let mut sum = 0.0;
        for j in 0..n {
            let d = y[j] - y[i];
            if d.abs() <= 3.0 * sigma {
                let w = (-d * d / (2.0 * sigma * sigma)).exp();
                g[i * n + j] = w;
                sum += w;
            }
        }
        for j in 0..n {
            g[i * n + j] /= sum;
        }
    }

    // second difference; a mirrored row beyond each end has the end row's value
    let mut d2 = vec![0.0f64; n * n];
    for i in 0..n {
        let hm = if i > 0 { h[i - 1] } else { h[0] };
        let hp = if i < n - 1 { h[i] } else { h[n - 2] };
        if hm < 0.1 || hp < 0.1 {
            continue;
        }
        let cm = 2.0 / (hm * (hm + hp));
        let cp = 2.0 / (hp * (hm + hp));
        d2[i * n + i] -= cm + cp;
        d2[i * n + if i > 0 { i - 1 } else { i }] += cm;
        d2[i * n + if i < n - 1 { i + 1 } else { i }] += cp;
    }

    // combined operator −(ref² + σ²)·D·G, kept sparse (a few taps per row)
    let scale = -(CSD_REF_UM * CSD_REF_UM + sigma * sigma);
    let m: Vec<Vec<(usize, f32)>> = (0..n)
        .map(|i| {
            (0..n)
                .filter_map(|j| {
                    let v: f64 = (0..n).map(|k| d2[i * n + k] * g[k * n + j]).sum::<f64>() * scale;
                    (v.abs() > 1e-9).then_some((j, v as f32))
                })
                .collect()
        })
        .collect();

    let mut out = vec![0.0f32; n * n_samp];
    {
        let src: Vec<&[f32]> = rows.iter().map(|&r| &data[r * n_samp..(r + 1) * n_samp]).collect();
        out.par_chunks_mut(n_samp).zip(&m).for_each(|(o, taps)| {
            for &(j, w) in taps {
                for (o, &s) in o.iter_mut().zip(src[j]) {
                    *o += w * s;
                }
            }
        });
    }
    for (i, &r) in rows.iter().enumerate() {
        data[r * n_samp..(r + 1) * n_samp].copy_from_slice(&out[i * n_samp..(i + 1) * n_samp]);
    }
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

fn apply_global_cmr(data: &mut [f32], n_rows: usize, n_samp: usize) {
    if n_rows == 0 || n_samp == 0 {
        return;
    }
    let trace = crate::stats::median_trace(data, n_rows, n_samp);
    data.par_chunks_mut(n_samp).for_each(|row| {
        for (v, &m) in row.iter_mut().zip(&trace) {
            *v -= m;
        }
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
        for (t, &x) in src.iter().enumerate() {
            prefix[t + 1] = prefix[t] + x.abs() as f64;
        }
        for (t, gt) in g.iter_mut().enumerate() {
            let lo = t.saturating_sub(half);
            let hi = (t + half + 1).min(n_samp);
            let count = hi - lo;
            let mean_abs = (prefix[hi] - prefix[lo]) / count as f64;
            *gt = (mean_abs as f32).max(epsilon);
        }
    });
    gain
}

// ---------------------------------------------------------------------------
// Spatial kfilt (IBL destripe)
// ---------------------------------------------------------------------------

thread_local! {
    /// one padded time block of the rows, `[n_padded][n_t]`, reused across calls
    static KFILT_BLOCK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

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

    // the block's rows in depth order; `gain` has the same order
    let mut block_rows = select_rows_mut(data, n_samp, rows);

    // divide by gain
    block_rows.par_iter_mut().zip(gain.par_chunks(n_samp)).for_each(|(row, g)| {
        for (v, &gv) in row.iter_mut().zip(g) {
            *v /= gv;
        }
    });

    let sos = &filt.kfilt;
    let blocks = time_blocks_mut(block_rows, SPATIAL_BLOCK_T);
    blocks.into_par_iter().enumerate().for_each(|(b, mut block)| {
        let t_start = b * SPATIAL_BLOCK_T;
        let n_t = block[0].len();
        KFILT_BLOCK.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.resize(n_padded * n_t, 0.0);

            // the rows in depth order, between `pad` mirrored rows on each side
            for (i, row) in block.iter().enumerate() {
                buf[(pad + i) * n_t..(pad + i + 1) * n_t].copy_from_slice(row);
            }
            for p in 0..pad {
                let src = pad + (pad - 1 - p).min(n_rows - 1);
                buf.copy_within(src * n_t..(src + 1) * n_t, p * n_t);
            }
            for p in 0..pad {
                let src = pad + n_rows.saturating_sub(1 + p);
                buf.copy_within(src * n_t..(src + 1) * n_t, (pad + n_rows + p) * n_t);
            }

            // filter along depth, all time samples of the block at once
            SCRATCH.with(|s| sos.filtfilt_columns(&mut buf[..n_padded * n_t], n_padded, n_t, &mut s.borrow_mut()));

            // back into the rows, restoring the gain
            for (i, row) in block.iter_mut().enumerate() {
                let filtered = &buf[(pad + i) * n_t..(pad + i + 1) * n_t];
                let g = &gain[i * n_samp + t_start..i * n_samp + t_start + n_t];
                for ((o, &v), &gv) in row.iter_mut().zip(filtered).zip(g) {
                    *o = v * gv;
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
    fn filtfilt_columns_matches_filtfilt_exactly() {
        let f = butter_highpass(3, 0.01);
        let (n_rows, n_cols) = (97, 13);
        let x: Vec<f32> = (0..n_rows * n_cols).map(|i| ((i * 7919) % 1000) as f32 / 10.0 - 50.0).collect();
        let mut by_cols = x.clone();
        let mut scratch = Vec::new();
        f.filtfilt_columns(&mut by_cols, n_rows, n_cols, &mut scratch);
        for c in 0..n_cols {
            let mut col: Vec<f32> = (0..n_rows).map(|r| x[r * n_cols + c]).collect();
            f.filtfilt(&mut col, &mut scratch);
            for r in 0..n_rows {
                assert_eq!(col[r].to_bits(), by_cols[r * n_cols + c].to_bits(), "row {r} col {c}");
            }
        }
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
