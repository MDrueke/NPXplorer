// Per-channel power spectrum (Welch PSD), used by the "Power Spectrum" overlay: a
// log-frequency heatmap drawn to the right of the main heatmap, one row per channel,
// aligned with its rows. Two independent axes of settings: where the samples come
// from (current view window vs. evenly-spaced chunks across the whole recording) and
// which signal (raw voltage vs. the currently preprocessed buffer).
//
// The FFT uses `rustfft` rather than the hand-rolled radix-2 FFT in
// channel_classify.rs, since this needs a full per-bin spectrum (not just a single
// scalar HF feature) and benefits from rustfft's non-power-of-two support.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use rayon::prelude::*;
use rustfft::num_complex::Complex;

use crate::data::{DisplayRow, Meta, RawData};
use crate::preprocess::PreprocConfig;

/// Evenly-spaced chunk length for whole-recording mode: long enough for several
/// Welch segments' worth of averaging within one chunk.
pub const SPECTRUM_CHUNK_DUR_S: f64 = 1.0;

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SpectrumTimeScope {
    CurrentView,
    WholeRecordingChunks,
}
impl Default for SpectrumTimeScope {
    fn default() -> Self {
        Self::WholeRecordingChunks
    }
}

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SpectrumSource {
    Raw,
    Preprocessed,
}
impl Default for SpectrumSource {
    fn default() -> Self {
        Self::Raw
    }
}

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SpectrumScaling {
    Linear,
    Db,
}
impl Default for SpectrumScaling {
    fn default() -> Self {
        Self::Db
    }
}

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SpectrumNormalization {
    PerChannel,
    Global,
}
impl Default for SpectrumNormalization {
    fn default() -> Self {
        Self::Global
    }
}

/// One PSD per data row (same order as the `DisplayRow::Data` rows of the
/// (unzoomed) display-row list it was computed from), up to Nyquist.
pub struct SpectrumResult {
    /// bin centre frequencies, Hz, 0..=nyquist, linearly spaced
    pub freqs: Vec<f32>,
    /// `power[row][bin]`, one-sided PSD (µV²/Hz)
    pub power: Vec<Vec<f32>>,
}

/// A result together with what it was computed from, so it can tell whether it
/// still lines up with the current display.
pub struct ComputedSpectrum {
    pub result: SpectrumResult,
    /// preprocessing/layout config at compute time: `power`'s rows are indexed by
    /// the data rows of the display-row list this config produces
    pub cfg: PreprocConfig,
    pub source: SpectrumSource,
    /// `(view_first, view_n)` for a current-view result, `None` for whole-recording
    pub view: Option<(usize, usize)>,
}

impl ComputedSpectrum {
    /// Whether this result still describes what the display shows under `cfg`: a raw
    /// spectrum only depends on the row layout, a preprocessed one on every setting.
    pub fn matches(&self, cfg: &PreprocConfig) -> bool {
        match self.source {
            SpectrumSource::Raw => {
                self.cfg.avg_depths == cfg.avg_depths
                    && self.cfg.removed_channels == cfg.removed_channels
                    && self.cfg.channel_order == cfg.channel_order
                    && self.cfg.shank_order == cfg.shank_order
            }
            SpectrumSource::Preprocessed => self.cfg == *cfg,
        }
    }
}

// ---------------------------------------------------------------------------
// Welch PSD
// ---------------------------------------------------------------------------

const NPERSEG_TARGET: usize = 1024;

struct WelchPlan {
    nperseg: usize,
    noverlap: usize,
    window: Vec<f32>,
    win_sum_sq: f32,
    fs: f64,
    n_bins: usize,
    freqs: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

fn build_welch_plan(ns: usize, fs: f64) -> WelchPlan {
    let nperseg = if ns >= NPERSEG_TARGET {
        NPERSEG_TARGET
    } else {
        let mut p = 1usize;
        while p * 2 <= ns.max(2) {
            p *= 2;
        }
        p.max(2)
    };
    let noverlap = nperseg / 2;
    let window = hann_window(nperseg);
    let win_sum_sq: f32 = window.iter().map(|w| w * w).sum();
    let n_bins = nperseg / 2 + 1;
    let freqs: Vec<f32> = (0..n_bins).map(|k| (k as f64 * fs / nperseg as f64) as f32).collect();
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(nperseg);
    WelchPlan { nperseg, noverlap, window, win_sum_sq, fs, n_bins, freqs, fft }
}

/// Welch PSD of one channel/row, all bins (0..=Nyquist). Matches scipy.signal.welch's
/// defaults: periodic Hann window, noverlap = nperseg/2, per-segment mean removed,
/// one-sided density scaling with all bins but DC/Nyquist doubled. Returns all-zero
/// bins if `chan` is shorter than one segment.
fn welch_psd_full(chan: &[f32], plan: &WelchPlan) -> Vec<f32> {
    let ns = chan.len();
    let nperseg = plan.nperseg;
    let mut accum = vec![0f32; plan.n_bins];
    if ns < nperseg {
        return accum;
    }
    let step = nperseg - plan.noverlap;
    let mut buf = vec![Complex::new(0f32, 0f32); nperseg];
    let mut n_seg = 0usize;
    let mut start = 0;
    while start + nperseg <= ns {
        let seg = &chan[start..start + nperseg];
        let mean = seg.iter().sum::<f32>() / nperseg as f32;
        for i in 0..nperseg {
            buf[i] = Complex::new((seg[i] - mean) * plan.window[i], 0.0);
        }
        plan.fft.process(&mut buf);
        for k in 0..plan.n_bins {
            let mag2 = buf[k].re * buf[k].re + buf[k].im * buf[k].im;
            let mut p = mag2 / (plan.fs as f32 * plan.win_sum_sq);
            let is_nyquist = nperseg % 2 == 0 && k == plan.n_bins - 1;
            if k != 0 && !is_nyquist {
                p *= 2.0;
            }
            accum[k] += p;
        }
        n_seg += 1;
        start += step;
    }
    if n_seg > 0 {
        let inv = 1.0 / n_seg as f32;
        for v in &mut accum {
            *v *= inv;
        }
    }
    accum
}

/// PSD of every row of a `[n_rows][n_samp]` buffer, in parallel.
fn psd_rows_from_flat(flat: &[f32], n_rows: usize, n_samp: usize, fs: f64) -> (Vec<f32>, Vec<Vec<f32>>) {
    let plan = build_welch_plan(n_samp, fs);
    let power: Vec<Vec<f32>> = (0..n_rows)
        .into_par_iter()
        .map(|r| welch_psd_full(&flat[r * n_samp..(r + 1) * n_samp], &plan))
        .collect();
    (plan.freqs.clone(), power)
}

fn n_data_rows(display_rows: &[DisplayRow]) -> usize {
    display_rows.iter().filter(|r| matches!(r, DisplayRow::Data { .. })).count()
}

// ---------------------------------------------------------------------------
// Current-view (one-shot) computations
// ---------------------------------------------------------------------------

/// Raw voltage, current view window: reads straight from the mapped file.
pub fn compute_psd_raw_current_view(
    raw: &RawData,
    meta: &Meta,
    display_rows: &[DisplayRow],
    view_first: usize,
    view_n: usize,
) -> Option<SpectrumResult> {
    let fs = meta.sample_rate;
    let n_rows = n_data_rows(display_rows);
    if n_rows == 0 || view_n == 0 {
        return None;
    }
    let first = view_first.min(meta.n_samples);
    let n_samp = view_n.min(meta.n_samples.saturating_sub(first));
    if n_samp < 2 {
        return None;
    }
    let flat = raw.read_rows(first, n_samp, meta, display_rows, false);
    let (freqs, power) = psd_rows_from_flat(&flat, n_rows, n_samp, fs);
    Some(SpectrumResult { freqs, power })
}

/// Preprocessed buffer, current view window: `data` is the worker's `[n_data_rows][data_stride]`
/// buffer; `None` if the requested view isn't (yet) fully covered by it.
pub fn compute_psd_preprocessed_current_view(
    data: &[f32],
    display_rows: &[DisplayRow],
    data_stride: usize,
    buf_first: usize,
    buf_n_samp: usize,
    view_first: usize,
    view_n: usize,
    fs: f64,
) -> Option<SpectrumResult> {
    let buf_end = buf_first + buf_n_samp;
    if view_n == 0 || view_first < buf_first || view_first + view_n > buf_end {
        return None;
    }
    let lo = view_first - buf_first;
    let row_data_idx: Vec<usize> = display_rows
        .iter()
        .filter_map(|r| match r {
            DisplayRow::Data { data_idx, .. } => Some(*data_idx),
            _ => None,
        })
        .collect();
    if row_data_idx.is_empty() {
        return None;
    }
    let plan = build_welch_plan(view_n, fs);
    let power: Vec<Vec<f32>> = row_data_idx
        .par_iter()
        .map(|&data_idx| {
            let base = data_idx * data_stride;
            let seg = &data[base + lo..base + lo + view_n];
            welch_psd_full(seg, &plan)
        })
        .collect();
    Some(SpectrumResult { freqs: plan.freqs.clone(), power })
}

// ---------------------------------------------------------------------------
// Whole-recording (background, cancellable) computation — evenly-spaced raw
// chunks, PSDs averaged across chunks. Mirrors channel_classify::classify_recording.
// ---------------------------------------------------------------------------

pub fn compute_psd_whole_recording(
    raw: &RawData,
    meta: &Meta,
    display_rows: &[DisplayRow],
    n_chunks: usize,
    // restricts the span chunks are evenly spaced across to [start_s, end_s]
    // (clamped to the recording); None uses the whole recording
    time_range: Option<(f64, f64)>,
    cancel: &AtomicBool,
    progress: &AtomicUsize,
) -> Result<SpectrumResult, String> {
    const CANCELLED: &str = "cancelled";
    let fs = meta.sample_rate;
    let n_rows = n_data_rows(display_rows);
    if n_rows == 0 {
        return Err("no channels to analyse".into());
    }
    let total_dur = meta.n_samples as f64 / fs;
    // order-independent; an empty window falls back to the whole recording
    let (t_start, t_end) = match time_range {
        Some((a, b)) => {
            let (s, e) = (a.min(b).clamp(0.0, total_dur), a.max(b).clamp(0.0, total_dur));
            if e > s { (s, e) } else { (0.0, total_dur) }
        }
        None => (0.0, total_dur),
    };
    // chunks never reach outside the window, even one shorter than a chunk
    let chunk_s = SPECTRUM_CHUNK_DUR_S.min(t_end - t_start);
    let chunk_samp = ((chunk_s * fs) as usize).clamp(2, meta.n_samples.max(2));
    let max_t0 = t_start + (t_end - t_start - chunk_s).max(0.0);
    let plan = build_welch_plan(chunk_samp, fs);
    let n_bins = plan.n_bins;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(crate::worker::compute_thread_count())
        .build()
        .map_err(|e| e.to_string())?;

    // (per-row PSD sums, number of chunks that contributed)
    let (sums, used) = pool.install(|| {
        (0..n_chunks)
            .into_par_iter()
            .fold(
                || (vec![vec![0f32; n_bins]; n_rows], 0usize),
                |(mut acc, mut used), i| {
                    if cancel.load(Ordering::Relaxed) {
                        return (acc, used);
                    }
                    let t0 = if n_chunks <= 1 {
                        t_start
                    } else {
                        t_start + (max_t0 - t_start) * i as f64 / (n_chunks - 1) as f64
                    };
                    let first_sample = (t0 * fs) as usize;
                    let n_samp = chunk_samp.min(meta.n_samples.saturating_sub(first_sample));
                    if n_samp >= plan.nperseg {
                        let flat = raw.read_rows(first_sample, n_samp, meta, display_rows, false);
                        if !cancel.load(Ordering::Relaxed) {
                            for r in 0..n_rows {
                                let psd = welch_psd_full(&flat[r * n_samp..(r + 1) * n_samp], &plan);
                                for k in 0..n_bins {
                                    acc[r][k] += psd[k];
                                }
                            }
                            used += 1;
                        }
                    }
                    progress.fetch_add(1, Ordering::Relaxed);
                    (acc, used)
                },
            )
            .reduce(
                || (vec![vec![0f32; n_bins]; n_rows], 0usize),
                |(mut a, ua), (b, ub)| {
                    for r in 0..n_rows {
                        for k in 0..n_bins {
                            a[r][k] += b[r][k];
                        }
                    }
                    (a, ua + ub)
                },
            )
    });

    if cancel.load(Ordering::Relaxed) {
        return Err(CANCELLED.into());
    }
    if used == 0 {
        return Err("the time window is too short for a spectrum".into());
    }
    let div = used as f32;
    let power: Vec<Vec<f32>> = sums
        .into_iter()
        .map(|row| row.into_iter().map(|v| v / div).collect())
        .collect();
    Ok(SpectrumResult { freqs: plan.freqs, power })
}
