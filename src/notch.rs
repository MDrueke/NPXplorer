//! Notch filters against narrowband noise, and the scan that finds it.
//!
//! The notches are part of the preprocessing (after DC removal, before the highpass).
//! They are recording-specific: saved with the recording's settings (AP and LF
//! separately — their noise differs, see settings.rs) and active again when the file
//! is reopened.
//!
//! The scan samples evenly spaced chunks, preprocesses them with the current settings
//! (without notches), and averages each channel's power spectrum over the chunks.
//! Narrow peaks are detected on every channel's own spectrum — so noise that reaches
//! the channels at different times (a diagonal pattern that CMR/destripe cannot
//! remove), or only some channels, is found too — and listed once they occur on enough
//! of a shank's channels. Line noise and electronics are narrow, brain oscillations broad.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use rayon::prelude::*;
use rustfft::num_complex::Complex;

use crate::data::{DisplayRow, Meta, RawData};
use crate::preprocess::{preprocess, Filters, PreprocConfig};

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Notch {
    pub freq_hz: f64,
    /// -3 dB bandwidth of one pass (Hz)
    pub bw_hz: f64,
}

pub const MIN_BW_HZ: f64 = 0.5;

/// `[b0, b1, b2, a1, a2]` of a second-order IIR notch (`scipy.signal.iirnotch`).
fn design(n: &Notch, fs: f64) -> Option<[f64; 5]> {
    let nyq = fs / 2.0;
    if !(n.freq_hz > 0.0 && n.freq_hz < nyq && n.bw_hz > 0.0) {
        return None;
    }
    let w0 = 2.0 * std::f64::consts::PI * n.freq_hz / fs;
    let bw = 2.0 * std::f64::consts::PI * n.bw_hz.max(MIN_BW_HZ) / fs;
    let gain = 1.0 / (1.0 + (bw / 2.0).tan());
    let c = w0.cos();
    Some([gain, -2.0 * gain * c, gain, -2.0 * gain * c, 2.0 * gain - 1.0])
}

/// Steady-state section states for a unit step (`sosfilt_zi`), as in `SosFilter::new`.
fn sos_zi(sos: &[[f64; 5]]) -> Vec<[f64; 2]> {
    let mut scale = 1.0;
    sos.iter()
        .map(|&[b0, b1, b2, a1, a2]| {
            let g = (b0 + b1 + b2) / (1.0 + a1 + a2);
            let z1 = b2 - a2 * g;
            let z0 = b1 - a1 * g + z1;
            let zi = [z0 * scale, z1 * scale];
            scale *= g;
            zi
        })
        .collect()
}

fn filt_with_zi(x: &mut [f64], sos: &[[f64; 5]], zi: &[[f64; 2]], x0: f64) {
    for (&[b0, b1, b2, a1, a2], z) in sos.iter().zip(zi) {
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

/// Zero-phase notch filtering of one channel, in f64: the poles of a narrow notch sit
/// so close to the unit circle that f32 state loses the attenuation.
fn filtfilt(x: &mut [f32], sos: &[[f64; 5]], zi: &[[f64; 2]], scratch: &mut Vec<f64>) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let padlen = (3 * (2 * sos.len() + 1)).min(n - 1);
    scratch.clear();
    for i in 0..padlen {
        scratch.push(2.0 * x[0] as f64 - x[padlen - i] as f64);
    }
    scratch.extend(x.iter().map(|&v| v as f64));
    for i in 0..padlen {
        scratch.push(2.0 * x[n - 1] as f64 - x[n - 2 - i] as f64);
    }
    let x0 = scratch[0];
    filt_with_zi(scratch, sos, zi, x0);
    scratch.reverse();
    let y0 = scratch[0];
    filt_with_zi(scratch, sos, zi, y0);
    scratch.reverse();
    for (o, &v) in x.iter_mut().zip(&scratch[padlen..padlen + n]) {
        *o = v as f32;
    }
}

/// Notch every row of `data` (`[rows][n_samp]`) in place.
pub fn apply_notches(data: &mut [f32], n_samp: usize, notches: &[Notch], fs: f64) {
    let sos: Vec<[f64; 5]> = notches.iter().filter_map(|n| design(n, fs)).collect();
    if sos.is_empty() || n_samp == 0 {
        return;
    }
    let zi = sos_zi(&sos);
    data.par_chunks_mut(n_samp).for_each_init(Vec::new, |scratch, x| filtfilt(x, &sos, &zi, scratch));
}

/// Seconds the enabled notches need to settle at the edge of a chunk (4 time
/// constants of the narrowest one, ~2 % left); 0 without notches.
pub fn settle_s(cfg: &PreprocConfig) -> f64 {
    if !cfg.notch_enabled {
        return 0.0;
    }
    cfg.notches
        .iter()
        .filter(|n| design(n, cfg.sample_rate).is_some())
        .map(|n| 4.0 / (std::f64::consts::PI * n.bw_hz.max(MIN_BW_HZ)))
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ScanSettings {
    pub n_chunks: usize,
    /// a peak must stand this far (dB) above the running-median baseline
    pub min_prominence_db: f32,
    /// ...and be at most this wide (Hz, -3 dB)
    pub max_width_hz: f32,
    /// ...and be present in at least this fraction of the chunks (estimated from its
    /// variation across chunks): transient peaks (e.g. stimulation artifacts) are no
    /// case for a notch on the whole recording
    pub min_presence: f32,
    /// a frequency is listed once it is a peak on this fraction of a shank's channels
    pub min_channel_frac: f32,
}

impl Default for ScanSettings {
    fn default() -> Self {
        ScanSettings { n_chunks: 100, min_prominence_db: 6.0, max_width_hz: 3.0, min_presence: 0.25, min_channel_frac: 0.3 }
    }
}

/// Mean of the shank's channel spectra, for the plot.
pub struct ShankSpectrum {
    pub shank: u32,
    pub psd_db: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Peak {
    /// median over the channels it was found on
    pub freq_hz: f64,
    /// median -3 dB width over those channels
    pub width_hz: f64,
    /// (shank, fraction of the shank's channels) for every shank it was found on
    pub shank_frac: Vec<(u32, f32)>,
    pub n_channels: usize,
    /// median and largest prominence (dB) over the channels
    pub prom_median: f32,
    pub prom_max: f32,
    /// median estimated fraction of chunks in which it is present
    pub presence: f32,
    /// lowest other listed peak this one is an integer multiple of
    pub harmonic_of: Option<f64>,
}

pub struct ScanResult {
    /// frequency of bin k = k × df
    pub df: f64,
    pub shanks: Vec<ShankSpectrum>,
    pub peaks: Vec<Peak>,
    /// table rows: indices into `peaks`, one per row, or several for a series of evenly
    /// spaced lines (one source, e.g. a carrier with mains sidebands); by frequency
    pub groups: Vec<Vec<usize>>,
}

/// Groups `freqs` (ascending) into series of at least 3 evenly spaced lines; the rest
/// stay single. A series may skip one line at a time (a sideband too weak to be
/// listed); `tol` is how far a line may sit from its expected place.
fn group_series(freqs: &[f64], tol: f64, min_spacing: f64) -> Vec<Vec<usize>> {
    let n = freqs.len();
    let mut free = vec![true; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    loop {
        let mut best: Vec<usize> = Vec::new();
        for i in (0..n).filter(|&i| free[i]) {
            for j in (i + 1..n).filter(|&j| free[j]) {
                let d0 = freqs[j] - freqs[i];
                if d0 < min_spacing {
                    continue;
                }
                let mut members = vec![i, j];
                // spacing re-estimated from the series' ends as it grows
                let (mut last_m, mut d) = (1usize, d0);
                let mut m = 2;
                while m - last_m <= 2 {
                    let target = freqs[i] + m as f64 * d;
                    if target > freqs[n - 1] + tol {
                        break;
                    }
                    if let Some(k) = (j + 1..n).find(|&k| free[k] && (freqs[k] - target).abs() <= tol) {
                        members.push(k);
                        last_m = m;
                        d = (freqs[k] - freqs[i]) / m as f64;
                    }
                    m += 1;
                }
                // ...and downwards from i, with the same spacing and allowance
                let mut m = 1;
                let mut last_m = 0;
                while m - last_m <= 2 {
                    let target = freqs[i] - m as f64 * d;
                    if target < freqs[0] - tol {
                        break;
                    }
                    if let Some(k) = (0..i).rev().find(|&k| free[k] && (freqs[k] - target).abs() <= tol) {
                        members.push(k);
                        last_m = m;
                    }
                    m += 1;
                }
                members.sort_unstable();
                if members.len() > best.len() {
                    best = members;
                }
            }
        }
        if best.len() < 3 {
            break;
        }
        for &k in &best {
            free[k] = false;
        }
        groups.push(best);
    }
    groups.extend((0..n).filter(|&k| free[k]).map(|k| vec![k]));
    groups.sort_by(|a, b| freqs[a[0]].total_cmp(&freqs[b[0]]));
    groups
}

/// Lowest frequency considered for peaks (Hz).
const F_MIN_HZ: f64 = 1.0;

/// One channel's narrow peak.
struct ChannelPeak {
    freq_hz: f64,
    width_hz: f64,
    row: usize,
    prom: f32,
    presence: f32,
}

/// Fraction of chunks a peak is present in, estimated from its power's coefficient of
/// variation across chunks: a line that is on (with constant power) in a fraction q of
/// the chunks and off otherwise has CV² = (1 - q) / q, so q = 1 / (1 + CV²). Steady
/// lines on top of noise come out at 0.7–1. A line whose strength drifts reads lower
/// (40–60 % on real data), transient ones far lower (1–4 %), hence the 25 % default.
fn presence_from_moments(sum: f64, sum_sq: f64, n: f64) -> f32 {
    let mean = sum / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let cv2 = ((sum_sq / n - mean * mean) / (mean * mean)).max(0.0);
    (1.0 / (1.0 + cv2)) as f32
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

pub fn scan(
    raw: &RawData,
    meta: &Meta,
    cfg: &PreprocConfig,
    settings: &ScanSettings,
    cancel: &AtomicBool,
    progress: &AtomicUsize,
) -> Result<ScanResult, String> {
    let fs = meta.sample_rate;
    // what is left after the current preprocessing, without the notches themselves
    let mut cfg = cfg.clone();
    cfg.notch_enabled = false;
    let rows = meta.build_display_rows(cfg.avg_depths, &cfg.removed_channels, cfg.channel_order, cfg.shank_order);
    let row_shank: Vec<u32> = rows
        .iter()
        .filter_map(|r| if let DisplayRow::Data { shank, .. } = r { Some(*shank) } else { None })
        .collect();
    if row_shank.is_empty() {
        return Err("no channels to analyse".into());
    }
    let mut shanks: Vec<u32> = row_shank.clone();
    shanks.sort_unstable();
    shanks.dedup();
    let shank_idx: Vec<usize> = row_shank.iter().map(|s| shanks.binary_search(s).unwrap()).collect();
    let rows_per_shank: Vec<usize> = (0..shanks.len()).map(|i| shank_idx.iter().filter(|&&s| s == i).count()).collect();

    // one Hann-windowed segment of ~1 s per chunk (~1 Hz resolution), with a margin on
    // both sides for the preprocessing filters to settle
    let nfft = (fs.round() as usize).next_power_of_two().max(256);
    let pad = (0.1 * fs).round() as usize;
    let len = nfft + 2 * pad;
    if meta.n_samples < len {
        return Err("the recording is too short for a scan".into());
    }
    let n_bins = nfft / 2 + 1;
    let df = fs / nfft as f64;
    let window: Vec<f32> = (0..nfft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / nfft as f32).cos())
        .collect();
    let fft = rustfft::FftPlanner::<f32>::new().plan_fft_forward(nfft);
    let filters = Filters::new(&cfg);

    let n_chunks = settings.n_chunks.max(1);
    let max_first = meta.n_samples - len;
    // per channel: sum of the chunks' power and of its square, per frequency bin
    let mut moments: Vec<(Vec<f64>, Vec<f64>)> = vec![(vec![0.0; n_bins], vec![0.0; n_bins]); row_shank.len()];
    let mut used = 0usize;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(crate::worker::compute_thread_count())
        .build()
        .map_err(|e| e.to_string())?;
    pool.install(|| -> Result<(), String> {
        for c in 0..n_chunks {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let first = if n_chunks == 1 { max_first / 2 } else { max_first * c / (n_chunks - 1) };
            let mut data = raw.read_rows(first, len, meta, &rows, cfg.phase_shift);
            preprocess(&mut data, len, &cfg, &filters, cancel, &rows, None);
            data.par_chunks(len).zip(moments.par_iter_mut()).for_each_init(
                || vec![Complex::new(0.0f32, 0.0); nfft],
                |buf, (row, (s1, s2))| {
                    for (b, (&v, &w)) in buf.iter_mut().zip(row[pad..pad + nfft].iter().zip(&window)) {
                        *b = Complex::new(v * w, 0.0);
                    }
                    fft.process(buf);
                    for ((a, q), z) in s1.iter_mut().zip(s2.iter_mut()).zip(&buf[..n_bins]) {
                        let p = z.norm_sqr() as f64;
                        *a += p;
                        *q += p * p;
                    }
                },
            );
            used += 1;
            progress.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    })?;
    let n = used as f64;

    let k_min = (F_MIN_HZ / df).ceil() as usize;
    // stay clear of the anti-alias rolloff at the top of the band
    let k_max = ((n_bins - 1) as f64 * 0.98) as usize;

    // narrow, steady peaks of every channel's own mean spectrum
    let mut found: Vec<ChannelPeak> = pool.install(|| {
        moments
            .par_iter()
            .enumerate()
            .flat_map_iter(|(row, (s1, s2))| {
                let db: Vec<f32> = s1.iter().map(|&v| to_db(v / n)).collect();
                let base = running_median_baseline(&db, df);
                find_peaks(&db, &base, df, k_min, k_max, settings)
                    .into_iter()
                    .filter_map(|(k, width_hz, freq_hz, prom)| {
                        let presence = presence_from_moments(s1[k], s2[k], n);
                        (presence >= settings.min_presence).then_some(ChannelPeak { freq_hz, width_hz, row, prom, presence })
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    });

    // the same line on different channels: chains of peaks less than 1.5 bins apart
    found.sort_by(|a, b| a.freq_hz.total_cmp(&b.freq_hz));
    let mut groups: Vec<Vec<ChannelPeak>> = Vec::new();
    for p in found {
        match groups.last_mut() {
            Some(g) if p.freq_hz - g.last().unwrap().freq_hz <= 1.5 * df => g.push(p),
            _ => groups.push(vec![p]),
        }
    }
    let mut peaks: Vec<Peak> = groups
        .into_iter()
        .filter_map(|g| {
            let mut chans: Vec<usize> = g.iter().map(|p| p.row).collect();
            chans.sort_unstable();
            chans.dedup();
            let mut per_shank = vec![0usize; shanks.len()];
            for &r in &chans {
                per_shank[shank_idx[r]] += 1;
            }
            let shank_frac: Vec<(u32, f32)> = per_shank
                .iter()
                .enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(s, &c)| (shanks[s], c as f32 / rows_per_shank[s] as f32))
                .collect();
            if !shank_frac.iter().any(|&(_, f)| f >= settings.min_channel_frac) {
                return None;
            }
            let v = |f: &dyn Fn(&ChannelPeak) -> f64| median(&mut g.iter().map(f).collect::<Vec<_>>());
            Some(Peak {
                freq_hz: v(&|p| p.freq_hz),
                width_hz: v(&|p| p.width_hz),
                prom_median: v(&|p| p.prom as f64) as f32,
                presence: v(&|p| p.presence as f64) as f32,
                prom_max: g.iter().map(|p| p.prom).fold(f32::MIN, f32::max),
                n_channels: chans.len(),
                shank_frac,
                harmonic_of: None,
            })
        })
        .collect();
    for i in 0..peaks.len() {
        let f = peaks[i].freq_hz;
        peaks[i].harmonic_of = peaks[..i].iter().map(|p| p.freq_hz).find(|&f0| {
            let k = (f / f0).round();
            k >= 2.0 && (f - k * f0).abs() <= 1.5 * df
        });
    }

    let out_shanks = shanks
        .iter()
        .enumerate()
        .map(|(s, &shank)| {
            let mut mean = vec![0.0f64; n_bins];
            for (_, (s1, _)) in moments.iter().enumerate().filter(|(r, _)| shank_idx[*r] == s) {
                for (m, &v) in mean.iter_mut().zip(s1) {
                    *m += v;
                }
            }
            let norm = n * rows_per_shank[s] as f64;
            ShankSpectrum { shank, psd_db: mean.iter().map(|&v| to_db(v / norm)).collect() }
        })
        .collect();
    let freqs: Vec<f64> = peaks.iter().map(|p| p.freq_hz).collect();
    let groups = group_series(&freqs, 1.5 * df, 3.0 * df);
    Ok(ScanResult { df, shanks: out_shanks, peaks, groups })
}

fn to_db(p: f64) -> f32 {
    (10.0 * (p + 1e-30).log10()) as f32
}

/// Running median of the dB spectrum over ±max(5 Hz, 5 % of f): follows the 1/f slope
/// and broad bumps, but not narrow lines. Evaluated on every 4th bin and interpolated.
fn running_median_baseline(db: &[f32], df: f64) -> Vec<f32> {
    let n = db.len();
    let step = 4;
    let at = |k: usize| {
        let half = ((5.0f64).max(0.05 * k as f64 * df) / df).ceil() as usize;
        let (lo, hi) = (k.saturating_sub(half), (k + half + 1).min(n));
        let mut w: Vec<f32> = db[lo..hi].to_vec();
        let mid = w.len() / 2;
        *w.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
    };
    let knots: Vec<(usize, f32)> = (0..n).step_by(step).chain(std::iter::once(n - 1)).map(|k| (k, at(k))).collect();
    let mut base = vec![0.0f32; n];
    for w in knots.windows(2) {
        let ((k0, v0), (k1, v1)) = (w[0], w[1]);
        for k in k0..=k1 {
            let t = if k1 > k0 { (k - k0) as f32 / (k1 - k0) as f32 } else { 0.0 };
            base[k] = v0 + (v1 - v0) * t;
        }
    }
    base
}

/// Narrow local maxima: (bin, -3 dB width in Hz, refined frequency, prominence dB).
fn find_peaks(
    db: &[f32],
    base: &[f32],
    df: f64,
    k_min: usize,
    k_max: usize,
    settings: &ScanSettings,
) -> Vec<(usize, f64, f64, f32)> {
    let mut out = Vec::new();
    let max_bins = (settings.max_width_hz as f64 / df).ceil() as usize + 2;
    for k in k_min.max(1)..k_max.min(db.len() - 1) {
        let prom = db[k] - base[k];
        if prom < settings.min_prominence_db || db[k] < db[k - 1] || db[k] < db[k + 1] {
            continue;
        }
        // -3 dB points, linearly interpolated; give up (= broad) beyond max_bins
        let level = db[k] - 3.0;
        let edge = |dir: isize| -> Option<f64> {
            let mut j = k as isize;
            for _ in 0..max_bins {
                let next = j + dir;
                if next < 0 || next as usize >= db.len() {
                    return None;
                }
                if db[next as usize] < level {
                    let (a, b) = (db[j as usize], db[next as usize]);
                    return Some(j as f64 + dir as f64 * ((a - level) / (a - b)) as f64);
                }
                j = next;
            }
            None
        };
        let (Some(l), Some(r)) = (edge(-1), edge(1)) else { continue };
        let width_hz = (r - l) * df;
        if width_hz > settings.max_width_hz as f64 {
            continue;
        }
        // parabolic interpolation of the maximum
        let (a, b, c) = (db[k - 1] as f64, db[k] as f64, db[k + 1] as f64);
        let denom = a - 2.0 * b + c;
        let delta = if denom.abs() > 1e-12 { (0.5 * (a - c) / denom).clamp(-0.5, 0.5) } else { 0.0 };
        out.push((k, width_hz, (k as f64 + delta) * df, prom));
    }
    out
}

// ---------------------------------------------------------------------------
// Window section
// ---------------------------------------------------------------------------

struct Job {
    rx: mpsc::Receiver<Result<ScanResult, String>>,
    cancel: Arc<AtomicBool>,
    progress: Arc<AtomicUsize>,
    total: usize,
}

/// "Notch filters" section of the Noise Suppression window.
pub struct NotchPanel {
    settings: ScanSettings,
    job: Option<Job>,
    result: Option<ScanResult>,
    error: Option<String>,
    /// edited list; Apply hands it to the preprocessing
    pub draft: Vec<Notch>,
    manual: Notch,
}

impl NotchPanel {
    pub fn new(applied: &[Notch], settings: ScanSettings) -> Self {
        NotchPanel {
            settings,
            job: None,
            result: None,
            error: None,
            draft: applied.to_vec(),
            manual: Notch { freq_hz: 50.0, bw_hz: 1.0 },
        }
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.job else { return };
        match job.rx.try_recv() {
            Ok(res) => {
                self.job = None;
                match res {
                    Ok(r) => {
                        self.result = Some(r);
                        self.error = None;
                    }
                    Err(e) if e == "cancelled" => {}
                    Err(e) => self.error = Some(e),
                }
            }
            Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(80)),
            Err(mpsc::TryRecvError::Disconnected) => self.job = None,
        }
    }

    /// Stop a running scan (the window is being closed).
    pub fn abort(&self) {
        if let Some(job) = &self.job {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    pub fn scan_settings(&self) -> &ScanSettings {
        &self.settings
    }

    /// Returns the new notch list when Apply was clicked.
    pub fn draw(
        &mut self,
        ui: &mut egui::Ui,
        raw: &Arc<RawData>,
        meta: &Arc<Meta>,
        cfg: &PreprocConfig,
    ) -> Option<Vec<Notch>> {
        ui.label(
            egui::RichText::new(
                "Part of the preprocessing: changes the data everywhere (heatmap, firing rate, \
                 PSTH, spectrum). Saved next to the data file and active again when it is reopened.",
            )
            .small()
            .weak(),
        );

        // --- scan ---
        egui::Grid::new("notch_scan_grid").num_columns(2).show(ui, |ui| {
            ui.label("Chunks to sample:");
            ui.add(egui::DragValue::new(&mut self.settings.n_chunks).range(1..=1000).speed(1.0))
                .on_hover_text("Evenly spaced ~1 s chunks across the recording.");
            ui.end_row();
            ui.label("Min. prominence:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.settings.min_prominence_db).range(1.0..=60.0).speed(0.1))
                    .on_hover_text("How far a peak must stand above the smoothed spectrum.");
                ui.label("dB");
            });
            ui.end_row();
            ui.label("Max. width:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.settings.max_width_hz).range(0.5..=50.0).speed(0.05))
                    .on_hover_text("Peaks wider than this (-3 dB) are not listed: broad peaks are usually brain oscillations, not noise.");
                ui.label("Hz");
            });
            ui.end_row();
            ui.label("Min. channels:");
            ui.horizontal(|ui| {
                let mut pct = self.settings.min_channel_frac * 100.0;
                if ui
                    .add(egui::DragValue::new(&mut pct).range(0.0..=100.0).speed(1.0))
                    .on_hover_text("A frequency is listed once it is a peak on at least this share of one shank's channels.")
                    .changed()
                {
                    self.settings.min_channel_frac = pct / 100.0;
                }
                ui.label("% of a shank");
            });
            ui.end_row();
            ui.label("Min. presence:");
            ui.horizontal(|ui| {
                let mut pct = self.settings.min_presence * 100.0;
                if ui
                    .add(egui::DragValue::new(&mut pct).range(0.0..=100.0).speed(1.0))
                    .on_hover_text("Peaks that appear in fewer chunks are not listed: transient noise (e.g. stimulation artifacts) is no case for a notch on the whole recording. Estimated from how much the peak's power varies between chunks.")
                    .changed()
                {
                    self.settings.min_presence = pct / 100.0;
                }
                ui.label("% of chunks");
            });
            ui.end_row();
        });
        if let Some(job) = &self.job {
            let done = job.progress.load(Ordering::Relaxed);
            ui.horizontal(|ui| {
                ui.add(
                    egui::ProgressBar::new(done as f32 / job.total.max(1) as f32)
                        .desired_width(220.0)
                        .text(format!("{done} / {} chunks", job.total)),
                );
                if job.cancel.load(Ordering::Relaxed) {
                    ui.label("Stopping…");
                } else if ui.button("Abort").clicked() {
                    job.cancel.store(true, Ordering::Relaxed);
                }
            });
        } else if ui
            .button("Find noise peaks")
            .on_hover_text(
                "Averages each channel's power spectrum over chunks of the recording, after the \
                 current preprocessing (without notches), finds narrow peaks on every channel, and \
                 lists those found on enough channels.",
            )
            .clicked()
        {
            let (tx, rx) = mpsc::channel();
            let cancel = Arc::new(AtomicBool::new(false));
            let progress = Arc::new(AtomicUsize::new(0));
            let (raw, meta, cfg, settings) = (Arc::clone(raw), Arc::clone(meta), cfg.clone(), self.settings.clone());
            let (c, p) = (Arc::clone(&cancel), Arc::clone(&progress));
            std::thread::spawn(move || {
                let _ = tx.send(scan(&raw, &meta, &cfg, &settings, &c, &p));
            });
            self.job = Some(Job { rx, cancel, progress, total: self.settings.n_chunks.max(1) });
            self.error = None;
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), e);
        }

        // --- results ---
        if let Some(res) = &self.result {
            draw_plot(ui, res, &self.draft);
            if res.peaks.is_empty() {
                ui.label("No narrow peaks found.");
            } else {
                let multi = res.shanks.len() > 1;
                egui::Grid::new("notch_peaks_grid").num_columns(6).striped(true).show(ui, |ui| {
                    ui.label("");
                    ui.label("Frequency");
                    ui.label("Width");
                    ui.label("Channels").on_hover_text("Share of each shank's channels the peak was found on.");
                    ui.label("Prominence").on_hover_text("Above the smoothed spectrum: median (max) over the channels.");
                    ui.label("Present").on_hover_text("Estimated share of chunks in which the peak was there (median over the channels).");
                    ui.end_row();
                    for g in &res.groups {
                        let members: Vec<&Peak> = g.iter().map(|&k| &res.peaks[k]).collect();
                        let has = |draft: &[Notch], p: &Peak| draft.iter().position(|n| (n.freq_hz - p.freq_hz).abs() <= 1.5 * res.df);
                        let n_on = members.iter().filter(|p| has(&self.draft, p).is_some()).count();
                        let mut on = n_on == members.len();
                        // one checkbox for the whole series
                        if ui.add(egui::Checkbox::new(&mut on, "").indeterminate(n_on > 0 && n_on < members.len())).changed() {
                            for p in &members {
                                match has(&self.draft, p) {
                                    Some(i) if !on => {
                                        self.draft.remove(i);
                                    }
                                    None if on => self.draft.push(Notch {
                                        freq_hz: (p.freq_hz * 10.0).round() / 10.0,
                                        bw_hz: p.width_hz.clamp(1.0, 20.0),
                                    }),
                                    _ => {}
                                }
                            }
                        }
                        let first = members[0];
                        let last = members[members.len() - 1];
                        // the shanks' channel shares: range over the lines of a series
                        let mut shank_ids: Vec<u32> = members.iter().flat_map(|p| p.shank_frac.iter().map(|&(s, _)| s)).collect();
                        shank_ids.sort_unstable();
                        shank_ids.dedup();
                        let chans: Vec<String> = shank_ids
                            .iter()
                            .map(|&s| {
                                let fr: Vec<f32> = members
                                    .iter()
                                    .filter_map(|p| p.shank_frac.iter().find(|&&(t, _)| t == s).map(|&(_, f)| f * 100.0))
                                    .collect();
                                let (lo, hi) = fr.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                                let pct = if (hi - lo).abs() < 0.5 { format!("{hi:.0} %") } else { format!("{lo:.0}–{hi:.0} %") };
                                if multi { format!("S{s} {pct}") } else { pct }
                            })
                            .collect();
                        let prom_med = members.iter().map(|p| p.prom_median).fold(f32::MIN, f32::max);
                        let prom_max = members.iter().map(|p| p.prom_max).fold(f32::MIN, f32::max);
                        let mut pres: Vec<f64> = members.iter().map(|p| p.presence as f64).collect();
                        let pres = median(&mut pres);
                        let mut widths: Vec<f64> = members.iter().map(|p| p.width_hz).collect();
                        let width = median(&mut widths);
                        if members.len() == 1 {
                            let mut f = format!("{:.1} Hz", first.freq_hz);
                            if let Some(f0) = first.harmonic_of {
                                f += &format!("  (×{} of {:.1})", (first.freq_hz / f0).round(), f0);
                            }
                            ui.label(f);
                        } else {
                            // smallest gap, refined over the whole span (lines may be missing)
                            let gap = members.windows(2).map(|w| w[1].freq_hz - w[0].freq_hz).fold(f64::INFINITY, f64::min);
                            let span = last.freq_hz - first.freq_hz;
                            let spacing = span / (span / gap).round().max(1.0);
                            let detail: Vec<String> = members
                                .iter()
                                .map(|p| format!("{:.1} Hz  +{:.0} dB, {} channels", p.freq_hz, p.prom_median, p.n_channels))
                                .collect();
                            ui.label(format!(
                                "{:.1}–{:.1} Hz, every {:.1} Hz ({} lines)",
                                first.freq_hz,
                                last.freq_hz,
                                spacing,
                                members.len()
                            ))
                            .on_hover_text(format!("One source: evenly spaced lines, notched one by one.\n{}", detail.join("\n")));
                        }
                        ui.label(format!("{width:.1} Hz"));
                        ui.label(chans.join(", ")).on_hover_text(format!(
                            "{} channels",
                            members.iter().map(|p| p.n_channels).max().unwrap_or(0)
                        ));
                        ui.label(format!("+{prom_med:.0} dB ({prom_max:.0})"));
                        ui.label(format!("{:.0} %", pres * 100.0));
                        ui.end_row();
                    }
                });
            }
        }

        // --- notch list ---
        ui.add_space(4.0);
        ui.label("Notches:");
        let mut remove = None;
        for (i, n) in self.draft.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut n.freq_hz).range(0.1..=meta.sample_rate / 2.0).speed(0.1));
                ui.label("Hz, width");
                ui.add(egui::DragValue::new(&mut n.bw_hz).range(MIN_BW_HZ..=200.0).speed(0.05))
                    .on_hover_text("-3 dB width. Narrower removes less of the signal but takes longer to settle.");
                ui.label("Hz");
                if ui.small_button("✖").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.draft.remove(i);
        }
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.manual.freq_hz).range(0.1..=meta.sample_rate / 2.0).speed(0.1));
            ui.label("Hz, width");
            ui.add(egui::DragValue::new(&mut self.manual.bw_hz).range(MIN_BW_HZ..=200.0).speed(0.05));
            ui.label("Hz");
            if ui.button("Add").clicked() {
                self.draft.push(self.manual);
            }
        });

        let mut out = None;
        ui.horizontal(|ui| {
            let changed = self.draft != cfg.notches;
            if ui
                .add_enabled(changed, egui::Button::new("Apply notches"))
                .on_disabled_hover_text("Nothing to apply: the list is the one already in use.")
                .clicked()
            {
                self.draft.sort_by(|a, b| a.freq_hz.total_cmp(&b.freq_hz));
                out = Some(self.draft.clone());
            }
            // what is in use right now (e.g. loaded from the file when the recording opened)
            let status = if changed {
                "not applied yet".to_string()
            } else if cfg.notches.is_empty() {
                String::new()
            } else if cfg.notch_enabled {
                format!("{} notch{} active (saved next to the data file)", cfg.notches.len(), if cfg.notches.len() == 1 { "" } else { "es" })
            } else {
                format!("{} notch{} saved, switched off (Notch in the Preprocessing bar)", cfg.notches.len(), if cfg.notches.len() == 1 { "" } else { "es" })
            };
            if !status.is_empty() {
                ui.label(egui::RichText::new(status).small().weak());
            }
        });
        out
    }
}

/// Per-shank spectra (dB, log frequency) with the peaks marked; peaks with a notch in
/// the list are drawn solid.
fn draw_plot(ui: &mut egui::Ui, res: &ScanResult, draft: &[Notch]) {
    let n_bins = res.shanks.first().map_or(0, |s| s.psd_db.len());
    if n_bins < 4 {
        return;
    }
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width().max(300.0), 150.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let plot = rect.shrink2(egui::vec2(4.0, 14.0));
    let (f_lo, f_hi) = (F_MIN_HZ.max(res.df), (n_bins - 1) as f64 * res.df);
    let x_of = |f: f64| plot.left() + ((f / f_lo).ln() / (f_hi / f_lo).ln()) as f32 * plot.width();
    let f_of = |x: f32| f_lo * (f_hi / f_lo).powf(((x - plot.left()) / plot.width()) as f64);

    let k_lo = (f_lo / res.df).ceil() as usize;
    let top = res.shanks.iter().flat_map(|s| s.psd_db[k_lo..].iter().copied()).fold(f32::MIN, f32::max);
    let (y_hi, y_lo) = (top + 3.0, top - 60.0);
    let y_of = |db: f32| plot.bottom() - ((db.clamp(y_lo, y_hi) - y_lo) / (y_hi - y_lo)) * plot.height();

    let text = ui.visuals().weak_text_color();
    for f in crate::render::spectrum_ticks(f_lo as f32, f_hi as f32, false) {
        let x = x_of(f as f64);
        painter.line_segment([egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())], egui::Stroke::new(0.5_f32, text.gamma_multiply(0.4)));
        painter.text(egui::pos2(x, rect.bottom() - 1.0), egui::Align2::CENTER_BOTTOM, crate::render::spectrum_tick_label(f), egui::FontId::proportional(10.0), text);
    }

    // one point per pixel column: the column's maximum keeps narrow peaks visible
    let n_cols = plot.width().max(1.0) as usize;
    for (i, s) in res.shanks.iter().enumerate() {
        let t = if res.shanks.len() > 1 { i as f32 / (res.shanks.len() - 1) as f32 } else { 0.8 };
        let [r, g, b] = crate::render::spectrum_color(t);
        let mut pts = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let x0 = plot.left() + c as f32;
            let (ka, kb) = ((f_of(x0) / res.df) as usize, ((f_of(x0 + 1.0) / res.df).ceil() as usize).min(n_bins - 1));
            if let Some(v) = s.psd_db[ka.min(kb)..=kb].iter().copied().reduce(f32::max) {
                pts.push(egui::pos2(x0, y_of(v)));
            }
        }
        painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(r, g, b))));
        if res.shanks.len() > 1 {
            painter.text(
                egui::pos2(rect.right() - 4.0, rect.top() + 1.0 + 11.0 * i as f32),
                egui::Align2::RIGHT_TOP,
                format!("shank {}", s.shank),
                egui::FontId::proportional(10.0),
                egui::Color32::from_rgb(r, g, b),
            );
        }
    }
    for p in &res.peaks {
        let x = x_of(p.freq_hz);
        let on = draft.iter().any(|n| (n.freq_hz - p.freq_hz).abs() <= 1.5 * res.df);
        let col = if on { egui::Color32::from_rgb(0xff, 0x66, 0x66) } else { text };
        painter.line_segment([egui::pos2(x, plot.top() - 10.0), egui::pos2(x, plot.top() - 2.0)], egui::Stroke::new(if on { 2.0_f32 } else { 1.0 }, col));
    }
    if let Some(pos) = resp.hover_pos() {
        if plot.x_range().contains(pos.x) {
            painter.text(rect.left_top() + egui::vec2(4.0, 1.0), egui::Align2::LEFT_TOP, format!("{:.1} Hz", f_of(pos.x)), egui::FontId::proportional(10.0), text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(f: f64, fs: f64, n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| amp * (2.0 * std::f64::consts::PI * f * i as f64 / fs).sin() as f32).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn notch_removes_its_line_and_keeps_others() {
        let fs = 30_000.0;
        let n = 90_000;
        let notches = [Notch { freq_hz: 50.0, bw_hz: 1.0 }, Notch { freq_hz: 3000.0, bw_hz: 2.0 }];
        // away from the edges (settling) the lines are gone
        let settle = (settle_s(&PreprocConfig { notches: notches.to_vec(), notch_enabled: true, ..test_cfg(fs) }) * fs) as usize;
        for f in [50.0, 3000.0] {
            let mut x = sine(f, fs, n, 100.0);
            apply_notches(&mut x, n, &notches, fs);
            assert!(rms(&x[settle..n - settle]) < 2.0, "{f} Hz left: {}", rms(&x[settle..n - settle]));
        }
        let mut x = sine(1000.0, fs, n, 100.0);
        apply_notches(&mut x, n, &notches, fs);
        assert!((rms(&x[settle..n - settle]) - 70.7).abs() < 0.5);
    }

    fn test_cfg(fs: f64) -> PreprocConfig {
        PreprocConfig {
            dc_removal: false,
            phase_shift: false,
            highpass: false,
            spatial_filter: crate::preprocess::SpatialFilter::Off,
            avg_depths: false,
            sample_rate: fs,
            removed_channels: Default::default(),
            channel_order: crate::data::ChannelOrder::Depth,
            shank_order: crate::data::ShankOrder::Id,
            notches: Vec::new(),
            notch_enabled: false,
        }
    }

    #[test]
    fn peak_finder_reports_narrow_lines_only() {
        // synthetic spectrum: 1/f slope, a narrow line at 60 Hz, a broad bump at 8 Hz
        let df = 0.5;
        let n = 4001;
        let db: Vec<f32> = (0..n)
            .map(|k| {
                let f = k as f64 * df;
                let mut v = -10.0 * (f.max(0.5)).log10() as f32;
                v += 20.0 * (-((f - 60.0) / 0.4).powi(2)).exp() as f32;
                v += 10.0 * (-((f - 8.0) / 3.0).powi(2)).exp() as f32;
                v
            })
            .collect();
        let base = running_median_baseline(&db, df);
        let s = ScanSettings::default();
        let peaks = find_peaks(&db, &base, df, 2, n - 2, &s);
        assert_eq!(peaks.len(), 1, "{peaks:?}");
        assert!((peaks[0].2 - 60.0).abs() < 0.3);
    }

    #[test]
    fn presence_estimate_separates_steady_from_transient() {
        // power 100 in 10 of 100 chunks, 0 otherwise -> present 10 %
        let (s1, s2) = (10.0 * 100.0, 10.0 * 100.0f64 * 100.0);
        assert!((presence_from_moments(s1, s2, 100.0) - 0.1).abs() < 1e-6);
        // constant power -> 100 %
        assert!((presence_from_moments(100.0 * 5.0, 100.0 * 25.0, 100.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn evenly_spaced_lines_form_one_series() {
        // a carrier with ±k × 50 Hz sidebands (10943.3 missing), plus two unrelated lines
        let f = [5004.2, 9709.8, 10893.3, 10993.3, 11043.3, 11093.3, 11143.3, 11193.3, 11243.3];
        let g = group_series(&f, 1.4, 2.7);
        assert_eq!(g, vec![vec![0], vec![1], vec![2, 3, 4, 5, 6, 7, 8]]);
        // two lines alone are no series
        assert_eq!(group_series(&[50.0, 100.0], 1.4, 2.7).len(), 2);
    }
}
