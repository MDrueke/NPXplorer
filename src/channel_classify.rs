// Port of IBL's automated channel-QC algorithm (ibldsp/voltage.py::detect_bad_channels,
// as used by iblsorter/preprocess.py::get_good_channels). Classifies each channel from
// a handful of short raw-data snippets spread across the recording, then takes the
// per-channel majority vote. Labels: 0 good, 1 dead, 2 noisy, 3 outside of the brain.

use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::data::{Meta, RawData};
use crate::preprocess::{butter_highpass_sos, sosfiltfilt_inplace};

pub const DEFAULT_N_CLASSIFY_CHUNKS: usize = 100;
pub const CLASSIFY_CHUNK_DUR_S: f64 = 0.3;

const SIMILARITY_LOW: f32 = -0.5; // below this: dead
const SIMILARITY_HIGH: f32 = 1.0; // above this: noisy
const DETREND_NMED: usize = 11;

// ---------------------------------------------------------------------------
// Small hand-rolled radix-2 FFT (nperseg is always a power of two here), so
// the Welch PSD feature below doesn't need an external DSP crate — same
// rationale as the hand-written Butterworth/SOS filters in preprocess.rs.
// ---------------------------------------------------------------------------

fn fft_radix2(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());

    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut cwr = 1.0f32;
            let mut cwi = 0.0f32;
            for k in 0..half {
                let ur = re[i + k];
                let ui = im[i + k];
                let vr = re[i + k + half] * cwr - im[i + k + half] * cwi;
                let vi = re[i + k + half] * cwi + im[i + k + half] * cwr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + half] = ur - vr;
                im[i + k + half] = ui - vi;
                let ncwr = cwr * wr - cwi * wi;
                let ncwi = cwr * wi + cwi * wr;
                cwr = ncwr;
                cwi = ncwi;
            }
            i += len;
        }
        len <<= 1;
    }
}

// ---------------------------------------------------------------------------
// Welch PSD, restricted to the mean power above 80% Nyquist (the only part of
// the spectrum the classifier needs). Matches scipy.signal.welch's defaults:
// periodic Hann window, noverlap = nperseg/2, per-segment mean removed,
// one-sided density scaling with all bins but DC/Nyquist doubled.
// ---------------------------------------------------------------------------

const NPERSEG_DEFAULT: usize = 256;

struct WelchPlan {
    nperseg: usize,
    noverlap: usize,
    window: Vec<f32>,
    win_sum_sq: f32,
    hf_start_bin: usize,
    n_bins: usize,
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

fn build_welch_plan(ns: usize, fs: f64) -> WelchPlan {
    let nperseg = if ns >= NPERSEG_DEFAULT {
        NPERSEG_DEFAULT
    } else {
        let mut p = 1usize;
        while p * 2 <= ns {
            p *= 2;
        }
        p.max(2)
    };
    let noverlap = nperseg / 2;
    let window = hann_window(nperseg);
    let win_sum_sq: f32 = window.iter().map(|w| w * w).sum();
    let n_bins = nperseg / 2 + 1;
    let hf_cut = (fs / 2.0) * 0.8;
    let hf_start_bin = (0..n_bins)
        .find(|&k| (k as f64) * fs / (nperseg as f64) > hf_cut)
        .unwrap_or(n_bins);
    WelchPlan { nperseg, noverlap, window, win_sum_sq, hf_start_bin, n_bins }
}

fn welch_psd_hf(chan: &[f32], fs: f64, plan: &WelchPlan) -> f32 {
    let ns = chan.len();
    let nperseg = plan.nperseg;
    if ns < nperseg || plan.hf_start_bin >= plan.n_bins {
        return 0.0;
    }
    let step = nperseg - plan.noverlap;
    let mut accum = vec![0f32; plan.n_bins];
    let mut re = vec![0f32; nperseg];
    let mut im = vec![0f32; nperseg];
    let mut n_seg = 0usize;
    let mut start = 0;
    while start + nperseg <= ns {
        let seg = &chan[start..start + nperseg];
        let mean = seg.iter().sum::<f32>() / nperseg as f32;
        for i in 0..nperseg {
            re[i] = (seg[i] - mean) * plan.window[i];
            im[i] = 0.0;
        }
        fft_radix2(&mut re, &mut im);
        for k in 0..plan.n_bins {
            let mag2 = re[k] * re[k] + im[k] * im[k];
            let mut p = mag2 / (fs as f32 * plan.win_sum_sq);
            let is_nyquist = nperseg % 2 == 0 && k == plan.n_bins - 1;
            if k != 0 && !is_nyquist {
                p *= 2.0;
            }
            accum[k] += p;
        }
        n_seg += 1;
        start += step;
    }
    if n_seg == 0 {
        return 0.0;
    }
    let hf_bins = &accum[plan.hf_start_bin..plan.n_bins];
    if hf_bins.is_empty() {
        return 0.0;
    }
    hf_bins.iter().map(|&v| v / n_seg as f32).sum::<f32>() / hf_bins.len() as f32
}

// ---------------------------------------------------------------------------
// Per-time-sample median across channels (the "reference trace"), same
// select_nth_unstable approach as apply_global_cmr in preprocess.rs.
// ---------------------------------------------------------------------------

fn median_trace(raw: &[f32], nc: usize, ns: usize) -> Vec<f32> {
    let mut out = vec![0f32; ns];
    out.par_iter_mut().enumerate().for_each(|(t, o)| {
        let mut col: Vec<f32> = (0..nc).map(|ch| raw[ch * ns + t]).collect();
        let half = nc / 2;
        col.select_nth_unstable_by(half, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        *o = if nc % 2 == 0 {
            let upper = col[half];
            col[..half].select_nth_unstable_by(half - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            (upper + col[half - 1]) / 2.0
        } else {
            col[half]
        };
    });
    out
}

/// Zero-lag cross-correlation of each channel with the median reference trace.
/// The IBL reference computes this via an FFT circular cross-correlation and
/// only ever reads the zero-lag term back out, which collapses to a plain dot
/// product — so no FFT is needed for this part.
fn channels_similarity(raw: &[f32], nc: usize, ns: usize) -> Vec<f32> {
    let ref_trace = median_trace(raw, nc, ns);
    let ref_mean = ref_trace.iter().sum::<f32>() / ns as f32;
    let ref_dm: Vec<f32> = ref_trace.iter().map(|&v| v - ref_mean).collect();
    let denom: f32 = ref_dm.iter().map(|&v| v * v).sum::<f32>().max(1e-20);

    (0..nc)
        .into_par_iter()
        .map(|ch| {
            let row = &raw[ch * ns..(ch + 1) * ns];
            let mean = row.iter().sum::<f32>() / ns as f32;
            let numer: f32 = row.iter().zip(ref_dm.iter()).map(|(&x, &r)| (x - mean) * r).sum();
            numer / denom
        })
        .collect()
}

/// Subtract an `nmed`-point median filter (edge-tapered by repeating the
/// first/last value), matching ibldsp.voltage.detect_bad_channels's detrend().
fn detrend(x: &[f32], nmed: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let ntap = (nmed as f32 / 2.0).ceil() as usize;
    let mut xf = vec![x[0]; ntap];
    xf.extend_from_slice(x);
    xf.extend(std::iter::repeat(*x.last().unwrap()).take(ntap));

    let half = nmed / 2;
    (0..n)
        .map(|i| {
            let center = i + ntap;
            let lo = center.saturating_sub(half);
            let hi = (center + half + 1).min(xf.len());
            let mut w: Vec<f32> = xf[lo..hi].to_vec();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            x[i] - w[w.len() / 2]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Single-chunk classification
// ---------------------------------------------------------------------------

/// Classify one raw chunk (`[nc][ns]` µV, un-preprocessed). Returns a label
/// per channel: 0 good, 1 dead, 2 noisy, 3 outside of the brain.
fn classify_chunk(raw_in: &[f32], nc: usize, ns: usize, fs: f64) -> Vec<u8> {
    let mut raw = raw_in.to_vec();
    raw.par_chunks_mut(ns).for_each(|row| {
        let mean = row.iter().sum::<f32>() / ns as f32;
        row.iter_mut().for_each(|v| *v -= mean);
    });

    let xcor = channels_similarity(&raw, nc, ns);

    let is_ap = fs > 2600.0;
    let psd_hf_threshold = if is_ap { 0.02 } else { 1.4 };
    let wn = if is_ap { 300.0 / fs * 2.0 } else { 1.0 / fs * 2.0 };
    let sos = butter_highpass_sos(3, wn);

    let mut hf = raw.clone();
    hf.par_chunks_mut(ns).for_each(|row| {
        sosfiltfilt_inplace(&sos, row);
    });
    let xcorf = channels_similarity(&hf, nc, ns);

    let xcor_hf = detrend(&xcor, DETREND_NMED);
    let xcorf_detrend = detrend(&xcorf, DETREND_NMED);
    let xcor_lf: Vec<f32> = xcorf.iter().zip(xcorf_detrend.iter()).map(|(&a, &b)| a - b - 1.0).collect();

    // psd_hf is computed on the DC-removed raw signal, not the highpassed one
    // (matches the Python reference, which calls welch() on `raw`, not `hf`)
    let plan = build_welch_plan(ns, fs);
    let psd_hf: Vec<f32> = (0..nc)
        .into_par_iter()
        .map(|ch| welch_psd_hf(&raw[ch * ns..(ch + 1) * ns], fs, &plan))
        .collect();

    let mut labels = vec![0u8; nc];
    for ch in 0..nc {
        if xcor_hf[ch] < SIMILARITY_LOW {
            labels[ch] = 1;
        }
    }
    for ch in 0..nc {
        if psd_hf[ch] > psd_hf_threshold || xcor_hf[ch] > SIMILARITY_HIGH {
            labels[ch] = 2;
        }
    }

    // outside-of-brain: contiguous run of low-xcor_lf channels touching the
    // end of the probe (uppermost channels), detected from a smoothed trend
    if nc >= 2 {
        let window_size = 25usize;
        let half_k = (window_size - 1) / 2;
        let signal_filtered: Vec<f32> = (0..nc)
            .map(|i| {
                let mut sum = 0f32;
                for k in 0..window_size {
                    let idx = i as isize - half_k as isize + k as isize;
                    if idx >= 0 && (idx as usize) < nc {
                        sum += xcor_lf[idx as usize];
                    }
                }
                sum / window_size as f32
            })
            .collect();

        let diff_x: Vec<f32> = (1..nc).map(|i| signal_filtered[i] - signal_filtered[i - 1]).collect();
        let indx: Vec<usize> = diff_x
            .iter()
            .enumerate()
            .filter(|(_, &d)| d < -0.02)
            .map(|(i, _)| i)
            .collect();

        if !indx.is_empty() {
            let m = indx.len() / 2;
            let median_idx = if indx.len() % 2 == 1 {
                indx[m] as f64
            } else {
                (indx[m - 1] + indx[m]) as f64 / 2.0
            };
            let indx_threshold = (median_idx.floor() as usize).min(nc - 1);
            let threshold = xcor_lf[indx_threshold];
            let ioutside: Vec<usize> = (0..nc).filter(|&i| xcor_lf[i] < threshold).collect();

            if let Some(&last) = ioutside.last() {
                if last == nc - 1 {
                    // keep only the run contiguous with the end of the probe
                    let mut a = vec![0i64; ioutside.len()];
                    let mut acc = 0i64;
                    for i in 1..ioutside.len() {
                        acc += (ioutside[i] as i64 - ioutside[i - 1] as i64) - 1;
                        a[i] = acc;
                    }
                    let max_a = *a.iter().max().unwrap_or(&0);
                    let kept: Vec<usize> = ioutside
                        .into_iter()
                        .zip(a.iter())
                        .filter(|(_, &v)| v == max_a)
                        .map(|(i, _)| i)
                        .collect();
                    for i in kept {
                        labels[i] = 3;
                    }
                }
            }
        }
    }

    labels
}

// ---------------------------------------------------------------------------
// Whole-recording classification: majority vote across N_CLASSIFY_CHUNKS
// evenly-spaced chunks. Runs on whatever thread calls it — the caller (app.rs)
// spawns this on a background thread and polls `progress`/`cancel`.
// ---------------------------------------------------------------------------

/// Returns `None` if cancelled partway through.
pub fn classify_recording(
    raw: &RawData,
    meta: &Meta,
    n_chunks: usize,
    cancel: &AtomicBool,
    progress: &AtomicUsize,
) -> Option<Vec<u8>> {
    let nc = meta.n_ap_chans;
    let fs = meta.sample_rate;
    let total_dur = meta.n_samples as f64 / fs;
    let max_t0 = (total_dur - CLASSIFY_CHUNK_DUR_S).max(0.0);

    let mut votes = vec![[0u32; 4]; nc];

    for i in 0..n_chunks {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let t0 = if n_chunks <= 1 {
            0.0
        } else {
            max_t0 * i as f64 / (n_chunks - 1) as f64
        };
        let first_sample = (t0 * fs) as usize;
        let n_samp = ((CLASSIFY_CHUNK_DUR_S * fs) as usize).min(meta.n_samples.saturating_sub(first_sample));

        if n_samp > 0 {
            let chunk = raw.read_chunk_uv(first_sample, n_samp, meta);
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let labels = classify_chunk(&chunk, nc, n_samp, fs);
            for (ch, &l) in labels.iter().enumerate() {
                votes[ch][l as usize] += 1;
            }
        }
        progress.fetch_add(1, Ordering::Relaxed);
    }

    let final_labels: Vec<u8> = votes
        .iter()
        .map(|counts| {
            let mut best_label = 0u8;
            let mut best_count = counts[0];
            for l in 1..4 {
                if counts[l] > best_count {
                    best_count = counts[l];
                    best_label = l as u8;
                }
            }
            best_label
        })
        .collect();

    Some(final_labels)
}
