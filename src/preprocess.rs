use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// SOS biquad — f32 arithmetic
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Sos {
    pub b0: f32, pub b1: f32, pub b2: f32,
    pub a1: f32, pub a2: f32,
}

pub fn sosfilt_inplace(sos: &[Sos], x: &mut [f32]) {
    for s in sos {
        let (b0, b1, b2, a1, a2) = (s.b0, s.b1, s.b2, s.a1, s.a2);
        let (mut z0, mut z1) = (0.0f32, 0.0f32);
        for v in x.iter_mut() {
            let xi = *v;
            let yi = b0 * xi + z0;
            z0 = b1 * xi - a1 * yi + z1;
            z1 = b2 * xi - a2 * yi;
            *v = yi;
        }
    }
}

pub fn sosfiltfilt_inplace(sos: &[Sos], x: &mut [f32]) {
    sosfilt_inplace(sos, x);
    x.reverse();
    sosfilt_inplace(sos, x);
    x.reverse();
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
    pub phase_shift: bool,
    pub highpass: bool,
    pub spatial_filter: SpatialFilter,
    pub avg_depths: bool,
    pub sample_rate: f64,
    pub im_dat_prb_type: u32,
}

#[derive(Clone)]
pub struct Filters {
    pub hp_sos: Vec<Sos>,
    pub kfilt_sos: Vec<Sos>,
    pub kfilt_lagc: usize,
}

impl Filters {
    pub fn new(cfg: &PreprocConfig) -> Self {
        let fs = cfg.sample_rate;
        Filters {
            hp_sos: butter_highpass_sos(3, 300.0 / fs * 2.0),
            kfilt_sos: butter_highpass_sos(3, 0.01),
            kfilt_lagc: (fs / 10.0).round() as usize,
        }
    }
}

use crate::data::DisplayRow;

// ---------------------------------------------------------------------------
// Top-level entry point
// Order: depth-averaging is done by the caller before this function.
// Here: DC offset -> Phase shift -> Temporal HP -> Spatial filter
// ---------------------------------------------------------------------------

pub fn preprocess(
    data: &mut [f32],
    n_samp: usize,
    cfg: &PreprocConfig,
    filt: &Filters,
    cancel: &AtomicBool,
    display_rows: &[DisplayRow],
) {
    if display_rows.is_empty() || n_samp == 0 { return; }

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
        if cancel.load(Ordering::Relaxed) { return; }
        
        let n_shank_rows = end_row - start_row;
        let n_shank_samples = n_shank_rows * n_samp;
        
        let shank_data = &mut data[current_data_offset .. current_data_offset + n_shank_samples];
        let shank_data_rows = &data_rows[start_row .. end_row];
        
        // 1. DC Offset Correction
        if cfg.dc_removal {
            apply_dc_removal(shank_data, n_samp);
        }
        if cancel.load(Ordering::Relaxed) { return; }

        // 2. Phase Shift Correction
        if cfg.phase_shift {
            apply_phase_shift(shank_data, n_samp, shank_data_rows, cfg.im_dat_prb_type);
        }
        if cancel.load(Ordering::Relaxed) { return; }

        // 3. Highpass Filter
        if cfg.highpass {
            apply_temporal_hp(shank_data, n_samp, &filt.hp_sos);
        }
        if cancel.load(Ordering::Relaxed) { return; }

        // 4. Spatial Filter
        match cfg.spatial_filter {
            SpatialFilter::Off => {}
            SpatialFilter::GlobalCmr => apply_global_cmr(shank_data, n_shank_rows, n_samp),
            SpatialFilter::LocalCmr => apply_local_cmr(shank_data, n_samp, shank_data_rows),
            SpatialFilter::Destripe => {
                apply_kfilt(shank_data, n_shank_rows, n_samp, filt, cancel);
            }
        }
        
        current_data_offset += n_shank_samples;
    }
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

// ---------------------------------------------------------------------------
// Phase Shift Correction (Fractional delay via FFT)
// ---------------------------------------------------------------------------

fn apply_phase_shift(data: &mut [f32], n_samp: usize, data_rows: &[&DisplayRow], im_dat_prb_type: u32) {
    data.par_chunks_mut(n_samp).enumerate().for_each(|(row_idx, row)| {
        if let DisplayRow::Data { first_ch, .. } = data_rows[row_idx] {
            let shift_samples = match im_dat_prb_type {
                21 | 24 | 2013 | 2014 => (*first_ch % 384 / 24) as f32 / 16.0,
                _ => (*first_ch % 384 / 32) as f32 / 13.0,
            };
            
            if shift_samples == 0.0 { return; }

            let delta = shift_samples;
            let inv_delta = 1.0 - delta;
            
            for t in (1..n_samp).rev() {
                row[t] = row[t] * inv_delta + row[t - 1] * delta;
            }
            // For t=0, we don't have row[-1], so keep row[0] unchanged or scale it
        }
    });
}

use std::cell::RefCell;

thread_local! {
    static LOCAL_CMR_BUF: RefCell<(Vec<f32>, Vec<f32>)> = RefCell::new((Vec::new(), Vec::new()));
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

fn apply_temporal_hp(data: &mut [f32], n_samp: usize, sos: &[Sos]) {
    data.par_chunks_mut(n_samp).for_each(|ch| {
        sosfiltfilt_inplace(sos, ch);
    });
}

// ---------------------------------------------------------------------------
// Global CMR
// ---------------------------------------------------------------------------

thread_local! {
    static CMR_COL: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

fn apply_global_cmr(data: &mut [f32], n_rows: usize, n_samp: usize) {
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

fn compute_agc_gain(data: &[f32], n_rows: usize, n_samp: usize, win: usize) -> Vec<f32> {
    let n_total = n_rows * n_samp;
    let mean = data.iter().sum::<f32>() / n_total as f32;
    let var = data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n_total as f32;
    let epsilon = (var.sqrt() * 0.003).max(1e-8f32);

    let half = win / 2;
    let mut gain = vec![epsilon; n_rows * n_samp];

    gain.par_chunks_mut(n_samp).enumerate().for_each(|(ch, g)| {
        let src = &data[ch * n_samp..(ch + 1) * n_samp];
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

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

fn apply_kfilt(data: &mut [f32], n_rows: usize, n_samp: usize, filt: &Filters, cancel: &AtomicBool) {
    let pad = 60usize.min(n_rows);
    let n_padded = n_rows + 2 * pad;
    let gain = compute_agc_gain(data, n_rows, n_samp, filt.kfilt_lagc);

    // divide by gain
    for (ch, g) in gain.chunks(n_samp).enumerate() {
        let row = &mut data[ch * n_samp..(ch + 1) * n_samp];
        for (v, &gv) in row.iter_mut().zip(g.iter()) {
            *v /= gv;
        }
    }

    if cancel.load(Ordering::Relaxed) { return; }

    let data_ptr = SendPtr(data.as_mut_ptr());
    let gain_ptr = SendPtr(gain.as_ptr() as *mut f32);
    let sos = filt.kfilt_sos.clone();

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

            // Read row by row for contiguous memory access
            for ch in 0..n_rows {
                let src_row = unsafe { std::slice::from_raw_parts(dp.add(ch * n_samp + t_start), n_t) };
                let dst_row = &mut buf[(pad + ch) * n_t .. (pad + ch + 1) * n_t];
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
            for i in 0..n_t {
                for r in 0..n_padded {
                    col[r] = buf[r * n_t + i];
                }
                sosfiltfilt_inplace(&sos, &mut col);
                for r in 0..n_padded {
                    buf[r * n_t + i] = col[r];
                }
            }

            // Write back row by row
            for ch in 0..n_rows {
                let dst_row = unsafe { std::slice::from_raw_parts_mut(dp.add(ch * n_samp + t_start), n_t) };
                let src_row = &buf[(pad + ch) * n_t .. (pad + ch + 1) * n_t];
                let g_row = unsafe { std::slice::from_raw_parts(gp.add(ch * n_samp + t_start), n_t) };
                for i in 0..n_t {
                    dst_row[i] = src_row[i] * g_row[i];
                }
            }
        });
    });
}
