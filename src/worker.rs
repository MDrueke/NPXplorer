use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use crate::data::{DisplayRow, Meta, RawData};
use crate::preprocess::{Filters, PreprocConfig, preprocess};

// overlap samples used when stitching an extension chunk — gives the temporal
// highpass enough data to settle before the new samples we actually keep
pub const EXTENSION_OVERLAP_SAMP: usize = 2000;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WorkerStatus {
    Idle,
    Computing,
    Done,
}

/// Percentile lookup table: vmax_pct[p] = p-th percentile of |data| (p = 0..=100.0, 0.01 steps).
pub type PctTable = Box<[f32; 10001]>;

pub struct PreprocBuffer {
    pub first_sample: usize,
    pub n_samp: usize,
    /// Layout: data[row_idx * n_samp .. (row_idx+1) * n_samp], µV
    pub data: Arc<Vec<f32>>,
    pub cfg: PreprocConfig,
    pub display_rows: Arc<Vec<DisplayRow>>,
    /// Percentile table of |data| values (0..=100.0).
    pub vmax_pct: PctTable,
}

pub struct WorkerState {
    pub buffer: Option<PreprocBuffer>,
    pub status: WorkerStatus,
    pub request: Option<WorkerRequest>,
    pub active_request: Option<WorkerRequest>,
}

impl WorkerState {
    pub fn new() -> Self {
        WorkerState { buffer: None, status: WorkerStatus::Idle, request: None, active_request: None }
    }
}

#[derive(Clone, Debug)]
pub enum RequestKind {
    Full { center_sample: usize, half_window: usize },
    Extend {
        direction: i32,
        extension_samp: usize,
        view_first: usize,
        view_n: usize,
        /// cap on total buffer size (samples) — growth stops (net-zero) once reached
        max_buffer_samp: usize,
        /// trigger net-zero growth early if system-available memory drops below this
        /// percentage of total (0-100)
        mem_pressure_pct: f32,
        /// ...or below this absolute number of free bytes, whichever hits first
        mem_reserve_bytes: u64,
    },
}

#[derive(Clone, Debug)]
pub struct WorkerRequest {
    pub kind: RequestKind,
    pub cfg: PreprocConfig,
}

pub type SharedWorkerState = Arc<(Mutex<WorkerState>, Condvar)>;
pub type SharedCancel = Arc<AtomicBool>;

// ---------------------------------------------------------------------------
// Depth averaging
// ---------------------------------------------------------------------------

pub(crate) fn average_depth_rows(raw: &[f32], n_samp: usize, display_rows: &[DisplayRow]) -> Vec<f32> {
    use rayon::prelude::*;
    let n_data_rows = display_rows.iter().filter(|r| matches!(r, DisplayRow::Data { .. })).count();
    let mut out = vec![0.0f32; n_data_rows * n_samp];

    let data_rows: Vec<&DisplayRow> = display_rows.iter()
        .filter(|r| matches!(r, DisplayRow::Data { .. }))
        .collect();

    out.par_chunks_mut(n_samp).enumerate().for_each(|(row_idx, dst)| {
        if let DisplayRow::Data { channels, .. } = data_rows[row_idx] {
            let n = channels.len() as f32;
            for t in 0..n_samp {
                let sum: f32 = channels.iter().map(|&ch| raw[ch * n_samp + t]).sum();
                dst[t] = sum / n;
            }
        }
    });
    out
}

// ---------------------------------------------------------------------------
// Percentile table
// ---------------------------------------------------------------------------

fn compute_pct_table(data: &[f32]) -> PctTable {
    let step = (data.len() / 2_000_000).max(1);
    let mut vals: Vec<f32> = data.iter().step_by(step).map(|v| v.abs()).collect();
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    let mut table = Box::new([0.0f32; 10001]);
    for p in 0..=10000usize {
        let idx = ((p * (n - 1)) / 10000).min(n - 1);
        table[p] = vals[idx];
    }
    table
}

// ---------------------------------------------------------------------------
// Half-window size
// ---------------------------------------------------------------------------

/// Half of the initial/full-recompute buffer size, in samples. `initial_buffer_s` is
/// the total width of that buffer (view sits at its center); this also doubles as the
/// steady-state cap that incremental extension growth is trimmed back to.
pub fn compute_half_window(initial_buffer_s: f64, sample_rate: f64) -> usize {
    (initial_buffer_s / 2.0 * sample_rate) as usize
}

// ---------------------------------------------------------------------------
// Memory pressure check
// ---------------------------------------------------------------------------

/// True if system-available memory is low by either the percentage or absolute
/// threshold. Used to decide whether buffer growth should stop (net-zero) early.
fn memory_pressure(mem_pressure_pct: f32, mem_reserve_bytes: u64) -> bool {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    let total = sys.total_memory();
    if total == 0 {
        return false; // can't determine — assume no pressure rather than stall growth
    }
    let available = sys.available_memory();
    let pct_free = available as f64 / total as f64 * 100.0;
    pct_free < mem_pressure_pct as f64 || available < mem_reserve_bytes
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

/// Number of worker threads to use for heavy CPU work (preprocessing, PSTH).
/// Leaves 2 cores free for the UI thread and the OS so the app stays responsive.
pub fn compute_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2).max(2))
        .unwrap_or(2)
}

pub fn spawn_worker(
    raw: Arc<RawData>,
    meta: Arc<Meta>,
    filt: Arc<Mutex<Filters>>,
    shared: SharedWorkerState,
    cancel: SharedCancel,
    ctx: egui::Context,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let n_threads = compute_thread_count();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();

        let (lock, cvar) = &*shared;
        loop {
            let req = {
                let mut st = lock.lock().unwrap();
                loop {
                    if let Some(r) = st.request.take() {
                        st.status = WorkerStatus::Computing;
                        st.active_request = Some(r.clone());
                        break r;
                    }
                    st = cvar.wait(st).unwrap();
                }
            };

            cancel.store(false, Ordering::Relaxed);

            pool.install(|| {
                match req.kind.clone() {
                    RequestKind::Full { center_sample, half_window } => {
                        run_full(&req.cfg, center_sample, half_window, &raw, &meta, &filt, &lock, &cancel, &ctx);
                    }
                    RequestKind::Extend { direction, extension_samp, view_first, view_n, max_buffer_samp, mem_pressure_pct, mem_reserve_bytes } => {
                        run_extend(&req.cfg, direction, extension_samp, view_first, view_n, max_buffer_samp, mem_pressure_pct, mem_reserve_bytes, &raw, &meta, &filt, &lock, &cancel, &ctx);
                    }
                }
            });
        }
    })
}

// ---------------------------------------------------------------------------
// Full recompute
// ---------------------------------------------------------------------------

fn run_full(
    cfg: &PreprocConfig,
    center_sample: usize,
    half_window: usize,
    raw: &RawData,
    meta: &Meta,
    filt: &Mutex<Filters>,
    lock: &Mutex<WorkerState>,
    cancel: &AtomicBool,
    ctx: &egui::Context,
) {
    let display_rows = Arc::new(meta.build_display_rows(cfg.avg_depths));

    let first = center_sample.saturating_sub(half_window);
    let n_samp = (half_window * 2).min(meta.n_samples.saturating_sub(first));

    let raw_chunk = raw.read_chunk_uv(first, n_samp, meta);
    if cancel.load(Ordering::Relaxed) {
        finish_cancelled(lock);
        return;
    }

    let mut data = average_depth_rows(&raw_chunk, n_samp, &display_rows);
    if cancel.load(Ordering::Relaxed) {
        finish_cancelled(lock);
        return;
    }

    let filt_g = filt.lock().unwrap().clone();
    preprocess(&mut data, n_samp, cfg, &filt_g, cancel, &display_rows);
    if cancel.load(Ordering::Relaxed) {
        finish_cancelled(lock);
        return;
    }

    let vmax_pct = compute_pct_table(&data);
    let n_data_rows = display_rows.iter().filter(|r| matches!(r, DisplayRow::Data { .. })).count();
    debug_assert_eq!(data.len(), n_data_rows * n_samp);

    let buf = PreprocBuffer {
        first_sample: first,
        n_samp,
        data: Arc::new(data),
        cfg: cfg.clone(),
        display_rows,
        vmax_pct,
    };
    publish(lock, buf, ctx);
}

// ---------------------------------------------------------------------------
// Incremental extension
// ---------------------------------------------------------------------------

fn run_extend(
    cfg: &PreprocConfig,
    direction: i32,
    extension_samp: usize,
    view_first: usize,
    view_n: usize,
    max_buffer_samp: usize,
    mem_pressure_pct: f32,
    mem_reserve_bytes: u64,
    raw: &RawData,
    meta: &Meta,
    filt: &Mutex<Filters>,
    lock: &Mutex<WorkerState>,
    cancel: &AtomicBool,
    ctx: &egui::Context,
) {
    // snapshot current buffer
    let current = {
        let st = lock.lock().unwrap();
        st.buffer.as_ref().map(|b| (
            Arc::clone(&b.data),
            b.first_sample,
            b.n_samp,
            Arc::clone(&b.display_rows),
            b.cfg.clone(),
        ))
    };

    let (old_data, old_first, old_n_samp, display_rows, _old_cfg) = match current {
        Some(v) if v.4 == *cfg => v,
        _ => {
            // no buffer or config changed — app will issue a Full request
            finish_cancelled(lock);
            return;
        }
    };

    let n_data_rows = display_rows.iter()
        .filter(|r| matches!(r, DisplayRow::Data { .. }))
        .count();

    // small fixed safety margin — trimming never eats into this much space behind/ahead
    // of the visible view, regardless of cap or memory pressure (correctness floor)
    let safety_retain = EXTENSION_OVERLAP_SAMP;
    let pressure = memory_pressure(mem_pressure_pct, mem_reserve_bytes);

    let (read_start, read_n, clean_offset, actual_ext, drop_samp, new_first) = if direction > 0 {
        // extend right: read [buf_end - overlap, buf_end + ext]
        let buf_end = old_first + old_n_samp;
        let available = meta.n_samples.saturating_sub(buf_end);
        let ext = extension_samp.min(available);
        if ext == 0 { finish_cancelled(lock); return; }

        let rs = buf_end.saturating_sub(EXTENSION_OVERLAP_SAMP);
        let act_overlap = buf_end - rs;
        let rn = act_overlap + ext;

        // no trim below the cap and without memory pressure (pure growth); once over
        // the cap or under pressure, drop enough for net-zero growth from here on
        let over_cap = (old_n_samp + ext).saturating_sub(max_buffer_samp);
        let target_drop = if pressure { over_cap.max(ext) } else { over_cap };
        let headroom = (view_first as isize - old_first as isize - safety_retain as isize).max(0) as usize;
        let drop = target_drop.min(headroom).min(old_n_samp);

        (rs, rn, act_overlap, ext, drop, old_first + drop)
    } else {
        // extend left: read [buf_first - ext, buf_first + overlap]
        let ext = extension_samp.min(old_first);
        if ext == 0 { finish_cancelled(lock); return; }

        let rs = old_first - ext;
        let act_overlap = EXTENSION_OVERLAP_SAMP.min(old_n_samp);
        let rn = ext + act_overlap;

        let over_cap = (old_n_samp + ext).saturating_sub(max_buffer_samp);
        let target_drop = if pressure { over_cap.max(ext) } else { over_cap };
        let view_end = view_first + view_n;
        let buf_end = old_first + old_n_samp;
        let headroom = (buf_end as isize - view_end as isize - safety_retain as isize).max(0) as usize;
        let drop = target_drop.min(headroom).min(old_n_samp);

        (rs, rn, 0usize, ext, drop, rs)
    };

    // read and preprocess the extension + overlap chunk
    let raw_chunk = raw.read_chunk_uv(read_start, read_n, meta);
    if cancel.load(Ordering::Relaxed) { finish_cancelled(lock); return; }

    let mut ext_proc = average_depth_rows(&raw_chunk, read_n, &display_rows);
    if cancel.load(Ordering::Relaxed) { finish_cancelled(lock); return; }

    let filt_g = filt.lock().unwrap().clone();
    preprocess(&mut ext_proc, read_n, cfg, &filt_g, cancel, &display_rows);
    if cancel.load(Ordering::Relaxed) { finish_cancelled(lock); return; }

    // stitch: shift old buffer + append clean new samples
    let keep_old = old_n_samp - drop_samp;
    let new_n_samp = keep_old + actual_ext;
    let mut new_data = vec![0.0f32; n_data_rows * new_n_samp];

    for r in 0..n_data_rows {
        let new_row = r * new_n_samp;
        if direction > 0 {
            // old (shifted left) | new ext
            let old_src = r * old_n_samp + drop_samp;
            new_data[new_row..new_row + keep_old]
                .copy_from_slice(&old_data[old_src..old_src + keep_old]);
            let ext_src = r * read_n + clean_offset;
            new_data[new_row + keep_old..new_row + new_n_samp]
                .copy_from_slice(&ext_proc[ext_src..ext_src + actual_ext]);
        } else {
            // new ext | old (minus right drop)
            let ext_src = r * read_n + clean_offset;
            new_data[new_row..new_row + actual_ext]
                .copy_from_slice(&ext_proc[ext_src..ext_src + actual_ext]);
            let old_src = r * old_n_samp;
            new_data[new_row + actual_ext..new_row + new_n_samp]
                .copy_from_slice(&old_data[old_src..old_src + keep_old]);
        }
    }

    let vmax_pct = compute_pct_table(&new_data);
    let buf = PreprocBuffer {
        first_sample: new_first,
        n_samp: new_n_samp,
        data: Arc::new(new_data),
        cfg: cfg.clone(),
        display_rows,
        vmax_pct,
    };
    publish(lock, buf, ctx);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn finish_cancelled(lock: &Mutex<WorkerState>) {
    let mut st = lock.lock().unwrap();
    st.status = WorkerStatus::Idle;
    st.active_request = None;
}

fn publish(lock: &Mutex<WorkerState>, buf: PreprocBuffer, ctx: &egui::Context) {
    let mut st = lock.lock().unwrap();
    st.active_request = None;
    if st.request.is_some() {
        st.status = WorkerStatus::Idle;
    } else {
        st.buffer = Some(buf);
        st.status = WorkerStatus::Done;
        ctx.request_repaint();
    }
}
