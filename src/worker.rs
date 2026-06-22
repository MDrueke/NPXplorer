
macro_rules! file_log {
    ($($arg:tt)*) => {
        {
            let log_path = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("debug.log")))
                .unwrap_or_else(|| std::path::PathBuf::from("debug.log"));
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log_path) {
                use std::io::Write;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                let _ = write!(f, "[{:.3}] ", now);
                let _ = writeln!(f, $($arg)*);
            }
        }
    };
}

use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use crate::data::{DisplayRow, Meta, RawData};
use crate::preprocess::{Filters, PreprocConfig, preprocess};

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

#[derive(Clone)]
pub struct WorkerRequest {
    pub center_sample: usize,
    pub half_window: usize,
    pub cfg: PreprocConfig,
}

pub type SharedWorkerState = Arc<(Mutex<WorkerState>, Condvar)>;
pub type SharedCancel = Arc<AtomicBool>;

// ---------------------------------------------------------------------------
// Depth averaging — averages same-row channel pairs before preprocessing
// ---------------------------------------------------------------------------

/// Given raw data [n_ap][n_samp] and display_rows, produce averaged data [n_data_rows][n_samp].
fn average_depth_rows(
    raw: &[f32],
    n_samp: usize,
    display_rows: &[DisplayRow],
) -> Vec<f32> {
    use rayon::prelude::*;
    let n_data_rows = display_rows.iter().filter(|r| matches!(r, DisplayRow::Data { .. })).count();
    let mut out = vec![0.0f32; n_data_rows * n_samp];

    // collect only Data rows in order
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
// Percentile table — sample |data| and compute 0..=100 percentiles
// ---------------------------------------------------------------------------

fn compute_pct_table(data: &[f32]) -> PctTable {
    // sub-sample to at most 2M values for speed
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
// Compute the dynamic half-window size based on available RAM
// ---------------------------------------------------------------------------

pub fn compute_half_window(view_dur_s: f64, sample_rate: f64) -> usize {
    // We want the total buffer to be view_dur_s plus 2.5s padding on each side.
    // The half_window is computed around the center.
    ((view_dur_s / 2.0 + 2.5) * sample_rate) as usize
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

pub fn spawn_worker(
    raw: Arc<RawData>,
    meta: Arc<Meta>,
    filt: Arc<Mutex<Filters>>,
    shared: SharedWorkerState,
    cancel: SharedCancel,
    ctx: egui::Context,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(2).max(2))
            .unwrap_or(2);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();

        let (lock, cvar) = &*shared;
        loop {
            // wait for a request
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

            // reset cancel flag for this newly grabbed request
            cancel.store(false, Ordering::Relaxed);
            file_log!("WORKER: Picked up request center={} half_window={} cfg={:?}", req.center_sample, req.half_window, req.cfg);

            // perform heavy computation within the dedicated thread pool
            pool.install(|| {

            // build display rows for this config
            let display_rows = Arc::new(meta.build_display_rows(req.cfg.avg_depths));

            let first = req.center_sample.saturating_sub(req.half_window);
            let n_samp = (req.half_window * 2)
                .min(meta.n_samples.saturating_sub(first));

            // read raw chunk [n_ap][n_samp]
            let raw_chunk = raw.read_chunk_uv(first, n_samp, &meta);
            if cancel.load(Ordering::Relaxed) {
                file_log!("WORKER: Cancelled after read_chunk_uv");
                let mut st = lock.lock().unwrap();
                st.status = WorkerStatus::Idle;
                st.active_request = None;
                return;
            }

            // average_depth_rows correctly reorders raw_chunk into display_rows order
            // If avg_depths is false, it still reorders the single-channel groups correctly.
            let mut data = average_depth_rows(&raw_chunk, n_samp, &display_rows);

            if cancel.load(Ordering::Relaxed) {
                let mut st = lock.lock().unwrap();
                st.status = WorkerStatus::Idle;
                st.active_request = None;
                return;
            }

            // preprocess in-place
            let filt_g = filt.lock().unwrap().clone();
            file_log!("WORKER: Starting preprocess, data.len()={}, n_samp={}, display_rows={}", data.len(), n_samp, display_rows.len());
            preprocess(&mut data, n_samp, &req.cfg, &filt_g, &cancel, &display_rows);

            if cancel.load(Ordering::Relaxed) {
                let mut st = lock.lock().unwrap();
                st.status = WorkerStatus::Idle;
                st.active_request = None;
                return;
            }

            // compute percentile table
            let vmax_pct = compute_pct_table(&data);

            let buf = PreprocBuffer {
                first_sample: first,
                n_samp,
                data: Arc::new(data),
                cfg: req.cfg,
                display_rows,
                vmax_pct,
            };

            let mut st = lock.lock().unwrap();
            st.active_request = None;
            if st.request.is_some() {
                file_log!("WORKER: Finished but discarded — new request waiting");
                // newer request arrived — discard this result
                st.status = WorkerStatus::Idle;
            } else {
                file_log!("WORKER: Done. buf first={}, n_samp={}, data.len()={}, display_rows={}", buf.first_sample, buf.n_samp, buf.data.len(), buf.display_rows.len());
                st.buffer = Some(buf);
                st.status = WorkerStatus::Done;
                ctx.request_repaint(); // Wake UI immediately
            }
        }); // end pool.install

        }
    })
}
