use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use rayon::prelude::*;

use crate::data::{DisplayRow, Meta, RawData};
use crate::preprocess::{preprocess, Filters, PreprocConfig, SpatialFilter};

// samples read beyond each edge of a block so the filters (filtfilt) settle before
// the samples that are kept
pub const MIN_OVERLAP_SAMP: usize = 2000;
const MIN_BLOCK_S: f64 = 2.0;
// blocks are at least this many overlaps long, so the overlap costs ≤ ~10 % extra work
const BLOCK_PER_OVERLAP: usize = 10;
// background fill reaches this many times further ahead (scroll direction) than behind
const BEHIND_WEIGHT: usize = 3;
// values sampled for the colour-scale percentile
const PCT_MAX_SAMPLES: usize = 200_000;
// i16 storage range relative to the ADC's full range (room for CMR / filter overshoot)
const QUANT_HEADROOM: f64 = 4.0;
// how often an idle worker re-checks free memory (other apps may grow or shrink)
const IDLE_RECHECK: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------------------
// Block cache
// ---------------------------------------------------------------------------

struct Block {
    /// [row][sample], µV = value * step_uv
    data: Vec<i16>,
}

impl Block {
    fn bytes(&self) -> usize {
        self.data.len() * 2
    }
}

static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(0);

/// Preprocessed data for one config, in fixed-size time blocks.
pub struct BlockCache {
    id: u64,
    cfg: PreprocConfig,
    display_rows: Arc<Vec<DisplayRow>>,
    n_rows: usize,
    n_samples: usize,
    block_samp: usize,
    overlap: usize,
    step_uv: f32,
    /// destripe AGC floors of the first block computed — reused by every later block,
    /// so neighbouring blocks get the same floor and no seam appears
    agc_eps: Option<Arc<Vec<f32>>>,
    blocks: HashMap<usize, Arc<Block>>,
    bytes: usize,
}

impl BlockCache {
    fn new(cfg: &PreprocConfig, meta: &Meta, filt: &Filters) -> Self {
        let display_rows = Arc::new(meta.build_display_rows(cfg.avg_depths, &cfg.removed_channels, cfg.channel_order, cfg.shank_order));
        let n_rows = display_rows.iter().filter(|r| matches!(r, DisplayRow::Data { .. })).count();
        let fs = meta.sample_rate;
        let settle_s = crate::notch::settle_s(cfg).max(crate::preprocess::bandpass_settle_s(cfg));
        let mut overlap = MIN_OVERLAP_SAMP.max((settle_s * fs).ceil() as usize);
        if cfg.spatial_filter == SpatialFilter::Destripe {
            overlap = overlap.max(filt.kfilt_lagc);
        }
        let block_samp = ((MIN_BLOCK_S * fs) as usize).max(BLOCK_PER_OVERLAP * overlap);
        let max_uv_per_bit = meta.uv_per_bit.iter().fold(0.0f32, |a, &b| a.max(b)) as f64;
        let step_uv = (max_uv_per_bit * meta.max_int * QUANT_HEADROOM / 32767.0).max(1e-3) as f32;
        BlockCache {
            id: NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            cfg: cfg.clone(),
            display_rows,
            n_rows,
            n_samples: meta.n_samples,
            block_samp,
            overlap,
            step_uv,
            agc_eps: None,
            blocks: HashMap::new(),
            bytes: 0,
        }
    }

    fn n_blocks(&self) -> usize {
        self.n_samples.div_ceil(self.block_samp)
    }

    fn block_len(&self, b: usize) -> usize {
        self.block_samp.min(self.n_samples - b * self.block_samp)
    }

    /// Blocks overlapping samples [first, end).
    fn range(&self, first: usize, end: usize) -> std::ops::Range<usize> {
        let end = end.min(self.n_samples);
        if end <= first {
            return 0..0;
        }
        first / self.block_samp..(end - 1) / self.block_samp + 1
    }

    fn block_bytes(&self) -> usize {
        self.n_rows * self.block_samp * 2
    }

    fn insert(&mut self, b: usize, block: Block) {
        self.bytes += block.bytes();
        if let Some(old) = self.blocks.insert(b, Arc::new(block)) {
            self.bytes -= old.bytes();
        }
    }

    fn evict(&mut self, b: usize) {
        if let Some(old) = self.blocks.remove(&b) {
            self.bytes -= old.bytes();
        }
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// What the UI wants preprocessed; posted every frame it changes.
#[derive(Clone, PartialEq, Debug)]
pub struct Plan {
    pub cfg: PreprocConfig,
    pub view_first: usize,
    pub view_n: usize,
    /// last scroll direction: +1 forward, -1 backward
    pub direction: i32,
    /// kept preprocessed on both sides of the view, before anything else
    pub around_samp: usize,
    /// keep filling beyond `around_samp` until the memory budget is used
    pub background: bool,
    /// the cache grows to at most this share of available memory (its own size counted as available)
    pub budget_pct: f32,
    /// blocks are evicted when available memory drops below this
    pub reserve_bytes: u64,
}

pub struct WorkerState {
    cache: Option<BlockCache>,
    plan: Option<Plan>,
    /// block being computed, and whether it is background fill
    active: Option<(usize, bool)>,
    shutdown: bool,
}

impl WorkerState {
    pub fn new() -> Self {
        WorkerState { cache: None, plan: None, active: None, shutdown: false }
    }
}

pub type SharedWorkerState = Arc<(Mutex<WorkerState>, Condvar)>;
pub type SharedCancel = Arc<AtomicBool>;

/// Hand the worker a new plan. The block being computed is cancelled if it was made
/// for another config, or if the view now lacks a block and this one isn't it.
pub fn post_plan(shared: &SharedWorkerState, cancel: &SharedCancel, plan: Plan) {
    let (lock, cvar) = &**shared;
    let mut st = lock.lock().unwrap();
    if st.plan.as_ref() == Some(&plan) {
        return;
    }
    if let (Some((b, _)), Some(c)) = (st.active, &st.cache) {
        let view = c.range(plan.view_first, plan.view_first + plan.view_n);
        let view_missing = view.clone().any(|v| !c.blocks.contains_key(&v));
        if c.cfg != plan.cfg || (view_missing && !view.contains(&b)) {
            cancel.store(true, Ordering::Relaxed);
        }
    }
    st.plan = Some(plan);
    cvar.notify_one();
}

/// Ask the worker to exit; returns immediately.
pub fn request_shutdown(shared: &SharedWorkerState, cancel: &SharedCancel) {
    cancel.store(true, Ordering::Relaxed);
    let (lock, cvar) = &**shared;
    lock.lock().unwrap().shutdown = true;
    cvar.notify_one();
}

/// True while a block the view needs (not background fill) is being computed.
pub fn computing_foreground(shared: &SharedWorkerState) -> bool {
    matches!(shared.0.lock().unwrap().active, Some((_, false)))
}

/// Cached extents as (first_sample, n_samp), consecutive blocks merged.
pub fn cached_runs(shared: &SharedWorkerState) -> Vec<(usize, usize)> {
    let st = shared.0.lock().unwrap();
    let Some(c) = &st.cache else { return Vec::new() };
    let mut idx: Vec<usize> = c.blocks.keys().copied().collect();
    idx.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for b in idx {
        let (first, n) = (b * c.block_samp, c.block_len(b));
        match runs.last_mut() {
            Some(r) if r.0 + r.1 == first => r.1 += n,
            _ => runs.push((first, n)),
        }
    }
    runs
}

pub fn cache_display_rows(shared: &SharedWorkerState) -> Option<Arc<Vec<DisplayRow>>> {
    shared.0.lock().unwrap().cache.as_ref().map(|c| Arc::clone(&c.display_rows))
}

// ---------------------------------------------------------------------------
// View window
// ---------------------------------------------------------------------------

/// Contiguous f32 copy of the cached data around the view, in display units (µV, or
/// mV/mm² for CSD) — what the heatmap, waveform view and overlays read.
/// Layout: data[row * n_samp + t].
pub struct ViewWindow {
    pub first_sample: usize,
    pub n_samp: usize,
    pub data: Arc<Vec<f32>>,
    pub cfg: PreprocConfig,
    pub display_rows: Arc<Vec<DisplayRow>>,
    cache_id: u64,
}

/// Keep `window` covering the view ± `pad` samples, as far as the cached blocks allow
/// (with gaps: the run of cached blocks that covers most of the view). Rebuilt when the
/// view comes within half of `pad` of its edge, or when more of that range is cached.
/// Left untouched if the cache is for another config or has nothing near the view.
pub fn update_view_window(
    shared: &SharedWorkerState,
    window: &mut Option<ViewWindow>,
    view_first: usize,
    view_n: usize,
    pad: usize,
    cfg: &PreprocConfig,
) {
    let (lo, hi, blocks, cfg, display_rows, cache_id, block_samp, scale, n_rows) = {
        let st = shared.0.lock().unwrap();
        let Some(c) = &st.cache else { return };
        if c.cfg != *cfg {
            return;
        }
        let view_end = view_first + view_n;
        let view = c.range(view_first, view_end);
        let wanted = c.range(view_first.saturating_sub(pad), view_end + pad);

        // runs of cached blocks inside `wanted`; keep the one sharing most blocks with the view
        let mut best: Option<std::ops::Range<usize>> = None;
        let mut b = wanted.start;
        while b < wanted.end {
            if !c.blocks.contains_key(&b) {
                b += 1;
                continue;
            }
            let start = b;
            while b < wanted.end && c.blocks.contains_key(&b) {
                b += 1;
            }
            let shared_blocks = |r: &std::ops::Range<usize>| r.end.min(view.end).saturating_sub(r.start.max(view.start));
            if best.as_ref().map_or(true, |r| shared_blocks(&(start..b)) > shared_blocks(r)) {
                best = Some(start..b);
            }
        }
        let Some(run) = best else { return };
        let run_lo = run.start * c.block_samp;
        let run_hi = (run.end * c.block_samp).min(c.n_samples);
        let clip = |lo: usize, hi: usize| (lo.max(run_lo), hi.min(run_hi));

        // still good: covers the view ± pad/2, as far as cached
        let (need_lo, need_hi) = clip(view_first.saturating_sub(pad / 2), view_end + pad / 2);
        if let Some(w) = window.as_ref() {
            if w.cache_id == c.id && w.first_sample <= need_lo && w.first_sample + w.n_samp >= need_hi {
                return;
            }
        }
        let (lo, hi) = clip(view_first.saturating_sub(pad), view_end + pad);
        if lo >= hi {
            return;
        }
        let blocks: Vec<(usize, Arc<Block>)> = c.range(lo, hi).map(|b| (b, Arc::clone(&c.blocks[&b]))).collect();
        let scale = c.step_uv * crate::preprocess::csd_to_mv_per_mm2(&c.cfg);
        (lo, hi, blocks, c.cfg.clone(), Arc::clone(&c.display_rows), c.id, c.block_samp, scale, c.n_rows)
    };

    let n_samp = hi - lo;
    let mut data = vec![0.0f32; n_rows * n_samp];
    data.par_chunks_mut(n_samp).enumerate().for_each(|(r, row)| {
        for (b, blk) in &blocks {
            let b_first = b * block_samp;
            let len = blk.data.len() / n_rows;
            // overlap of this block with [lo, hi)
            let (s, e) = (b_first.max(lo), (b_first + len).min(hi));
            if s >= e {
                continue;
            }
            let src = &blk.data[r * len + (s - b_first)..r * len + (e - b_first)];
            for (o, &v) in row[s - lo..e - lo].iter_mut().zip(src) {
                *o = v as f32 * scale;
            }
        }
    });

    *window = Some(ViewWindow { first_sample: lo, n_samp, data: Arc::new(data), cfg, display_rows, cache_id });
}

/// `pct`-th percentile (0–100) of |values| in samples [lo, hi) of `w` (clipped to it),
/// from an evenly strided sample of at most `PCT_MAX_SAMPLES` values.
pub fn window_percentile(w: &ViewWindow, lo: usize, hi: usize, pct: f32) -> Option<f32> {
    let (lo, hi) = (lo.max(w.first_sample), hi.min(w.first_sample + w.n_samp));
    if lo >= hi || w.n_samp == 0 {
        return None;
    }
    let (off, n) = (lo - w.first_sample, hi - lo);
    let n_rows = w.data.len() / w.n_samp;
    let step = (n_rows * n / PCT_MAX_SAMPLES).max(1);
    let mut vals: Vec<f32> = (0..n_rows)
        .flat_map(|r| w.data[r * w.n_samp + off..r * w.n_samp + off + n].iter().step_by(step).map(|v| v.abs()))
        .collect();
    let k = ((pct as f64 / 100.0 * (vals.len() - 1) as f64).round() as usize).min(vals.len() - 1);
    Some(*vals.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1)
}

// ---------------------------------------------------------------------------
// Scheduling
// ---------------------------------------------------------------------------

/// Fill / eviction priority of block `b` (lower first): distance from the view, with
/// distance behind the scroll direction weighted `BEHIND_WEIGHT` times.
fn score(c: &BlockCache, p: &Plan, b: usize) -> usize {
    let start = b * c.block_samp;
    let end = start + c.block_len(b);
    let view_end = p.view_first + p.view_n;
    let (d, ahead) = if start >= view_end {
        (start - view_end, p.direction >= 0)
    } else if end <= p.view_first {
        (p.view_first - end, p.direction < 0)
    } else {
        (0, true)
    };
    if ahead { d } else { d * BEHIND_WEIGHT }
}

/// Next block to compute (and whether it is background fill), evicting blocks to stay
/// within budget. View blocks first, then the `around` margin, then background fill.
/// The view and margin are always computed and never evicted, whatever the memory.
fn next_block(c: &mut BlockCache, p: &Plan, available: u64) -> Option<(usize, bool)> {
    let view = c.range(p.view_first, p.view_first + p.view_n);
    if let Some(b) = view.clone().find(|b| !c.blocks.contains_key(b)) {
        return Some((b, false));
    }
    let protected = c.range(p.view_first.saturating_sub(p.around_samp), p.view_first + p.view_n + p.around_samp);
    if let Some(b) = protected.clone().filter(|b| !c.blocks.contains_key(b)).min_by_key(|&b| score(c, p, b)) {
        return Some((b, false));
    }

    let block_bytes = c.block_bytes();
    let reserve = p.reserve_bytes as usize;
    // memory freed by evictions here counts as available again
    let mut avail = available as usize;
    let cap = if p.background {
        ((avail + c.bytes) as f64 * p.budget_pct as f64 / 100.0) as usize
    } else {
        // without background fill keep twice the margin, so scrolling back and forth
        // doesn't recompute what was just evicted
        2 * protected.len() * block_bytes
    };
    let farthest = |c: &BlockCache| {
        c.blocks.keys().copied().filter(|b| !protected.contains(b)).max_by_key(|&b| score(c, p, b))
    };
    while c.bytes > cap || avail < reserve {
        let Some(b) = farthest(c) else { break };
        let before = c.bytes;
        c.evict(b);
        avail += before - c.bytes;
    }
    if !p.background {
        return None;
    }

    // nearest missing block on each side of the margin
    let n_blocks = c.n_blocks();
    let after = (protected.end..n_blocks).find(|b| !c.blocks.contains_key(b));
    let before = (0..protected.start).rev().find(|b| !c.blocks.contains_key(b));
    let cand = [after, before].into_iter().flatten().min_by_key(|&b| score(c, p, b))?;
    let cand_score = score(c, p, cand);
    // make room only by dropping blocks that matter less than the candidate
    while c.bytes + block_bytes > cap || avail < reserve + block_bytes {
        let b = farthest(c).filter(|&b| score(c, p, b) > cand_score)?;
        let before = c.bytes;
        c.evict(b);
        avail += before - c.bytes;
    }
    Some((cand, true))
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

struct Job {
    b: usize,
    cache_id: u64,
    cfg: PreprocConfig,
    display_rows: Arc<Vec<DisplayRow>>,
    n_rows: usize,
    first: usize,
    len: usize,
    overlap: usize,
    step_uv: f32,
    agc_eps: Option<Arc<Vec<f32>>>,
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
        let pool = rayon::ThreadPoolBuilder::new().num_threads(compute_thread_count()).build().unwrap();
        let mut sys = sysinfo::System::new();
        let (lock, cvar) = &*shared;
        loop {
            let job = {
                let mut st = lock.lock().unwrap();
                loop {
                    if st.shutdown {
                        st.cache = None;
                        return;
                    }
                    if let Some(plan) = st.plan.clone() {
                        if st.cache.as_ref().map_or(true, |c| c.cfg != plan.cfg) {
                            let filt_g = filt.lock().unwrap().clone();
                            st.cache = Some(BlockCache::new(&plan.cfg, &meta, &filt_g));
                        }
                        sys.refresh_memory();
                        let c = st.cache.as_mut().unwrap();
                        if let Some((b, background)) = next_block(c, &plan, sys.available_memory()) {
                            let job = Job {
                                b,
                                cache_id: c.id,
                                cfg: c.cfg.clone(),
                                display_rows: Arc::clone(&c.display_rows),
                                n_rows: c.n_rows,
                                first: b * c.block_samp,
                                len: c.block_len(b),
                                overlap: c.overlap,
                                step_uv: c.step_uv,
                                agc_eps: c.agc_eps.clone(),
                            };
                            st.active = Some((b, background));
                            // reset under the lock, so a cancel posted for this block isn't lost
                            cancel.store(false, Ordering::Relaxed);
                            break job;
                        }
                    }
                    st = cvar.wait_timeout(st, IDLE_RECHECK).unwrap().0;
                }
            };

            let result = pool.install(|| compute_block(&job, &raw, &meta, &filt, &cancel));

            let mut st = lock.lock().unwrap();
            st.active = None;
            if let (Some((block, eps)), Some(c)) = (result, st.cache.as_mut()) {
                if c.id == job.cache_id {
                    if job.agc_eps.is_none() && job.cfg.spatial_filter == SpatialFilter::Destripe {
                        c.agc_eps = Some(Arc::new(eps));
                    }
                    c.insert(job.b, block);
                    ctx.request_repaint();
                }
            }
        }
    })
}

/// Read and preprocess one block plus its overlap, keep the block, store it as i16.
/// Returns the block and the destripe AGC floors used; None if cancelled.
fn compute_block(job: &Job, raw: &RawData, meta: &Meta, filt: &Mutex<Filters>, cancel: &AtomicBool) -> Option<(Block, Vec<f32>)> {
    let read_first = job.first.saturating_sub(job.overlap);
    let read_end = (job.first + job.len + job.overlap).min(meta.n_samples);
    let read_n = read_end - read_first;

    let mut data = raw.read_rows(read_first, read_n, meta, &job.display_rows, job.cfg.phase_shift);
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let filt_g = filt.lock().unwrap().clone();
    let eps = preprocess(&mut data, read_n, &job.cfg, &filt_g, cancel, &job.display_rows, job.agc_eps.as_deref().map(|v| &v[..]));
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    let (off, len) = (job.first - read_first, job.len);
    let inv = 1.0 / job.step_uv;
    let mut q = vec![0i16; job.n_rows * len];
    if len > 0 {
        q.par_chunks_mut(len).enumerate().for_each(|(r, row)| {
            let src = &data[r * read_n + off..r * read_n + off + len];
            for (o, &v) in row.iter_mut().zip(src) {
                *o = (v * inv).round().clamp(-32768.0, 32767.0) as i16;
            }
        });
    }
    Some((Block { data: q }, eps))
}
