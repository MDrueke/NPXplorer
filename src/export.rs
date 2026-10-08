//! "File → Export preprocessed data…": a copy of the recording with the current
//! preprocessing applied, in the input's format (SpikeGLX `.bin` + `.meta`, or an Open
//! Ephys folder), plus a provenance file, NPXplorer settings for the copy (with
//! preprocessing off, so it isn't filtered twice) and optionally a Kilosort 4 probe file.
//!
//! The recording is processed in chunks with margins on both sides that are thrown
//! away, so the result matches processing it in one piece. Steps that take a value
//! from the data itself (DC offset, destripe's AGC floor) use values estimated once,
//! from chunks spread over the exported range, instead of each chunk's own.
//!
//! Pipeline: reader → compute → writer (→ SHA-1 for SpikeGLX), one thread each,
//! connected by bounded queues; the parallel work inside every stage shares one thread
//! pool with all cores.

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use crate::data::{DisplayRow, Meta, RawData};
use crate::preprocess::{preprocess, Filters, PreprocConfig, SpatialFilter};

// ---------------------------------------------------------------------------
// Settings (export window, saved per recording)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum BadChannelAction {
    Keep,
    Remove,
    Interpolate,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum BlankMode {
    /// samples in the window set to 0
    Zero,
    /// a straight line per channel from the sample before the window to the one after
    Interpolate,
}

/// Blanking of windows around event on- and offsets.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BlankSettings {
    pub enabled: bool,
    pub mode: BlankMode,
    /// window around each onset (ms, relative to it; start may be negative)
    pub onset_start_ms: f64,
    pub onset_end_ms: f64,
    pub offsets: bool,
    /// window around each offset (ms, relative to it)
    pub offset_start_ms: f64,
    pub offset_end_ms: f64,
    /// offset = onset + this, when the event file has no offset column
    pub event_length_ms: f64,
}

impl Default for BlankSettings {
    fn default() -> Self {
        BlankSettings {
            enabled: false,
            mode: BlankMode::Zero,
            onset_start_ms: -1.0,
            onset_end_ms: 3.0,
            offsets: false,
            offset_start_ms: -1.0,
            offset_end_ms: 3.0,
            event_length_ms: 100.0,
        }
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportSettings {
    /// exported time range (s); `None` = the whole recording
    pub range_s: Option<[f64; 2]>,
    pub average_depths: bool,
    pub dead: BadChannelAction,
    pub noisy: BadChannelAction,
    pub remove_outside: bool,
    pub blank: BlankSettings,
    pub probe_file: bool,
    pub open_when_done: bool,
}

impl Default for ExportSettings {
    fn default() -> Self {
        ExportSettings {
            range_s: None,
            average_depths: false,
            dead: BadChannelAction::Keep,
            noisy: BadChannelAction::Keep,
            remove_outside: false,
            blank: BlankSettings::default(),
            probe_file: false,
            open_when_done: false,
        }
    }
}

impl ExportSettings {
    /// Whether the export needs the channel classification's labels.
    pub fn needs_labels(&self) -> bool {
        self.dead != BadChannelAction::Keep || self.noisy != BadChannelAction::Keep || self.remove_outside
    }
}

// ---------------------------------------------------------------------------
// Formats and file names
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    SpikeGlx,
    OpenEphys,
}

impl Format {
    /// Same rule as `Meta::from_data_path`: a sibling `.meta` file means SpikeGLX.
    pub fn of(src: &Path) -> Format {
        if src.with_extension("meta").is_file() {
            Format::SpikeGlx
        } else {
            Format::OpenEphys
        }
    }

    fn label(self) -> &'static str {
        match self {
            Format::SpikeGlx => "SpikeGLX",
            Format::OpenEphys => "Open Ephys",
        }
    }
}

/// Seconds for a file name: up to 3 decimals, trailing zeros dropped.
fn fmt_s(s: f64) -> String {
    let t = format!("{s:.3}");
    let t = t.trim_end_matches('0').trim_end_matches('.');
    if t.is_empty() { "0".into() } else { t.to_string() }
}

/// Text added to the exported recording's name, e.g. `_preprocessed_120-300s`.
pub fn name_tag(range_s: Option<[f64; 2]>) -> String {
    match range_s {
        Some([a, b]) => format!("_preprocessed_{}-{}s", fmt_s(a), fmt_s(b)),
        None => "_preprocessed".to_string(),
    }
}

/// Top folder of an Open Ephys session (the one holding the Record Node folders).
fn open_ephys_root(src: &Path) -> Option<PathBuf> {
    let (_, settings) = crate::data::find_open_ephys_meta(src)?;
    settings.parent()?.parent().map(Path::to_path_buf)
}

/// Proposed destination: SpikeGLX — the data file with the tag inserted before
/// `_g0_t0` (so the name stays readable by SpikeGLX tools), always as `.bin`;
/// Open Ephys — the session folder with the tag appended.
pub fn default_destination(src: &Path, format: Format, range_s: Option<[f64; 2]>) -> PathBuf {
    let tag = name_tag(range_s);
    match format {
        Format::SpikeGlx => {
            let name = src.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let name = name.strip_suffix(".cbin").map(|s| format!("{s}.bin")).unwrap_or(name);
            let at = spikeglx_run_end(&name).unwrap_or_else(|| name.find('.').unwrap_or(name.len()));
            src.with_file_name(format!("{}{tag}{}", &name[..at], &name[at..]))
        }
        Format::OpenEphys => {
            let root = open_ephys_root(src).unwrap_or_else(|| src.parent().unwrap_or(src).to_path_buf());
            let name = root.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            root.with_file_name(format!("{name}{tag}"))
        }
    }
}

/// Byte index where a SpikeGLX run name ends: the start of its last `_g<N>_t<N>.`.
fn spikeglx_run_end(name: &str) -> Option<usize> {
    let b = name.as_bytes();
    (0..b.len()).rev().find(|&i| {
        let rest = &name[i..];
        let Some(r) = rest.strip_prefix("_g") else { return false };
        let d1 = r.bytes().take_while(u8::is_ascii_digit).count();
        let Some(r) = r[d1..].strip_prefix("_t").filter(|_| d1 > 0) else { return false };
        let d2 = r.bytes().take_while(u8::is_ascii_digit).count();
        d2 > 0 && (r[d2..].starts_with('.') || r[d2..].is_empty())
    })
}

// ---------------------------------------------------------------------------
// Job, progress, outcome
// ---------------------------------------------------------------------------

/// Event times loaded in the Events window.
#[derive(Clone)]
pub struct EventTimes {
    pub onsets: Vec<f64>,
    pub offsets: Option<Vec<f64>>,
    pub file: Option<PathBuf>,
}

pub struct Job {
    pub src: PathBuf,
    /// SpikeGLX: the new `.bin`; Open Ephys: the new session folder
    pub dst: PathBuf,
    pub format: Format,
    pub raw: Arc<RawData>,
    pub meta: Arc<Meta>,
    /// the view's preprocessing; its removed channels are the user's
    pub cfg: PreprocConfig,
    pub settings: ExportSettings,
    /// classification labels if computed (0 good, 1 dead, 2 noisy, 3 outside)
    pub labels: Option<Arc<Vec<u8>>>,
    pub classify_chunks: usize,
    pub outside_rule: crate::channel_classify::OutsideRule,
    pub events: Option<EventTimes>,
    /// the source recording's settings file (its atlas section is carried over)
    pub source_settings: toml::Table,
    /// chunk length in samples; `None` = chosen from the available memory (tests set
    /// it to force many chunk boundaries)
    pub chunk_samples: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Phase {
    Starting = 0,
    Classifying,
    Estimating,
    Writing,
    Finishing,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Phase::Starting => "Starting…",
            Phase::Classifying => "Classifying channels…",
            Phase::Estimating => "Estimating DC offsets and filter levels…",
            Phase::Writing => "Writing preprocessed data…",
            Phase::Finishing => "Finishing (flushing to disk, metadata)…",
        }
    }
}

#[derive(Default)]
pub struct Progress {
    phase: AtomicU8,
    /// classification chunks done (its own counter type)
    pub classify_done: AtomicUsize,
    pub classify_total: AtomicUsize,
    /// samples written / to write
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub bytes: AtomicU64,
}

impl Progress {
    pub fn phase(&self) -> Phase {
        match self.phase.load(Ordering::Relaxed) {
            1 => Phase::Classifying,
            2 => Phase::Estimating,
            3 => Phase::Writing,
            4 => Phase::Finishing,
            _ => Phase::Starting,
        }
    }

    fn set_phase(&self, p: Phase) {
        self.phase.store(p as u8, Ordering::Relaxed);
    }
}

pub struct Done {
    /// the exported data file (to open it)
    pub data_path: PathBuf,
    /// what was written, for the summary
    pub files: Vec<PathBuf>,
    pub n_channels: usize,
    pub duration_s: f64,
    pub clipped: u64,
    pub elapsed_s: f64,
    pub bytes: u64,
}

pub enum Outcome {
    Done(Done),
    Cancelled,
    Failed(String),
}

pub struct Handle {
    pub progress: Arc<Progress>,
    pub cancel: Arc<AtomicBool>,
    pub started: Instant,
    rx: mpsc::Receiver<(Outcome, Option<Vec<u8>>)>,
}

impl Handle {
    /// The result, once the export has ended, with the classification labels if the
    /// export computed them.
    pub fn poll(&self) -> Option<(Outcome, Option<Vec<u8>>)> {
        match self.rx.try_recv() {
            Ok(v) => Some(v),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                Some((Outcome::Failed("the export stopped unexpectedly".into()), None))
            }
        }
    }
}

pub fn spawn(job: Job, ctx: egui::Context) -> Handle {
    let progress = Arc::new(Progress::default());
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let (p, c) = (Arc::clone(&progress), Arc::clone(&cancel));
    std::thread::spawn(move || {
        let mut labels_out = None;
        let outcome = match run(&job, &p, &c, &mut labels_out) {
            Ok(done) => Outcome::Done(done),
            Err(_) if c.load(Ordering::Relaxed) => Outcome::Cancelled,
            Err(e) => Outcome::Failed(format!("{e:#}")),
        };
        if !matches!(outcome, Outcome::Done(_)) {
            cleanup(&job);
        }
        let _ = tx.send((outcome, labels_out));
        ctx.request_repaint();
    });
    Handle { progress, cancel, started: Instant::now(), rx }
}

/// Remove what a failed or cancelled export left behind.
fn cleanup(job: &Job) {
    match job.format {
        Format::SpikeGlx => {
            let _ = std::fs::remove_file(part_path(&job.dst));
        }
        Format::OpenEphys => {
            if job.dst.join(CREATED_MARKER).is_file() {
                let _ = std::fs::remove_dir_all(&job.dst);
            }
        }
    }
}

/// Marks an Open Ephys output folder as created by this export (removed when done),
/// so a failed export only ever deletes a folder it made itself.
const CREATED_MARKER: &str = ".npxplorer_export_incomplete";

fn part_path(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

// ---------------------------------------------------------------------------
// Channel plan
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Source {
    /// a preprocessed data row
    Row(usize),
    /// weighted sum of preprocessed data rows (an interpolated channel)
    Mix(Vec<(usize, f32)>),
}

#[derive(Clone, Debug)]
struct OutChannel {
    /// file channel the output channel is named, placed and scaled after
    ch: usize,
    src: Source,
    x_um: f32,
    y_um: f32,
    shank: u32,
}

struct Plan {
    /// rows the preprocessing runs on (without removed and interpolated channels)
    rows: Vec<DisplayRow>,
    n_rows: usize,
    /// output channels in file order
    out: Vec<OutChannel>,
    removed_dead: Vec<usize>,
    removed_noisy: Vec<usize>,
    removed_outside: Vec<usize>,
    interpolated: Vec<usize>,
}

/// IBL's `interpolate_bad_channels`: weights exp(-(d / 20 µm)^1.3) over the good
/// channels of the same shank, small weights dropped, normalised.
const KRIGING_DISTANCE_UM: f32 = 20.0;
const KRIGING_P: f32 = 1.3;
const KRIGING_MIN_WEIGHT: f32 = 0.005;

fn plan_channels(meta: &Meta, user_removed: &BTreeSet<usize>, avg: bool, labels: Option<&[u8]>, s: &ExportSettings) -> Plan {
    let with_label = |l: u8| -> Vec<usize> {
        labels
            .map(|lab| (0..meta.n_ap_chans).filter(|&c| lab.get(c) == Some(&l) && !user_removed.contains(&c)).collect())
            .unwrap_or_default()
    };
    let (dead, noisy, outside) = (with_label(1), with_label(2), with_label(3));
    let mut excluded = user_removed.clone();
    let mut interpolated = Vec::new();
    let mut take = |chans: &[usize], action: BadChannelAction, removed: &mut Vec<usize>| match action {
        BadChannelAction::Keep => {}
        BadChannelAction::Remove => {
            removed.extend_from_slice(chans);
            excluded.extend(chans);
        }
        BadChannelAction::Interpolate => {
            interpolated.extend_from_slice(chans);
            excluded.extend(chans);
        }
    };
    let (mut removed_dead, mut removed_noisy, mut removed_outside) = (Vec::new(), Vec::new(), Vec::new());
    take(&dead, s.dead, &mut removed_dead);
    take(&noisy, s.noisy, &mut removed_noisy);
    take(&outside, if s.remove_outside { BadChannelAction::Remove } else { BadChannelAction::Keep }, &mut removed_outside);
    interpolated.sort_unstable();

    let rows = meta.build_display_rows(avg, &excluded, crate::data::ChannelOrder::Id, crate::data::ShankOrder::Id);
    let data_rows: Vec<(usize, usize, f32, f32, u32)> = rows
        .iter()
        .filter_map(|r| match r {
            DisplayRow::Data { data_idx, first_ch, x_um, y_um, shank, .. } => Some((*data_idx, *first_ch, *x_um, *y_um, *shank)),
            _ => None,
        })
        .collect();

    let mut out: Vec<OutChannel> = data_rows
        .iter()
        .map(|&(i, ch, x, y, shank)| OutChannel { ch, src: Source::Row(i), x_um: x, y_um: y, shank })
        .collect();
    for &c in &interpolated {
        let g = &meta.channel_geom[c];
        // with depth averaging, a bad channel at a depth that still has a row is
        // simply left out of that row's average
        if avg && data_rows.iter().any(|r| r.4 == g.shank && (r.3 - g.y_um).abs() < 0.5) {
            continue;
        }
        let mut w: Vec<(usize, f32)> = data_rows
            .iter()
            .filter(|r| r.4 == g.shank)
            .map(|r| {
                let d = ((r.2 - g.x_um).powi(2) + (r.3 - g.y_um).powi(2)).sqrt();
                (r.0, (-(d / KRIGING_DISTANCE_UM).powf(KRIGING_P)).exp())
            })
            .filter(|&(_, w)| w >= KRIGING_MIN_WEIGHT)
            .collect();
        let sum: f32 = w.iter().map(|p| p.1).sum();
        w.iter_mut().for_each(|p| p.1 /= sum.max(f32::MIN_POSITIVE));
        out.push(OutChannel { ch: c, src: Source::Mix(w), x_um: g.x_um, y_um: g.y_um, shank: g.shank });
    }
    out.sort_by_key(|o| o.ch);
    Plan { n_rows: data_rows.len(), rows, out, removed_dead, removed_noisy, removed_outside, interpolated }
}

/// Number of channels the export will contain (without sync channels).
pub fn output_channel_count(meta: &Meta, user_removed: &BTreeSet<usize>, labels: Option<&[u8]>, s: &ExportSettings) -> usize {
    plan_channels(meta, user_removed, s.average_depths, labels, s).out.len()
}

// ---------------------------------------------------------------------------
// Event blanking
// ---------------------------------------------------------------------------

/// Sample ranges `[lo, hi)` to blank, sorted and merged.
fn blank_intervals(ev: &EventTimes, b: &BlankSettings, fs: f64, n_total: usize) -> Vec<(usize, usize)> {
    let to_sample = |t: f64| (t * fs).round().clamp(0.0, n_total as f64) as usize;
    let mut v = Vec::new();
    let mut push = |t: f64, start_ms: f64, end_ms: f64| {
        let (lo, hi) = (to_sample(t + start_ms / 1000.0), to_sample(t + end_ms / 1000.0));
        if hi > lo {
            v.push((lo, hi));
        }
    };
    for (i, &on) in ev.onsets.iter().enumerate() {
        push(on, b.onset_start_ms, b.onset_end_ms);
        if b.offsets {
            let off = ev.offsets.as_ref().and_then(|o| o.get(i).copied()).unwrap_or(on + b.event_length_ms / 1000.0);
            push(off, b.offset_start_ms, b.offset_end_ms);
        }
    }
    v.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(v.len());
    for (lo, hi) in v {
        match merged.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    merged
}

/// Blank the intervals that overlap `data` (`[n_rows][n_samp]`, starting at absolute
/// sample `first`).
fn apply_blanking(data: &mut [f32], n_samp: usize, first: usize, intervals: &[(usize, usize)], mode: BlankMode) {
    let end = first + n_samp;
    let start_idx = intervals.partition_point(|iv| iv.1 <= first);
    let local: Vec<(usize, usize)> = intervals[start_idx..]
        .iter()
        .take_while(|iv| iv.0 < end)
        .map(|&(lo, hi)| (lo.max(first) - first, hi.min(end) - first))
        .collect();
    if local.is_empty() {
        return;
    }
    data.par_chunks_mut(n_samp).for_each(|row| {
        for &(lo, hi) in &local {
            match mode {
                BlankMode::Zero => row[lo..hi].fill(0.0),
                BlankMode::Interpolate => {
                    // from the last sample before to the first after; at a data edge
                    // the available side is held
                    let a = if lo > 0 { Some(row[lo - 1]) } else { None };
                    let b = if hi < n_samp { Some(row[hi]) } else { None };
                    let (a, b) = match (a, b) {
                        (Some(a), Some(b)) => (a, b),
                        (Some(a), None) => (a, a),
                        (None, Some(b)) => (b, b),
                        (None, None) => (0.0, 0.0),
                    };
                    let span = (hi - lo + 1) as f32;
                    for (k, v) in row[lo..hi].iter_mut().enumerate() {
                        *v = a + (b - a) * (k + 1) as f32 / span;
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// The export
// ---------------------------------------------------------------------------

/// Chunks the DC offsets and destripe AGC floors are estimated from.
const ESTIMATE_CHUNKS: usize = 16;

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mid = v.len() / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    *m
}

/// Where the output goes, per format.
struct Targets {
    /// the data file being written (final name)
    data: PathBuf,
}

pub(crate) fn run(job: &Job, p: &Progress, cancel: &AtomicBool, labels_out: &mut Option<Vec<u8>>) -> Result<Done> {
    let t_start = Instant::now();
    let meta = &*job.meta;
    let fs = meta.sample_rate;
    let s = &job.settings;
    if !cfg!(target_endian = "little") {
        bail!("exporting is only supported on little-endian machines");
    }

    // --- destination ---------------------------------------------------------
    let src_canon = std::fs::canonicalize(&job.src).unwrap_or_else(|_| job.src.clone());
    let targets = match job.format {
        Format::SpikeGlx => {
            let dst_canon = job.dst.parent().and_then(|d| std::fs::canonicalize(d).ok()).map(|d| d.join(job.dst.file_name().unwrap_or_default()));
            if dst_canon.as_deref() == Some(src_canon.as_path()) || job.dst.with_extension("meta") == job.src.with_extension("meta") {
                bail!("the export would overwrite the original recording: choose another file name");
            }
            Targets { data: job.dst.clone() }
        }
        Format::OpenEphys => {
            let root = open_ephys_root(&job.src).context("no Open Ephys session folder (settings.xml) found above the recording")?;
            let root_canon = std::fs::canonicalize(&root).unwrap_or(root.clone());
            let dst_abs = job.dst.parent().and_then(|d| std::fs::canonicalize(d).ok()).map(|d| d.join(job.dst.file_name().unwrap_or_default())).unwrap_or(job.dst.clone());
            if dst_abs.starts_with(&root_canon) || root_canon.starts_with(&dst_abs) {
                bail!("the export folder must be outside the original session folder: choose another name");
            }
            if job.dst.exists() && std::fs::read_dir(&job.dst).map(|mut d| d.next().is_some()).unwrap_or(true) {
                bail!("{} already exists and is not empty: choose another name", job.dst.display());
            }
            std::fs::create_dir_all(&job.dst).with_context(|| format!("creating {}", job.dst.display()))?;
            std::fs::write(job.dst.join(CREATED_MARKER), b"")?;
            let rel = job.src.strip_prefix(&root).context("recording is not inside its session folder")?;
            // a compressed continuous.cbin is written uncompressed
            Targets { data: job.dst.join(rel).with_extension("dat") }
        }
    };

    // --- channel classification (if needed and not done) -----------------------
    let labels: Option<Arc<Vec<u8>>> = if s.needs_labels() && job.labels.is_none() {
        p.set_phase(Phase::Classifying);
        p.classify_total.store(job.classify_chunks.max(1), Ordering::Relaxed);
        let l = crate::channel_classify::classify_recording(
            &job.raw,
            meta,
            &job.cfg.removed_channels,
            job.classify_chunks.max(1),
            job.outside_rule,
            cancel,
            &p.classify_done,
        )
        .context("cancelled")?;
        *labels_out = Some(l.clone());
        Some(Arc::new(l))
    } else {
        job.labels.clone()
    };

    // --- plan ----------------------------------------------------------------------
    let plan = plan_channels(meta, &job.cfg.removed_channels, s.average_depths, labels.as_deref().map(|v| v.as_slice()), s);
    if plan.out.is_empty() {
        bail!("no channels left to export");
    }
    let sync_chans: Vec<usize> = (meta.n_ap_chans..meta.n_saved_chans).collect();
    let n_out = plan.out.len();
    let n_cols = n_out + sync_chans.len();

    let (first, n_samples) = match s.range_s {
        Some([a, b]) => {
            let lo = ((a.max(0.0) * fs).round() as usize).min(meta.n_samples);
            let hi = ((b * fs).round() as usize).clamp(lo, meta.n_samples);
            (lo, hi - lo)
        }
        None => (0, meta.n_samples),
    };
    if n_samples == 0 {
        bail!("the time range is empty");
    }
    let out_bytes = n_samples as u64 * n_cols as u64 * 2;
    check_free_space(&targets.data, out_bytes)?;

    let intervals = match (&job.events, s.blank.enabled) {
        (Some(ev), true) => blank_intervals(ev, &s.blank, fs, meta.n_samples),
        (None, true) => bail!("event blanking needs an event file: load one in the Events window"),
        _ => Vec::new(),
    };

    // the view's preprocessing; DC removal is done here, with fixed offsets
    let cfg = PreprocConfig { avg_depths: s.average_depths, ..job.cfg.clone() };
    let cfg_no_dc = PreprocConfig { dc_removal: false, ..cfg.clone() };
    let filters = Filters::new(&cfg);

    // margins thrown away on both sides of every chunk: enough for the highpass and
    // notches to settle, destripe's AGC window, and any blanking window to fit
    let longest_blank = intervals.iter().map(|iv| iv.1 - iv.0).max().unwrap_or(0);
    let margin = crate::worker::EXTENSION_OVERLAP_SAMP
        .max((crate::notch::settle_s(&cfg) * fs).ceil() as usize)
        .max(if cfg.spatial_filter == SpatialFilter::Destripe { filters.kfilt_lagc / 2 + 1 } else { 0 })
        .max(longest_blank + 16);

    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().context("creating the thread pool")?;

    // --- estimates from chunks spread over the range ----------------------------
    p.set_phase(Phase::Estimating);
    let est_len = (fs as usize).min(n_samples);
    let starts: Vec<usize> = (0..ESTIMATE_CHUNKS.min(n_samples / est_len.max(1)).max(1))
        .map(|i| first + (n_samples - est_len) * i / (ESTIMATE_CHUNKS - 1).max(1))
        .collect();
    let (dc, eps) = pool.install(|| -> Result<(Option<Vec<f32>>, Option<Vec<f32>>)> {
        let dc = if cfg.dc_removal {
            let mut per_row: Vec<Vec<f32>> = vec![Vec::new(); plan.n_rows];
            for &st in &starts {
                if cancel.load(Ordering::Relaxed) {
                    bail!("cancelled");
                }
                let d = job.raw.read_rows(st, est_len, meta, &plan.rows, cfg.phase_shift);
                for (r, row) in d.chunks(est_len).enumerate() {
                    per_row[r].push(row.iter().map(|&v| v as f64).sum::<f64>() as f32 / est_len as f32);
                }
            }
            Some(per_row.iter_mut().map(|v| median(v)).collect())
        } else {
            None
        };
        let eps = if cfg.spatial_filter == SpatialFilter::Destripe {
            let mut per_block: Vec<Vec<f32>> = Vec::new();
            for &st in &starts {
                if cancel.load(Ordering::Relaxed) {
                    bail!("cancelled");
                }
                let rs = st.saturating_sub(margin);
                let rn = (st + est_len + margin).min(meta.n_samples) - rs;
                let mut d = job.raw.read_rows(rs, rn, meta, &plan.rows, cfg.phase_shift);
                prepare(&mut d, rn, rs, dc.as_deref(), &intervals, s.blank.mode);
                let e = preprocess(&mut d, rn, &cfg_no_dc, &filters, cancel, &plan.rows, None);
                if per_block.is_empty() {
                    per_block = vec![Vec::new(); e.len()];
                }
                for (b, v) in e.into_iter().enumerate() {
                    if let Some(pb) = per_block.get_mut(b) {
                        pb.push(v);
                    }
                }
            }
            Some(per_block.iter_mut().map(|v| median(v)).collect())
        } else {
            None
        };
        Ok((dc, eps))
    })?;

    let t_estimated = t_start.elapsed();

    // --- chunk size from the memory available ---------------------------------
    // in flight: ~2 chunks queued + 1 in each stage, plus destripe's gain buffer
    let bytes_per_sample = 4 * plan.n_rows.max(1) + 2 * n_cols;
    let avail = {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.available_memory()
    };
    let budget = (avail / 2).clamp(512 << 20, 8 << 30) as usize;
    let mem_len = (budget / (7 * bytes_per_sample)).saturating_sub(2 * margin);
    let chunk_len = job
        .chunk_samples
        .unwrap_or_else(|| (20 * margin).max((5.0 * fs) as usize).min((30.0 * fs) as usize).min(mem_len).max(margin.max(1024)));

    // --- pipeline ------------------------------------------------------------------
    p.set_phase(Phase::Writing);
    p.total.store(n_samples as u64, Ordering::Relaxed);
    if let Some(dir) = targets.data.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let part = part_path(&targets.data);
    let mut file = std::fs::File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let _ = file.set_len(out_bytes); // preallocate; harmless if unsupported

    let fail: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let stop = AtomicBool::new(false);
    let stopped = || stop.load(Ordering::Relaxed) || cancel.load(Ordering::Relaxed);
    let record = |e: anyhow::Error| {
        fail.lock().unwrap().get_or_insert(e);
        stop.store(true, Ordering::Relaxed);
    };
    let clipped = AtomicU64::new(0);
    let hasher_result: Mutex<Option<String>> = Mutex::new(None);
    let scale: Vec<f32> = plan.out.iter().map(|o| 1.0 / meta.uv_per_bit[o.ch]).collect();
    // busy time per stage (ns), for the debug log
    let busy: [AtomicU64; 5] = Default::default();
    let timed = |i: usize, t: Instant| busy[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);

    std::thread::scope(|sc| {
        let (to_compute, from_reader) = mpsc::sync_channel::<ReadChunk>(1);
        let (to_writer, from_compute) = mpsc::sync_channel::<ReadChunk>(1);
        let (to_hasher, from_writer) = mpsc::sync_channel::<Vec<i16>>(2);

        // reader
        sc.spawn(|| {
            let to_compute = to_compute;
            let mut core_first = first;
            while core_first < first + n_samples && !stopped() {
                let core_len = chunk_len.min(first + n_samples - core_first);
                let read_first = core_first.saturating_sub(margin);
                let read_end = (core_first + core_len + margin).min(meta.n_samples);
                let read_n = read_end - read_first;
                let t = Instant::now();
                let (data, sync) = pool.install(|| {
                    let d = job.raw.read_rows(read_first, read_n, meta, &plan.rows, cfg.phase_shift);
                    let y = job.raw.read_i16(core_first, core_len, meta, &sync_chans);
                    (d, y)
                });
                timed(0, t);
                let chunk = ReadChunk { read_first, read_n, core_off: core_first - read_first, core_len, data, sync };
                if to_compute.send(chunk).is_err() {
                    break;
                }
                core_first += core_len;
            }
        });

        // compute
        sc.spawn(|| {
            let to_writer = to_writer;
            for mut c in from_reader {
                if stopped() {
                    break;
                }
                let t = Instant::now();
                pool.install(|| {
                    prepare(&mut c.data, c.read_n, c.read_first, dc.as_deref(), &intervals, s.blank.mode);
                    preprocess(&mut c.data, c.read_n, &cfg_no_dc, &filters, &stop, &plan.rows, eps.as_deref());
                });
                timed(1, t);
                if stopped() || to_writer.send(c).is_err() {
                    break;
                }
            }
        });

        // hasher (SpikeGLX's fileSHA1)
        if job.format == Format::SpikeGlx {
            sc.spawn(|| {
                use sha1::Digest;
                let mut h = sha1::Sha1::new();
                for buf in from_writer {
                    let t = Instant::now();
                    h.update(bytemuck::cast_slice::<i16, u8>(&buf));
                    timed(4, t);
                }
                let hex: String = h.finalize().iter().map(|b| format!("{b:02X}")).collect();
                *hasher_result.lock().unwrap() = Some(hex);
            });
        } else {
            drop(from_writer);
        }

        // writer (this thread)
        let to_hasher = (job.format == Format::SpikeGlx).then_some(to_hasher);
        for c in from_compute {
            if stopped() {
                break;
            }
            let t = Instant::now();
            let buf = pool.install(|| interleave(&c, &plan.out, &scale, n_cols, &clipped));
            timed(2, t);
            let bytes: &[u8] = bytemuck::cast_slice(&buf);
            let t = Instant::now();
            let written = file.write_all(bytes);
            timed(3, t);
            if let Err(e) = written {
                record(anyhow::Error::new(e).context(format!("writing {}", part.display())));
                break;
            }
            p.done.fetch_add(c.core_len as u64, Ordering::Relaxed);
            p.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            if let Some(tx) = &to_hasher {
                let _ = tx.send(buf); // hashed while the next chunk is written
            }
        }
        drop(to_hasher);
        if cancel.load(Ordering::Relaxed) {
            stop.store(true, Ordering::Relaxed);
        }
    });

    let ms = |i: usize| busy[i].load(Ordering::Relaxed) / 1_000_000;
    crate::file_log!(
        "export: {} chunks of {chunk_len} samples (margin {margin}); busy ms: read {}, preprocess {}, interleave {}, write {}, sha1 {}; total {} ms",
        n_samples.div_ceil(chunk_len),
        ms(0), ms(1), ms(2), ms(3), ms(4),
        t_start.elapsed().as_millis()
    );
    let timing = std::env::var_os("NPX_EXPORT_TIMING").is_some();
    if timing {
        eprintln!(
            "  stages (busy ms): read {}, preprocess {}, interleave {}, write {}, sha1 {}; chunk {chunk_len} samples, margin {margin}",
            ms(0), ms(1), ms(2), ms(3), ms(4)
        );
        eprintln!("  estimates done at {:.1} s, data written at {:.1} s", t_estimated.as_secs_f64(), t_start.elapsed().as_secs_f64());
    }
    if let Some(e) = fail.lock().unwrap().take() {
        return Err(e);
    }
    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }

    // --- finish: data file, then the metadata around it ---------------------------
    p.set_phase(Phase::Finishing);
    file.flush()?;
    file.sync_all().with_context(|| format!("writing {}", part.display()))?;
    drop(file);
    if timing {
        eprintln!("  synced at {:.1} s", t_start.elapsed().as_secs_f64());
    }
    std::fs::rename(&part, &targets.data).with_context(|| format!("renaming {} to {}", part.display(), targets.data.display()))?;

    let info = ExportInfo {
        job,
        plan: &plan,
        first,
        n_samples,
        sha1: hasher_result.lock().unwrap().take(),
        clipped: clipped.load(Ordering::Relaxed),
        dc: dc.as_deref(),
        eps: eps.as_deref(),
        n_blanked: intervals.len(),
        margin,
        chunk_len,
        data_path: &targets.data,
    };
    let mut files = vec![targets.data.clone()];
    match job.format {
        Format::SpikeGlx => files.extend(write_spikeglx_meta(&info)?),
        Format::OpenEphys => files.extend(write_open_ephys_meta(&info)?),
    }
    files.extend(write_companions(&info)?);
    if job.format == Format::OpenEphys {
        let _ = std::fs::remove_file(job.dst.join(CREATED_MARKER));
    }
    Ok(Done {
        data_path: targets.data.clone(),
        files,
        n_channels: n_out,
        duration_s: n_samples as f64 / fs,
        clipped: info.clipped,
        elapsed_s: t_start.elapsed().as_secs_f64(),
        bytes: out_bytes,
    })
}

struct ReadChunk {
    read_first: usize,
    read_n: usize,
    /// position of the kept part within the read range
    core_off: usize,
    core_len: usize,
    /// `[n_rows][read_n]`, µV
    data: Vec<f32>,
    /// raw sync channel(s), `[core_len][n_sync]`
    sync: Vec<i16>,
}

/// Samples per parallel work item when interleaving.
const INTERLEAVE_BLOCK: usize = 1024;
/// Samples transposed at a time within a work item: the output rows they fill stay
/// in L1 cache while every channel writes into them.
const INTERLEAVE_TILE: usize = 32;

/// The kept part of a preprocessed chunk as interleaved int16 (`[t][column]`): output
/// channels at their source channel's µV/bit, then the raw sync channel(s).
fn interleave(c: &ReadChunk, out: &[OutChannel], scale: &[f32], n_cols: usize, clipped: &AtomicU64) -> Vec<i16> {
    let n_out = out.len();
    let n_sync = n_cols - n_out;
    let mut buf = vec![0i16; c.core_len * n_cols];
    let rn = c.read_n;
    let row = |r: usize, t0: usize, nt: usize| &c.data[r * rn + c.core_off + t0..r * rn + c.core_off + t0 + nt];
    buf.par_chunks_mut(INTERLEAVE_BLOCK * n_cols).enumerate().for_each(|(b, block)| {
        let nt_block = block.len() / n_cols;
        let mut clips = 0u64;
        let mut mixed = [0.0f32; INTERLEAVE_TILE];
        for tile in (0..nt_block).step_by(INTERLEAVE_TILE) {
            let nt = INTERLEAVE_TILE.min(nt_block - tile);
            let t0 = b * INTERLEAVE_BLOCK + tile;
            let dst = &mut block[tile * n_cols..(tile + nt) * n_cols];
            for (k, o) in out.iter().enumerate() {
                let src: &[f32] = match &o.src {
                    Source::Row(r) => row(*r, t0, nt),
                    Source::Mix(w) => {
                        mixed[..nt].fill(0.0);
                        for &(r, wr) in w {
                            for (m, &v) in mixed.iter_mut().zip(row(r, t0, nt)) {
                                *m += wr * v;
                            }
                        }
                        &mixed[..nt]
                    }
                };
                let sc = scale[k];
                for (dt, &v) in src.iter().enumerate() {
                    // round half away from zero (as f32::round) by truncation, which
                    // vectorizes; then clip to the int16 range
                    let y = v * sc;
                    let r = y + 0.5f32.copysign(y);
                    clips += u64::from(!(-32769.0 < r && r < 32768.0));
                    dst[dt * n_cols + k] = r.clamp(-32768.0, 32767.0) as i16;
                }
            }
            for dt in 0..nt {
                let s0 = (t0 + dt) * n_sync;
                dst[dt * n_cols + n_out..(dt + 1) * n_cols].copy_from_slice(&c.sync[s0..s0 + n_sync]);
            }
        }
        if clips > 0 {
            clipped.fetch_add(clips, Ordering::Relaxed);
        }
    });
    buf
}

/// What the metadata writers need to know about the finished export.
struct ExportInfo<'a> {
    job: &'a Job,
    plan: &'a Plan,
    first: usize,
    n_samples: usize,
    sha1: Option<String>,
    clipped: u64,
    dc: Option<&'a [f32]>,
    eps: Option<&'a [f32]>,
    n_blanked: usize,
    margin: usize,
    chunk_len: usize,
    data_path: &'a Path,
}

/// Fixed DC offsets, then blanking — the steps the export does before handing the
/// chunk to `preprocess` (with its own DC removal off).
fn prepare(data: &mut [f32], n_samp: usize, first: usize, dc: Option<&[f32]>, intervals: &[(usize, usize)], mode: BlankMode) {
    if let Some(dc) = dc {
        data.par_chunks_mut(n_samp).zip(dc.par_iter()).for_each(|(row, &o)| row.iter_mut().for_each(|v| *v -= o));
    }
    if !intervals.is_empty() {
        apply_blanking(data, n_samp, first, intervals, mode);
    }
}

fn check_free_space(target: &Path, need: u64) -> Result<()> {
    let dir = target.ancestors().skip(1).find(|d| d.exists()).unwrap_or(Path::new("."));
    let dir = std::fs::canonicalize(dir).unwrap_or(dir.to_path_buf());
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let disk = disks
        .list()
        .iter()
        .filter(|d| dir.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len());
    if let Some(d) = disk {
        // a little headroom for the metadata files
        if d.available_space() < need + (64 << 20) {
            bail!(
                "not enough free disk space: the export needs {:.1} GB, {} has {:.1} GB free",
                need as f64 / 1e9,
                d.mount_point().display(),
                d.available_space() as f64 / 1e9
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SpikeGLX metadata
// ---------------------------------------------------------------------------

/// `(header)(entry)(entry)...` → header and entries, without the parentheses.
fn split_paren_list(v: &str) -> (String, Vec<String>) {
    let mut parts = v.split(')').map(|p| p.trim_start_matches('(').to_string()).filter(|p| !p.is_empty());
    let header = parts.next().unwrap_or_default();
    (header, parts.collect())
}

fn join_paren_list(header: &str, entries: &[&String]) -> String {
    let mut s = format!("({header})");
    for e in entries {
        s.push('(');
        s.push_str(e);
        s.push(')');
    }
    s
}

/// "0:383,768" → [0, 1, …, 383, 768]
fn expand_subset(v: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in v.split(',') {
        let part = part.trim();
        match part.split_once(':') {
            Some((a, b)) => out.extend(a.trim().parse::<usize>().ok()?..=b.trim().parse::<usize>().ok()?),
            None => out.push(part.parse().ok()?),
        }
    }
    Some(out)
}

/// [0, 1, 2, 5, 7, 8] → "0:2,5,7:8"
fn compress_subset(v: &[usize]) -> String {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        parts.push(if j > i { format!("{}:{}", v[i], v[j]) } else { v[i].to_string() });
        i = j + 1;
    }
    parts.join(",")
}

/// The source `.meta` with the keys that describe the file's channels, size and
/// position rewritten; everything else (probe table, gains, …) is kept, so the
/// export reads like a SpikeGLX recording that saved a channel subset.
fn spikeglx_meta_text(text: &str, info: &ExportInfo) -> String {
    let meta = &*info.job.meta;
    let kept: Vec<usize> = info.plan.out.iter().map(|o| o.ch).collect();
    let n_out = kept.len();
    // source file positions of the saved channels, in the export's order
    let saved: Vec<usize> = kept.iter().copied().chain(meta.n_ap_chans..meta.n_saved_chans).collect();
    let n_cols = saved.len();
    let fs = meta.sample_rate;

    // acquisition channel numbers, from the channel map, else from the old subset
    let field = |name: &str| {
        text.lines().find_map(|l| {
            let (k, v) = l.trim_end_matches('\r').split_once('=')?;
            (k.trim_start_matches('~') == name).then(|| v.to_string())
        })
    };
    let chan_map = field("snsChanMap").map(|v| split_paren_list(&v));
    let acq: Vec<usize> = chan_map
        .as_ref()
        .filter(|(_, e)| e.len() == meta.n_saved_chans)
        .and_then(|(_, e)| e.iter().map(|x| x.split_once(';')?.1.split(':').next()?.parse().ok()).collect())
        .or_else(|| field("snsSaveChanSubset").and_then(|v| expand_subset(&v)).filter(|v| v.len() == meta.n_saved_chans))
        .unwrap_or_else(|| (0..meta.n_saved_chans).collect());

    let mut out = String::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        let Some((k, v)) = line.split_once('=') else {
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let new_v: Option<String> = match k.trim_start_matches('~') {
            "nSavedChans" => Some(n_cols.to_string()),
            "snsApLfSy" => {
                let mut c: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if c.len() >= 3 {
                    if c[0] > 0 { c[0] = n_out } else { c[1] = n_out }
                    Some(c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
                } else {
                    Some(v.to_string())
                }
            }
            "snsSaveChanSubset" => Some(compress_subset(&saved.iter().map(|&i| acq[i]).collect::<Vec<_>>())),
            "snsChanMap" => {
                let (h, e) = split_paren_list(v);
                if e.len() == meta.n_saved_chans {
                    Some(join_paren_list(&h, &saved.iter().map(|&i| &e[i]).collect::<Vec<_>>()))
                } else {
                    None
                }
            }
            "snsGeomMap" | "snsShankMap" => {
                let (h, e) = split_paren_list(v);
                (e.len() == meta.n_ap_chans).then(|| join_paren_list(&h, &kept.iter().map(|&i| &e[i]).collect::<Vec<_>>()))
            }
            "fileSizeBytes" => Some((info.n_samples * n_cols * 2).to_string()),
            "fileTimeSecs" => Some(format!("{}", info.n_samples as f64 / fs)),
            "fileSHA1" => info.sha1.clone(),
            "firstSample" => v.trim().parse::<u64>().ok().map(|f| (f + info.first as u64).to_string()),
            "fileName" => Some(info.data_path.to_string_lossy().replace('\\', "/")),
            _ => Some(v.to_string()),
        };
        // keys that can't be adapted are dropped: readers then fall back to defaults
        // instead of reading wrong values
        if let Some(nv) = new_v {
            out.push_str(&format!("{k}={nv}\n"));
        }
    }
    out
}

fn write_spikeglx_meta(info: &ExportInfo) -> Result<Vec<PathBuf>> {
    let src = info.job.src.with_extension("meta");
    let text = std::fs::read_to_string(&src).with_context(|| format!("reading {}", src.display()))?;
    let dst = info.data_path.with_extension("meta");
    std::fs::write(&dst, spikeglx_meta_text(&text, info)).with_context(|| format!("writing {}", dst.display()))?;
    Ok(vec![dst])
}

// ---------------------------------------------------------------------------
// Open Ephys metadata
// ---------------------------------------------------------------------------

fn xml_escape(v: &str) -> String {
    v.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// `settings.xml` with the probe's per-channel lists (`CHANNELS`, `ELECTRODE_XPOS`,
/// `ELECTRODE_YPOS`, … — every element of the probe whose attributes are `CH<n>`)
/// reduced to the exported channels and renumbered from 0, in the probe the reader
/// (`parse_open_ephys_geometry`) takes positions from.
fn rewrite_settings_xml(xml: &str, node_id: u32, kept: &[usize]) -> Result<String> {
    let doc = roxmltree::Document::parse(xml).context("parsing settings.xml")?;
    let processor = doc
        .descendants()
        .find(|n| n.has_tag_name("PROCESSOR") && n.attribute("NodeId").and_then(|s| s.parse::<u32>().ok()) == Some(node_id))
        .with_context(|| format!("no PROCESSOR with NodeId={node_id} in settings.xml"))?;
    let probe = processor.descendants().find(|n| n.has_tag_name("NP_PROBE")).context("no NP_PROBE in settings.xml")?;
    let mut edits: Vec<(std::ops::Range<usize>, String)> = Vec::new();
    for el in probe.children().filter(|n| n.is_element() && !n.has_children()) {
        let ch_of = |name: &str| name.strip_prefix("CH").and_then(|n| n.parse::<usize>().ok());
        let by_ch: std::collections::HashMap<usize, &str> =
            el.attributes().filter_map(|a| Some((ch_of(a.name())?, a.value()))).collect();
        if by_ch.is_empty() {
            continue;
        }
        let mut tag = format!("<{}", el.tag_name().name());
        for a in el.attributes().filter(|a| ch_of(a.name()).is_none()) {
            tag.push_str(&format!(" {}=\"{}\"", a.name(), xml_escape(a.value())));
        }
        for (i, c) in kept.iter().enumerate() {
            if let Some(v) = by_ch.get(c) {
                tag.push_str(&format!(" CH{i}=\"{}\"", xml_escape(v)));
            }
        }
        tag.push_str("/>");
        edits.push((el.range(), tag));
    }
    let mut out = xml.to_string();
    edits.sort_by_key(|e| std::cmp::Reverse(e.0.start));
    for (r, t) in edits {
        out.replace_range(r, &t);
    }
    Ok(out)
}

/// Copy the per-sample part `[first, first + n)` of a 1-D `.npy` array of `n_total`
/// values (timestamps, sample numbers) to `dst`. `Ok(false)` if `src` isn't one.
fn slice_npy(src: &Path, dst: &Path, n_total: usize, first: usize, n: usize) -> Result<bool> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(src)?;
    let mut pre = [0u8; 12];
    if f.read_exact(&mut pre[..10]).is_err() || &pre[..6] != b"\x93NUMPY" {
        return Ok(false);
    }
    let (hlen, hstart) = if pre[6] == 1 {
        (u16::from_le_bytes([pre[8], pre[9]]) as usize, 10)
    } else {
        f.read_exact(&mut pre[10..12])?;
        (u32::from_le_bytes([pre[8], pre[9], pre[10], pre[11]]) as usize, 12)
    };
    let mut header = vec![0u8; hlen];
    f.read_exact(&mut header)?;
    let header = String::from_utf8_lossy(&header);
    let between = |key: &str, open: char, close: char| -> Option<String> {
        let i = header.find(key)? + key.len();
        let rest = &header[i..];
        let a = rest.find(open)? + 1;
        let b = a + rest[a..].find(close)?;
        Some(rest[a..b].to_string())
    };
    let (Some(descr), Some(shape)) = (between("'descr'", '\'', '\''), between("'shape'", '(', ')')) else {
        return Ok(false);
    };
    let dims: Vec<usize> = shape.split(',').filter_map(|d| d.trim().parse().ok()).collect();
    let item: usize = descr.trim_start_matches(|c: char| !c.is_ascii_digit()).parse().unwrap_or(0);
    if dims != [n_total] || item == 0 || header.contains("'fortran_order': True") {
        return Ok(false);
    }
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}");
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    let mut out = std::io::BufWriter::new(std::fs::File::create(dst)?);
    out.write_all(b"\x93NUMPY\x01\x00")?;
    out.write_all(&((dict.len() + pad + 1) as u16).to_le_bytes())?;
    out.write_all(dict.as_bytes())?;
    out.write_all(&vec![b' '; pad])?;
    out.write_all(b"\n")?;
    f.seek(SeekFrom::Start((hstart + hlen + first * item) as u64))?;
    std::io::copy(&mut f.take((n * item) as u64), &mut out)?;
    out.flush()?;
    Ok(true)
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let to = dst.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_dir(&e.path(), &to)?;
        } else {
            std::fs::copy(e.path(), &to)?;
        }
    }
    Ok(())
}

fn write_open_ephys_meta(info: &ExportInfo) -> Result<Vec<PathBuf>> {
    let job = info.job;
    let (oebin, settings) = crate::data::find_open_ephys_meta(&job.src).context("structure.oebin / settings.xml not found")?;
    let root = open_ephys_root(&job.src).context("session folder not found")?;
    let under = |p: &Path| -> Result<PathBuf> { Ok(job.dst.join(p.strip_prefix(&root)?)) };
    let kept: Vec<usize> = info.plan.out.iter().map(|o| o.ch).collect();
    let mut files = Vec::new();

    // structure.oebin: this stream only, with the exported channels
    let stream_dir = job.src.parent().context("no stream folder")?;
    let stream_name = stream_dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let text = std::fs::read_to_string(&oebin)?;
    let mut j: serde_json::Value = serde_json::from_str(&text).context("parsing structure.oebin")?;
    let mut node_id = 0u32;
    if let Some(cont) = j.get_mut("continuous").and_then(|c| c.as_array_mut()) {
        cont.retain(|c| c.get("folder_name").and_then(|f| f.as_str()).map(|f| f.trim_end_matches('/') == stream_name).unwrap_or(false));
        for c in cont.iter_mut() {
            node_id = c.get("source_processor_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let chans = c.get("channels").and_then(|v| v.as_array()).cloned().unwrap_or_default();
            c["num_channels"] = serde_json::json!(kept.len());
            c["channels"] = serde_json::Value::Array(kept.iter().filter_map(|&k| chans.get(k).cloned()).collect());
        }
    }
    let dst_oebin = under(&oebin)?;
    std::fs::create_dir_all(dst_oebin.parent().unwrap())?;
    std::fs::write(&dst_oebin, serde_json::to_string_pretty(&j)?)?;
    files.push(dst_oebin.clone());

    // settings.xml: the probe's channel lists reduced to the exported channels
    let xml = std::fs::read_to_string(&settings)?;
    let dst_settings = under(&settings)?;
    std::fs::create_dir_all(dst_settings.parent().unwrap())?;
    std::fs::write(&dst_settings, rewrite_settings_xml(&xml, node_id, &kept)?)?;
    files.push(dst_settings);

    // the recording folder's other files (sync_messages.txt, …) and its events
    let rec_dir = oebin.parent().context("no recording folder")?;
    for e in std::fs::read_dir(rec_dir)? {
        let e = e?;
        let path = e.path();
        if e.file_type()?.is_file() && path != oebin {
            std::fs::copy(&path, under(&path)?)?;
        } else if e.file_type()?.is_dir() && e.file_name() == "events" {
            copy_dir(&path, &under(&path)?)?;
        }
    }

    // the stream folder: per-sample arrays cut to the range, other files copied
    let n_total = job.meta.n_samples;
    for e in std::fs::read_dir(stream_dir)? {
        let e = e?;
        let path = e.path();
        if !e.file_type()?.is_file() || path == job.src {
            continue;
        }
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if matches!(ext, "dat" | "cbin" | "ch") {
            continue; // the data itself (written above) and mtscomp's index
        }
        let to = under(&path)?;
        if ext == "npy" && slice_npy(&path, &to, n_total, info.first, info.n_samples)? {
            files.push(to);
            continue;
        }
        std::fs::copy(&path, &to)?;
    }
    Ok(files)
}

// ---------------------------------------------------------------------------
// Provenance, settings for the export, events, probe file
// ---------------------------------------------------------------------------

/// `<data file stem><suffix>` next to the exported data file.
fn beside_data(info: &ExportInfo, suffix: &str) -> PathBuf {
    let stem = info.data_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    info.data_path.with_file_name(format!("{stem}{suffix}"))
}

/// UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
fn iso_utc(t: std::time::SystemTime) -> String {
    let secs = t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn write_companions(info: &ExportInfo) -> Result<Vec<PathBuf>> {
    let job = info.job;
    let mut files = Vec::new();
    let (t0, t1) = (info.first as f64 / job.meta.sample_rate, (info.first + info.n_samples) as f64 / job.meta.sample_rate);

    // events shifted to the export's start, so they line up with the new file
    let events_csv = match &job.events {
        Some(ev) => {
            let path = beside_data(info, ".events.csv");
            let mut text = String::from(if ev.offsets.is_some() { "onset_s,offset_s\n" } else { "onset_s\n" });
            for (i, &on) in ev.onsets.iter().enumerate() {
                if on < t0 || on >= t1 {
                    continue;
                }
                match ev.offsets.as_ref().and_then(|o| o.get(i)) {
                    Some(off) => text.push_str(&format!("{},{}\n", on - t0, off - t0)),
                    None => text.push_str(&format!("{}\n", on - t0)),
                }
            }
            std::fs::write(&path, text)?;
            files.push(path.clone());
            Some((path, if ev.offsets.is_some() { "header\no,f\n" } else { "header\no\n" }))
        }
        None => None,
    };

    // NPXplorer settings for the export: preprocessing off (it is in the data now),
    // nothing removed, the atlas registration and events carried over
    let band = crate::settings::Band::of_sample_rate(job.meta.sample_rate);
    let mut table = toml::Table::new();
    table.insert("removed_channels".into(), toml::Value::Array(Vec::new()));
    let mut preproc = toml::Table::new();
    for k in ["dc_removal", "phase_shift", "highpass", "avg_depths", "notch_enabled"] {
        preproc.insert(k.into(), toml::Value::Boolean(false));
    }
    preproc.insert("spatial_filter".into(), toml::Value::String("Off".into()));
    preproc.insert("notches".into(), toml::Value::Array(Vec::new()));
    let mut band_table = toml::Table::new();
    band_table.insert("preproc".into(), toml::Value::Table(preproc));
    table.insert(band.key().into(), toml::Value::Table(band_table));
    if let Some(atlas) = job.source_settings.get("atlas") {
        table.insert("atlas".into(), atlas.clone());
    }
    if let Some((path, layout)) = &events_csv {
        let mut ev = toml::Table::new();
        ev.insert("event_file".into(), toml::Value::String(path.to_string_lossy().into_owned()));
        ev.insert("layout".into(), toml::Value::String(layout.to_string()));
        table.insert("events".into(), toml::Value::Table(ev));
    }
    let sidecar = crate::settings::path(info.data_path);
    std::fs::write(&sidecar, toml::to_string_pretty(&table)?)?;
    files.push(sidecar);

    // Kilosort 4 probe file (its own JSON format: chanMap, xc, yc, kcoords, n_chans)
    if job.settings.probe_file {
        let out = &info.plan.out;
        let probe = serde_json::json!({
            "chanMap": (0..out.len()).collect::<Vec<_>>(),
            "xc": out.iter().map(|o| o.x_um).collect::<Vec<_>>(),
            "yc": out.iter().map(|o| o.y_um).collect::<Vec<_>>(),
            "kcoords": out.iter().map(|o| o.shank as f32).collect::<Vec<_>>(),
            "n_chans": out.len(),
        });
        let path = beside_data(info, ".kilosort4_probe.json");
        std::fs::write(&path, serde_json::to_string_pretty(&probe)?)?;
        files.push(path);
    }

    let path = match job.format {
        Format::SpikeGlx => beside_data(info, ".preprocessing.toml"),
        Format::OpenEphys => job.dst.join("preprocessing.toml"),
    };
    std::fs::write(&path, provenance(info, t0, t1)?)?;
    files.push(path);
    Ok(files)
}

/// The provenance file: where the export came from and every step applied, in order.
fn provenance(info: &ExportInfo, t0: f64, t1: f64) -> Result<String> {
    use toml::Value as V;
    let job = info.job;
    let meta = &*job.meta;
    let cfg = &job.cfg;
    let s = &job.settings;
    let ids = |chans: &[usize]| V::Array(chans.iter().map(|&c| V::String(meta.channel_id(c).to_string())).collect());
    let str_v = |x: &str| V::String(x.to_string());
    let mut t = toml::Table::new();

    let mut ex = toml::Table::new();
    ex.insert("app".into(), str_v(concat!("NPXplorer v", env!("CARGO_PKG_VERSION"))));
    ex.insert("date".into(), V::String(iso_utc(std::time::SystemTime::now())));
    ex.insert("format".into(), str_v(job.format.label()));
    ex.insert("source".into(), V::String(job.src.to_string_lossy().into_owned()));
    if let Ok(m) = std::fs::metadata(&job.src) {
        ex.insert("source_size_bytes".into(), V::Integer(m.len() as i64));
        if let Ok(mt) = m.modified() {
            ex.insert("source_modified".into(), V::String(iso_utc(mt)));
        }
    }
    ex.insert("output".into(), V::String(info.data_path.to_string_lossy().into_owned()));
    ex.insert("sample_rate_hz".into(), V::Float(meta.sample_rate));
    ex.insert("time_range_s".into(), V::Array(vec![V::Float(t0), V::Float(t1)]));
    ex.insert("first_sample_in_source".into(), V::Integer(info.first as i64));
    ex.insert("n_samples".into(), V::Integer(info.n_samples as i64));
    t.insert("export".into(), V::Table(ex));

    let mut ch = toml::Table::new();
    ch.insert("exported".into(), V::Array(info.plan.out.iter().map(|o| V::String(meta.channel_id(o.ch).to_string())).collect()));
    ch.insert("removed_by_user".into(), ids(&cfg.removed_channels.iter().copied().collect::<Vec<_>>()));
    ch.insert("removed_dead".into(), ids(&info.plan.removed_dead));
    ch.insert("removed_noisy".into(), ids(&info.plan.removed_noisy));
    ch.insert("removed_outside_brain".into(), ids(&info.plan.removed_outside));
    ch.insert("interpolated".into(), ids(&info.plan.interpolated));
    ch.insert("averaged_same_depth".into(), V::Boolean(s.average_depths));
    ch.insert(
        "copied_unchanged".into(),
        V::Array((meta.n_ap_chans..meta.n_saved_chans).map(|c| V::String(format!("sync (file channel {c})"))).collect()),
    );
    ch.insert("order".into(), str_v("file order of the source (each averaged channel at its first channel's place)"));
    t.insert("channels".into(), V::Table(ch));

    let mut steps: Vec<V> = Vec::new();
    let mut step = |name: &str, detail: String| {
        let mut st = toml::Table::new();
        st.insert("step".into(), str_v(name));
        st.insert("detail".into(), V::String(detail));
        steps.push(V::Table(st));
    };
    step("read", format!(
        "int16 × µV per bit{}{}",
        if cfg.phase_shift { "; ADC sampling delay corrected (fractional-delay filter, 16-tap Lanczos sinc)" } else { "" },
        if s.average_depths { "; channels at the same depth averaged" } else { "" },
    ));
    if let Some(dc) = info.dc {
        step("dc_removal", format!("fixed offset per channel: median of the means of {ESTIMATE_CHUNKS} 1 s chunks spread over the range ({} offsets)", dc.len()));
    }
    if s.blank.enabled {
        let b = &s.blank;
        let mut d = format!(
            "{} windows ({} after merging overlaps) of [{}, {}] ms around each onset",
            match b.mode { BlankMode::Zero => "set to 0:", BlankMode::Interpolate => "linearly interpolated:" },
            info.n_blanked,
            b.onset_start_ms,
            b.onset_end_ms,
        );
        if b.offsets {
            d.push_str(&format!(" and [{}, {}] ms around each offset", b.offset_start_ms, b.offset_end_ms));
            if job.events.as_ref().is_some_and(|e| e.offsets.is_none()) {
                d.push_str(&format!(" (offset = onset + {} ms)", b.event_length_ms));
            }
        }
        if let Some(f) = job.events.as_ref().and_then(|e| e.file.as_ref()) {
            d.push_str(&format!("; events from {}", f.display()));
        }
        step("event_blanking", d);
    }
    if cfg.notch_enabled && !cfg.notches.is_empty() {
        let list: Vec<String> = cfg.notches.iter().map(|n| format!("{} Hz (bw {} Hz)", n.freq_hz, n.bw_hz)).collect();
        step("notch", format!("zero-phase biquad notches: {}", list.join(", ")));
    }
    if cfg.highpass {
        step("highpass", "Butterworth, order 3, 300 Hz, zero-phase (forward-backward, scipy sosfiltfilt edge handling)".into());
    }
    match cfg.spatial_filter {
        SpatialFilter::Off => {}
        SpatialFilter::GlobalCmr => step("spatial_filter", "global CMR: median over the shank's channels subtracted, per sample".into()),
        SpatialFilter::LocalCmr => step("spatial_filter", "local CMR: median of the channels 100–400 µm away subtracted, per sample".into()),
        SpatialFilter::Destripe => step("spatial_filter", format!(
            "destripe (IBL kfilt): AGC over {} samples, spatial Butterworth highpass order 3 at 0.01 along depth, per run of adjacent channels; AGC floors (fixed, median over {ESTIMATE_CHUNKS} chunks): {:?}",
            Filters::new(cfg).kfilt_lagc,
            info.eps.unwrap_or(&[]),
        )),
    }
    if !info.plan.interpolated.is_empty() {
        step("interpolate_bad_channels", format!(
            "IBL interpolate_bad_channels: weighted sum of the preprocessed channels of the same shank, weights exp(-(d / {KRIGING_DISTANCE_UM} µm)^{KRIGING_P}), weights < {KRIGING_MIN_WEIGHT} dropped"
        ));
    }
    step("quantization", format!("rounded to int16 at each channel's original µV per bit; {} samples clipped", info.clipped));
    t.insert("steps".into(), V::Array(steps));

    let mut not = toml::Table::new();
    not.insert("noise_suppression".into(), str_v("display-only filter, not exported"));
    t.insert("not_applied".into(), V::Table(not));

    let mut tech = toml::Table::new();
    tech.insert("chunk_samples".into(), V::Integer(info.chunk_len as i64));
    tech.insert("chunk_margin_samples".into(), V::Integer(info.margin as i64));
    t.insert("processing".into(), V::Table(tech));

    Ok(format!(
        "# How this recording was preprocessed by NPXplorer. Steps are listed in the order they ran.\n{}",
        toml::to_string_pretty(&t)?
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::SpatialFilter;

    const N_NEURAL: usize = 32;
    const FS: f64 = 30_000.0;
    const UV_PER_BIT: f32 = (0.6 / 512.0 / 500.0 * 1e6) as f32;

    /// A small SpikeGLX recording: 2 shanks × 16 channels (2 columns, 20 µm pitch,
    /// interleaved in the file like real 4-shank probes), plus a sync channel; noise,
    /// a per-channel DC offset, a line tone and a few spikes.
    fn write_recording(dir: &Path, n_samples: usize) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let bin = dir.join("rec_g0_t0.imec0.ap.bin");
        let n_ch = N_NEURAL + 1;
        let mut data = vec![0i16; n_samples * n_ch];
        let mut rng = 12345u64;
        let mut rand = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for t in 0..n_samples {
            let tone = 20.0 * (2.0 * std::f64::consts::PI * 50.0 * t as f64 / FS).sin();
            for c in 0..N_NEURAL {
                let spike = if t % 7919 < 20 && c % 5 == 0 { -150.0 * (1.0 - (t % 7919) as f64 / 20.0) } else { 0.0 };
                let uv = 30.0 * rand() + tone + 100.0 * c as f64 + spike;
                data[t * n_ch + c] = (uv / UV_PER_BIT as f64).round() as i16;
            }
            data[t * n_ch + N_NEURAL] = ((t / 1000) % 2) as i16 * 64;
        }
        std::fs::write(&bin, bytemuck::cast_slice::<i16, u8>(&data)).unwrap();

        // file order: shank 1's channels first, then shank 0's (like the real probe)
        let geom = |c: usize| {
            let shank = if c < 16 { 1 } else { 0 };
            let i = c % 16;
            (shank, if i % 2 == 0 { 27 } else { 59 }, (i / 2) * 20)
        };
        let mut chan_map = format!("({N_NEURAL},0,1)");
        let mut geom_map = String::from("(NP2014,2,250,70)");
        for c in 0..N_NEURAL {
            chan_map.push_str(&format!("(AP{c};{c}:{c})"));
            let (s, x, y) = geom(c);
            geom_map.push_str(&format!("({s}:{x}:{y}:1)"));
        }
        chan_map.push_str("(SY0;768:768)");
        let meta = format!(
            "nSavedChans={n_ch}\nimSampRate={FS}\nfileSizeBytes={}\nfileTimeSecs={}\nfileSHA1=00\nfirstSample=1000\n\
             imAiRangeMax=0.6\nimMaxInt=512\nimChan0apGain=500\nimDatPrb_type=0\nsnsApLfSy={N_NEURAL},0,1\n\
             snsSaveChanSubset=0:{},768\n~snsChanMap={chan_map}\n~snsGeomMap={geom_map}\n",
            n_samples * n_ch * 2,
            n_samples as f64 / FS,
            N_NEURAL - 1,
        );
        std::fs::write(bin.with_extension("meta"), meta).unwrap();
        bin
    }

    fn job(src: &Path, dst: &Path, cfg: PreprocConfig, settings: ExportSettings, chunk: Option<usize>) -> Job {
        let meta = Arc::new(Meta::from_data_path(src).unwrap());
        let raw = Arc::new(crate::data::open_data(src, &meta).unwrap());
        Job {
            src: src.to_path_buf(),
            dst: dst.to_path_buf(),
            format: Format::of(src),
            raw,
            meta,
            cfg,
            settings,
            labels: None,
            classify_chunks: 4,
            outside_rule: Default::default(),
            events: None,
            source_settings: toml::Table::new(),
            chunk_samples: chunk,
        }
    }

    fn cfg(meta_fs: f64) -> PreprocConfig {
        PreprocConfig {
            dc_removal: true,
            phase_shift: true,
            highpass: true,
            spatial_filter: SpatialFilter::Destripe,
            avg_depths: false,
            sample_rate: meta_fs,
            removed_channels: BTreeSet::new(),
            channel_order: Default::default(),
            shank_order: Default::default(),
            notches: vec![crate::notch::Notch { freq_hz: 50.0, bw_hz: 2.0 }],
            notch_enabled: true,
        }
    }

    fn run_job(j: &Job) -> Done {
        let mut labels = None;
        run(j, &Progress::default(), &AtomicBool::new(false), &mut labels).unwrap()
    }

    fn read_i16(p: &Path) -> Vec<i16> {
        bytemuck::cast_slice::<u8, i16>(&std::fs::read(p).unwrap()).to_vec()
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("npx_export_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn chunk_boundaries_do_not_show() {
        let dir = tmp("chunks");
        let src = write_recording(&dir, 60_000);
        let mut c = cfg(FS);
        c.removed_channels = [3usize, 20].into_iter().collect();
        let settings = ExportSettings { range_s: Some([0.2, 1.9]), ..Default::default() };
        // many short chunks vs one piece: every seam must be invisible (≤ 1 LSB of
        // rounding, from float summation order)
        let a = dir.join("a_g0_t0.imec0.ap.bin");
        let b = dir.join("b_g0_t0.imec0.ap.bin");
        run_job(&job(&src, &a, c.clone(), settings.clone(), Some(4_111)));
        run_job(&job(&src, &b, c, settings, Some(1_000_000)));
        let (x, y) = (read_i16(&a), read_i16(&b));
        assert_eq!(x.len(), y.len());
        let max = x.iter().zip(&y).map(|(p, q)| (*p as i32 - *q as i32).abs()).max().unwrap();
        assert!(max <= 1, "max difference {max} LSB");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn spikeglx_metadata_describes_the_export() {
        let dir = tmp("meta");
        let src = write_recording(&dir, 30_000);
        let mut c = cfg(FS);
        c.removed_channels = [0usize, 17].into_iter().collect();
        let dst = default_destination(&src, Format::SpikeGlx, Some([0.1, 0.6]));
        assert_eq!(dst.file_name().unwrap(), "rec_preprocessed_0.1-0.6s_g0_t0.imec0.ap.bin");
        let settings = ExportSettings { range_s: Some([0.1, 0.6]), probe_file: true, ..Default::default() };
        let done = run_job(&job(&src, &dst, c, settings, None));

        let src_meta = Meta::from_data_path(&src).unwrap();
        let m = Meta::from_data_path(&dst).unwrap();
        assert_eq!(m.n_ap_chans, N_NEURAL - 2);
        assert_eq!(m.n_saved_chans, N_NEURAL - 1);
        assert_eq!(m.n_samples, 15_000);
        assert_eq!(done.n_channels, N_NEURAL - 2);
        // channels keep their names and positions
        let kept: Vec<usize> = (0..N_NEURAL).filter(|c| *c != 0 && *c != 17).collect();
        for (i, &c) in kept.iter().enumerate() {
            assert_eq!(m.channel_ids[i], src_meta.channel_ids[c]);
            assert_eq!(m.channel_geom[i].y_um, src_meta.channel_geom[c].y_um);
            assert_eq!(m.channel_geom[i].shank, src_meta.channel_geom[c].shank);
            assert_eq!(m.uv_per_bit[i], src_meta.uv_per_bit[c]);
        }
        let text = std::fs::read_to_string(dst.with_extension("meta")).unwrap();
        let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(&format!("{k}="))).unwrap().to_string();
        assert_eq!(get("firstSample"), "4000");
        assert_eq!(get("snsSaveChanSubset"), "1:16,18:31,768");
        assert!(get("~snsChanMap").starts_with("(32,0,1)(AP1;1:1)") && get("~snsChanMap").ends_with("(SY0;768:768)"));
        {
            use sha1::Digest;
            let digest: String = sha1::Sha1::digest(std::fs::read(&dst).unwrap()).iter().map(|b| format!("{b:02X}")).collect();
            assert_eq!(get("fileSHA1"), digest);
        }
        // the sync channel is copied unchanged
        let out = read_i16(&dst);
        let raw = read_i16(&src);
        let n_out_cols = N_NEURAL - 1;
        for t in (0..15_000).step_by(997) {
            assert_eq!(out[t * n_out_cols + n_out_cols - 1], raw[(3000 + t) * (N_NEURAL + 1) + N_NEURAL]);
        }
        // the export opens without preprocessing and with nothing removed
        let side = crate::settings::load_table(&dst);
        assert_eq!(side["ap"]["preproc"]["highpass"].as_bool(), Some(false));
        assert_eq!(side["removed_channels"].as_array().map(|a| a.len()), Some(0));
        assert!(beside_data_path(&dst, ".kilosort4_probe.json").is_file());
        assert!(beside_data_path(&dst, ".preprocessing.toml").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn beside_data_path(data: &Path, suffix: &str) -> PathBuf {
        let stem = data.file_stem().unwrap().to_string_lossy().into_owned();
        data.with_file_name(format!("{stem}{suffix}"))
    }

    #[test]
    fn source_is_never_overwritten() {
        let dir = tmp("overwrite");
        let src = write_recording(&dir, 3_000);
        let mut labels = None;
        let r = run(&job(&src, &src, cfg(FS), ExportSettings::default(), None), &Progress::default(), &AtomicBool::new(false), &mut labels);
        assert!(r.is_err());
        assert_eq!(read_i16(&src).len(), 3_000 * (N_NEURAL + 1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn channel_plan_removes_interpolates_and_averages() {
        let dir = tmp("plan");
        let src = write_recording(&dir, 1_000);
        let meta = Meta::from_data_path(&src).unwrap();
        let mut labels = vec![0u8; N_NEURAL];
        labels[4] = 1; // dead
        labels[5] = 2; // noisy
        labels[30] = 3; // outside
        let s = ExportSettings {
            dead: BadChannelAction::Interpolate,
            noisy: BadChannelAction::Remove,
            remove_outside: true,
            ..Default::default()
        };
        let p = plan_channels(&meta, &[7usize].into_iter().collect(), false, Some(&labels), &s);
        let chans: Vec<usize> = p.out.iter().map(|o| o.ch).collect();
        assert!(!chans.contains(&5) && !chans.contains(&7) && !chans.contains(&30));
        assert_eq!(p.interpolated, vec![4]);
        let interp = p.out.iter().find(|o| o.ch == 4).unwrap();
        let Source::Mix(w) = &interp.src else { panic!("channel 4 not interpolated") };
        assert!((w.iter().map(|p| p.1).sum::<f32>() - 1.0).abs() < 1e-5 && !w.is_empty());
        assert!(chans.windows(2).all(|w| w[0] < w[1]), "file order");

        // averaging: one channel per depth (8 depths × 2 shanks). Channel 30's depth
        // keeps channel 31; channel 4's depth lost both channels (4 interpolated,
        // 5 removed), so it is interpolated as a whole
        let p = plan_channels(&meta, &BTreeSet::new(), true, Some(&labels), &s);
        assert_eq!(p.out.len(), 16);
        let mixed: Vec<usize> = p.out.iter().filter(|o| matches!(o.src, Source::Mix(_))).map(|o| o.ch).collect();
        assert_eq!(mixed, vec![4]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn blanking_windows_and_modes() {
        let ev = EventTimes { onsets: vec![0.010, 0.0105, 0.050], offsets: None, file: None };
        let b = BlankSettings { enabled: true, onset_start_ms: -1.0, onset_end_ms: 1.0, offsets: true, offset_start_ms: 0.0, offset_end_ms: 1.0, event_length_ms: 20.0, ..Default::default() };
        let iv = blank_intervals(&ev, &b, 1000.0, 1000);
        // onsets 10 and 10.5 ms merge; offsets at 30 / 30.5 / 70 ms
        assert_eq!(iv, vec![(9, 12), (30, 32), (49, 51), (70, 71)]);

        let mut row: Vec<f32> = (0..20).map(|i| i as f32).collect();
        apply_blanking(&mut row, 20, 100, &[(105, 108)], BlankMode::Interpolate);
        assert_eq!(&row[4..9], &[4.0, 5.0, 6.0, 7.0, 8.0]); // a line stays a line
        let mut row = vec![1.0f32; 20];
        apply_blanking(&mut row, 20, 100, &[(95, 102), (118, 130)], BlankMode::Zero);
        assert!(row[..2].iter().chain(&row[18..]).all(|&v| v == 0.0) && row[2..18].iter().all(|&v| v == 1.0));
    }

    #[test]
    fn names_keep_the_spikeglx_suffix() {
        let p = Path::new("/d/run_x_g0_t12.imec1.lf.cbin");
        assert_eq!(
            default_destination(p, Format::SpikeGlx, None).file_name().unwrap(),
            "run_x_preprocessed_g0_t12.imec1.lf.bin"
        );
        assert_eq!(name_tag(Some([120.0, 300.25])), "_preprocessed_120-300.25s");
        assert_eq!(compress_subset(&[0, 1, 2, 5, 7, 8]), "0:2,5,7:8");
        assert_eq!(expand_subset("0:2,5").unwrap(), vec![0, 1, 2, 5]);
    }

    #[test]
    fn settings_xml_channel_lists_are_renumbered() {
        let xml = r#"<S><PROCESSOR NodeId="100"><EDITOR><NP_PROBE probe_part_number="PRB_1_4_0480_1_C">
<CHANNELS CH0="0" CH1="0" CH2="1" CH3="0"/>
<ELECTRODE_XPOS CH0="27" CH1="59" CH2="11" CH3="43"/>
<ELECTRODE_YPOS CH0="0" CH1="0" CH3="20"/>
</NP_PROBE></EDITOR></PROCESSOR></S>"#;
        let out = rewrite_settings_xml(xml, 100, &[1, 2, 3]).unwrap();
        assert!(out.contains(r#"<ELECTRODE_XPOS CH0="59" CH1="11" CH2="43"/>"#), "{out}");
        assert!(out.contains(r#"<ELECTRODE_YPOS CH0="0" CH2="20"/>"#), "{out}");
        assert!(out.contains(r#"<CHANNELS CH0="0" CH1="1" CH2="0"/>"#), "{out}");
        assert!(out.contains(r#"probe_part_number="PRB_1_4_0480_1_C""#));
    }

    #[test]
    fn npy_per_sample_arrays_are_cut() {
        let dir = tmp("npy");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("ts.npy");
        let dict = "{'descr': '<i8', 'fortran_order': False, 'shape': (100,), }";
        let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend(((dict.len() + pad + 1) as u16).to_le_bytes());
        bytes.extend(dict.as_bytes());
        bytes.extend(vec![b' '; pad]);
        bytes.push(b'\n');
        for i in 0..100i64 {
            bytes.extend((5000 + i).to_le_bytes());
        }
        std::fs::write(&src, bytes).unwrap();
        let dst = dir.join("cut.npy");
        assert!(slice_npy(&src, &dst, 100, 10, 5).unwrap());
        let out = std::fs::read(&dst).unwrap();
        assert_eq!(out.len() % 8, 0);
        let vals: Vec<i64> = out[out.len() - 40..].chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(vals, vec![5010, 5011, 5012, 5013, 5014]);
        assert!(String::from_utf8_lossy(&out[..128]).contains("'shape': (5,)"));
        assert!(!slice_npy(&src, &dir.join("x.npy"), 99, 0, 5).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An Open Ephys session: Record Node/settings.xml, structure.oebin, one NP 1.0
    /// stream with continuous.dat and per-sample timestamps, and an events folder.
    fn write_open_ephys(root: &Path, n_ch: usize, n_samples: usize) -> PathBuf {
        let rec = root.join("Record Node 101/experiment1/recording1");
        let stream = rec.join("continuous/Neuropix-PXI-100.0");
        std::fs::create_dir_all(&stream).unwrap();
        std::fs::create_dir_all(rec.join("events/Neuropix-PXI-100.0/TTL")).unwrap();
        std::fs::write(rec.join("events/Neuropix-PXI-100.0/TTL/states.npy"), b"x").unwrap();
        std::fs::write(rec.join("sync_messages.txt"), "start time: 0\n").unwrap();
        let attrs = |f: &dyn Fn(usize) -> String| (0..n_ch).map(|c| format!(" CH{c}=\"{}\"", f(c))).collect::<String>();
        let xml = format!(
            "<SETTINGS><SIGNALCHAIN><PROCESSOR name=\"Neuropix-PXI\" NodeId=\"100\"><EDITOR><NP_PROBE probe_part_number=\"PRB_1_4_0480_1_C\">\
             <CHANNELS{}/><ELECTRODE_XPOS{}/><ELECTRODE_YPOS{}/></NP_PROBE></EDITOR></PROCESSOR></SIGNALCHAIN></SETTINGS>",
            attrs(&|_| "0".into()),
            attrs(&|c| if c % 2 == 0 { "27".into() } else { "59".into() }),
            attrs(&|c| ((c / 2) * 20).to_string()),
        );
        std::fs::write(root.join("Record Node 101/settings.xml"), xml).unwrap();
        let channels: Vec<serde_json::Value> = (0..n_ch)
            .map(|c| serde_json::json!({"channel_name": format!("CH{}", c + 1), "bit_volts": 0.195, "units": "uV"}))
            .collect();
        let oebin = serde_json::json!({
            "GUI version": "0.6.4",
            "continuous": [{"folder_name": "Neuropix-PXI-100.0/", "sample_rate": 30000.0, "source_processor_id": 100, "num_channels": n_ch, "channels": channels}],
            "events": [], "spikes": []
        });
        std::fs::write(rec.join("structure.oebin"), oebin.to_string()).unwrap();
        let data: Vec<i16> = (0..n_samples * n_ch).map(|i| ((i * 7919) % 200) as i16 - 100).collect();
        std::fs::write(stream.join("continuous.dat"), bytemuck::cast_slice::<i16, u8>(&data)).unwrap();
        let dict = format!("{{'descr': '<i8', 'fortran_order': False, 'shape': ({n_samples},), }}");
        let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
        let mut npy = b"\x93NUMPY\x01\x00".to_vec();
        npy.extend(((dict.len() + pad + 1) as u16).to_le_bytes());
        npy.extend(dict.as_bytes());
        npy.extend(vec![b' '; pad]);
        npy.push(b'\n');
        for i in 0..n_samples as i64 {
            npy.extend((1_000_000 + i).to_le_bytes());
        }
        std::fs::write(stream.join("sample_numbers.npy"), npy).unwrap();
        stream.join("continuous.dat")
    }

    #[test]
    fn open_ephys_round_trip() {
        let dir = tmp("oe");
        let session = dir.join("2024-01-01_10-00-00");
        let src = write_open_ephys(&session, 16, 30_000);
        let dst = default_destination(&src, Format::OpenEphys, Some([0.25, 0.75]));
        assert_eq!(dst, dir.join("2024-01-01_10-00-00_preprocessed_0.25-0.75s"));
        let mut c = cfg(FS);
        c.spatial_filter = SpatialFilter::GlobalCmr;
        c.removed_channels = [2usize, 9].into_iter().collect();
        let settings = ExportSettings { range_s: Some([0.25, 0.75]), ..Default::default() };
        let done = run_job(&job(&src, &dst, c, settings, Some(3_001)));
        assert_eq!(done.data_path, dst.join("Record Node 101/experiment1/recording1/continuous/Neuropix-PXI-100.0/continuous.dat"));
        assert!(!dst.join(CREATED_MARKER).exists());

        let src_meta = Meta::from_data_path(&src).unwrap();
        let m = Meta::from_data_path(&done.data_path).unwrap();
        assert_eq!((m.n_ap_chans, m.n_samples), (14, 15_000));
        let kept: Vec<usize> = (0..16).filter(|c| *c != 2 && *c != 9).collect();
        for (i, &k) in kept.iter().enumerate() {
            assert_eq!(m.channel_ids[i], src_meta.channel_ids[k]);
            assert_eq!((m.channel_geom[i].x_um, m.channel_geom[i].y_um), (src_meta.channel_geom[k].x_um, src_meta.channel_geom[k].y_um));
        }
        // timestamps cut to the range, events and other files copied
        let npy = std::fs::read(done.data_path.with_file_name("sample_numbers.npy")).unwrap();
        let first = i64::from_le_bytes(npy[npy.len() - 15_000 * 8..][..8].try_into().unwrap());
        assert_eq!(first, 1_000_000 + 7_500);
        assert!(dst.join("Record Node 101/experiment1/recording1/events/Neuropix-PXI-100.0/TTL/states.npy").is_file());
        assert!(dst.join("Record Node 101/experiment1/recording1/sync_messages.txt").is_file());
        assert!(dst.join("preprocessing.toml").is_file());

        // a second export into the same (now non-empty) folder is refused, and the
        // failed attempt leaves the first export alone
        let mut labels = None;
        let j = job(&src, &dst, cfg(FS), ExportSettings::default(), None);
        let r = run(&j, &Progress::default(), &AtomicBool::new(false), &mut labels);
        assert!(r.is_err());
        cleanup(&j);
        assert!(done.data_path.is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn iso_dates() {
        assert_eq!(iso_utc(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_759_910_400)), "2025-10-08T08:00:00Z");
    }
}
