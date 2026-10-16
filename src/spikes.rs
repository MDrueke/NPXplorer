// Spike-sorted data (phy / Kilosort output folder): every spike is drawn as a dot on
// the heatmap, at its time and on the peak channel of its template.
//
// Files read (see https://phy.readthedocs.io): spike_times.npy (samples),
// spike_templates.npy, spike_clusters.npy (optional, after curation), templates.npy
// (n_templates x n_time x n_temp_channels), templates_ind.npy (optional),
// channel_map.npy (row of each sorted channel in the data file), params.py
// (sample_rate, n_channels_dat), and the cluster tables (cluster_*.tsv) for filtering
// the units: phy's curation label (`group`), Kilosort's label (`KSLabel`) and any
// other column.

use anyhow::{bail, Context, Result};
use egui::Color32;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use crate::colormap::ColorMapChoice;
use crate::data::{DisplayRow, Meta};

/// More dots than this in the view are not drawn (zoom in to see them).
const MAX_DOTS: usize = 20_000;
/// dot radius (px) in a `DOT_REF_DUR_S` window; it grows in proportion as the time
/// window shrinks, up to `DOT_MAX_RADIUS`
const DOT_RADIUS: f32 = 1.25;
const DOT_REF_DUR_S: f64 = 0.3;
const DOT_MIN_RADIUS: f32 = 0.75;
const DOT_MAX_RADIUS: f32 = 5.0;

const VALUE_HELP: &str = "Units whose value in this column matches are kept. \
Text: one value, or several separated by commas (e.g. Th,CA1). \
Numbers: a comparison, e.g. >50, <=10, =3, !=0.\n\n\
The column is looked up in the cluster tables (cluster_*.tsv, e.g. a label added in phy, or cluster_info.tsv). \
Leave the column empty to skip this filter.";

/// Filter on a label column (`group`, `KSLabel`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum LabelFilter {
    Good,
    GoodMua,
    All,
}

impl LabelFilter {
    const ALL: [LabelFilter; 3] = [LabelFilter::Good, LabelFilter::GoodMua, LabelFilter::All];

    fn name(self) -> &'static str {
        match self {
            LabelFilter::Good => "good",
            LabelFilter::GoodMua => "good + mua",
            LabelFilter::All => "all",
        }
    }

    fn accepts(self, label: &str) -> bool {
        match self {
            LabelFilter::Good => label == "good",
            LabelFilter::GoodMua => label == "good" || label == "mua",
            LabelFilter::All => true,
        }
    }
}

/// Spike-sorted data settings, saved per recording (see settings.rs); the filters
/// last used are the default for recordings opened for the first time.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SpikeSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub folder: Option<PathBuf>,
    /// phy's curation label
    pub group: LabelFilter,
    /// Kilosort's label
    pub ks_label: LabelFilter,
    pub user_column: String,
    pub user_value: String,
}

impl Default for SpikeSettings {
    fn default() -> Self {
        Self {
            folder: None,
            group: LabelFilter::Good,
            ks_label: LabelFilter::All,
            user_column: String::new(),
            user_value: String::new(),
        }
    }
}

pub struct SortedSpikes {
    dir: PathBuf,
    /// sorted
    times_s: Vec<f64>,
    /// 0-based channel in file order, same order as `times_s`
    chans: Vec<u16>,
    clusters: Vec<u32>,
    /// every cluster with spikes, ascending
    units: Vec<u32>,
}

/// Whether `dir` looks like a phy folder.
pub fn is_phy_dir(dir: &Path) -> bool {
    dir.join("spike_times.npy").is_file()
}

pub fn load(dir: &Path, meta: &Meta) -> Result<SortedSpikes> {
    let params = read_params(&dir.join("params.py"));
    let fs = match params.sample_rate {
        Some(fs) => fs,
        // without params.py, only an AP-band recording has the right sample rate
        None if meta.sample_rate >= 10_000.0 => meta.sample_rate,
        None => bail!("params.py with the sorting's sample_rate not found in {}", dir.display()),
    };
    if let Some(n) = params.n_channels_dat {
        if n != meta.n_saved_chans && n != meta.n_ap_chans {
            bail!(
                "the sorted file has {n} channels, this recording {}. Open the file that was sorted.",
                meta.n_saved_chans
            );
        }
    }

    let (_, times) = read_npy(&dir.join("spike_times.npy"))?;
    let (_, spike_templates) = read_npy(&dir.join("spike_templates.npy"))?;
    if spike_templates.len() != times.len() {
        bail!("spike_templates.npy and spike_times.npy differ in length");
    }
    let clusters = if dir.join("spike_clusters.npy").is_file() {
        let (_, c) = read_npy(&dir.join("spike_clusters.npy"))?;
        if c.len() != times.len() {
            bail!("spike_clusters.npy and spike_times.npy differ in length");
        }
        c
    } else {
        spike_templates.clone()
    };
    let (_, chan_map) = read_npy(&dir.join("channel_map.npy"))?;
    let chan_map: Vec<usize> = chan_map.iter().map(|&c| c as usize).collect();
    if let Some(&m) = chan_map.iter().max() {
        if m >= meta.n_ap_chans {
            bail!("channel_map.npy refers to channel {m}, this recording has {}", meta.n_ap_chans);
        }
    }

    let peak = template_peak_channels(dir, &chan_map)?;

    let mut idx: Vec<usize> = (0..times.len()).collect();
    if times.windows(2).any(|w| w[1] < w[0]) {
        idx.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
    }
    let mut times_s = Vec::with_capacity(idx.len());
    let mut chans = Vec::with_capacity(idx.len());
    let mut spike_clusters = Vec::with_capacity(idx.len());
    for &i in &idx {
        let t = spike_templates[i] as usize;
        let ch = *peak.get(t).with_context(|| format!("spike template {t} not in templates.npy"))?;
        times_s.push(times[i] / fs);
        chans.push(ch as u16);
        spike_clusters.push(clusters[i] as u32);
    }
    let mut units = spike_clusters.clone();
    units.sort_unstable();
    units.dedup();
    Ok(SortedSpikes { dir: dir.to_path_buf(), times_s, chans, clusters: spike_clusters, units })
}

/// Recording channel with the largest peak-to-peak amplitude of each template. Kilosort's
/// templates are whitened: like phy, they are unwhitened with whitening_mat_inv.npy
/// first (when present), which puts the peak on the channel where the spike is
/// largest in the recording.
fn template_peak_channels(dir: &Path, chan_map: &[usize]) -> Result<Vec<usize>> {
    let path = dir.join("templates.npy");
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let h = crate::atlas::parse_npy_header(&bytes).with_context(|| path.display().to_string())?;
    if h.shape.len() != 3 || h.fortran_order {
        bail!("templates.npy: expected a C-order 3-D array, found shape {:?}", h.shape);
    }
    let (n_t, n_time, n_ch) = (h.shape[0], h.shape[1], h.shape[2]);
    let (size, get) = decoder(&h.descr)?;
    let data = &bytes[h.data_offset..];
    if data.len() < n_t * n_time * n_ch * size {
        bail!("templates.npy is truncated");
    }

    // templates_ind only matters for sparse templates: when they span every channel of
    // channel_map it is the identity (and iblsorter writes it transposed)
    let ind = if n_ch == chan_map.len() {
        None
    } else if dir.join("templates_ind.npy").is_file() {
        let (shape, v) = read_npy(&dir.join("templates_ind.npy"))?;
        if shape != [n_t, n_ch] {
            bail!("templates_ind.npy has shape {shape:?}, expected [{n_t}, {n_ch}]");
        }
        Some(v)
    } else {
        bail!("templates.npy has {n_ch} channels, channel_map.npy {}", chan_map.len());
    };

    // whitening_mat_inv as sparse columns: (row, weight) of its nonzeros, per column
    let wmi_path = dir.join("whitening_mat_inv.npy");
    let unwhiten: Option<Vec<Vec<(usize, f64)>>> = if ind.is_none() && wmi_path.is_file() {
        let (shape, w) = read_npy(&wmi_path)?;
        if shape != [n_ch, n_ch] {
            bail!("whitening_mat_inv.npy has shape {shape:?}, expected [{n_ch}, {n_ch}]");
        }
        Some((0..n_ch).map(|d| (0..n_ch).filter(|&c| w[c * n_ch + d] != 0.0).map(|c| (c, w[c * n_ch + d])).collect()).collect())
    } else {
        None
    };

    use rayon::prelude::*;
    (0..n_t)
        .into_par_iter()
        .map(|t| {
            let base = t * n_time * n_ch * size;
            let x: Vec<f64> = (0..n_time * n_ch).map(|j| get(&data[base + j * size..])).collect();
            let mut lo = vec![f64::INFINITY; n_ch];
            let mut hi = vec![f64::NEG_INFINITY; n_ch];
            for s in 0..n_time {
                let row = &x[s * n_ch..(s + 1) * n_ch];
                for d in 0..n_ch {
                    let v = match &unwhiten {
                        Some(cols) => cols[d].iter().map(|&(c, w)| row[c] * w).sum(),
                        None => row[d],
                    };
                    lo[d] = lo[d].min(v);
                    hi[d] = hi[d].max(v);
                }
            }
            let best = (0..n_ch).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap_or(0);
            let k = ind.as_ref().map_or(best, |ind| ind[t * n_ch + best] as usize);
            chan_map.get(k).copied().with_context(|| format!("template channel {k} not in channel_map.npy"))
        })
        .collect()
}

/// Every column of the folder's cluster tables: column -> cluster id -> value. The
/// single-column files phy writes (cluster_group.tsv, cluster_<label>.tsv, ...) take
/// precedence over cluster_info.tsv; copies like "cluster_info (copy 1).tsv" are
/// ignored.
fn cluster_columns(dir: &Path) -> HashMap<String, HashMap<u32, String>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            name.starts_with("cluster_")
                && name.ends_with(".tsv")
                && (name == "cluster_info.tsv" || !name.starts_with("cluster_info"))
        })
        .collect();
    // cluster_info.tsv first, so the others overwrite it
    files.sort_by_key(|p| p.file_name().is_none_or(|n| n != "cluster_info.tsv"));

    let mut cols: HashMap<String, HashMap<u32, String>> = HashMap::new();
    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().unwrap_or_default().split('\t').map(str::trim).collect();
        let Some(id_col) = header.iter().position(|&h| h == "cluster_id" || h == "id") else { continue };
        for line in lines {
            let fields: Vec<&str> = line.split('\t').collect();
            let Some(id) = fields.get(id_col).and_then(|v| v.trim().parse::<u32>().ok()) else { continue };
            for (j, col) in header.iter().enumerate() {
                if j != id_col {
                    let v = fields.get(j).map_or("", |v| v.trim());
                    cols.entry(col.to_string()).or_default().insert(id, v.to_string());
                }
            }
        }
    }
    cols
}

/// The user filter's value: a list of texts, or a comparison with a number.
enum ValueMatch {
    Any(Vec<String>),
    Cmp(&'static str, f64),
}

impl ValueMatch {
    fn parse(text: &str, column: &str) -> Result<Self> {
        let v = text.trim();
        if v.is_empty() {
            bail!("enter a value for column '{column}'");
        }
        for op in [">=", "<=", "!=", ">", "<", "="] {
            if let Some(rest) = v.strip_prefix(op) {
                let x = rest.trim().parse::<f64>().with_context(|| format!("'{}' after '{op}' is not a number", rest.trim()))?;
                return Ok(ValueMatch::Cmp(op, x));
            }
        }
        Ok(ValueMatch::Any(v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()))
    }

    fn matches(&self, v: &str) -> bool {
        match self {
            ValueMatch::Any(list) => list.iter().any(|w| w == v),
            ValueMatch::Cmp(op, x) => v.parse::<f64>().is_ok_and(|y| match *op {
                ">=" => y >= *x,
                "<=" => y <= *x,
                "!=" => y != *x,
                ">" => y > *x,
                "<" => y < *x,
                _ => y == *x,
            }),
        }
    }

    /// Fails if a listed value occurs nowhere in the column, or the column has no
    /// numbers to compare.
    fn check(&self, column: &str, values: &HashMap<u32, String>) -> Result<()> {
        match self {
            ValueMatch::Any(list) => {
                let missing: Vec<&str> =
                    list.iter().filter(|w| !values.values().any(|v| v == *w)).map(|w| w.as_str()).collect();
                if !missing.is_empty() {
                    bail!("value '{}' not found in column '{column}'", missing.join("', '"));
                }
            }
            ValueMatch::Cmp(..) => {
                if !values.values().any(|v| v.parse::<f64>().is_ok()) {
                    bail!("column '{column}' has no numbers to compare");
                }
            }
        }
        Ok(())
    }
}

/// Shown when the curation-label filter is skipped.
const NO_GROUP_NOTE: &str = "not curated yet (no group column): curation filter skipped";

/// The units that pass the filters, in order: `group`, `KSLabel`, the user column; and
/// a note if the curation filter was skipped because the sorting isn't curated yet.
fn filter_units(dir: &Path, units: &[u32], s: &SpikeSettings) -> Result<(Vec<u32>, Option<&'static str>)> {
    let cols = cluster_columns(dir);
    let mut keep = units.to_vec();
    let mut note = None;
    for (col, f) in [("group", s.group), ("KSLabel", s.ks_label)] {
        if f == LabelFilter::All {
            continue;
        }
        if col == "group" && !cols.contains_key(col) {
            note = Some(NO_GROUP_NOTE);
            continue;
        }
        let c = cols.get(col).with_context(|| format!("no '{col}' column found (cluster_{col}.tsv or cluster_info.tsv)"))?;
        keep.retain(|u| c.get(u).is_some_and(|v| f.accepts(v)));
        if keep.is_empty() {
            bail!("no units left after the {col} filter ({})", f.name());
        }
    }
    let column = s.user_column.trim();
    if !column.is_empty() {
        let c = cols.get(column).with_context(|| format!("column '{column}' not found in the cluster tables"))?;
        let m = ValueMatch::parse(&s.user_value, column)?;
        m.check(column, c)?;
        keep.retain(|u| c.get(u).is_some_and(|v| m.matches(v)));
        if keep.is_empty() {
            bail!("no units left after the '{column}' filter");
        }
    }
    Ok((keep, note))
}

#[derive(Default)]
struct Params {
    sample_rate: Option<f64>,
    n_channels_dat: Option<usize>,
}

fn read_params(path: &Path) -> Params {
    let mut p = Params::default();
    for line in std::fs::read_to_string(path).unwrap_or_default().lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        let v = v.trim();
        match k.trim() {
            "sample_rate" => p.sample_rate = v.parse().ok(),
            "n_channels_dat" => p.n_channels_dat = v.parse().ok(),
            _ => {}
        }
    }
    p
}

/// A numeric .npy file as f64 (exact for integers below 2^53, i.e. any sample index).
fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let h = crate::atlas::parse_npy_header(&bytes).with_context(|| path.display().to_string())?;
    let (size, get) = decoder(&h.descr).with_context(|| path.display().to_string())?;
    let n: usize = h.shape.iter().product();
    let data = &bytes[h.data_offset..];
    if data.len() < n * size {
        bail!("{} is truncated", path.display());
    }
    Ok((h.shape, (0..n).map(|i| get(&data[i * size..])).collect()))
}

/// Item size and reader of a little-endian numeric .npy dtype.
fn decoder(descr: &str) -> Result<(usize, fn(&[u8]) -> f64)> {
    let d = descr.trim_start_matches(['<', '=', '|']);
    if d.starts_with('>') {
        bail!("big-endian .npy data ('{descr}') is not supported");
    }
    Ok(match d {
        "f4" => (4, |b| f32::from_le_bytes(b[..4].try_into().unwrap()) as f64),
        "f8" => (8, |b| f64::from_le_bytes(b[..8].try_into().unwrap())),
        "i1" => (1, |b| b[0] as i8 as f64),
        "u1" => (1, |b| b[0] as f64),
        "i2" => (2, |b| i16::from_le_bytes(b[..2].try_into().unwrap()) as f64),
        "u2" => (2, |b| u16::from_le_bytes(b[..2].try_into().unwrap()) as f64),
        "i4" => (4, |b| i32::from_le_bytes(b[..4].try_into().unwrap()) as f64),
        "u4" => (4, |b| u32::from_le_bytes(b[..4].try_into().unwrap()) as f64),
        "i8" => (8, |b| i64::from_le_bytes(b[..8].try_into().unwrap()) as f64),
        "u8" => (8, |b| u64::from_le_bytes(b[..8].try_into().unwrap()) as f64),
        _ => bail!("unsupported .npy dtype '{descr}'"),
    })
}

/// Spike-sorted data of the open recording: the window, loading, filtering and the
/// overlay.
pub struct SpikesState {
    pub open: bool,
    path_text: String,
    /// filters as set in the window
    filters: SpikeSettings,
    data: Option<SortedSpikes>,
    /// indices into `data` of the spikes that pass the filters, in time order
    shown: Vec<u32>,
    n_shown_units: usize,
    pub show_overlay: bool,
    load_rx: Option<mpsc::Receiver<Result<SortedSpikes>>>,
    pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    error: Option<String>,
    /// why a filter was skipped (shown in the window and the legend, nothing pops up)
    note: Option<&'static str>,
    /// spikes in the view last frame when there were too many to draw
    too_many: Option<usize>,
}

impl SpikesState {
    /// `default_dir`: the default location (Preferences), used when the recording has
    /// no saved folder.
    pub fn new(settings: SpikeSettings, default_dir: Option<PathBuf>) -> Self {
        let folder = settings.folder.clone().or(default_dir.filter(|d| d.is_dir()));
        Self {
            open: false,
            path_text: folder.map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
            filters: settings,
            data: None,
            shown: Vec::new(),
            n_shown_units: 0,
            show_overlay: true,
            load_rx: None,
            pick_rx: None,
            error: None,
            note: None,
            too_many: None,
        }
    }

    /// What is saved with the recording.
    pub fn settings(&self) -> SpikeSettings {
        let folder = self.path_text.trim();
        SpikeSettings { folder: (!folder.is_empty()).then(|| PathBuf::from(folder)), ..self.filters.clone() }
    }

    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }

    pub fn n_units(&self) -> usize {
        self.n_shown_units
    }

    fn loading(&self) -> bool {
        self.load_rx.is_some()
    }

    /// Why a filter was skipped, if one was.
    pub fn note(&self) -> Option<&'static str> {
        self.note.filter(|_| self.data.is_some())
    }

    /// Spikes in the view when there were too many to draw.
    pub fn too_many(&self) -> Option<usize> {
        self.too_many.filter(|_| self.show_overlay && self.data.is_some())
    }

    /// Load the folder in the window (Load on start); opens the window if that fails.
    pub fn load_on_start(&mut self, ctx: &egui::Context, meta: &Arc<Meta>) {
        if is_phy_dir(Path::new(self.path_text.trim())) {
            self.apply(ctx, meta);
        }
    }

    /// Load the folder (in the background) unless it is the loaded one, then filter.
    fn apply(&mut self, ctx: &egui::Context, meta: &Arc<Meta>) {
        let dir = PathBuf::from(self.path_text.trim());
        if self.data.as_ref().is_some_and(|d| d.dir == dir) {
            self.refilter();
            return;
        }
        if !is_phy_dir(&dir) {
            self.error = Some(format!("no spike_times.npy in {}", dir.display()));
            self.open = true;
            return;
        }
        let (tx, rx) = mpsc::channel();
        let (meta, ctx) = (Arc::clone(meta), ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(load(&dir, &meta).with_context(|| format!("loading {}", dir.display())));
            ctx.request_repaint();
        });
        self.load_rx = Some(rx);
        self.error = None;
    }

    /// Select the spikes of the units that pass the filters. On failure the window
    /// opens with the reason, and the spikes shown so far stay.
    fn refilter(&mut self) {
        let Some(d) = &self.data else { return };
        match filter_units(&d.dir, &d.units, &self.filters) {
            Ok((keep, note)) => {
                self.note = note;
                let mut mask = vec![false; d.units.last().map_or(0, |&u| u as usize + 1)];
                for &u in &keep {
                    mask[u as usize] = true;
                }
                self.shown = (0..d.clusters.len() as u32).filter(|&i| mask[d.clusters[i as usize] as usize]).collect();
                self.n_shown_units = keep.len();
                self.show_overlay = true;
                self.error = None;
            }
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                self.open = true;
            }
        }
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.pick_rx = None;
                    if let Some(dir) = picked {
                        self.path_text = dir.to_string_lossy().into_owned();
                    }
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => self.pick_rx = None,
            }
        }
        if let Some(rx) = &self.load_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.load_rx = None;
                    match res {
                        Ok(d) => {
                            self.data = Some(d);
                            self.shown.clear();
                            self.n_shown_units = 0;
                            self.refilter();
                        }
                        Err(e) => {
                            self.error = Some(format!("{e:#}"));
                            self.open = true;
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => self.load_rx = None,
            }
        }
    }

    pub fn draw_window(&mut self, ctx: &egui::Context, meta: &Arc<Meta>) {
        if !self.open {
            return;
        }
        let mut apply = false;
        let mut open = self.open;
        egui::Window::new("Spike-sorted data")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Folder:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.path_text)
                            .desired_width(300.0)
                            .hint_text("phy folder (spike_times.npy, ...)"),
                    );
                    if ui.button("Browse…").clicked() && self.pick_rx.is_none() {
                        let start = Some(PathBuf::from(self.path_text.trim())).filter(|d| d.is_dir());
                        let (tx, rx) = mpsc::channel();
                        std::thread::spawn(move || {
                            let mut dlg = rfd::FileDialog::new().set_title("Spike-sorted data (phy folder)");
                            if let Some(d) = start {
                                dlg = dlg.set_directory(d);
                            }
                            let _ = tx.send(dlg.pick_folder());
                        });
                        self.pick_rx = Some(rx);
                    }
                });

                ui.separator();
                ui.label(egui::RichText::new("Filter neurons").strong());
                egui::Grid::new("spike_filter_grid").num_columns(2).show(ui, |ui| {
                    for (label, hover, value) in [
                        ("Curation label (group):", "phy's manual label (cluster_group.tsv)", &mut self.filters.group),
                        ("Kilosort label (KSLabel):", "Kilosort's automatic label (cluster_KSLabel.tsv)", &mut self.filters.ks_label),
                    ] {
                        ui.label(label).on_hover_text(hover);
                        egui::ComboBox::from_id_salt(label).selected_text(value.name()).show_ui(ui, |ui| {
                            for f in LabelFilter::ALL {
                                ui.selectable_value(value, f, f.name());
                            }
                        });
                        ui.end_row();
                    }
                });
                ui.label("User:").on_hover_text(VALUE_HELP);
                ui.horizontal(|ui| {
                    ui.label("Column:");
                    ui.add(egui::TextEdit::singleline(&mut self.filters.user_column).desired_width(110.0).hint_text("e.g. layer"))
                        .on_hover_text(VALUE_HELP);
                    ui.label("Value:");
                    ui.add(egui::TextEdit::singleline(&mut self.filters.user_value).desired_width(110.0).hint_text("e.g. Th or >50"))
                        .on_hover_text(VALUE_HELP);
                });
                ui.label(
                    egui::RichText::new("applied in this order; \"all\" includes noise")
                        .small()
                        .color(Color32::GRAY),
                );

                ui.separator();
                ui.horizontal(|ui| {
                    let loaded = self.data.as_ref().is_some_and(|d| d.dir == Path::new(self.path_text.trim()));
                    let label = if loaded { "Apply" } else { "Load" };
                    apply = ui
                        .add_enabled(!self.loading() && !self.path_text.trim().is_empty(), egui::Button::new(label))
                        .clicked();
                    if self.loading() {
                        ui.spinner();
                        ui.label("Loading…");
                    }
                });
                if let Some(err) = &self.error {
                    ui.colored_label(Color32::from_rgb(0xff, 0x66, 0x66), err);
                } else if self.data.is_some() {
                    ui.colored_label(
                        Color32::from_rgb(0x55, 0xdd, 0x77),
                        format!("{} units, {} spikes shown", self.n_shown_units, self.shown.len()),
                    );
                    if let Some(note) = self.note {
                        ui.label(egui::RichText::new(note).small().color(Color32::GRAY));
                    }
                }
            });
        self.open = open;
        if apply {
            self.apply(ctx, meta);
        }
    }

    /// A dot per spike in the view, on its channel's row of the heatmap drawn in `rect`
    /// with rows `first_row..=last_row`. Spikes on removed channels are not drawn.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_overlay(
        &mut self,
        painter: &egui::Painter,
        rect: egui::Rect,
        view_start_s: f64,
        view_dur_s: f64,
        display_rows: &[DisplayRow],
        first_row: usize,
        last_row: usize,
        cmap: &ColorMapChoice,
    ) {
        self.too_many = None;
        let Some(d) = self.data.as_ref().filter(|_| self.show_overlay && view_dur_s > 0.0) else {
            return;
        };
        let time = |i: &u32| d.times_s[*i as usize];
        let lo = self.shown.partition_point(|i| time(i) < view_start_s);
        let hi = self.shown.partition_point(|i| time(i) <= view_start_s + view_dur_s);
        if hi - lo > MAX_DOTS {
            self.too_many = Some(hi - lo);
            return;
        }

        let n_rows = (last_row - first_row + 1) as f32;
        let mut row_y = Vec::new();
        for r in first_row..=last_row {
            if let DisplayRow::Data { channels, .. } = &display_rows[r] {
                let y = rect.top() + ((last_row - r) as f32 + 0.5) / n_rows * rect.height();
                for &c in channels {
                    if c >= row_y.len() {
                        row_y.resize(c + 1, None);
                    }
                    row_y[c] = Some(y);
                }
            }
        }

        let spec = cmap.spec();
        let fill = Color32::from_rgb(spec.spike_fill[0], spec.spike_fill[1], spec.spike_fill[2]);
        let [r, g, b] = spec.spike_outline;
        let radius = (DOT_RADIUS * (DOT_REF_DUR_S / view_dur_s) as f32).clamp(DOT_MIN_RADIUS, DOT_MAX_RADIUS);
        let stroke = egui::Stroke::new((radius * 0.4).clamp(0.5, 1.0), Color32::from_rgb(r, g, b));
        for &i in &self.shown[lo..hi] {
            let i = i as usize;
            if let Some(&Some(y)) = row_y.get(d.chans[i] as usize) {
                let x = rect.left() + ((d.times_s[i] - view_start_s) / view_dur_s) as f32 * rect.width();
                painter.circle(egui::pos2(x, y), radius, fill, stroke);
            }
        }
    }
}
