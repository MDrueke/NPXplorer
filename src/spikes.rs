// Spike-sorted data (phy / Kilosort output folder): every spike is drawn as a dot on
// the heatmap, at its time and on the peak channel of its template.
//
// Files read (see https://phy.readthedocs.io): spike_times.npy (samples),
// spike_templates.npy, spike_clusters.npy (optional, after curation), templates.npy
// (n_templates x n_time x n_temp_channels), templates_ind.npy (optional),
// channel_map.npy (row of each sorted channel in the data file), params.py
// (sample_rate, n_channels_dat) and cluster_group.tsv (optional; clusters labelled
// "noise" are left out).

use anyhow::{bail, Context, Result};
use egui::Color32;
use std::collections::HashSet;
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

pub struct SortedSpikes {
    /// sorted
    times_s: Vec<f64>,
    /// 0-based channel in file order, same order as `times_s`
    chans: Vec<u32>,
    pub n_units: usize,
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
    let noise = noise_clusters(dir);

    let mut idx: Vec<usize> = (0..times.len())
        .filter(|&i| !noise.contains(&(clusters[i] as i64)))
        .collect();
    if idx.windows(2).any(|w| times[w[1]] < times[w[0]]) {
        idx.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
    }
    let mut times_s = Vec::with_capacity(idx.len());
    let mut chans = Vec::with_capacity(idx.len());
    for &i in &idx {
        let t = spike_templates[i] as usize;
        let ch = *peak.get(t).with_context(|| format!("spike template {t} not in templates.npy"))?;
        times_s.push(times[i] / fs);
        chans.push(ch as u32);
    }
    let n_units = idx.iter().map(|&i| clusters[i] as i64).collect::<HashSet<_>>().len();
    Ok(SortedSpikes { times_s, chans, n_units })
}

/// Recording channel with the largest peak-to-peak amplitude of each template.
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

    let mut peak = Vec::with_capacity(n_t);
    let mut lo = vec![f64::INFINITY; n_ch];
    let mut hi = vec![f64::NEG_INFINITY; n_ch];
    for t in 0..n_t {
        lo.fill(f64::INFINITY);
        hi.fill(f64::NEG_INFINITY);
        for s in 0..n_time {
            let row = (t * n_time + s) * n_ch * size;
            for c in 0..n_ch {
                let v = get(&data[row + c * size..]);
                lo[c] = lo[c].min(v);
                hi[c] = hi[c].max(v);
            }
        }
        let best = (0..n_ch).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap_or(0);
        let k = ind.as_ref().map_or(best, |ind| ind[t * n_ch + best] as usize);
        peak.push(*chan_map.get(k).with_context(|| format!("template channel {k} not in channel_map.npy"))?);
    }
    Ok(peak)
}

/// Clusters labelled "noise" in cluster_group.tsv (phy's manual labels).
fn noise_clusters(dir: &Path) -> HashSet<i64> {
    let Ok(text) = std::fs::read_to_string(dir.join("cluster_group.tsv")) else {
        return HashSet::new();
    };
    text.lines()
        .skip(1)
        .filter_map(|l| {
            let mut f = l.split('\t');
            let id = f.next()?.trim().parse().ok()?;
            (f.next()?.trim() == "noise").then_some(id)
        })
        .collect()
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

/// Spike-sorted data of the open recording: loading, the overlay and its errors.
#[derive(Default)]
pub struct SpikesState {
    data: Option<SortedSpikes>,
    pub show_overlay: bool,
    load_rx: Option<mpsc::Receiver<Result<SortedSpikes>>>,
    pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    pub error: Option<String>,
    /// spikes in the view last frame when there were too many to draw
    too_many: Option<usize>,
}

impl SpikesState {
    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }

    pub fn n_units(&self) -> usize {
        self.data.as_ref().map_or(0, |d| d.n_units)
    }

    pub fn loading(&self) -> bool {
        self.load_rx.is_some() || self.pick_rx.is_some()
    }

    /// Spikes in the view when there were too many to draw.
    pub fn too_many(&self) -> Option<usize> {
        self.too_many.filter(|_| self.show_overlay && self.data.is_some())
    }

    pub fn start_load(&mut self, ctx: &egui::Context, dir: PathBuf, meta: &Arc<Meta>) {
        let (tx, rx) = mpsc::channel();
        let (meta, ctx) = (Arc::clone(meta), ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(load(&dir, &meta).with_context(|| format!("loading {}", dir.display())));
            ctx.request_repaint();
        });
        self.load_rx = Some(rx);
        self.error = None;
    }

    /// Folder picker for a phy folder, starting in `dir`; loads the picked one.
    pub fn pick_and_load(&mut self, dir: Option<PathBuf>) {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut dlg = rfd::FileDialog::new().set_title("Spike-sorted data (phy folder)");
            if let Some(d) = dir {
                dlg = dlg.set_directory(d);
            }
            let _ = tx.send(dlg.pick_folder());
        });
        self.pick_rx = Some(rx);
    }

    pub fn poll(&mut self, ctx: &egui::Context, meta: &Arc<Meta>) {
        if let Some(rx) = &self.pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.pick_rx = None;
                    if let Some(dir) = picked {
                        self.start_load(ctx, dir, meta);
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
                            self.show_overlay = true;
                        }
                        Err(e) => self.error = Some(format!("{e:#}")),
                    }
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => self.load_rx = None,
            }
        }
    }

    pub fn draw_error_window(&mut self, ctx: &egui::Context) {
        let Some(err) = &self.error else { return };
        let mut open = true;
        egui::Window::new("Loading spike-sorted data failed")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.colored_label(Color32::from_rgb(0xff, 0x66, 0x66), err);
            });
        if !open {
            self.error = None;
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
        let lo = d.times_s.partition_point(|&t| t < view_start_s);
        let hi = d.times_s.partition_point(|&t| t <= view_start_s + view_dur_s);
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
        for i in lo..hi {
            if let Some(&Some(y)) = row_y.get(d.chans[i] as usize) {
                let x = rect.left() + ((d.times_s[i] - view_start_s) / view_dur_s) as f32 * rect.width();
                painter.circle(egui::pos2(x, y), radius, fill, stroke);
            }
        }
    }
}
