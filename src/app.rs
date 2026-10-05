use egui::{CentralPanel, TextureHandle, TextureOptions, TopBottomPanel, Ui, Vec2};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};

use crate::data::{open_data, ChannelOrder, DisplayRow, Meta, RawData, ShankOrder};
use crate::preprocess::{Filters, PreprocConfig, SpatialFilter};
use crate::psth::{compute_psth, load_stim_times, PsthParams, PsthResult, StimLayout};
use crate::render::{build_heatmap_into, build_psth_heatmap_into};
use crate::worker::{
    compute_half_window, request_shutdown, spawn_worker, RequestKind, SharedCancel,
    SharedWorkerState, WorkerRequest, WorkerState, WorkerStatus,
};

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ColorMode {
    Percentile,
    Voltage,
}

use crate::colormap::ColorMapChoice;

fn default_initial_buffer_s() -> f64 {
    30.0
}
fn default_extension_margin_s() -> f64 {
    5.0
}
fn default_mem_pressure_pct() -> f32 {
    15.0
}
fn default_mem_reserve_mb() -> f64 {
    1500.0
}
fn default_true() -> bool {
    true
}
fn default_spike_overlay_scale() -> f32 {
    1.0
}
fn default_spike_smoothing_sigma() -> f32 {
    1.5
}
fn default_n_classify_chunks() -> usize {
    crate::channel_classify::DEFAULT_N_CLASSIFY_CHUNKS
}
fn default_atlas_min_region_channels() -> usize {
    4
}
fn default_bregma_lambda_mm() -> f64 {
    crate::atlas::REFERENCE_BREGMA_LAMBDA_MM
}
fn default_spectrum_n_chunks() -> usize {
    100
}
fn default_spectrum_freq_min_hz() -> f64 {
    1.0
}
fn default_spectrum_freq_max_hz() -> f64 {
    5000.0
}
fn default_spectrum_time_end_s() -> f64 {
    60.0
}

/// Largest buffer duration (s) that fits in currently-available system memory, minus
/// `mem_reserve_mb`. Used both to clamp saved preferences at load time and to bound
/// the "Initial buffer size" slider live in the Preferences window.
fn max_feasible_buffer_s(n_data_rows: usize, sample_rate: f64, mem_reserve_mb: f64) -> f64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    let available = sys.available_memory() as f64;
    let usable = (available - mem_reserve_mb * 1e6).max(0.0);
    let bytes_per_sample_all_rows = (n_data_rows.max(1) as f64) * 4.0; // f32
    (usable / bytes_per_sample_all_rows / sample_rate).max(1.0)
}

/// Largest extension margin (s) that can't oscillate against itself.
///
/// Once the buffer is at its `initial_buffer_s` (B) cap, extending by margin M on one
/// side trims the same M from the opposite side (net-zero growth). Right before that
/// fires, the far margin is at worst `B - view_n - M`; after the trim it drops by
/// another M, to `B - view_n - 2M`. For that not to already be below M (and
/// immediately trigger an extension back the other way), we need:
///   B - view_n - 2M >= M   =>   M <= (B - view_n) / 3
/// A 0.9 safety factor keeps clear of the exact boundary (float rounding, view_n
/// changes, etc.).
fn max_extension_margin_s(initial_buffer_s: f64, view_dur_s: f64) -> f64 {
    (((initial_buffer_s - view_dur_s).max(0.0) / 3.0) * 0.9).max(0.5)
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Preferences {
    pub preproc_cfg: PreprocConfig,
    pub view_dur_s: f64,
    pub color_mode: ColorMode,
    pub color_pct: f32,
    pub color_uv: f32,
    pub colormap_choice: ColorMapChoice,
    pub spike_threshold: f32,
    /// whether the firing-rate overlay is shown (and computed) at all
    #[serde(default = "default_true")]
    pub show_firing_rate_overlay: bool,
    /// user-configurable multiplier on the firing-rate overlay's width scaling
    #[serde(default = "default_spike_overlay_scale")]
    pub spike_overlay_scale: f32,
    /// std. dev. (in channels/display rows) of the Gaussian used to smooth the
    /// firing-rate overlay across depth
    #[serde(default = "default_spike_smoothing_sigma")]
    pub spike_smoothing_sigma: f32,
    /// number of evenly-spaced chunks sampled across the recording for the
    /// channel-classification majority vote
    #[serde(default = "default_n_classify_chunks")]
    pub n_classify_chunks: usize,
    /// how "outside of the brain" channels are detected (IBL fixed threshold / adaptive)
    #[serde(default)]
    pub classify_outside_rule: crate::channel_classify::OutsideRule,
    /// heatmap: show each pixel column's extreme sample instead of its mean
    #[serde(default)]
    pub peak_pooling: bool,
    /// total size (s) of the buffer loaded on initial load / full recompute; also the
    /// steady-state cap that incremental extension growth settles back to
    #[serde(default = "default_initial_buffer_s")]
    pub initial_buffer_s: f64,
    /// how close (s) the view can get to the edge of the preprocessed buffer before
    /// an extension is triggered; also the size of each extension step
    #[serde(default = "default_extension_margin_s")]
    pub extension_margin_s: f64,
    #[serde(default = "default_mem_pressure_pct")]
    pub mem_pressure_pct: f32,
    #[serde(default = "default_mem_reserve_mb")]
    pub mem_reserve_mb: f64,
    #[serde(default)]
    pub last_dir: Option<String>,
    /// paths of the most recently opened recordings, most recent first (max 5)
    #[serde(default)]
    pub recent_files: Vec<String>,
    /// folder holding the Allen CCF atlas files (Atlas Registration)
    #[serde(default)]
    pub atlas_dir: Option<String>,
    /// last-used bregma-lambda distance; a recording's saved insertion overrides it
    #[serde(default = "default_bregma_lambda_mm")]
    pub bregma_lambda_mm: f64,
    /// atlas overlay: regions with fewer channels are not drawn on their own
    #[serde(default = "default_atlas_min_region_channels")]
    pub atlas_min_region_channels: usize,
    /// applied noise-suppression settings (visual filter)
    #[serde(default)]
    pub noise_suppression: crate::noise::NoiseSuppression,
    /// power spectrum: time span the PSD is computed over
    #[serde(default)]
    pub spectrum_time_scope: crate::spectrum::SpectrumTimeScope,
    /// power spectrum: raw voltage or the current preprocessed buffer
    #[serde(default)]
    pub spectrum_source: crate::spectrum::SpectrumSource,
    /// power spectrum: linear power or dB
    #[serde(default)]
    pub spectrum_scaling: crate::spectrum::SpectrumScaling,
    /// power spectrum: colour range shared across channels, or per-channel
    #[serde(default)]
    pub spectrum_normalization: crate::spectrum::SpectrumNormalization,
    /// power spectrum: number of evenly-spaced chunks sampled in whole-recording mode
    #[serde(default = "default_spectrum_n_chunks")]
    pub spectrum_n_chunks: usize,
    /// power spectrum: restrict the displayed/coloured band to a sub-range
    #[serde(default)]
    pub spectrum_freq_restrict: bool,
    #[serde(default = "default_spectrum_freq_min_hz")]
    pub spectrum_freq_min_hz: f64,
    #[serde(default = "default_spectrum_freq_max_hz")]
    pub spectrum_freq_max_hz: f64,
    /// power spectrum: restrict the span whole-recording chunks are sourced from
    #[serde(default)]
    pub spectrum_time_restrict: bool,
    #[serde(default)]
    pub spectrum_time_start_s: f64,
    #[serde(default = "default_spectrum_time_end_s")]
    pub spectrum_time_end_s: f64,
}

impl Preferences {
    pub fn load() -> Option<Self> {
        // prefer the new config/ location; fall back to the legacy path next to the exe
        // so settings saved by older versions are not lost
        let text = std::fs::read_to_string(Self::path())
            .or_else(|_| std::fs::read_to_string(Self::legacy_path()))
            .ok()?;
        toml::from_str(&text).ok()
    }

    pub fn save(&self) {
        let path = Self::path();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(s) = toml::to_string_pretty(self) {
            let _ = std::fs::write(path, s);
        }
    }

    pub fn path() -> std::path::PathBuf {
        crate::psth::config_dir().join("npxplorer_prefs.toml")
    }

    fn legacy_path() -> std::path::PathBuf {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                return dir.join("npxplorer_prefs.toml");
            }
        }
        std::path::PathBuf::from("npxplorer_prefs.toml")
    }
}

/// State for the PSTH window: the peri-stimulus average of the preprocessed signal.
struct PsthState {
    open: bool,
    stim_path: Option<PathBuf>,
    // staged settings (only committed to a recompute when "Apply/Compute" is pressed)
    start_ms: f64,
    end_ms: f64,
    start_ms_str: String,
    end_ms_str: String,
    stim_t_start: f64,
    stim_t_end: f64,
    stim_t_start_str: String,
    stim_t_end_str: String,
    total_s: f64,
    color_mode: ColorMode,
    color_pct: f32,
    color_uv: f32,

    // per-channel line selection (mirrors the main window; independent of it)
    sel_ch1: Option<usize>,
    sel_ch2: Option<usize>,

    // async plumbing
    pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    compute_rx: Option<mpsc::Receiver<Result<PsthResult, String>>>,
    cancel: Arc<AtomicBool>,
    /// stimuli processed / to process by the in-flight compute
    progress: Arc<AtomicUsize>,
    progress_total: Arc<AtomicUsize>,
    computing: bool,
    apply_requested: bool,
    /// color-scale max when the result arrived: the trace plots are autoscaled at that
    /// value and zoom with later color-scale changes
    trace_vmax_ref: f32,

    result: Option<Arc<PsthResult>>,
    error: Option<String>,
    n_used: usize,
    n_skipped: usize,

    texture: Option<TextureHandle>,
    pixel_buf: Vec<u8>,
    tex_dirty: bool,
    last_tex_size: Option<[usize; 2]>,

    // rect of the plotted figure (logical coords), for PNG export cropping
    figure_rect: Option<egui::Rect>,
    export_pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    export_path: Option<PathBuf>,
    export_pending: bool,
}

impl PsthState {
    fn new(total_s: f64) -> Self {
        Self {
            open: false,
            stim_path: None,
            start_ms: -50.0,
            end_ms: 200.0,
            start_ms_str: "-50".to_string(),
            end_ms_str: "200".to_string(),
            stim_t_start: 0.0,
            stim_t_end: total_s,
            stim_t_start_str: "0.000".to_string(),
            stim_t_end_str: format!("{:.3}", total_s),
            total_s,
            color_mode: ColorMode::Percentile,
            color_pct: 99.0,
            color_uv: 50.0,
            sel_ch1: None,
            sel_ch2: None,
            pick_rx: None,
            compute_rx: None,
            cancel: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(AtomicUsize::new(0)),
            progress_total: Arc::new(AtomicUsize::new(0)),
            computing: false,
            apply_requested: false,
            trace_vmax_ref: 1.0,
            result: None,
            error: None,
            n_used: 0,
            n_skipped: 0,
            texture: None,
            pixel_buf: Vec::new(),
            tex_dirty: false,
            last_tex_size: None,
            figure_rect: None,
            export_pick_rx: None,
            export_path: None,
            export_pending: false,
        }
    }
}

impl PsthState {
    /// Fill in the file and settings saved for this recording.
    fn restore(&mut self, stim_file: Option<&std::path::Path>, s: &crate::ttl::PsthSettings) {
        self.stim_path = stim_file.filter(|p| p.is_file()).map(|p| p.to_path_buf());
        if let Some(m) = &s.color_mode {
            self.color_mode = m.clone();
        }
        self.color_pct = s.color_pct.unwrap_or(self.color_pct).clamp(95.0, 100.0);
        self.color_uv = s.color_uv.unwrap_or(self.color_uv).clamp(1.0, 200.0);
        if let (Some(a), Some(b)) = (s.start_ms, s.end_ms) {
            (self.start_ms, self.end_ms) = (a, b);
            self.start_ms_str = format!("{a}");
            self.end_ms_str = format!("{b}");
        }
        if let (Some(a), Some(b)) = (s.stim_t_start, s.stim_t_end) {
            self.stim_t_start = a.clamp(0.0, self.total_s);
            self.stim_t_end = b.clamp(0.0, self.total_s);
            self.stim_t_start_str = format!("{:.3}", self.stim_t_start);
            self.stim_t_end_str = format!("{:.3}", self.stim_t_end);
        }
    }

    /// What is saved with the recording.
    fn settings(&self) -> crate::ttl::PsthSettings {
        crate::ttl::PsthSettings {
            start_ms: Some(self.start_ms),
            end_ms: Some(self.end_ms),
            stim_t_start: Some(self.stim_t_start),
            stim_t_end: Some(self.stim_t_end),
            color_mode: Some(self.color_mode.clone()),
            color_pct: Some(self.color_pct),
            color_uv: Some(self.color_uv),
        }
    }

    /// Color-scale max of the heatmap (µV).
    fn vmax(&self, result: &PsthResult) -> f32 {
        match self.color_mode {
            ColorMode::Percentile => result.vmax_percentile(self.color_pct),
            ColorMode::Voltage => self.color_uv.max(1e-6),
        }
    }
}

/// Spawn the native picker for a stimulus-times file on a background thread (same
/// rationale as the main file picker: never block the egui event loop).
pub(crate) fn spawn_stim_picker(dir: Option<PathBuf>) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut dlg = rfd::FileDialog::new()
            .add_filter("Stimulus times", &["csv", "txt", "tsv", "dat"])
            .add_filter("All files", &["*"]);
        if let Some(d) = dir {
            dlg = dlg.set_directory(d);
        }
        let _ = tx.send(dlg.pick_file());
    });
    rx
}

/// Spawn the native picker for a channel-numbers-to-remove file on a background thread.
fn spawn_channel_list_picker(dir: Option<PathBuf>) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut dlg = rfd::FileDialog::new()
            .add_filter("Channel list", &["csv", "txt", "tsv", "dat"])
            .add_filter("All files", &["*"]);
        if let Some(d) = dir {
            dlg = dlg.set_directory(d);
        }
        let _ = tx.send(dlg.pick_file());
    });
    rx
}

/// Spawn the native save dialog for the PSTH PNG on a background thread.
fn spawn_png_saver(dir: Option<PathBuf>, default_name: String) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut dlg = rfd::FileDialog::new()
            .add_filter("PNG image", &["png"])
            .set_file_name(default_name);
        if let Some(d) = dir {
            dlg = dlg.set_directory(d);
        }
        let _ = tx.send(dlg.save_file());
    });
    rx
}

/// Index range (inclusive) of all display rows.
fn all_rows(display_rows: &[DisplayRow]) -> (usize, usize) {
    (0, display_rows.len().saturating_sub(1))
}

/// Rectangle zoom (left-drag on the heatmap): a channel range plus the time view from
/// before the first zoom, which Esc restores.
/// Everything the spectrum panel's texture depends on. Holds the result's `Arc`
/// (compared by pointer) rather than a bare address, so a freed result's address
/// being reused by the next one can't make a stale texture look current.
struct SpectrumTexKey {
    result: Arc<crate::spectrum::ComputedSpectrum>,
    size: [usize; 2],
    rows: (usize, usize),
    scaling: crate::spectrum::SpectrumScaling,
    normalization: crate::spectrum::SpectrumNormalization,
    freq_range: Option<(f32, f32)>,
}

impl PartialEq for SpectrumTexKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.result, &other.result)
            && self.size == other.size
            && self.rows == other.rows
            && self.scaling == other.scaling
            && self.normalization == other.normalization
            && self.freq_range == other.freq_range
    }
}

/// How long the view must stay put before a live (current-view) spectrum is
/// recomputed, so scrolling isn't slowed down by an FFT every frame.
const SPECTRUM_SETTLE: std::time::Duration = std::time::Duration::from_millis(150);

/// Rectangle zoom in the waveform view: the time window and voltage range from before
/// the first zoom, which Esc restores.
struct WaveformZoom {
    prev_start_s: f64,
    prev_dur_s: f64,
    prev_y_range_uv: f32,
}

struct Zoom {
    /// channels of the bottom and top rows; stored as channels rather than row indices
    /// because the rows are rebuilt when channels are removed or averaged
    ch_bottom: usize,
    ch_top: usize,
    prev_start_s: f64,
    prev_dur_s: f64,
}

/// Index range (inclusive) of the display rows on screen: the zoomed channel range, or
/// all rows when not zoomed (or when a zoom channel no longer has a row).
fn view_rows(display_rows: &[DisplayRow], zoom: Option<&Zoom>) -> (usize, usize) {
    let row_of = |ch: usize| {
        display_rows
            .iter()
            .position(|r| matches!(r, DisplayRow::Data { channels, .. } if channels.contains(&ch)))
    };
    match zoom.map(|z| (row_of(z.ch_bottom), row_of(z.ch_top))) {
        Some((Some(lo), Some(hi))) if lo <= hi => (lo, hi),
        _ => all_rows(display_rows),
    }
}

/// Smallest ± voltage range (µV) of the waveform view.
const WAVEFORM_MIN_RANGE_UV: f32 = 1.0;

/// Smallest drag (px, in each direction) that zooms; anything less stays a click.
const ZOOM_MIN_DRAG_PX: f32 = 4.0;

/// Channels of the bottom and top data rows inside the screen-space selection `sel` on
/// a heatmap drawn in `rect` with rows `first_row..=last_row`; at least two rows when
/// the heatmap has them. None if the selection holds no data row.
fn zoom_selection(
    display_rows: &[DisplayRow],
    first_row: usize,
    last_row: usize,
    rect: egui::Rect,
    sel: egui::Rect,
) -> Option<(usize, usize)> {
    let n_rows = last_row - first_row + 1;
    let row_at = |y: f32| {
        let frac = ((y - rect.top()) / rect.height()).clamp(0.0, 1.0);
        last_row
            .saturating_sub((frac as f64 * n_rows as f64) as usize)
            .clamp(first_row, last_row)
    };
    let data_rows: Vec<(usize, usize)> = (first_row..=last_row)
        .filter_map(|r| match &display_rows[r] {
            DisplayRow::Data { first_ch, .. } => Some((r, *first_ch)),
            _ => None,
        })
        .collect();
    let (lo, hi) = (row_at(sel.bottom()), row_at(sel.top()));
    let inside: Vec<usize> = (0..data_rows.len())
        .filter(|&i| (lo..=hi).contains(&data_rows[i].0))
        .collect();
    let (mut b, mut t) = (*inside.first()?, *inside.last()?);
    if b == t {
        if t + 1 < data_rows.len() {
            t += 1;
        } else if b > 0 {
            b -= 1;
        }
    }
    Some((data_rows[b].1, data_rows[t].1))
}

/// Channel IDs of the first and last data rows (bottom and top of the heatmap).
fn edge_channel_ids<'a>(meta: &'a Meta, display_rows: &[DisplayRow]) -> (&'a str, &'a str) {
    let first = display_rows.iter().find_map(|r| match r {
        DisplayRow::Data { first_ch, .. } => Some(meta.channel_id(*first_ch)),
        _ => None,
    });
    let last = display_rows.iter().rev().find_map(|r| match r {
        DisplayRow::Data { first_ch, .. } => Some(meta.channel_id(*first_ch)),
        _ => None,
    });
    (first.unwrap_or(""), last.unwrap_or(""))
}

/// Mirrors egui's internal (private) `menu::set_menu_style`, so a custom popup's
/// buttons render flat — transparent until hovered, no per-item border — exactly
/// like real menu items (e.g. the "File" dropdown's "Open"/"Recent files").
fn apply_menu_item_style(style: &mut egui::Style) {
    style.spacing.button_padding = egui::vec2(2.0, 0.0);
    style.visuals.widgets.active.bg_stroke = egui::Stroke::NONE;
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::NONE;
    style.visuals.widgets.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::NONE;
}

/// Zero horizontal gaps between widgets in this scope — the flat fill/corner/border
/// styling itself is set app-wide in `MainApp::new` (main.rs), so this only needs to
/// handle the toolbar-specific "buttons flush together, thin separators" layout.
fn zero_item_gap(ui: &mut Ui) {
    ui.spacing_mut().item_spacing.x = 0.0;
}

/// Opacity (0-255) of the per-row classification overlay stripes on the heatmap.
/// 26 ≈ 0.1. Legend swatches in the toggle box are always drawn fully opaque,
/// independent of this.
const CLASSIFICATION_OVERLAY_ALPHA: u8 = 5;

/// Color for a channel-classification label (1 dead, 2 noisy, 3 outside of the
/// brain), at the given alpha. Label 0 (good) has no overlay color.
fn classification_color(label: u8, alpha: u8) -> egui::Color32 {
    match label {
        1 => egui::Color32::from_rgba_unmultiplied(255, 0, 255, alpha), // dead: magenta
        2 => egui::Color32::from_rgba_unmultiplied(230, 40, 40, alpha), // noisy: red
        3 => egui::Color32::from_rgba_unmultiplied(40, 200, 80, alpha), // outside of brain: green
        _ => egui::Color32::TRANSPARENT,
    }
}

/// data-row index (into `result.data`) for a 1-based channel number, or None if the
/// channel is not among the computed display rows.
fn channel_row(result: &PsthResult, ch: usize) -> Option<usize> {
    result.display_rows.iter().find_map(|r| match r {
        DisplayRow::Data {
            data_idx, first_ch, ..
        } if *first_ch + 1 == ch => Some(*data_idx),
        _ => None,
    })
}

/// 1-based channel number under a heatmap y coordinate (last row at top, first row at
/// bottom — matching `build_psth_heatmap_into`).
fn channel_at_heatmap_y(result: &PsthResult, heat_rect: egui::Rect, y: f32) -> Option<usize> {
    let n_rows = result.display_rows.len();
    if n_rows == 0 {
        return None;
    }
    let frac = ((y - heat_rect.top()) / heat_rect.height()).clamp(0.0, 0.999_9);
    let disp_idx = (n_rows - 1).saturating_sub((frac * n_rows as f32) as usize);
    match result.display_rows.get(disp_idx) {
        Some(DisplayRow::Data { first_ch, .. }) => Some(*first_ch + 1),
        _ => None,
    }
}

/// "Esc to close zoom" at `pos`: solid text on a semi-transparent box, both switching
/// with the colormap (dark box / white text, or the reverse on Cool-Warm's light
/// background).
fn draw_zoom_notice(painter: &egui::Painter, pos: egui::Pos2, cmap: &ColorMapChoice) {
    let [fr, fg, fb] = cmap.spec().heatmap_fg;
    let fg_color = egui::Color32::from_rgb(fr, fg, fb);
    let [br, bg, bb] = cmap.spec().label_bg;
    let galley = painter.layout_no_wrap("Esc to close zoom".to_string(), egui::FontId::proportional(13.0), fg_color);
    painter.rect_filled(
        galley.rect.translate(pos.to_vec2()).expand2(egui::vec2(6.0, 4.0)),
        4.0,
        egui::Color32::from_rgba_unmultiplied(br, bg, bb, 160),
    );
    painter.galley(pos, galley, fg_color);
}

/// The hover readout (channel, time, voltage) at the bottom left of `rect`.
fn draw_readout(painter: &egui::Painter, rect: egui::Rect, label: String) {
    let color = egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200);
    let galley = painter.layout_no_wrap(label, egui::FontId::proportional(12.0), color);
    let text_pos = rect.left_bottom() + Vec2::new(6.0, -6.0 - galley.rect.height());
    let bg_rect = galley.rect.translate(text_pos.to_vec2()).expand(4.0);
    let [r, g, b] = crate::render::C_ZERO;
    painter.rect_filled(bg_rect, 2.0, egui::Color32::from_rgba_unmultiplied(r, g, b, 200));
    painter.galley(text_pos, galley, color);
}

/// Noise-filtered copy of the view; `key` = (buffer data pointer, buffer first sample,
/// buffer length, view first sample, view length).
struct NoiseView {
    key: (usize, usize, usize, usize, usize),
    settings: crate::noise::NoiseSuppression,
    data: Arc<Vec<f32>>,
    first: usize,
    n: usize,
}

pub struct NPXplorerApp {
    bin_path: PathBuf,
    meta: Arc<Meta>,
    raw: Arc<RawData>,
    is_compressed: bool,

    // view state
    view_start_s: f64,
    view_dur_s: f64,
    window_dur_str: String,
    jump_str: String,

    // preprocessing
    preproc_cfg: PreprocConfig,
    preproc_filters: Arc<Mutex<Filters>>,
    scroll_speed_fine: bool,

    // color scale
    color_mode: ColorMode,
    color_pct: f32,
    color_uv: f32,
    color_pct_str: String,
    color_uv_str: String,
    colormap_choice: ColorMapChoice,

    // preferences
    show_preferences: bool,
    spike_threshold: f32,
    show_firing_rate_overlay: bool,
    spike_overlay_scale: f32,
    spike_smoothing_sigma: f32,

    // selected channels
    selected_channel_1: Option<usize>,
    selected_channel_2: Option<usize>,

    // per-channel right-click context menu (plain right-click, no Alt)
    context_menu_channel: Option<usize>,
    context_menu_pos: Option<egui::Pos2>,

    // single-channel waveform view: Some(channel) replaces the heatmap with a line
    // plot of that channel; y-axis is ±waveform_y_range_uv, adjustable via Alt+scroll
    waveform_channel: Option<usize>,
    waveform_y_range_uv: f32,
    /// voltage at the middle of the waveform view: 0, except when zoomed into a range
    waveform_y_center_uv: f32,
    /// rectangle zoom in the waveform view (left-drag); Esc returns to `WaveformZoom`'s
    /// view before closing the waveform view
    waveform_zoom: Option<WaveformZoom>,

    // channel removal
    show_remove_channels: bool,
    remove_channels_text: String,
    remove_channels_error: Option<String>,
    remove_channels_pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,

    // IBL-style channel classification (dead/noisy/outside-of-brain/good)
    channel_labels: Option<Arc<Vec<u8>>>,
    show_classification_overlay: bool,
    classifying: bool,
    classify_error: Option<String>,
    classify_rx: Option<mpsc::Receiver<Result<Vec<u8>, String>>>,
    classify_cancel: Arc<AtomicBool>,
    classify_progress: Arc<AtomicUsize>,
    /// user-configurable number of chunks for the majority vote (Preferences)
    classify_n_chunks: usize,
    classify_outside_rule: crate::channel_classify::OutsideRule,
    /// heatmap pixel columns show their extreme sample instead of their mean
    peak_pooling: bool,
    /// snapshot of `classify_n_chunks` taken when the in-flight run was dispatched,
    /// so the progress bar stays correct even if the preference changes mid-run
    classify_run_total: usize,

    // noise suppression (visual filter on the view only); the window edits the draft,
    // Apply copies it to `noise`
    noise_open: bool,
    /// "Notch filters" section of the noise window (scan + notch list)
    notch_panel: crate::notch::NotchPanel,
    noise: crate::noise::NoiseSuppression,
    noise_draft: crate::noise::NoiseSuppression,
    /// filtered copy of the view (plus margin) and what it was computed from
    noise_view: Option<NoiseView>,
    /// noise settings the current heatmap texture was drawn with
    last_rendered_noise: Option<crate::noise::NoiseSuppression>,

    // power spectrum (per-channel PSD overlay)
    spectrum_open: bool,
    spectrum_time_scope: crate::spectrum::SpectrumTimeScope,
    spectrum_source: crate::spectrum::SpectrumSource,
    spectrum_scaling: crate::spectrum::SpectrumScaling,
    spectrum_normalization: crate::spectrum::SpectrumNormalization,
    spectrum_n_chunks: usize,
    /// whole-recording mode: whether chunks are sourced only from
    /// [spectrum_time_start_s, spectrum_time_end_s] rather than the whole recording
    spectrum_time_restrict: bool,
    spectrum_time_start_s: f64,
    spectrum_time_end_s: f64,
    /// whether the displayed/coloured band is restricted to [spectrum_freq_min_hz,
    /// spectrum_freq_max_hz] rather than the full 0..nyquist band
    spectrum_freq_restrict: bool,
    spectrum_freq_min_hz: f64,
    spectrum_freq_max_hz: f64,
    spectrum_show_overlay: bool,
    /// true once the user has pressed "Calculate" with scope = current view; keeps
    /// the result live-recomputed as the view scrolls, until the scope is changed
    spectrum_want_live: bool,
    spectrum_result: Option<Arc<crate::spectrum::ComputedSpectrum>>,
    spectrum_texture: Option<TextureHandle>,
    /// inputs the current `spectrum_texture` was built from; rebuilt only when they change
    spectrum_tex_key: Option<SpectrumTexKey>,
    spectrum_pixel_buf: Vec<u8>,
    spectrum_computing: bool,
    spectrum_error: Option<String>,
    spectrum_rx: Option<mpsc::Receiver<Result<crate::spectrum::ComputedSpectrum, String>>>,
    spectrum_cancel: Arc<AtomicBool>,
    spectrum_progress: Arc<AtomicUsize>,
    spectrum_run_total: usize,
    // live current-view recompute waits until the view has stopped moving: the
    // last (view_first, view_n) seen and when it last changed
    spectrum_view_seen: (usize, usize),
    spectrum_view_changed_at: std::time::Instant,

    // async worker
    worker_state: SharedWorkerState,
    worker_cancel: SharedCancel,
    worker_half_window: usize,
    initial_buffer_s: f64,
    extension_margin_s: f64,
    mem_pressure_pct: f32,
    mem_reserve_mb: f64,
    _worker_handle: std::thread::JoinHandle<()>,

    // rendering
    heatmap_texture: Option<TextureHandle>,
    pixel_buf: Vec<u8>,
    last_rendered_first: usize,
    last_rendered_cfg: Option<PreprocConfig>,
    last_rendered_n: usize,
    last_rendered_size: Option<[usize; 2]>,
    last_rendered_buf: Option<(usize, usize)>,
    last_rendered_rows: Option<(usize, usize)>,

    // rectangle zoom
    zoom: Option<Zoom>,
    /// screen position where the current zoom drag started
    zoom_drag_start: Option<egui::Pos2>,

    // UI state
    pending_cfg_recompute: bool,
    pub file_dialog_request: bool,
    /// set by the toolbar's "Recent files" menu; polled by MainApp to switch recordings
    pub open_recent_request: Option<PathBuf>,
    recent_files: Vec<PathBuf>,
    projection_sums: Vec<f32>,
    // spike projection cache keys
    proj_view_first: usize,
    proj_view_n: usize,
    proj_threshold: f32,
    proj_sigma: f32,
    proj_cfg: Option<PreprocConfig>,
    proj_rows: Option<(usize, usize)>,

    psth: PsthState,
    atlas: crate::atlas_ui::AtlasUi,
    ttl: crate::ttl::TtlState,
    /// stim-file format text shared by the PSTH and TTL windows
    stim_layout_text: String,
    /// stimulus file last loaded in the TTL or PSTH window (saved per recording)
    stim_file: Option<PathBuf>,

    // the recording's settings file (settings.rs): which section is ours, what was
    // last written, and since when the settings differ from it
    band: crate::settings::Band,
    settings_saved: Option<crate::settings::RecordingSettings>,
    settings_changed_at: Option<std::time::Instant>,
    settings_error: Option<String>,

    // screenshot of the plot area
    /// the central panel: heatmap plus spectrum strip, or the waveform view
    plot_rect: Option<egui::Rect>,
    screenshot: ScreenshotState,
    capture: Option<Capture>,
}

/// How long the settings must have differed from the file before it is rewritten,
/// so scrolling or dragging a value doesn't write on every frame.
const SETTINGS_SAVE_DELAY: std::time::Duration = std::time::Duration::from_secs(1);

/// Overlays that can be left out of a screenshot.
#[derive(Clone, Copy, PartialEq)]
struct ShotOverlays {
    ttl: bool,
    atlas: bool,
    firing_rate: bool,
    spectrum: bool,
    classification: bool,
    selection: bool,
    scale_bar: bool,
}

/// The "Screenshot" window.
#[derive(Default)]
struct ScreenshotState {
    open: bool,
    /// overlays to include; filled from what is shown when the window opens
    include: Option<ShotOverlays>,
    pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    /// result of the last screenshot: (message, success)
    note: Option<(String, bool)>,
}

/// A screenshot being taken: the windows, the legend and the hover readout are hidden,
/// the overlays set to the chosen ones, until the image arrives.
struct Capture {
    path: PathBuf,
    include: ShotOverlays,
    /// overlay visibility to restore afterwards
    restore: ShotOverlays,
    /// frames drawn so far in screenshot mode; the image is requested in the second,
    /// the first one drawn entirely without windows
    frames: u32,
}

/// Tags our screenshot request, so the PSTH export's doesn't take its image.
struct MainScreenshot;

impl NPXplorerApp {
    pub fn new(ctx: &egui::Context, bin_path: PathBuf) -> anyhow::Result<Self> {
        let meta = Arc::new(Meta::from_data_path(&bin_path)?);
        let raw = Arc::new(open_data(&bin_path, &meta)?);
        let fs = meta.sample_rate;

        let prefs = Preferences::load();

        let mut preproc_cfg = PreprocConfig {
            dc_removal: true,
            phase_shift: false,
            highpass: true,
            spatial_filter: SpatialFilter::GlobalCmr,
            avg_depths: true,
            sample_rate: fs,
            removed_channels: Default::default(),
            channel_order: Default::default(),
            shank_order: Default::default(),
            notches: Vec::new(),
            notch_enabled: false,
        };

        let mut view_dur_s = 0.5;
        let mut color_mode = ColorMode::Percentile;
        let mut color_pct = 99.0;
        let mut color_uv = 120.0;
        let mut colormap_choice = ColorMapChoice::IceFire;
        let mut spike_threshold = -40.0;
        let mut show_firing_rate_overlay = true;
        let mut spike_overlay_scale = default_spike_overlay_scale();
        let mut spike_smoothing_sigma = default_spike_smoothing_sigma();
        let mut initial_buffer_s = default_initial_buffer_s();
        let mut extension_margin_s = default_extension_margin_s();
        let mut mem_pressure_pct = default_mem_pressure_pct();
        let mut mem_reserve_mb = default_mem_reserve_mb();
        let mut recent_files: Vec<PathBuf> = Vec::new();
        let mut n_classify_chunks = default_n_classify_chunks();
        let mut classify_outside_rule = crate::channel_classify::OutsideRule::default();
        let mut peak_pooling = false;
        let mut atlas_dir = None;
        let mut bregma_lambda_mm = default_bregma_lambda_mm();
        let mut atlas_min_region_channels = default_atlas_min_region_channels();
        let mut noise = crate::noise::NoiseSuppression::default();
        let mut spectrum_time_scope = crate::spectrum::SpectrumTimeScope::default();
        let mut spectrum_source = crate::spectrum::SpectrumSource::default();
        let mut spectrum_scaling = crate::spectrum::SpectrumScaling::default();
        let mut spectrum_normalization = crate::spectrum::SpectrumNormalization::default();
        let mut spectrum_n_chunks = default_spectrum_n_chunks();
        let mut spectrum_freq_restrict = false;
        let mut spectrum_freq_min_hz = default_spectrum_freq_min_hz();
        let mut spectrum_freq_max_hz = default_spectrum_freq_max_hz();
        let mut spectrum_time_restrict = false;
        let mut spectrum_time_start_s = 0.0;
        let mut spectrum_time_end_s = default_spectrum_time_end_s();

        if let Some(p) = prefs {
            preproc_cfg = p.preproc_cfg;
            preproc_cfg.sample_rate = fs;
            view_dur_s = p.view_dur_s;
            color_mode = p.color_mode;
            color_pct = p.color_pct;
            color_uv = p.color_uv;
            colormap_choice = p.colormap_choice;
            spike_threshold = p.spike_threshold;
            show_firing_rate_overlay = p.show_firing_rate_overlay;
            spike_overlay_scale = p.spike_overlay_scale;
            spike_smoothing_sigma = p.spike_smoothing_sigma;
            initial_buffer_s = p.initial_buffer_s;
            extension_margin_s = p.extension_margin_s;
            mem_pressure_pct = p.mem_pressure_pct;
            mem_reserve_mb = p.mem_reserve_mb;
            recent_files = p.recent_files.iter().map(PathBuf::from).collect();
            n_classify_chunks = p.n_classify_chunks;
            classify_outside_rule = p.classify_outside_rule;
            peak_pooling = p.peak_pooling;
            atlas_dir = p.atlas_dir;
            bregma_lambda_mm = p.bregma_lambda_mm;
            atlas_min_region_channels = p.atlas_min_region_channels;
            noise = p.noise_suppression;
            spectrum_time_scope = p.spectrum_time_scope;
            spectrum_source = p.spectrum_source;
            spectrum_scaling = p.spectrum_scaling;
            spectrum_normalization = p.spectrum_normalization;
            spectrum_n_chunks = p.spectrum_n_chunks;
            spectrum_freq_restrict = p.spectrum_freq_restrict;
            spectrum_freq_min_hz = p.spectrum_freq_min_hz;
            spectrum_freq_max_hz = p.spectrum_freq_max_hz;
            spectrum_time_restrict = p.spectrum_time_restrict;
            spectrum_time_start_s = p.spectrum_time_start_s;
            spectrum_time_end_s = p.spectrum_time_end_s;
        }

        // the recording's own settings (see settings.rs); the band section is applied
        // once the app exists, below
        let settings_file = crate::settings::load_table(&bin_path);
        let band = crate::settings::Band::of_sample_rate(fs);

        // removed channels: as saved, else the reference sites, which carry no neural
        // signal (listed in the Remove-channels dialog; Reset brings them back). Set
        // before the filters and the first preprocessing request are made.
        let saved_removed = settings_file
            .get("removed_channels")
            .and_then(|v| v.clone().try_into::<Vec<usize>>().ok())
            .map(|v| v.into_iter().filter(|&c| c < meta.n_ap_chans).collect::<BTreeSet<usize>>())
            .filter(|set| set.len() < meta.n_ap_chans);
        preproc_cfg.removed_channels = match saved_removed {
            Some(set) => set,
            None if meta.reference_channels.len() < meta.n_ap_chans => meta.reference_channels.clone(),
            None => BTreeSet::new(),
        };

        let stim_settings = match settings_file.get("stim") {
            Some(v) => crate::settings::overlay(&crate::settings::StimSettings::default(), Some(v)),
            None => crate::settings::legacy_stim(&bin_path).unwrap_or_default(),
        };
        let atlas_default = crate::settings::AtlasSettings {
            insertion: crate::atlas::Insertion { bregma_lambda_mm, ..Default::default() },
            ..Default::default()
        };
        let atlas_settings = match settings_file.get("atlas") {
            Some(v) => crate::settings::overlay(&atlas_default, Some(v)),
            None => crate::settings::legacy_atlas(&bin_path).unwrap_or(atlas_default),
        };

        // move this recording to the front of the recent-files list (max 5)
        recent_files.retain(|p| p != &bin_path);
        recent_files.insert(0, bin_path.clone());
        recent_files.truncate(5);

        let filters = Arc::new(Mutex::new(Filters::new(&preproc_cfg)));
        let shared: SharedWorkerState = Arc::new((Mutex::new(WorkerState::new()), Condvar::new()));
        let cancel: SharedCancel = Arc::new(AtomicBool::new(false));
        // the first request is sent by `start`, once the settings are applied
        let handle = spawn_worker(
            Arc::clone(&raw),
            Arc::clone(&meta),
            Arc::clone(&filters),
            Arc::clone(&shared),
            Arc::clone(&cancel),
            ctx.clone(),
        );

        let is_compressed = bin_path.extension().and_then(|s| s.to_str()) == Some("cbin");

        let meta_for_atlas = Arc::clone(&meta);
        let psth_total_s = meta.n_samples as f64 / meta.sample_rate;
        let mut psth = PsthState::new(psth_total_s);
        psth.restore(stim_settings.stim_file.as_deref(), &stim_settings.psth);
        let remove_channels_text = crate::channel_remove::format_channel_list(
            &preproc_cfg.removed_channels,
            &meta.channel_ids,
        );
        let notch_panel = crate::notch::NotchPanel::new(&[], Default::default());

        let mut app = Self {
            bin_path,
            meta,
            raw: Arc::clone(&raw),
            is_compressed,
            view_start_s: 0.0,
            view_dur_s,
            window_dur_str: format!("{:.3}", view_dur_s),
            jump_str: "0.000".to_string(),
            preproc_cfg: preproc_cfg.clone(),
            preproc_filters: filters,
            scroll_speed_fine: true,
            color_mode,
            color_pct,
            color_uv,
            color_pct_str: format!("{:.2}", color_pct),
            color_uv_str: format!("{:.0}", color_uv),
            colormap_choice,
            show_preferences: false,
            spike_threshold,
            show_firing_rate_overlay,
            spike_overlay_scale,
            spike_smoothing_sigma,
            selected_channel_1: None,
            selected_channel_2: None,
            context_menu_channel: None,
            context_menu_pos: None,
            waveform_channel: None,
            waveform_y_range_uv: 200.0,
            waveform_y_center_uv: 0.0,
            waveform_zoom: None,
            show_remove_channels: false,
            remove_channels_text,
            remove_channels_error: None,
            remove_channels_pick_rx: None,
            channel_labels: None,
            show_classification_overlay: false,
            classifying: false,
            classify_error: None,
            classify_rx: None,
            classify_cancel: Arc::new(AtomicBool::new(false)),
            classify_progress: Arc::new(AtomicUsize::new(0)),
            classify_n_chunks: n_classify_chunks,
            classify_outside_rule,
            peak_pooling,
            classify_run_total: 0,
            noise_open: false,
            notch_panel,
            noise_draft: noise.clone(),
            noise,
            noise_view: None,
            last_rendered_noise: None,
            spectrum_open: false,
            spectrum_time_scope,
            spectrum_source,
            spectrum_scaling,
            spectrum_normalization,
            spectrum_n_chunks,
            spectrum_freq_restrict,
            spectrum_freq_min_hz,
            spectrum_freq_max_hz,
            spectrum_time_restrict,
            spectrum_time_start_s,
            spectrum_time_end_s,
            spectrum_show_overlay: false,
            spectrum_want_live: false,
            spectrum_result: None,
            spectrum_texture: None,
            spectrum_tex_key: None,
            spectrum_pixel_buf: Vec::new(),
            spectrum_computing: false,
            spectrum_error: None,
            spectrum_rx: None,
            spectrum_cancel: Arc::new(AtomicBool::new(false)),
            spectrum_progress: Arc::new(AtomicUsize::new(0)),
            spectrum_run_total: 0,
            spectrum_view_seen: (usize::MAX, 0),
            spectrum_view_changed_at: std::time::Instant::now(),
            worker_state: shared,
            worker_cancel: cancel,
            worker_half_window: compute_half_window(initial_buffer_s, fs),
            initial_buffer_s,
            extension_margin_s,
            mem_pressure_pct,
            mem_reserve_mb,
            _worker_handle: handle,
            heatmap_texture: None,
            pixel_buf: Vec::new(),
            last_rendered_first: usize::MAX,
            last_rendered_cfg: None,
            last_rendered_n: 0,
            last_rendered_size: None,
            last_rendered_buf: None,
            last_rendered_rows: None,
            zoom: None,
            zoom_drag_start: None,
            pending_cfg_recompute: false,
            file_dialog_request: false,
            open_recent_request: None,
            recent_files,
            projection_sums: Vec::new(),
            proj_view_first: usize::MAX,
            proj_view_n: 0,
            proj_threshold: 0.0,
            proj_sigma: 0.0,
            proj_cfg: None,
            proj_rows: None,
            psth,
            atlas: crate::atlas_ui::AtlasUi::new(&meta_for_atlas, atlas_dir, atlas_settings, atlas_min_region_channels),
            ttl: crate::ttl::TtlState::new(&stim_settings.ttl, stim_settings.stim_file.as_deref()),
            stim_layout_text: stim_settings.layout.clone().unwrap_or_else(crate::psth::default_layout_text),
            stim_file: stim_settings.stim_file.clone(),
            band,
            settings_saved: None,
            settings_changed_at: None,
            settings_error: None,
            plot_rect: None,
            screenshot: ScreenshotState::default(),
            capture: None,
        };

        // this band's settings, over the ones taken from the preferences above
        let key = band.key();
        let mut band_settings = crate::settings::overlay(&app.band_settings(), settings_file.get(key));
        if settings_file.get(key).is_none() {
            // notches saved by an earlier version are restored and switched on
            if let Some(notches) = crate::settings::legacy_notches(&app.bin_path) {
                band_settings.preproc.notch_enabled = !notches.is_empty();
                band_settings.preproc.notches = notches;
            }
        }
        app.apply_band_settings(band_settings);
        app.start(ctx);
        Ok(app)
    }

    /// Clamp the buffer settings, start preprocessing, and load the atlas if its
    /// overlay was shown last time. Writes the settings file, so a recording opened
    /// once keeps its settings even when the preferences change later.
    fn start(&mut self, ctx: &egui::Context) {
        let fs = self.meta.sample_rate;
        // defensively re-clamp in case prefs were saved on a machine with more RAM,
        // or with an initial_buffer_s/view_dur_s combination that no longer satisfies
        // the no-oscillation bound
        let n_data_rows = self
            .meta
            .build_display_rows(
                self.preproc_cfg.avg_depths,
                &self.preproc_cfg.removed_channels,
                self.preproc_cfg.channel_order,
                self.preproc_cfg.shank_order,
            )
            .iter()
            .filter(|r| matches!(r, DisplayRow::Data { .. }))
            .count();
        self.initial_buffer_s =
            self.initial_buffer_s.min(max_feasible_buffer_s(n_data_rows, fs, self.mem_reserve_mb));
        self.extension_margin_s =
            self.extension_margin_s.min(max_extension_margin_s(self.initial_buffer_s, self.view_dur_s));
        self.worker_half_window = compute_half_window(self.initial_buffer_s, fs);

        *self.preproc_filters.lock().unwrap() = Filters::new(&self.preproc_cfg);
        self.request_recompute();
        let meta = Arc::clone(&self.meta);
        self.atlas.register_on_open(ctx, &meta);
        self.save_settings();
    }
}

impl Drop for NPXplorerApp {
    /// Stop the worker thread when the recording is closed, so its thread pool, the
    /// preprocessed buffer and the mapped data file are released. Unsaved settings
    /// changes are written first.
    fn drop(&mut self) {
        self.flush_settings();
        request_shutdown(&self.worker_state, &self.worker_cancel, &self.preproc_cfg);
    }
}

impl NPXplorerApp {
    /// This band's settings, as saved in the recording's settings file.
    fn band_settings(&self) -> crate::settings::BandSettings {
        use crate::settings::*;
        let c = &self.preproc_cfg;
        // a zoomed view isn't restored; the view from before the (outermost) zoom is
        let (view_start_s, view_dur_s) = match (&self.zoom, &self.waveform_zoom) {
            (Some(z), _) => (z.prev_start_s, z.prev_dur_s),
            (None, Some(z)) => (z.prev_start_s, z.prev_dur_s),
            (None, None) => (self.view_start_s, self.view_dur_s),
        };
        BandSettings {
            view_start_s,
            view_dur_s,
            scroll_fine: self.scroll_speed_fine,
            color_mode: self.color_mode.clone(),
            color_pct: self.color_pct,
            color_uv: self.color_uv,
            colormap: self.colormap_choice.clone(),
            peak_pooling: self.peak_pooling,
            waveform_y_range_uv: self.waveform_zoom.as_ref().map_or(self.waveform_y_range_uv, |z| z.prev_y_range_uv),
            preproc: PreprocSettings {
                dc_removal: c.dc_removal,
                phase_shift: c.phase_shift,
                highpass: c.highpass,
                spatial_filter: c.spatial_filter,
                avg_depths: c.avg_depths,
                channel_order: c.channel_order,
                shank_order: c.shank_order,
                notch_enabled: c.notch_enabled,
                notches: c.notches.clone(),
            },
            firing_rate: FiringRateSettings {
                show: self.show_firing_rate_overlay,
                threshold_uv: self.spike_threshold,
                overlay_scale: self.spike_overlay_scale,
                smoothing_sigma: self.spike_smoothing_sigma,
            },
            classification: ClassificationSettings {
                n_chunks: self.classify_n_chunks,
                outside_rule: self.classify_outside_rule,
                show_overlay: self.show_classification_overlay,
            },
            spectrum: SpectrumSettings {
                time_scope: self.spectrum_time_scope,
                source: self.spectrum_source,
                scaling: self.spectrum_scaling,
                normalization: self.spectrum_normalization,
                n_chunks: self.spectrum_n_chunks,
                freq_restrict: self.spectrum_freq_restrict,
                freq_min_hz: self.spectrum_freq_min_hz,
                freq_max_hz: self.spectrum_freq_max_hz,
                time_restrict: self.spectrum_time_restrict,
                time_start_s: self.spectrum_time_start_s,
                time_end_s: self.spectrum_time_end_s,
                show_overlay: self.spectrum_show_overlay,
            },
            noise: self.noise.clone(),
            notch_scan: self.notch_panel.scan_settings().clone(),
        }
    }

    /// Take over a band's saved settings; called before preprocessing starts.
    fn apply_band_settings(&mut self, s: crate::settings::BandSettings) {
        let total_s = self.meta.n_samples as f64 / self.meta.sample_rate;
        self.view_dur_s = s.view_dur_s.clamp(0.01, 10.0);
        self.view_start_s = s.view_start_s.clamp(0.0, (total_s - self.view_dur_s).max(0.0));
        self.window_dur_str = format!("{:.3}", self.view_dur_s);
        self.jump_str = format!("{:.3}", self.view_start_s);
        self.scroll_speed_fine = s.scroll_fine;
        self.color_mode = s.color_mode;
        self.color_pct = s.color_pct.clamp(95.0, 100.0);
        self.color_uv = s.color_uv.clamp(10.0, 300.0);
        self.color_pct_str = format!("{:.2}", self.color_pct);
        self.color_uv_str = format!("{:.0}", self.color_uv);
        self.colormap_choice = s.colormap;
        self.peak_pooling = s.peak_pooling;
        self.waveform_y_range_uv = s.waveform_y_range_uv.clamp(WAVEFORM_MIN_RANGE_UV, 2000.0);

        let c = &mut self.preproc_cfg;
        c.dc_removal = s.preproc.dc_removal;
        c.phase_shift = s.preproc.phase_shift;
        c.spatial_filter = s.preproc.spatial_filter;
        // destripe includes the highpass
        c.highpass = s.preproc.highpass || c.spatial_filter == SpatialFilter::Destripe;
        c.avg_depths = s.preproc.avg_depths;
        c.channel_order = s.preproc.channel_order;
        c.shank_order = s.preproc.shank_order;
        c.notch_enabled = s.preproc.notch_enabled && !s.preproc.notches.is_empty();
        c.notches = s.preproc.notches;
        self.notch_panel = crate::notch::NotchPanel::new(&self.preproc_cfg.notches, s.notch_scan);

        self.show_firing_rate_overlay = s.firing_rate.show;
        self.spike_threshold = s.firing_rate.threshold_uv;
        self.spike_overlay_scale = s.firing_rate.overlay_scale;
        self.spike_smoothing_sigma = s.firing_rate.smoothing_sigma;
        self.classify_n_chunks = s.classification.n_chunks;
        self.classify_outside_rule = s.classification.outside_rule;
        self.show_classification_overlay = s.classification.show_overlay;

        self.spectrum_time_scope = s.spectrum.time_scope;
        self.spectrum_source = s.spectrum.source;
        self.spectrum_scaling = s.spectrum.scaling;
        self.spectrum_normalization = s.spectrum.normalization;
        self.spectrum_n_chunks = s.spectrum.n_chunks;
        self.spectrum_freq_restrict = s.spectrum.freq_restrict;
        self.spectrum_freq_min_hz = s.spectrum.freq_min_hz;
        self.spectrum_freq_max_hz = s.spectrum.freq_max_hz;
        self.spectrum_time_restrict = s.spectrum.time_restrict;
        self.spectrum_time_start_s = s.spectrum.time_start_s;
        self.spectrum_time_end_s = s.spectrum.time_end_s;
        self.spectrum_show_overlay = s.spectrum.show_overlay;

        self.noise_draft = s.noise.clone();
        self.noise = s.noise;
    }

    /// Everything saved in the recording's settings file.
    fn recording_settings(&self) -> crate::settings::RecordingSettings {
        let mut s = crate::settings::RecordingSettings {
            removed_channels: Some(self.preproc_cfg.removed_channels.iter().copied().collect()),
            stim: Some(crate::settings::StimSettings {
                stim_file: self.stim_file.clone(),
                layout: Some(self.stim_layout_text.clone()),
                ttl: self.ttl.settings(),
                psth: self.psth.settings(),
            }),
            atlas: Some(self.atlas.settings()),
            ..Default::default()
        };
        *s.band_mut(self.band) = Some(self.band_settings());
        s
    }

    /// Write the settings file (and the preferences, which hold the last-used settings
    /// as defaults for recordings opened for the first time).
    fn save_settings(&mut self) {
        let current = self.recording_settings();
        let mut to_write = current.clone();
        if let Some(stim) = &mut to_write.stim {
            stim.layout = stim.layout.as_deref().and_then(crate::psth::layout_to_save);
        }
        self.settings_error = crate::settings::save(&self.bin_path, self.band, &to_write)
            .err()
            .map(|e| format!("settings not saved: {e:#}"));
        self.settings_saved = Some(current);
        self.settings_changed_at = None;
        self.save_prefs();
    }

    /// Rewrite the settings file once they have differed from it for
    /// `SETTINGS_SAVE_DELAY` and no mouse button is held (a drag in progress).
    fn autosave_settings(&mut self, ctx: &egui::Context) {
        if self.capture.is_some() {
            return; // overlay visibility is temporarily changed for the screenshot
        }
        if self.settings_saved.as_ref() == Some(&self.recording_settings()) {
            self.settings_changed_at = None;
            return;
        }
        let since = *self.settings_changed_at.get_or_insert_with(std::time::Instant::now);
        if since.elapsed() >= SETTINGS_SAVE_DELAY && !ctx.input(|i| i.pointer.any_down()) {
            self.save_settings();
        } else {
            ctx.request_repaint_after(SETTINGS_SAVE_DELAY);
        }
    }

    /// Write unsaved settings changes now (closing the recording or the app).
    pub fn flush_settings(&mut self) {
        if let Some(c) = self.capture.take() {
            self.set_overlay_visibility(c.restore);
        }
        if self.settings_saved.as_ref() != Some(&self.recording_settings()) {
            self.save_settings();
        }
    }
}

impl NPXplorerApp {
    pub fn save_prefs(&self) {
        let last_dir = self
            .bin_path
            .parent()
            .map(|p| p.to_string_lossy().to_string());
        let prefs = Preferences {
            preproc_cfg: self.preproc_cfg.clone(),
            // a zoomed window length is not a sensible default for the next session
            view_dur_s: self.zoom.as_ref().map_or(self.view_dur_s, |z| z.prev_dur_s),
            color_mode: self.color_mode.clone(),
            color_pct: self.color_pct,
            color_uv: self.color_uv,
            colormap_choice: self.colormap_choice.clone(),
            spike_threshold: self.spike_threshold,
            show_firing_rate_overlay: self.show_firing_rate_overlay,
            spike_overlay_scale: self.spike_overlay_scale,
            spike_smoothing_sigma: self.spike_smoothing_sigma,
            n_classify_chunks: self.classify_n_chunks,
            classify_outside_rule: self.classify_outside_rule,
            peak_pooling: self.peak_pooling,
            initial_buffer_s: self.initial_buffer_s,
            extension_margin_s: self.extension_margin_s,
            mem_pressure_pct: self.mem_pressure_pct,
            mem_reserve_mb: self.mem_reserve_mb,
            last_dir,
            recent_files: self
                .recent_files
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect(),
            atlas_dir: self.atlas.atlas_dir(),
            bregma_lambda_mm: self.atlas.bregma_lambda_mm(),
            atlas_min_region_channels: self.atlas.min_region_channels(),
            noise_suppression: self.noise.clone(),
            spectrum_time_scope: self.spectrum_time_scope,
            spectrum_source: self.spectrum_source,
            spectrum_scaling: self.spectrum_scaling,
            spectrum_normalization: self.spectrum_normalization,
            spectrum_n_chunks: self.spectrum_n_chunks,
            spectrum_freq_restrict: self.spectrum_freq_restrict,
            spectrum_freq_min_hz: self.spectrum_freq_min_hz,
            spectrum_freq_max_hz: self.spectrum_freq_max_hz,
            spectrum_time_restrict: self.spectrum_time_restrict,
            spectrum_time_start_s: self.spectrum_time_start_s,
            spectrum_time_end_s: self.spectrum_time_end_s,
        };
        prefs.save();
    }

    fn request_recompute(&mut self) {
        let fs = self.meta.sample_rate;
        // use same formula as update() to avoid off-by-one from float rounding
        let view_first = (self.view_start_s * fs) as usize;
        let view_n = (self.view_dur_s * fs) as usize;
        let center = view_first + view_n / 2;
        file_log!(
            "UI: request_recompute center={} view_first={} view_n={} cfg={:?}",
            center,
            view_first,
            view_n,
            self.preproc_cfg
        );
        // cancel any in-flight computation
        self.worker_cancel.store(true, Ordering::Relaxed);
        let req = WorkerRequest {
            kind: RequestKind::Full {
                center_sample: center,
                half_window: self.worker_half_window,
            },
            cfg: self.preproc_cfg.clone(),
        };
        let (lock, cvar) = &*self.worker_state;
        lock.lock().unwrap().request = Some(req);
        cvar.notify_one();
    }

    // -----------------------------------------------------------------------
    // Channel removal
    // -----------------------------------------------------------------------

    fn apply_removed_channels(&mut self, set: BTreeSet<usize>) {
        let n_left = (0..self.meta.n_ap_chans)
            .filter(|c| !set.contains(c))
            .count();
        if n_left == 0 {
            self.remove_channels_error =
                Some("this would remove every channel — at least one must stay".to_string());
            return;
        }
        self.preproc_cfg.removed_channels = set;
        self.heatmap_texture = None;
        self.pending_cfg_recompute = true;
    }

    /// Parse `remove_channels_text` and apply it, or set an error message on failure.
    fn apply_remove_channels_text(&mut self) {
        match crate::channel_remove::parse_channel_list(
            &self.remove_channels_text,
            &self.meta.channel_ids,
        ) {
            Ok(set) => {
                self.remove_channels_error = None;
                self.apply_removed_channels(set);
            }
            Err(e) => self.remove_channels_error = Some(e.to_string()),
        }
    }

    /// Add a single 1-based channel to the removed-channels list (e.g. from the
    /// context menu), keeping the Remove-channels dialog's text field in sync, and
    /// force an immediate recompute.
    fn remove_channel(&mut self, ch: usize) {
        let mut set = self.preproc_cfg.removed_channels.clone();
        set.insert(ch - 1);
        self.remove_channels_text =
            crate::channel_remove::format_channel_list(&set, &self.meta.channel_ids);
        self.remove_channels_error = None;
        self.apply_removed_channels(set);
    }

    fn poll_remove_channels_picker(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.remove_channels_pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.remove_channels_pick_rx = None;
                    if let Some(path) = picked {
                        let ids = &self.meta.channel_ids;
                        let default_layout = crate::channel_remove::default_layout_path();
                        let result = crate::channel_remove::resolve_layout(&path, &default_layout)
                            .and_then(|layout| {
                                crate::channel_remove::load_removed_channels(&path, &layout, ids)
                            });
                        match result {
                            Ok(set) => {
                                self.remove_channels_text =
                                    crate::channel_remove::format_channel_list(
                                        &set,
                                        &self.meta.channel_ids,
                                    );
                                self.remove_channels_error = None;
                                self.apply_removed_channels(set);
                            }
                            Err(e) => self.remove_channels_error = Some(e.to_string()),
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => self.remove_channels_pick_rx = None,
            }
        }
    }

    fn draw_remove_channels_window(&mut self, ctx: &egui::Context) {
        if !self.show_remove_channels {
            return;
        }
        let mut open = self.show_remove_channels;
        egui::Window::new("Remove channels")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                let example = crate::channel_remove::example_list(&self.meta.channel_ids);
                ui.label(format!(
                    "Channels to exclude from display and computation, by channel ID \
                     (e.g. \"{example}\"; a bare number matches the ID's number):"
                ));
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.remove_channels_text)
                        .desired_width(250.0)
                        .hint_text(format!("e.g. {example}")),
                );
                if resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    self.apply_remove_channels_text();
                }

                ui.horizontal(|ui| {
                    if ui.button("Apply").clicked() {
                        self.apply_remove_channels_text();
                    }
                    if ui.button("Load from file…").clicked()
                        && self.remove_channels_pick_rx.is_none()
                    {
                        self.remove_channels_pick_rx = Some(spawn_channel_list_picker(
                            self.bin_path.parent().map(|p| p.to_path_buf()),
                        ));
                    }
                    if ui.button("Reset").clicked() {
                        self.remove_channels_text.clear();
                        self.remove_channels_error = None;
                        self.apply_removed_channels(BTreeSet::new());
                    }
                });

                let n_removed = self.preproc_cfg.removed_channels.len();
                if n_removed > 0 {
                    ui.label(format!("{n_removed} channel(s) currently removed."));
                }
                if let Some(err) = &self.remove_channels_error {
                    ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), err);
                }
            });
        self.show_remove_channels = open;
    }

    // -----------------------------------------------------------------------
    // IBL-style channel classification (dead/noisy/outside-of-brain/good)
    // -----------------------------------------------------------------------

    fn dispatch_classify(&mut self, ctx: &egui::Context) {
        self.classify_cancel.store(true, Ordering::Relaxed);
        let cancel = Arc::new(AtomicBool::new(false));
        self.classify_cancel = Arc::clone(&cancel);
        let progress = Arc::new(AtomicUsize::new(0));
        self.classify_progress = Arc::clone(&progress);

        let (tx, rx) = mpsc::channel();
        self.classify_rx = Some(rx);
        self.classifying = true;
        self.classify_error = None;

        let raw = Arc::clone(&self.raw);
        let meta = Arc::clone(&self.meta);
        let removed = self.preproc_cfg.removed_channels.clone();
        let n_chunks = self.classify_n_chunks.max(1);
        let outside_rule = self.classify_outside_rule;
        self.classify_run_total = n_chunks;
        let ctx = ctx.clone();

        std::thread::spawn(move || {
            let res = crate::channel_classify::classify_recording(
                &raw,
                &meta,
                &removed,
                n_chunks,
                outside_rule,
                &cancel,
                &progress,
            )
            .ok_or_else(|| "cancelled".to_string());
            let _ = tx.send(res);
            ctx.request_repaint();
        });
    }

    fn poll_classify(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.classify_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.classify_rx = None;
                    self.classifying = false;
                    match res {
                        Ok(labels) => {
                            self.channel_labels = Some(Arc::new(labels));
                            self.show_classification_overlay = true;
                            self.classify_error = None;
                        }
                        Err(e) if e == "cancelled" => {}
                        Err(e) => self.classify_error = Some(e),
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(80));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.classify_rx = None;
                    self.classifying = false;
                }
            }
        }
    }

    fn draw_classify_progress_window(&mut self, ctx: &egui::Context) {
        if self.classifying {
            let done = self.classify_progress.load(Ordering::Relaxed);
            let total = self.classify_run_total;
            egui::Window::new("Channel Classification")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.set_min_width(220.0);
                    ui.label("Scanning recording for dead / noisy / out-of-brain channels…");
                    ui.add(
                        egui::ProgressBar::new(done as f32 / total.max(1) as f32).show_percentage(),
                    );
                    ui.label(format!("{done} / {total} chunks"));
                    if ui.button("Abort").clicked() {
                        self.classify_cancel.store(true, Ordering::Relaxed);
                    }
                });
            return;
        }

        if let Some(err) = self.classify_error.clone() {
            let mut dismiss = false;
            egui::Window::new("Channel Classification failed")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), &err);
                    if ui.button("OK").clicked() {
                        dismiss = true;
                    }
                });
            if dismiss {
                self.classify_error = None;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Power spectrum overlay
    // -----------------------------------------------------------------------

    /// `None` shows the full band up to Nyquist; see `spectrum_freq_restrict`.
    fn spectrum_freq_range(&self) -> Option<(f32, f32)> {
        self.spectrum_freq_restrict
            .then(|| (self.spectrum_freq_min_hz as f32, self.spectrum_freq_max_hz as f32))
    }

    /// `None` sources whole-recording chunks from the whole recording; see
    /// `spectrum_time_restrict`.
    fn spectrum_time_range(&self) -> Option<(f64, f64)> {
        self.spectrum_time_restrict
            .then_some((self.spectrum_time_start_s, self.spectrum_time_end_s))
    }

    /// Background scan: evenly-spaced raw chunks across the whole recording.
    fn dispatch_spectrum(&mut self, ctx: &egui::Context) {
        self.spectrum_cancel.store(true, Ordering::Relaxed);
        let cancel = Arc::new(AtomicBool::new(false));
        self.spectrum_cancel = Arc::clone(&cancel);
        let progress = Arc::new(AtomicUsize::new(0));
        self.spectrum_progress = Arc::clone(&progress);

        let (tx, rx) = mpsc::channel();
        self.spectrum_rx = Some(rx);
        self.spectrum_computing = true;
        self.spectrum_error = None;

        let raw = Arc::clone(&self.raw);
        let meta = Arc::clone(&self.meta);
        let display_rows = self.meta.build_display_rows(
            self.preproc_cfg.avg_depths,
            &self.preproc_cfg.removed_channels,
            self.preproc_cfg.channel_order,
            self.preproc_cfg.shank_order,
        );
        let n_chunks = self.spectrum_n_chunks.max(1);
        let time_range = self.spectrum_time_range();
        self.spectrum_run_total = n_chunks;
        // the row layout this job's rows are indexed by travels with the result, so a
        // result is only ever shown against the layout it was computed for
        let cfg = self.preproc_cfg.clone();
        let ctx = ctx.clone();

        std::thread::spawn(move || {
            let res = crate::spectrum::compute_psd_whole_recording(
                &raw,
                &meta,
                &display_rows,
                n_chunks,
                time_range,
                &cancel,
                &progress,
            )
            .map(|result| crate::spectrum::ComputedSpectrum {
                result,
                cfg,
                source: crate::spectrum::SpectrumSource::Raw,
                view: None,
            });
            let _ = tx.send(res);
            ctx.request_repaint();
        });
    }

    fn poll_spectrum(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.spectrum_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.spectrum_rx = None;
                    self.spectrum_computing = false;
                    match res {
                        Ok(result) => {
                            self.spectrum_result = Some(Arc::new(result));
                            self.spectrum_show_overlay = true;
                            self.spectrum_error = None;
                        }
                        Err(e) if e == "cancelled" => {}
                        Err(e) => self.spectrum_error = Some(e),
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(80));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.spectrum_rx = None;
                    self.spectrum_computing = false;
                }
            }
        }
    }

    fn draw_spectrum_error_window(&mut self, ctx: &egui::Context) {
        if let Some(err) = self.spectrum_error.clone() {
            let mut dismiss = false;
            egui::Window::new("Power Spectrum failed")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), &err);
                    if ui.button("OK").clicked() {
                        dismiss = true;
                    }
                });
            if dismiss {
                self.spectrum_error = None;
            }
        }
    }

    /// Settings window opened by the "Noise Suppression" toolbar button.
    fn draw_noise_window(&mut self, ctx: &egui::Context) {
        if !self.noise_open {
            return;
        }
        let mut open = self.noise_open;
        egui::Window::new("Noise Suppression")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.strong("Display filters");
                    match crate::noise::draw_settings(ui, &mut self.noise_draft, &self.noise) {
                        crate::noise::Action::Apply => self.noise = self.noise_draft.clone(),
                        crate::noise::Action::DisableAll => {
                            self.noise_draft.disable_all();
                            self.noise.disable_all();
                        }
                        crate::noise::Action::None => {}
                    }
                    ui.add_space(8.0);
                    ui.separator();
                    ui.strong("Notch filters");
                    if let Some(notches) = self.notch_panel.draw(ui, &self.raw, &self.meta, &self.preproc_cfg) {
                        self.preproc_cfg.notch_enabled = !notches.is_empty();
                        self.preproc_cfg.notches = notches;
                        self.heatmap_texture = None;
                        self.pending_cfg_recompute = true;
                    }
                });
            });
        self.noise_open = open;
    }

    /// The samples to draw as `(data, first_sample, n_samp)`: the worker buffer, or
    /// with noise suppression on, a filtered copy of the view plus a margin. The copy
    /// is cached until the view, the buffer or the settings change.
    fn display_source(
        &mut self,
        buf_data: &Option<Arc<Vec<f32>>>,
        buf_display_rows: &Option<Arc<Vec<DisplayRow>>>,
        buf_first: usize,
        buf_n_samp: usize,
        view_first: usize,
        view_n: usize,
    ) -> (Option<Arc<Vec<f32>>>, usize, usize) {
        let unfiltered = (buf_data.clone(), buf_first, buf_n_samp);
        let (Some(data), Some(rows)) = (buf_data, buf_display_rows) else { return unfiltered };
        if !self.noise.active() || buf_n_samp == 0 {
            self.noise_view = None;
            return unfiltered;
        }
        let key = (Arc::as_ptr(data) as usize, buf_first, buf_n_samp, view_first, view_n);
        if let Some(v) = &self.noise_view {
            if v.key == key && v.settings == self.noise {
                return (Some(Arc::clone(&v.data)), v.first, v.n);
            }
        }
        let fs = self.meta.sample_rate;
        let margin = crate::noise::margin_samples(fs);
        let lo = view_first.saturating_sub(margin).max(buf_first);
        let hi = (view_first + view_n + margin).min(buf_first + buf_n_samp);
        if lo >= hi {
            return unfiltered;
        }
        let n = hi - lo;
        let off = lo - buf_first;
        let n_rows = data.len() / buf_n_samp;
        let mut out = vec![0.0f32; n_rows * n];
        {
            use rayon::prelude::*;
            out.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
                row.copy_from_slice(&data[r * buf_n_samp + off..r * buf_n_samp + off + n]);
            });
        }
        crate::noise::apply(&mut out, n, rows, &self.noise, fs);
        let out = Arc::new(out);
        self.noise_view = Some(NoiseView { key, settings: self.noise.clone(), data: Arc::clone(&out), first: lo, n });
        (Some(out), lo, n)
    }

    /// Settings window opened by the "Power Spectrum" toolbar button.
    fn draw_spectrum_window(&mut self, ctx: &egui::Context) {
        if !self.spectrum_open {
            return;
        }
        use crate::spectrum::{SpectrumNormalization, SpectrumScaling, SpectrumSource, SpectrumTimeScope};
        let mut open = self.spectrum_open;
        let mut dirty = false;
        egui::Window::new("Power Spectrum")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Time scope:");
                    let mut scope = self.spectrum_time_scope;
                    egui::ComboBox::from_id_salt("spec_scope_combo")
                        .selected_text(match scope {
                            SpectrumTimeScope::CurrentView => "Current view window",
                            SpectrumTimeScope::WholeRecordingChunks => "Whole recording (chunks)",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut scope, SpectrumTimeScope::CurrentView, "Current view window");
                            ui.selectable_value(
                                &mut scope,
                                SpectrumTimeScope::WholeRecordingChunks,
                                "Whole recording (chunks)",
                            );
                        });
                    if scope != self.spectrum_time_scope {
                        self.spectrum_time_scope = scope;
                        self.spectrum_want_live = false;
                        dirty = true;
                    }
                });

                if self.spectrum_time_scope == SpectrumTimeScope::CurrentView {
                    ui.label(
                        egui::RichText::new(
                            "Recomputes automatically as you scroll, once calculated.",
                        )
                        .small()
                        .color(egui::Color32::GRAY),
                    );
                } else {
                    ui.horizontal(|ui| {
                        ui.label("Chunks to sample:");
                        if ui
                            .add(egui::DragValue::new(&mut self.spectrum_n_chunks).speed(1.0).range(1..=500))
                            .changed()
                        {
                            dirty = true;
                        }
                    });
                    ui.label(
                        egui::RichText::new(
                            "number of 1 s snippets, evenly spaced across the recording, averaged together",
                        )
                        .small()
                        .color(egui::Color32::GRAY),
                    );

                    let total_s = self.meta.n_samples as f64 / self.meta.sample_rate;
                    if ui
                        .checkbox(&mut self.spectrum_time_restrict, "Restrict time window")
                        .changed()
                    {
                        if self.spectrum_time_restrict {
                            // seed with the full recording, so the fields start somewhere
                            // sensible instead of an arbitrary fixed default
                            self.spectrum_time_start_s = 0.0;
                            self.spectrum_time_end_s = total_s;
                        }
                        dirty = true;
                    }
                    if self.spectrum_time_restrict {
                        ui.horizontal(|ui| {
                            ui.label("s:");
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.spectrum_time_start_s)
                                        .speed(1.0)
                                        .range(0.0..=self.spectrum_time_end_s.clamp(0.0, total_s)),
                                )
                                .changed()
                            {
                                dirty = true;
                            }
                            ui.label("–");
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.spectrum_time_end_s)
                                        .speed(1.0)
                                        .range(self.spectrum_time_start_s.clamp(0.0, total_s)..=total_s),
                                )
                                .changed()
                            {
                                dirty = true;
                            }
                        });
                        ui.label(
                            egui::RichText::new("chunks are sourced only from this time window")
                                .small()
                                .color(egui::Color32::GRAY),
                        );
                    }
                }

                ui.separator();
                let forced_raw = self.spectrum_time_scope == SpectrumTimeScope::WholeRecordingChunks;
                ui.horizontal(|ui| {
                    ui.label("Source:");
                    let mut source = if forced_raw { SpectrumSource::Raw } else { self.spectrum_source };
                    ui.add_enabled_ui(!forced_raw, |ui| {
                        egui::ComboBox::from_id_salt("spec_source_combo")
                            .selected_text(match source {
                                SpectrumSource::Raw => "Raw voltage",
                                SpectrumSource::Preprocessed => "Preprocessed",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut source, SpectrumSource::Raw, "Raw voltage");
                                ui.selectable_value(&mut source, SpectrumSource::Preprocessed, "Preprocessed");
                            });
                    });
                    if forced_raw {
                        ui.label(
                            egui::RichText::new("(whole-recording mode always uses raw)")
                                .small()
                                .color(egui::Color32::GRAY),
                        );
                    } else if source != self.spectrum_source {
                        self.spectrum_source = source;
                        dirty = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Scaling:");
                    let mut scaling = self.spectrum_scaling;
                    egui::ComboBox::from_id_salt("spec_scaling_combo")
                        .selected_text(match scaling {
                            SpectrumScaling::Linear => "Linear power",
                            SpectrumScaling::Db => "dB",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut scaling, SpectrumScaling::Linear, "Linear power");
                            ui.selectable_value(&mut scaling, SpectrumScaling::Db, "dB");
                        });
                    if scaling != self.spectrum_scaling {
                        self.spectrum_scaling = scaling;
                        dirty = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Colour range:");
                    let mut norm = self.spectrum_normalization;
                    egui::ComboBox::from_id_salt("spec_norm_combo")
                        .selected_text(match norm {
                            SpectrumNormalization::PerChannel => "Per-channel",
                            SpectrumNormalization::Global => "Global",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut norm, SpectrumNormalization::PerChannel, "Per-channel");
                            ui.selectable_value(&mut norm, SpectrumNormalization::Global, "Global");
                        });
                    if norm != self.spectrum_normalization {
                        self.spectrum_normalization = norm;
                        dirty = true;
                    }
                });

                if ui
                    .checkbox(&mut self.spectrum_freq_restrict, "Restrict frequency range")
                    .changed()
                {
                    if self.spectrum_freq_restrict {
                        // seed with the current result's own band, so the fields start
                        // somewhere sensible instead of an arbitrary fixed default
                        let (lo, hi) = self
                            .spectrum_result
                            .as_ref()
                            .map(|c| crate::render::spectrum_freq_bounds(&c.result.freqs, None))
                            .unwrap_or((1.0, (self.meta.sample_rate / 2.0) as f32));
                        self.spectrum_freq_min_hz = lo as f64;
                        self.spectrum_freq_max_hz = hi as f64;
                    }
                    dirty = true;
                }
                if self.spectrum_freq_restrict {
                    ui.horizontal(|ui| {
                        let nyquist = self.meta.sample_rate / 2.0;
                        ui.label("Hz:");
                        if ui
                            .add(
                                egui::DragValue::new(&mut self.spectrum_freq_min_hz)
                                    .speed(1.0)
                                    .range(0.1..=self.spectrum_freq_max_hz.clamp(0.1, nyquist)),
                            )
                            .changed()
                        {
                            dirty = true;
                        }
                        ui.label("–");
                        if ui
                            .add(
                                egui::DragValue::new(&mut self.spectrum_freq_max_hz)
                                    .speed(1.0)
                                    .range(self.spectrum_freq_min_hz.clamp(0.1, nyquist)..=nyquist),
                            )
                            .changed()
                        {
                            dirty = true;
                        }
                    });
                }

                ui.separator();
                if self.spectrum_computing {
                    let done = self.spectrum_progress.load(Ordering::Relaxed);
                    let total = self.spectrum_run_total;
                    ui.label("Computing power spectrum across the recording…");
                    ui.add(
                        egui::ProgressBar::new(done as f32 / total.max(1) as f32).show_percentage(),
                    );
                    ui.label(format!("{done} / {total} chunks"));
                    if ui.button("Abort").clicked() {
                        self.spectrum_cancel.store(true, Ordering::Relaxed);
                    }
                } else {
                    ui.horizontal(|ui| {
                        if ui.button("Calculate").clicked() {
                            self.spectrum_show_overlay = true;
                            match self.spectrum_time_scope {
                                SpectrumTimeScope::WholeRecordingChunks => self.dispatch_spectrum(ui.ctx()),
                                // computed (immediately: no scrolling to wait for)
                                // by the live recompute in the central panel
                                SpectrumTimeScope::CurrentView => {
                                    self.spectrum_want_live = true;
                                    let now = std::time::Instant::now();
                                    self.spectrum_view_changed_at =
                                        now.checked_sub(SPECTRUM_SETTLE).unwrap_or(now);
                                }
                            }
                        }
                    });

                    if let Some(err) = &self.spectrum_error {
                        ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), err);
                    } else if self.spectrum_result.is_some() {
                        ui.colored_label(egui::Color32::from_rgb(0x55, 0xdd, 0x77), "Computed");
                    }
                }
            });
        self.spectrum_open = open;
        if dirty {
            self.save_prefs();
        }
    }

    // -----------------------------------------------------------------------
    // Screenshot of the plot area
    // -----------------------------------------------------------------------

    /// Which overlays are shown now.
    fn overlay_visibility(&self) -> ShotOverlays {
        ShotOverlays {
            ttl: self.ttl.show_overlay,
            atlas: self.atlas.show_overlay,
            firing_rate: self.show_firing_rate_overlay,
            spectrum: self.spectrum_show_overlay,
            classification: self.show_classification_overlay,
            selection: true,
            scale_bar: true,
        }
    }

    fn set_overlay_visibility(&mut self, v: ShotOverlays) {
        self.ttl.show_overlay = v.ttl;
        self.atlas.show_overlay = v.atlas;
        if v.firing_rate != self.show_firing_rate_overlay {
            self.show_firing_rate_overlay = v.firing_rate;
            // the (skipped while hidden) spike projection has to catch up
            self.proj_view_first = usize::MAX;
            self.heatmap_texture = None;
        }
        self.spectrum_show_overlay = v.spectrum;
        self.show_classification_overlay = v.classification;
    }

    /// The overlays of the current view in the screenshot, or all of them outside a
    /// screenshot.
    fn shot_includes(&self) -> ShotOverlays {
        self.capture.as_ref().map_or(
            ShotOverlays {
                ttl: true,
                atlas: true,
                firing_rate: true,
                spectrum: true,
                classification: true,
                selection: true,
                scale_bar: true,
            },
            |c| c.include,
        )
    }

    /// Default file name: recording, (channel,) start of the view.
    fn default_screenshot_name(&self) -> String {
        let stem = self.bin_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let start = self.view_start_s;
        match self.waveform_channel {
            Some(ch) => format!("{stem}_{}_{start:.3}s.png", self.meta.channel_id(ch - 1)),
            None => format!("{stem}_{start:.3}s.png"),
        }
    }

    /// The "Screenshot" window: which computed overlays to include, and Save.
    fn draw_screenshot_window(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.screenshot.pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.screenshot.pick_rx = None;
                    if let Some(mut path) = picked {
                        if path.extension().is_none() {
                            path.set_extension("png");
                        }
                        let include = self.screenshot.include.unwrap_or_else(|| self.overlay_visibility());
                        let restore = self.overlay_visibility();
                        self.set_overlay_visibility(include);
                        self.capture = Some(Capture { path, include, restore, frames: 0 });
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => self.screenshot.pick_rx = None,
            }
        }
        if !self.screenshot.open {
            self.screenshot.include = None; // taken from the view again on the next open
            return;
        }
        let waveform = self.waveform_channel.is_some();
        let spectrum = self.spectrum_result.as_ref().is_some_and(|c| c.matches(&self.preproc_cfg));
        let mut inc = self.screenshot.include.unwrap_or_else(|| self.overlay_visibility());
        let mut open = self.screenshot.open;
        egui::Window::new("Screenshot")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Saves the plot area as a PNG image, without the windows and the legend.");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Include").strong());
                let mut any = false;
                let mut item = |ui: &mut Ui, show: bool, value: &mut bool, label: &str| {
                    if show {
                        ui.checkbox(value, label);
                        any = true;
                    }
                };
                item(ui, self.ttl.has_data(), &mut inc.ttl, "TTL");
                item(ui, !waveform && self.atlas.has_data(), &mut inc.atlas, "Atlas regions");
                item(ui, !waveform && !self.projection_sums.is_empty(), &mut inc.firing_rate, "Firing rate");
                item(ui, !waveform && spectrum, &mut inc.spectrum, "Power spectrum");
                item(ui, !waveform && self.channel_labels.is_some(), &mut inc.classification, "Channel classification");
                let selected = self.selected_channel_1.is_some() || self.selected_channel_2.is_some();
                item(ui, !waveform && selected, &mut inc.selection, "Selected channels");
                item(ui, !waveform, &mut inc.scale_bar, "Scale bar");
                if !any {
                    ui.label(egui::RichText::new("no overlays calculated").color(egui::Color32::GRAY));
                }
                ui.add_space(4.0);
                if ui.add_enabled(self.screenshot.pick_rx.is_none(), egui::Button::new("Save…")).clicked() {
                    self.screenshot.note = None;
                    self.screenshot.pick_rx = Some(spawn_png_saver(
                        self.bin_path.parent().map(|p| p.to_path_buf()),
                        self.default_screenshot_name(),
                    ));
                }
                if let Some((msg, ok)) = &self.screenshot.note {
                    let c = if *ok { egui::Color32::from_rgb(0x66, 0xdd, 0x66) } else { egui::Color32::from_rgb(0xff, 0x66, 0x66) };
                    ui.colored_label(c, msg);
                }
            });
        self.screenshot.include = Some(inc);
        self.screenshot.open = open;
    }

    /// Once the requested image has arrived: crop it to the plot area, save it, and
    /// show the overlays as before.
    fn poll_capture(&mut self, ctx: &egui::Context) {
        if self.capture.is_none() {
            return;
        }
        let shot = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, user_data, .. }
                    if user_data.data.as_ref().is_some_and(|d| d.as_ref().is::<MainScreenshot>()) =>
                {
                    Some(image.clone())
                }
                _ => None,
            })
        });
        let Some(image) = shot else { return };
        let c = self.capture.take().unwrap();
        self.set_overlay_visibility(c.restore);
        let note = match self.plot_rect {
            Some(rect) => {
                let cropped = image.region(&rect, Some(ctx.pixels_per_point()));
                let [w, h] = cropped.size;
                match crate::psth::save_png(&c.path, w, h, cropped.as_raw()) {
                    Ok(()) => (format!("saved to {}", c.path.display()), true),
                    Err(e) => (format!("screenshot not saved: {e:#}"), false),
                }
            }
            None => ("nothing to save: no plot on screen".to_string(), false),
        };
        self.screenshot.note = Some(note);
    }

    // -----------------------------------------------------------------------
    // Per-channel context menu (plain right-click, no Alt)
    // -----------------------------------------------------------------------

    fn draw_channel_context_menu(&mut self, ctx: &egui::Context) {
        let Some(ch) = self.context_menu_channel else {
            return;
        };
        let pos = self.context_menu_pos.unwrap_or(egui::Pos2::ZERO);
        let ch_id = self.meta.channel_id(ch - 1).to_string();

        let mut still_open = true;
        let area_resp = egui::Area::new(egui::Id::new("channel_context_menu"))
            .fixed_pos(pos)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                apply_menu_item_style(ui.style_mut());
                egui::Frame::menu(ui.style()).show(ui, |ui| {
                    ui.set_min_width(170.0);
                    // justified so each item's hover highlight spans the full row,
                    // exactly like a real menu's entries (see menu_popup upstream)
                    ui.with_layout(egui::Layout::top_down_justified(egui::Align::LEFT), |ui| {
                        if ui.button(format!("View waveform ({ch_id})")).clicked() {
                            self.waveform_channel = Some(ch);
                            still_open = false;
                        }
                        if ui.button(format!("Remove channel {ch_id}")).clicked() {
                            self.remove_channel(ch);
                            still_open = false;
                        }
                    });
                });
            });

        if area_resp.response.clicked_elsewhere() || ctx.input(|i| i.key_pressed(egui::Key::Escape))
        {
            still_open = false;
        }
        if !still_open {
            self.context_menu_channel = None;
            self.context_menu_pos = None;
        }
    }

    // -----------------------------------------------------------------------
    // Single-channel waveform view
    // -----------------------------------------------------------------------

    /// Replaces the heatmap with a line plot of one channel's voltage over the
    /// current time window. Y-axis range is ±`waveform_y_range_uv` (Alt+scroll to
    /// adjust); line color follows the active colormap's accent color.
    fn draw_waveform_view(
        &mut self,
        ui: &mut Ui,
        ch: usize,
        matches_cfg: bool,
        view_first: usize,
        view_n: usize,
        buf_first: usize,
        buf_n_samp: usize,
        buf_data: &Option<Arc<Vec<f32>>>,
        buf_display_rows: &Option<Arc<Vec<DisplayRow>>>,
    ) {
        let rect = ui.available_rect_before_wrap();
        let resp = ui.interact(rect, ui.id().with("waveform_plot"), egui::Sense::click_and_drag());
        let hover = resp
            .hover_pos()
            .or_else(|| self.zoom_drag_start.and_then(|_| ui.input(|i| i.pointer.interact_pos())))
            .filter(|_| self.capture.is_none());
        let painter = ui.painter_at(rect);

        // vertical axis: ±half_range around the centre voltage
        let half_range = self.waveform_y_range_uv.max(WAVEFORM_MIN_RANGE_UV);
        let center_uv = self.waveform_y_center_uv;
        let mid_y = rect.center().y;
        let half_h = rect.height() * 0.5 - 4.0;
        let y_of = |v: f32| mid_y - ((v - center_uv) / half_range).clamp(-1.0, 1.0) * half_h;
        let uv_at = |y: f32| center_uv + (mid_y - y) / half_h * half_range;
        let (view_start_s, view_dur_s) = (self.view_start_s, self.view_dur_s);
        let t_at = |x: f32| view_start_s + ((x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64 * view_dur_s;

        // rectangle zoom: left-drag selects a time and voltage range
        if resp.drag_started_by(egui::PointerButton::Primary) {
            self.zoom_drag_start = ui.input(|i| i.pointer.press_origin());
        }
        let mut drag_sel = None;
        if let Some(start) = self.zoom_drag_start {
            let end = ui.input(|i| i.pointer.interact_pos()).unwrap_or(start);
            let sel = egui::Rect::from_two_pos(start, end).intersect(rect);
            if resp.drag_stopped() || !ui.input(|i| i.pointer.primary_down()) {
                self.zoom_drag_start = None;
                if sel.width() >= ZOOM_MIN_DRAG_PX && sel.height() >= ZOOM_MIN_DRAG_PX {
                    let (t0, t1) = (t_at(sel.left()), t_at(sel.right()));
                    let (v_lo, v_hi) = (uv_at(sel.bottom()), uv_at(sel.top()));
                    // nested zooms keep the view from before the first one
                    if self.waveform_zoom.is_none() {
                        self.waveform_zoom = Some(WaveformZoom {
                            prev_start_s: self.view_start_s,
                            prev_dur_s: self.view_dur_s,
                            prev_y_range_uv: self.waveform_y_range_uv,
                        });
                    }
                    let total_s = self.meta.n_samples as f64 / self.meta.sample_rate;
                    self.view_dur_s = (t1 - t0).max(0.01);
                    self.view_start_s = t0.clamp(0.0, (total_s - self.view_dur_s).max(0.0));
                    self.window_dur_str = format!("{:.3}", self.view_dur_s);
                    self.waveform_y_center_uv = (v_lo + v_hi) / 2.0;
                    self.waveform_y_range_uv = ((v_hi - v_lo) / 2.0).max(WAVEFORM_MIN_RANGE_UV);
                    ui.ctx().request_repaint();
                }
            } else {
                drag_sel = Some(sel);
            }
        }
        painter.rect_filled(
            rect,
            0.0,
            egui::Color32::from_rgb(
                crate::render::C_ZERO[0],
                crate::render::C_ZERO[1],
                crate::render::C_ZERO[2],
            ),
        );
        self.ttl.draw_overlay(
            &painter,
            rect,
            self.view_start_s,
            self.view_dur_s,
            &self.colormap_choice,
        );

        let row_data: Option<(Arc<Vec<f32>>, usize)> = if matches_cfg {
            buf_display_rows
                .as_ref()
                .zip(buf_data.as_ref())
                .and_then(|(rows, data)| {
                    rows.iter()
                        .find_map(|r| match r {
                            DisplayRow::Data {
                                data_idx, first_ch, ..
                            } if *first_ch + 1 == ch => Some(*data_idx),
                            _ => None,
                        })
                        .map(|data_idx| (Arc::clone(data), data_idx))
                })
        } else {
            None
        };

        match row_data {
            Some((data, data_idx)) => {
                let row_base = data_idx * buf_n_samp;
                let ov_start = view_first.max(buf_first);
                let ov_end = (view_first + view_n).min(buf_first + buf_n_samp);

                if ov_start < ov_end && row_base + buf_n_samp <= data.len() {
                    let lo = ov_start - buf_first;
                    let hi = ov_end - buf_first;
                    let samples = &data[row_base + lo..row_base + hi];
                    let n = samples.len();

                    if n >= 2 {
                        let [ar, ag, ab] = self.colormap_choice.spec().accent;
                        let line_color = egui::Color32::from_rgb(ar, ag, ab);

                        // base line width — tune WAVEFORM_LINE_BASE_WIDTH below; it
                        // scales gently with the panel height so it isn't too thin on
                        // a large window or too thick on a small one
                        const WAVEFORM_LINE_BASE_WIDTH: f32 = 1.0;
                        let line_width =
                            WAVEFORM_LINE_BASE_WIDTH * (rect.height() / 400.0).clamp(0.5, 2.5);

                        let pts: Vec<egui::Pos2> = samples
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| {
                                let x = rect.left() + (i as f32 / (n - 1) as f32) * rect.width();
                                egui::pos2(x, y_of(v))
                            })
                            .collect();

                        // 0 µV line, while it is in the shown range
                        if (center_uv - 0.0).abs() <= half_range {
                            let y0 = y_of(0.0);
                            painter.line_segment(
                                [egui::pos2(rect.left(), y0), egui::pos2(rect.right(), y0)],
                                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(80)),
                            );
                        }
                        // hovered sample: marked on the trace, values in the readout
                        let hovered = hover.map(|p| {
                            let frac = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                            let i = (frac * (n - 1) as f32).round() as usize;
                            (i, pts[i])
                        });
                        painter.add(egui::Shape::line(
                            pts,
                            egui::Stroke::new(line_width, line_color),
                        ));

                        let hint = if self.capture.is_some() { "" } else { " (Alt+scroll to rescale, drag to zoom)" };
                        let range = if center_uv == 0.0 {
                            format!("±{half_range:.0} µV")
                        } else {
                            format!("{:.1} to {:.1} µV", center_uv - half_range, center_uv + half_range)
                        };
                        painter.text(
                            egui::pos2(rect.left() + 6.0, rect.top() + 4.0),
                            egui::Align2::LEFT_TOP,
                            format!("{}  ·  {range}{hint}", self.meta.channel_id(ch - 1)),
                            egui::FontId::proportional(13.0),
                            egui::Color32::from_gray(200),
                        );

                        let id = self.meta.channel_id(ch - 1);
                        match (drag_sel, hovered) {
                            // while dragging a zoom rectangle: the ranges it covers
                            (Some(sel), _) => draw_readout(
                                &painter,
                                rect,
                                format!(
                                    "{id}  t = {:.4}–{:.4} s  {:.1} to {:.1} µV",
                                    t_at(sel.left()),
                                    t_at(sel.right()),
                                    uv_at(sel.bottom()),
                                    uv_at(sel.top())
                                ),
                            ),
                            (None, Some((i, pt))) => {
                                painter.circle_filled(pt, line_width + 2.0, line_color);
                                let t = (ov_start + i) as f64 / self.meta.sample_rate;
                                draw_readout(&painter, rect, format!("{id}  t = {t:.4} s  {:.1} µV", samples[i]));
                            }
                            (None, None) => {}
                        }
                    }
                } else {
                    Self::draw_centered_message(&painter, rect, "⏳ Loading…");
                }
            }
            None => {
                let msg = if matches_cfg {
                    format!("Channel {} isn't currently displayed (removed, or outside the loaded buffer).", self.meta.channel_id(ch - 1))
                } else {
                    "⏳ Loading…".to_string()
                };
                Self::draw_centered_message(&painter, rect, &msg);
            }
        }

        if self.capture.is_some() {
            return;
        }
        if let Some(sel) = drag_sel {
            painter.rect_filled(sel, 0.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 3));
            painter.rect_stroke(sel, 0.0, egui::Stroke::new(1.0_f32, egui::Color32::WHITE), egui::StrokeKind::Inside);
        }
        if self.waveform_zoom.is_some() {
            draw_zoom_notice(&painter, rect.left_top() + egui::vec2(10.0, 28.0), &self.colormap_choice);
        }
        let close_rect = egui::Rect::from_min_size(
            rect.right_top() + egui::vec2(-34.0, 6.0),
            egui::vec2(28.0, 24.0),
        );
        if ui
            .put(close_rect, egui::Button::new("✖"))
            .on_hover_text("Back to heatmap (Esc)")
            .clicked()
        {
            self.leave_waveform_zoom();
            self.waveform_channel = None;
        }
    }

    /// Return to the waveform view from before the zoom; false if not zoomed.
    fn leave_waveform_zoom(&mut self) -> bool {
        let Some(z) = self.waveform_zoom.take() else { return false };
        self.view_start_s = z.prev_start_s;
        self.view_dur_s = z.prev_dur_s;
        self.window_dur_str = format!("{:.3}", self.view_dur_s);
        self.waveform_y_range_uv = z.prev_y_range_uv;
        self.waveform_y_center_uv = 0.0;
        true
    }

    fn draw_centered_message(painter: &egui::Painter, rect: egui::Rect, msg: &str) {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            msg,
            egui::FontId::proportional(16.0),
            egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200),
        );
    }

    // -----------------------------------------------------------------------
    // PSTH
    // -----------------------------------------------------------------------

    /// Poll the PSTH picker/compute/export channels and dispatch a computation only
    /// when the user has pressed "Apply/Compute".
    fn poll_and_maybe_dispatch_psth(&mut self, ctx: &egui::Context) {
        // stimulus-file picker
        if let Some(rx) = &self.psth.pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.psth.pick_rx = None;
                    if let Some(path) = picked {
                        self.psth.stim_path = Some(path);
                        self.psth.open = true;
                        // the old plots belong to the previous file; compute on Apply/Compute
                        self.psth.result = None;
                        self.psth.error = None;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => self.psth.pick_rx = None,
            }
        }

        // compute result
        if let Some(rx) = &self.psth.compute_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.psth.compute_rx = None;
                    self.psth.computing = false;
                    match res {
                        Ok(r) => {
                            self.psth.trace_vmax_ref = self.psth.vmax(&r);
                            self.psth.n_used = r.n_used;
                            self.psth.n_skipped = r.n_skipped;
                            self.psth.result = Some(Arc::new(r));
                            self.psth.error = None;
                            self.psth.tex_dirty = true;
                        }
                        Err(e) if e == "cancelled" => {}
                        Err(e) => {
                            self.psth.error = Some(e);
                            self.psth.result = None;
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(80));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.psth.compute_rx = None;
                    self.psth.computing = false;
                }
            }
        }

        // dispatch only when Apply was pressed
        if self.psth.apply_requested && self.psth.stim_path.is_some() {
            self.psth.apply_requested = false;
            self.dispatch_psth_compute(ctx);
        }

        self.poll_psth_export(ctx);
    }

    fn dispatch_psth_compute(&mut self, ctx: &egui::Context) {
        let stim_path = match &self.psth.stim_path {
            Some(p) => p.clone(),
            None => return,
        };
        let layout = match StimLayout::parse(&self.stim_layout_text) {
            Ok(l) => l,
            Err(e) => {
                self.psth.error = Some(e.to_string());
                return;
            }
        };
        self.stim_file = Some(stim_path.clone());

        // cancel any in-flight compute and install a fresh cancel flag
        self.psth.cancel.store(true, Ordering::Relaxed);
        let cancel = Arc::new(AtomicBool::new(false));
        self.psth.cancel = Arc::clone(&cancel);
        let progress = Arc::new(AtomicUsize::new(0));
        self.psth.progress = Arc::clone(&progress);
        let progress_total = Arc::new(AtomicUsize::new(0));
        self.psth.progress_total = Arc::clone(&progress_total);

        let (tx, rx) = mpsc::channel();
        self.psth.compute_rx = Some(rx);
        self.psth.computing = true;
        self.psth.error = None;

        let raw = Arc::clone(&self.raw);
        let meta = Arc::clone(&self.meta);
        let cfg = self.preproc_cfg.clone();
        let params = PsthParams {
            start_ms: self.psth.start_ms,
            end_ms: self.psth.end_ms,
        };
        let (t_start, t_end) = (self.psth.stim_t_start, self.psth.stim_t_end);
        let ctx = ctx.clone();

        std::thread::spawn(move || {
            let res = (|| -> Result<PsthResult, String> {
                let all = load_stim_times(&stim_path, &layout).map_err(|e| e.to_string())?;
                let times: Vec<f64> = all
                    .into_iter()
                    .filter(|&t| t >= t_start && t <= t_end)
                    .collect();
                if times.is_empty() {
                    return Err(format!(
                        "no stimuli fall within the selected time range {:.3}–{:.3} s.",
                        t_start, t_end
                    ));
                }
                compute_psth(
                    &raw,
                    &meta,
                    &cfg,
                    &times,
                    &params,
                    &cancel,
                    &progress,
                    &progress_total,
                )
                .map_err(|e| e.to_string())
            })();
            let _ = tx.send(res);
            ctx.request_repaint();
        });
    }

    /// Poll the PNG-export save dialog and, once the requested screenshot arrives,
    /// crop it to the figure area and write the file.
    fn poll_psth_export(&mut self, ctx: &egui::Context) {
        // save-file dialog result
        if let Some(rx) = &self.psth.export_pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.psth.export_pick_rx = None;
                    if let Some(mut path) = picked {
                        if path.extension().is_none() {
                            path.set_extension("png");
                        }
                        self.psth.export_path = Some(path);
                        self.psth.export_pending = true;
                        // request a full-viewport screenshot; we crop it when it arrives
                        ctx.send_viewport_cmd(
                            egui::ViewportCommand::Screenshot(Default::default()),
                        );
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => self.psth.export_pick_rx = None,
            }
        }

        // screenshot reply
        if self.psth.export_pending {
            let shot = ctx.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let (Some(image), Some(rect), Some(path)) =
                (shot, self.psth.figure_rect, self.psth.export_path.clone())
            {
                self.psth.export_pending = false;
                self.psth.export_path = None;
                let ppp = ctx.pixels_per_point();
                let cropped = image.region(&rect, Some(ppp));
                let [w, h] = cropped.size;
                match crate::psth::save_png(&path, w, h, cropped.as_raw()) {
                    Ok(()) => {}
                    Err(e) => self.psth.error = Some(format!("PNG export failed: {e}")),
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // UI panels
    // -----------------------------------------------------------------------

    fn draw_toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.scope(|ui| {
                zero_item_gap(ui);
                ui.menu_button("File", |ui| {
                    if ui.button("Open").clicked() {
                        self.file_dialog_request = true;
                        ui.close_menu();
                    }
                    ui.menu_button("Recent files", |ui| {
                        if let Some(path) = crate::draw_recent_files_menu(ui, &self.recent_files) {
                            self.open_recent_request = Some(path);
                            ui.close_menu();
                        }
                    });
                });
                ui.add(egui::Separator::default().vertical().spacing(1.0));
            });
            ui.label(format!(
                "{}",
                self.bin_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            ))
            .on_hover_text(format!(
                "{}\nprobe type {}, {} channels, {:.0} Hz",
                self.bin_path.display(),
                self.meta.im_dat_prb_type,
                self.meta.n_ap_chans,
                self.meta.sample_rate
            ));
            ui.separator();
            // window duration text field — stored separately to avoid overwrite each frame
            ui.label("Window:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.window_dur_str)
                    .desired_width(55.0),
            );
            ui.label("s");
            if resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                if let Ok(v) = self.window_dur_str.trim().parse::<f64>() {
                    let new_dur = v.clamp(0.01, 10.0);
                    if (new_dur - self.view_dur_s).abs() > 1e-6 {
                        self.view_dur_s = new_dur;
                        self.heatmap_texture = None;
                    }
                }
                // re-sync display string to actual value
                self.window_dur_str = format!("{:.3}", self.view_dur_s);
            }

            ui.separator();
            ui.label("Scroll:");
            ui.radio_value(&mut self.scroll_speed_fine, true, "Fine");
            ui.radio_value(&mut self.scroll_speed_fine, false, "Coarse");

            ui.separator();

            // Color scale controls
            ui.label("Color scale:");

            if ui
                .radio_value(&mut self.color_mode, ColorMode::Percentile, "%ile")
                .changed()
                || ui
                    .radio_value(&mut self.color_mode, ColorMode::Voltage, "±µV")
                    .changed()
            {
                self.heatmap_texture = None;
            }

            if self.color_mode == ColorMode::Percentile {
                if ui
                    .add(
                        egui::Slider::new(&mut self.color_pct, 95.0..=100.0)
                            .step_by(0.1)
                            .text("%"),
                    )
                    .changed()
                {
                    self.color_pct_str = format!("{:.2}", self.color_pct);
                    self.heatmap_texture = None;
                }
            } else {
                if ui
                    .add(
                        egui::Slider::new(&mut self.color_uv, 10.0..=300.0)
                            .integer()
                            .text("µV"),
                    )
                    .changed()
                {
                    self.color_uv_str = format!("{:.0}", self.color_uv);
                    self.heatmap_texture = None;
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Preferences").clicked() {
                    self.show_preferences = !self.show_preferences;
                }
                if ui.button("Screenshot").on_hover_text("Save the plot area as an image").clicked() {
                    self.screenshot.open = !self.screenshot.open;
                }
            });
        });
    }

    fn draw_preproc_panel(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Preprocessing:");

            let mut dc = self.preproc_cfg.dc_removal;
            if ui.checkbox(&mut dc, "DC").changed() {
                self.preproc_cfg.dc_removal = dc;
                self.pending_cfg_recompute = true;
            }

            ui.separator();

            let mut phase = self.preproc_cfg.phase_shift;
            if ui.checkbox(&mut phase, "Phase Shift").changed() {
                self.preproc_cfg.phase_shift = phase;
                self.pending_cfg_recompute = true;
            }

            ui.separator();

            // only once notches exist (Noise Suppression window); switches them all
            if !self.preproc_cfg.notches.is_empty() {
                let mut on = self.preproc_cfg.notch_enabled;
                let list: Vec<String> = self.preproc_cfg.notches.iter().map(|n| format!("{:.1} Hz (width {:.1} Hz)", n.freq_hz, n.bw_hz)).collect();
                if ui
                    .checkbox(&mut on, format!("Notch ({})", self.preproc_cfg.notches.len()))
                    .on_hover_text(format!("Notch filters for this file, edited in the Noise Suppression window:\n{}", list.join("\n")))
                    .changed()
                {
                    self.preproc_cfg.notch_enabled = on;
                    self.pending_cfg_recompute = true;
                }
                ui.separator();
            }

            let hp_enabled = self.preproc_cfg.spatial_filter != SpatialFilter::Destripe;
            let mut hp = self.preproc_cfg.highpass;
            if ui
                .add_enabled(hp_enabled, egui::Checkbox::new(&mut hp, "300 Hz HP"))
                .on_disabled_hover_text("Included in destripe")
                .changed()
            {
                self.preproc_cfg.highpass = hp;
                self.pending_cfg_recompute = true;
            }

            ui.separator();
            ui.label("Spatial:");

            let mut spatial = self.preproc_cfg.spatial_filter;
            let changed = ui
                .radio_value(&mut spatial, SpatialFilter::Off, "Off")
                .changed()
                || ui
                    .radio_value(&mut spatial, SpatialFilter::GlobalCmr, "Global CMR")
                    .changed()
                || ui
                    .radio_value(&mut spatial, SpatialFilter::LocalCmr, "Local CMR")
                    .changed()
                || ui
                    .radio_value(&mut spatial, SpatialFilter::Destripe, "Destripe")
                    .changed();

            if changed {
                if spatial == SpatialFilter::Destripe {
                    self.preproc_cfg.highpass = true;
                }
                self.preproc_cfg.spatial_filter = spatial;
                {
                    let mut f = self.preproc_filters.lock().unwrap();
                    *f = Filters::new(&self.preproc_cfg);
                }
                self.heatmap_texture = None;
                self.pending_cfg_recompute = true;
            }

            ui.separator();

            // depth averaging checkbox
            let mut avg = self.preproc_cfg.avg_depths;
            if ui.checkbox(&mut avg, "Avg adjacent chans").changed() {
                self.preproc_cfg.avg_depths = avg;
                self.heatmap_texture = None;
                self.pending_cfg_recompute = true;
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // right-to-left: the field goes first so the label ends up left of it
                let resp =
                    ui.add(egui::TextEdit::singleline(&mut self.jump_str).desired_width(70.0));
                ui.label("Jump to (s):");
                if resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(t) = self.jump_str.trim().parse::<f64>() {
                        let max_t =
                            self.meta.n_samples as f64 / self.meta.sample_rate - self.view_dur_s;
                        self.view_start_s = t.clamp(0.0, max_t.max(0.0));
                    }
                    self.jump_str = format!("{:.3}", self.view_start_s);
                }
                // keep jump field synced when not being edited
                if !resp.has_focus() {
                    self.jump_str = format!("{:.3}", self.view_start_s);
                }
            });
        });
    }

    fn draw_channel_controls(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.scope(|ui| {
                zero_item_gap(ui);
                if ui.button("PSTH").clicked() {
                    self.psth.open = true;
                    if self.psth.pick_rx.is_none() {
                        self.psth.pick_rx = Some(spawn_stim_picker(
                            self.bin_path.parent().map(|p| p.to_path_buf()),
                        ));
                    }
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                if ui.button("TTL").clicked() {
                    self.ttl.open = !self.ttl.open;
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                if ui.button("Atlas Registration").clicked() {
                    self.atlas.open = !self.atlas.open;
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                if ui
                    .add_enabled(
                        !self.classifying,
                        egui::Button::new("Channel Classification"),
                    )
                    .clicked()
                {
                    self.dispatch_classify(ui.ctx());
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                if ui.button("Power Spectrum").clicked() {
                    self.spectrum_open = !self.spectrum_open;
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                // highlighted while any suppression step is on
                if ui
                    .add(egui::Button::new("Noise Suppression").selected(self.noise.active()))
                    .clicked()
                {
                    self.noise_open = !self.noise_open;
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                if ui.button("Remove channels…").clicked() {
                    self.show_remove_channels = !self.show_remove_channels;
                }
            });

            let display_rows_arc = {
                let (lock, _) = &*self.worker_state;
                lock.lock()
                    .unwrap()
                    .buffer
                    .as_ref()
                    .map(|b| Arc::clone(&b.display_rows))
            };

            let mut ch1_visible = false;
            let mut ch2_visible = false;

            if let Some(rows) = &display_rows_arc {
                for r in rows.iter() {
                    if let DisplayRow::Data { first_ch, .. } = r {
                        let ch = *first_ch + 1;
                        if Some(ch) == self.selected_channel_1 {
                            ch1_visible = true;
                        }
                        if Some(ch) == self.selected_channel_2 {
                            ch2_visible = true;
                        }
                    }
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let (Some(ch1), Some(ch2)) = (self.selected_channel_1, self.selected_channel_2) {
                    if ch1 > 0
                        && ch1 <= self.meta.channel_geom.len()
                        && ch2 > 0
                        && ch2 <= self.meta.channel_geom.len()
                    {
                        let y1 = self.meta.channel_geom[ch1 - 1].y_um;
                        let y2 = self.meta.channel_geom[ch2 - 1].y_um;
                        let dist = (y1 - y2).abs();
                        ui.label(
                            egui::RichText::new(format!("Δ = {:.1} µm", dist))
                                .strong()
                                .color(egui::Color32::WHITE),
                        );
                        ui.separator();
                    }
                }

                if ch2_visible {
                    if let Some(ch2) = self.selected_channel_2 {
                        if ui.button("✖").clicked() {
                            self.selected_channel_2 = None;
                        }
                        ui.label(
                            egui::RichText::new(format!(
                                "Selected Channel 2: {}",
                                self.meta.channel_id(ch2 - 1)
                            ))
                            .color(egui::Color32::from_rgb(0xff, 0xb6, 0x17)),
                        );
                    }
                }

                if ch1_visible {
                    if let Some(ch1) = self.selected_channel_1 {
                        if ui.button("✖").clicked() {
                            self.selected_channel_1 = None;
                        }
                        let id = self.meta.channel_id(ch1 - 1);
                        let text = if self.selected_channel_2.is_some() {
                            format!("Selected Channel 1: {id}")
                        } else {
                            format!("Selected Channel: {id}")
                        };
                        ui.label(egui::RichText::new(text).color(egui::Color32::WHITE));
                    }
                }
            });
        });
    }

    fn draw_status_bar(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let status = {
                let (lock, _) = &*self.worker_state;
                lock.lock().unwrap().status.clone()
            };
            if status == WorkerStatus::Computing {
                ui.spinner();
                ui.label("Computing…");
            }

            if self.is_compressed {
                ui.separator();
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "Reading from compressed .cbin is slower",
                );
            }

            if let Some(e) = &self.settings_error {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), format!("⚠ {e}"))
                    .on_hover_text(e);
            }

            // values the metadata did not provide (gain, geometry, ...)
            for w in &self.meta.warnings {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(0xff, 0xcc, 0x55), format!("⚠ {w}"))
                    .on_hover_text(w);
            }
        });
    }

    fn draw_nav_bar(&mut self, ui: &mut Ui) {
        let total_s = self.meta.n_samples as f64 / self.meta.sample_rate;

        let (response, painter) =
            ui.allocate_painter(Vec2::new(ui.available_width(), 32.0), egui::Sense::click());
        let rect = response.rect;
        let w = rect.width();

        painter.rect_filled(rect, 2.0, egui::Color32::BLACK);

        let [ar, ag, ab] = self.colormap_choice.spec().accent;

        // preprocessed-buffer extent, drawn first so it sits beneath the view marker
        let buf_extent = {
            let (lock, _) = &*self.worker_state;
            lock.lock()
                .unwrap()
                .buffer
                .as_ref()
                .map(|b| (b.first_sample, b.n_samp))
        };
        if let Some((first, n_samp)) = buf_extent {
            let buf_frac = (first as f64 / self.meta.sample_rate / total_s) as f32;
            let buf_w_frac = (n_samp as f64 / self.meta.sample_rate / total_s) as f32;
            let buf_rect = egui::Rect::from_min_size(
                egui::pos2(rect.min.x + w * buf_frac, rect.min.y),
                Vec2::new((w * buf_w_frac).max(2.0), rect.height()),
            );
            painter.rect_filled(
                buf_rect,
                1.0,
                egui::Color32::from_rgba_unmultiplied(
                    ar,
                    ag,
                    ab,
                    crate::render::BUFFER_EXTENT_ALPHA,
                ),
            );
        }

        // view marker
        let view_frac = (self.view_start_s / total_s) as f32;
        let view_w_frac = (self.view_dur_s / total_s) as f32;
        let view_rect = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + w * view_frac, rect.min.y),
            Vec2::new((w * view_w_frac).max(2.0), rect.height()),
        );
        painter.rect_filled(
            view_rect,
            1.0,
            egui::Color32::from_rgba_unmultiplied(ar, ag, ab, crate::render::VIEW_MARKER_ALPHA),
        );

        // time labels
        let n_labels = 8;
        for i in 0..=n_labels {
            let frac = i as f32 / n_labels as f32;
            let t = frac as f64 * total_s;
            let x = rect.min.x + w * frac;
            painter.line_segment(
                [egui::pos2(x, rect.max.y - 6.0), egui::pos2(x, rect.max.y)],
                egui::Stroke::new(1.0_f32, egui::Color32::GRAY),
            );
            painter.text(
                egui::pos2(x, rect.max.y - 8.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{:.0}s", t),
                egui::FontId::proportional(9.0),
                egui::Color32::GRAY,
            );
        }

        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let frac = ((pos.x - rect.min.x) / w).clamp(0.0, 1.0) as f64;
                let max_t = total_s - self.view_dur_s;
                self.view_start_s = (frac * total_s).clamp(0.0, max_t.max(0.0));
            }
        }
    }
}

impl NPXplorerApp {
    fn draw_psth_window(&mut self, ctx: &egui::Context) {
        if !self.psth.open {
            return;
        }
        let c_zero = egui::Color32::from_rgb(
            crate::render::C_ZERO[0],
            crate::render::C_ZERO[1],
            crate::render::C_ZERO[2],
        );
        let [ar, ag, ab] = self.colormap_choice.spec().accent;
        let accent = egui::Color32::from_rgb(ar, ag, ab);
        let accent_50 = egui::Color32::from_rgba_unmultiplied(ar, ag, ab, 128);
        let cmap = self.colormap_choice.clone();
        let total_s = self.psth.total_s;

        // size the window to 2/3 of the main window and center it on first open
        let screen = ctx.screen_rect();
        let win_size = screen.size() * (2.0 / 3.0);
        let win_pos = screen.center() - (win_size * 0.5);

        let mut open = self.psth.open;
        let result = self.psth.result.clone();

        let win_resp = egui::Window::new(
            egui::RichText::new("Peri-Stimulus Time Histogram").color(egui::Color32::WHITE),
        )
        .open(&mut open)
        .default_size(win_size)
        .default_pos(win_pos)
        .frame(
            egui::Frame::new()
                .fill(c_zero)
                .inner_margin(8.0)
                .stroke(egui::Stroke::new(2.0_f32, accent_50)),
        )
        .show(ctx, |ui| {
            // force every widget in this window onto the app background
            ui.visuals_mut().panel_fill = c_zero;
            ui.visuals_mut().window_fill = c_zero;

            ui.horizontal(|ui| {
                if ui.button("Change file…").clicked() && self.psth.pick_rx.is_none() {
                    self.psth.pick_rx = Some(spawn_stim_picker(
                        self.bin_path.parent().map(|p| p.to_path_buf()),
                    ));
                }
                let name = self
                    .psth
                    .stim_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "(no file)".into());
                ui.label(name);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let can_export = self.psth.result.is_some()
                        && self.psth.export_pick_rx.is_none()
                        && !self.psth.export_pending;
                    if ui
                        .add_enabled(can_export, egui::Button::new("Export PNG…"))
                        .clicked()
                    {
                        self.psth.export_pick_rx = Some(spawn_png_saver(
                            self.bin_path.parent().map(|p| p.to_path_buf()),
                            self.default_psth_png_name(),
                        ));
                    }
                });
            });

            crate::ttl::format_editor(ui, &mut self.stim_layout_text);

            ui.horizontal(|ui| {
                // stimulus time-range selector (seconds)
                ui.label("Stim time (s):");
                let r1 = ui.add(
                    egui::TextEdit::singleline(&mut self.psth.stim_t_start_str)
                        .desired_width(70.0)
                        .hint_text("start"),
                );
                if r1.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(v) = self.psth.stim_t_start_str.trim().parse::<f64>() {
                        self.psth.stim_t_start = v.clamp(0.0, total_s);
                    }
                    self.psth.stim_t_start_str = format!("{:.3}", self.psth.stim_t_start);
                }
                ui.label("to");
                let r2 = ui.add(
                    egui::TextEdit::singleline(&mut self.psth.stim_t_end_str)
                        .desired_width(70.0)
                        .hint_text("end"),
                );
                if r2.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(v) = self.psth.stim_t_end_str.trim().parse::<f64>() {
                        self.psth.stim_t_end = v.clamp(0.0, total_s);
                    }
                    self.psth.stim_t_end_str = format!("{:.3}", self.psth.stim_t_end);
                }
            });

            ui.horizontal(|ui| {
                ui.label("Window (ms):");
                let r1 = ui.add(
                    egui::TextEdit::singleline(&mut self.psth.start_ms_str)
                        .desired_width(55.0)
                        .hint_text("start"),
                );
                if r1.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(v) = self.psth.start_ms_str.trim().parse::<f64>() {
                        self.psth.start_ms = v;
                    }
                    self.psth.start_ms_str = format!("{}", self.psth.start_ms);
                }
                ui.label("to");
                let r2 = ui.add(
                    egui::TextEdit::singleline(&mut self.psth.end_ms_str)
                        .desired_width(55.0)
                        .hint_text("end"),
                );
                if r2.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(v) = self.psth.end_ms_str.trim().parse::<f64>() {
                        self.psth.end_ms = v;
                    }
                    self.psth.end_ms_str = format!("{}", self.psth.end_ms);
                }

                ui.separator();

                // Apply: commit the staged settings and recompute
                let apply = ui.add_enabled(
                    self.psth.stim_path.is_some() && !self.psth.computing,
                    egui::Button::new(egui::RichText::new("Apply/Compute").color(c_zero))
                        .fill(accent_50),
                );
                if apply.clicked() {
                    self.psth.apply_requested = true;
                }
            });

            ui.horizontal(|ui| {
                ui.label("Color scale:");
                let mut changed = false;
                changed |= ui
                    .radio_value(&mut self.psth.color_mode, ColorMode::Percentile, "%ile")
                    .changed();
                changed |= ui
                    .radio_value(&mut self.psth.color_mode, ColorMode::Voltage, "±µV")
                    .changed();
                if self.psth.color_mode == ColorMode::Percentile {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.psth.color_pct, 95.0..=100.0)
                                .step_by(0.1)
                                .text("%"),
                        )
                        .changed();
                } else {
                    changed |= ui
                        .add(egui::Slider::new(&mut self.psth.color_uv, 1.0..=200.0).text("µV"))
                        .changed();
                }
                if changed {
                    self.psth.tex_dirty = true;
                }
            });

            if self.psth.computing {
                let done = self.psth.progress.load(Ordering::Relaxed);
                let total = self.psth.progress_total.load(Ordering::Relaxed);
                ui.horizontal(|ui| {
                    if total == 0 {
                        // still reading the stimulus file / setting up
                        ui.spinner();
                        ui.label("Preparing…");
                    } else {
                        ui.add(
                            egui::ProgressBar::new(done as f32 / total as f32)
                                .desired_width(240.0)
                                .show_percentage(),
                        );
                        ui.label(format!("{done} / {total} stimuli"));
                    }
                    if ui.button("Abort").clicked() {
                        self.psth.cancel.store(true, Ordering::Relaxed);
                    }
                });
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(100));
            } else if let Some(err) = &self.psth.error {
                ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), err);
            } else if self.psth.result.is_some() {
                ui.horizontal(|ui| {
                    let base = if self.psth.n_skipped > 0 {
                        format!(
                            "{} stimuli averaged ({} skipped near edges)",
                            self.psth.n_used, self.psth.n_skipped
                        )
                    } else {
                        format!("{} stimuli averaged", self.psth.n_used)
                    };
                    ui.label(base);
                    ui.label("·  left-click / right-click the heatmap to plot a channel:");
                    if let Some(c) = self.psth.sel_ch1 {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 255, 255),
                            self.meta.channel_id(c - 1),
                        );
                    }
                    if let Some(c) = self.psth.sel_ch2 {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 182, 23),
                            self.meta.channel_id(c - 1),
                        );
                    }
                    if (self.psth.sel_ch1.is_some() || self.psth.sel_ch2.is_some())
                        && ui.button("Deselect").clicked()
                    {
                        self.psth.sel_ch1 = None;
                        self.psth.sel_ch2 = None;
                    }
                });
            }

            ui.separator();

            if let Some(result) = &result {
                self.draw_psth_plots(ui, result, &cmap, accent, c_zero);
            } else if !self.psth.computing && self.psth.error.is_none() {
                if self.psth.stim_path.is_some() {
                    ui.label("Press Apply/Compute to compute the PSTH.");
                } else {
                    ui.label("Pick a stimulus-times file to compute the PSTH.");
                }
            }
        });

        // Alt+scroll over the window adjusts the color scale, like on the main heatmap
        let hovered = win_resp.is_some_and(|r| r.response.contains_pointer());
        if hovered && ctx.input(|i| i.modifiers.alt) {
            let ticks: f32 = ctx.input(|i| {
                i.events
                    .iter()
                    .filter_map(|e| match e {
                        egui::Event::MouseWheel { delta, .. } => Some(delta.y.signum()),
                        _ => None,
                    })
                    .sum()
            });
            if ticks != 0.0 {
                if self.psth.color_mode == ColorMode::Percentile {
                    self.psth.color_pct = (self.psth.color_pct - ticks * 0.1).clamp(95.0, 100.0);
                } else {
                    // multiplicative, so the step suits both 2 µV and 200 µV
                    self.psth.color_uv =
                        (self.psth.color_uv * 1.1f32.powf(-ticks)).clamp(1.0, 200.0);
                }
                self.psth.tex_dirty = true;
            }
        }

        self.psth.open = open;
    }

    /// Default filename for a PSTH PNG: `<recording>_<stim file>_psth.png`.
    fn default_psth_png_name(&self) -> String {
        let rec = self
            .bin_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let stim = self
            .psth
            .stim_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        format!("{rec}_{stim}_psth.png")
    }

    fn draw_psth_plots(
        &mut self,
        ui: &mut Ui,
        result: &Arc<PsthResult>,
        cmap: &ColorMapChoice,
        accent: egui::Color32,
        c_zero: egui::Color32,
    ) {
        // colors for the two selectable channel traces (match the main window)
        let ch1_color = egui::Color32::from_rgb(255, 255, 255);
        let ch2_color = egui::Color32::from_rgb(255, 182, 23);

        let avail = ui.available_size();
        if avail.x < 40.0 || avail.y < 90.0 {
            return;
        }
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        self.psth.figure_rect = Some(rect);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, c_zero);

        let gutter = 46.0;
        let x_axis_h = 22.0;
        let gap = 6.0;
        let plot_left = rect.left() + gutter;
        let plot_right = rect.right() - 6.0;

        // two stacked line plots (selected-channel, then mean) above the heatmap
        let line_h = ((rect.height() - x_axis_h) * 0.20).clamp(40.0, 130.0);
        let sel_rect = egui::Rect::from_min_max(
            egui::pos2(plot_left, rect.top()),
            egui::pos2(plot_right, rect.top() + line_h),
        );
        let avg_rect = egui::Rect::from_min_max(
            egui::pos2(plot_left, sel_rect.bottom() + gap),
            egui::pos2(plot_right, sel_rect.bottom() + gap + line_h),
        );
        let heat_rect = egui::Rect::from_min_max(
            egui::pos2(plot_left, avg_rect.bottom() + gap),
            egui::pos2(plot_right, rect.bottom() - x_axis_h),
        );
        if heat_rect.width() < 2.0 || heat_rect.height() < 2.0 {
            return;
        }

        let start_ms = result.start_ms;
        let end_ms = start_ms + result.n_win as f64 * result.dt_ms;
        let span_ms = (end_ms - start_ms).max(1e-6);
        let x_of_ms = |ms: f64| -> f32 {
            heat_rect.left() + ((ms - start_ms) / span_ms) as f32 * heat_rect.width()
        };
        let n = result.n_win.max(2);
        let x_of_i = |rct: &egui::Rect, i: usize| -> f32 {
            rct.left() + (i as f32 / (n - 1) as f32) * rct.width()
        };

        // heatmap texture (rebuilt on color/size change)
        let pw = heat_rect.width().round() as usize;
        let ph = heat_rect.height().round() as usize;
        let vmax = self.psth.vmax(result);
        // traces zoom with the color scale; out-of-range parts are clipped to their plot
        let zoom = self.psth.trace_vmax_ref / vmax;
        let size_changed = self.psth.last_tex_size != Some([pw, ph]);
        if self.psth.tex_dirty || size_changed || self.psth.texture.is_none() {
            build_psth_heatmap_into(
                &mut self.psth.pixel_buf,
                result,
                pw,
                ph,
                vmax,
                self.peak_pooling,
                cmap,
            );
            let img = egui::ColorImage::from_rgba_unmultiplied([pw, ph], &self.psth.pixel_buf);
            self.psth.texture = Some(ui.ctx().load_texture(
                "psth_heatmap",
                img,
                TextureOptions::NEAREST,
            ));
            self.psth.last_tex_size = Some([pw, ph]);
            self.psth.tex_dirty = false;
        }
        if let Some(tex) = &self.psth.texture {
            painter.image(
                tex.id(),
                heat_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }

        // marker lines for the selected channels (like the main window)
        let n_disp = result.display_rows.len().max(1);
        let draw_marker =
            |ch: usize, color: egui::Color32| {
                if let Some(d) = result.display_rows.iter().position(
                    |r| matches!(r, DisplayRow::Data { first_ch, .. } if *first_ch + 1 == ch),
                ) {
                    let frac_y = ((n_disp - 1 - d) as f32 + 0.5) / n_disp as f32;
                    let y = heat_rect.top() + frac_y * heat_rect.height();
                    painter.line_segment(
                        [
                            egui::pos2(heat_rect.left(), y),
                            egui::pos2(heat_rect.right(), y),
                        ],
                        egui::Stroke::new(2.0_f32, color),
                    );
                }
            };
        if let Some(c) = self.psth.sel_ch1 {
            let [fr, fg, fb] = self.colormap_choice.spec().heatmap_fg;
            draw_marker(c, egui::Color32::from_rgba_unmultiplied(fr, fg, fb, 128));
        }
        if let Some(c) = self.psth.sel_ch2 {
            draw_marker(c, egui::Color32::from_rgba_unmultiplied(255, 182, 23, 128));
        }

        // channel selection: click the heatmap
        let resp = ui.interact(
            heat_rect,
            ui.id().with("psth_heat_click"),
            egui::Sense::click(),
        );
        let click = if resp.clicked() {
            resp.interact_pointer_pos().map(|p| (p, false))
        } else if resp.secondary_clicked() {
            resp.interact_pointer_pos().map(|p| (p, true))
        } else {
            None
        };
        if let Some((pos, right)) = click {
            if let Some(ch) = channel_at_heatmap_y(result, heat_rect, pos.y) {
                if right {
                    self.psth.sel_ch2 = Some(ch);
                } else {
                    self.psth.sel_ch1 = Some(ch);
                }
            }
        }

        // ---- selected-channel line plot ----
        painter.rect_filled(sel_rect, 0.0, c_zero);
        let mut sel_traces: Vec<(usize, egui::Color32)> = Vec::new();
        if let Some(c) = self.psth.sel_ch1 {
            sel_traces.push((c, ch1_color));
        }
        if let Some(c) = self.psth.sel_ch2 {
            sel_traces.push((c, ch2_color));
        }
        // symmetric scale across all shown channel traces
        let mut sel_max = 1e-6f32;
        for (ch, _) in &sel_traces {
            if let Some(row) = channel_row(result, *ch) {
                let s = &result.data[row * result.n_win..(row + 1) * result.n_win];
                sel_max = sel_max.max(s.iter().fold(0.0f32, |m, &v| m.max(v.abs())));
            }
        }
        sel_max /= zoom;
        let sel_mid = sel_rect.center().y;
        let sel_half = sel_rect.height() * 0.5 - 2.0;
        painter.line_segment(
            [
                egui::pos2(sel_rect.left(), sel_mid),
                egui::pos2(sel_rect.right(), sel_mid),
            ],
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(80)),
        );
        for (ch, color) in &sel_traces {
            if let Some(row) = channel_row(result, *ch) {
                let s = &result.data[row * result.n_win..(row + 1) * result.n_win];
                let pts: Vec<egui::Pos2> = (0..result.n_win)
                    .map(|i| {
                        egui::pos2(x_of_i(&sel_rect, i), sel_mid - (s[i] / sel_max) * sel_half)
                    })
                    .collect();
                painter
                    .with_clip_rect(sel_rect)
                    .add(egui::Shape::line(pts, egui::Stroke::new(1.5_f32, *color)));
            }
        }

        // ---- mean-across-channels line plot ----
        painter.rect_filled(avg_rect, 0.0, c_zero);
        let tmax = result
            .avg_trace
            .iter()
            .fold(0.0f32, |m, &v| m.max(v.abs()))
            .max(1e-6)
            / zoom;
        let avg_mid = avg_rect.center().y;
        let avg_half = avg_rect.height() * 0.5 - 2.0;
        painter.line_segment(
            [
                egui::pos2(avg_rect.left(), avg_mid),
                egui::pos2(avg_rect.right(), avg_mid),
            ],
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(80)),
        );
        let pts: Vec<egui::Pos2> = (0..result.n_win)
            .map(|i| {
                egui::pos2(
                    x_of_i(&avg_rect, i),
                    avg_mid - (result.avg_trace[i] / tmax) * avg_half,
                )
            })
            .collect();
        painter
            .with_clip_rect(avg_rect)
            .add(egui::Shape::line(pts, egui::Stroke::new(1.5_f32, accent)));

        // onset marker at t = 0 across all three plots
        if start_ms < 0.0 && end_ms > 0.0 {
            let x0 = x_of_ms(0.0);
            let stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_gray(150));
            painter.line_segment(
                [
                    egui::pos2(x0, sel_rect.top()),
                    egui::pos2(x0, sel_rect.bottom()),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    egui::pos2(x0, avg_rect.top()),
                    egui::pos2(x0, avg_rect.bottom()),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    egui::pos2(x0, heat_rect.top()),
                    egui::pos2(x0, heat_rect.bottom()),
                ],
                stroke,
            );
        }

        // labels
        let txt = egui::Color32::from_gray(200);
        let fid = egui::FontId::proportional(11.0);
        painter.text(
            egui::pos2(rect.left() + 2.0, sel_mid),
            egui::Align2::LEFT_CENTER,
            "chan µV",
            fid.clone(),
            txt,
        );
        painter.text(
            egui::pos2(rect.left() + 2.0, avg_mid),
            egui::Align2::LEFT_CENTER,
            "mean µV",
            fid.clone(),
            txt,
        );
        let (bottom_id, top_id) = edge_channel_ids(&self.meta, &result.display_rows);
        painter.text(
            egui::pos2(rect.left() + 2.0, heat_rect.top() + 2.0),
            egui::Align2::LEFT_TOP,
            top_id,
            fid.clone(),
            txt,
        );
        painter.text(
            egui::pos2(rect.left() + 2.0, heat_rect.bottom() - 2.0),
            egui::Align2::LEFT_BOTTOM,
            bottom_id,
            fid.clone(),
            txt,
        );
        let y_txt = rect.bottom() - x_axis_h + 4.0;
        painter.text(
            egui::pos2(heat_rect.left(), y_txt),
            egui::Align2::LEFT_TOP,
            format!("{:.0} ms", start_ms),
            fid.clone(),
            txt,
        );
        if start_ms < 0.0 && end_ms > 0.0 {
            painter.text(
                egui::pos2(x_of_ms(0.0), y_txt),
                egui::Align2::CENTER_TOP,
                "0",
                fid.clone(),
                txt,
            );
        }
        painter.text(
            egui::pos2(heat_rect.right(), y_txt),
            egui::Align2::RIGHT_TOP,
            format!("{:.0} ms", end_ms),
            fid,
            txt,
        );
    }

    /// Esc closes the topmost open tool window (PSTH, Atlas Registration, Preferences,
    /// ...), one per press; with no window open it closes the waveform view, then
    /// returns from the rectangle zoom. It is left alone while a text field, combo box or the
    /// channel context menu has it, and progress windows (with Abort) are never closed.
    fn close_top_window_on_escape(&mut self, ctx: &egui::Context) {
        if !ctx.input(|i| i.key_pressed(egui::Key::Escape))
            || ctx.wants_keyboard_input()
            || ctx.memory(|m| m.any_popup_open())
            || self.context_menu_channel.is_some()
        {
            return;
        }
        // General rule: every `egui::Window` other than the main one should close on
        // Escape, topmost first. Area ids are the window titles (see egui::Window::new),
        // so each one needs an entry here — a window-less "open" flag can't be
        // discovered automatically. A transient progress popup with no open/close
        // state of its own (classification/atlas/spectrum "computing") doesn't need
        // one: it closes itself when the job finishes, and Escape isn't a stand-in
        // for its "Abort" button.
        let open: Vec<(egui::Id, u8)> = [
            (self.psth.open, "Peri-Stimulus Time Histogram", 0),
            (self.atlas.open, "Atlas Registration", 1),
            (self.show_remove_channels, "Remove channels", 2),
            (self.show_preferences, "Preferences", 3),
            (self.ttl.open, "TTL", 4),
            (
                self.classify_error.is_some(),
                "Channel Classification failed",
                5,
            ),
            (self.spectrum_open, "Power Spectrum", 6),
            (self.spectrum_error.is_some(), "Power Spectrum failed", 7),
            (self.noise_open, "Noise Suppression", 8),
            (self.screenshot.open, "Screenshot", 9),
        ]
        .into_iter()
        .filter(|(is_open, ..)| *is_open)
        .map(|(_, title, which)| (egui::Id::new(title), which))
        .collect();
        if open.is_empty() {
            // no window left to close: Esc leaves the waveform zoom, the waveform view,
        // then the heatmap zoom
            if self.waveform_channel.is_some() {
                if !self.leave_waveform_zoom() {
                    self.waveform_channel = None;
                }
                self.heatmap_texture = None;
                return;
            }
            if let Some(z) = self.zoom.take() {
                self.view_start_s = z.prev_start_s;
                self.view_dur_s = z.prev_dur_s;
                self.window_dur_str = format!("{:.3}", self.view_dur_s);
                self.heatmap_texture = None;
            }
            return;
        }
        // back-to-front stacking order; a window not in it yet counts as the bottom one
        let depth = |id: egui::Id| ctx.memory(|m| m.layer_ids().position(|l| l.id == id));
        let (_, which) = open.into_iter().max_by_key(|(id, _)| depth(*id)).unwrap();
        match which {
            0 => self.psth.open = false,
            1 => self.atlas.open = false,
            2 => self.show_remove_channels = false,
            3 => self.show_preferences = false,
            4 => self.ttl.open = false,
            5 => self.classify_error = None,
            6 => self.spectrum_open = false,
            7 => self.spectrum_error = None,
            8 => self.noise_open = false,
            _ => self.screenshot.open = false,
        }
    }

    pub fn update(&mut self, ctx: &egui::Context) {
        self.poll_capture(ctx);
        // taking a screenshot: no windows, menus or hover effects over the plot
        let capturing = self.capture.is_some();
        self.atlas.no_hover = capturing;
        self.close_top_window_on_escape(ctx);
        self.poll_and_maybe_dispatch_psth(ctx);
        self.poll_remove_channels_picker(ctx);
        self.poll_classify(ctx);
        self.poll_spectrum(ctx);
        self.notch_panel.poll(ctx);
        self.atlas.poll(ctx);
        if !capturing {
            self.draw_psth_window(ctx);
            self.draw_remove_channels_window(ctx);
            self.draw_classify_progress_window(ctx);
            self.draw_spectrum_error_window(ctx);
            self.draw_spectrum_window(ctx);
            self.draw_noise_window(ctx);
            self.draw_channel_context_menu(ctx);
            self.atlas
                .draw_window(ctx, &self.meta, &self.bin_path, &self.colormap_choice);
            self.atlas.draw_progress_window(ctx);
            self.ttl.draw_window(
                ctx,
                &mut self.stim_layout_text,
                &self.bin_path,
                &mut self.stim_file,
                &mut self.view_start_s,
                self.view_dur_s,
                self.meta.n_samples as f64 / self.meta.sample_rate,
            );
            self.draw_screenshot_window(ctx);
        }
        self.autosave_settings(ctx);
        if self.atlas.take_prefs_dirty() {
            self.save_prefs();
        }

        let mut show_prefs = self.show_preferences;
        if show_prefs && !capturing {
            egui::Window::new("Preferences")
                .anchor(egui::Align2::RIGHT_TOP, [-10.0, 40.0])
                .collapsible(false)
                .open(&mut show_prefs)
                .show(ctx, |ui| {
                    // scrollable so every section stays reachable regardless of window
                    // height (the content has grown past a typical screen's height)
                    egui::ScrollArea::vertical()
                        .max_height(ui.ctx().screen_rect().height() - 80.0)
                        .show(ui, |ui| {
                    ui.label(egui::RichText::new("Appearance").strong());

                    ui.horizontal(|ui| {
                        ui.label("Colormap:");
                        let mut cm = self.colormap_choice.clone();
                        egui::ComboBox::from_id_salt("cm_combo")
                            .selected_text(cm.spec().name)
                            .show_ui(ui, |ui| {
                                for c in ColorMapChoice::ALL {
                                    ui.selectable_value(&mut cm, c.clone(), c.spec().name);
                                }
                            });
                        if cm != self.colormap_choice {
                            self.colormap_choice = cm;
                            self.heatmap_texture = None; // Force redraw
                            self.psth.tex_dirty = true; // PSTH heatmap tracks the same colormap
                        }
                    });

                    if ui
                        .checkbox(&mut self.peak_pooling, "Peak pooling")
                        .on_hover_text("How a heatmap pixel column shows the many samples it covers (e.g. ~11 samples per column in a 0.5 s window, ~200 in a 10 s window).\n\nOn (peak): each column shows the sample with the largest magnitude, sign kept. A spike keeps its full amplitude at any window length, like an oscilloscope's min/max display. The background looks noisier at long windows, and in %ile mode the colour range follows the values on screen.\n\nOff (mean): each column shows the average of its samples. Smooth, but it acts as a lowpass: a -100 µV spike fades to about -5 µV in a 10 s window, so mostly the LFP remains visible.")
                        .changed()
                    {
                        self.heatmap_texture = None;
                        self.psth.tex_dirty = true;
                        self.save_prefs();
                    }

                    ui.separator();
                    ui.label(egui::RichText::new("Channel layout").strong());

                    ui.horizontal(|ui| {
                        ui.label("Order channels by:");
                        let mut co = self.preproc_cfg.channel_order;
                        egui::ComboBox::from_id_salt("channel_order_combo")
                            .selected_text(match co {
                                ChannelOrder::Id => "ID",
                                ChannelOrder::Depth => "Depth",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut co, ChannelOrder::Id, "ID");
                                ui.selectable_value(&mut co, ChannelOrder::Depth, "Depth");
                            });
                        if co != self.preproc_cfg.channel_order {
                            self.preproc_cfg.channel_order = co;
                            self.heatmap_texture = None;
                            self.pending_cfg_recompute = true;
                            self.save_prefs();
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Order shanks by:");
                        let mut so = self.preproc_cfg.shank_order;
                        egui::ComboBox::from_id_salt("shank_order_combo")
                            .selected_text(match so {
                                ShankOrder::Id => "ID",
                                ShankOrder::XCoord => "x coordinate",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut so, ShankOrder::Id, "ID");
                                ui.selectable_value(&mut so, ShankOrder::XCoord, "x coordinate");
                            });
                        if so != self.preproc_cfg.shank_order {
                            self.preproc_cfg.shank_order = so;
                            self.heatmap_texture = None;
                            self.pending_cfg_recompute = true;
                            self.save_prefs();
                        }
                    });

                    ui.separator();
                    ui.label(egui::RichText::new("Firing rate overlay").strong());

                    if ui.checkbox(&mut self.show_firing_rate_overlay, "Show firing rate overlay").changed() {
                        // force the (skipped while hidden) spike projection to recompute
                        self.proj_view_first = usize::MAX;
                        self.heatmap_texture = None;
                        self.save_prefs();
                    }

                    ui.horizontal(|ui| {
                        ui.label("Spike Threshold (µV):");
                        if ui.add(egui::DragValue::new(&mut self.spike_threshold).speed(1.0)).changed() {
                            self.heatmap_texture = None;
                            self.save_prefs();
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Overlay scale:");
                        if ui.add(
                            egui::DragValue::new(&mut self.spike_overlay_scale)
                                .speed(0.05).range(0.1..=10.0)
                        ).changed() {
                            self.heatmap_texture = None;
                            self.save_prefs();
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Depth smoothing sigma (channels):");
                        if ui.add(
                            egui::DragValue::new(&mut self.spike_smoothing_sigma)
                                .speed(0.1).range(0.1..=10.0)
                        ).changed() {
                            self.heatmap_texture = None;
                            self.save_prefs();
                        }
                    });

                    ui.separator();
                    ui.label(egui::RichText::new("Channel Classification").strong());

                    ui.horizontal(|ui| {
                        ui.label("Chunks to sample:");
                        if ui.add(
                            egui::DragValue::new(&mut self.classify_n_chunks)
                                .speed(1.0).range(1..=500)
                        ).changed() {
                            self.save_prefs();
                        }
                    });
                    ui.label(
                        egui::RichText::new(
                            "number of 300 ms snippets, evenly spaced across the recording, that \
                             the majority vote is taken over on the next run — more chunks are \
                             more robust but slower"
                        ).small().color(egui::Color32::GRAY)
                    );

                    ui.horizontal(|ui| {
                        use crate::channel_classify::OutsideRule;
                        ui.label("Outside of brain:").on_hover_text("How channels outside the brain are found, from each channel's low-frequency similarity to its neighbours (xcor_lf; strongly negative above the brain surface). Only a run of such channels reaching the top of the shank is labelled.\n\nAdaptive (default): the threshold is taken where the smoothed similarity trend drops most steeply along the shank (IBL's optional 'adaptive' mode). Finds the brain surface even when the contrast is weak, e.g. in LFP recordings.\n\nFixed threshold: xcor_lf < -0.75, the default of IBL's ibldsp. Stricter; can miss the surface entirely.");
                        let mut rule = self.classify_outside_rule;
                        egui::ComboBox::from_id_salt("outside_rule_combo")
                            .selected_text(match rule {
                                OutsideRule::Fixed => "Fixed threshold (IBL)",
                                OutsideRule::Adaptive => "Adaptive (default)",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut rule, OutsideRule::Fixed, "Fixed threshold (IBL)");
                                ui.selectable_value(&mut rule, OutsideRule::Adaptive, "Adaptive (default)");
                            })
                            .response
                            .on_hover_text("How channels outside the brain are found, from each channel's low-frequency similarity to its neighbours (xcor_lf; strongly negative above the brain surface). Only a run of such channels reaching the top of the shank is labelled.\n\nAdaptive (default): the threshold is taken where the smoothed similarity trend drops most steeply along the shank (IBL's optional 'adaptive' mode). Finds the brain surface even when the contrast is weak, e.g. in LFP recordings.\n\nFixed threshold: xcor_lf < -0.75, the default of IBL's ibldsp. Stricter; can miss the surface entirely.");
                        if rule != self.classify_outside_rule {
                            self.classify_outside_rule = rule;
                            self.save_prefs();
                        }
                    });

                    ui.separator();
                    ui.label(egui::RichText::new("Buffer").strong());

                    let n_data_rows = self.meta.build_display_rows(self.preproc_cfg.avg_depths, &self.preproc_cfg.removed_channels, self.preproc_cfg.channel_order, self.preproc_cfg.shank_order)
                        .iter().filter(|r| matches!(r, DisplayRow::Data { .. })).count();
                    let max_feasible = max_feasible_buffer_s(n_data_rows, self.meta.sample_rate, self.mem_reserve_mb);

                    ui.horizontal(|ui| {
                        ui.label("Initial buffer size (s):");
                        if ui.add(
                            egui::DragValue::new(&mut self.initial_buffer_s)
                                .speed(0.5).range(1.0..=max_feasible)
                        ).changed() {
                            let fs = self.meta.sample_rate;
                            self.worker_half_window = compute_half_window(self.initial_buffer_s, fs);
                            let max_margin = max_extension_margin_s(self.initial_buffer_s, self.view_dur_s);
                            self.extension_margin_s = self.extension_margin_s.min(max_margin);
                            self.pending_cfg_recompute = true;
                            self.save_prefs();
                        }
                    });
                    ui.label(
                        egui::RichText::new(format!("(max given available memory: {:.1} s)", max_feasible))
                            .small().color(egui::Color32::GRAY)
                    );

                    let max_margin = max_extension_margin_s(self.initial_buffer_s, self.view_dur_s);
                    ui.horizontal(|ui| {
                        ui.label("Extension margin (s):");
                        if ui.add(
                            egui::DragValue::new(&mut self.extension_margin_s)
                                .speed(0.1).range(0.5..=max_margin)
                        ).changed() {
                            self.save_prefs();
                        }
                    });
                    ui.label(
                        egui::RichText::new(format!(
                            "distance from the buffer edge that triggers (and size of) each extension — capped at {:.1} s to prevent the buffer from ping-ponging between edges",
                            max_margin
                        )).small().color(egui::Color32::GRAY)
                    );

                    ui.horizontal(|ui| {
                        ui.label("Memory pressure threshold (%):");
                        if ui.add(
                            egui::DragValue::new(&mut self.mem_pressure_pct)
                                .speed(1.0).range(1.0..=90.0)
                        ).changed() {
                            self.save_prefs();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Memory reserve (MB):");
                        if ui.add(
                            egui::DragValue::new(&mut self.mem_reserve_mb)
                                .speed(50.0).range(100.0..=20000.0)
                        ).changed() {
                            self.save_prefs();
                        }
                    });
                    ui.label(
                        egui::RichText::new("buffer growth stops once free memory drops below either threshold")
                            .small().color(egui::Color32::GRAY)
                    );
                        });
                });
        }
        self.show_preferences = show_prefs;

        // mouse-wheel scroll — 5% of window per tick; Alt+scroll instead adjusts the
        // color scale (or, in single-channel waveform view, the waveform's y-axis range)
        let alt_held = ctx.input(|i| i.modifiers.alt);
        // wheel over a window (Atlas Registration, Preferences, PSTH, ...) scrolls only
        // that window; panels, incl. the heatmap, live on the Background layer
        let pointer_over_window = ctx
            .input(|i| i.pointer.hover_pos())
            .and_then(|p| ctx.layer_id_at(p))
            .is_some_and(|layer| layer.order != egui::Order::Background);
        let ticks = if pointer_over_window {
            0.0
        } else {
            ctx.input(|i| {
                i.events
                    .iter()
                    .filter_map(|e| match e {
                        egui::Event::MouseWheel { delta, .. } => Some(delta.y.signum()),
                        _ => None,
                    })
                    .sum::<f32>()
            })
        };
        if ticks != 0.0 && alt_held {
            // inverted relative to a plain sum of tick signs: scrolling up now
            // decreases the value (mirrors the "zoom out" feel of scroll-up elsewhere)
            if self.waveform_channel.is_some() {
                // proportional steps, so the scale stays usable when zoomed to a few µV
                self.waveform_y_range_uv = (self.waveform_y_range_uv * 1.05f32.powf(-ticks))
                    .clamp(WAVEFORM_MIN_RANGE_UV, 2000.0);
            } else if self.color_mode == ColorMode::Percentile {
                self.color_pct = (self.color_pct - ticks * 0.1).clamp(95.0, 100.0);
                self.color_pct_str = format!("{:.2}", self.color_pct);
                self.heatmap_texture = None;
            } else {
                self.color_uv = (self.color_uv - ticks * 5.0).clamp(10.0, 300.0);
                self.color_uv_str = format!("{:.0}", self.color_uv);
                self.heatmap_texture = None;
            }
        } else if ticks != 0.0 {
            let fs = self.meta.sample_rate;
            let total_s = self.meta.n_samples as f64 / fs;
            let max_start = (total_s - self.view_dur_s).max(0.0);
            let pct = if self.scroll_speed_fine { 0.05 } else { 0.30 };
            let step = ticks as f64 * self.view_dur_s * pct;
            self.view_start_s = (self.view_start_s + step).clamp(0.0, max_start);
        }

        // keyboard scroll (not while typing into a text field)
        let typing = ctx.wants_keyboard_input();
        ctx.input(|i| {
            if typing {
                return;
            }
            let fs = self.meta.sample_rate;
            let total_s = self.meta.n_samples as f64 / fs;
            let step = self.view_dur_s * 0.5;
            let max_start = (total_s - self.view_dur_s).max(0.0);
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::D) {
                self.view_start_s = (self.view_start_s + step).min(max_start);
            }
            if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::A) {
                self.view_start_s = (self.view_start_s - step).max(0.0);
            }
        });

        TopBottomPanel::top("toolbar").show(ctx, |ui| {
            self.draw_toolbar(ui);
        });
        TopBottomPanel::top("preproc").show(ctx, |ui| {
            self.draw_preproc_panel(ui);
        });
        TopBottomPanel::top("chan_ctrl").show(ctx, |ui| {
            self.draw_channel_controls(ui);
        });
        TopBottomPanel::bottom("status_bar")
            .exact_height(20.0)
            .show(ctx, |ui| {
                self.draw_status_bar(ui);
            });
        TopBottomPanel::bottom("nav_bar").show(ctx, |ui| {
            self.draw_nav_bar(ui);
        });

        CentralPanel::default()
            .frame(egui::Frame::new().fill(egui::Color32::from_rgb(crate::render::C_ZERO[0], crate::render::C_ZERO[1], crate::render::C_ZERO[2])))
            .show(ctx, |ui| {
                self.plot_rect = Some(ui.available_rect_before_wrap());
                let shot = self.shot_includes();
                let avail = ui.available_size();
                // power spectrum panel: a fixed 20% strip to the right of the main
                // heatmap, once a result exists and the user hasn't hidden it
                let spec_visible = self.spectrum_show_overlay
                    && self.waveform_channel.is_none()
                    && self.spectrum_result.as_ref().is_some_and(|c| c.matches(&self.preproc_cfg));
                let heat_w = if spec_visible { (avail.x * 0.8).max(1.0) } else { avail.x };
                let pw = heat_w as usize;
                let ph = avail.y as usize;
                if pw < 2 || ph < 2 { return; }

                let fs = self.meta.sample_rate;
                let view_first = (self.view_start_s * fs) as usize;
                let view_n = (self.view_dur_s * fs) as usize;
                let center = view_first + view_n / 2;

                // === single snapshot of worker state for all reads this frame ===
                let (
                    w_status, w_has_request,
                    buf_first, buf_n_samp, buf_data, buf_cfg, buf_display_rows,
                    matches_view, matches_cfg,
                    req_center_cfg, act_center_cfg,
                ) = {
                    let (lock, _) = &*self.worker_state;
                    let st = lock.lock().unwrap();
                    let status = st.status.clone();
                    let has_req = st.request.is_some();
                    let req_cc = st.request.as_ref().and_then(|r| {
                        if let RequestKind::Full { center_sample, half_window } = &r.kind {
                            Some((*center_sample, *half_window, r.cfg.clone()))
                        } else { None }
                    });
                    let act_cc = st.active_request.as_ref().and_then(|r| {
                        if let RequestKind::Full { center_sample, half_window } = &r.kind {
                            Some((*center_sample, *half_window, r.cfg.clone()))
                        } else { None }
                    });

                    if let Some(buf) = &st.buffer {
                        let buf_end = buf.first_sample + buf.n_samp;
                        let max_view_n = self.meta.n_samples.saturating_sub(view_first);
                        let expected_end = view_first + view_n.min(max_view_n);
                        let m_view = buf.first_sample <= view_first && expected_end <= buf_end;
                        let m_cfg = buf.cfg == self.preproc_cfg;

                        (status, has_req,
                         buf.first_sample, buf.n_samp,
                         Some(Arc::clone(&buf.data)), Some(buf.cfg.clone()),
                         Some(Arc::clone(&buf.display_rows)),
                         m_view, m_cfg, req_cc, act_cc)
                    } else {
                        (status, has_req, 0, 0, None, None, None, false, false, req_cc, act_cc)
                    }
                };

                // what the heatmap and waveform view draw (noise-filtered or not)
                let (src_data, src_first, src_n) = if matches_cfg {
                    self.display_source(&buf_data, &buf_display_rows, buf_first, buf_n_samp, view_first, view_n)
                } else {
                    (buf_data.clone(), buf_first, buf_n_samp)
                };

                // request repaint while worker is busy (moved here from top to use snapshot)
                if w_status == WorkerStatus::Computing || w_has_request {
                    ctx.request_repaint_after(std::time::Duration::from_millis(50));
                }

                // colour range from current UI settings — valid whenever cfg matches,
                // regardless of exact time coverage (percentile table covers whatever's
                // currently in the buffer). With peak pooling the %ile is taken from the
                // pooled values on screen instead: pooled extremes sit well above the raw
                // samples' percentiles, so a raw-sample range would saturate the map.
                let vmax = if matches_cfg {
                    if self.color_mode == ColorMode::Percentile && self.peak_pooling {
                        0.0 // unused: ColorScale::ViewPercentile below
                    } else if self.color_mode == ColorMode::Percentile {
                        // need percentile table — quick lock just for the lookup
                        let (lock, _) = &*self.worker_state;
                        let st = lock.lock().unwrap();
                        if let Some(buf) = &st.buffer {
                            let pct_idx = (self.color_pct * 100.0).round() as usize;
                            buf.vmax_pct[pct_idx.min(10000)].max(1.0)
                        } else { 250.0 }
                    } else {
                        self.color_uv.max(1.0)
                    }
                } else { 250.0 };

                // Rebuild whenever we have a cfg-matching buffer, rendering whatever time
                // overlap exists and letting build_heatmap_into background-fill the rest —
                // keeps the view live instead of freezing on a stale frame while extension
                // or a full recompute catches up.
                let rows_now = buf_display_rows.as_ref().map(|r| view_rows(r, self.zoom.as_ref()));
                if matches_cfg {
                    let pos_changed = self.last_rendered_first != view_first;
                    let cfg_changed = self.last_rendered_cfg.as_ref() != Some(&self.preproc_cfg);
                    let size_changed = self.last_rendered_size != Some([pw, ph]);
                    let buf_changed = self.last_rendered_buf != Some((buf_first, buf_n_samp));

                    let need_rebuild = self.heatmap_texture.is_none()
                        || pos_changed || view_n != self.last_rendered_n
                        || size_changed || buf_changed
                        || cfg_changed || rows_now != self.last_rendered_rows
                        || self.last_rendered_noise.as_ref() != Some(&self.noise);

                    if need_rebuild && view_n > 0 {
                        if let (Some(data_arc), Some(display_rows)) = (&buf_data, &buf_display_rows) {
                            let stride = buf_n_samp;

                            let (first_row, last_row) = view_rows(display_rows, self.zoom.as_ref());

                            // spike projection over whatever time overlap currently exists
                            // between the view and the buffer (may be partial or none)
                            let ov_start = view_first.max(buf_first);
                            let ov_end = (view_first + view_n).min(buf_first + buf_n_samp);
                            let (offset, n) = if ov_start < ov_end {
                                (ov_start - buf_first, ov_end - ov_start)
                            } else {
                                (0, 0)
                            };

                            // spike projection: only recompute if view/threshold/cfg changed,
                            // or the buffer itself changed (e.g. extension filled in previously
                            // out-of-range samples while the view stayed put)
                            let proj_stale = self.proj_view_first != view_first
                                || self.proj_view_n != view_n
                                || self.proj_threshold != self.spike_threshold
                                || self.proj_sigma != self.spike_smoothing_sigma
                                || self.proj_cfg.as_ref() != Some(&self.preproc_cfg)
                                || self.proj_rows != Some((first_row, last_row))
                                || buf_changed;

                            if proj_stale && self.show_firing_rate_overlay {
                                let visible = &display_rows[first_row..=last_row];
                                let mut sums = vec![0.0f32; visible.len()];

                                let sample_rate = self.meta.sample_rate;
                                let refractory_samples = (1.5 * sample_rate as f32 / 1000.0) as usize;
                                let threshold = self.spike_threshold;

                                use rayon::prelude::*;
                                sums.par_iter_mut().enumerate().for_each(|(i, count)| {
                                    if let DisplayRow::Data { data_idx, .. } = &visible[i] {
                                        let base = data_idx * stride + offset;
                                        if n > 0 && base + n <= data_arc.len() {
                                            let ch_data = &data_arc[base..base + n];
                                            let mut spikes = 0.0f32;
                                            let mut last_spike = None;
                                            for (t, &v) in ch_data.iter().enumerate() {
                                                if v < threshold {
                                                    if let Some(last_t) = last_spike {
                                                        if t - last_t > refractory_samples {
                                                            spikes += 1.0;
                                                            last_spike = Some(t);
                                                        }
                                                    } else {
                                                        spikes += 1.0;
                                                        last_spike = Some(t);
                                                    }
                                                }
                                            }
                                            *count = spikes;
                                        }
                                    }
                                });

                                // Gaussian convolution across depth, radius 3 (7-tap);
                                // sigma is user-configurable (default 1.5, matching the
                                // previous fixed kernel exactly)
                                let sigma = self.spike_smoothing_sigma.max(0.01);
                                let k_rad: isize = 3;
                                let kernel: [f32; 7] = std::array::from_fn(|j| {
                                    let x = j as f32 - k_rad as f32;
                                    (-x * x / (2.0 * sigma * sigma)).exp()
                                });
                                let mut smoothed = vec![0.0f32; sums.len()];
                                for i in 0..sums.len() {
                                    let mut v = 0.0;
                                    let mut weight_sum = 0.0;
                                    for j in 0..=6 {
                                        let idx = i as isize + (j as isize - k_rad);
                                        if idx >= 0 && idx < sums.len() as isize {
                                            v += sums[idx as usize] * kernel[j];
                                            weight_sum += kernel[j];
                                        }
                                    }
                                    if weight_sum > 0.0 {
                                        smoothed[i] = v / weight_sum;
                                    }
                                }
                                self.projection_sums = smoothed;
                                self.proj_view_first = view_first;
                                self.proj_view_n = view_n;
                                self.proj_threshold = self.spike_threshold;
                                self.proj_sigma = self.spike_smoothing_sigma;
                                self.proj_cfg = Some(self.preproc_cfg.clone());
                                self.proj_rows = Some((first_row, last_row));
                            }

                            let scale = if self.color_mode == ColorMode::Percentile && self.peak_pooling {
                                crate::render::ColorScale::ViewPercentile(self.color_pct)
                            } else {
                                crate::render::ColorScale::Fixed(vmax)
                            };
                            let shown_rows = if display_rows.is_empty() {
                                &display_rows[..]
                            } else {
                                &display_rows[first_row..=last_row]
                            };
                            build_heatmap_into(
                                &mut self.pixel_buf,
                                src_data.as_deref().map_or(&data_arc[..], |d| &d[..]),
                                shown_rows,
                                src_n, src_first, src_n, view_first, view_n,
                                pw, ph, scale, self.peak_pooling,
                                &self.colormap_choice,
                            );
                            let img = egui::ColorImage::from_rgba_unmultiplied([pw, ph], &self.pixel_buf);
                            // update the existing GPU texture in place while the size is
                            // unchanged (every scroll frame) instead of allocating a new one
                            match &mut self.heatmap_texture {
                                Some(tex) if !size_changed => tex.set(img, TextureOptions::NEAREST),
                                _ => self.heatmap_texture = Some(ctx.load_texture("heatmap", img, TextureOptions::NEAREST)),
                            }
                            self.last_rendered_first = view_first;
                            self.last_rendered_n = view_n;
                            self.last_rendered_cfg = buf_cfg;
                            self.last_rendered_size = Some([pw, ph]);
                            self.last_rendered_buf = Some((buf_first, buf_n_samp));
                            self.last_rendered_rows = Some((first_row, last_row));
                            self.last_rendered_noise = Some(self.noise.clone());
                        }
                    }
                }

                // request new background computation if needed (uses snapshot for checks, locks only for writes)
                let mut requested_new = false;
                if !matches_view || !matches_cfg || self.pending_cfg_recompute {
                    // a queued or running full recompute is good enough if it has the
                    // current config and buffer size and its window covers the view plus
                    // the extension margin — re-requesting on every scroll frame would
                    // cancel it each time and nothing would ever finish
                    let fs = self.meta.sample_rate;
                    let margin = (self.extension_margin_s * fs) as usize;
                    let need_lo = view_first.saturating_sub(margin);
                    let need_hi = (view_first + view_n + margin).min(self.meta.n_samples);
                    let adequate = |c: usize, hw: usize, cfg: &PreprocConfig| {
                        *cfg == self.preproc_cfg
                            && hw == self.worker_half_window
                            && c.saturating_sub(hw) <= need_lo
                            && (c + hw).min(self.meta.n_samples) >= need_hi
                    };
                    let already_requested = req_center_cfg.as_ref().map_or(false, |(c, hw, cfg)| adequate(*c, *hw, cfg))
                        || act_center_cfg.as_ref().map_or(false, |(c, hw, cfg)| adequate(*c, *hw, cfg));

                    if !already_requested {
                        file_log!("UI: Requesting recompute. matches_view={}, matches_cfg={}, pending_cfg={}, center={}", matches_view, matches_cfg, self.pending_cfg_recompute, center);
                        file_log!("UI: buf first={}, n={}, view_first={}, view_n={}", buf_first, buf_n_samp, view_first, view_n);
                        self.request_recompute();
                        requested_new = true;
                    }
                }

                if matches_view && matches_cfg && !requested_new {
                    let fs = self.meta.sample_rate;
                    let buf_end = buf_first + buf_n_samp;
                    let target_margin = (self.extension_margin_s * fs) as usize;

                    // spatial, direction-agnostic: whichever side the view is closest to
                    // the buffer edge on is (by construction) the side being scrolled toward,
                    // so this re-fires every frame the worker is idle until margin is restored —
                    // no need to track scroll direction explicitly, and it doesn't depend on a
                    // scroll event having fired this exact frame (unlike the old proximity check)
                    let left_margin = view_first.saturating_sub(buf_first);
                    let right_margin = buf_end.saturating_sub(view_first + view_n);
                    let left_needs = left_margin < target_margin && buf_first > 0;
                    let right_needs = right_margin < target_margin && buf_end < self.meta.n_samples;

                    let extend_dir: i32 = if left_needs && (!right_needs || left_margin <= right_margin) {
                        -1
                    } else if right_needs {
                        1
                    } else {
                        0
                    };

                    if extend_dir != 0 {
                        // only submit if the worker is free (Idle or Done — Done is the
                        // steady-state after any successful compute, so it must count as
                        // free too) and never cancel in-progress work
                        let (lock, cvar) = &*self.worker_state;
                        let mut st = lock.lock().unwrap();
                        if st.status != WorkerStatus::Computing && st.request.is_none() {
                            st.request = Some(WorkerRequest {
                                kind: RequestKind::Extend {
                                    direction: extend_dir,
                                    extension_samp: target_margin,
                                    view_first,
                                    view_n,
                                    max_buffer_samp: (self.initial_buffer_s * fs) as usize,
                                    mem_pressure_pct: self.mem_pressure_pct,
                                    mem_reserve_bytes: (self.mem_reserve_mb * 1e6) as u64,
                                },
                                cfg: self.preproc_cfg.clone(),
                            });
                            cvar.notify_one();
                        }
                    }
                }

                self.pending_cfg_recompute = false;

                // power spectrum, current-view scope: recomputed once "Calculate" has been
                // pressed, whenever the result no longer matches the view/settings — but
                // only after the view has stopped moving for SPECTRUM_SETTLE, so scrolling
                // isn't slowed down by an FFT of every channel on each frame. Skipped while
                // the panel is hidden or the waveform view replaces the heatmap.
                if (view_first, view_n) != self.spectrum_view_seen {
                    self.spectrum_view_seen = (view_first, view_n);
                    self.spectrum_view_changed_at = std::time::Instant::now();
                }
                if self.spectrum_want_live
                    && self.spectrum_show_overlay
                    && self.waveform_channel.is_none()
                    && self.spectrum_time_scope == crate::spectrum::SpectrumTimeScope::CurrentView
                {
                    let stale = self.spectrum_result.as_ref().is_none_or(|c| {
                        c.view != Some((view_first, view_n))
                            || c.source != self.spectrum_source
                            || !c.matches(&self.preproc_cfg)
                    });
                    let still_for = self.spectrum_view_changed_at.elapsed();
                    if stale && still_for < SPECTRUM_SETTLE {
                        ctx.request_repaint_after(SPECTRUM_SETTLE - still_for);
                    } else if stale {
                        let new_result = match self.spectrum_source {
                            crate::spectrum::SpectrumSource::Raw => {
                                let full_rows = self.meta.build_display_rows(
                                    self.preproc_cfg.avg_depths,
                                    &self.preproc_cfg.removed_channels,
                                    self.preproc_cfg.channel_order,
                                    self.preproc_cfg.shank_order,
                                );
                                crate::spectrum::compute_psd_raw_current_view(
                                    &self.raw, &self.meta, &full_rows, view_first, view_n,
                                )
                            }
                            // the buffer's rows are indexed by this same config's layout
                            crate::spectrum::SpectrumSource::Preprocessed => {
                                match (&buf_data, &buf_display_rows) {
                                    (Some(data), Some(rows)) if matches_cfg => {
                                        crate::spectrum::compute_psd_preprocessed_current_view(
                                            data, rows, buf_n_samp, buf_first, buf_n_samp, view_first,
                                            view_n, fs,
                                        )
                                    }
                                    _ => None,
                                }
                            }
                        };
                        if let Some(result) = new_result {
                            self.spectrum_result = Some(Arc::new(crate::spectrum::ComputedSpectrum {
                                result,
                                cfg: self.preproc_cfg.clone(),
                                source: self.spectrum_source,
                                view: Some((view_first, view_n)),
                            }));
                        }
                    }
                }

                if let Some(ch) = self.waveform_channel {
                    self.draw_waveform_view(ui, ch, matches_cfg, view_first, view_n, src_first, src_n, &src_data, &buf_display_rows);
                } else {

                // loading indicator
                if self.heatmap_texture.is_none() {
                    ui.painter().text(
                        ui.clip_rect().center(),
                        egui::Align2::CENTER_CENTER,
                        "⏳ Loading…",
                        egui::FontId::proportional(18.0),
                        egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200),
                    );
                }

                // draw texture
                if let Some(tex) = &self.heatmap_texture {
                    let img_widget = egui::Image::new(tex)
                        .fit_to_exact_size(egui::vec2(heat_w, avail.y))
                        .sense(egui::Sense::click_and_drag());
                    let resp = ui.add(img_widget);
                    self.ttl.draw_overlay(&ui.painter_at(resp.rect), resp.rect, self.view_start_s, self.view_dur_s, &self.colormap_choice);

                    // power spectrum panel: opaque heatmap in the strip to the right of
                    // the main view, one row per channel, aligned with the main heatmap's
                    // rows (including the current zoom)
                    if spec_visible {
                        if let Some(computed) = self.spectrum_result.clone() {
                            let full_rows: Arc<Vec<DisplayRow>> = match &buf_display_rows {
                                Some(rows) if matches_cfg => Arc::clone(rows),
                                _ => Arc::new(self.meta.build_display_rows(
                                    self.preproc_cfg.avg_depths,
                                    &self.preproc_cfg.removed_channels,
                                    self.preproc_cfg.channel_order,
                                    self.preproc_cfg.shank_order,
                                )),
                            };
                            let (sfirst_row, slast_row) = view_rows(&full_rows, self.zoom.as_ref());
                            let shown_rows = if full_rows.is_empty() {
                                &full_rows[..]
                            } else {
                                &full_rows[sfirst_row..=slast_row]
                            };
                            let row_offset = full_rows[..sfirst_row.min(full_rows.len())]
                                .iter()
                                .filter(|r| matches!(r, DisplayRow::Data { .. }))
                                .count();
                            let spec_w = (avail.x - resp.rect.width()).max(1.0) as usize;
                            let spec_h = ph;
                            let freq_range = self.spectrum_freq_range();
                            let key = SpectrumTexKey {
                                result: Arc::clone(&computed),
                                size: [spec_w, spec_h],
                                rows: (sfirst_row, slast_row),
                                scaling: self.spectrum_scaling,
                                normalization: self.spectrum_normalization,
                                freq_range,
                            };
                            if self.spectrum_tex_key.as_ref() != Some(&key) || self.spectrum_texture.is_none() {
                                crate::render::build_spectrum_heatmap_into(
                                    &mut self.spectrum_pixel_buf,
                                    &computed.result.power,
                                    &computed.result.freqs,
                                    shown_rows,
                                    row_offset,
                                    spec_w,
                                    spec_h,
                                    self.spectrum_scaling,
                                    self.spectrum_normalization,
                                    freq_range,
                                );
                                let simg = egui::ColorImage::from_rgba_unmultiplied(
                                    [spec_w, spec_h],
                                    &self.spectrum_pixel_buf,
                                );
                                match &mut self.spectrum_texture {
                                    Some(stex) if stex.size() == [spec_w, spec_h] => {
                                        stex.set(simg, TextureOptions::NEAREST)
                                    }
                                    _ => {
                                        self.spectrum_texture =
                                            Some(ctx.load_texture("spectrum", simg, TextureOptions::NEAREST))
                                    }
                                }
                                self.spectrum_tex_key = Some(key);
                            }
                            if let Some(stex) = &self.spectrum_texture {
                                let spec_rect = egui::Rect::from_min_max(
                                    egui::pos2(resp.rect.right(), resp.rect.top()),
                                    egui::pos2(resp.rect.right() + spec_w as f32, resp.rect.bottom()),
                                );
                                ui.painter().image(
                                    stex.id(),
                                    spec_rect,
                                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                    egui::Color32::WHITE,
                                );

                                // frequency axis: ticks + labels, black, at the bottom
                                let (f_lo, f_hi) = crate::render::spectrum_freq_bounds(&computed.result.freqs, freq_range);
                                let log_lo = f_lo.ln();
                                let log_span = (f_hi.ln() - log_lo).max(1e-6);
                                let painter = ui.painter();
                                let font = egui::FontId::proportional(9.0);
                                let y_bottom = spec_rect.bottom();
                                for f in crate::render::spectrum_ticks(f_lo, f_hi, freq_range.is_some()) {
                                    let frac = ((f.ln() - log_lo) / log_span).clamp(0.0, 1.0);
                                    let x = (spec_rect.left() + frac * spec_rect.width())
                                        .clamp(spec_rect.left() + 0.5, spec_rect.right() - 0.5);
                                    painter.line_segment(
                                        [egui::pos2(x, y_bottom - 5.5), egui::pos2(x, y_bottom - 1.0)],
                                        egui::Stroke::new(1.0_f32, egui::Color32::BLACK),
                                    );
                                    // centred on the tick, but kept inside the panel so labels
                                    // at the band edges aren't cut off (only the label moves)
                                    let galley = painter.layout_no_wrap(
                                        crate::render::spectrum_tick_label(f),
                                        font.clone(),
                                        egui::Color32::BLACK,
                                    );
                                    let size = galley.size();
                                    let lx = (x - size.x / 2.0)
                                        .min(spec_rect.right() - 2.0 - size.x)
                                        .max(spec_rect.left() + 2.0);
                                    painter.galley(egui::pos2(lx, y_bottom - 6.5 - size.y), galley, egui::Color32::BLACK);
                                }
                            }
                        }
                    }

                    // right edge the channel-selection lines and atlas region borders
                    // visually extend to: the spectrum panel's right edge when it's
                    // showing, otherwise the same as resp.rect.right()
                    let full_right = resp.rect.left() + avail.x;

                    // click detection — channel selection now requires Alt (plain
                    // left-click is a no-op; plain right-click opens the context menu)
                    let mut click_pos = None;
                    let mut is_left_click = false;
                    let mut is_right_click = false;
                    let mut is_context_click = false;

                    if resp.clicked() {
                        click_pos = resp.interact_pointer_pos().or_else(|| ctx.input(|i| i.pointer.interact_pos()));
                        if alt_held {
                            is_left_click = true;
                        }
                    }
                    if resp.secondary_clicked() {
                        click_pos = resp.interact_pointer_pos().or_else(|| ctx.input(|i| i.pointer.interact_pos()));
                        if alt_held {
                            is_right_click = true;
                        } else {
                            is_context_click = true;
                        }
                    }

                    if let Some(display_rows) = buf_display_rows.as_ref().filter(|r| !r.is_empty()) {
                        let (first_row, last_row) = view_rows(display_rows, self.zoom.as_ref());
                        let n_rows = last_row.saturating_sub(first_row) + 1;

                        // rectangle zoom: plain left-drag (Alt+drag is left to the atlas
                        // borders). Atlas border grab zones sit on top of the heatmap, so a
                        // drag starting on a border never reaches here.
                        if resp.drag_started_by(egui::PointerButton::Primary) && !alt_held {
                            self.zoom_drag_start = ctx.input(|i| i.pointer.press_origin());
                        }
                        if let Some(start) = self.zoom_drag_start {
                            let end = ctx.input(|i| i.pointer.interact_pos()).unwrap_or(start);
                            let sel = egui::Rect::from_two_pos(start, end).intersect(resp.rect);
                            if resp.drag_stopped() || !ctx.input(|i| i.pointer.primary_down()) {
                                self.zoom_drag_start = None;
                                let chans = zoom_selection(display_rows, first_row, last_row, resp.rect, sel);
                                if sel.width() >= ZOOM_MIN_DRAG_PX && sel.height() >= ZOOM_MIN_DRAG_PX {
                                    if let Some((ch_bottom, ch_top)) = chans {
                                        let w = resp.rect.width() as f64;
                                        let t0 = self.view_start_s + (sel.left() - resp.rect.left()) as f64 / w * self.view_dur_s;
                                        let t1 = self.view_start_s + (sel.right() - resp.rect.left()) as f64 / w * self.view_dur_s;
                                        // nested zooms keep the view from before the first one
                                        let (prev_start_s, prev_dur_s) = self
                                            .zoom
                                            .as_ref()
                                            .map_or((self.view_start_s, self.view_dur_s), |z| (z.prev_start_s, z.prev_dur_s));
                                        let total_s = self.meta.n_samples as f64 / self.meta.sample_rate;
                                        self.view_dur_s = (t1 - t0).max(0.01);
                                        self.view_start_s = t0.clamp(0.0, (total_s - self.view_dur_s).max(0.0));
                                        self.window_dur_str = format!("{:.3}", self.view_dur_s);
                                        self.zoom = Some(Zoom { ch_bottom, ch_top, prev_start_s, prev_dur_s });
                                        ctx.request_repaint();
                                    }
                                }
                            } else {
                                ui.painter().rect_filled(sel, 0.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 3));
                                ui.painter().rect_stroke(
                                    sel,
                                    0.0,
                                    egui::Stroke::new(1.0_f32, egui::Color32::WHITE),
                                    egui::StrokeKind::Inside,
                                );
                            }
                        }

                        // handle clicks
                        if let Some(pos) = click_pos {
                            let frac_y = ((pos.y - resp.rect.top()) / resp.rect.height()).clamp(0.0, 1.0);
                            let disp_idx = last_row.saturating_sub(
                                (frac_y as f64 * n_rows as f64) as usize
                            ).clamp(first_row, last_row);

                            if let DisplayRow::Data { first_ch, .. } = &display_rows[disp_idx] {
                                let ch = *first_ch + 1;
                                if is_left_click {
                                    self.selected_channel_1 = Some(ch);
                                }
                                if is_right_click {
                                    self.selected_channel_2 = Some(ch);
                                }
                                if is_context_click {
                                    self.context_menu_channel = Some(ch);
                                    self.context_menu_pos = Some(pos);
                                }
                            }
                        }

                        // draw channel marker lines
                        let draw_line = |ch_to_draw: usize, color: egui::Color32| {
                            for r in first_row..=last_row {
                                if let DisplayRow::Data { first_ch, .. } = &display_rows[r] {
                                    if *first_ch + 1 == ch_to_draw {
                                        let frac_y = (last_row - r) as f32 / n_rows as f32 + (0.5 / n_rows as f32);
                                        let y = resp.rect.top() + frac_y * resp.rect.height();
                                        ui.painter().line_segment(
                                            [egui::pos2(resp.rect.left(), y), egui::pos2(full_right, y)],
                                            egui::Stroke::new(2.0_f32, color)
                                        );
                                        break;
                                    }
                                }
                            }
                        };

                        let [fr, fg, fb] = self.colormap_choice.spec().heatmap_fg;
                        if shot.selection {
                            if let Some(ch1) = self.selected_channel_1 {
                                draw_line(ch1, egui::Color32::from_rgba_unmultiplied(fr, fg, fb, 128));
                            }
                            if let Some(ch2) = self.selected_channel_2 {
                                draw_line(ch2, egui::Color32::from_rgba_unmultiplied(255, 182, 23, 128));
                            }
                        }
                        if let Some(ctx_ch) = self.context_menu_channel.filter(|_| self.capture.is_none()) {
                            draw_line(ctx_ch, egui::Color32::from_rgba_unmultiplied(fr, fg, fb, 100));
                        }

                        // label each shank's section with "shank N", at the top-right
                        // corner of its (possibly partial) visible span — only when more
                        // than one shank is actually visible
                        let mut shank_tops: Vec<(u32, usize)> = Vec::new(); // (shank, topmost visible row idx)
                        for r in first_row..=last_row {
                            if let DisplayRow::Data { shank, .. } = &display_rows[r] {
                                match shank_tops.last_mut() {
                                    Some((s, top)) if *s == *shank => *top = r,
                                    _ => shank_tops.push((*shank, r)),
                                }
                            }
                        }
                        if shank_tops.len() > 1 {
                            for (shank, top_idx) in shank_tops {
                                let frac_top = (last_row - top_idx) as f32 / n_rows as f32;
                                let y = resp.rect.top() + frac_top * resp.rect.height();
                                ui.painter().text(
                                    egui::pos2(resp.rect.right() - 6.0, y + 2.0),
                                    egui::Align2::RIGHT_TOP,
                                    format!("shank {shank}"),
                                    egui::FontId::proportional(12.0),
                                    egui::Color32::from_rgba_unmultiplied(fr, fg, fb, 200),
                                );
                            }
                        }

                        // draw projection overlay
                        if self.show_firing_rate_overlay && !self.projection_sums.is_empty() {
                            // scale with window duration and threshold (baseline: -20 µV → 1x),
                            // plus a user-configurable multiplier (spike_overlay_scale, default 1)
                            let threshold_scale = self.spike_threshold.abs() / 20.0;
                            let spike_scale_factor = threshold_scale * (0.5 / self.view_dur_s) as f32 * self.spike_overlay_scale;

                            // per-map alpha is tuned for this overlay's many-triangle accumulation,
                            // so it is a separate field from the shared accent RGB in colormap.rs
                            let overlay_alpha = self.colormap_choice.spec().overlay_alpha;
                            let [pr, pg, pb] = self.colormap_choice.spec().accent;
                            let color = egui::Color32::from_rgba_unmultiplied(pr, pg, pb, overlay_alpha);

                            let min_x = resp.rect.left();
                            let max_x = resp.rect.right();
                            let top_y = resp.rect.top();
                            let h = resp.rect.height();
                            let row_h = h / n_rows as f32;

                            let mut mesh = egui::epaint::Mesh::default();

                            for (i, &count) in self.projection_sums.iter().enumerate() {
                                let x = min_x + count * spike_scale_factor;
                                let x = x.min(max_x);

                                let y = top_y + h - (i as f32 + 0.5) * row_h;

                                let idx_base = mesh.vertices.len() as u32;
                                mesh.vertices.push(egui::epaint::Vertex {
                                    pos: egui::pos2(min_x, y),
                                    uv: egui::epaint::WHITE_UV,
                                    color,
                                });
                                mesh.vertices.push(egui::epaint::Vertex {
                                    pos: egui::pos2(x, y),
                                    uv: egui::epaint::WHITE_UV,
                                    color,
                                });

                                if i > 0 {
                                    mesh.indices.push(idx_base - 2);
                                    mesh.indices.push(idx_base - 1);
                                    mesh.indices.push(idx_base);

                                    mesh.indices.push(idx_base - 1);
                                    mesh.indices.push(idx_base + 1);
                                    mesh.indices.push(idx_base);
                                }
                            }

                            if !mesh.is_empty() {
                                ui.painter().add(egui::Shape::mesh(mesh));
                            }
                        }

                        // channel classification overlay: horizontal stripe per
                        // non-good row (worst label wins when avg_depths merges
                        // several raw channels into one row: dead > noisy > outside)
                        if self.show_classification_overlay {
                            if let Some(labels) = &self.channel_labels {
                                let top_y = resp.rect.top();
                                let h = resp.rect.height();
                                let row_h = h / n_rows as f32;
                                let min_x = resp.rect.left();
                                let max_x = resp.rect.right();

                                for r in first_row..=last_row {
                                    if let DisplayRow::Data { channels, .. } = &display_rows[r] {
                                        let worst = channels
                                            .iter()
                                            .filter_map(|&ch| labels.get(ch).copied())
                                            .max_by_key(|&l| match l { 1 => 3, 2 => 2, 3 => 1, _ => 0 });
                                        if let Some(l) = worst {
                                            if l != 0 {
                                                let frac_top = (last_row - r) as f32 / n_rows as f32;
                                                let y0 = top_y + frac_top * h;
                                                let rect = egui::Rect::from_min_size(
                                                    egui::pos2(min_x, y0),
                                                    egui::vec2(max_x - min_x, row_h),
                                                );
                                                ui.painter().rect_filled(rect, 0.0, classification_color(l, CLASSIFICATION_OVERLAY_ALPHA));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // atlas region borders + labels, on top of the firing-rate overlay
                        self.atlas.draw_overlay(
                            ui,
                            &self.meta,
                            resp.rect,
                            full_right,
                            display_rows,
                            first_row,
                            last_row,
                            self.preproc_cfg.channel_order,
                            &self.colormap_choice,
                        );

                        // legend box: one Show/Hide row per overlay that has been computed
                        // this session (TTL, Atlas, firing rate, power spectrum, channel
                        // classification), pinned to the heatmap's bottom-right corner just
                        // above the scale bar. Laid out inside the heatmap panel itself (not
                        // a floating Area), so every window the user opens stays on top of it.
                        if self.capture.is_none()
                            && (self.channel_labels.is_some()
                                || self.ttl.has_data()
                                || self.atlas.has_data()
                                || !self.projection_sums.is_empty()
                                || self.spectrum_result.is_some())
                        {
                            // anchored by its bottom-right corner; the scale bar's top
                            // edge sits 30 px above the heatmap bottom. The box's size is
                            // only known after layout, so last frame's size places it and
                            // a size change (legend shown/hidden) re-runs the layout pass.
                            let size_id = egui::Id::new("classify_box_size");
                            let prev_size: egui::Vec2 =
                                ctx.data(|d| d.get_temp(size_id)).unwrap_or_default();
                            let anchor = egui::pos2(resp.rect.right() - 8.0, resp.rect.bottom() - 38.0);
                            let mut box_ui = ui.new_child(
                                egui::UiBuilder::new()
                                    .max_rect(egui::Rect::from_min_max(anchor - prev_size, anchor))
                                    .layout(egui::Layout::top_down(egui::Align::Min)),
                            );
                            box_ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                            box_ui.visuals_mut().override_text_color = Some(egui::Color32::WHITE);
                            let bg = egui::Color32::from_rgba_unmultiplied(
                                crate::render::C_ZERO[0],
                                crate::render::C_ZERO[1],
                                crate::render::C_ZERO[2],
                                77,
                            );
                            let box_size = egui::Frame::new()
                                .fill(bg)
                                .corner_radius(8.0)
                                .inner_margin(8.0)
                                .show(&mut box_ui, |ui| {
                                    ui.vertical(|ui| {
                                        let legend_row = |ui: &mut Ui, color: egui::Color32, text: &str| {
                                            ui.horizontal(|ui| {
                                                let (rect, _) = ui.allocate_exact_size(
                                                    egui::vec2(12.0, 12.0),
                                                    egui::Sense::hover(),
                                                );
                                                ui.painter().rect_filled(rect, 2.0, color);
                                                ui.label(text);
                                            });
                                        };
                                        let mut any_above = false;

                                        if self.ttl.has_data() {
                                            if self.ttl.show_overlay {
                                                let [r, g, b] = self.colormap_choice.spec().accent;
                                                legend_row(ui, egui::Color32::from_rgb(r, g, b), "TTL");
                                            }
                                            let label = if self.ttl.show_overlay { "Hide TTL" } else { "Show TTL" };
                                            ui.toggle_value(&mut self.ttl.show_overlay, label);
                                            any_above = true;
                                        }

                                        if self.atlas.has_data() {
                                            if any_above {
                                                ui.add_space(4.0);
                                            }
                                            if self.atlas.show_overlay {
                                                let [r, g, b] = self.colormap_choice.spec().atlas;
                                                legend_row(ui, egui::Color32::from_rgb(r, g, b), "Atlas");
                                            }
                                            let label = if self.atlas.show_overlay { "Hide Atlas" } else { "Show Atlas" };
                                            ui.toggle_value(&mut self.atlas.show_overlay, label);
                                            any_above = true;
                                        }

                                        if !self.projection_sums.is_empty() {
                                            if any_above {
                                                ui.add_space(4.0);
                                            }
                                            if self.show_firing_rate_overlay {
                                                let [r, g, b] = self.colormap_choice.spec().accent;
                                                legend_row(ui, egui::Color32::from_rgb(r, g, b), "Firing rate");
                                            }
                                            let label = if self.show_firing_rate_overlay {
                                                "Hide firing rate"
                                            } else {
                                                "Show firing rate"
                                            };
                                            if ui.toggle_value(&mut self.show_firing_rate_overlay, label).changed() {
                                                self.proj_view_first = usize::MAX;
                                                self.heatmap_texture = None;
                                            }
                                            any_above = true;
                                        }

                                        if self.spectrum_result.is_some() {
                                            if any_above {
                                                ui.add_space(4.0);
                                            }
                                            if self.spectrum_show_overlay {
                                                // swatch: the spectrum palette, low to high power
                                                ui.horizontal(|ui| {
                                                    let (rect, _) = ui.allocate_exact_size(
                                                        egui::vec2(12.0, 12.0),
                                                        egui::Sense::hover(),
                                                    );
                                                    for i in 0..12 {
                                                        let [r, g, b] = crate::render::spectrum_color(i as f32 / 11.0);
                                                        let x = rect.left() + i as f32;
                                                        ui.painter().rect_filled(
                                                            egui::Rect::from_x_y_ranges(x..=x + 1.0, rect.y_range()),
                                                            0.0,
                                                            egui::Color32::from_rgb(r, g, b),
                                                        );
                                                    }
                                                    ui.label("Power spectrum");
                                                });
                                            }
                                            let label =
                                                if self.spectrum_show_overlay { "Hide Spectra" } else { "Show Spectra" };
                                            ui.toggle_value(&mut self.spectrum_show_overlay, label);
                                            any_above = true;
                                        }

                                        if self.channel_labels.is_some() {
                                            if any_above {
                                                ui.add_space(4.0);
                                            }
                                            if self.show_classification_overlay {
                                                legend_row(ui, classification_color(1, 255), "Dead");
                                                legend_row(ui, classification_color(2, 255), "Noisy");
                                                legend_row(ui, classification_color(3, 255), "Out of brain");
                                            }
                                            if self.show_classification_overlay || any_above {
                                                ui.add_space(4.0);
                                            }
                                            let label = if self.show_classification_overlay {
                                                "Hide Chan Classification"
                                            } else {
                                                "Show Chan Classification"
                                            };
                                            ui.toggle_value(&mut self.show_classification_overlay, label);
                                        }
                                    });
                                })
                                .response
                                .rect
                                .size();
                            if (box_size - prev_size).length() > 0.5 {
                                ctx.data_mut(|d| d.insert_temp(size_id, box_size));
                                ctx.request_discard("classification box resized");
                            }
                        }
                    }

                    // hover overlay: ch / time / voltage
                    let hover_pos = resp.hover_pos().or_else(|| {
                        if resp.dragged() || resp.is_pointer_button_down_on() {
                            ctx.input(|i| i.pointer.interact_pos())
                        } else {
                            None
                        }
                    }).filter(|_| self.capture.is_none());

                    if let Some(pos) = hover_pos {
                        if let Some(display_rows) = buf_display_rows.as_ref().filter(|r| !r.is_empty()) {
                            let (first_row, last_row) = view_rows(display_rows, self.zoom.as_ref());
                            let n_rows = last_row.saturating_sub(first_row) + 1;

                            let frac_y = ((pos.y - resp.rect.top()) / resp.rect.height()).clamp(0.0, 1.0);
                            let disp_idx = last_row.saturating_sub(
                                (frac_y as f64 * n_rows as f64) as usize
                            ).clamp(first_row, last_row);

                            let ch_str = match &display_rows[disp_idx] {
                                DisplayRow::Data { first_ch, .. } => {
                                    let id = self.meta.channel_id(*first_ch);
                                    match self.atlas.channel_region_label(*first_ch) {
                                        Some(region) => format!("{id}  {region}  "),
                                        None => format!("{id}  "),
                                    }
                                }
                                DisplayRow::IntraShankGap => "Channel gap  ".to_string(),
                                DisplayRow::ShankBoundary => "Shank gap  ".to_string(),
                            };

                            let frac_x = ((pos.x - resp.rect.left()) / resp.rect.width()).clamp(0.0, 1.0);
                            let t = self.view_start_s + frac_x as f64 * self.view_dur_s;

                            // voltage readout from snapshot data
                            let voltage_uv: Option<f32> = if let Some(DisplayRow::Data { data_idx, .. }) = display_rows.get(disp_idx) {
                                // same samples as drawn (noise-filtered when that is on)
                                if let Some(data) = &src_data {
                                    let t_sample = (t * self.meta.sample_rate) as usize;
                                    if t_sample >= src_first {
                                        let off = t_sample - src_first;
                                        let idx = data_idx * src_n + off;
                                        if off < src_n && idx < data.len() {
                                            Some(data[idx])
                                        } else { None }
                                    } else { None }
                                } else { None }
                            } else { None };

                            let volt_str = voltage_uv.map(|v| format!("  {:.1} µV", v)).unwrap_or_default();
                            let mut label = format!("{}t = {:.4} s{}", ch_str, t, volt_str);
                            // while dragging a zoom rectangle: the channel and time ranges it covers
                            if let Some(start) = self.zoom_drag_start {
                                let sel = egui::Rect::from_two_pos(start, pos).intersect(resp.rect);
                                let t_at = |x: f32| {
                                    self.view_start_s
                                        + ((x - resp.rect.left()) / resp.rect.width()) as f64 * self.view_dur_s
                                };
                                let chans = zoom_selection(display_rows, first_row, last_row, resp.rect, sel)
                                    .map(|(b, t)| format!("{}–{}  ", self.meta.channel_id(b), self.meta.channel_id(t)))
                                    .unwrap_or_default();
                                label = format!("{chans}t = {:.4}–{:.4} s", t_at(sel.left()), t_at(sel.right()));
                            }

                            draw_readout(ui.painter(), resp.rect, label);
                        }
                    }

                    // scale bar overlay (10% of view_dur_s) bottom right
                    let [fr, fg, fb] = self.colormap_choice.spec().heatmap_fg;
                    let fg_color = egui::Color32::from_rgb(fr, fg, fb);
                    if shot.scale_bar {
                        let scale_bar_frac = 0.1;
                        let scale_bar_w = resp.rect.width() * scale_bar_frac;
                        let scale_bar_h = 4.0;
                        let bar_min = resp.rect.right_bottom() - egui::vec2(scale_bar_w + 20.0, 30.0);
                        let bar_rect = egui::Rect::from_min_size(bar_min, egui::vec2(scale_bar_w, scale_bar_h));
                        ui.painter().rect_filled(bar_rect, 0.0, fg_color);

                        let dur_ms = self.view_dur_s * (scale_bar_frac as f64) * 1000.0;
                        ui.painter().text(
                            bar_rect.right_bottom() + egui::vec2(0.0, 5.0),
                            egui::Align2::RIGHT_TOP,
                            format!("{:.0} ms", dur_ms),
                            egui::FontId::proportional(14.0),
                            fg_color,
                        );
                    }

                    // zoom notice, top left
                    if self.zoom.is_some() && self.capture.is_none() {
                        draw_zoom_notice(ui.painter(), resp.rect.left_top() + egui::vec2(10.0, 10.0), &self.colormap_choice);
                    }
                }

                } // end else (heatmap view)
            });

        if let Some(c) = &mut self.capture {
            c.frames += 1;
            if c.frames == 2 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(MainScreenshot)));
            }
            ctx.request_repaint();
        }

        // Idle heartbeat: some Wayland compositors flag a window that stops submitting
        // frames entirely (fully event-driven idle, no pending repaint requests) as
        // "not responding", even though the event loop is fine. Keep a low-frequency
        // repaint going at all times so a frame always lands within ~1s.
        ctx.request_repaint_after(std::time::Duration::from_secs(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short synthetic NP 1.0 recording without a geometry map (so the reference
    /// site comes from the probe type); returns the AP and LF data files.
    fn write_recording(dir: &std::path::Path) -> (PathBuf, PathBuf) {
        std::fs::create_dir_all(dir).unwrap();
        let mut out = Vec::new();
        for (band, fs, n_samp, ap_lf) in [("ap", 30000.0, 9000usize, "384,0,1"), ("lf", 2500.0, 750, "0,384,1")] {
            let bin = dir.join(format!("rec_g0_t0.imec0.{band}.bin"));
            let n_ch = 385;
            let data: Vec<u8> = (0..n_samp * n_ch).flat_map(|i| (((i * 7) % 200) as i16 - 100).to_le_bytes()).collect();
            std::fs::write(&bin, &data).unwrap();
            std::fs::write(
                bin.with_extension("meta"),
                format!(
                    "nSavedChans={n_ch}\nimSampRate={fs}\nfileSizeBytes={}\nsnsApLfSy={ap_lf}\nimDatPrb_type=0\n",
                    data.len()
                ),
            )
            .unwrap();
            out.push(bin);
        }
        (out[0].clone(), out[1].clone())
    }

    #[test]
    fn waveform_zoom_returns_to_the_previous_view() {
        let dir = std::env::temp_dir().join(format!("npx_app_wzoom_{}", std::process::id()));
        let (ap, _) = write_recording(&dir);
        let mut app = NPXplorerApp::new(&egui::Context::default(), ap).unwrap();
        app.waveform_channel = Some(5);
        let before = (app.view_start_s, app.view_dur_s, app.waveform_y_range_uv);
        app.waveform_zoom = Some(WaveformZoom { prev_start_s: before.0, prev_dur_s: before.1, prev_y_range_uv: before.2 });
        (app.view_start_s, app.view_dur_s, app.waveform_y_range_uv, app.waveform_y_center_uv) = (0.1, 0.02, 12.0, -30.0);
        // the zoom isn't what is saved
        let s = app.band_settings();
        assert_eq!((s.view_start_s, s.view_dur_s, s.waveform_y_range_uv), before);

        assert!(app.leave_waveform_zoom());
        assert_eq!((app.view_start_s, app.view_dur_s, app.waveform_y_range_uv), before);
        assert_eq!(app.waveform_y_center_uv, 0.0);
        assert!(!app.leave_waveform_zoom());
        drop(app);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn settings_are_restored_on_reopen() {
        let dir = std::env::temp_dir().join(format!("npx_app_settings_{}", std::process::id()));
        let (ap, lf) = write_recording(&dir);
        let ctx = egui::Context::default();

        // first open: the reference site is removed and the settings file written
        let mut app = NPXplorerApp::new(&ctx, ap.clone()).unwrap();
        assert_eq!(app.preproc_cfg.removed_channels, BTreeSet::from([191]));
        assert!(crate::settings::path(&ap).is_file());

        app.preproc_cfg.removed_channels = BTreeSet::from([3, 191]);
        app.preproc_cfg.spatial_filter = SpatialFilter::Destripe;
        app.preproc_cfg.notches = vec![crate::notch::Notch { freq_hz: 50.0, bw_hz: 1.0 }];
        app.preproc_cfg.notch_enabled = true;
        app.view_dur_s = 0.1;
        app.view_start_s = 0.05;
        app.color_mode = ColorMode::Voltage;
        app.color_uv = 77.0;
        app.colormap_choice = ColorMapChoice::Vanimo;
        app.waveform_y_range_uv = 333.0;
        app.noise.sigmoid_enabled = true;
        app.spectrum_n_chunks = 42;
        app.show_classification_overlay = true;
        app.stim_layout_text = "header\no,f\n".to_string();
        app.psth.color_pct = 97.5;
        drop(app);

        let app = NPXplorerApp::new(&ctx, ap.clone()).unwrap();
        assert_eq!(app.preproc_cfg.removed_channels, BTreeSet::from([3, 191]));
        assert_eq!(app.preproc_cfg.spatial_filter, SpatialFilter::Destripe);
        assert!(app.preproc_cfg.highpass && app.preproc_cfg.notch_enabled);
        assert_eq!(app.preproc_cfg.notches.len(), 1);
        assert_eq!((app.view_dur_s, app.view_start_s), (0.1, 0.05));
        assert!(app.color_mode == ColorMode::Voltage && app.colormap_choice == ColorMapChoice::Vanimo);
        assert_eq!((app.color_uv, app.waveform_y_range_uv), (77.0, 333.0));
        assert!(app.noise.sigmoid_enabled && app.noise_draft.sigmoid_enabled);
        assert_eq!(app.spectrum_n_chunks, 42);
        assert!(app.show_classification_overlay);
        assert_eq!(app.stim_layout_text, "header\no,f\n");
        assert_eq!(app.psth.color_pct, 97.5);
        drop(app);

        // the LF file shares the removed channels and stimulus settings, but has its
        // own band settings (from the preferences, not the AP band's)
        let app = NPXplorerApp::new(&ctx, lf.clone()).unwrap();
        assert_eq!(app.band, crate::settings::Band::Lf);
        assert_eq!(app.preproc_cfg.removed_channels, BTreeSet::from([3, 191]));
        assert_eq!(app.stim_layout_text, "header\no,f\n");
        assert!(app.preproc_cfg.notches.is_empty());
        drop(app);
        let t = crate::settings::load_table(&ap);
        assert!(t.contains_key("ap") && t.contains_key("lf"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
