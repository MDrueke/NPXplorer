use egui::{CentralPanel, TextureHandle, TextureOptions, TopBottomPanel, Ui, Vec2};
mod central;
mod prefs;
mod psth;
mod screenshot;
mod toolbar;
mod windows;
#[cfg(test)]
mod tests;

use central::{HeatmapCache, NoiseView, ProjectionCache};
pub use prefs::{FileLocations, Preferences};
use prefs::{
    default_around_s, default_atlas_min_region_channels, default_bregma_lambda_mm, default_mem_budget_pct,
    default_mem_reserve_mb, default_n_classify_chunks, default_spectrum_freq_max_hz, default_spectrum_freq_min_hz,
    default_spectrum_n_chunks, default_spectrum_time_end_s, default_spike_overlay_scale, default_spike_smoothing_sigma,
};
use psth::PsthState;
use screenshot::{Capture, MainScreenshot, ScreenshotState, ShotOverlays};

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};

use crate::data::{open_data, ChannelOrder, DisplayRow, Meta, RawData, ShankOrder};
use crate::preprocess::{PreprocConfig, SpatialFilter};
use crate::psth::{compute_psth, load_event_times, PsthParams, PsthResult, EventLayout};
use crate::jobs::{BackgroundJob, Outcome};
use crate::render::{build_heatmap_into, build_psth_heatmap_into};
use crate::worker::{
    request_shutdown, spawn_worker, Plan, SharedCancel, SharedWorkerState, ViewWindow, WorkerState,
};

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ColorMode {
    Percentile,
    Voltage,
}

use crate::colormap::ColorMapChoice;

/// Spawn the native picker for an event-times file on a background thread (same
/// rationale as the main file picker: never block the egui event loop).
pub(crate) fn spawn_event_picker(dir: Option<PathBuf>) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut dlg = rfd::FileDialog::new()
            .add_filter("Event times", &["csv", "txt", "tsv", "dat"])
            .add_filter("All files", &["*"]);
        if let Some(d) = dir {
            dlg = dlg.set_directory(d);
        }
        let _ = tx.send(dlg.pick_file());
    });
    rx
}

/// The preprocessing steps `cfg` applies, in order, for display.
fn preprocessing_summary(cfg: &PreprocConfig) -> String {
    let mut steps = Vec::new();
    if cfg.phase_shift {
        steps.push("Phase shift".to_string());
    }
    if cfg.dc_removal {
        steps.push("DC removal".to_string());
    }
    if cfg.notch_enabled && !cfg.notches.is_empty() {
        steps.push(format!("Notch ({})", cfg.notches.len()));
    }
    if cfg.highpass {
        steps.push("Highpass".to_string());
    }
    match cfg.spatial_filter {
        SpatialFilter::Off => {}
        SpatialFilter::GlobalCmr => steps.push("Global CMR".to_string()),
        SpatialFilter::LocalCmr => steps.push("Local CMR".to_string()),
        SpatialFilter::Destripe => steps.push("Destripe".to_string()),
        SpatialFilter::Csd => {
            let p = &cfg.csd;
            if p.bandpass {
                steps.push(format!("Bandpass {}–{} Hz", p.low_hz, p.high_hz));
            }
            steps.push(if p.smoothing { format!("CSD (σ {} µm)", p.sigma_um) } else { "CSD".to_string() });
        }
    }
    steps.join(" · ")
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

/// Where the heatmap's rows are on screen: rows `first..=last` of `rows`, bottom to
/// top, filling `rect`.
#[derive(Clone, Copy)]
pub(crate) struct RowsOnScreen<'a> {
    pub rows: &'a [DisplayRow],
    pub first: usize,
    pub last: usize,
    pub rect: egui::Rect,
}

/// Index range (inclusive) of all display rows.
fn all_rows(display_rows: &[DisplayRow]) -> (usize, usize) {
    (0, display_rows.len().saturating_sub(1))
}

/// Rectangle zoom in the waveform view: the time window and voltage range from before
/// the first zoom, which Esc restores.
struct WaveformZoom {
    prev_start_s: f64,
    prev_dur_s: f64,
    prev_y_range_uv: f32,
}

/// Rectangle zoom (left-drag on the heatmap): a channel range plus the time view from
/// before the first zoom, which Esc restores.
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
// display ranges while CSD is on (mV/mm²)
const CSD_PCT_MIN: f32 = 80.0;
const CSD_SCALE_RANGE: (f32, f32) = (0.1, 1000.0);
const CSD_WAVEFORM_MIN_RANGE: f32 = 0.01;

/// Smallest drag (px, in each direction) that zooms; anything less stays a click.
const ZOOM_MIN_DRAG_PX: f32 = 4.0;

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

/// Tooltip of the "Outside of brain" preference.
const OUTSIDE_RULE_HELP: &str = "How channels outside the brain are found, from each channel's low-frequency similarity to its neighbours (xcor_lf; strongly negative above the brain surface). Only a run of such channels reaching the top of the shank is labelled.\n\nAdaptive (default): the threshold is taken where the smoothed similarity trend drops most steeply along the shank (IBL's optional 'adaptive' mode). Finds the brain surface even when the contrast is weak, e.g. in LFP recordings.\n\nFixed threshold: xcor_lf < -0.75, the default of IBL's ibldsp. Stricter; can miss the surface entirely.";

const ENTER_TARGET: &str = "npx_enter_target";
const ENTER_WINDOWS: &str = "npx_enter_windows";
const ENTER_BLOCKED: &str = "npx_enter_blocked";

/// Enter applies the main action of one tool window: the window holding the keyboard
/// focus, otherwise the topmost window (if it has a main action). Decided at the start of the frame, before widgets
/// move the focus (a text field gives it up on Enter); the windows that asked for Enter
/// last frame are the candidates. Nothing happens while a popup is open, a field
/// outside such a window has the focus, or a multi-line text field has it.
fn route_enter(ctx: &egui::Context) {
    use egui::{Id, LayerId};
    let windows: Vec<LayerId> = ctx.data_mut(|d| d.remove_temp(Id::new(ENTER_WINDOWS))).unwrap_or_default();
    let blocked = ctx.data_mut(|d| d.remove_temp::<bool>(Id::new(ENTER_BLOCKED))).unwrap_or(false);
    let pressed = ctx.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.is_none());
    let target = if pressed && !blocked && !ctx.memory(|m| m.any_popup_open()) {
        let focus_layer = ctx.memory(|m| m.focused()).and_then(|id| ctx.read_response(id)).map(|r| r.layer_id);
        match focus_layer {
            Some(l) => windows.contains(&l).then_some(l),
            // the topmost window, if it has a main action (windows are the Middle layers)
            None => ctx.memory(|m| {
                m.layer_ids()
                    .filter(|l| l.order == egui::Order::Middle && m.areas().visible_last_frame(l))
                    .last()
                    .filter(|l| windows.contains(l))
            }),
        }
    } else {
        None
    };
    ctx.data_mut(|d| match target {
        Some(l) => d.insert_temp(Id::new(ENTER_TARGET), l),
        None => d.remove::<LayerId>(Id::new(ENTER_TARGET)),
    });
}

/// Call next to a window's main button: true when Enter should trigger it this frame.
pub(crate) fn enter_applies(ui: &egui::Ui) -> bool {
    let layer = ui.layer_id();
    let id = egui::Id::new(ENTER_WINDOWS);
    ui.ctx().data_mut(|d| {
        let w = d.get_temp_mut_or_default::<Vec<egui::LayerId>>(id);
        if !w.contains(&layer) {
            w.push(layer);
        }
    });
    ui.ctx().data(|d| d.get_temp::<egui::LayerId>(egui::Id::new(ENTER_TARGET))) == Some(layer)
}

/// For a multi-line text field: while it has the focus, Enter types a new line instead.
pub(crate) fn enter_types_newline(resp: &egui::Response) {
    if resp.has_focus() {
        resp.ctx.data_mut(|d| d.insert_temp(egui::Id::new(ENTER_BLOCKED), true));
    }
}

/// Options CSD needs: one row per depth, and its bandpass in place of the 300 Hz highpass.
fn enforce_csd_options(c: &mut PreprocConfig) {
    c.avg_depths = true;
    if c.csd.bandpass {
        c.highpass = false;
    }
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
    show_file_locations: bool,
    file_locations: FileLocations,
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
    /// the colour / waveform values of the unit system not shown now (CSD mV/mm² vs µV);
    /// swapped with the active ones when CSD switches on or off
    display_stash: crate::settings::CsdDisplay,
    display_is_csd: bool,
    /// colour-scale percentile cache: (window data, from, to, percentile) → value
    pct_cache: Option<((u64, usize, usize, u32), f32)>,
    /// rectangle zoom in the waveform view (left-drag); Esc returns to `WaveformZoom`'s
    /// view before closing the waveform view
    waveform_zoom: Option<WaveformZoom>,

    // channel removal
    show_remove_channels: bool,
    remove_channels_text: String,
    remove_channels_error: Option<String>,
    remove_channels_pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,

    /// channel classification: labels, overlay and the background run
    classify: crate::classify_ui::ClassifyUi,
    /// heatmap pixel columns show their extreme sample instead of their mean
    peak_pooling: bool,

    // noise suppression (visual filter on the view only); the window edits the draft,
    // Apply copies it to `noise`
    noise_open: bool,
    /// CSD window: the parameters being edited (applied with Apply / Enter)
    csd_edit: Option<crate::preprocess::CsdParams>,
    /// "Avg adjacent chans" and the 300 Hz HP as they were before CSD switched them
    csd_restore: Option<(bool, bool)>,
    /// "Notch filters" section of the noise window (scan + notch list)
    notch_panel: crate::notch_ui::NotchPanel,
    noise: crate::noise::NoiseSuppression,
    noise_draft: crate::noise::NoiseSuppression,
    /// filtered copy of the view (plus margin) and what it was computed from
    noise_view: Option<NoiseView>,

    /// power spectrum: settings window, computation and the strip beside the heatmap
    spectrum: crate::spectrum_ui::SpectrumUi,

    // async worker
    worker_state: SharedWorkerState,
    worker_cancel: SharedCancel,
    around_s: f64,
    background_fill: bool,
    mem_budget_pct: f32,
    mem_reserve_mb: f64,
    /// contiguous copy of the preprocessed blocks around the view
    view_window: Option<ViewWindow>,
    /// last scroll direction (+1 / -1) and the view position it was taken from
    scroll_dir: i32,
    last_view_first: usize,
    _worker_handle: std::thread::JoinHandle<()>,

    /// the heatmap texture and what it was drawn from
    heatmap: HeatmapCache,

    // rectangle zoom
    zoom: Option<Zoom>,
    /// screen position where the current zoom drag started
    zoom_drag_start: Option<egui::Pos2>,

    // UI state
    pub file_dialog_request: bool,
    /// like `file_dialog_request`, for the folder picker ("Open Zarr folder…")
    pub folder_dialog_request: bool,
    /// set by the toolbar's "Recent files" menu; polled by MainApp to switch recordings
    pub open_recent_request: Option<PathBuf>,
    recent_files: Vec<PathBuf>,
    /// firing-rate overlay values and what they were computed from
    projection: ProjectionCache,

    psth: PsthState,
    atlas: crate::atlas_ui::AtlasUi,
    events: crate::events::EventsState,
    spikes: crate::spikes_ui::SpikesState,
    /// event-file format text shared by the PSTH and Events windows
    event_layout_text: String,
    /// event file last loaded in the Events or PSTH window (saved per recording)
    event_file: Option<PathBuf>,

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
    /// "Export preprocessed data" window and a running export
    export_ui: crate::export_ui::ExportUi,
}

/// How long the settings must have differed from the file before it is rewritten,
/// so scrolling or dragging a value doesn't write on every frame.
const SETTINGS_SAVE_DELAY: std::time::Duration = std::time::Duration::from_secs(1);

impl NPXplorerApp {
    pub fn new(ctx: &egui::Context, bin_path: PathBuf) -> anyhow::Result<Self> {
        // a Zarr recording is its folder, whichever file in it was picked
        let bin_path = crate::zarr::root_of(&bin_path).unwrap_or(bin_path);
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
            csd: Default::default(),
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
        let mut around_s = default_around_s();
        let mut background_fill = false;
        let mut mem_budget_pct = default_mem_budget_pct();
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
        let mut file_locations = FileLocations::default();
        let mut spike_filter = crate::spikes::SpikeSettings::default();

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
            around_s = p.around_s;
            background_fill = p.background_fill;
            mem_budget_pct = p.mem_budget_pct;
            mem_reserve_mb = p.mem_reserve_mb;
            recent_files = p.recent_files.iter().map(PathBuf::from).collect();
            n_classify_chunks = p.n_classify_chunks;
            classify_outside_rule = p.classify_outside_rule;
            peak_pooling = p.peak_pooling;
            atlas_dir = p.atlas_dir;
            file_locations = p.file_locations;
            spike_filter = p.spike_filter;
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

        let event_settings =
            crate::settings::overlay(&crate::settings::EventSettings::default(), settings_file.get("events"));
        let spikes = crate::spikes_ui::SpikesState::new(
            crate::settings::overlay(&spike_filter, settings_file.get("spikes")),
            FileLocations::resolve(&file_locations.spikes, &bin_path),
        );
        let export_settings =
            crate::settings::overlay(&crate::export::ExportSettings::default(), settings_file.get("export"));
        let export_ui = crate::export_ui::ExportUi::new(export_settings, meta.n_samples as f64 / meta.sample_rate);
        let atlas_default = crate::settings::AtlasSettings {
            insertion: crate::atlas::Insertion {
                bregma_lambda_mm,
                tip_offset_um: crate::probe::spec_for_type(meta.im_dat_prb_type)
                    .map_or(crate::atlas::DEFAULT_TIP_OFFSET_UM, |s| s.tip_length_um),
                ..Default::default()
            },
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

        let shared: SharedWorkerState = Arc::new((Mutex::new(WorkerState::new()), Condvar::new()));
        let cancel: SharedCancel = Arc::new(AtomicBool::new(false));
        // the first plan is posted by the first frame, once the settings are applied
        let handle = spawn_worker(
            Arc::clone(&raw),
            Arc::clone(&meta),
            Arc::clone(&shared),
            Arc::clone(&cancel),
            {
                let ctx = ctx.clone();
                move || ctx.request_repaint()
            },
        );

        let is_compressed = bin_path.extension().and_then(|s| s.to_str()) == Some("cbin");

        let meta_for_atlas = Arc::clone(&meta);
        let psth_total_s = meta.n_samples as f64 / meta.sample_rate;
        let mut psth = PsthState::new(psth_total_s);
        psth.restore(event_settings.event_file.as_deref(), &event_settings.psth);
        let remove_channels_text = crate::channel_remove::format_channel_list(
            &preproc_cfg.removed_channels,
            &meta.channel_ids,
        );
        let notch_panel = crate::notch_ui::NotchPanel::new(&[], Default::default());

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
            scroll_speed_fine: true,
            color_mode,
            color_pct,
            color_uv,
            color_pct_str: format!("{:.2}", color_pct),
            color_uv_str: format!("{:.0}", color_uv),
            colormap_choice,
            show_preferences: false,
            show_file_locations: false,
            file_locations,
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
            display_stash: Default::default(),
            display_is_csd: false,
            pct_cache: None,
            waveform_zoom: None,
            show_remove_channels: false,
            remove_channels_text,
            remove_channels_error: None,
            remove_channels_pick_rx: None,
            classify: crate::classify_ui::ClassifyUi::new(n_classify_chunks, classify_outside_rule),
            peak_pooling,
            noise_open: false,
            csd_edit: None,
            csd_restore: None,
            notch_panel,
            noise_draft: noise.clone(),
            noise,
            noise_view: None,
            spectrum: crate::spectrum_ui::SpectrumUi::from_settings(&crate::settings::SpectrumSettings {
                time_scope: spectrum_time_scope,
                source: spectrum_source,
                scaling: spectrum_scaling,
                normalization: spectrum_normalization,
                n_chunks: spectrum_n_chunks,
                freq_restrict: spectrum_freq_restrict,
                freq_min_hz: spectrum_freq_min_hz,
                freq_max_hz: spectrum_freq_max_hz,
                time_restrict: spectrum_time_restrict,
                time_start_s: spectrum_time_start_s,
                time_end_s: spectrum_time_end_s,
                show_overlay: false,
            }),
            worker_state: shared,
            worker_cancel: cancel,
            around_s,
            background_fill,
            mem_budget_pct,
            mem_reserve_mb,
            view_window: None,
            scroll_dir: 1,
            last_view_first: 0,
            _worker_handle: handle,
            heatmap: HeatmapCache::default(),
            zoom: None,
            zoom_drag_start: None,
            file_dialog_request: false,
            folder_dialog_request: false,
            open_recent_request: None,
            recent_files,
            projection: ProjectionCache::default(),
            psth,
            atlas: crate::atlas_ui::AtlasUi::new(&meta_for_atlas, atlas_dir, atlas_settings, atlas_min_region_channels),
            events: crate::events::EventsState::new(&event_settings.window, event_settings.event_file.as_deref()),
            spikes,
            event_layout_text: event_settings.layout.clone().unwrap_or_else(crate::psth::default_layout_text),
            event_file: event_settings.event_file.clone(),
            band,
            settings_saved: None,
            settings_changed_at: None,
            settings_error: None,
            plot_rect: None,
            screenshot: ScreenshotState::default(),
            capture: None,
            export_ui,
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

    /// Load the atlas if its overlay was shown last time.
    /// Preprocessing starts with the first frame. Writes the settings file, so a
    /// recording opened once keeps its settings even when the preferences change later.
    fn start(&mut self, ctx: &egui::Context) {
        let meta = Arc::clone(&self.meta);
        self.atlas.register_on_open(ctx, &meta);
        self.load_on_start(ctx);
        self.save_settings();
    }

    /// Events and spike-sorted data from their default locations (Preferences → Default
    /// file locations). A recording's own event file takes precedence over the default.
    fn load_on_start(&mut self, ctx: &egui::Context) {
        let loc = self.file_locations.clone();
        let default_events = FileLocations::resolve(&loc.events, &self.bin_path).filter(|p| p.is_file());
        if let Some(p) = &default_events {
            self.events.suggest_path(p);
        }
        if loc.load_events_on_start {
            let saved = self.event_file.clone().filter(|p| p.is_file());
            if let Some(p) = saved.or(default_events) {
                if self.events.load_path(&p, &self.event_layout_text) {
                    self.event_file = Some(p);
                } else {
                    self.events.open = true;
                }
            }
        }
        if loc.load_spikes_on_start {
            let meta = Arc::clone(&self.meta);
            self.spikes.load_on_start(ctx, &meta);
        }
    }
}

impl Drop for NPXplorerApp {
    /// Stop the worker thread when the recording is closed, so its thread pool, the
    /// preprocessed buffer and the mapped data file are released. Unsaved settings
    /// changes are written first.
    fn drop(&mut self) {
        self.flush_settings();
        request_shutdown(&self.worker_state, &self.worker_cancel);
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
        let (raw, csd) = self.display_sets();
        BandSettings {
            view_start_s,
            view_dur_s,
            scroll_fine: self.scroll_speed_fine,
            color_mode: self.color_mode,
            color_pct: raw.color_pct,
            color_uv: raw.color_scale,
            colormap: self.colormap_choice.clone(),
            peak_pooling: self.peak_pooling,
            waveform_y_range_uv: raw.waveform_range,
            csd_display: csd,
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
                csd: c.csd.clone(),
            },
            firing_rate: FiringRateSettings {
                show: self.show_firing_rate_overlay,
                threshold_uv: self.spike_threshold,
                overlay_scale: self.spike_overlay_scale,
                smoothing_sigma: self.spike_smoothing_sigma,
            },
            classification: self.classify.settings(),
            spectrum: self.spectrum.settings(),
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
        // the µV values are active now; the next frame swaps in the CSD ones if needed
        let d = s.csd_display;
        self.display_stash = crate::settings::CsdDisplay {
            color_pct: d.color_pct.clamp(CSD_PCT_MIN, 100.0),
            color_scale: d.color_scale.clamp(CSD_SCALE_RANGE.0, CSD_SCALE_RANGE.1),
            waveform_range: d.waveform_range.clamp(CSD_WAVEFORM_MIN_RANGE, 2000.0),
        };
        self.display_is_csd = false;
        self.waveform_zoom = None;

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
        c.csd = s.preproc.csd;
        if c.spatial_filter == SpatialFilter::Csd {
            enforce_csd_options(c);
        }
        self.notch_panel = crate::notch_ui::NotchPanel::new(&self.preproc_cfg.notches, s.notch_scan);

        self.show_firing_rate_overlay = s.firing_rate.show;
        self.spike_threshold = s.firing_rate.threshold_uv;
        self.spike_overlay_scale = s.firing_rate.overlay_scale;
        self.spike_smoothing_sigma = s.firing_rate.smoothing_sigma;
        self.classify.apply_settings(&s.classification);

        self.spectrum.apply_settings(&s.spectrum);

        self.noise_draft = s.noise.clone();
        self.noise = s.noise;
    }

    /// Everything saved in the recording's settings file.
    fn recording_settings(&self) -> crate::settings::RecordingSettings {
        let mut s = crate::settings::RecordingSettings {
            removed_channels: Some(self.preproc_cfg.removed_channels.iter().copied().collect()),
            events: Some(crate::settings::EventSettings {
                event_file: self.event_file.clone(),
                layout: Some(self.event_layout_text.clone()),
                window: self.events.settings(),
                psth: self.psth.settings(),
            }),
            atlas: Some(self.atlas.settings()),
            spikes: Some(self.spikes.settings()),
            export: Some(self.export_ui.settings.clone()),
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
        if let Some(events) = &mut to_write.events {
            events.layout = events.layout.as_deref().and_then(crate::psth::layout_to_save);
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

    // -----------------------------------------------------------------------
    // Channel removal
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // IBL-style channel classification (dead/noisy/outside-of-brain/good)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Power spectrum overlay
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Screenshot of the plot area
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Export of the preprocessed recording
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Per-channel context menu (plain right-click, no Alt)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Single-channel waveform view
    // -----------------------------------------------------------------------

    /// Return to the waveform view from before the zoom; false if not zoomed.
    /// Display values (µV ones, CSD ones), whichever set is active.
    fn display_sets(&self) -> (crate::settings::CsdDisplay, crate::settings::CsdDisplay) {
        let active = crate::settings::CsdDisplay {
            color_pct: self.color_pct,
            color_scale: self.color_uv,
            waveform_range: self.waveform_zoom.as_ref().map_or(self.waveform_y_range_uv, |z| z.prev_y_range_uv),
        };
        if self.display_is_csd {
            (self.display_stash.clone(), active)
        } else {
            (active, self.display_stash.clone())
        }
    }

    /// Swap in the colour / waveform values of the unit system CSD needs (mV/mm²), or
    /// back to µV, when CSD has been switched since the last frame.
    fn sync_display_units(&mut self) {
        let csd = self.preproc_cfg.spatial_filter == SpatialFilter::Csd;
        if csd == self.display_is_csd {
            return;
        }
        self.leave_waveform_zoom();
        let s = &mut self.display_stash;
        std::mem::swap(&mut self.color_pct, &mut s.color_pct);
        std::mem::swap(&mut self.color_uv, &mut s.color_scale);
        std::mem::swap(&mut self.waveform_y_range_uv, &mut s.waveform_range);
        self.color_pct_str = format!("{:.2}", self.color_pct);
        self.color_uv_str = format!("{:.0}", self.color_uv);
        self.display_is_csd = csd;
    }

    fn unit(&self) -> &'static str {
        if self.display_is_csd { "mV/mm²" } else { "µV" }
    }

    fn waveform_min_range(&self) -> f32 {
        if self.display_is_csd { CSD_WAVEFORM_MIN_RANGE } else { WAVEFORM_MIN_RANGE_UV }
    }

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

    // -----------------------------------------------------------------------
    // UI panels
    // -----------------------------------------------------------------------

}

impl NPXplorerApp {

    // -----------------------------------------------------------------------
    // Central panel: the heatmap (or waveform view) and everything drawn on it
    // -----------------------------------------------------------------------

    pub fn update(&mut self, ctx: &egui::Context) {
        // a running export freezes the window: nothing else is drawn or computed
        let mut export_requests = Vec::new();
        let frozen = self.export_ui.draw_frozen(ctx, &mut export_requests);
        self.handle_export_requests(export_requests, ctx);
        if frozen {
            return;
        }
        self.poll_capture(ctx);
        // taking a screenshot: no windows, menus or hover effects over the plot
        let capturing = self.capture.is_some();
        self.atlas.no_hover = capturing;
        self.close_top_window_on_escape(ctx);
        route_enter(ctx);
        self.sync_display_units();
        self.poll_and_maybe_dispatch_psth(ctx);
        self.poll_remove_channels_picker(ctx);
        self.classify.poll(ctx);
        self.spectrum.poll(ctx);
        self.notch_panel.poll(ctx);
        self.atlas.poll(ctx);
        self.spikes.poll(ctx);
        if !capturing {
            let meta = Arc::clone(&self.meta);
            self.spikes.draw_window(ctx, &meta);
            self.draw_file_locations_window(ctx);
            self.draw_psth_window(ctx);
            self.draw_remove_channels_window(ctx);
            self.classify.draw_progress_window(ctx);
            self.spectrum.draw_error_window(ctx);
            if self.spectrum.draw_window(ctx, &self.raw, &self.meta, &self.preproc_cfg) {
                self.save_prefs();
            }
            self.draw_csd_window(ctx);
            self.draw_noise_window(ctx);
            self.draw_channel_context_menu(ctx);
            self.atlas
                .draw_window(ctx, &self.meta, &self.bin_path, &self.colormap_choice);
            self.atlas.draw_progress_window(ctx);
            let view = crate::events::TimeView {
                start_s: &mut self.view_start_s,
                dur_s: self.view_dur_s,
                total_s: self.meta.n_samples as f64 / self.meta.sample_rate,
            };
            self.events.draw_window(ctx, &mut self.event_layout_text, &self.bin_path, &mut self.event_file, view);
            self.draw_screenshot_window(ctx);
            self.draw_export_window(ctx);
        }
        self.autosave_settings(ctx);
        if self.atlas.take_prefs_dirty() {
            self.save_prefs();
        }

        self.draw_preferences_window(ctx);

        // mouse wheel and arrow keys move the view (Alt+wheel: the colour scale)
        let alt_held = ctx.input(|i| i.modifiers.alt);
        self.handle_view_input(ctx, alt_held);

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
            .show(ctx, |ui| self.draw_central_panel(ctx, ui, alt_held));

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
