use egui::{CentralPanel, TextureHandle, TextureOptions, TopBottomPanel, Ui, Vec2};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::data::{DisplayRow, Meta, RawData, open_data};
use crate::preprocess::{Filters, PreprocConfig, SpatialFilter};
use crate::psth::{PsthParams, PsthResult, compute_psth, resolve_layout, load_stim_times, default_layout_path};
use crate::render::{build_heatmap_into, build_psth_heatmap_into};
use crate::worker::{
    RequestKind, SharedCancel, SharedWorkerState, WorkerRequest, WorkerState, WorkerStatus,
    compute_half_window, spawn_worker,
};

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ColorMode {
    Percentile,
    Voltage,
}

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ColorMapChoice {
    YellowMagenta,
    RedBlue,
    OrangeBlue,
    IceFire,
    Vanimo,
    GreyScale,
}

fn default_initial_buffer_s() -> f64 { 30.0 }
fn default_extension_margin_s() -> f64 { 5.0 }
fn default_mem_pressure_pct() -> f32 { 15.0 }
fn default_mem_reserve_mb() -> f64 { 1500.0 }
fn default_spike_overlay_scale() -> f32 { 1.0 }
fn default_spike_smoothing_sigma() -> f32 { 1.5 }

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
    /// user-configurable multiplier on the firing-rate overlay's width scaling
    #[serde(default = "default_spike_overlay_scale")]
    pub spike_overlay_scale: f32,
    /// std. dev. (in channels/display rows) of the Gaussian used to smooth the
    /// firing-rate overlay across depth
    #[serde(default = "default_spike_smoothing_sigma")]
    pub spike_smoothing_sigma: f32,
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
    // staged settings (only committed to a recompute when "Apply settings" is pressed)
    ch_first: usize,
    ch_last: usize,
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
    computing: bool,
    apply_requested: bool,

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
    fn new(ch_first: usize, ch_last: usize, total_s: f64) -> Self {
        Self {
            open: false,
            stim_path: None,
            ch_first,
            ch_last,
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
            computing: false,
            apply_requested: false,
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

/// Spawn the native picker for a stimulus-times file on a background thread (same
/// rationale as the main file picker: never block the egui event loop).
fn spawn_stim_picker(dir: Option<PathBuf>) -> mpsc::Receiver<Option<PathBuf>> {
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

/// data-row index (into `result.data`) for a 1-based channel number, or None if the
/// channel is not among the computed display rows.
fn channel_row(result: &PsthResult, ch: usize) -> Option<usize> {
    result.display_rows.iter().find_map(|r| match r {
        DisplayRow::Data { data_idx, first_ch, .. } if *first_ch + 1 == ch => Some(*data_idx),
        _ => None,
    })
}

/// 1-based channel number under a heatmap y coordinate (ch_last at top, ch_first at
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
    ch_first: usize,
    ch_last: usize,

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
    spike_overlay_scale: f32,
    spike_smoothing_sigma: f32,

    // selected channels
    selected_channel_1: Option<usize>,
    selected_channel_2: Option<usize>,

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

    // smooth-scroll state
    waiting_since: Option<Instant>,
    last_requested_center: usize,

    // UI state
    pending_cfg_recompute: bool,
    pub file_dialog_request: bool,
    projection_sums: Vec<f32>,
    // spike projection cache keys
    proj_view_first: usize,
    proj_view_n: usize,
    proj_threshold: f32,
    proj_sigma: f32,
    proj_cfg: Option<PreprocConfig>,

    psth: PsthState,
}

impl NPXplorerApp {
    pub fn new(ctx: &egui::Context, bin_path: PathBuf) -> anyhow::Result<Self> {
        let meta = Arc::new(Meta::from_data_path(&bin_path)?);
        let (raw, _) = open_data(&bin_path, &meta)?;
        let raw = Arc::new(raw);
        let fs = meta.sample_rate;

        let prefs = Preferences::load();

        let mut preproc_cfg = PreprocConfig {
            dc_removal: true,
            phase_shift: false,
            highpass: true,
            spatial_filter: SpatialFilter::GlobalCmr,
            avg_depths: true,
            sample_rate: fs,
            im_dat_prb_type: meta.im_dat_prb_type,
        };

        let mut view_dur_s = 0.5;
        let mut color_mode = ColorMode::Percentile;
        let mut color_pct = 99.0;
        let mut color_uv = 120.0;
        let mut colormap_choice = ColorMapChoice::IceFire;
        let mut spike_threshold = -40.0;
        let mut spike_overlay_scale = default_spike_overlay_scale();
        let mut spike_smoothing_sigma = default_spike_smoothing_sigma();
        let mut initial_buffer_s = default_initial_buffer_s();
        let mut extension_margin_s = default_extension_margin_s();
        let mut mem_pressure_pct = default_mem_pressure_pct();
        let mut mem_reserve_mb = default_mem_reserve_mb();

        if let Some(p) = prefs {
            preproc_cfg = p.preproc_cfg;
            preproc_cfg.sample_rate = fs;
            preproc_cfg.im_dat_prb_type = meta.im_dat_prb_type;
            view_dur_s = p.view_dur_s;
            color_mode = p.color_mode;
            color_pct = p.color_pct;
            color_uv = p.color_uv;
            colormap_choice = p.colormap_choice;
            spike_threshold = p.spike_threshold;
            spike_overlay_scale = p.spike_overlay_scale;
            spike_smoothing_sigma = p.spike_smoothing_sigma;
            initial_buffer_s = p.initial_buffer_s;
            extension_margin_s = p.extension_margin_s;
            mem_pressure_pct = p.mem_pressure_pct;
            mem_reserve_mb = p.mem_reserve_mb;
        }

        // defensively re-clamp in case prefs were saved on a machine with more RAM,
        // or with an initial_buffer_s/view_dur_s combination that no longer satisfies
        // the no-oscillation bound
        let n_data_rows = meta.build_display_rows(preproc_cfg.avg_depths)
            .iter().filter(|r| matches!(r, DisplayRow::Data { .. })).count();
        initial_buffer_s = initial_buffer_s.min(max_feasible_buffer_s(n_data_rows, fs, mem_reserve_mb));
        extension_margin_s = extension_margin_s.min(max_extension_margin_s(initial_buffer_s, view_dur_s));

        let filters = Arc::new(Mutex::new(Filters::new(&preproc_cfg)));
        let shared: SharedWorkerState = Arc::new((Mutex::new(WorkerState::new()), Condvar::new()));
        let cancel: SharedCancel = Arc::new(AtomicBool::new(false));

        let half_window = compute_half_window(initial_buffer_s, fs);

        let handle = spawn_worker(
            Arc::clone(&raw),
            Arc::clone(&meta),
            Arc::clone(&filters),
            Arc::clone(&shared),
            Arc::clone(&cancel),
            ctx.clone(),
        );

        // send initial request
        {
            let (lock, cvar) = &*shared;
            lock.lock().unwrap().request = Some(WorkerRequest {
                kind: RequestKind::Full {
                    center_sample: half_window,
                    half_window,
                },
                cfg: preproc_cfg.clone(),
            });
            cvar.notify_one();
        }

        let is_compressed = bin_path.extension().and_then(|s| s.to_str()) == Some("cbin");

        let n_ap = meta.n_ap_chans;
        let psth_total_s = meta.n_samples as f64 / meta.sample_rate;
        Ok(Self {
            bin_path,
            meta,
            raw: Arc::clone(&raw),
            is_compressed,
            view_start_s: 0.0,
            view_dur_s,
            window_dur_str: format!("{:.3}", view_dur_s),
            jump_str: "0.000".to_string(),
            ch_first: 0,
            ch_last: n_ap.saturating_sub(1),
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
            spike_overlay_scale,
            spike_smoothing_sigma,
            selected_channel_1: None,
            selected_channel_2: None,
            worker_state: shared,
            worker_cancel: cancel,
            worker_half_window: half_window,
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
            waiting_since: None,
            last_requested_center: 0,
            pending_cfg_recompute: false,
            file_dialog_request: false,
            projection_sums: Vec::new(),
            proj_view_first: usize::MAX,
            proj_view_n: 0,
            proj_threshold: 0.0,
            proj_sigma: 0.0,
            proj_cfg: None,
            psth: PsthState::new(0, n_ap.saturating_sub(1), psth_total_s),
        })
    }

    pub fn save_prefs(&self) {
        let last_dir = self.bin_path.parent().map(|p| p.to_string_lossy().to_string());
        let prefs = Preferences {
            preproc_cfg: self.preproc_cfg.clone(),
            view_dur_s: self.view_dur_s,
            color_mode: self.color_mode.clone(),
            color_pct: self.color_pct,
            color_uv: self.color_uv,
            colormap_choice: self.colormap_choice.clone(),
            spike_threshold: self.spike_threshold,
            spike_overlay_scale: self.spike_overlay_scale,
            spike_smoothing_sigma: self.spike_smoothing_sigma,
            initial_buffer_s: self.initial_buffer_s,
            extension_margin_s: self.extension_margin_s,
            mem_pressure_pct: self.mem_pressure_pct,
            mem_reserve_mb: self.mem_reserve_mb,
            last_dir,
        };
        prefs.save();
    }

    fn request_recompute(&mut self) {
        let fs = self.meta.sample_rate;
        // use same formula as update() to avoid off-by-one from float rounding
        let view_first = (self.view_start_s * fs) as usize;
        let view_n = (self.view_dur_s * fs) as usize;
        let center = view_first + view_n / 2;
        file_log!("UI: request_recompute center={} view_first={} view_n={} cfg={:?}", center, view_first, view_n, self.preproc_cfg);
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
    // PSTH
    // -----------------------------------------------------------------------

    /// Poll the PSTH picker/compute/export channels and dispatch a computation only
    /// when the user has pressed "Apply settings" (or just picked a file).
    fn poll_and_maybe_dispatch_psth(&mut self, ctx: &egui::Context) {
        // stimulus-file picker
        if let Some(rx) = &self.psth.pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.psth.pick_rx = None;
                    if let Some(path) = picked {
                        self.psth.stim_path = Some(path);
                        self.psth.open = true;
                        self.psth.apply_requested = true; // auto-compute on first load
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

        // cancel any in-flight compute and install a fresh cancel flag
        self.psth.cancel.store(true, Ordering::Relaxed);
        let cancel = Arc::new(AtomicBool::new(false));
        self.psth.cancel = Arc::clone(&cancel);

        let (tx, rx) = mpsc::channel();
        self.psth.compute_rx = Some(rx);
        self.psth.computing = true;
        self.psth.error = None;

        let raw = Arc::clone(&self.raw);
        let meta = Arc::clone(&self.meta);
        let cfg = self.preproc_cfg.clone();
        let params = PsthParams {
            ch_first: self.psth.ch_first,
            ch_last: self.psth.ch_last,
            start_ms: self.psth.start_ms,
            end_ms: self.psth.end_ms,
        };
        let (t_start, t_end) = (self.psth.stim_t_start, self.psth.stim_t_end);
        let default_layout = default_layout_path();
        let ctx = ctx.clone();

        std::thread::spawn(move || {
            let res = (|| -> Result<PsthResult, String> {
                let layout = resolve_layout(&stim_path, &default_layout).map_err(|e| e.to_string())?;
                let all = load_stim_times(&stim_path, &layout).map_err(|e| e.to_string())?;
                let times: Vec<f64> = all.into_iter().filter(|&t| t >= t_start && t <= t_end).collect();
                if times.is_empty() {
                    return Err(format!(
                        "no stimuli fall within the selected time range {:.3}–{:.3} s.",
                        t_start, t_end
                    ));
                }
                compute_psth(&raw, &meta, &cfg, &times, &params, &cancel).map_err(|e| e.to_string())
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
                        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
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
    // resolve channel slider values → indices into display_rows
    // -----------------------------------------------------------------------

    /// Find the display_row index range corresponding to ch_first..ch_last.
    fn visible_row_range(&self, display_rows: &[DisplayRow]) -> (usize, usize) {
        let mut first_idx = 0usize;
        let mut last_idx = display_rows.len().saturating_sub(1);
        let mut found_first = false;
        for (i, row) in display_rows.iter().enumerate() {
            if let DisplayRow::Data { first_ch, .. } = row {
                if !found_first && *first_ch >= self.ch_first {
                    first_idx = i;
                    found_first = true;
                }
                if *first_ch <= self.ch_last {
                    last_idx = i;
                }
            }
        }
        (first_idx, last_idx)
    }

    // -----------------------------------------------------------------------
    // UI panels
    // -----------------------------------------------------------------------

    fn draw_toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open").clicked() {
                    self.file_dialog_request = true;
                    ui.close_menu();
                }
            });
            ui.separator();
            ui.label(format!(
                "{}",
                self.bin_path.file_name().unwrap_or_default().to_string_lossy()
            ));
            ui.separator();
            // window duration text field — stored separately to avoid overwrite each frame
            ui.label("Window:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.window_dur_str)
                    .desired_width(55.0)
                    .hint_text("s"),
            );
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

            if ui.radio_value(&mut self.color_mode, ColorMode::Percentile, "%ile").changed()
                || ui.radio_value(&mut self.color_mode, ColorMode::Voltage, "±µV").changed() {
                self.heatmap_texture = None;
            }

            if self.color_mode == ColorMode::Percentile {
                if ui.add(
                    egui::Slider::new(&mut self.color_pct, 95.0..=100.0)
                        .step_by(0.1)
                        .suffix("%")
                ).changed() {
                    self.color_pct_str = format!("{:.2}", self.color_pct);
                    self.heatmap_texture = None;
                }
            } else {
                if ui.add(
                    egui::Slider::new(&mut self.color_uv, 10.0..=300.0)
                        .integer()
                        .suffix("µV")
                ).changed() {
                    self.color_uv_str = format!("{:.0}", self.color_uv);
                    self.heatmap_texture = None;
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Preferences").clicked() {
                    self.show_preferences = !self.show_preferences;
                }
                if ui.button("PSTH").clicked() {
                    self.psth.open = true;
                    if self.psth.pick_rx.is_none() {
                        self.psth.pick_rx = Some(spawn_stim_picker(
                            self.bin_path.parent().map(|p| p.to_path_buf()),
                        ));
                    }
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

            let hp_enabled = self.preproc_cfg.spatial_filter != SpatialFilter::Destripe;
            let mut hp = self.preproc_cfg.highpass;
            if ui.add_enabled(hp_enabled, egui::Checkbox::new(&mut hp, "300 Hz HP"))
                .on_disabled_hover_text("Included in destripe")
                .changed()
            {
                self.preproc_cfg.highpass = hp;
                self.pending_cfg_recompute = true;
            }

            ui.separator();
            ui.label("Spatial:");

            let mut spatial = self.preproc_cfg.spatial_filter;
            let changed = ui.radio_value(&mut spatial, SpatialFilter::Off, "Off").changed()
                || ui.radio_value(&mut spatial, SpatialFilter::GlobalCmr, "Global CMR").changed()
                || ui.radio_value(&mut spatial, SpatialFilter::LocalCmr, "Local CMR").changed()
                || ui.radio_value(&mut spatial, SpatialFilter::Destripe, "Destripe").changed();
            
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
            if ui.checkbox(&mut avg, "Avg depths").changed() {
                self.preproc_cfg.avg_depths = avg;
                self.heatmap_texture = None;
                self.pending_cfg_recompute = true;
            }

            // computing spinner
            let _status = {
                let (lock, _) = &*self.worker_state;
                lock.lock().unwrap().status.clone()
            };
        });
    }

    fn draw_channel_controls(&mut self, ui: &mut Ui) {
        let n_ap = self.meta.n_ap_chans;
        ui.horizontal(|ui| {
            ui.label("Channels:");
            let mut cf = self.ch_first + 1;
            let mut cl = self.ch_last + 1;
            let mut changed = false;
            changed |= ui.add(egui::Slider::new(&mut cf, 1..=n_ap).text("First")).changed();
            changed |= ui.add(egui::Slider::new(&mut cl, 1..=n_ap).text("Last")).changed();
            if changed {
                self.ch_first = (cf - 1).min(cl - 1);
                self.ch_last = (cl - 1).max(cf - 1);
                self.heatmap_texture = None;
            }

            ui.separator();
            ui.label("Jump to (s):");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.jump_str).desired_width(70.0));
            if resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                if let Ok(t) = self.jump_str.trim().parse::<f64>() {
                    let max_t = self.meta.n_samples as f64 / self.meta.sample_rate - self.view_dur_s;
                    self.view_start_s = t.clamp(0.0, max_t.max(0.0));
                }
                self.jump_str = format!("{:.3}", self.view_start_s);
            }
            // keep jump field synced when not being edited
            if !resp.has_focus() {
                self.jump_str = format!("{:.3}", self.view_start_s);
            }

            let display_rows_arc = {
                let (lock, _) = &*self.worker_state;
                lock.lock().unwrap().buffer.as_ref().map(|b| Arc::clone(&b.display_rows))
            };

            let mut ch1_visible = false;
            let mut ch2_visible = false;

            if let Some(rows) = &display_rows_arc {
                let (first_row, last_row) = self.visible_row_range(rows);
                for r in first_row..=last_row {
                    if let DisplayRow::Data { first_ch, .. } = &rows[r] {
                        let ch = *first_ch + 1;
                        if Some(ch) == self.selected_channel_1 { ch1_visible = true; }
                        if Some(ch) == self.selected_channel_2 { ch2_visible = true; }
                    }
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let (Some(ch1), Some(ch2)) = (self.selected_channel_1, self.selected_channel_2) {
                    if ch1 > 0 && ch1 <= self.meta.channel_geom.len() && ch2 > 0 && ch2 <= self.meta.channel_geom.len() {
                        let y1 = self.meta.channel_geom[ch1 - 1].y_um;
                        let y2 = self.meta.channel_geom[ch2 - 1].y_um;
                        let dist = (y1 - y2).abs();
                        ui.label(
                            egui::RichText::new(format!("Δ = {:.1} µm", dist))
                                .strong()
                                .color(egui::Color32::WHITE)
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
                            egui::RichText::new(format!("Selected Channel 2: {}", ch2))
                                .color(egui::Color32::from_rgb(0xff, 0xb6, 0x17))
                        );
                    }
                }
                
                if ch1_visible {
                    if let Some(ch1) = self.selected_channel_1 {
                        if ui.button("✖").clicked() {
                            self.selected_channel_1 = None;
                        }
                        let text = if self.selected_channel_2.is_some() {
                            format!("Selected Channel 1: {}", ch1)
                        } else {
                            format!("Selected Channel: {}", ch1)
                        };
                        ui.label(
                            egui::RichText::new(text)
                                .color(egui::Color32::WHITE)
                        );
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
                ui.colored_label(egui::Color32::YELLOW, "Reading from compressed .cbin is slower");
            }
        });
    }

    fn draw_nav_bar(&mut self, ui: &mut Ui) {
        let total_s = self.meta.n_samples as f64 / self.meta.sample_rate;

        let (response, painter) = ui.allocate_painter(
            Vec2::new(ui.available_width(), 32.0),
            egui::Sense::click(),
        );
        let rect = response.rect;
        let w = rect.width();

        painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(0x18, 0x1a, 0x1f));

        let [ar, ag, ab] = crate::render::colormap_accent(&self.colormap_choice);

        // preprocessed-buffer extent, drawn first so it sits beneath the view marker
        let buf_extent = {
            let (lock, _) = &*self.worker_state;
            lock.lock().unwrap().buffer.as_ref().map(|b| (b.first_sample, b.n_samp))
        };
        if let Some((first, n_samp)) = buf_extent {
            let buf_frac = (first as f64 / self.meta.sample_rate / total_s) as f32;
            let buf_w_frac = (n_samp as f64 / self.meta.sample_rate / total_s) as f32;
            let buf_rect = egui::Rect::from_min_size(
                egui::pos2(rect.min.x + w * buf_frac, rect.min.y),
                Vec2::new((w * buf_w_frac).max(2.0), rect.height()),
            );
            painter.rect_filled(buf_rect, 1.0, egui::Color32::from_rgba_unmultiplied(ar, ag, ab, crate::render::BUFFER_EXTENT_ALPHA));
        }

        // view marker
        let view_frac = (self.view_start_s / total_s) as f32;
        let view_w_frac = (self.view_dur_s / total_s) as f32;
        let view_rect = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + w * view_frac, rect.min.y),
            Vec2::new((w * view_w_frac).max(2.0), rect.height()),
        );
        painter.rect_filled(view_rect, 1.0, egui::Color32::from_rgba_unmultiplied(ar, ag, ab, crate::render::VIEW_MARKER_ALPHA));

        // time labels
        let n_labels = 8;
        for i in 0..=n_labels {
            let frac = i as f32 / n_labels as f32;
            let t = frac as f64 * total_s;
            let x = rect.min.x + w * frac;
            painter.line_segment(
                [egui::pos2(x, rect.max.y - 6.0), egui::pos2(x, rect.max.y)],
                egui::Stroke::new(1.0, egui::Color32::GRAY),
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
            crate::render::C_ZERO[0], crate::render::C_ZERO[1], crate::render::C_ZERO[2],
        );
        let [ar, ag, ab] = crate::render::colormap_accent(&self.colormap_choice);
        let accent = egui::Color32::from_rgb(ar, ag, ab);
        let accent_50 = egui::Color32::from_rgba_unmultiplied(ar, ag, ab, 128);
        let cmap = self.colormap_choice.clone();
        let n_ap = self.meta.n_ap_chans;
        let total_s = self.psth.total_s;

        // size the window to 2/3 of the main window and center it on first open
        let screen = ctx.screen_rect();
        let win_size = screen.size() * (2.0 / 3.0);
        let win_pos = screen.center() - (win_size * 0.5);

        let mut open = self.psth.open;
        let result = self.psth.result.clone();

        egui::Window::new(
            egui::RichText::new("Peri-Stimulus Time Histogram").color(egui::Color32::WHITE),
        )
            .open(&mut open)
            .default_size(win_size)
            .default_pos(win_pos)
            .frame(egui::Frame::new().fill(c_zero).inner_margin(8.0)
                .stroke(egui::Stroke::new(2.0, accent_50)))
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
                    let name = self.psth.stim_path.as_ref()
                        .and_then(|p| p.file_name())
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "(no file)".into());
                    ui.label(name);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let can_export = self.psth.result.is_some()
                            && self.psth.export_pick_rx.is_none()
                            && !self.psth.export_pending;
                        if ui.add_enabled(can_export, egui::Button::new("Export PNG…")).clicked() {
                            self.psth.export_pick_rx = Some(spawn_png_saver(
                                self.bin_path.parent().map(|p| p.to_path_buf()),
                                self.default_psth_png_name(),
                            ));
                        }
                    });
                });

                ui.horizontal(|ui| {
                    ui.label("Channels:");
                    let mut cf = self.psth.ch_first + 1;
                    let mut cl = self.psth.ch_last + 1;
                    let mut changed = false;
                    changed |= ui.add(egui::Slider::new(&mut cf, 1..=n_ap).text("First")).changed();
                    changed |= ui.add(egui::Slider::new(&mut cl, 1..=n_ap).text("Last")).changed();
                    if changed {
                        self.psth.ch_first = (cf - 1).min(cl - 1);
                        self.psth.ch_last = (cl - 1).max(cf - 1);
                    }

                    ui.separator();

                    // stimulus time-range selector (seconds)
                    ui.label("Stim time (s):");
                    let r1 = ui.add(egui::TextEdit::singleline(&mut self.psth.stim_t_start_str)
                        .desired_width(70.0).hint_text("start"));
                    if r1.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if let Ok(v) = self.psth.stim_t_start_str.trim().parse::<f64>() {
                            self.psth.stim_t_start = v.clamp(0.0, total_s);
                        }
                        self.psth.stim_t_start_str = format!("{:.3}", self.psth.stim_t_start);
                    }
                    ui.label("to");
                    let r2 = ui.add(egui::TextEdit::singleline(&mut self.psth.stim_t_end_str)
                        .desired_width(70.0).hint_text("end"));
                    if r2.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if let Ok(v) = self.psth.stim_t_end_str.trim().parse::<f64>() {
                            self.psth.stim_t_end = v.clamp(0.0, total_s);
                        }
                        self.psth.stim_t_end_str = format!("{:.3}", self.psth.stim_t_end);
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Window (ms):");
                    let r1 = ui.add(egui::TextEdit::singleline(&mut self.psth.start_ms_str)
                        .desired_width(55.0).hint_text("start"));
                    if r1.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if let Ok(v) = self.psth.start_ms_str.trim().parse::<f64>() {
                            self.psth.start_ms = v;
                        }
                        self.psth.start_ms_str = format!("{}", self.psth.start_ms);
                    }
                    ui.label("to");
                    let r2 = ui.add(egui::TextEdit::singleline(&mut self.psth.end_ms_str)
                        .desired_width(55.0).hint_text("end"));
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
                        egui::Button::new(egui::RichText::new("Apply settings").color(c_zero))
                            .fill(accent_50),
                    );
                    if apply.clicked() {
                        self.psth.apply_requested = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Color scale:");
                    let mut changed = false;
                    changed |= ui.radio_value(&mut self.psth.color_mode, ColorMode::Percentile, "%ile").changed();
                    changed |= ui.radio_value(&mut self.psth.color_mode, ColorMode::Voltage, "±µV").changed();
                    if self.psth.color_mode == ColorMode::Percentile {
                        changed |= ui.add(egui::Slider::new(&mut self.psth.color_pct, 95.0..=100.0)
                            .step_by(0.1).suffix("%")).changed();
                    } else {
                        changed |= ui.add(egui::Slider::new(&mut self.psth.color_uv, 1.0..=200.0)
                            .suffix("µV")).changed();
                    }
                    if changed {
                        self.psth.tex_dirty = true;
                    }
                });

                if self.psth.computing {
                    ui.horizontal(|ui| { ui.spinner(); ui.label("Computing…"); });
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
                } else if let Some(err) = &self.psth.error {
                    ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), err);
                } else if self.psth.result.is_some() {
                    ui.horizontal(|ui| {
                        let base = if self.psth.n_skipped > 0 {
                            format!("{} stimuli averaged ({} skipped near edges)",
                                self.psth.n_used, self.psth.n_skipped)
                        } else {
                            format!("{} stimuli averaged", self.psth.n_used)
                        };
                        ui.label(base);
                        ui.label("·  left-click / right-click the heatmap to plot a channel:");
                        if let Some(c) = self.psth.sel_ch1 {
                            ui.colored_label(egui::Color32::from_rgb(255, 255, 255), format!("ch {c}"));
                        }
                        if let Some(c) = self.psth.sel_ch2 {
                            ui.colored_label(egui::Color32::from_rgb(255, 182, 23), format!("ch {c}"));
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
                    ui.label("Pick a stimulus-times file to compute the PSTH.");
                }
            });

        self.psth.open = open;
    }

    /// Default filename for a PSTH PNG: `<recording>_<stim file>_psth.png`.
    fn default_psth_png_name(&self) -> String {
        let rec = self.bin_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let stim = self.psth.stim_path.as_ref()
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
        let vmax = match self.psth.color_mode {
            ColorMode::Percentile => result.vmax_percentile(self.psth.color_pct),
            ColorMode::Voltage => self.psth.color_uv.max(1e-6),
        };
        let size_changed = self.psth.last_tex_size != Some([pw, ph]);
        if self.psth.tex_dirty || size_changed || self.psth.texture.is_none() {
            build_psth_heatmap_into(&mut self.psth.pixel_buf, result, pw, ph, vmax, cmap);
            let img = egui::ColorImage::from_rgba_unmultiplied([pw, ph], &self.psth.pixel_buf);
            self.psth.texture = Some(ui.ctx().load_texture("psth_heatmap", img, TextureOptions::NEAREST));
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
        let draw_marker = |ch: usize, color: egui::Color32| {
            if let Some(d) = result.display_rows.iter().position(
                |r| matches!(r, DisplayRow::Data { first_ch, .. } if *first_ch + 1 == ch),
            ) {
                let frac_y = ((n_disp - 1 - d) as f32 + 0.5) / n_disp as f32;
                let y = heat_rect.top() + frac_y * heat_rect.height();
                painter.line_segment(
                    [egui::pos2(heat_rect.left(), y), egui::pos2(heat_rect.right(), y)],
                    egui::Stroke::new(2.0, color),
                );
            }
        };
        if let Some(c) = self.psth.sel_ch1 {
            draw_marker(c, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 128));
        }
        if let Some(c) = self.psth.sel_ch2 {
            draw_marker(c, egui::Color32::from_rgba_unmultiplied(255, 182, 23, 128));
        }

        // channel selection: click the heatmap
        let resp = ui.interact(heat_rect, ui.id().with("psth_heat_click"), egui::Sense::click());
        let click = if resp.clicked() {
            resp.interact_pointer_pos().map(|p| (p, false))
        } else if resp.secondary_clicked() {
            resp.interact_pointer_pos().map(|p| (p, true))
        } else {
            None
        };
        if let Some((pos, right)) = click {
            if let Some(ch) = channel_at_heatmap_y(result, heat_rect, pos.y) {
                if right { self.psth.sel_ch2 = Some(ch); } else { self.psth.sel_ch1 = Some(ch); }
            }
        }

        // ---- selected-channel line plot ----
        painter.rect_filled(sel_rect, 0.0, c_zero);
        let mut sel_traces: Vec<(usize, egui::Color32)> = Vec::new();
        if let Some(c) = self.psth.sel_ch1 { sel_traces.push((c, ch1_color)); }
        if let Some(c) = self.psth.sel_ch2 { sel_traces.push((c, ch2_color)); }
        // symmetric scale across all shown channel traces
        let mut sel_max = 1e-6f32;
        for (ch, _) in &sel_traces {
            if let Some(row) = channel_row(result, *ch) {
                let s = &result.data[row * result.n_win..(row + 1) * result.n_win];
                sel_max = sel_max.max(s.iter().fold(0.0f32, |m, &v| m.max(v.abs())));
            }
        }
        let sel_mid = sel_rect.center().y;
        let sel_half = sel_rect.height() * 0.5 - 2.0;
        painter.line_segment(
            [egui::pos2(sel_rect.left(), sel_mid), egui::pos2(sel_rect.right(), sel_mid)],
            egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
        );
        for (ch, color) in &sel_traces {
            if let Some(row) = channel_row(result, *ch) {
                let s = &result.data[row * result.n_win..(row + 1) * result.n_win];
                let pts: Vec<egui::Pos2> = (0..result.n_win).map(|i| {
                    egui::pos2(x_of_i(&sel_rect, i), sel_mid - (s[i] / sel_max) * sel_half)
                }).collect();
                painter.add(egui::Shape::line(pts, egui::Stroke::new(1.5, *color)));
            }
        }

        // ---- mean-across-channels line plot ----
        painter.rect_filled(avg_rect, 0.0, c_zero);
        let tmax = result.avg_trace.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
        let avg_mid = avg_rect.center().y;
        let avg_half = avg_rect.height() * 0.5 - 2.0;
        painter.line_segment(
            [egui::pos2(avg_rect.left(), avg_mid), egui::pos2(avg_rect.right(), avg_mid)],
            egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
        );
        let pts: Vec<egui::Pos2> = (0..result.n_win).map(|i| {
            egui::pos2(x_of_i(&avg_rect, i), avg_mid - (result.avg_trace[i] / tmax) * avg_half)
        }).collect();
        painter.add(egui::Shape::line(pts, egui::Stroke::new(1.5, accent)));

        // onset marker at t = 0 across all three plots
        if start_ms < 0.0 && end_ms > 0.0 {
            let x0 = x_of_ms(0.0);
            let stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(150));
            painter.line_segment([egui::pos2(x0, sel_rect.top()), egui::pos2(x0, sel_rect.bottom())], stroke);
            painter.line_segment([egui::pos2(x0, avg_rect.top()), egui::pos2(x0, avg_rect.bottom())], stroke);
            painter.line_segment([egui::pos2(x0, heat_rect.top()), egui::pos2(x0, heat_rect.bottom())], stroke);
        }

        // labels
        let txt = egui::Color32::from_gray(200);
        let fid = egui::FontId::proportional(11.0);
        painter.text(egui::pos2(rect.left() + 2.0, sel_mid), egui::Align2::LEFT_CENTER, "chan µV", fid.clone(), txt);
        painter.text(egui::pos2(rect.left() + 2.0, avg_mid), egui::Align2::LEFT_CENTER, "mean µV", fid.clone(), txt);
        painter.text(egui::pos2(rect.left() + 2.0, heat_rect.top() + 2.0), egui::Align2::LEFT_TOP, format!("ch {}", self.psth.ch_last + 1), fid.clone(), txt);
        painter.text(egui::pos2(rect.left() + 2.0, heat_rect.bottom() - 2.0), egui::Align2::LEFT_BOTTOM, format!("ch {}", self.psth.ch_first + 1), fid.clone(), txt);
        let y_txt = rect.bottom() - x_axis_h + 4.0;
        painter.text(egui::pos2(heat_rect.left(), y_txt), egui::Align2::LEFT_TOP, format!("{:.0} ms", start_ms), fid.clone(), txt);
        if start_ms < 0.0 && end_ms > 0.0 {
            painter.text(egui::pos2(x_of_ms(0.0), y_txt), egui::Align2::CENTER_TOP, "0", fid.clone(), txt);
        }
        painter.text(egui::pos2(heat_rect.right(), y_txt), egui::Align2::RIGHT_TOP, format!("{:.0} ms", end_ms), fid, txt);
    }

    pub fn update(&mut self, ctx: &egui::Context) {
        self.poll_and_maybe_dispatch_psth(ctx);
        self.draw_psth_window(ctx);

        let mut show_prefs = self.show_preferences;
        if show_prefs {
            egui::Window::new("Preferences")
                .anchor(egui::Align2::RIGHT_TOP, [-10.0, 40.0])
                .collapsible(false)
                .open(&mut show_prefs)
                .show(ctx, |ui| {
                    ui.label(egui::RichText::new("Appearance").strong());

                    ui.horizontal(|ui| {
                        ui.label("Colormap:");
                        let mut cm = self.colormap_choice.clone();
                        egui::ComboBox::from_id_salt("cm_combo")
                            .selected_text(match cm {
                                ColorMapChoice::YellowMagenta => "Yellow-Magenta",
                                ColorMapChoice::RedBlue => "Red-Blue",
                                ColorMapChoice::OrangeBlue => "Orange-Blue",
                                ColorMapChoice::IceFire => "Ice-Fire",
                                ColorMapChoice::Vanimo => "Vanimo",
                                ColorMapChoice::GreyScale => "Greyscale",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut cm, ColorMapChoice::YellowMagenta, "Yellow-Magenta");
                                ui.selectable_value(&mut cm, ColorMapChoice::RedBlue, "Red-Blue");
                                ui.selectable_value(&mut cm, ColorMapChoice::OrangeBlue, "Orange-Blue");
                                ui.selectable_value(&mut cm, ColorMapChoice::IceFire, "Ice-Fire");
                                ui.selectable_value(&mut cm, ColorMapChoice::Vanimo, "Vanimo");
                                ui.selectable_value(&mut cm, ColorMapChoice::GreyScale, "Greyscale");
                            });
                        if cm != self.colormap_choice {
                            self.colormap_choice = cm;
                            self.heatmap_texture = None; // Force redraw
                            self.psth.tex_dirty = true; // PSTH heatmap tracks the same colormap
                        }
                    });

                    ui.separator();
                    ui.label(egui::RichText::new("Firing rate overlay").strong());

                    ui.horizontal(|ui| {
                        ui.label("Spike Threshold (µV):");
                        if ui.add(egui::DragValue::new(&mut self.spike_threshold).speed(1.0).suffix(" µV")).changed() {
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
                    ui.label(egui::RichText::new("Buffer").strong());

                    let n_data_rows = self.meta.build_display_rows(self.preproc_cfg.avg_depths)
                        .iter().filter(|r| matches!(r, DisplayRow::Data { .. })).count();
                    let max_feasible = max_feasible_buffer_s(n_data_rows, self.meta.sample_rate, self.mem_reserve_mb);

                    ui.horizontal(|ui| {
                        ui.label("Initial buffer size (s):");
                        if ui.add(
                            egui::DragValue::new(&mut self.initial_buffer_s)
                                .speed(0.5).range(1.0..=max_feasible).suffix(" s")
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
                                .speed(0.1).range(0.5..=max_margin).suffix(" s")
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
                                .speed(1.0).range(1.0..=90.0).suffix(" %")
                        ).changed() {
                            self.save_prefs();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Memory reserve (MB):");
                        if ui.add(
                            egui::DragValue::new(&mut self.mem_reserve_mb)
                                .speed(50.0).range(100.0..=20000.0).suffix(" MB")
                        ).changed() {
                            self.save_prefs();
                        }
                    });
                    ui.label(
                        egui::RichText::new("buffer growth stops once free memory drops below either threshold")
                            .small().color(egui::Color32::GRAY)
                    );
                });
        }
        self.show_preferences = show_prefs;

        // mouse-wheel scroll — 5% of window per tick
        let ticks = ctx.input(|i| {
            i.events.iter().filter_map(|e| match e {
                egui::Event::MouseWheel { delta, .. } => Some(delta.y.signum()),
                _ => None,
            }).sum::<f32>()
        });
        if ticks != 0.0 {
            let fs = self.meta.sample_rate;
            let total_s = self.meta.n_samples as f64 / fs;
            let max_start = (total_s - self.view_dur_s).max(0.0);
            let pct = if self.scroll_speed_fine { 0.05 } else { 0.30 };
            let step = ticks as f64 * self.view_dur_s * pct;
            self.view_start_s = (self.view_start_s + step).clamp(0.0, max_start);
        }

        // keyboard scroll
        ctx.input(|i| {
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

        TopBottomPanel::top("toolbar").show(ctx, |ui| { self.draw_toolbar(ui); });
        TopBottomPanel::top("preproc").show(ctx, |ui| { self.draw_preproc_panel(ui); });
        TopBottomPanel::top("chan_ctrl").show(ctx, |ui| { self.draw_channel_controls(ui); });
        TopBottomPanel::bottom("status_bar").exact_height(20.0).show(ctx, |ui| { self.draw_status_bar(ui); });
        TopBottomPanel::bottom("nav_bar").show(ctx, |ui| { self.draw_nav_bar(ui); });

        CentralPanel::default()
            .frame(egui::Frame::new().fill(egui::Color32::from_rgb(crate::render::C_ZERO[0], crate::render::C_ZERO[1], crate::render::C_ZERO[2])))
            .show(ctx, |ui| {
                let avail = ui.available_size();
                let pw = avail.x as usize;
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
                        if let RequestKind::Full { center_sample, .. } = &r.kind {
                            Some((*center_sample, r.cfg.clone()))
                        } else { None }
                    });
                    let act_cc = st.active_request.as_ref().and_then(|r| {
                        if let RequestKind::Full { center_sample, .. } = &r.kind {
                            Some((*center_sample, r.cfg.clone()))
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

                // request repaint while worker is busy (moved here from top to use snapshot)
                if w_status == WorkerStatus::Computing || w_has_request {
                    ctx.request_repaint_after(std::time::Duration::from_millis(50));
                }

                // compute vmax from current UI settings — valid whenever cfg matches,
                // regardless of exact time coverage (percentile table covers whatever's
                // currently in the buffer)
                let vmax = if matches_cfg {
                    if self.color_mode == ColorMode::Percentile {
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

                if matches_view {
                    self.waiting_since = None;
                }

                // Rebuild whenever we have a cfg-matching buffer, rendering whatever time
                // overlap exists and letting build_heatmap_into background-fill the rest —
                // keeps the view live instead of freezing on a stale frame while extension
                // or a full recompute catches up.
                if matches_cfg {
                    let pos_changed = self.last_rendered_first != view_first;
                    let cfg_changed = self.last_rendered_cfg.as_ref() != Some(&self.preproc_cfg);
                    let size_changed = self.last_rendered_size != Some([pw, ph]);
                    let buf_changed = self.last_rendered_buf != Some((buf_first, buf_n_samp));

                    let need_rebuild = self.heatmap_texture.is_none()
                        || pos_changed || view_n != self.last_rendered_n
                        || size_changed || buf_changed
                        || cfg_changed;

                    if need_rebuild && view_n > 0 {
                        if let (Some(data_arc), Some(display_rows)) = (&buf_data, &buf_display_rows) {
                            let stride = buf_n_samp;

                            let (first_row, last_row) = self.visible_row_range(display_rows);

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
                                || buf_changed;

                            if proj_stale {
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
                            }

                            build_heatmap_into(
                                &mut self.pixel_buf,
                                data_arc,
                                display_rows,
                                first_row, last_row,
                                stride, buf_first, buf_n_samp, view_first, view_n,
                                pw, ph, vmax,
                                &self.colormap_choice,
                            );
                            let img = egui::ColorImage::from_rgba_unmultiplied([pw, ph], &self.pixel_buf);
                            self.heatmap_texture = Some(ctx.load_texture("heatmap", img, TextureOptions::NEAREST));
                            self.last_rendered_first = view_first;
                            self.last_rendered_n = view_n;
                            self.last_rendered_cfg = buf_cfg;
                            self.last_rendered_size = Some([pw, ph]);
                            self.last_rendered_buf = Some((buf_first, buf_n_samp));
                        }
                    }
                }

                // request new background computation if needed (uses snapshot for checks, locks only for writes)
                let mut requested_new = false;
                if !matches_view || !matches_cfg || self.pending_cfg_recompute {
                    let already_requested = {
                        let req_match = req_center_cfg.as_ref().map_or(false, |(c, cfg)| *c == center && *cfg == self.preproc_cfg);
                        let act_match = act_center_cfg.as_ref().map_or(false, |(c, cfg)| *c == center && *cfg == self.preproc_cfg);
                        req_match || act_match
                    };

                    if !already_requested {
                        file_log!("UI: Requesting recompute. matches_view={}, matches_cfg={}, pending_cfg={}, center={}", matches_view, matches_cfg, self.pending_cfg_recompute, center);
                        file_log!("UI: buf first={}, n={}, view_first={}, view_n={}", buf_first, buf_n_samp, view_first, view_n);
                        self.request_recompute();
                        self.last_requested_center = center;
                        requested_new = true;
                    }
                    if !matches_view && self.waiting_since.is_none() {
                        self.waiting_since = Some(Instant::now());
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

                self.pending_cfg_recompute = false;

                // draw texture
                if let Some(tex) = &self.heatmap_texture {
                    let img_widget = egui::Image::new(tex)
                        .fit_to_exact_size(avail)
                        .sense(egui::Sense::click());
                    let resp = ui.add(img_widget);

                    // click detection
                    let mut click_pos = None;
                    let mut is_left_click = false;
                    let mut is_right_click = false;
                    
                    if resp.clicked() {
                        click_pos = resp.interact_pointer_pos().or_else(|| ctx.input(|i| i.pointer.interact_pos()));
                        is_left_click = true;
                    }
                    if resp.secondary_clicked() {
                        click_pos = resp.interact_pointer_pos().or_else(|| ctx.input(|i| i.pointer.interact_pos()));
                        is_right_click = true;
                    }

                    if let Some(display_rows) = &buf_display_rows {
                        let (first_row, last_row) = self.visible_row_range(display_rows);
                        let n_rows = last_row.saturating_sub(first_row) + 1;

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
                                            [egui::pos2(resp.rect.left(), y), egui::pos2(resp.rect.right(), y)],
                                            egui::Stroke::new(2.0, color)
                                        );
                                        break;
                                    }
                                }
                            }
                        };

                        if let Some(ch1) = self.selected_channel_1 {
                            draw_line(ch1, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 128));
                        }
                        if let Some(ch2) = self.selected_channel_2 {
                            draw_line(ch2, egui::Color32::from_rgba_unmultiplied(255, 182, 23, 128));
                        }

                        // draw projection overlay
                        if !self.projection_sums.is_empty() {
                            // scale with window duration and threshold (baseline: -20 µV → 1x),
                            // plus a user-configurable multiplier (spike_overlay_scale, default 1)
                            let threshold_scale = self.spike_threshold.abs() / 20.0;
                            let spike_scale_factor = threshold_scale * (0.5 / self.view_dur_s) as f32 * self.spike_overlay_scale;
                            
                            // per-map alpha is tuned for this overlay's many-triangle accumulation,
                            // so it stays separate from the shared RGB in render::colormap_accent
                            let overlay_alpha = match self.colormap_choice {
                                ColorMapChoice::YellowMagenta => 5,
                                ColorMapChoice::RedBlue => 8,
                                ColorMapChoice::OrangeBlue => 5,
                                ColorMapChoice::IceFire => 8,
                                ColorMapChoice::Vanimo => 5,
                                ColorMapChoice::GreyScale => 2,
                            };
                            let [pr, pg, pb] = crate::render::colormap_accent(&self.colormap_choice);
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
                    }

                    // hover overlay: ch / time / voltage
                    let hover_pos = resp.hover_pos().or_else(|| {
                        if resp.dragged() || resp.is_pointer_button_down_on() {
                            ctx.input(|i| i.pointer.interact_pos())
                        } else {
                            None
                        }
                    });

                    if let Some(pos) = hover_pos {
                        if let Some(display_rows) = &buf_display_rows {
                            let (first_row, last_row) = self.visible_row_range(display_rows);
                            let n_rows = last_row.saturating_sub(first_row) + 1;

                            let frac_y = ((pos.y - resp.rect.top()) / resp.rect.height()).clamp(0.0, 1.0);
                            let disp_idx = last_row.saturating_sub(
                                (frac_y as f64 * n_rows as f64) as usize
                            ).clamp(first_row, last_row);

                            let ch_str = match &display_rows[disp_idx] {
                                DisplayRow::Data { first_ch, .. } => format!("Ch {}  ", first_ch),
                                DisplayRow::IntraShankGap => "Channel gap  ".to_string(),
                                DisplayRow::ShankBoundary => "Shank gap  ".to_string(),
                            };

                            let frac_x = ((pos.x - resp.rect.left()) / resp.rect.width()).clamp(0.0, 1.0);
                            let t = self.view_start_s + frac_x as f64 * self.view_dur_s;

                            // voltage readout from snapshot data
                            let voltage_uv: Option<f32> = if let Some(DisplayRow::Data { data_idx, .. }) = display_rows.get(disp_idx) {
                                if let Some(data) = &buf_data {
                                    let t_sample = (t * self.meta.sample_rate) as usize;
                                    if t_sample >= buf_first {
                                        let off = t_sample - buf_first;
                                        let idx = data_idx * buf_n_samp + off;
                                        if off < buf_n_samp && idx < data.len() {
                                            Some(data[idx])
                                        } else { None }
                                    } else { None }
                                } else { None }
                            } else { None };

                            let volt_str = voltage_uv.map(|v| format!("  {:.1} µV", v)).unwrap_or_default();
                            let label = format!("{}t = {:.4} s{}", ch_str, t, volt_str);
                            
                            let font_id = egui::FontId::proportional(12.0);
                            let galley = ui.painter().layout_no_wrap(label, font_id.clone(), egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200));
                            let text_pos = resp.rect.left_bottom() + Vec2::new(6.0, -6.0 - galley.rect.height());
                            
                            let bg_rect = galley.rect.translate(text_pos.to_vec2()).expand(4.0);
                            ui.painter().rect_filled(bg_rect, 2.0, egui::Color32::from_rgba_unmultiplied(crate::render::C_ZERO[0], crate::render::C_ZERO[1], crate::render::C_ZERO[2], 200));
                            
                            ui.painter().galley(text_pos, galley, egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200));
                        }
                    }

                    // scale bar overlay (10% of view_dur_s) bottom right
                    let scale_bar_frac = 0.1;
                    let scale_bar_w = avail.x * scale_bar_frac;
                    let scale_bar_h = 4.0;
                    let bar_min = resp.rect.right_bottom() - egui::vec2(scale_bar_w + 20.0, 30.0);
                    let bar_rect = egui::Rect::from_min_size(bar_min, egui::vec2(scale_bar_w, scale_bar_h));
                    ui.painter().rect_filled(bar_rect, 0.0, egui::Color32::WHITE);
                    
                    let dur_ms = self.view_dur_s * (scale_bar_frac as f64) * 1000.0;
                    ui.painter().text(
                        bar_rect.right_bottom() + egui::vec2(0.0, 5.0),
                        egui::Align2::RIGHT_TOP,
                        format!("{:.0} ms", dur_ms),
                        egui::FontId::proportional(14.0),
                        egui::Color32::WHITE,
                    );
                }
            });
    }
}
