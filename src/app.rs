use egui::{CentralPanel, TextureHandle, TextureOptions, TopBottomPanel, Ui, Vec2};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::data::{DisplayRow, Meta, open_data};
use crate::preprocess::{Filters, PreprocConfig, SpatialFilter};
use crate::render::build_heatmap_into;
use crate::worker::{
    SharedCancel, SharedWorkerState, WorkerRequest, WorkerState, WorkerStatus,
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

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Preferences {
    pub preproc_cfg: PreprocConfig,
    pub view_dur_s: f64,
    pub color_mode: ColorMode,
    pub color_pct: f32,
    pub color_uv: f32,
    pub colormap_choice: ColorMapChoice,
    pub spike_threshold: f32,
}

impl Preferences {
    pub fn load() -> Option<Self> {
        let path = Self::path();
        if let Ok(s) = std::fs::read_to_string(path) {
            toml::from_str(&s).ok()
        } else {
            None
        }
    }

    pub fn save(&self) {
        let path = Self::path();
        if let Ok(s) = toml::to_string_pretty(self) {
            let _ = std::fs::write(path, s);
        }
    }

    fn path() -> std::path::PathBuf {
        if let Ok(home) = std::env::var("HOME") {
            std::path::PathBuf::from(home).join(".rawviewer_prefs.toml")
        } else {
            std::path::PathBuf::from("rawviewer_prefs.toml")
        }
    }
}

pub struct RawViewerApp {
    bin_path: PathBuf,
    meta: Arc<Meta>,
    is_compressed: bool,

    // view state
    view_start_s: f64,
    view_dur_s: f64,
    window_dur_str: String,   // owned string for the text field
    jump_str: String,         // owned string for the jump text field
    ch_first: usize,          // first_ch of first visible data row
    ch_last: usize,           // first_ch of last visible data row

    // preprocessing
    preproc_cfg: PreprocConfig,
    preproc_filters: Arc<Mutex<Filters>>,
    scroll_speed_fine: bool,

    // color scale
    color_mode: ColorMode,
    color_pct: f32,           // percentile (80.0–100.0), default 99.0
    color_uv: f32,            // voltage default 120.0
    color_pct_str: String,
    color_uv_str: String,
    colormap_choice: ColorMapChoice,

    // preferences
    show_preferences: bool,
    spike_threshold: f32,

    // selected channels
    selected_channel_1: Option<usize>,
    selected_channel_2: Option<usize>,

    // async worker
    worker_state: SharedWorkerState,
    worker_cancel: SharedCancel,
    worker_half_window: usize,
    _worker_handle: std::thread::JoinHandle<()>,

    // rendering
    heatmap_texture: Option<TextureHandle>,
    pixel_buf: Vec<u8>,
    last_rendered_first: usize,
    last_rendered_cfg: Option<PreprocConfig>,
    last_rendered_n: usize,
    last_rendered_size: Option<[usize; 2]>,

    // smooth-scroll state
    waiting_since: Option<Instant>,
    last_requested_center: usize,

    // UI state
    pending_cfg_recompute: bool,
    pub file_dialog_request: bool,
    projection_sums: Vec<f32>,
}

impl RawViewerApp {
    pub fn new(ctx: &egui::Context, bin_path: PathBuf) -> anyhow::Result<Self> {
        let meta_path = bin_path.with_extension("meta");
        let meta = Arc::new(Meta::from_file(&meta_path)?);
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
        }
        let filters = Arc::new(Mutex::new(Filters::new(&preproc_cfg)));
        let shared: SharedWorkerState = Arc::new((Mutex::new(WorkerState::new()), Condvar::new()));
        let cancel: SharedCancel = Arc::new(AtomicBool::new(false));

        let half_window = compute_half_window(view_dur_s, fs);

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
                center_sample: (half_window as f64 * 0.5 * fs / fs) as usize,
                half_window,
                cfg: preproc_cfg.clone(),
            });
            cvar.notify_one();
        }

        let is_compressed = bin_path.extension().and_then(|s| s.to_str()) == Some("cbin");

        let n_ap = meta.n_ap_chans;
        Ok(Self {
            bin_path,
            meta,
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
            selected_channel_1: None,
            selected_channel_2: None,
            worker_state: shared,
            worker_cancel: cancel,
            worker_half_window: half_window,
            _worker_handle: handle,
            heatmap_texture: None,
            pixel_buf: Vec::new(),
            last_rendered_first: usize::MAX,
            last_rendered_cfg: None,
            last_rendered_n: 0,
            last_rendered_size: None,
            waiting_since: None,
            last_requested_center: 0,
            pending_cfg_recompute: false,
            file_dialog_request: false,
            projection_sums: Vec::new(),
        })
    }

    pub fn save_prefs(&self) {
        let prefs = Preferences {
            preproc_cfg: self.preproc_cfg.clone(),
            view_dur_s: self.view_dur_s,
            color_mode: self.color_mode.clone(),
            color_pct: self.color_pct,
            color_uv: self.color_uv,
            colormap_choice: self.colormap_choice.clone(),
            spike_threshold: self.spike_threshold,
        };
        prefs.save();
    }

    fn request_recompute(&mut self) {
        let fs = self.meta.sample_rate;
        let center = ((self.view_start_s + self.view_dur_s / 2.0) * fs) as usize;
        // cancel any in-flight computation
        self.worker_cancel.store(true, Ordering::Relaxed);
        let req = WorkerRequest {
            center_sample: center,
            half_window: self.worker_half_window,
            cfg: self.preproc_cfg.clone(),
        };
        let (lock, cvar) = &*self.worker_state;
        lock.lock().unwrap().request = Some(req);
        cvar.notify_one();
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
            ui.label(format!("t = {:.3} s", self.view_start_s));

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
                        self.worker_half_window = compute_half_window(self.view_dur_s, self.meta.sample_rate);
                        self.heatmap_texture = None;
                        self.pending_cfg_recompute = true;
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
            ui.label("Color:");

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
                        ui.label(text);
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

        painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(0x18, 0x18, 0x18));

        // view marker
        let view_frac = (self.view_start_s / total_s) as f32;
        let view_w_frac = (self.view_dur_s / total_s) as f32;
        let view_rect = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + w * view_frac, rect.min.y),
            Vec2::new((w * view_w_frac).max(2.0), rect.height()),
        );
        painter.rect_filled(view_rect, 1.0, egui::Color32::from_rgba_premultiplied(200, 200, 255, 180));

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

impl RawViewerApp {
    pub fn update(&mut self, ctx: &egui::Context) {
        // request repaint while worker is busy
        {
            let (lock, _) = &*self.worker_state;
            let st = lock.lock().unwrap();
            if st.status == WorkerStatus::Computing || st.request.is_some() {
                ctx.request_repaint_after(std::time::Duration::from_millis(50));
            }
        }

        let mut show_prefs = self.show_preferences;
        if show_prefs {
            egui::Window::new("Preferences")
                .anchor(egui::Align2::RIGHT_TOP, [-10.0, 40.0])
                .collapsible(false)
                .open(&mut show_prefs)
                .show(ctx, |ui| {
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
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Spike Threshold (µV):");
                        if ui.add(egui::DragValue::new(&mut self.spike_threshold).speed(1.0).suffix(" µV")).changed() {
                            self.heatmap_texture = None; // Redraw to update projection
                        }
                    });
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
        TopBottomPanel::bottom("status_bar").show(ctx, |ui| { self.draw_status_bar(ui); });
        TopBottomPanel::bottom("nav_bar").show(ctx, |ui| { self.draw_nav_bar(ui); });

        CentralPanel::default()
            .frame(egui::Frame::new().fill(egui::Color32::from_rgb(0x18, 0x18, 0x18)))
            .show(ctx, |ui| {
                let avail = ui.available_size();
                let pw = avail.x as usize;
                let ph = avail.y as usize;
                if pw < 2 || ph < 2 { return; }

                let fs = self.meta.sample_rate;
                let view_first = (self.view_start_s * fs) as usize;
                let view_n = (self.view_dur_s * fs) as usize;
                let center = view_first + view_n / 2;

                // check buffer coverage
                let (matches_view, matches_cfg, display_rows_opt, vmax, buf_cfg) = {
                    let (lock, _) = &*self.worker_state;
                    let st = lock.lock().unwrap();
                    if let Some(buf) = &st.buffer {
                        let buf_end = buf.first_sample + buf.n_samp;
                        let m_view = buf.first_sample <= view_first && view_first + view_n <= buf_end;
                        let m_cfg = buf.cfg == self.preproc_cfg;
                        
                        let v = if self.color_mode == ColorMode::Percentile {
                            let pct_idx = (self.color_pct * 100.0).round() as usize;
                            buf.vmax_pct[pct_idx.min(10000)]
                        } else {
                            self.color_uv
                        };
                        
                        (m_view, m_cfg, Some(Arc::clone(&buf.display_rows)), v, Some(buf.cfg.clone()))
                    } else {
                        (false, false, None, 250.0f32, None)
                    }
                };

                let vmax = vmax.max(1.0);

                // Rebuild heatmap if we have data for the view
                if matches_view {
                    self.waiting_since = None;

                    let pos_changed = self.last_rendered_first != view_first;
                    let cfg_changed = self.last_rendered_cfg.as_ref() != Some(&self.preproc_cfg);
                    let size_changed = self.last_rendered_size != Some([pw, ph]);
                    
                    let need_rebuild = self.heatmap_texture.is_none()
                        || pos_changed || view_n != self.last_rendered_n
                        || size_changed
                        || (cfg_changed && matches_cfg);

                    if need_rebuild && view_n > 0 {
                        if let Some(display_rows) = &display_rows_opt {
                            let (data_arc, stride, offset, n) = {
                                let (lock, _) = &*self.worker_state;
                                let st = lock.lock().unwrap();
                                let buf = st.buffer.as_ref().unwrap();
                                let off = view_first - buf.first_sample;
                                let n = view_n.min(buf.n_samp - off);
                                (Arc::clone(&buf.data), buf.n_samp, off, n)
                            };

                            let (first_row, last_row) = self.visible_row_range(display_rows);
                            
                            // Compute projection sums (spike counts)
                            let visible = &display_rows[first_row..=last_row];
                            let mut sums = vec![0.0f32; visible.len()];
                            
                            let sample_rate = self.meta.sample_rate;
                            let refractory_samples = (1.5 * sample_rate as f32 / 1000.0) as usize;
                            let threshold = self.spike_threshold;

                            use rayon::prelude::*;
                            sums.par_iter_mut().enumerate().for_each(|(i, count)| {
                                if let DisplayRow::Data { data_idx, .. } = &visible[i] {
                                    let base = data_idx * stride + offset;
                                    if base + n <= data_arc.len() {
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

                            // Apply Gaussian convolution (sigma = 1.5, kernel size 7)
                            let kernel = [0.0366, 0.111, 0.217, 0.271, 0.217, 0.111, 0.0366];
                            let k_rad = 3;
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

                            build_heatmap_into(
                                &mut self.pixel_buf,
                                &data_arc,
                                display_rows,
                                first_row, last_row,
                                stride, offset, n,
                                pw, ph, vmax,
                                &self.colormap_choice,
                            );
                            let img = egui::ColorImage::from_rgba_unmultiplied([pw, ph], &self.pixel_buf);
                            self.heatmap_texture = Some(ctx.load_texture("heatmap", img, TextureOptions::NEAREST));
                            self.last_rendered_first = view_first;
                            self.last_rendered_n = view_n;
                            self.last_rendered_cfg = buf_cfg;
                            self.last_rendered_size = Some([pw, ph]);
                        }
                    }
                }

                // Request new background computation if needed
                let mut requested_new = false;
                if !matches_view || !matches_cfg || self.pending_cfg_recompute {
                    // We need a new buffer either because we are out of bounds, or settings changed.
                    // But we ONLY send a new request if we haven't already requested it.
                    let already_requested = {
                        let (lock, _) = &*self.worker_state;
                        let st = lock.lock().unwrap();
                        let req_match = st.request.as_ref().map_or(false, |r| r.center_sample == center && r.cfg == self.preproc_cfg);
                        let act_match = st.active_request.as_ref().map_or(false, |r| r.center_sample == center && r.cfg == self.preproc_cfg);
                        req_match || act_match
                    };

                    if !already_requested {
                        self.request_recompute();
                        self.last_requested_center = center;
                        requested_new = true;
                    }
                    if !matches_view && self.waiting_since.is_none() {
                        self.waiting_since = Some(Instant::now());
                    }
                }

                if matches_view && matches_cfg && !requested_new {
                    // Prefetch logic: request next chunk if approaching edge,
                    // but ONLY if we haven't requested it already.
                    let prefetch_margin = self.worker_half_window / 4;
                    let (buf_first, buf_end) = {
                        let (lock, _) = &*self.worker_state;
                        let st = lock.lock().unwrap();
                        if let Some(buf) = &st.buffer {
                            (buf.first_sample, buf.first_sample + buf.n_samp)
                        } else { (0, 0) }
                    };
                    
                    let mut near_left = view_first < buf_first + prefetch_margin;
                    let mut near_right = view_first + view_n + prefetch_margin > buf_end;

                    // Prevent prefetching past the file bounds!
                    if near_left && buf_first == 0 {
                        near_left = false;
                    }
                    if near_right && buf_end >= self.meta.n_samples {
                        near_right = false;
                    }
                        
                    if near_left || near_right {
                        // The next center should be shifted towards the direction we are heading.
                        let next_center = if near_left {
                            view_first.saturating_sub(self.worker_half_window / 2)
                        } else {
                            view_first + view_n + self.worker_half_window / 2
                        };
                        
                        let already_requested = {
                            let (lock, _) = &*self.worker_state;
                            let st = lock.lock().unwrap();
                            let req_match = st.request.as_ref().map_or(false, |r| r.center_sample == next_center && r.cfg == self.preproc_cfg);
                            let act_match = st.active_request.as_ref().map_or(false, |r| r.center_sample == next_center && r.cfg == self.preproc_cfg);
                            req_match || act_match
                        };

                        if !already_requested {
                            // cancel any in-flight computation
                            self.worker_cancel.store(true, Ordering::Relaxed);
                            let req = WorkerRequest {
                                center_sample: next_center,
                                half_window: self.worker_half_window,
                                cfg: self.preproc_cfg.clone(),
                            };
                            let (lock, cvar) = &*self.worker_state;
                            lock.lock().unwrap().request = Some(req);
                            cvar.notify_one();
                            
                            self.last_requested_center = next_center;
                        }
                    }
                }

                // Show loading warning only if we have no texture to display
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

                    // Click detection
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

                    let display_rows_arc = {
                        let (lock, _) = &*self.worker_state;
                        let st = lock.lock().unwrap();
                        st.buffer.as_ref().map(|b| Arc::clone(&b.display_rows))
                    };

                    if let Some(display_rows) = &display_rows_arc {
                        let (first_row, last_row) = self.visible_row_range(&display_rows);
                        let n_rows = last_row.saturating_sub(first_row) + 1;

                        // Handle clicks
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

                        // Draw lines
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

                        // Draw Projection Overlay
                        if !self.projection_sums.is_empty() {
                            // Adaptive scaling: inversely proportional to the window size. Base is calibrated to 0.5s.
                            let spike_scale_factor = 1.0 * (0.5 / self.view_dur_s) as f32;
                            
                            // 10% opacity colors for each colormap
                            let color = match self.colormap_choice {
                                ColorMapChoice::YellowMagenta => egui::Color32::from_rgba_unmultiplied(250, 234, 130, 5),
                                ColorMapChoice::RedBlue => egui::Color32::from_rgba_unmultiplied(204, 103, 230, 8),
                                ColorMapChoice::OrangeBlue => egui::Color32::from_rgba_unmultiplied(242, 171, 126, 5),
                                ColorMapChoice::IceFire => egui::Color32::from_rgba_unmultiplied(166, 217, 237, 8),
                                ColorMapChoice::Vanimo => egui::Color32::from_rgba_unmultiplied(202, 237, 166, 5),
                                ColorMapChoice::GreyScale => egui::Color32::from_rgba_unmultiplied(255, 255, 255, 2),
                            };

                            let min_x = resp.rect.left();
                            let max_x = resp.rect.right(); // Can project across the entire image if scaled high enough
                            let top_y = resp.rect.top();
                            let h = resp.rect.height();

                            let row_h = h / n_rows as f32;

                            let mut mesh = egui::epaint::Mesh::default();

                            for (i, &count) in self.projection_sums.iter().enumerate() {
                                let x = min_x + count * spike_scale_factor;
                                let x = x.min(max_x); // clamp to the right edge of the image
                                
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
                        if let Some(display_rows) = &display_rows_arc {
                            let (first_row, last_row) = self.visible_row_range(&display_rows);
                            let n_rows = last_row.saturating_sub(first_row) + 1;

                            let frac_y = ((pos.y - resp.rect.top()) / resp.rect.height()).clamp(0.0, 1.0);
                            let disp_idx = last_row.saturating_sub(
                                (frac_y as f64 * n_rows as f64) as usize
                            ).clamp(first_row, last_row);

                            let first_ch_hover = if let DisplayRow::Data { first_ch, .. } = &display_rows[disp_idx] {
                                Some(*first_ch + 1)
                            } else { None };

                            let frac_x = ((pos.x - resp.rect.left()) / resp.rect.width()).clamp(0.0, 1.0);
                            let t = self.view_start_s + frac_x as f64 * self.view_dur_s;

                            let voltage_uv: Option<f32> = if let Some(DisplayRow::Data { data_idx, .. }) = display_rows.get(disp_idx) {
                                let (lock, _) = &*self.worker_state;
                                let st = lock.lock().unwrap();
                                if let Some(buf) = &st.buffer {
                                    let t_sample = (t * self.meta.sample_rate) as usize;
                                    if t_sample >= buf.first_sample {
                                        let off = t_sample - buf.first_sample;
                                        if off < buf.n_samp {
                                            Some(buf.data[data_idx * buf.n_samp + off])
                                        } else { None }
                                    } else { None }
                                } else { None }
                            } else { None };

                            let ch_str = first_ch_hover.map(|c| format!("Ch {}  ", c)).unwrap_or_default();
                            let volt_str = voltage_uv.map(|v| format!("  {:.1} µV", v)).unwrap_or_default();
                            let label = format!("{}t = {:.4} s{}", ch_str, t, volt_str);
                            
                            // Measure text to draw background
                            let font_id = egui::FontId::proportional(12.0);
                            let galley = ui.painter().layout_no_wrap(label, font_id.clone(), egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200));
                            let text_pos = resp.rect.left_bottom() + Vec2::new(6.0, -6.0 - galley.rect.height());
                            
                            // Background box
                            let bg_rect = galley.rect.translate(text_pos.to_vec2()).expand(4.0);
                            ui.painter().rect_filled(bg_rect, 2.0, egui::Color32::from_rgba_unmultiplied(0x18, 0x18, 0x18, 200));
                            
                            ui.painter().galley(text_pos, galley, egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200));
                        }
                    }

                    // Scale bar overlay (10% of view_dur_s) bottom right
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
