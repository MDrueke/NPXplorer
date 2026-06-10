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

#[derive(Clone, PartialEq, Eq)]
pub enum ColorMode {
    Percentile,
    Voltage,
}

#[derive(Clone, PartialEq, Eq)]
pub enum ColorMapChoice {
    YellowMagenta,
    RedBlue,
    OrangeBlue,
    IceFire,
    Vanimo,
    GreyScale,
}

pub struct RawViewerApp {
    bin_path: PathBuf,
    meta: Arc<Meta>,

    // view state
    view_start_s: f64,
    view_dur_s: f64,
    window_dur_str: String,   // owned string for the text field
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
}

impl RawViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>, bin_path: PathBuf) -> anyhow::Result<Self> {
        let meta_path = bin_path.with_extension("meta");
        let meta = Arc::new(Meta::from_file(&meta_path)?);
        let (raw, _) = open_data(&bin_path, &meta)?;
        let raw = Arc::new(raw);
        let fs = meta.sample_rate;

        let preproc_cfg = PreprocConfig {
            dc_removal: true,
            phase_shift: false,
            highpass: true,
            spatial_filter: SpatialFilter::GlobalCmr,
            avg_depths: true,
            sample_rate: fs,
            im_dat_prb_type: meta.im_dat_prb_type,
        };
        let filters = Arc::new(Mutex::new(Filters::new(&preproc_cfg)));
        let shared: SharedWorkerState = Arc::new((Mutex::new(WorkerState::new()), Condvar::new()));
        let cancel: SharedCancel = Arc::new(AtomicBool::new(false));

        // compute display rows to size the buffer
        let display_rows = meta.build_display_rows(preproc_cfg.avg_depths);
        let n_data_rows = display_rows.iter()
            .filter(|r| matches!(r, DisplayRow::Data { .. }))
            .count();
        let half_window = compute_half_window(n_data_rows, fs);

        let handle = spawn_worker(
            Arc::clone(&raw),
            Arc::clone(&meta),
            Arc::clone(&filters),
            Arc::clone(&shared),
            Arc::clone(&cancel),
            cc.egui_ctx.clone(),
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

        let n_ap = meta.n_ap_chans;
        Ok(Self {
            bin_path,
            meta,
            view_start_s: 0.0,
            view_dur_s: 0.5,
            window_dur_str: "0.500".to_string(),
            ch_first: 0,
            ch_last: n_ap.saturating_sub(1),
            preproc_cfg: preproc_cfg.clone(),
            preproc_filters: filters,
            scroll_speed_fine: true,
            color_mode: ColorMode::Percentile,
            color_pct: 99.0,
            color_uv: 120.0,
            color_pct_str: "99.0".to_string(),
            color_uv_str: "120.0".to_string(),
            colormap_choice: ColorMapChoice::YellowMagenta,
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
        })
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
            let status = {
                let (lock, _) = &*self.worker_state;
                lock.lock().unwrap().status.clone()
            };
            if status == WorkerStatus::Computing {
                ui.spinner();
                ui.label("Computing…");
            }
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
            let mut jump_str = format!("{:.3}", self.view_start_s);
            let resp = ui.add(egui::TextEdit::singleline(&mut jump_str).desired_width(70.0));
            if resp.lost_focus() {
                if let Ok(t) = jump_str.parse::<f64>() {
                    let max_t = self.meta.n_samples as f64 / self.meta.sample_rate - self.view_dur_s;
                    self.view_start_s = t.clamp(0.0, max_t.max(0.0));
                }
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

impl eframe::App for RawViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // request repaint while worker is busy
        {
            let (lock, _) = &*self.worker_state;
            let st = lock.lock().unwrap();
            if st.status == WorkerStatus::Computing || st.request.is_some() {
                ctx.request_repaint_after(std::time::Duration::from_millis(50));
            }
        }


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
