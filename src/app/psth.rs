//! The PSTH window: event-aligned averages of the preprocessed data, computed in
//! the background, drawn as a heatmap and traces, and exported as a PNG.

use super::*;

/// State for the PSTH window: the peri-stimulus average of the preprocessed signal.
pub(super) struct PsthState {
    pub(super) open: bool,
    pub(super) event_path: Option<PathBuf>,
    // staged settings (only committed to a recompute when "Apply/Compute" is pressed)
    pub(super) start_ms: f64,
    pub(super) end_ms: f64,
    pub(super) start_ms_str: String,
    pub(super) end_ms_str: String,
    pub(super) event_t_start: f64,
    pub(super) event_t_end: f64,
    pub(super) event_t_start_str: String,
    pub(super) event_t_end_str: String,
    pub(super) total_s: f64,
    pub(super) color_mode: ColorMode,
    pub(super) color_pct: f32,
    pub(super) color_uv: f32,

    // per-channel line selection (mirrors the main window; independent of it)
    pub(super) sel_ch1: Option<usize>,
    pub(super) sel_ch2: Option<usize>,

    // async plumbing
    pub(super) pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    /// the PSTH being computed; its progress counts events
    pub(super) job: Option<crate::jobs::BackgroundJob<PsthResult>>,
    pub(super) apply_requested: bool,
    /// color-scale max when the result arrived: the trace plots are autoscaled at that
    /// value and zoom with later color-scale changes
    pub(super) trace_vmax_ref: f32,

    pub(super) result: Option<Arc<PsthResult>>,
    pub(super) error: Option<String>,
    pub(super) n_used: usize,
    pub(super) n_skipped: usize,

    pub(super) texture: Option<TextureHandle>,
    pub(super) pixel_buf: Vec<u8>,
    pub(super) tex_dirty: bool,
    pub(super) last_tex_size: Option<[usize; 2]>,

    // rect of the plotted figure (logical coords), for PNG export cropping
    pub(super) figure_rect: Option<egui::Rect>,
    pub(super) export_pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    pub(super) export_path: Option<PathBuf>,
    pub(super) export_pending: bool,
}

impl PsthState {
    pub(super) fn new(total_s: f64) -> Self {
        Self {
            open: false,
            event_path: None,
            start_ms: -50.0,
            end_ms: 200.0,
            start_ms_str: "-50".to_string(),
            end_ms_str: "200".to_string(),
            event_t_start: 0.0,
            event_t_end: total_s,
            event_t_start_str: "0.000".to_string(),
            event_t_end_str: format!("{:.3}", total_s),
            total_s,
            color_mode: ColorMode::Percentile,
            color_pct: 99.0,
            color_uv: 50.0,
            sel_ch1: None,
            sel_ch2: None,
            pick_rx: None,
            job: None,
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
    pub(super) fn restore(&mut self, event_file: Option<&std::path::Path>, s: &crate::events::PsthSettings) {
        self.event_path = event_file.filter(|p| p.is_file()).map(|p| p.to_path_buf());
        if let Some(m) = &s.color_mode {
            self.color_mode = *m;
        }
        self.color_pct = s.color_pct.unwrap_or(self.color_pct).clamp(95.0, 100.0);
        self.color_uv = s.color_uv.unwrap_or(self.color_uv).clamp(1.0, 200.0);
        if let (Some(a), Some(b)) = (s.start_ms, s.end_ms) {
            (self.start_ms, self.end_ms) = (a, b);
            self.start_ms_str = format!("{a}");
            self.end_ms_str = format!("{b}");
        }
        if let (Some(a), Some(b)) = (s.event_t_start, s.event_t_end) {
            self.event_t_start = a.clamp(0.0, self.total_s);
            self.event_t_end = b.clamp(0.0, self.total_s);
            self.event_t_start_str = format!("{:.3}", self.event_t_start);
            self.event_t_end_str = format!("{:.3}", self.event_t_end);
        }
    }

    /// What is saved with the recording.
    pub(super) fn settings(&self) -> crate::events::PsthSettings {
        crate::events::PsthSettings {
            start_ms: Some(self.start_ms),
            end_ms: Some(self.end_ms),
            event_t_start: Some(self.event_t_start),
            event_t_end: Some(self.event_t_end),
            color_mode: Some(self.color_mode),
            color_pct: Some(self.color_pct),
            color_uv: Some(self.color_uv),
        }
    }

    /// Color-scale max of the heatmap (µV).
    pub(super) fn vmax(&self, result: &PsthResult) -> f32 {
        match self.color_mode {
            ColorMode::Percentile => result.vmax_percentile(self.color_pct),
            ColorMode::Voltage => self.color_uv.max(1e-6),
        }
    }
}


impl NPXplorerApp {
    /// Poll the PSTH picker/compute/export channels and dispatch a computation only
    /// when the user has pressed "Apply/Compute".
    pub(super) fn poll_and_maybe_dispatch_psth(&mut self, ctx: &egui::Context) {
        // event-file picker
        if let Some(rx) = &self.psth.pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.psth.pick_rx = None;
                    if let Some(path) = picked {
                        self.psth.event_path = Some(path);
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
        if let Some(outcome) = self.psth.job.as_ref().and_then(|j| j.poll(ctx)) {
            self.psth.job = None;
            match outcome {
                Outcome::Done(r) => {
                    self.psth.trace_vmax_ref = self.psth.vmax(&r);
                    self.psth.n_used = r.n_used;
                    self.psth.n_skipped = r.n_skipped;
                    self.psth.result = Some(Arc::new(r));
                    self.psth.error = None;
                    self.psth.tex_dirty = true;
                }
                Outcome::Failed(e) => {
                    self.psth.error = Some(e);
                    self.psth.result = None;
                }
                Outcome::Cancelled => {}
            }
        }

        // dispatch only when Apply was pressed
        if self.psth.apply_requested && self.psth.event_path.is_some() {
            self.psth.apply_requested = false;
            self.dispatch_psth_compute(ctx);
        }

        self.poll_psth_export(ctx);
    }

    pub(super) fn dispatch_psth_compute(&mut self, ctx: &egui::Context) {
        let event_path = match &self.psth.event_path {
            Some(p) => p.clone(),
            None => return,
        };
        let layout = match EventLayout::parse(&self.event_layout_text) {
            Ok(l) => l,
            Err(e) => {
                self.psth.error = Some(e.to_string());
                return;
            }
        };
        self.event_file = Some(event_path.clone());

        // a new job replaces (and so cancels) any in-flight one
        self.psth.error = None;

        let raw = Arc::clone(&self.raw);
        let meta = Arc::clone(&self.meta);
        let cfg = self.preproc_cfg.clone();
        let params = PsthParams {
            start_ms: self.psth.start_ms,
            end_ms: self.psth.end_ms,
        };
        let (t_start, t_end) = (self.psth.event_t_start, self.psth.event_t_end);
        // the total is set once the event file is read (0 shows "Preparing…")
        self.psth.job = Some(BackgroundJob::spawn(ctx, 0, move |c| {
            let all = load_event_times(&event_path, &layout).map_err(|e| e.to_string())?;
            let times: Vec<f64> = all.into_iter().filter(|&t| t >= t_start && t <= t_end).collect();
            if times.is_empty() {
                return Err(format!("no events fall within the selected time range {:.3}–{:.3} s.", t_start, t_end));
            }
            compute_psth(&raw, &meta, &cfg, &times, &params, c.control()).map_err(|e| e.to_string())
        }));
    }

    /// Poll the PNG-export save dialog and, once the requested screenshot arrives,
    /// crop it to the figure area and write the file.
    pub(super) fn poll_psth_export(&mut self, ctx: &egui::Context) {
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

    pub(super) fn draw_psth_window(&mut self, ctx: &egui::Context) {
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
                    self.psth.pick_rx = Some(spawn_event_picker(
                        self.bin_path.parent().map(|p| p.to_path_buf()),
                    ));
                }
                let name = self
                    .psth
                    .event_path
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

            crate::events::format_editor(ui, &mut self.event_layout_text);

            ui.horizontal(|ui| {
                // event time-range selector (seconds)
                ui.label("Event time (s):");
                let r1 = ui.add(
                    egui::TextEdit::singleline(&mut self.psth.event_t_start_str)
                        .desired_width(70.0)
                        .hint_text("start"),
                );
                if r1.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(v) = self.psth.event_t_start_str.trim().parse::<f64>() {
                        self.psth.event_t_start = v.clamp(0.0, total_s);
                    }
                    self.psth.event_t_start_str = format!("{:.3}", self.psth.event_t_start);
                }
                ui.label("to");
                let r2 = ui.add(
                    egui::TextEdit::singleline(&mut self.psth.event_t_end_str)
                        .desired_width(70.0)
                        .hint_text("end"),
                );
                if r2.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(v) = self.psth.event_t_end_str.trim().parse::<f64>() {
                        self.psth.event_t_end = v.clamp(0.0, total_s);
                    }
                    self.psth.event_t_end_str = format!("{:.3}", self.psth.event_t_end);
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
                let enabled = self.psth.event_path.is_some() && self.psth.job.is_none();
                let enter = enter_applies(ui);
                let apply = ui.add_enabled(
                    enabled,
                    egui::Button::new(egui::RichText::new("Apply/Compute").color(c_zero))
                        .fill(accent_50),
                );
                if apply.clicked() || (enter && enabled) {
                    self.psth.apply_requested = true;
                }
            });

            ui.horizontal(|ui| {
                ui.label("Color scale:");
                let mut changed = false;
                changed |= ui
                    .radio_value(&mut self.psth.color_mode, ColorMode::Percentile, "%ile")
                    .changed();
                let csd = self.psth.result.as_ref().is_some_and(|r| r.csd);
                let unit = if csd { "mV/mm²" } else { "µV" };
                changed |= ui
                    .radio_value(&mut self.psth.color_mode, ColorMode::Voltage, format!("±{unit}"))
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
                    let min = if csd { 0.01 } else { 1.0 };
                    changed |= ui
                        .add(egui::Slider::new(&mut self.psth.color_uv, min..=200.0).logarithmic(csd).text(unit))
                        .changed();
                }
                if changed {
                    self.psth.tex_dirty = true;
                }
            });

            if let Some(job) = &self.psth.job {
                let (done, total) = job.progress();
                ui.horizontal(|ui| {
                    if total == 0 {
                        // still reading the event file / setting up
                        ui.spinner();
                        ui.label("Preparing…");
                    } else {
                        ui.add(
                            egui::ProgressBar::new(done as f32 / total as f32)
                                .desired_width(240.0)
                                .show_percentage(),
                        );
                        ui.label(format!("{done} / {total} events"));
                    }
                    if job.is_cancelling() {
                        ui.label("Stopping…");
                    } else if ui.button("Abort").clicked() {
                        job.cancel();
                    }
                });
            } else if let Some(err) = &self.psth.error {
                ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), err);
            } else if self.psth.result.is_some() {
                ui.horizontal(|ui| {
                    let base = if self.psth.n_skipped > 0 {
                        format!(
                            "{} events averaged ({} skipped near edges)",
                            self.psth.n_used, self.psth.n_skipped
                        )
                    } else {
                        format!("{} events averaged", self.psth.n_used)
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
            } else if self.psth.job.is_none() && self.psth.error.is_none() {
                if self.psth.event_path.is_some() {
                    ui.label("Press Apply/Compute to compute the PSTH.");
                } else {
                    ui.label("Pick an event-times file to compute the PSTH.");
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
                    let min = if self.psth.result.as_ref().is_some_and(|r| r.csd) { 0.01 } else { 1.0 };
                    self.psth.color_uv = (self.psth.color_uv * 1.1f32.powf(-ticks)).clamp(min, 200.0);
                }
                self.psth.tex_dirty = true;
            }
        }

        self.psth.open = open;
    }

    /// Default filename for a PSTH PNG: `<recording>_<event file>_psth.png`.
    pub(super) fn default_psth_png_name(&self) -> String {
        let rec = self
            .bin_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let events = self
            .psth
            .event_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        format!("{rec}_{events}_psth.png")
    }

    pub(super) fn draw_psth_plots(
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
            if result.csd { "chan mV/mm²" } else { "chan µV" },
            fid.clone(),
            txt,
        );
        painter.text(
            egui::pos2(rect.left() + 2.0, avg_mid),
            egui::Align2::LEFT_CENTER,
            if result.csd { "mean mV/mm²" } else { "mean µV" },
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
}
