//! The smaller windows: remove channels, noise suppression, CSD, export, the
//! channel context menu, and closing the topmost window with Esc.

use super::*;

/// Spawn the native picker for a channel-numbers-to-remove file on a background thread.
pub(super) fn spawn_channel_list_picker(dir: Option<PathBuf>) -> mpsc::Receiver<Option<PathBuf>> {
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

impl NPXplorerApp {
    pub(super) fn apply_removed_channels(&mut self, set: BTreeSet<usize>) {
        let n_left = (0..self.meta.n_ap_chans)
            .filter(|c| !set.contains(c))
            .count();
        if n_left == 0 {
            self.remove_channels_error =
                Some("this would remove every channel — at least one must stay".to_string());
            return;
        }
        self.preproc_cfg.removed_channels = set;
    }

    /// Parse `remove_channels_text` and apply it, or set an error message on failure.
    pub(super) fn apply_remove_channels_text(&mut self) {
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
    pub(super) fn remove_channel(&mut self, ch: usize) {
        let mut set = self.preproc_cfg.removed_channels.clone();
        set.insert(ch - 1);
        self.remove_channels_text =
            crate::channel_remove::format_channel_list(&set, &self.meta.channel_ids);
        self.remove_channels_error = None;
        self.apply_removed_channels(set);
    }

    pub(super) fn poll_remove_channels_picker(&mut self, ctx: &egui::Context) {
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

    pub(super) fn draw_remove_channels_window(&mut self, ctx: &egui::Context) {
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
                let enter = enter_applies(ui);
                if resp.lost_focus() || enter {
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

    /// `None` shows the full band up to Nyquist; see `spectrum_freq_restrict`.
    /// Settings window opened by the "Noise Suppression" toolbar button.
    pub(super) fn draw_noise_window(&mut self, ctx: &egui::Context) {
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
                    match crate::noise_ui::draw_settings(ui, &mut self.noise_draft, &self.noise) {
                        crate::noise_ui::Action::Apply => self.noise = self.noise_draft.clone(),
                        crate::noise_ui::Action::DisableAll => {
                            self.noise_draft.disable_all();
                            self.noise.disable_all();
                        }
                        crate::noise_ui::Action::None => {}
                    }
                    ui.add_space(8.0);
                    ui.separator();
                    ui.strong("Notch filters");
                    if let Some(notches) = self.notch_panel.draw(ui, &self.raw, &self.meta, &self.preproc_cfg) {
                        self.preproc_cfg.notch_enabled = !notches.is_empty();
                        self.preproc_cfg.notches = notches;
                    }
                });
            });
        self.noise_open = open;
    }

    /// CSD settings, opened by the "CSD" spatial filter. Apply (or Enter) switches CSD on
    /// with them; Cancel (or Esc) leaves everything as it was.
    pub(super) fn draw_csd_window(&mut self, ctx: &egui::Context) {
        let Some(mut p) = self.csd_edit.clone() else { return };
        let nyq = self.meta.sample_rate / 2.0;
        let valid = !p.bandpass || p.high_hz > p.low_hz;
        let (mut apply, mut cancel) = (false, false);
        egui::Window::new("CSD").collapsible(false).resizable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut p.smoothing, "Gaussian smoothing, σ:").on_hover_text(
                    "Smoothing along depth before the second derivative, which strongly amplifies noise that differs from row to row. For LFP, 40–80 µm; below ~30 µm the result is mostly noise.",
                );
                ui.add_enabled(p.smoothing, egui::DragValue::new(&mut p.sigma_um).speed(1.0).range(1.0..=300.0).suffix(" µm"));
            });
            ui.checkbox(&mut p.bandpass, "Bandpass before CSD").on_hover_text(
                "Zero-phase Butterworth, order 3 at each edge; replaces the 300 Hz highpass. For LFP sinks and sources use e.g. 1–300 Hz. NP 1.0 AP data is usually highpassed in hardware (last imro field), so it has little signal below 300 Hz; NP 2.0 AP data is wideband.",
            );
            ui.add_enabled_ui(p.bandpass, |ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut p.low_hz).speed(0.5).range(0.1..=nyq * 0.9).suffix(" Hz"));
                    ui.label("to");
                    ui.add(egui::DragValue::new(&mut p.high_hz).speed(5.0).range(1.0..=nyq * 0.95).suffix(" Hz"));
                });
            });
            if !valid {
                ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), "The upper cutoff must be above the lower one.");
            }
            ui.label(
                egui::RichText::new("Shown as −∂²φ/∂z² in mV/mm² (no tissue conductivity assumed); sinks are negative. Without a bandpass, AP data gives a spike-band Laplacian rather than an LFP CSD.")
                    .small()
                    .color(egui::Color32::GRAY),
            );
            ui.horizontal(|ui| {
                let enter = enter_applies(ui);
                if ui.add_enabled(valid, egui::Button::new("Apply")).clicked() || (enter && valid) {
                    apply = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            });
        });
        if apply {
            let c = &mut self.preproc_cfg;
            if c.spatial_filter != SpatialFilter::Csd {
                self.csd_restore = Some((c.avg_depths, c.highpass));
                c.spatial_filter = SpatialFilter::Csd;
            }
            c.csd = p;
            // the 300 Hz HP comes back when the bandpass is switched off
            if let (false, Some((_, hp))) = (c.csd.bandpass, self.csd_restore) {
                c.highpass = hp;
            }
            enforce_csd_options(c);
            self.csd_edit = None;
        } else {
            self.csd_edit = (!cancel).then_some(p);
        }
    }

    pub(super) fn draw_export_window(&mut self, ctx: &egui::Context) {
        let steps = preprocessing_summary(&self.preproc_cfg);
        let inputs = crate::export_ui::WindowInputs {
            meta: &self.meta,
            src: &self.bin_path,
            removed: &self.preproc_cfg.removed_channels,
            labels: self.classify.labels.as_deref().map(|v| v.as_slice()),
            has_events: self.events.has_data(),
            view_s: (self.view_start_s, self.view_dur_s),
            steps,
            export_dir: FileLocations::resolve(&self.file_locations.export, &self.bin_path),
            csd: self.preproc_cfg.spatial_filter == SpatialFilter::Csd,
        };
        let req = self.export_ui.draw_window(ctx, &inputs);
        self.handle_export_requests(vec![req], ctx);
    }

    pub(super) fn handle_export_requests(&mut self, reqs: Vec<crate::export_ui::Request>, ctx: &egui::Context) {
        use crate::export_ui::Request;
        for r in reqs {
            match r {
                Request::None => {}
                Request::Export(dst) => {
                    // the settings file is current before the export reads it
                    self.save_settings();
                    let job = self.build_export_job(dst);
                    self.export_ui.start(job, ctx);
                }
                Request::OpenEventsWindow => self.events.open = true,
                Request::Open(path) => self.open_recent_request = Some(path),
                Request::Labels(l) => self.classify.set_labels(l),
            }
        }
    }

    pub(super) fn build_export_job(&self, dst: PathBuf) -> crate::export::Job {
        let events = self.events.times().map(|(on, off)| crate::export::EventTimes {
            onsets: on.to_vec(),
            offsets: off.map(|o| o.to_vec()),
            file: self.event_file.clone(),
        });
        crate::export::Job {
            src: self.bin_path.clone(),
            dst,
            format: crate::export::Format::of(&self.bin_path),
            raw: Arc::clone(&self.raw),
            meta: Arc::clone(&self.meta),
            cfg: self.preproc_cfg.clone(),
            settings: self.export_ui.settings.clone(),
            labels: self.classify.labels.clone(),
            classify_chunks: self.classify.n_chunks,
            outside_rule: self.classify.outside_rule,
            events,
            source_settings: crate::settings::load_table(&self.bin_path),
            chunk_samples: None,
        }
    }

    pub(super) fn draw_channel_context_menu(&mut self, ctx: &egui::Context) {
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

    /// Esc closes the topmost open tool window (PSTH, Atlas Registration, Preferences,
    /// ...), one per press; with no window open it closes the waveform view, then
    /// returns from the rectangle zoom. It is left alone while a text field, combo box or the
    /// channel context menu has it. A window with a running job (progress bar) aborts
    /// that job as it closes; a progress window aborts its job and closes itself once
    /// the job has stopped. A running export handles Esc itself (see `draw_frozen`).
    pub(super) fn close_top_window_on_escape(&mut self, ctx: &egui::Context) {
        if !ctx.input(|i| i.key_pressed(egui::Key::Escape))
            || ctx.wants_keyboard_input()
            || ctx.memory(|m| m.any_popup_open())
            || self.context_menu_channel.is_some()
            || self.export_ui.is_running()
        {
            return;
        }
        // General rule: every `egui::Window` other than the main one should close on
        // Escape, topmost first. Area ids are the window titles (see egui::Window::new)
        // unless set with `.id()`, so each one needs an entry here — a window-less
        // "open" flag can't be discovered automatically.
        let title = egui::Id::new;
        let export_result = self.export_ui.result_title();
        let open: Vec<(egui::Id, u8)> = [
            (self.psth.open, title("Peri-Stimulus Time Histogram"), 0),
            (self.atlas.open, title("Atlas Registration"), 1),
            (self.show_remove_channels, title("Remove channels"), 2),
            (self.show_preferences, title("Preferences"), 3),
            (self.events.open, title("Events"), 4),
            (self.classify.error.is_some(), title("Channel Classification failed"), 5),
            (self.spectrum.open, title("Power Spectrum"), 6),
            (self.spectrum.error.is_some(), title("Power Spectrum failed"), 7),
            (self.noise_open, title("Noise Suppression"), 8),
            (self.screenshot.open, title("Screenshot"), 9),
            (self.classify.is_running(), title("Channel Classification"), 10),
            (self.atlas.busy(), title("atlas_progress_window"), 11),
            (self.export_ui.open, title("Export preprocessed data"), 12),
            (export_result.is_some(), title(export_result.unwrap_or_default()), 13),
            (self.show_file_locations, title("Default file locations"), 14),
            (self.spikes.open, title("Spike-sorted data"), 15),
            (self.csd_edit.is_some(), title("CSD"), 16),
        ]
        .into_iter()
        .filter(|(is_open, ..)| *is_open)
        .map(|(_, id, which)| (id, which))
        .collect();
        if open.is_empty() {
            // no window left to close: Esc leaves the waveform zoom, the waveform view,
        // then the heatmap zoom
            if self.waveform_channel.is_some() {
                if !self.leave_waveform_zoom() {
                    self.waveform_channel = None;
                }
                return;
            }
            if let Some(z) = self.zoom.take() {
                self.view_start_s = z.prev_start_s;
                self.view_dur_s = z.prev_dur_s;
                self.window_dur_str = format!("{:.3}", self.view_dur_s);
            }
            return;
        }
        // back-to-front stacking order; a window not in it yet counts as the bottom one
        let depth = |id: egui::Id| ctx.memory(|m| m.layer_ids().position(|l| l.id == id));
        let (_, which) = open.into_iter().max_by_key(|(id, _)| depth(*id)).unwrap();
        match which {
            0 => {
                self.psth.open = false;
                if let Some(job) = &self.psth.job {
                    job.cancel();
                }
            }
            1 => self.atlas.open = false,
            2 => self.show_remove_channels = false,
            3 => self.show_preferences = false,
            4 => self.events.open = false,
            5 => self.classify.error = None,
            6 => {
                self.spectrum.open = false;
                self.spectrum.cancel();
            }
            7 => self.spectrum.error = None,
            8 => {
                self.noise_open = false;
                self.notch_panel.abort();
            }
            9 => self.screenshot.open = false,
            10 => self.classify.cancel(),
            11 => self.atlas.abort(),
            12 => self.export_ui.open = false,
            13 => self.export_ui.close_result(),
            14 => self.show_file_locations = false,
            15 => self.spikes.open = false,
            _ => self.csd_edit = None,
        }
    }
}
