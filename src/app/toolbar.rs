//! The panels around the plot: toolbar, preprocessing and channel controls,
//! navigation bar and status bar.

use super::*;


impl NPXplorerApp {
    pub(super) fn draw_toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.scope(|ui| {
                zero_item_gap(ui);
                ui.menu_button("File", |ui| {
                    if ui.button("Open").clicked() {
                        self.file_dialog_request = true;
                        ui.close_menu();
                    }
                    if ui.button("Open Zarr folder…").clicked() {
                        self.folder_dialog_request = true;
                        ui.close_menu();
                    }
                    ui.menu_button("Recent files", |ui| {
                        if let Some(path) = crate::draw_recent_files_menu(ui, &self.recent_files) {
                            self.open_recent_request = Some(path);
                            ui.close_menu();
                        }
                    });
                    ui.separator();
                    // an export is written in the format of the original, which needs its
                    // SpikeGLX or Open Ephys metadata files
                    let exportable = crate::zarr::root_of(&self.bin_path).is_none() && !crate::nwb::is_nwb(&self.bin_path);
                    let export = ui
                        .add_enabled(exportable, egui::Button::new("Export preprocessed data…"))
                        .on_disabled_hover_text("Not available for Zarr and NWB recordings.");
                    if export.clicked() {
                        self.export_ui.open_window(&self.bin_path);
                        ui.close_menu();
                    }
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
            let unit_label = format!("±{}", self.unit());

            ui.radio_value(&mut self.color_mode, ColorMode::Percentile, "%ile");
            ui.radio_value(&mut self.color_mode, ColorMode::Voltage, unit_label);

            let csd = self.display_is_csd;
            if self.color_mode == ColorMode::Percentile {
                let (min, step) = if csd { (CSD_PCT_MIN, 0.5) } else { (95.0, 0.1) };
                if ui
                    .add(
                        egui::Slider::new(&mut self.color_pct, min..=100.0)
                            .step_by(step)
                            .text("%"),
                    )
                    .changed()
                {
                    self.color_pct_str = format!("{:.2}", self.color_pct);
                }
            } else if csd {
                ui.add(
                    egui::Slider::new(&mut self.color_uv, CSD_SCALE_RANGE.0..=CSD_SCALE_RANGE.1)
                        .logarithmic(true)
                        .max_decimals(2)
                        .text("mV/mm²"),
                );
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

    pub(super) fn draw_preproc_panel(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Preprocessing:");

            let mut dc = self.preproc_cfg.dc_removal;
            if ui.checkbox(&mut dc, "DC").changed() {
                self.preproc_cfg.dc_removal = dc;
            }

            ui.separator();

            let mut phase = self.preproc_cfg.phase_shift;
            if ui.checkbox(&mut phase, "Phase Shift").changed() {
                self.preproc_cfg.phase_shift = phase;
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
                }
                ui.separator();
            }

            let is_csd = self.preproc_cfg.spatial_filter == SpatialFilter::Csd;
            let csd_bandpass = is_csd && self.preproc_cfg.csd.bandpass;
            let hp_enabled = self.preproc_cfg.spatial_filter != SpatialFilter::Destripe && !csd_bandpass;
            let mut hp = self.preproc_cfg.highpass;
            if ui
                .add_enabled(hp_enabled, egui::Checkbox::new(&mut hp, "300 Hz HP"))
                .on_disabled_hover_text(if csd_bandpass { "Replaced by the CSD bandpass" } else { "Included in destripe" })
                .changed()
            {
                self.preproc_cfg.highpass = hp;
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
            // CSD is chosen in its window (opened here), which applies it
            if ui
                .radio(is_csd, "CSD")
                .on_hover_text("Current source density along depth. Opens its settings; Enter applies.")
                .clicked()
            {
                self.csd_edit = Some(self.preproc_cfg.csd.clone());
            }

            if changed {
                // leaving CSD: give back the options it switched
                if is_csd {
                    if let Some((avg, hp)) = self.csd_restore.take() {
                        self.preproc_cfg.avg_depths = avg;
                        self.preproc_cfg.highpass = hp;
                    }
                }
                if spatial == SpatialFilter::Destripe {
                    self.preproc_cfg.highpass = true;
                }
                self.preproc_cfg.spatial_filter = spatial;
            }

            ui.separator();

            // depth averaging checkbox
            let mut avg = self.preproc_cfg.avg_depths;
            if ui
                .add_enabled(!is_csd, egui::Checkbox::new(&mut avg, "Avg adjacent chans"))
                .on_disabled_hover_text("CSD needs one row per depth")
                .changed()
            {
                self.preproc_cfg.avg_depths = avg;
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

    pub(super) fn draw_channel_controls(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.scope(|ui| {
                zero_item_gap(ui);
                if ui.button("PSTH").clicked() {
                    self.psth.open = true;
                    if self.psth.pick_rx.is_none() {
                        self.psth.pick_rx = Some(spawn_event_picker(
                            self.bin_path.parent().map(|p| p.to_path_buf()),
                        ));
                    }
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                if ui.button("Events").clicked() {
                    self.events.open = !self.events.open;
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                if ui.button("Atlas Registration").clicked() {
                    self.atlas.open = !self.atlas.open;
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                if ui
                    .button("Load Spikes")
                    .on_hover_text("Load spike-sorted data (a phy / Kilosort output folder), filter the units, and show their spikes on the heatmap")
                    .clicked()
                {
                    self.spikes.open = !self.spikes.open;
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                if ui
                    .add_enabled(
                        !self.classify.is_running(),
                        egui::Button::new("Channel Classification"),
                    )
                    .clicked()
                {
                    self.classify.dispatch(ui.ctx(), &self.raw, &self.meta, &self.preproc_cfg.removed_channels);
                }
                ui.add(egui::Separator::default().vertical().spacing(1.0));
                if ui.button("Power Spectrum").clicked() {
                    self.spectrum.open = !self.spectrum.open;
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

            let display_rows_arc = crate::worker::cache_display_rows(&self.worker_state);

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

    pub(super) fn draw_status_bar(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if crate::worker::computing_foreground(&self.worker_state) {
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

    pub(super) fn draw_nav_bar(&mut self, ui: &mut Ui) {
        let total_s = self.meta.n_samples as f64 / self.meta.sample_rate;

        let (response, painter) =
            ui.allocate_painter(Vec2::new(ui.available_width(), 32.0), egui::Sense::click());
        let rect = response.rect;
        let w = rect.width();

        painter.rect_filled(rect, 2.0, egui::Color32::BLACK);

        let [ar, ag, ab] = self.colormap_choice.spec().accent;

        // preprocessed extents, drawn first so they sit beneath the view marker
        for (first, n_samp) in crate::worker::cached_runs(&self.worker_state) {
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
