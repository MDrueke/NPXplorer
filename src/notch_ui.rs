//! The "Notch filters" section of the Noise Suppression window: the scan for noise
//! peaks (in the background), its plot, and the list of notches to apply.

use std::sync::Arc;

use crate::data::{Meta, RawData};
use crate::jobs::{BackgroundJob, Outcome};
use crate::notch::{scan, Notch, Peak, ScanResult, ScanSettings, F_MIN_HZ, MIN_BW_HZ};
use crate::preprocess::PreprocConfig;

// ---------------------------------------------------------------------------
// Window section
// ---------------------------------------------------------------------------

/// "Notch filters" section of the Noise Suppression window.
pub struct NotchPanel {
    settings: ScanSettings,
    job: Option<BackgroundJob<ScanResult>>,
    result: Option<ScanResult>,
    error: Option<String>,
    /// edited list; Apply hands it to the preprocessing
    pub draft: Vec<Notch>,
    manual: Notch,
}

impl NotchPanel {
    pub fn new(applied: &[Notch], settings: ScanSettings) -> Self {
        NotchPanel {
            settings,
            job: None,
            result: None,
            error: None,
            draft: applied.to_vec(),
            manual: Notch { freq_hz: 50.0, bw_hz: 1.0 },
        }
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        let Some(outcome) = self.job.as_ref().and_then(|j| j.poll(ctx)) else { return };
        self.job = None;
        match outcome {
            Outcome::Done(r) => {
                self.result = Some(r);
                self.error = None;
            }
            Outcome::Failed(e) => self.error = Some(e),
            Outcome::Cancelled => {}
        }
    }

    /// Stop a running scan (the window is being closed).
    pub fn abort(&self) {
        if let Some(job) = &self.job {
            job.cancel();
        }
    }

    pub fn scan_settings(&self) -> &ScanSettings {
        &self.settings
    }

    /// Returns the new notch list when Apply was clicked.
    pub fn draw(
        &mut self,
        ui: &mut egui::Ui,
        raw: &Arc<RawData>,
        meta: &Arc<Meta>,
        cfg: &PreprocConfig,
    ) -> Option<Vec<Notch>> {
        ui.label(
            egui::RichText::new(
                "Part of the preprocessing: changes the data everywhere (heatmap, firing rate, \
                 PSTH, spectrum). Saved next to the data file and active again when it is reopened.",
            )
            .small()
            .weak(),
        );

        // --- scan ---
        egui::Grid::new("notch_scan_grid").num_columns(2).show(ui, |ui| {
            ui.label("Chunks to sample:");
            ui.add(egui::DragValue::new(&mut self.settings.n_chunks).range(1..=1000).speed(1.0))
                .on_hover_text("Evenly spaced ~1 s chunks across the recording.");
            ui.end_row();
            ui.label("Min. prominence:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.settings.min_prominence_db).range(1.0..=60.0).speed(0.1))
                    .on_hover_text("How far a peak must stand above the smoothed spectrum.");
                ui.label("dB");
            });
            ui.end_row();
            ui.label("Max. width:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.settings.max_width_hz).range(0.5..=50.0).speed(0.05))
                    .on_hover_text("Peaks wider than this (-3 dB) are not listed: broad peaks are usually brain oscillations, not noise.");
                ui.label("Hz");
            });
            ui.end_row();
            ui.label("Min. channels:");
            ui.horizontal(|ui| {
                let mut pct = self.settings.min_channel_frac * 100.0;
                if ui
                    .add(egui::DragValue::new(&mut pct).range(0.0..=100.0).speed(1.0))
                    .on_hover_text("A frequency is listed once it is a peak on at least this share of one shank's channels.")
                    .changed()
                {
                    self.settings.min_channel_frac = pct / 100.0;
                }
                ui.label("% of a shank");
            });
            ui.end_row();
            ui.label("Min. presence:");
            ui.horizontal(|ui| {
                let mut pct = self.settings.min_presence * 100.0;
                if ui
                    .add(egui::DragValue::new(&mut pct).range(0.0..=100.0).speed(1.0))
                    .on_hover_text("Peaks that appear in fewer chunks are not listed: transient noise (e.g. stimulation artifacts) is no case for a notch on the whole recording. Estimated from how much the peak's power varies between chunks.")
                    .changed()
                {
                    self.settings.min_presence = pct / 100.0;
                }
                ui.label("% of chunks");
            });
            ui.end_row();
        });
        if let Some(job) = &self.job {
            let (done, total) = job.progress();
            ui.horizontal(|ui| {
                ui.add(
                    egui::ProgressBar::new(done as f32 / total.max(1) as f32)
                        .desired_width(220.0)
                        .text(format!("{done} / {total} chunks")),
                );
                if job.is_cancelling() {
                    ui.label("Stopping…");
                } else if ui.button("Abort").clicked() {
                    job.cancel();
                }
            });
        } else if ui
            .button("Find noise peaks")
            .on_hover_text(
                "Averages each channel's power spectrum over chunks of the recording, after the \
                 current preprocessing (without notches), finds narrow peaks on every channel, and \
                 lists those found on enough channels.",
            )
            .clicked()
        {
            let (raw, meta, cfg, settings) = (Arc::clone(raw), Arc::clone(meta), cfg.clone(), self.settings.clone());
            self.job = Some(BackgroundJob::spawn(ui.ctx(), self.settings.n_chunks.max(1), move |c| {
                scan(&raw, &meta, &cfg, &settings, c.control())
            }));
            self.error = None;
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), e);
        }

        // --- results ---
        if let Some(res) = &self.result {
            draw_plot(ui, res, &self.draft);
            if res.peaks.is_empty() {
                ui.label("No narrow peaks found.");
            } else {
                let multi = res.shanks.len() > 1;
                egui::Grid::new("notch_peaks_grid").num_columns(6).striped(true).show(ui, |ui| {
                    ui.label("");
                    ui.label("Frequency");
                    ui.label("Width");
                    ui.label("Channels").on_hover_text("Share of each shank's channels the peak was found on.");
                    ui.label("Prominence").on_hover_text("Above the smoothed spectrum: median (max) over the channels.");
                    ui.label("Present").on_hover_text("Estimated share of chunks in which the peak was there (median over the channels).");
                    ui.end_row();
                    for g in &res.groups {
                        let members: Vec<&Peak> = g.iter().map(|&k| &res.peaks[k]).collect();
                        let has = |draft: &[Notch], p: &Peak| draft.iter().position(|n| (n.freq_hz - p.freq_hz).abs() <= 1.5 * res.df);
                        let n_on = members.iter().filter(|p| has(&self.draft, p).is_some()).count();
                        let mut on = n_on == members.len();
                        // one checkbox for the whole series
                        if ui.add(egui::Checkbox::new(&mut on, "").indeterminate(n_on > 0 && n_on < members.len())).changed() {
                            for p in &members {
                                match has(&self.draft, p) {
                                    Some(i) if !on => {
                                        self.draft.remove(i);
                                    }
                                    None if on => self.draft.push(Notch {
                                        freq_hz: (p.freq_hz * 10.0).round() / 10.0,
                                        bw_hz: p.width_hz.clamp(1.0, 20.0),
                                    }),
                                    _ => {}
                                }
                            }
                        }
                        let first = members[0];
                        let last = members[members.len() - 1];
                        // the shanks' channel shares: range over the lines of a series
                        let mut shank_ids: Vec<u32> = members.iter().flat_map(|p| p.shank_frac.iter().map(|&(s, _)| s)).collect();
                        shank_ids.sort_unstable();
                        shank_ids.dedup();
                        let chans: Vec<String> = shank_ids
                            .iter()
                            .map(|&s| {
                                let fr: Vec<f32> = members
                                    .iter()
                                    .filter_map(|p| p.shank_frac.iter().find(|&&(t, _)| t == s).map(|&(_, f)| f * 100.0))
                                    .collect();
                                let (lo, hi) = fr.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                                let pct = if (hi - lo).abs() < 0.5 { format!("{hi:.0} %") } else { format!("{lo:.0}–{hi:.0} %") };
                                if multi { format!("S{s} {pct}") } else { pct }
                            })
                            .collect();
                        let prom_med = members.iter().map(|p| p.prom_median).fold(f32::MIN, f32::max);
                        let prom_max = members.iter().map(|p| p.prom_max).fold(f32::MIN, f32::max);
                        let mut pres: Vec<f64> = members.iter().map(|p| p.presence as f64).collect();
                        let pres = crate::stats::upper_median(&mut pres).expect("groups have members");
                        let mut widths: Vec<f64> = members.iter().map(|p| p.width_hz).collect();
                        let width = crate::stats::upper_median(&mut widths).expect("groups have members");
                        if members.len() == 1 {
                            let mut f = format!("{:.1} Hz", first.freq_hz);
                            if let Some(f0) = first.harmonic_of {
                                f += &format!("  (×{} of {:.1})", (first.freq_hz / f0).round(), f0);
                            }
                            ui.label(f);
                        } else {
                            // smallest gap, refined over the whole span (lines may be missing)
                            let gap = members.windows(2).map(|w| w[1].freq_hz - w[0].freq_hz).fold(f64::INFINITY, f64::min);
                            let span = last.freq_hz - first.freq_hz;
                            let spacing = span / (span / gap).round().max(1.0);
                            let detail: Vec<String> = members
                                .iter()
                                .map(|p| format!("{:.1} Hz  +{:.0} dB, {} channels", p.freq_hz, p.prom_median, p.n_channels))
                                .collect();
                            ui.label(format!(
                                "{:.1}–{:.1} Hz, every {:.1} Hz ({} lines)",
                                first.freq_hz,
                                last.freq_hz,
                                spacing,
                                members.len()
                            ))
                            .on_hover_text(format!("One source: evenly spaced lines, notched one by one.\n{}", detail.join("\n")));
                        }
                        ui.label(format!("{width:.1} Hz"));
                        ui.label(chans.join(", ")).on_hover_text(format!(
                            "{} channels",
                            members.iter().map(|p| p.n_channels).max().unwrap_or(0)
                        ));
                        ui.label(format!("+{prom_med:.0} dB ({prom_max:.0})"));
                        ui.label(format!("{:.0} %", pres * 100.0));
                        ui.end_row();
                    }
                });
            }
        }

        // --- notch list ---
        ui.add_space(4.0);
        ui.label("Notches:");
        let mut remove = None;
        for (i, n) in self.draft.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut n.freq_hz).range(0.1..=meta.sample_rate / 2.0).speed(0.1));
                ui.label("Hz, width");
                ui.add(egui::DragValue::new(&mut n.bw_hz).range(MIN_BW_HZ..=200.0).speed(0.05))
                    .on_hover_text("-3 dB width. Narrower removes less of the signal but takes longer to settle.");
                ui.label("Hz");
                if ui.small_button("✖").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.draft.remove(i);
        }
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.manual.freq_hz).range(0.1..=meta.sample_rate / 2.0).speed(0.1));
            ui.label("Hz, width");
            ui.add(egui::DragValue::new(&mut self.manual.bw_hz).range(MIN_BW_HZ..=200.0).speed(0.05));
            ui.label("Hz");
            if ui.button("Add").clicked() {
                self.draft.push(self.manual);
            }
        });

        let mut out = None;
        ui.horizontal(|ui| {
            let changed = self.draft != cfg.notches;
            let enter = crate::app::enter_applies(ui) && changed;
            if ui
                .add_enabled(changed, egui::Button::new("Apply notches"))
                .on_disabled_hover_text("Nothing to apply: the list is the one already in use.")
                .clicked()
                || enter
            {
                self.draft.sort_by(|a, b| a.freq_hz.total_cmp(&b.freq_hz));
                out = Some(self.draft.clone());
            }
            // what is in use right now (e.g. loaded from the file when the recording opened)
            let status = if changed {
                "not applied yet".to_string()
            } else if cfg.notches.is_empty() {
                String::new()
            } else if cfg.notch_enabled {
                format!("{} notch{} active (saved next to the data file)", cfg.notches.len(), if cfg.notches.len() == 1 { "" } else { "es" })
            } else {
                format!("{} notch{} saved, switched off (Notch in the Preprocessing bar)", cfg.notches.len(), if cfg.notches.len() == 1 { "" } else { "es" })
            };
            if !status.is_empty() {
                ui.label(egui::RichText::new(status).small().weak());
            }
        });
        out
    }
}

/// Per-shank spectra (dB, log frequency) with the peaks marked; peaks with a notch in
/// the list are drawn solid.
fn draw_plot(ui: &mut egui::Ui, res: &ScanResult, draft: &[Notch]) {
    let n_bins = res.shanks.first().map_or(0, |s| s.psd_db.len());
    if n_bins < 4 {
        return;
    }
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width().max(300.0), 150.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let plot = rect.shrink2(egui::vec2(4.0, 14.0));
    let (f_lo, f_hi) = (F_MIN_HZ.max(res.df), (n_bins - 1) as f64 * res.df);
    let x_of = |f: f64| plot.left() + ((f / f_lo).ln() / (f_hi / f_lo).ln()) as f32 * plot.width();
    let f_of = |x: f32| f_lo * (f_hi / f_lo).powf(((x - plot.left()) / plot.width()) as f64);

    let k_lo = (f_lo / res.df).ceil() as usize;
    let top = res.shanks.iter().flat_map(|s| s.psd_db[k_lo..].iter().copied()).fold(f32::MIN, f32::max);
    let (y_hi, y_lo) = (top + 3.0, top - 60.0);
    let y_of = |db: f32| plot.bottom() - ((db.clamp(y_lo, y_hi) - y_lo) / (y_hi - y_lo)) * plot.height();

    let text = ui.visuals().weak_text_color();
    for f in crate::render::spectrum_ticks(f_lo as f32, f_hi as f32, false) {
        let x = x_of(f as f64);
        painter.line_segment([egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())], egui::Stroke::new(0.5_f32, text.gamma_multiply(0.4)));
        painter.text(egui::pos2(x, rect.bottom() - 1.0), egui::Align2::CENTER_BOTTOM, crate::render::spectrum_tick_label(f), egui::FontId::proportional(10.0), text);
    }

    // one point per pixel column: the column's maximum keeps narrow peaks visible
    let n_cols = plot.width().max(1.0) as usize;
    for (i, s) in res.shanks.iter().enumerate() {
        let t = if res.shanks.len() > 1 { i as f32 / (res.shanks.len() - 1) as f32 } else { 0.8 };
        let [r, g, b] = crate::render::spectrum_color(t);
        let mut pts = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let x0 = plot.left() + c as f32;
            let (ka, kb) = ((f_of(x0) / res.df) as usize, ((f_of(x0 + 1.0) / res.df).ceil() as usize).min(n_bins - 1));
            if let Some(v) = s.psd_db[ka.min(kb)..=kb].iter().copied().reduce(f32::max) {
                pts.push(egui::pos2(x0, y_of(v)));
            }
        }
        painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(r, g, b))));
        if res.shanks.len() > 1 {
            painter.text(
                egui::pos2(rect.right() - 4.0, rect.top() + 1.0 + 11.0 * i as f32),
                egui::Align2::RIGHT_TOP,
                format!("shank {}", s.shank),
                egui::FontId::proportional(10.0),
                egui::Color32::from_rgb(r, g, b),
            );
        }
    }
    for p in &res.peaks {
        let x = x_of(p.freq_hz);
        let on = draft.iter().any(|n| (n.freq_hz - p.freq_hz).abs() <= 1.5 * res.df);
        let col = if on { egui::Color32::from_rgb(0xff, 0x66, 0x66) } else { text };
        painter.line_segment([egui::pos2(x, plot.top() - 10.0), egui::pos2(x, plot.top() - 2.0)], egui::Stroke::new(if on { 2.0_f32 } else { 1.0 }, col));
    }
    if let Some(pos) = resp.hover_pos() {
        if plot.x_range().contains(pos.x) {
            painter.text(rect.left_top() + egui::vec2(4.0, 1.0), egui::Align2::LEFT_TOP, format!("{:.1} Hz", f_of(pos.x)), egui::FontId::proportional(10.0), text);
        }
    }
}
