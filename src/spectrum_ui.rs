//! Power spectrum: the settings window, the background computation over the whole
//! recording, and the state of the strip drawn to the right of the heatmap.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::TextureHandle;

use crate::data::{DisplayRow, Meta, RawData};
use crate::jobs::{BackgroundJob, Outcome};
use crate::preprocess::PreprocConfig;
use crate::settings::SpectrumSettings;
use crate::spectrum::{ComputedSpectrum, SpectrumNormalization, SpectrumScaling, SpectrumSource, SpectrumTimeScope};

/// How long the view must stay put before a live (current-view) spectrum is
/// recomputed, so scrolling isn't slowed down by an FFT every frame.
pub const SETTLE: Duration = Duration::from_millis(150);

/// Inputs the strip's texture was built from; it is rebuilt only when they change.
/// Holds the result's `Arc` (compared by pointer) rather than a bare address, so a
/// freed result's address being reused by the next one can't make a stale texture
/// look current.
pub struct TexKey {
    pub result: Arc<ComputedSpectrum>,
    pub size: [usize; 2],
    pub rows: (usize, usize),
    pub scaling: SpectrumScaling,
    pub normalization: SpectrumNormalization,
    pub freq_range: Option<(f32, f32)>,
}

impl PartialEq for TexKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.result, &other.result)
            && self.size == other.size
            && self.rows == other.rows
            && self.scaling == other.scaling
            && self.normalization == other.normalization
            && self.freq_range == other.freq_range
    }
}

pub struct SpectrumUi {
    pub open: bool,
    pub time_scope: SpectrumTimeScope,
    pub source: SpectrumSource,
    pub scaling: SpectrumScaling,
    pub normalization: SpectrumNormalization,
    pub n_chunks: usize,
    /// whole-recording mode: chunks come only from [time_start_s, time_end_s]
    pub time_restrict: bool,
    pub time_start_s: f64,
    pub time_end_s: f64,
    /// the shown band is [freq_min_hz, freq_max_hz] instead of 0..Nyquist
    pub freq_restrict: bool,
    pub freq_min_hz: f64,
    pub freq_max_hz: f64,
    pub show_overlay: bool,
    /// set by "Calculate" with scope = current view: the result is recomputed as the
    /// view scrolls, until the scope changes
    pub want_live: bool,
    pub result: Option<Arc<ComputedSpectrum>>,
    /// whole-recording computation in progress
    job: Option<BackgroundJob<ComputedSpectrum>>,
    pub error: Option<String>,
    /// live recompute waits until the view stops moving: the last (first, n) seen
    /// and when it changed
    pub view_seen: (usize, usize),
    pub view_changed_at: Instant,
    pub texture: Option<TextureHandle>,
    pub tex_key: Option<TexKey>,
    pub pixel_buf: Vec<u8>,
}

impl Default for SpectrumUi {
    fn default() -> Self {
        SpectrumUi {
            open: false,
            time_scope: SpectrumTimeScope::default(),
            source: SpectrumSource::default(),
            scaling: SpectrumScaling::default(),
            normalization: SpectrumNormalization::default(),
            n_chunks: 1,
            time_restrict: false,
            time_start_s: 0.0,
            time_end_s: 0.0,
            freq_restrict: false,
            freq_min_hz: 0.0,
            freq_max_hz: 0.0,
            show_overlay: false,
            want_live: false,
            result: None,
            job: None,
            error: None,
            view_seen: (usize::MAX, 0),
            view_changed_at: Instant::now(),
            texture: None,
            tex_key: None,
            pixel_buf: Vec::new(),
        }
    }
}

impl SpectrumUi {
    pub fn from_settings(s: &SpectrumSettings) -> Self {
        let mut ui = SpectrumUi::default();
        ui.apply_settings(s);
        ui
    }

    /// The settings saved with the recording.
    pub fn settings(&self) -> SpectrumSettings {
        SpectrumSettings {
            time_scope: self.time_scope,
            source: self.source,
            scaling: self.scaling,
            normalization: self.normalization,
            n_chunks: self.n_chunks,
            freq_restrict: self.freq_restrict,
            freq_min_hz: self.freq_min_hz,
            freq_max_hz: self.freq_max_hz,
            time_restrict: self.time_restrict,
            time_start_s: self.time_start_s,
            time_end_s: self.time_end_s,
            show_overlay: self.show_overlay,
        }
    }

    pub fn apply_settings(&mut self, s: &SpectrumSettings) {
        self.time_scope = s.time_scope;
        self.source = s.source;
        self.scaling = s.scaling;
        self.normalization = s.normalization;
        self.n_chunks = s.n_chunks;
        self.freq_restrict = s.freq_restrict;
        self.freq_min_hz = s.freq_min_hz;
        self.freq_max_hz = s.freq_max_hz;
        self.time_restrict = s.time_restrict;
        self.time_start_s = s.time_start_s;
        self.time_end_s = s.time_end_s;
        self.show_overlay = s.show_overlay;
    }

    /// The shown band, `None` = all of it.
    pub fn freq_range(&self) -> Option<(f32, f32)> {
        self.freq_restrict.then_some((self.freq_min_hz as f32, self.freq_max_hz as f32))
    }

    /// Where whole-recording chunks come from, `None` = the whole recording.
    fn time_range(&self) -> Option<(f64, f64)> {
        self.time_restrict.then_some((self.time_start_s, self.time_end_s))
    }

    pub fn is_computing(&self) -> bool {
        self.job.is_some()
    }

    /// Stop a running whole-recording computation.
    pub fn cancel(&self) {
        if let Some(job) = &self.job {
            job.cancel();
        }
    }

    /// Start the whole-recording computation: evenly spaced raw chunks.
    pub fn dispatch(&mut self, ctx: &egui::Context, raw: &Arc<RawData>, meta: &Arc<Meta>, cfg: &PreprocConfig) {
        self.error = None;
        let raw = Arc::clone(raw);
        let meta = Arc::clone(meta);
        let display_rows = meta.build_display_rows(cfg.avg_depths, &cfg.removed_channels, cfg.channel_order, cfg.shank_order);
        let n_chunks = self.n_chunks.max(1);
        let time_range = self.time_range();
        // the row layout this job's rows are indexed by travels with the result, so a
        // result is only ever shown against the layout it was computed for
        let cfg = cfg.clone();
        self.job = Some(BackgroundJob::spawn(ctx, n_chunks, move |c| {
            crate::spectrum::compute_psd_whole_recording(&raw, &meta, &display_rows, n_chunks, time_range, c.control())
                .map(|result| ComputedSpectrum { result, cfg, source: SpectrumSource::Raw, view: None })
        }));
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        let Some(outcome) = self.job.as_ref().and_then(|j| j.poll(ctx)) else { return };
        self.job = None;
        match outcome {
            Outcome::Done(result) => {
                self.result = Some(Arc::new(result));
                self.show_overlay = true;
                self.error = None;
            }
            Outcome::Failed(e) => self.error = Some(e),
            Outcome::Cancelled => {}
        }
    }

    pub fn draw_error_window(&mut self, ctx: &egui::Context) {
        let Some(err) = self.error.clone() else { return };
        let mut dismiss = false;
        egui::Window::new("Power Spectrum failed")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), &err);
                let enter = crate::app::enter_applies(ui);
                if ui.button("OK").clicked() || enter {
                    dismiss = true;
                }
            });
        if dismiss {
            self.error = None;
        }
    }

    /// The settings window opened by the "Power Spectrum" toolbar button. Returns
    /// whether a setting that is saved in the preferences changed.
    pub fn draw_window(&mut self, ctx: &egui::Context, raw: &Arc<RawData>, meta: &Arc<Meta>, cfg: &PreprocConfig) -> bool {
        if !self.open {
            return false;
        }
        let mut open = self.open;
        let mut dirty = false;
        egui::Window::new("Power Spectrum")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Time scope:");
                    let mut scope = self.time_scope;
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
                    if scope != self.time_scope {
                        self.time_scope = scope;
                        self.want_live = false;
                        dirty = true;
                    }
                });

                if self.time_scope == SpectrumTimeScope::CurrentView {
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
                            .add(egui::DragValue::new(&mut self.n_chunks).speed(1.0).range(1..=500))
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

                    let total_s = meta.n_samples as f64 / meta.sample_rate;
                    if ui
                        .checkbox(&mut self.time_restrict, "Restrict time window")
                        .changed()
                    {
                        if self.time_restrict {
                            // seed with the full recording, so the fields start somewhere
                            // sensible instead of an arbitrary fixed default
                            self.time_start_s = 0.0;
                            self.time_end_s = total_s;
                        }
                        dirty = true;
                    }
                    if self.time_restrict {
                        ui.horizontal(|ui| {
                            ui.label("s:");
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.time_start_s)
                                        .speed(1.0)
                                        .range(0.0..=self.time_end_s.clamp(0.0, total_s)),
                                )
                                .changed()
                            {
                                dirty = true;
                            }
                            ui.label("–");
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.time_end_s)
                                        .speed(1.0)
                                        .range(self.time_start_s.clamp(0.0, total_s)..=total_s),
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
                let forced_raw = self.time_scope == SpectrumTimeScope::WholeRecordingChunks;
                ui.horizontal(|ui| {
                    ui.label("Source:");
                    let mut source = if forced_raw { SpectrumSource::Raw } else { self.source };
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
                    } else if source != self.source {
                        self.source = source;
                        dirty = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Scaling:");
                    let mut scaling = self.scaling;
                    egui::ComboBox::from_id_salt("spec_scaling_combo")
                        .selected_text(match scaling {
                            SpectrumScaling::Linear => "Linear power",
                            SpectrumScaling::Db => "dB",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut scaling, SpectrumScaling::Linear, "Linear power");
                            ui.selectable_value(&mut scaling, SpectrumScaling::Db, "dB");
                        });
                    if scaling != self.scaling {
                        self.scaling = scaling;
                        dirty = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Colour range:");
                    let mut norm = self.normalization;
                    egui::ComboBox::from_id_salt("spec_norm_combo")
                        .selected_text(match norm {
                            SpectrumNormalization::PerChannel => "Per-channel",
                            SpectrumNormalization::Global => "Global",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut norm, SpectrumNormalization::PerChannel, "Per-channel");
                            ui.selectable_value(&mut norm, SpectrumNormalization::Global, "Global");
                        });
                    if norm != self.normalization {
                        self.normalization = norm;
                        dirty = true;
                    }
                });

                if ui
                    .checkbox(&mut self.freq_restrict, "Restrict frequency range")
                    .changed()
                {
                    if self.freq_restrict {
                        // seed with the current result's own band, so the fields start
                        // somewhere sensible instead of an arbitrary fixed default
                        let (lo, hi) = self
                            .result
                            .as_ref()
                            .map(|c| crate::render::spectrum_freq_bounds(&c.result.freqs, None))
                            .unwrap_or((1.0, (meta.sample_rate / 2.0) as f32));
                        self.freq_min_hz = lo as f64;
                        self.freq_max_hz = hi as f64;
                    }
                    dirty = true;
                }
                if self.freq_restrict {
                    ui.horizontal(|ui| {
                        let nyquist = meta.sample_rate / 2.0;
                        ui.label("Hz:");
                        if ui
                            .add(
                                egui::DragValue::new(&mut self.freq_min_hz)
                                    .speed(1.0)
                                    .range(0.1..=self.freq_max_hz.clamp(0.1, nyquist)),
                            )
                            .changed()
                        {
                            dirty = true;
                        }
                        ui.label("–");
                        if ui
                            .add(
                                egui::DragValue::new(&mut self.freq_max_hz)
                                    .speed(1.0)
                                    .range(self.freq_min_hz.clamp(0.1, nyquist)..=nyquist),
                            )
                            .changed()
                        {
                            dirty = true;
                        }
                    });
                }

                ui.separator();
                if let Some(job) = &self.job {
                    let (done, total) = job.progress();
                    ui.label("Computing power spectrum across the recording…");
                    ui.add(
                        egui::ProgressBar::new(done as f32 / total.max(1) as f32).show_percentage(),
                    );
                    ui.label(format!("{done} / {total} chunks"));
                    if job.is_cancelling() {
                        ui.label("Stopping…");
                    } else if ui.button("Abort").clicked() {
                        job.cancel();
                    }
                } else {
                    ui.horizontal(|ui| {
                        let enter = crate::app::enter_applies(ui);
                        if ui.button("Calculate").clicked() || enter {
                            self.show_overlay = true;
                            match self.time_scope {
                                SpectrumTimeScope::WholeRecordingChunks => self.dispatch(ui.ctx(), raw, meta, cfg),
                                // computed (immediately: no scrolling to wait for)
                                // by the live recompute in the central panel
                                SpectrumTimeScope::CurrentView => {
                                    self.want_live = true;
                                    let now = std::time::Instant::now();
                                    self.view_changed_at =
                                        now.checked_sub(SETTLE).unwrap_or(now);
                                }
                            }
                        }
                    });

                    if let Some(err) = &self.error {
                        ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), err);
                    } else if self.result.is_some() {
                        ui.colored_label(egui::Color32::from_rgb(0x55, 0xdd, 0x77), "Computed");
                    }
                }
            });
        self.open = open;
        dirty
    }

    /// Draw the strip right of the heatmap `heat`, `size` pixels: one row per display
    /// row `first..=last` of `full_rows` (as in the heatmap, zoom included), with the
    /// frequency axis at the bottom. The texture is rebuilt only when its inputs change.
    pub fn draw_strip(
        &mut self,
        ctx: &egui::Context,
        ui: &egui::Ui,
        heat: egui::Rect,
        size: [usize; 2],
        full_rows: &[DisplayRow],
        (first_row, last_row): (usize, usize),
    ) {
        let Some(computed) = self.result.clone() else { return };
        let shown_rows = if full_rows.is_empty() { full_rows } else { &full_rows[first_row..=last_row] };
        let row_offset = full_rows[..first_row.min(full_rows.len())]
            .iter()
            .filter(|r| matches!(r, DisplayRow::Data { .. }))
            .count();
        let freq_range = self.freq_range();
        let key = TexKey {
            result: Arc::clone(&computed),
            size,
            rows: (first_row, last_row),
            scaling: self.scaling,
            normalization: self.normalization,
            freq_range,
        };
        if self.tex_key.as_ref() != Some(&key) || self.texture.is_none() {
            let style = crate::render::SpectrumStyle { scaling: self.scaling, normalization: self.normalization, freq_range };
            crate::render::build_spectrum_heatmap_into(&mut self.pixel_buf, &computed.result, shown_rows, row_offset, size, &style);
            let img = egui::ColorImage::from_rgba_unmultiplied(size, &self.pixel_buf);
            match &mut self.texture {
                Some(tex) if tex.size() == size => tex.set(img, egui::TextureOptions::NEAREST),
                _ => self.texture = Some(ctx.load_texture("spectrum", img, egui::TextureOptions::NEAREST)),
            }
            self.tex_key = Some(key);
        }
        let Some(tex) = &self.texture else { return };
        let rect = egui::Rect::from_min_max(
            egui::pos2(heat.right(), heat.top()),
            egui::pos2(heat.right() + size[0] as f32, heat.bottom()),
        );
        ui.painter().image(
            tex.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // frequency axis: ticks + labels, black, at the bottom
        let (f_lo, f_hi) = crate::render::spectrum_freq_bounds(&computed.result.freqs, freq_range);
        let log_lo = f_lo.ln();
        let log_span = (f_hi.ln() - log_lo).max(1e-6);
        let painter = ui.painter();
        let font = egui::FontId::proportional(9.0);
        let y_bottom = rect.bottom();
        for f in crate::render::spectrum_ticks(f_lo, f_hi, freq_range.is_some()) {
            let frac = ((f.ln() - log_lo) / log_span).clamp(0.0, 1.0);
            let x = (rect.left() + frac * rect.width()).clamp(rect.left() + 0.5, rect.right() - 0.5);
            painter.line_segment(
                [egui::pos2(x, y_bottom - 5.5), egui::pos2(x, y_bottom - 1.0)],
                egui::Stroke::new(1.0_f32, egui::Color32::BLACK),
            );
            // centred on the tick, but kept inside the panel so labels at the band
            // edges aren't cut off (only the label moves)
            let galley = painter.layout_no_wrap(crate::render::spectrum_tick_label(f), font.clone(), egui::Color32::BLACK);
            let label = galley.size();
            let lx = (x - label.x / 2.0).min(rect.right() - 2.0 - label.x).max(rect.left() + 2.0);
            painter.galley(egui::pos2(lx, y_bottom - 6.5 - label.y), galley, egui::Color32::BLACK);
        }
    }
}
