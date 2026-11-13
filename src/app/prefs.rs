//! Preferences: the settings shared by all recordings, their file, and the
//! Preferences and Default file locations windows.

use super::*;

pub(super) fn default_around_s() -> f64 {
    10.0
}

pub(super) fn default_mem_budget_pct() -> f32 {
    50.0
}

pub(super) fn default_mem_reserve_mb() -> f64 {
    1500.0
}

pub(super) fn default_true() -> bool {
    true
}

pub(super) fn default_spike_overlay_scale() -> f32 {
    1.0
}

pub(super) fn default_spike_smoothing_sigma() -> f32 {
    1.5
}

pub(super) fn default_n_classify_chunks() -> usize {
    crate::channel_classify::DEFAULT_N_CLASSIFY_CHUNKS
}

pub(super) fn default_atlas_min_region_channels() -> usize {
    4
}

pub(super) fn default_bregma_lambda_mm() -> f64 {
    crate::atlas::REFERENCE_BREGMA_LAMBDA_MM
}

pub(super) fn default_spectrum_n_chunks() -> usize {
    100
}

pub(super) fn default_spectrum_freq_min_hz() -> f64 {
    1.0
}

pub(super) fn default_spectrum_freq_max_hz() -> f64 {
    5000.0
}

pub(super) fn default_spectrum_time_end_s() -> f64 {
    60.0
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
    /// seconds kept preprocessed on each side of the view
    #[serde(default = "default_around_s")]
    pub around_s: f64,
    /// keep preprocessing beyond `around_s` until the memory budget is used
    #[serde(default)]
    pub background_fill: bool,
    /// share (%) of available memory the preprocessed cache may use
    #[serde(default = "default_mem_budget_pct")]
    pub mem_budget_pct: f32,
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
    #[serde(default)]
    pub file_locations: FileLocations,
    /// unit filters last used (without the folder)
    #[serde(default)]
    pub spike_filter: crate::spikes::SpikeSettings,
}

/// Where to look for files that belong to a recording, relative to its folder
/// (Preferences → Default file locations).
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FileLocations {
    /// event-times file
    #[serde(alias = "ttl")]
    pub events: String,
    /// phy folder with the spike-sorted data
    pub spikes: String,
    /// folder the export's save dialog starts in (empty: next to the recording)
    pub export: String,
    #[serde(alias = "load_ttl_on_start")]
    pub load_events_on_start: bool,
    pub load_spikes_on_start: bool,
}

impl Default for FileLocations {
    fn default() -> Self {
        Self {
            events: String::new(),
            spikes: "ks".into(),
            export: String::new(),
            load_events_on_start: false,
            load_spikes_on_start: false,
        }
    }
}

impl FileLocations {
    /// `rel` in the recording's folder (`None` if not set).
    pub(super) fn resolve(rel: &str, bin_path: &Path) -> Option<PathBuf> {
        let rel = rel.trim();
        (!rel.is_empty()).then(|| bin_path.parent().unwrap_or(Path::new("")).join(rel))
    }
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
        crate::config::config_dir().join("npxplorer_prefs.toml")
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
            color_mode: self.color_mode,
            color_pct: self.display_sets().0.color_pct,
            color_uv: self.display_sets().0.color_scale,
            colormap_choice: self.colormap_choice.clone(),
            spike_threshold: self.spike_threshold,
            show_firing_rate_overlay: self.show_firing_rate_overlay,
            spike_overlay_scale: self.spike_overlay_scale,
            spike_smoothing_sigma: self.spike_smoothing_sigma,
            n_classify_chunks: self.classify.n_chunks,
            classify_outside_rule: self.classify.outside_rule,
            peak_pooling: self.peak_pooling,
            around_s: self.around_s,
            background_fill: self.background_fill,
            mem_budget_pct: self.mem_budget_pct,
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
            spectrum_time_scope: self.spectrum.time_scope,
            spectrum_source: self.spectrum.source,
            spectrum_scaling: self.spectrum.scaling,
            spectrum_normalization: self.spectrum.normalization,
            spectrum_n_chunks: self.spectrum.n_chunks,
            spectrum_freq_restrict: self.spectrum.freq_restrict,
            spectrum_freq_min_hz: self.spectrum.freq_min_hz,
            spectrum_freq_max_hz: self.spectrum.freq_max_hz,
            spectrum_time_restrict: self.spectrum.time_restrict,
            spectrum_time_start_s: self.spectrum.time_start_s,
            spectrum_time_end_s: self.spectrum.time_end_s,
            file_locations: self.file_locations.clone(),
            spike_filter: crate::spikes::SpikeSettings { folder: None, ..self.spikes.settings() },
        };
        prefs.save();
    }

    /// Preferences → Default file locations: where the event file, the spike-sorted data
    /// and exports go, relative to the recording's folder, whether events and spikes are
    /// loaded when a recording is opened, and the atlas folder.
    pub(super) fn draw_file_locations_window(&mut self, ctx: &egui::Context) {
        if !self.show_file_locations {
            return;
        }
        let before = self.file_locations.clone();
        let mut open = true;
        egui::Window::new("Default file locations")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                let loc = &mut self.file_locations;
                ui.label(
                    egui::RichText::new("relative to the recording's folder (.. goes up a folder, e.g. ../ks); absolute paths work too")
                        .small()
                        .color(egui::Color32::GRAY),
                );
                egui::Grid::new("file_locations_grid").num_columns(2).show(ui, |ui| {
                    ui.label("Event file:");
                    ui.add(egui::TextEdit::singleline(&mut loc.events).desired_width(260.0).hint_text("e.g. events.csv"));
                    ui.end_row();
                    ui.label("Spike-sorted data:");
                    ui.add(egui::TextEdit::singleline(&mut loc.spikes).desired_width(260.0).hint_text("phy folder, e.g. ks"));
                    ui.end_row();
                    ui.label("Export folder:");
                    ui.add(egui::TextEdit::singleline(&mut loc.export).desired_width(260.0).hint_text("empty: next to the recording"));
                    ui.end_row();
                });

                ui.add_space(4.0);
                ui.label(egui::RichText::new("Load on start").strong());
                ui.checkbox(&mut loc.load_events_on_start, "Events")
                    .on_hover_text("Loads the recording's last event file, or else the default one, with the recording's event-file format");
                ui.checkbox(&mut loc.load_spikes_on_start, "Spike-sorted data");

                ui.separator();
                ui.label(egui::RichText::new("Atlas folder").strong());
                self.atlas.dir_editor(ui);
            });
        self.show_file_locations = open;
        if self.file_locations != before {
            self.save_prefs();
        }
    }

    /// The Preferences window (toolbar button "Preferences").
    pub(super) fn draw_preferences_window(&mut self, ctx: &egui::Context) {
        let mut show_prefs = self.show_preferences;
        if show_prefs && self.capture.is_none() {
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
                    if ui.button("Default file locations…").clicked() {
                        self.show_file_locations = !self.show_file_locations;
                    }
                    ui.separator();
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
                            self.psth.tex_dirty = true; // PSTH heatmap tracks the same colormap
                        }
                    });

                    if ui
                        .checkbox(&mut self.peak_pooling, "Peak pooling")
                        .on_hover_text("How a heatmap pixel column shows the many samples it covers (e.g. ~11 samples per column in a 0.5 s window, ~200 in a 10 s window).\n\nOn (peak): each column shows the sample with the largest magnitude, sign kept. A spike keeps its full amplitude at any window length, like an oscilloscope's min/max display. The background looks noisier at long windows, and in %ile mode the colour range follows the values on screen.\n\nOff (mean): each column shows the average of its samples. Smooth, but it acts as a lowpass: a -100 µV spike fades to about -5 µV in a 10 s window, so mostly the LFP remains visible.")
                        .changed()
                    {
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
                            self.save_prefs();
                        }
                    });

                    ui.separator();
                    ui.label(egui::RichText::new("Firing rate overlay").strong());

                    if ui.checkbox(&mut self.show_firing_rate_overlay, "Show firing rate overlay").changed() {
                        self.save_prefs();
                    }

                    ui.horizontal(|ui| {
                        ui.label("Spike Threshold (µV):");
                        if ui.add(egui::DragValue::new(&mut self.spike_threshold).speed(1.0)).changed() {
                            self.save_prefs();
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Overlay scale:");
                        if ui.add(
                            egui::DragValue::new(&mut self.spike_overlay_scale)
                                .speed(0.05).range(0.1..=10.0)
                        ).changed() {
                            self.save_prefs();
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Depth smoothing sigma (channels):");
                        if ui.add(
                            egui::DragValue::new(&mut self.spike_smoothing_sigma)
                                .speed(0.1).range(0.1..=10.0)
                        ).changed() {
                            self.save_prefs();
                        }
                    });

                    ui.separator();
                    ui.label(egui::RichText::new("Channel Classification").strong());

                    ui.horizontal(|ui| {
                        ui.label("Chunks to sample:");
                        if ui.add(
                            egui::DragValue::new(&mut self.classify.n_chunks)
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
                        ui.label("Outside of brain:").on_hover_text(OUTSIDE_RULE_HELP);
                        let mut rule = self.classify.outside_rule;
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
                            .on_hover_text(OUTSIDE_RULE_HELP);
                        if rule != self.classify.outside_rule {
                            self.classify.outside_rule = rule;
                            self.save_prefs();
                        }
                    });

                    ui.separator();
                    ui.label(egui::RichText::new("Performance").strong());

                    ui.horizontal(|ui| {
                        ui.label("Preprocess around view (s):");
                        if ui.add(egui::DragValue::new(&mut self.around_s).speed(0.5).range(0.0..=600.0)).changed() {
                            self.save_prefs();
                        }
                    });
                    if ui.checkbox(&mut self.background_fill, "Background preprocessing")
                        .on_hover_text("Keep preprocessing further from the view (mostly in the scroll direction) while nothing else needs computing, until the memory budget is used. Pauses while Power spectrum, Classification, PSTH or Export run.")
                        .changed()
                    {
                        self.save_prefs();
                    }
                    ui.add_enabled_ui(self.background_fill, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Memory budget (% of available):");
                            if ui.add(egui::DragValue::new(&mut self.mem_budget_pct).speed(1.0).range(1.0..=95.0)).changed() {
                                self.save_prefs();
                            }
                        });
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
                        egui::RichText::new("preprocessed data beyond the view's surroundings is dropped once free memory falls below the reserve")
                            .small().color(egui::Color32::GRAY)
                    );
                        });
                });
        }
        self.show_preferences = show_prefs;
    }
}
