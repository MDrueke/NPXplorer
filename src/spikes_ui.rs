//! The "Load Spikes" window and the spike dots drawn on the heatmap.

use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use anyhow::{Context, Result};
use egui::Color32;

use crate::app::RowsOnScreen;
use crate::colormap::ColorMapChoice;
use crate::data::{DisplayRow, Meta};
use crate::spikes::*;

/// More dots than this in the view are not drawn (zoom in to see them).
const MAX_DOTS: usize = 20_000;
/// dot radius (px) in a `DOT_REF_DUR_S` window; it grows in proportion as the time
/// window shrinks, up to `DOT_MAX_RADIUS`
const DOT_RADIUS: f32 = 1.25;
const DOT_REF_DUR_S: f64 = 0.3;
const DOT_MIN_RADIUS: f32 = 0.75;
const DOT_MAX_RADIUS: f32 = 5.0;

const VALUE_HELP: &str = "Units whose value in this column matches are kept. \
Text: one value, or several separated by commas (e.g. Th,CA1). \
Numbers: a comparison, e.g. >50, <=10, =3, !=0.\n\n\
The column is looked up in the cluster tables (cluster_*.tsv, e.g. a label added in phy, or cluster_info.tsv). \
Leave the column empty to skip this filter.";

/// Spike-sorted data of the open recording: the window, loading, filtering and the
/// overlay.
pub struct SpikesState {
    pub open: bool,
    path_text: String,
    /// filters as set in the window
    filters: SpikeSettings,
    data: Option<SortedSpikes>,
    /// indices into `data` of the spikes that pass the filters, in time order
    shown: Vec<u32>,
    n_shown_units: usize,
    pub show_overlay: bool,
    load_rx: Option<mpsc::Receiver<Result<SortedSpikes>>>,
    pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    error: Option<String>,
    /// why a filter was skipped (shown in the window and the legend, nothing pops up)
    note: Option<&'static str>,
    /// spikes in the view last frame when there were too many to draw
    too_many: Option<usize>,
}

impl SpikesState {
    /// `default_dir`: the default location (Preferences), used when the recording has
    /// no saved folder.
    pub fn new(settings: SpikeSettings, default_dir: Option<PathBuf>) -> Self {
        let folder = settings.folder.clone().or(default_dir.filter(|d| d.is_dir()));
        Self {
            open: false,
            path_text: folder.map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
            filters: settings,
            data: None,
            shown: Vec::new(),
            n_shown_units: 0,
            show_overlay: true,
            load_rx: None,
            pick_rx: None,
            error: None,
            note: None,
            too_many: None,
        }
    }

    /// What is saved with the recording.
    pub fn settings(&self) -> SpikeSettings {
        let folder = self.path_text.trim();
        SpikeSettings { folder: (!folder.is_empty()).then(|| PathBuf::from(folder)), ..self.filters.clone() }
    }

    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }

    pub fn n_units(&self) -> usize {
        self.n_shown_units
    }

    fn loading(&self) -> bool {
        self.load_rx.is_some()
    }

    /// Why a filter was skipped, if one was.
    pub fn note(&self) -> Option<&'static str> {
        self.note.filter(|_| self.data.is_some())
    }

    /// Spikes in the view when there were too many to draw.
    pub fn too_many(&self) -> Option<usize> {
        self.too_many.filter(|_| self.show_overlay && self.data.is_some())
    }

    /// Load the folder in the window (Load on start); opens the window if that fails.
    pub fn load_on_start(&mut self, ctx: &egui::Context, meta: &Arc<Meta>) {
        if is_phy_dir(Path::new(self.path_text.trim())) {
            self.apply(ctx, meta);
        }
    }

    /// Load the folder (in the background) unless it is the loaded one, then filter.
    fn apply(&mut self, ctx: &egui::Context, meta: &Arc<Meta>) {
        let dir = PathBuf::from(self.path_text.trim());
        if self.data.as_ref().is_some_and(|d| d.dir == dir) {
            self.refilter();
            return;
        }
        if !is_phy_dir(&dir) {
            self.error = Some(format!("no spike_times.npy in {}", dir.display()));
            self.open = true;
            return;
        }
        let (tx, rx) = mpsc::channel();
        let (meta, ctx) = (Arc::clone(meta), ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(load(&dir, &meta).with_context(|| format!("loading {}", dir.display())));
            ctx.request_repaint();
        });
        self.load_rx = Some(rx);
        self.error = None;
    }

    /// Select the spikes of the units that pass the filters. On failure the window
    /// opens with the reason, and the spikes shown so far stay.
    fn refilter(&mut self) {
        let Some(d) = &self.data else { return };
        match filter_units(&d.dir, &d.units, &self.filters) {
            Ok((keep, note)) => {
                self.note = note;
                let mut mask = vec![false; d.units.last().map_or(0, |&u| u as usize + 1)];
                for &u in &keep {
                    mask[u as usize] = true;
                }
                self.shown = (0..d.clusters.len() as u32).filter(|&i| mask[d.clusters[i as usize] as usize]).collect();
                self.n_shown_units = keep.len();
                self.show_overlay = true;
                self.error = None;
            }
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                self.open = true;
            }
        }
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.pick_rx = None;
                    if let Some(dir) = picked {
                        self.path_text = dir.to_string_lossy().into_owned();
                    }
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => self.pick_rx = None,
            }
        }
        if let Some(rx) = &self.load_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.load_rx = None;
                    match res {
                        Ok(d) => {
                            self.data = Some(d);
                            self.shown.clear();
                            self.n_shown_units = 0;
                            self.refilter();
                        }
                        Err(e) => {
                            self.error = Some(format!("{e:#}"));
                            self.open = true;
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => self.load_rx = None,
            }
        }
    }

    pub fn draw_window(&mut self, ctx: &egui::Context, meta: &Arc<Meta>) {
        if !self.open {
            return;
        }
        let mut apply = false;
        let mut open = self.open;
        egui::Window::new("Spike-sorted data")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Folder:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.path_text)
                            .desired_width(300.0)
                            .hint_text("phy folder (spike_times.npy, ...)"),
                    );
                    if ui.button("Browse…").clicked() && self.pick_rx.is_none() {
                        let start = Some(PathBuf::from(self.path_text.trim())).filter(|d| d.is_dir());
                        let (tx, rx) = mpsc::channel();
                        std::thread::spawn(move || {
                            let mut dlg = rfd::FileDialog::new().set_title("Spike-sorted data (phy folder)");
                            if let Some(d) = start {
                                dlg = dlg.set_directory(d);
                            }
                            let _ = tx.send(dlg.pick_folder());
                        });
                        self.pick_rx = Some(rx);
                    }
                });

                ui.separator();
                ui.label(egui::RichText::new("Filter neurons").strong());
                egui::Grid::new("spike_filter_grid").num_columns(2).show(ui, |ui| {
                    for (label, hover, value) in [
                        ("Curation label (group):", "phy's manual label (cluster_group.tsv)", &mut self.filters.group),
                        ("Kilosort label (KSLabel):", "Kilosort's automatic label (cluster_KSLabel.tsv)", &mut self.filters.ks_label),
                    ] {
                        ui.label(label).on_hover_text(hover);
                        egui::ComboBox::from_id_salt(label).selected_text(value.name()).show_ui(ui, |ui| {
                            for f in LabelFilter::ALL {
                                ui.selectable_value(value, f, f.name());
                            }
                        });
                        ui.end_row();
                    }
                });
                ui.label("User:").on_hover_text(VALUE_HELP);
                ui.horizontal(|ui| {
                    ui.label("Column:");
                    ui.add(egui::TextEdit::singleline(&mut self.filters.user_column).desired_width(110.0).hint_text("e.g. layer"))
                        .on_hover_text(VALUE_HELP);
                    ui.label("Value:");
                    ui.add(egui::TextEdit::singleline(&mut self.filters.user_value).desired_width(110.0).hint_text("e.g. Th or >50"))
                        .on_hover_text(VALUE_HELP);
                });
                ui.label(
                    egui::RichText::new("applied in this order; \"all\" includes noise")
                        .small()
                        .color(Color32::GRAY),
                );

                ui.separator();
                ui.horizontal(|ui| {
                    let loaded = self.data.as_ref().is_some_and(|d| d.dir == Path::new(self.path_text.trim()));
                    let label = if loaded { "Apply" } else { "Load" };
                    let enabled = !self.loading() && !self.path_text.trim().is_empty();
                    let enter = crate::app::enter_applies(ui) && enabled;
                    apply = ui.add_enabled(enabled, egui::Button::new(label)).clicked() || enter;
                    if self.loading() {
                        ui.spinner();
                        ui.label("Loading…");
                    }
                });
                if let Some(err) = &self.error {
                    ui.colored_label(Color32::from_rgb(0xff, 0x66, 0x66), err);
                } else if self.data.is_some() {
                    ui.colored_label(
                        Color32::from_rgb(0x55, 0xdd, 0x77),
                        format!("{} units, {} spikes shown", self.n_shown_units, self.shown.len()),
                    );
                    if let Some(note) = self.note {
                        ui.label(egui::RichText::new(note).small().color(Color32::GRAY));
                    }
                }
            });
        self.open = open;
        if apply {
            self.apply(ctx, meta);
        }
    }

    /// A dot per spike in the view, on its channel's row of the heatmap drawn in `rect`
    /// with rows `first_row..=last_row`. Spikes on removed channels are not drawn.
    pub fn draw_overlay(&mut self, painter: &egui::Painter, shown: RowsOnScreen, (view_start_s, view_dur_s): (f64, f64), cmap: &ColorMapChoice) {
        let RowsOnScreen { rows: display_rows, first: first_row, last: last_row, rect } = shown;
        self.too_many = None;
        let Some(d) = self.data.as_ref().filter(|_| self.show_overlay && view_dur_s > 0.0) else {
            return;
        };
        let time = |i: &u32| d.times_s[*i as usize];
        let lo = self.shown.partition_point(|i| time(i) < view_start_s);
        let hi = self.shown.partition_point(|i| time(i) <= view_start_s + view_dur_s);
        if hi - lo > MAX_DOTS {
            self.too_many = Some(hi - lo);
            return;
        }

        let n_rows = (last_row - first_row + 1) as f32;
        let mut row_y = Vec::new();
        for (r, row) in (first_row..).zip(&display_rows[first_row..=last_row]) {
            if let DisplayRow::Data { channels, .. } = row {
                let y = rect.top() + ((last_row - r) as f32 + 0.5) / n_rows * rect.height();
                for &c in channels {
                    if c >= row_y.len() {
                        row_y.resize(c + 1, None);
                    }
                    row_y[c] = Some(y);
                }
            }
        }

        let spec = cmap.spec();
        let fill = Color32::from_rgb(spec.spike_fill[0], spec.spike_fill[1], spec.spike_fill[2]);
        let [r, g, b] = spec.spike_outline;
        let radius = (DOT_RADIUS * (DOT_REF_DUR_S / view_dur_s) as f32).clamp(DOT_MIN_RADIUS, DOT_MAX_RADIUS);
        let stroke = egui::Stroke::new((radius * 0.4).clamp(0.5, 1.0), Color32::from_rgb(r, g, b));
        for &i in &self.shown[lo..hi] {
            let i = i as usize;
            if let Some(&Some(y)) = row_y.get(d.chans[i] as usize) {
                let x = rect.left() + ((d.times_s[i] - view_start_s) / view_dur_s) as f32 * rect.width();
                painter.circle(egui::pos2(x, y), radius, fill, stroke);
            }
        }
    }
}
