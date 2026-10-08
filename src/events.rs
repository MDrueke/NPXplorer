// Events window: loads event on/offset times and shades them on the heatmap and
// the waveform view. The event-file format is the one the PSTH uses (see psth.rs).

use egui::{Color32, Ui};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use crate::colormap::ColorMapChoice;
use crate::psth::{self, EventLayout};

const FORMAT_HELP: &str = "One line per line of the event file (lines starting with # are ignored). \
Lines without 'o' are header rows to skip. The first line with 'o' marks the onset column(s), \
'f' the offset column(s) and 'x' columns to ignore.\n\n\
Edits are saved with the recording's settings and used by both PSTH and Events. \
The default in config/ is not changed.";

const DURATION_HELP: &str = "Width of each shaded area. Only used when the file format marks \
no offset column: offset columns ('f') take precedence.";

/// Events window settings, saved per recording (see settings.rs).
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq)]
#[serde(default)]
pub struct EventsWindowSettings {
    pub duration_ms: f64,
    pub opacity_pct: f32,
    pub emphasize_edges: bool,
    pub show_overlay: bool,
}

impl Default for EventsWindowSettings {
    fn default() -> Self {
        Self { duration_ms: 100.0, opacity_pct: 10.0, emphasize_edges: false, show_overlay: true }
    }
}

/// PSTH window settings, saved per recording; `None` = the window's default
#[derive(serde::Serialize, serde::Deserialize, Clone, Default, PartialEq)]
#[serde(default)]
pub struct PsthSettings {
    pub start_ms: Option<f64>,
    pub end_ms: Option<f64>,
    pub event_t_start: Option<f64>,
    pub event_t_end: Option<f64>,
    pub color_mode: Option<crate::app::ColorMode>,
    pub color_pct: Option<f32>,
    pub color_uv: Option<f32>,
}

pub struct EventsState {
    pub open: bool,
    path_text: String,
    pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    duration_ms: f64,
    /// opacity of the shaded areas, %
    opacity_pct: f32,
    pub(crate) show_overlay: bool,
    emphasize_edges: bool,
    onsets: Vec<f64>,
    /// offsets from 'f' columns, same order as `onsets`
    offsets: Option<Vec<f64>>,
    error: Option<String>,
}

impl EventsState {
    pub fn new(settings: &EventsWindowSettings, event_file: Option<&Path>) -> Self {
        Self {
            open: false,
            path_text: event_file.map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
            pick_rx: None,
            duration_ms: settings.duration_ms,
            opacity_pct: settings.opacity_pct,
            show_overlay: settings.show_overlay,
            emphasize_edges: settings.emphasize_edges,
            onsets: Vec::new(),
            offsets: None,
            error: None,
        }
    }

    pub fn settings(&self) -> EventsWindowSettings {
        EventsWindowSettings {
            duration_ms: self.duration_ms,
            opacity_pct: self.opacity_pct,
            emphasize_edges: self.emphasize_edges,
            show_overlay: self.show_overlay,
        }
    }

    /// Whether the shading is drawn (and so listed in the heatmap legend).
    pub fn overlay_visible(&self) -> bool {
        self.show_overlay && !self.onsets.is_empty()
    }

    /// Whether an event file has been loaded this session (regardless of
    /// `show_overlay`) — gates showing the legend's Show/Hide events toggle.
    pub fn has_data(&self) -> bool {
        !self.onsets.is_empty()
    }

    /// The loaded on- and offset times (s), if an event file has been loaded.
    pub fn times(&self) -> Option<(&[f64], Option<&[f64]>)> {
        (!self.onsets.is_empty()).then(|| (self.onsets.as_slice(), self.offsets.as_deref()))
    }

    /// Returns whether the file was loaded.
    fn load(&mut self, layout_text: &str) -> bool {
        let path = PathBuf::from(self.path_text.trim());
        let res = (|| -> anyhow::Result<(Vec<f64>, Option<Vec<f64>>)> {
            let layout = EventLayout::parse(layout_text)?;
            let onsets = psth::load_event_times(&path, &layout)?;
            let offsets = psth::load_event_offsets(&path, &layout)?;
            if let Some(off) = &offsets {
                if let Some(i) = (0..onsets.len()).find(|&i| off[i] < onsets[i]) {
                    anyhow::bail!(
                        "event {} ends before it starts (onset {} s, offset {} s). \
                         Check the 'o' and 'f' columns in the file format.",
                        i + 1,
                        onsets[i],
                        off[i]
                    );
                }
            }
            Ok((onsets, offsets))
        })();
        match res {
            Ok((onsets, offsets)) => {
                self.onsets = onsets;
                self.offsets = offsets;
                self.show_overlay = true;
                self.error = None;
                true
            }
            Err(e) => {
                self.error = Some(e.to_string());
                false
            }
        }
    }

    /// Onset strictly after `after`, closest one first (`None` if there isn't one).
    fn next_onset(&self, after: f64) -> Option<f64> {
        self.onsets.iter().copied().filter(|&t| t > after + 1e-9).fold(None, |best, t| {
            Some(best.map_or(t, |b: f64| b.min(t)))
        })
    }

    /// Onset strictly before `before`, closest one first (`None` if there isn't one).
    fn prev_onset(&self, before: f64) -> Option<f64> {
        self.onsets.iter().copied().filter(|&t| t < before - 1e-9).fold(None, |best, t| {
            Some(best.map_or(t, |b: f64| b.max(t)))
        })
    }

    /// A successfully loaded file is stored in `event_file`. `view_start_s` is updated
    /// in place by the "jump to event" buttons, centering the targeted event in the view.
    pub fn draw_window(
        &mut self,
        ctx: &egui::Context,
        layout_text: &mut String,
        bin_path: &Path,
        event_file: &mut Option<PathBuf>,
        view_start_s: &mut f64,
        view_dur_s: f64,
        total_s: f64,
    ) {
        if !self.open {
            return;
        }
        let mut loaded = false;
        if let Some(rx) = &self.pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.pick_rx = None;
                    if let Some(p) = picked {
                        self.path_text = p.to_string_lossy().into_owned();
                        loaded |= self.load(layout_text);
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => self.pick_rx = None,
            }
        }

        let mut open = self.open;
        egui::Window::new("Events")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(460.0)
            .show(ctx, |ui| {
                ui.label(egui::RichText::new("Event file").strong());
                let mut load = false;
                ui.horizontal(|ui| {
                    ui.label("File:");
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.path_text)
                            .desired_width(280.0)
                            .hint_text("path to the event-times file"),
                    );
                    load |= resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.button("Browse…").clicked() && self.pick_rx.is_none() {
                        let typed = Path::new(self.path_text.trim()).parent().filter(|d| d.is_dir());
                        let dir = typed.or(bin_path.parent()).map(Path::to_path_buf);
                        self.pick_rx = Some(crate::app::spawn_event_picker(dir));
                    }
                    load |= ui
                        .add_enabled(!self.path_text.trim().is_empty(), egui::Button::new("Load"))
                        .clicked();
                });

                ui.add_space(4.0);
                format_editor(ui, layout_text);

                ui.separator();
                ui.horizontal(|ui| {
                    let from_file = self.offsets.is_some();
                    ui.add_enabled_ui(!from_file, |ui| {
                        ui.label("Duration (ms):")
                            .on_hover_text(DURATION_HELP)
                            .on_disabled_hover_text(DURATION_HELP);
                        ui.add(
                            egui::DragValue::new(&mut self.duration_ms)
                                .speed(1.0)
                                .range(0.0..=1.0e6),
                        )
                        .on_hover_text(DURATION_HELP)
                        .on_disabled_hover_text(DURATION_HELP);
                    });
                    if from_file {
                        ui.label(egui::RichText::new("(offsets from file)").color(Color32::GRAY));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Opacity:");
                    ui.add(egui::Slider::new(&mut self.opacity_pct, 0.0..=100.0).text("%"));
                });
                ui.checkbox(&mut self.show_overlay, "Show overlay");
                ui.checkbox(&mut self.emphasize_edges, "Emphasize on/offset");

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Jump to event:");
                    let center = *view_start_s + view_dur_s / 2.0;
                    let prev = self.prev_onset(center);
                    let next = self.next_onset(center);
                    let max_start = (total_s - view_dur_s).max(0.0);
                    if ui
                        .add_enabled(prev.is_some(), egui::Button::new("◀"))
                        .on_hover_text("Previous event")
                        .clicked()
                    {
                        if let Some(t) = prev {
                            *view_start_s = (t - view_dur_s / 2.0).clamp(0.0, max_start);
                        }
                    }
                    if ui
                        .add_enabled(next.is_some(), egui::Button::new("▶"))
                        .on_hover_text("Next event")
                        .clicked()
                    {
                        if let Some(t) = next {
                            *view_start_s = (t - view_dur_s / 2.0).clamp(0.0, max_start);
                        }
                    }
                });

                if load {
                    loaded |= self.load(layout_text);
                }
                if let Some(err) = &self.error {
                    ui.colored_label(Color32::from_rgb(0xff, 0x66, 0x66), err);
                } else if !self.onsets.is_empty() {
                    ui.horizontal(|ui| {
                        ui.colored_label(Color32::from_rgb(0x55, 0xdd, 0x77), "Loaded");
                        ui.label(format!("{} events", self.onsets.len()));
                    });
                }
            });
        self.open = open;
        if loaded {
            *event_file = Some(PathBuf::from(self.path_text.trim()));
        }
    }

    /// Shade every event overlapping the view in `rect` (x axis = the displayed
    /// time window), with optional full-opacity lines at on- and offsets.
    pub fn draw_overlay(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        view_start_s: f64,
        view_dur_s: f64,
        cmap: &ColorMapChoice,
    ) {
        if !self.overlay_visible() || view_dur_s <= 0.0 {
            return;
        }
        let [r, g, b] = cmap.spec().accent;
        let alpha = (self.opacity_pct / 100.0 * 255.0).round() as u8;
        let fill = Color32::from_rgba_unmultiplied(r, g, b, alpha);
        let edge = egui::Stroke::new(1.0_f32, Color32::from_rgb(r, g, b));
        let view_end_s = view_start_s + view_dur_s;
        let to_x = |t: f64| rect.left() + ((t - view_start_s) / view_dur_s) as f32 * rect.width();
        let dur_s = self.duration_ms / 1000.0;

        for (i, &on) in self.onsets.iter().enumerate() {
            let off = self.offsets.as_ref().map_or(on + dur_s, |o| o[i]);
            if off < view_start_s || on > view_end_s {
                continue;
            }
            let x0 = to_x(on).max(rect.left());
            // at least 1 px, so short events stay visible in long windows
            let x1 = to_x(off).min(rect.right()).max(x0 + 1.0);
            painter.rect_filled(
                egui::Rect::from_x_y_ranges(x0..=x1, rect.y_range()),
                0.0,
                fill,
            );
            if self.emphasize_edges {
                for t in [on, off] {
                    if (view_start_s..=view_end_s).contains(&t) {
                        let x = to_x(t);
                        painter.line_segment(
                            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                            edge,
                        );
                    }
                }
            }
        }
    }
}

/// Multi-line editor for the event-file format, shared by the PSTH and Events windows.
/// The text is saved with the recording's settings; "Reset to default" restores the
/// default from `config/`.
pub fn format_editor(ui: &mut Ui, text: &mut String) {
    ui.horizontal(|ui| {
        ui.label("File format:").on_hover_text(FORMAT_HELP);
        if ui.button("Reset to default").clicked() {
            *text = psth::default_layout_text();
        }
    });
    ui.add(
        egui::TextEdit::multiline(text)
            .code_editor()
            .desired_rows(5)
            .desired_width(f32::INFINITY),
    );
}
