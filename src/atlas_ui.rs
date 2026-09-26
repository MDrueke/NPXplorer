// Atlas Registration window, background load/registration job, and the region
// overlay drawn on the heatmap. The geometry itself lives in atlas.rs.

use egui::{Color32, Ui};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::app::ColorMapChoice;
use crate::atlas::{
    self, az_el_to_polar, polar_to_az_el, AngleConvention, Atlas, Insertion, RegionEdit,
    Registration, PROGRESS_TOTAL,
};
use crate::data::{ChannelOrder, DisplayRow, Meta};

/// Deepest selectable hierarchy level before an atlas has been loaded (the 2017
/// structure tree goes down to about this depth).
const FALLBACK_MAX_LEVEL: u32 = 10;
/// Step of the depth ⏶/⏷ buttons (mm).
const DEPTH_STEP_MM: f64 = 0.01;
/// Opacity (0-255) of region border lines; 51 = 20%. The hovered/dragged line is
/// drawn fully opaque.
const BORDER_ALPHA: u8 = 30;
/// Half-height (px) of the grab zone around a border line.
const BORDER_GRAB_PX: f32 = 4.0;
/// How long the "saved to …" note stays next to the Save button.
const SAVED_NOTE_DURATION: Duration = Duration::from_secs(6);

/// An Alt+drag on a region border, which shifts all borders by changing the depth.
struct DepthDrag {
    start_depth_mm: f64,
    /// probe position (µm) of the row under the pointer when the drag started
    start_y_um: f32,
    shank: u32,
}

struct JobOutput {
    atlas: Arc<Atlas>,
    registration: Registration,
    /// insertion the registration was computed for
    ins: Insertion,
}

pub struct AtlasUi {
    pub open: bool,
    /// atlas folder as typed/pasted or picked
    dir_text: String,
    pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    ins: Insertion,
    show_overlay: bool,
    table_shank: usize,

    atlas: Option<Arc<Atlas>>,
    registration: Option<Arc<Registration>>,
    /// region per raw channel at `ins.level`: the atlas result with `edits` applied
    channel_regions: Vec<Option<usize>>,
    /// corrections made by dragging borders, in brain coordinates — they follow depth
    /// changes and are only cleared by "Reset borders"
    edits: Vec<RegionEdit>,
    depth_drag: Option<DepthDrag>,
    /// insertion the current registration was computed for
    registered_ins: Option<Insertion>,
    meta: Arc<Meta>,
    /// half the electrode pitch per shank (µm): the depth range one edited row covers
    half_pitch: HashMap<u32, f64>,

    job_rx: Option<mpsc::Receiver<Result<Option<JobOutput>, String>>>,
    job_cancel: Arc<AtomicBool>,
    job_progress: Arc<AtomicUsize>,
    error: Option<String>,

    /// insertion changed live (depth slider); written once the mouse is released
    sidecar_dirty: bool,
    /// result of the last CSV save: (message, success, when)
    save_note: Option<(String, bool, Instant)>,

    /// set when a value that lives in the global preferences changed
    prefs_dirty: bool,
}

fn spawn_folder_picker(dir: Option<PathBuf>) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut dlg = rfd::FileDialog::new();
        if let Some(d) = dir {
            dlg = dlg.set_directory(d);
        }
        let _ = tx.send(dlg.pick_folder());
    });
    rx
}

fn color(cmap: &ColorMapChoice, alpha: u8) -> Color32 {
    let [r, g, b] = crate::render::atlas_color(cmap);
    Color32::from_rgba_unmultiplied(r, g, b, alpha)
}

/// Opacity of the boxes behind region labels / table names.
const LABEL_BG_ALPHA: u8 = 170;

fn label_bg(cmap: &ColorMapChoice) -> Color32 {
    let [r, g, b] = crate::render::atlas_label_bg(cmap);
    Color32::from_rgba_unmultiplied(r, g, b, LABEL_BG_ALPHA)
}

impl AtlasUi {
    /// `atlas_dir`/`bregma_lambda_mm` come from the global preferences; a sidecar file
    /// next to the recording, if present, prefills the insertion (its BL distance wins).
    pub fn new(bin_path: &Path, meta: &Arc<Meta>, atlas_dir: Option<String>, bregma_lambda_mm: f64) -> Self {
        let (ins, edits) = atlas::load_sidecar(bin_path)
            .unwrap_or((Insertion { bregma_lambda_mm, ..Default::default() }, Vec::new()));
        Self {
            open: false,
            dir_text: atlas_dir.unwrap_or_default(),
            pick_rx: None,
            ins,
            show_overlay: false,
            table_shank: 0,
            atlas: None,
            registration: None,
            channel_regions: Vec::new(),
            edits,
            depth_drag: None,
            registered_ins: None,
            meta: Arc::clone(meta),
            half_pitch: atlas::half_pitch_per_shank(meta),
            job_rx: None,
            job_cancel: Arc::new(AtomicBool::new(false)),
            job_progress: Arc::new(AtomicUsize::new(0)),
            error: None,
            sidecar_dirty: false,
            save_note: None,
            prefs_dirty: false,
        }
    }

    pub fn atlas_dir(&self) -> Option<String> {
        let d = self.dir_text.trim();
        (!d.is_empty()).then(|| d.to_string())
    }

    pub fn bregma_lambda_mm(&self) -> f64 {
        self.ins.bregma_lambda_mm
    }

    /// true once after a preference-backed value changed; the app then saves prefs
    pub fn take_prefs_dirty(&mut self) -> bool {
        std::mem::take(&mut self.prefs_dirty)
    }

    fn busy(&self) -> bool {
        self.job_rx.is_some()
    }

    fn overlay_visible(&self) -> bool {
        self.show_overlay && self.registration.is_some()
    }

    /// Recompute the per-channel regions: the atlas result at the chosen level, with
    /// the dragged-border edits applied at each channel's current depth in the brain.
    fn refresh_channel_regions(&mut self) {
        let (Some(reg), Some(atlas), Some(ins)) = (&self.registration, &self.atlas, &self.registered_ins) else {
            self.channel_regions = Vec::new();
            return;
        };
        let level = self.ins.level;
        self.channel_regions = reg
            .channel_regions
            .iter()
            .enumerate()
            .map(|(ch, &r)| {
                let g = &self.meta.channel_geom[ch];
                match atlas::edit_at(&self.edits, g.shank, atlas::brain_depth_um(ins, g.y_um)) {
                    Some(e) => e.region.and_then(|id| atlas.tree.row_of_id(id)).map(|r| atlas.tree.at_level(r, level)),
                    None => r.map(|r| atlas.tree.at_level(r, level)),
                }
            })
            .collect();
    }

    /// Record that the rows' channels now belong to `region` (a structure-tree row at
    /// the current level): stored as depth ranges in the brain, so the correction
    /// moves along with the atlas when the depth changes.
    fn record_edit(&mut self, display_rows: &[DisplayRow], rows: &[usize], region: Option<usize>) {
        let (Some(reg), Some(atlas), Some(ins)) = (&self.registration, &self.atlas, &self.registered_ins) else {
            return;
        };
        let region_id = region.map(|r| atlas.tree.rows[r].id);
        for &r in rows {
            let DisplayRow::Data { channels, .. } = &display_rows[r] else { continue };
            for &ch in channels {
                let g = &self.meta.channel_geom[ch];
                let d = atlas::brain_depth_um(ins, g.y_um);
                let h = self.half_pitch.get(&g.shank).copied().unwrap_or(10.0);
                // an edit back to what the atlas says just removes the correction
                let base = reg.channel_regions[ch].map(|r| atlas.tree.at_level(r, self.ins.level));
                let edit = RegionEdit { shank: g.shank, from_um: d - h, to_um: d + h, region: region_id };
                atlas::upsert_edit(&mut self.edits, edit, base != region);
            }
        }
        self.refresh_channel_regions();
        self.sidecar_dirty = true;
    }

    fn save_sidecar(&mut self, bin_path: &Path) {
        self.sidecar_dirty = false;
        if let Err(e) = atlas::save_sidecar(bin_path, &self.ins, &self.edits) {
            self.error = Some(format!("{e:#}"));
        }
    }

    fn region_names(&self, region: Option<usize>) -> (&str, &str) {
        match (region, &self.atlas) {
            (Some(r), Some(atlas)) => (atlas.tree.rows[r].acronym.as_str(), atlas.tree.rows[r].name.as_str()),
            _ => ("outside", "Outside the brain"),
        }
    }

    // -----------------------------------------------------------------------
    // background job
    // -----------------------------------------------------------------------

    /// Start loading the atlas (if needed) and registering in the background; false if
    /// it couldn't be started (the reason is shown in the window).
    fn dispatch(&mut self, ctx: &egui::Context, meta: &Arc<Meta>, bin_path: &Path) -> bool {
        let Some(dir) = self.atlas_dir().map(PathBuf::from) else {
            self.error = Some("select the folder containing the Allen atlas first".into());
            return false;
        };
        if let Err(e) = atlas::save_sidecar(bin_path, &self.ins, &self.edits) {
            self.error = Some(format!("{e:#}"));
            return false;
        }
        self.sidecar_dirty = false;
        self.prefs_dirty = true;

        self.job_cancel.store(true, Ordering::Relaxed);
        let cancel = Arc::new(AtomicBool::new(false));
        self.job_cancel = Arc::clone(&cancel);
        let progress = Arc::new(AtomicUsize::new(0));
        self.job_progress = Arc::clone(&progress);
        self.error = None;

        // reuse the loaded atlas unless the folder changed
        let loaded = self.atlas.clone().filter(|a| a.dir == dir);
        let meta = Arc::clone(meta);
        let ins = self.ins.clone();
        let job_ins = ins.clone();
        let ctx = ctx.clone();
        let (tx, rx) = mpsc::channel();
        self.job_rx = Some(rx);

        std::thread::spawn(move || {
            let run = || -> anyhow::Result<Option<JobOutput>> {
                let atlas = match loaded {
                    Some(a) => a,
                    None => match Atlas::load(&dir, &cancel, &progress)? {
                        Some(a) => Arc::new(a),
                        None => return Ok(None),
                    },
                };
                progress.store(PROGRESS_TOTAL * 3 / 10, Ordering::Relaxed);
                Ok(atlas::register(&atlas, &meta, &ins, &cancel, &progress)?
                    .map(|registration| JobOutput { atlas, registration, ins: job_ins }))
            };
            let _ = tx.send(run().map_err(|e| format!("{e:#}")));
            ctx.request_repaint();
        });
        true
    }

    /// Re-register synchronously with the already-loaded atlas — fast enough (well
    /// under a millisecond) to follow the depth slider live.
    fn register_now(&mut self, meta: &Meta) {
        let Some(atlas) = self.atlas.clone() else { return };
        match atlas::register(&atlas, meta, &self.ins, &AtomicBool::new(false), &AtomicUsize::new(0)) {
            Ok(Some(reg)) => {
                self.registration = Some(Arc::new(reg));
                self.registered_ins = Some(self.ins.clone());
                self.refresh_channel_regions();
                self.error = None;
            }
            Ok(None) => {}
            Err(e) => self.error = Some(format!("{e:#}")),
        }
        self.sidecar_dirty = true;
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.job_rx else { return };
        match rx.try_recv() {
            Ok(res) => {
                self.job_rx = None;
                match res {
                    Ok(Some(out)) => {
                        self.atlas = Some(out.atlas);
                        self.registration = Some(Arc::new(out.registration));
                        self.registered_ins = Some(out.ins);
                        self.show_overlay = true;
                        self.refresh_channel_regions();
                    }
                    Ok(None) => {} // aborted
                    Err(e) => {
                        self.error = Some(e);
                        if self.registration.is_none() {
                            self.show_overlay = false;
                        }
                    }
                }
            }
            Err(mpsc::TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(80));
            }
            Err(mpsc::TryRecvError::Disconnected) => self.job_rx = None,
        }
    }

    pub fn draw_progress_window(&mut self, ctx: &egui::Context) {
        if !self.busy() {
            return;
        }
        let done = self.job_progress.load(Ordering::Relaxed);
        let stage = if done < PROGRESS_TOTAL * 3 / 10 {
            "Loading atlas…"
        } else {
            "Registering probe trajectory…"
        };
        egui::Window::new("Atlas Registration")
            .id(egui::Id::new("atlas_progress_window"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(220.0);
                ui.label(stage);
                ui.add(egui::ProgressBar::new(done as f32 / PROGRESS_TOTAL as f32).show_percentage());
                if ui.button("Abort").clicked() {
                    self.job_cancel.store(true, Ordering::Relaxed);
                }
            });
    }

    /// Write `<recording>_regions.csv` (channel ID from the meta file, region acronym)
    /// next to the data file.
    fn save_csv(&mut self, bin_path: &Path, meta: &Meta) {
        let rows: Vec<(String, String)> = self
            .channel_regions
            .iter()
            .enumerate()
            .map(|(ch, &r)| (meta.channel_id(ch).to_string(), self.region_names(r).0.to_string()))
            .collect();
        self.save_note = Some(match atlas::save_regions_csv(bin_path, &rows) {
            Ok(path) => (format!("saved to {}", path.display()), true, Instant::now()),
            Err(e) => (format!("{e:#}"), false, Instant::now()),
        });
    }

    // -----------------------------------------------------------------------
    // window
    // -----------------------------------------------------------------------

    pub fn draw_window(&mut self, ctx: &egui::Context, meta: &Arc<Meta>, bin_path: &Path, cmap: &ColorMapChoice) {
        if let Some(rx) = &self.pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.pick_rx = None;
                    if let Some(p) = picked {
                        self.dir_text = p.to_string_lossy().into_owned();
                        self.prefs_dirty = true;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => self.pick_rx = None,
            }
        }

        // live depth changes / dragged borders: write the sidecar once the mouse is released
        if self.sidecar_dirty && !ctx.input(|i| i.pointer.any_down()) {
            self.save_sidecar(bin_path);
        }

        let mut open = self.open;
        egui::Window::new("Atlas Registration")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(460.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(ui.ctx().screen_rect().height() - 80.0)
                    .show(ui, |ui| self.window_contents(ui, meta, bin_path, cmap));
            });
        self.open = open;
    }

    fn window_contents(&mut self, ui: &mut Ui, meta: &Arc<Meta>, bin_path: &Path, cmap: &ColorMapChoice) {
        let busy = self.busy();

        ui.label(egui::RichText::new("Atlas").strong());
        ui.horizontal(|ui| {
            ui.label("Folder:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.dir_text)
                    .desired_width(300.0)
                    .hint_text("folder with the Allen CCF .npy files"),
            );
            if resp.lost_focus() {
                self.prefs_dirty = true;
            }
            if ui.button("Browse…").clicked() && self.pick_rx.is_none() {
                self.pick_rx = Some(spawn_folder_picker(self.atlas_dir().map(PathBuf::from)));
            }
        });
        ui.label(
            egui::RichText::new(format!(
                "needs {} and {}",
                atlas::ANNOTATION_FILE,
                atlas::STRUCTURE_TREE_FILE
            ))
            .small()
            .color(Color32::GRAY),
        );
        ui.horizontal(|ui| {
            ui.label("Bregma–lambda distance:");
            if ui
                .add(
                    egui::DragValue::new(&mut self.ins.bregma_lambda_mm)
                        .speed(0.01)
                        .range(2.0..=6.0)
                        .fixed_decimals(2)
                        .suffix(" mm"),
                )
                .changed()
            {
                self.prefs_dirty = true;
            }
        });

        ui.separator();
        ui.label(egui::RichText::new("Insertion").strong());
        ui.horizontal(|ui| {
            ui.label("Angles:");
            ui.radio_value(
                &mut self.ins.angle_convention,
                AngleConvention::AzimuthElevation,
                "Azimuth / elevation / rotation",
            );
            ui.radio_value(
                &mut self.ins.angle_convention,
                AngleConvention::PolarAzimuth,
                "Polar / azimuth / roll",
            );
        });

        let num = |ui: &mut Ui, v: &mut f64, speed: f64, lo: f64, hi: f64, dec: usize, suffix: &str| {
            ui.add(egui::DragValue::new(v).speed(speed).range(lo..=hi).fixed_decimals(dec).suffix(suffix))
                .changed()
        };
        let mut depth_changed = false;
        egui::Grid::new("atlas_insertion_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            ui.label("AP (from bregma, + anterior)");
            num(ui, &mut self.ins.ap_mm, 0.01, -10.0, 10.0, 2, " mm");
            ui.end_row();
            ui.label("ML (from bregma, + right)");
            num(ui, &mut self.ins.ml_mm, 0.01, -10.0, 10.0, 2, " mm");
            ui.end_row();

            match self.ins.angle_convention {
                AngleConvention::AzimuthElevation => {
                    ui.label("Azimuth (from lambda→bregma axis)");
                    num(ui, &mut self.ins.azimuth_deg, 0.5, 0.0, 360.0, 1, "°");
                    ui.end_row();
                    ui.label("Elevation (from horizontal)");
                    num(ui, &mut self.ins.elevation_deg, 0.5, 0.0, 90.0, 1, "°");
                    ui.end_row();
                    ui.label("Rotation (around probe axis)");
                    num(ui, &mut self.ins.rotation_deg, 0.5, -360.0, 360.0, 1, "°");
                    ui.end_row();
                }
                AngleConvention::PolarAzimuth => {
                    let (mut theta, mut phi) = az_el_to_polar(self.ins.azimuth_deg, self.ins.elevation_deg);
                    ui.label("Polar angle (from vertical)");
                    let c1 = num(ui, &mut theta, 0.5, 0.0, 90.0, 1, "°");
                    ui.end_row();
                    ui.label("Azimuth (from +ML, counter-clockwise)");
                    let c2 = num(ui, &mut phi, 0.5, 0.0, 360.0, 1, "°");
                    ui.end_row();
                    if c1 || c2 {
                        let (az, el) = polar_to_az_el(theta, phi);
                        self.ins.azimuth_deg = az;
                        self.ins.elevation_deg = el;
                    }
                    ui.label("Roll (around probe axis)");
                    num(ui, &mut self.ins.rotation_deg, 0.5, -360.0, 360.0, 1, "°");
                    ui.end_row();
                }
            }

            ui.label("Depth (brain surface to tip)");
            ui.horizontal(|ui| {
                let max = atlas::SHANK_LENGTH_UM / 1000.0;
                depth_changed |= num(ui, &mut self.ins.depth_mm, 0.01, 0.0, max, 3, " mm");
                ui.spacing_mut().item_spacing.x = 2.0;
                if ui.small_button("⏶").on_hover_text("10 µm deeper (borders move up)").clicked() {
                    self.ins.depth_mm = (self.ins.depth_mm + DEPTH_STEP_MM).min(max);
                    depth_changed = true;
                }
                if ui.small_button("⏷").on_hover_text("10 µm shallower (borders move down)").clicked() {
                    self.ins.depth_mm = (self.ins.depth_mm - DEPTH_STEP_MM).max(0.0);
                    depth_changed = true;
                }
                ui.add_space(6.0);
                depth_changed |= ui
                    .add(egui::Slider::new(&mut self.ins.depth_mm, 0.0..=max).show_value(false))
                    .changed();
            });
            ui.end_row();
            ui.label("Tip to first electrode row");
            num(ui, &mut self.ins.tip_offset_um, 1.0, 0.0, 1000.0, 0, " µm");
            ui.end_row();
        });
        // once an overlay exists, depth changes are applied immediately
        if depth_changed && self.registration.is_some() && !busy {
            self.register_now(meta);
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Region level:");
            let max_level = self.atlas.as_ref().map_or(FALLBACK_MAX_LEVEL, |a| a.tree.max_depth);
            let label = |l: Option<u32>| match l {
                None => "Finest (incl. cortical layers)".to_string(),
                Some(l) => format!("Hierarchy level {l}"),
            };
            let mut level = self.ins.level;
            egui::ComboBox::from_id_salt("atlas_level_combo")
                .selected_text(label(level))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut level, None, label(None));
                    for l in (1..max_level).rev() {
                        ui.selectable_value(&mut level, Some(l), label(Some(l)));
                    }
                });
            if level != self.ins.level {
                self.ins.level = level;
                self.refresh_channel_regions();
                self.sidecar_dirty = true;
            }
        });

        ui.horizontal(|ui| {
            if ui.add_enabled(!busy, egui::Button::new("Apply")).clicked() {
                self.dispatch(ui.ctx(), meta, bin_path);
            }
            let mut show = self.show_overlay;
            if ui.add_enabled(!busy, egui::Checkbox::new(&mut show, "Show overlay")).changed() {
                self.show_overlay = show;
                if show && self.registration.is_none() && !self.dispatch(ui.ctx(), meta, bin_path) {
                    self.show_overlay = false;
                }
            }
            if ui
                .add_enabled(self.registration.is_some() && !busy, egui::Button::new("Save"))
                .on_hover_text("write each channel's region to a .csv next to the recording")
                .clicked()
            {
                self.save_csv(bin_path, meta);
            }
            if !self.edits.is_empty()
                && ui.button("Reset borders").on_hover_text("undo all dragged region borders").clicked()
            {
                self.edits.clear();
                self.refresh_channel_regions();
                self.sidecar_dirty = true;
            }
        });
        if let Some((msg, ok, when)) = &self.save_note {
            let elapsed = when.elapsed();
            if elapsed < SAVED_NOTE_DURATION {
                let c = if *ok { Color32::from_rgb(0x66, 0xdd, 0x66) } else { Color32::from_rgb(0xff, 0x66, 0x66) };
                ui.colored_label(c, msg);
                ui.ctx().request_repaint_after(SAVED_NOTE_DURATION - elapsed);
            }
        }

        if let Some(err) = &self.error {
            ui.colored_label(Color32::from_rgb(0xff, 0x66, 0x66), err);
        }

        let (Some(reg), Some(atlas)) = (self.registration.clone(), self.atlas.clone()) else {
            return;
        };
        for w in &reg.warnings {
            ui.colored_label(Color32::from_rgb(0xff, 0xcc, 0x55), w);
        }
        let fmt = |p: [f64; 3]| format!("AP {:.2}, ML {:.2}, DV {:.2} mm", p[1], p[0], p[2]);
        ui.label(format!("Brain entry: {}", fmt(reg.entry)));
        ui.label(format!("Tip:  {}", fmt(reg.tip)));

        ui.separator();
        ui.label(egui::RichText::new("Regions along the probe").strong());
        if reg.shanks.len() > 1 {
            self.table_shank = self.table_shank.min(reg.shanks.len() - 1);
            ui.horizontal(|ui| {
                ui.label("Shank:");
                egui::ComboBox::from_id_salt("atlas_shank_combo")
                    .selected_text(format!("{}", reg.shanks[self.table_shank].shank))
                    .show_ui(ui, |ui| {
                        for (i, s) in reg.shanks.iter().enumerate() {
                            ui.selectable_value(&mut self.table_shank, i, format!("{}", s.shank));
                        }
                    });
            });
        }
        let Some(samples) = reg.shanks.get(self.table_shank) else { return };
        let segs = atlas::shank_segments(samples, &atlas.tree, self.ins.level);
        let region_color = color(cmap, 255);
        egui::Grid::new("atlas_region_table").striped(true).num_columns(4).show(ui, |ui| {
            ui.label(egui::RichText::new("Region").strong());
            ui.label(egui::RichText::new("From (µm)").strong());
            ui.label(egui::RichText::new("To (µm)").strong());
            ui.label(egui::RichText::new("Recorded").strong());
            ui.end_row();
            for s in &segs {
                let (acr, name) = self.region_names(s.region);
                ui.label(egui::RichText::new(acr).color(region_color).background_color(label_bg(cmap)))
                    .on_hover_text(name);
                ui.label(format!("{:.0}", s.from_um));
                ui.label(format!("{:.0}", s.to_um));
                ui.label(if s.recorded { "✔" } else { "" });
                ui.end_row();
            }
        });
        ui.label(
            egui::RichText::new(
                "depths along the probe axis from the brain-surface entry point, from the atlas \
                 (borders dragged on the heatmap are not reflected here)",
            )
            .small()
            .color(Color32::GRAY),
        );
    }

    // -----------------------------------------------------------------------
    // heatmap overlay
    // -----------------------------------------------------------------------

    /// Acronym of the region under a raw channel (0-based), for the hover readout.
    pub fn channel_region_label(&self, ch: usize) -> Option<&str> {
        if !self.overlay_visible() {
            return None;
        }
        Some(self.region_names(*self.channel_regions.get(ch)?).0)
    }

    /// Region borders and acronym labels over the heatmap `rect`, which shows
    /// `display_rows[first_row..=last_row]` bottom-to-top. Rows averaged over several
    /// channels take the region of their first channel. Borders can be dragged up or
    /// down (not past the neighboring borders), which reassigns the rows in between;
    /// Alt+drag moves all borders together by adjusting the insertion depth.
    /// Skipped in ID order, where row adjacency isn't spatial.
    pub fn draw_overlay(
        &mut self,
        ui: &Ui,
        meta: &Meta,
        rect: egui::Rect,
        display_rows: &[DisplayRow],
        first_row: usize,
        last_row: usize,
        channel_order: ChannelOrder,
        cmap: &ColorMapChoice,
    ) {
        if !self.overlay_visible() || channel_order == ChannelOrder::Id || last_row < first_row {
            return;
        }
        let n_rows = last_row - first_row + 1;
        let row_h = rect.height() / n_rows as f32;
        let row_top = |r: usize| rect.top() + (last_row - r) as f32 * row_h;
        let row_bottom = |r: usize| row_top(r) + row_h;

        // spans of consecutive data rows (gap rows don't interrupt) with one region
        struct Span {
            shank: u32,
            region: Option<usize>,
            /// display-row indices of the span's data rows, ascending (bottom to top)
            rows: Vec<usize>,
        }
        let mut spans: Vec<Span> = Vec::new();
        for r in first_row..=last_row.min(display_rows.len().saturating_sub(1)) {
            if let DisplayRow::Data { first_ch, shank, .. } = &display_rows[r] {
                let region = self.channel_regions.get(*first_ch).copied().flatten();
                match spans.last_mut() {
                    Some(s) if s.shank == *shank && s.region == region => s.rows.push(r),
                    _ => spans.push(Span { shank: *shank, region, rows: vec![r] }),
                }
            }
        }

        // probe position (y_um) of the data row on `shank` closest to screen height `py`
        let y_um_at = |py: f32, shank: u32| -> Option<f32> {
            (first_row..=last_row.min(display_rows.len().saturating_sub(1)))
                .filter_map(|r| match &display_rows[r] {
                    DisplayRow::Data { shank: s, y_um, .. } if *s == shank => {
                        Some(((row_top(r) + row_bottom(r)) / 2.0 - py).abs()).zip(Some(*y_um))
                    }
                    _ => None,
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, y)| y)
        };
        let alt = ui.input(|i| i.modifiers.alt);

        // borders: grab zones first (so a drag can be applied), then draw
        let mut reassign: Option<(Vec<usize>, Option<usize>)> = None;
        let mut borders: Vec<(f32, bool)> = Vec::new(); // (screen y, hovered/dragged)
        let mut index_in_shank = 0usize;
        for w in spans.windows(2) {
            let (below, above) = (&w[0], &w[1]);
            if below.shank != above.shank {
                index_in_shank = 0;
                continue;
            }
            let border_y = |split: &[usize], k: usize| (row_top(split[k - 1]) + row_bottom(split[k])) / 2.0;
            let rows: Vec<usize> = below.rows.iter().chain(&above.rows).copied().collect();
            let k = below.rows.len();
            let y = border_y(&rows, k);

            let id = ui.id().with(("atlas_border", below.shank, index_in_shank));
            index_in_shank += 1;
            let grab = egui::Rect::from_x_y_ranges(rect.x_range(), (y - BORDER_GRAB_PX)..=(y + BORDER_GRAB_PX));
            let resp = ui.interact(grab, id, egui::Sense::drag());
            let active = resp.hovered() || resp.dragged();
            if active {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
            }
            if resp.drag_started() && alt {
                // Alt+drag: move all borders together by changing the insertion depth
                let start_y_um = resp.interact_pointer_pos().and_then(|p| y_um_at(p.y, below.shank));
                if let Some(start_y_um) = start_y_um {
                    self.depth_drag = Some(DepthDrag {
                        start_depth_mm: self.ins.depth_mm,
                        start_y_um,
                        shank: below.shank,
                    });
                }
            }
            if resp.dragged() && reassign.is_none() && self.depth_drag.is_none() {
                if let Some(p) = resp.interact_pointer_pos() {
                    // nearest split that leaves at least one row on either side
                    let k_new = (1..rows.len())
                        .min_by(|&a, &b| {
                            (border_y(&rows, a) - p.y).abs().total_cmp(&(border_y(&rows, b) - p.y).abs())
                        })
                        .unwrap_or(k);
                    if k_new < k {
                        reassign = Some((rows[k_new..k].to_vec(), above.region));
                    } else if k_new > k {
                        reassign = Some((rows[k..k_new].to_vec(), below.region));
                    }
                }
            }
            borders.push((y, active));
        }

        // Alt+drag in progress: the row under the pointer follows the grabbed border, so
        // the depth changes by the probe distance (µm) the pointer moved along the shank.
        // Tracked here rather than via the grabbed border's response, since re-registering
        // can change which borders exist mid-drag.
        let mut new_depth = None;
        if let Some(d) = &self.depth_drag {
            let (down, pos) = ui.input(|i| (i.pointer.primary_down(), i.pointer.interact_pos()));
            if !down {
                self.depth_drag = None;
            } else {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                if let Some(y_um) = pos.and_then(|p| y_um_at(p.y, d.shank)) {
                    let max = atlas::SHANK_LENGTH_UM / 1000.0;
                    let depth = (d.start_depth_mm + (y_um - d.start_y_um) as f64 / 1000.0).clamp(0.0, max);
                    if (depth - self.ins.depth_mm).abs() > 1e-9 {
                        new_depth = Some(depth);
                    }
                }
            }
        }

        // with Alt held over a border (or while Alt-dragging), all borders light up
        let all_active = self.depth_drag.is_some() || (alt && borders.iter().any(|b| b.1));
        let painter = ui.painter();
        for &(y, active) in &borders {
            let alpha = if active || all_active { 255 } else { BORDER_ALPHA };
            painter.line_segment(
                [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                egui::Stroke::new(1.5_f32, color(cmap, alpha)),
            );
        }

        // labels
        let col = color(cmap, 255);
        let font = egui::FontId::proportional(12.0);
        let bg = label_bg(cmap);
        let hover = ui.ctx().input(|i| i.pointer.hover_pos());
        for (i, s) in spans.iter().enumerate() {
            let (top, bottom) = (row_top(*s.rows.last().unwrap()), row_bottom(s.rows[0]));
            let (acr, name) = self.region_names(s.region);
            let galley = painter.layout_no_wrap(acr.to_string(), font.clone(), col);
            if bottom - top < galley.size().y + 2.0 {
                continue;
            }
            let pos = egui::pos2(rect.left() + 6.0, (top + bottom) / 2.0 - galley.size().y / 2.0);
            let bg_rect = egui::Rect::from_min_size(pos, galley.size()).expand(2.0);
            painter.rect_filled(bg_rect, 2.0, bg);
            painter.galley(pos, galley, col);
            if hover.is_some_and(|p| bg_rect.contains(p)) {
                egui::show_tooltip_text(ui.ctx(), ui.layer_id(), egui::Id::new(("atlas_label", i)), name);
            }
        }

        if let Some(depth) = new_depth {
            self.ins.depth_mm = depth;
            self.register_now(meta);
            return;
        }

        if let Some((rows, region)) = reassign {
            self.record_edit(display_rows, &rows, region);
        }
    }
}
