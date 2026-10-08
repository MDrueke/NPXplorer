//! The "Export preprocessed data" window, the save dialog, and the frozen main
//! window shown while an export runs: an in-memory screenshot of the window,
//! darkened and without colour, under a progress panel. The screenshot never touches
//! the disk and is freed (image and GPU texture) when the export ends.

use egui::{Color32, RichText, Ui};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Instant;

use crate::export::{BadChannelAction, BlankMode, ExportSettings, Format, Handle, Outcome, Phase};

const ERROR_RED: Color32 = Color32::from_rgb(0xff, 0x66, 0x66);
const WARN_ORANGE: Color32 = Color32::from_rgb(0xff, 0xb6, 0x17);
/// Brightness of the frozen screenshot (luminance × this).
const FREEZE_DIM: f32 = 0.45;

/// Marks the screenshot requested for the freeze, so other screenshot handlers
/// (Screenshot window, PSTH PNG) leave it alone.
pub struct FreezeShot;

/// What the window needs to know about the open recording.
pub struct WindowInputs<'a> {
    pub meta: &'a crate::data::Meta,
    pub src: &'a std::path::Path,
    pub removed: &'a std::collections::BTreeSet<usize>,
    pub labels: Option<&'a [u8]>,
    pub has_events: bool,
    pub view_s: (f64, f64),
    /// the preprocessing steps that will be applied, for display
    pub steps: String,
}

/// What the app should do after drawing.
pub enum Request {
    None,
    /// the user picked the destination: build the job and call `start`
    Export(PathBuf),
    OpenEventsWindow,
    /// open the exported recording
    Open(PathBuf),
    /// classification labels computed by the export
    Labels(Vec<u8>),
}

struct Running {
    handle: Handle,
    /// the frozen look, once the screenshot has arrived
    texture: Option<egui::TextureHandle>,
    shot_requested_at: Instant,
    writing_since: Option<Instant>,
    dst: PathBuf,
    /// close the app once the export has stopped (close was requested meanwhile)
    close_after: bool,
}

pub struct ExportUi {
    pub open: bool,
    pub settings: ExportSettings,
    /// time range as edited (s)
    from_s: f64,
    to_s: f64,
    errors: Vec<String>,
    /// the user pressed Export without an event file while blanking is on
    events_missing: bool,
    pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    pick_format: Format,
    running: Option<Running>,
    result: Option<(Outcome, PathBuf)>,
    /// free space at the source's folder (bytes), measured when the window opens
    free_bytes: Option<u64>,
}

impl ExportUi {
    pub fn new(settings: ExportSettings, total_s: f64) -> Self {
        let [from_s, to_s] = settings.range_s.unwrap_or([0.0, total_s]);
        ExportUi {
            open: false,
            settings,
            from_s,
            to_s: to_s.min(total_s),
            errors: Vec::new(),
            events_missing: false,
            pick_rx: None,
            pick_format: Format::SpikeGlx,
            running: None,
            result: None,
            free_bytes: None,
        }
    }

    pub fn open_window(&mut self, src: &std::path::Path) {
        self.open = true;
        self.errors.clear();
        self.events_missing = false;
        self.free_bytes = free_space(src);
    }

    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    /// Title of the finished / failed / cancelled window, if it is shown.
    pub fn result_title(&self) -> Option<&'static str> {
        self.result.as_ref().map(|(outcome, _)| match outcome {
            Outcome::Done(_) => "Export finished",
            Outcome::Cancelled => "Export cancelled",
            Outcome::Failed(_) => "Export failed",
        })
    }

    pub fn close_result(&mut self) {
        self.result = None;
    }

    // -----------------------------------------------------------------------
    // The export window
    // -----------------------------------------------------------------------

    pub fn draw_window(&mut self, ctx: &egui::Context, inp: &WindowInputs) -> Request {
        let mut req = Request::None;
        if let Some(r) = self.poll_pick() {
            return r;
        }
        if let Some(r) = self.draw_result(ctx) {
            return r;
        }
        if !self.open || self.running.is_some() {
            return Request::None;
        }
        let total_s = inp.meta.n_samples as f64 / inp.meta.sample_rate;
        let format = Format::of(inp.src);
        let mut open = self.open;
        egui::Window::new("Export preprocessed data")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(460.0)
            .show(ctx, |ui| {
                req = self.draw_contents(ui, inp, total_s, format);
            });
        self.open = open && self.open;
        req
    }

    fn draw_contents(&mut self, ui: &mut Ui, inp: &WindowInputs, total_s: f64, format: Format) -> Request {
        let mut req = Request::None;
        let s = &mut self.settings;
        let weak = |ui: &mut Ui, t: &str| {
            ui.label(RichText::new(t).small().color(Color32::GRAY));
        };

        // time range
        ui.label(RichText::new("Time range").strong());
        ui.horizontal(|ui| {
            ui.label("From");
            ui.add(egui::DragValue::new(&mut self.from_s).speed(0.1).range(0.0..=total_s).max_decimals(3));
            ui.label("s");
            ui.add_space(8.0);
            ui.label("To");
            ui.add(egui::DragValue::new(&mut self.to_s).speed(0.1).range(0.0..=total_s).max_decimals(3));
            ui.label("s");
        });
        ui.horizontal(|ui| {
            if ui.button("Whole recording").clicked() {
                self.from_s = 0.0;
                self.to_s = total_s;
            }
            if ui.button("Current view").clicked() {
                self.from_s = inp.view_s.0.clamp(0.0, total_s);
                self.to_s = (inp.view_s.0 + inp.view_s.1).clamp(0.0, total_s);
            }
        });
        s.range_s = {
            let whole = self.from_s <= 0.0 && self.to_s >= total_s - 0.5 / inp.meta.sample_rate;
            (!whole).then_some([self.from_s, self.to_s])
        };

        // channels
        ui.separator();
        ui.label(RichText::new("Channels").strong());
        let needs_labels = s.needs_labels();
        let n_out = (!needs_labels || inp.labels.is_some())
            .then(|| crate::export::output_channel_count(inp.meta, inp.removed, inp.labels, s));
        match n_out {
            Some(n) => ui.label(format!(
                "{n} of {} channels ({} removed in Remove channels)",
                inp.meta.n_ap_chans,
                inp.removed.len()
            )),
            None => ui.label(format!(
                "{} channels minus the bad channels found by the classification",
                inp.meta.n_ap_chans - inp.removed.len()
            )),
        };
        ui.label("Bad channels (from the channel classification):").on_hover_text(
            "Uses the labels of Preferences → Channel Classification. If it hasn't been run for this recording, it runs before the export, with the settings in Preferences.",
        );
        let action_row = |ui: &mut Ui, label: &str, a: &mut BadChannelAction, hover: &str| {
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.label(label).on_hover_text(hover);
                ui.radio_value(a, BadChannelAction::Keep, "Keep");
                ui.radio_value(a, BadChannelAction::Remove, "Remove");
                ui.radio_value(a, BadChannelAction::Interpolate, "Interpolate")
                    .on_hover_text("Rebuilt from the neighbouring channels of the same shank after preprocessing (IBL's interpolate_bad_channels); left out of CMR and destripe.");
            });
        };
        action_row(ui, "Dead:", &mut s.dead, "Channels with (almost) no signal.");
        action_row(ui, "Noisy:", &mut s.noisy, "Channels much noisier than their neighbours.");
        ui.horizontal(|ui| {
            ui.add_space(12.0);
            ui.checkbox(&mut s.remove_outside, "Remove channels outside the brain").on_hover_text(
                "The outside-of-brain detection can be wrong (e.g. in shallow recordings): check the classification overlay first.",
            );
        });
        if needs_labels && inp.labels.is_none() {
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                weak(ui, "Not classified yet: the classification runs before the export.");
            });
        }
        ui.horizontal(|ui| {
            ui.checkbox(&mut s.average_depths, "Average same-depth channels");
            ui.label(RichText::new("breaks probe geometry for spike sorters and most other analyses").small().color(WARN_ORANGE));
        });

        // event blanking
        ui.separator();
        ui.horizontal(|ui| {
            ui.checkbox(&mut s.blank.enabled, RichText::new("Event blanking").strong())
                .on_hover_text("Removes stimulation artifacts around the events of the Events window, before notch, highpass and spatial filters, so the filters don't spread them.");
            if !inp.has_events {
                weak(ui, "(no event file loaded)");
            }
        });
        ui.add_enabled_ui(s.blank.enabled, |ui| {
            let b = &mut s.blank;
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.radio_value(&mut b.mode, BlankMode::Zero, "Blank").on_hover_text("Set the samples in the window to 0.");
                ui.radio_value(&mut b.mode, BlankMode::Interpolate, "Interpolate")
                    .on_hover_text("A straight line per channel from the sample before the window to the one after it.");
            });
            let window = |ui: &mut Ui, start: &mut f64, end: &mut f64| {
                ui.label("Start");
                ui.add(egui::DragValue::new(start).speed(0.1).max_decimals(3));
                ui.label("ms");
                ui.label("End");
                ui.add(egui::DragValue::new(end).speed(0.1).max_decimals(3));
                ui.label("ms");
            };
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.label("Around onsets:");
                window(ui, &mut b.onset_start_ms, &mut b.onset_end_ms);
            });
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.checkbox(&mut b.offsets, "Around offsets:");
                ui.add_enabled_ui(b.offsets, |ui| window(ui, &mut b.offset_start_ms, &mut b.offset_end_ms));
            });
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.add_enabled_ui(b.offsets, |ui| {
                    ui.label("Event length");
                    ui.add(egui::DragValue::new(&mut b.event_length_ms).speed(1.0).range(0.0..=f64::MAX).max_decimals(3));
                    ui.label("ms");
                });
                weak(ui, "used when the event file has no offset column");
            });
        });

        // preprocessing and output
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Preprocessing:").strong());
            ui.label(if inp.steps.is_empty() { "none (raw data)" } else { inp.steps.as_str() });
        });
        weak(ui, "Noise suppression is a display-only filter and is not exported.");
        ui.checkbox(&mut s.probe_file, "Write probe file for Kilosort 4 (.json)");
        ui.checkbox(&mut s.open_when_done, "Open export when done");

        ui.separator();
        let mut export_clicked = false;
        let n_sync = inp.meta.n_saved_chans - inp.meta.n_ap_chans;
        let n_cols = n_out.unwrap_or(inp.meta.n_ap_chans - inp.removed.len()) + n_sync;
        let dur = (self.to_s - self.from_s).max(0.0);
        let size = dur * inp.meta.sample_rate * n_cols as f64 * 2.0;
        ui.horizontal(|ui| {
            ui.label(format!(
                "{} · ≈ {:.1} GB{}",
                match format {
                    Format::SpikeGlx => "SpikeGLX (.bin + .meta)",
                    Format::OpenEphys => "Open Ephys (session folder)",
                },
                size / 1e9,
                self.free_bytes.map(|f| format!(" · {:.0} GB free", f as f64 / 1e9)).unwrap_or_default()
            ));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                export_clicked = ui.button("Export…").clicked();
                if ui.button("Cancel").clicked() {
                    self.open = false;
                }
            });
        });
        if export_clicked && self.validate(inp) && self.pick_rx.is_none() {
            let dst = crate::export::default_destination(inp.src, format, self.settings.range_s);
            self.pick_rx = Some(spawn_save_dialog(dst, format));
            self.pick_format = format;
        }
        for e in &self.errors {
            ui.colored_label(ERROR_RED, e);
        }
        if self.events_missing && ui.button("Open Events window").clicked() {
            req = Request::OpenEventsWindow;
        }
        req
    }

    /// Checks before the save dialog; fills `errors`.
    fn validate(&mut self, inp: &WindowInputs) -> bool {
        self.errors.clear();
        self.events_missing = false;
        let s = &self.settings;
        if self.to_s <= self.from_s {
            self.errors.push("The time range is empty: To must be after From.".into());
        }
        if s.blank.enabled {
            if !inp.has_events {
                self.events_missing = true;
                self.errors.push("Event blanking needs an event file: load one in the Events window.".into());
            }
            if s.blank.onset_end_ms <= s.blank.onset_start_ms || (s.blank.offsets && s.blank.offset_end_ms <= s.blank.offset_start_ms) {
                self.errors.push("Each blanking window's End must be after its Start.".into());
            }
        }
        if (!s.needs_labels() || inp.labels.is_some())
            && crate::export::output_channel_count(inp.meta, inp.removed, inp.labels, s) == 0
        {
            self.errors.push("No channels left to export.".into());
        }
        self.errors.is_empty()
    }

    fn poll_pick(&mut self) -> Option<Request> {
        let rx = self.pick_rx.as_ref()?;
        match rx.try_recv() {
            Ok(picked) => {
                self.pick_rx = None;
                // a SpikeGLX name typed without the extension: the .meta goes next to
                // the .bin, so the extension must be there
                let format = self.pick_format;
                picked.map(|p| match format {
                    Format::SpikeGlx if p.extension().is_none_or(|e| e != "bin") => {
                        let mut s = p.into_os_string();
                        s.push(".bin");
                        Request::Export(PathBuf::from(s))
                    }
                    _ => Request::Export(p),
                })
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.pick_rx = None;
                None
            }
        }
    }

    // -----------------------------------------------------------------------
    // Running: freeze, progress, result
    // -----------------------------------------------------------------------

    /// Start the export and freeze the window. The screenshot of the current frame
    /// arrives with the next one.
    pub fn start(&mut self, job: crate::export::Job, ctx: &egui::Context) {
        let dst = job.dst.clone();
        let handle = crate::export::spawn(job, ctx.clone());
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(FreezeShot)));
        self.open = false;
        self.running = Some(Running {
            handle,
            texture: None,
            shot_requested_at: Instant::now(),
            writing_since: None,
            dst,
            close_after: false,
        });
    }

    /// While an export runs: draw the frozen window and the progress panel instead of
    /// the app, and return true. When it ends, unfreeze (freeing the screenshot) and
    /// return false. Requests (labels, opening the export) come back in `out`.
    pub fn draw_frozen(&mut self, ctx: &egui::Context, out: &mut Vec<Request>) -> bool {
        let Some(run) = &mut self.running else { return false };

        // Esc is the Abort button (the app's Esc handling is off while exporting)
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            run.handle.cancel.store(true, Ordering::Relaxed);
        }

        // the screenshot of the last live frame
        if run.texture.is_none() {
            let shot = ctx.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Screenshot { image, user_data, .. }
                        if user_data.data.as_ref().is_some_and(|d| d.as_ref().is::<FreezeShot>()) =>
                    {
                        Some(image.clone())
                    }
                    _ => None,
                })
            });
            if let Some(image) = shot {
                let grey = desaturate_dim(&image);
                drop(image);
                run.texture = Some(ctx.load_texture("export_freeze", grey, egui::TextureOptions::LINEAR));
            } else if run.shot_requested_at.elapsed().as_secs_f32() < 1.0 {
                // one more live frame for the screenshot to be taken of
                ctx.request_repaint();
                return false;
            }
        }

        // closing the app: stop the export first, close when it has cleaned up
        if ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            run.handle.cancel.store(true, Ordering::Relaxed);
            run.close_after = true;
        }

        if let Some((outcome, labels)) = run.handle.poll() {
            let run = self.running.take().unwrap(); // drops the texture: freed
            if let Some(l) = labels {
                out.push(Request::Labels(l));
            }
            if run.close_after {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return false;
            }
            if let Outcome::Done(d) = &outcome {
                if self.settings.open_when_done {
                    out.push(Request::Open(d.data_path.clone()));
                    return false;
                }
            }
            self.result = Some((outcome, run.dst));
            ctx.request_repaint();
            return false;
        }

        // the frozen window, swallowing all input
        let screen = ctx.screen_rect();
        egui::Area::new(egui::Id::new("export_freeze"))
            .order(egui::Order::Foreground)
            .fixed_pos(screen.min)
            .show(ctx, |ui| {
                let (rect, _) = ui.allocate_exact_size(screen.size(), egui::Sense::click_and_drag());
                match &run.texture {
                    Some(t) => {
                        ui.painter().image(t.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
                    }
                    None => {
                        ui.painter().rect_filled(rect, 0.0, Color32::from_gray(12));
                    }
                }
            });
        self.draw_progress(ctx);
        ctx.request_repaint_after(std::time::Duration::from_millis(250));
        true
    }

    fn draw_progress(&mut self, ctx: &egui::Context) {
        let Some(run) = &mut self.running else { return };
        let p = &run.handle.progress;
        let phase = p.phase();
        if phase == Phase::Writing && run.writing_since.is_none() {
            run.writing_since = Some(Instant::now());
        }
        egui::Window::new("Exporting preprocessed data")
            .order(egui::Order::Tooltip)
            .collapsible(false)
            .resizable(false)
            .title_bar(true)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                ui.label(phase.label());
                match phase {
                    Phase::Classifying => {
                        let (d, t) = (p.classify_done.load(Ordering::Relaxed), p.classify_total.load(Ordering::Relaxed));
                        ui.add(egui::ProgressBar::new(d as f32 / t.max(1) as f32).show_percentage());
                        ui.label(format!("{d} / {t} chunks"));
                    }
                    Phase::Writing => {
                        let (d, t) = (p.done.load(Ordering::Relaxed), p.total.load(Ordering::Relaxed));
                        let frac = d as f64 / t.max(1) as f64;
                        ui.add(egui::ProgressBar::new(frac as f32).show_percentage());
                        let el = run.writing_since.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
                        let mb_s = p.bytes.load(Ordering::Relaxed) as f64 / 1e6 / el.max(1e-3);
                        let left = if frac > 0.01 { format!(" · {} left", fmt_duration(el / frac - el)) } else { String::new() };
                        ui.label(format!("{mb_s:.0} MB/s{left}"));
                    }
                    _ => {
                        ui.add(egui::Spinner::new());
                    }
                }
                ui.label(RichText::new(run.dst.display().to_string()).small().color(Color32::GRAY));
                ui.label(RichText::new(format!("elapsed {}", fmt_duration(run.handle.started.elapsed().as_secs_f64()))).small().color(Color32::GRAY));
                if run.handle.cancel.load(Ordering::Relaxed) {
                    ui.label("Stopping…");
                } else if ui.button("Abort").clicked() {
                    run.handle.cancel.store(true, Ordering::Relaxed);
                }
            });
    }

    /// The finished / failed / cancelled message after an export.
    fn draw_result(&mut self, ctx: &egui::Context) -> Option<Request> {
        let title = self.result_title()?;
        let (outcome, dst) = self.result.as_ref()?;
        // Enter is OK; Esc is handled with the other windows
        let mut close = ctx.input(|i| i.key_pressed(egui::Key::Enter));
        let mut req = None;
        egui::Window::new(title)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_max_width(520.0);
                match outcome {
                    Outcome::Done(d) => {
                        ui.label(format!(
                            "{} channels, {:.1} s of data, {:.2} GB, in {}.",
                            d.n_channels,
                            d.duration_s,
                            d.bytes as f64 / 1e9,
                            fmt_duration(d.elapsed_s)
                        ));
                        if d.clipped > 0 {
                            ui.colored_label(WARN_ORANGE, format!("{} samples exceeded the int16 range and were clipped.", d.clipped));
                        }
                        ui.label("Written:");
                        for f in &d.files {
                            ui.label(RichText::new(f.display().to_string()).small().color(Color32::GRAY));
                        }
                    }
                    Outcome::Cancelled => {
                        ui.label(format!("Nothing was kept of {}.", dst.display()));
                    }
                    Outcome::Failed(e) => {
                        ui.colored_label(ERROR_RED, e);
                        ui.label(format!("Nothing was kept of {}.", dst.display()));
                    }
                }
                ui.horizontal(|ui| {
                    if let Outcome::Done(d) = outcome {
                        if ui.button("Open export").clicked() {
                            req = Some(Request::Open(d.data_path.clone()));
                            close = true;
                        }
                    }
                    if ui.button("OK").clicked() {
                        close = true;
                    }
                });
            });
        if close {
            self.result = None;
        }
        req
    }
}

fn fmt_duration(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

/// The screenshot without colour and darker.
fn desaturate_dim(image: &egui::ColorImage) -> egui::ColorImage {
    let pixels: Vec<Color32> = image
        .pixels
        .par_iter()
        .map(|c| {
            let l = 0.299 * c.r() as f32 + 0.587 * c.g() as f32 + 0.114 * c.b() as f32;
            Color32::from_gray((l * FREEZE_DIM) as u8)
        })
        .collect();
    egui::ColorImage { size: image.size, pixels }
}

/// Free space on the disk holding `path`'s folder.
fn free_space(path: &std::path::Path) -> Option<u64> {
    let dir = std::fs::canonicalize(path.parent()?).ok()?;
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|d| dir.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

/// Native save dialog, on a background thread, with the proposed name filled in.
fn spawn_save_dialog(proposed: PathBuf, format: Format) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut dlg = rfd::FileDialog::new()
            .set_title(match format {
                Format::SpikeGlx => "Export preprocessed data",
                Format::OpenEphys => "Export preprocessed data (name of the new session folder)",
            })
            .set_file_name(proposed.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
        if format == Format::SpikeGlx {
            dlg = dlg.add_filter("SpikeGLX binary", &["bin"]);
        }
        if let Some(d) = proposed.parent() {
            dlg = dlg.set_directory(d);
        }
        let _ = tx.send(dlg.save_file());
    });
    rx
}
