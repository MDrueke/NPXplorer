//! Screenshot of the plot area: the window that chooses the overlays and the file,
//! and the capture itself (windows hidden for two frames, then the image is saved).

use super::*;

/// Overlays that can be left out of a screenshot.
#[derive(Clone, Copy, PartialEq)]
pub(super) struct ShotOverlays {
    pub(super) events: bool,
    pub(super) spikes: bool,
    pub(super) atlas: bool,
    pub(super) firing_rate: bool,
    pub(super) spectrum: bool,
    pub(super) classification: bool,
    pub(super) selection: bool,
    pub(super) scale_bar: bool,
}

/// The "Screenshot" window.
#[derive(Default)]
pub(super) struct ScreenshotState {
    pub(super) open: bool,
    /// overlays to include; filled from what is shown when the window opens
    pub(super) include: Option<ShotOverlays>,
    pub(super) pick_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    /// why the last screenshot wasn't saved (after a successful one the window closes)
    pub(super) error: Option<String>,
}

/// A screenshot being taken: the windows, the legend and the hover readout are hidden,
/// the overlays set to the chosen ones, until the image arrives.
pub(super) struct Capture {
    pub(super) path: PathBuf,
    pub(super) include: ShotOverlays,
    /// overlay visibility to restore afterwards
    pub(super) restore: ShotOverlays,
    /// frames drawn so far in screenshot mode; the image is requested in the second,
    /// the first one drawn entirely without windows
    pub(super) frames: u32,
}

/// Tags our screenshot request, so the PSTH export's doesn't take its image.
pub(super) struct MainScreenshot;

impl NPXplorerApp {
    /// Which overlays are shown now.
    pub(super) fn overlay_visibility(&self) -> ShotOverlays {
        ShotOverlays {
            events: self.events.show_overlay,
            spikes: self.spikes.show_overlay,
            atlas: self.atlas.show_overlay,
            firing_rate: self.show_firing_rate_overlay,
            spectrum: self.spectrum.show_overlay,
            classification: self.classify.show_overlay,
            selection: true,
            scale_bar: true,
        }
    }

    pub(super) fn set_overlay_visibility(&mut self, v: ShotOverlays) {
        self.events.show_overlay = v.events;
        self.spikes.show_overlay = v.spikes;
        self.atlas.show_overlay = v.atlas;
        self.show_firing_rate_overlay = v.firing_rate;
        self.spectrum.show_overlay = v.spectrum;
        self.classify.show_overlay = v.classification;
    }

    /// The overlays of the current view in the screenshot, or all of them outside a
    /// screenshot.
    pub(super) fn shot_includes(&self) -> ShotOverlays {
        self.capture.as_ref().map_or(
            ShotOverlays {
                events: true,
                spikes: true,
                atlas: true,
                firing_rate: true,
                spectrum: true,
                classification: true,
                selection: true,
                scale_bar: true,
            },
            |c| c.include,
        )
    }

    /// Default file name: recording, (channel,) start of the view.
    pub(super) fn default_screenshot_name(&self) -> String {
        let stem = self.bin_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let start = self.view_start_s;
        match self.waveform_channel {
            Some(ch) => format!("{stem}_{}_{start:.3}s.png", self.meta.channel_id(ch - 1)),
            None => format!("{stem}_{start:.3}s.png"),
        }
    }

    /// The "Screenshot" window: which computed overlays to include, and Save.
    pub(super) fn draw_screenshot_window(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.screenshot.pick_rx {
            match rx.try_recv() {
                Ok(picked) => {
                    self.screenshot.pick_rx = None;
                    if let Some(mut path) = picked {
                        if path.extension().is_none() {
                            path.set_extension("png");
                        }
                        let include = self.screenshot.include.unwrap_or_else(|| self.overlay_visibility());
                        let restore = self.overlay_visibility();
                        self.set_overlay_visibility(include);
                        self.capture = Some(Capture { path, include, restore, frames: 0 });
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => self.screenshot.pick_rx = None,
            }
        }
        if !self.screenshot.open {
            self.screenshot.include = None; // taken from the view again on the next open
            return;
        }
        let waveform = self.waveform_channel.is_some();
        let spectrum = self.spectrum.result.as_ref().is_some_and(|c| c.matches(&self.preproc_cfg));
        let mut inc = self.screenshot.include.unwrap_or_else(|| self.overlay_visibility());
        let mut open = self.screenshot.open;
        egui::Window::new("Screenshot")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Saves the plot area as a PNG image, without the windows and the legend.");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Include").strong());
                let mut any = false;
                let mut item = |ui: &mut Ui, show: bool, value: &mut bool, label: &str| {
                    if show {
                        ui.checkbox(value, label);
                        any = true;
                    }
                };
                item(ui, self.events.has_data(), &mut inc.events, "Events");
                item(ui, !waveform && self.spikes.has_data(), &mut inc.spikes, "Spikes");
                item(ui, !waveform && self.atlas.has_data(), &mut inc.atlas, "Atlas regions");
                item(ui, !waveform && !self.projection.sums.is_empty(), &mut inc.firing_rate, "Firing rate");
                item(ui, !waveform && spectrum, &mut inc.spectrum, "Power spectrum");
                item(ui, !waveform && self.classify.labels.is_some(), &mut inc.classification, "Channel classification");
                let selected = self.selected_channel_1.is_some() || self.selected_channel_2.is_some();
                item(ui, !waveform && selected, &mut inc.selection, "Selected channels");
                item(ui, !waveform, &mut inc.scale_bar, "Scale bar");
                if !any {
                    ui.label(egui::RichText::new("no overlays calculated").color(egui::Color32::GRAY));
                }
                ui.add_space(4.0);
                let enter = enter_applies(ui);
                let can_save = self.screenshot.pick_rx.is_none();
                if ui.add_enabled(can_save, egui::Button::new("Save…")).clicked() || (enter && can_save) {
                    self.screenshot.error = None;
                    self.screenshot.pick_rx = Some(spawn_png_saver(
                        self.bin_path.parent().map(|p| p.to_path_buf()),
                        self.default_screenshot_name(),
                    ));
                }
                if let Some(e) = &self.screenshot.error {
                    ui.colored_label(egui::Color32::from_rgb(0xff, 0x66, 0x66), e);
                }
            });
        self.screenshot.include = Some(inc);
        self.screenshot.open = open;
    }

    /// Once the requested image has arrived: crop it to the plot area, save it, and
    /// show the overlays as before.
    pub(super) fn poll_capture(&mut self, ctx: &egui::Context) {
        if self.capture.is_none() {
            return;
        }
        let shot = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, user_data, .. }
                    if user_data.data.as_ref().is_some_and(|d| d.as_ref().is::<MainScreenshot>()) =>
                {
                    Some(image.clone())
                }
                _ => None,
            })
        });
        let Some(image) = shot else { return };
        let c = self.capture.take().unwrap();
        self.set_overlay_visibility(c.restore);
        let error = match self.plot_rect {
            Some(rect) => {
                let cropped = image.region(&rect, Some(ctx.pixels_per_point()));
                let [w, h] = cropped.size;
                crate::psth::save_png(&c.path, w, h, cropped.as_raw())
                    .err()
                    .map(|e| format!("screenshot not saved: {e:#}"))
            }
            None => Some("nothing to save: no plot on screen".to_string()),
        };
        // saved: the window is done; on failure it stays open with the reason
        self.screenshot.open = error.is_some();
        self.screenshot.error = error;
    }
}
