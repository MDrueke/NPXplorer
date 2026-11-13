//! IBL-style channel classification (dead / noisy / outside the brain / good): the
//! background run over the recording, its progress window and the labels it found.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::channel_classify::OutsideRule;
use crate::data::{Meta, RawData};
use crate::jobs::{BackgroundJob, Outcome};
use crate::settings::ClassificationSettings;

pub struct ClassifyUi {
    /// one label per channel (0 good, 1 dead, 2 noisy, 3 outside), once computed
    pub labels: Option<Arc<Vec<u8>>>,
    pub show_overlay: bool,
    /// number of chunks the majority vote is taken over (Preferences)
    pub n_chunks: usize,
    pub outside_rule: OutsideRule,
    job: Option<BackgroundJob<Vec<u8>>>,
    pub error: Option<String>,
}

impl ClassifyUi {
    pub fn new(n_chunks: usize, outside_rule: OutsideRule) -> Self {
        ClassifyUi { labels: None, show_overlay: false, n_chunks, outside_rule, job: None, error: None }
    }

    /// The settings saved with the recording.
    pub fn settings(&self) -> ClassificationSettings {
        ClassificationSettings { n_chunks: self.n_chunks, outside_rule: self.outside_rule, show_overlay: self.show_overlay }
    }

    pub fn apply_settings(&mut self, s: &ClassificationSettings) {
        self.n_chunks = s.n_chunks;
        self.outside_rule = s.outside_rule;
        self.show_overlay = s.show_overlay;
    }

    pub fn is_running(&self) -> bool {
        self.job.is_some()
    }

    pub fn cancel(&self) {
        if let Some(job) = &self.job {
            job.cancel();
        }
    }

    /// Labels computed elsewhere (an export that classified the channels), shown at once.
    pub fn set_labels(&mut self, labels: Vec<u8>) {
        self.labels = Some(Arc::new(labels));
        self.show_overlay = true;
    }

    /// Classify the recording's channels, leaving out `removed`.
    pub fn dispatch(&mut self, ctx: &egui::Context, raw: &Arc<RawData>, meta: &Arc<Meta>, removed: &BTreeSet<usize>) {
        self.error = None;
        let raw = Arc::clone(raw);
        let meta = Arc::clone(meta);
        let removed = removed.clone();
        let n_chunks = self.n_chunks.max(1);
        let outside_rule = self.outside_rule;
        self.job = Some(BackgroundJob::spawn(ctx, n_chunks, move |c| {
            // `None` means cancelled, which the job reports from its cancel flag
            crate::channel_classify::classify_recording(&raw, &meta, &removed, n_chunks, outside_rule, c.control())
                .ok_or_else(|| "cancelled".to_string())
        }));
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        let Some(outcome) = self.job.as_ref().and_then(|j| j.poll(ctx)) else { return };
        self.job = None;
        match outcome {
            Outcome::Done(labels) => {
                self.set_labels(labels);
                self.error = None;
            }
            Outcome::Failed(e) => self.error = Some(e),
            Outcome::Cancelled => {}
        }
    }

    /// Progress while running; afterwards the error, if the run failed.
    pub fn draw_progress_window(&mut self, ctx: &egui::Context) {
        if let Some(job) = &self.job {
            let (done, total) = job.progress();
            egui::Window::new("Channel Classification")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.set_min_width(220.0);
                    ui.label("Scanning recording for dead / noisy / out-of-brain channels…");
                    ui.add(egui::ProgressBar::new(done as f32 / total.max(1) as f32).show_percentage());
                    ui.label(format!("{done} / {total} chunks"));
                    if job.is_cancelling() {
                        ui.label("Stopping…");
                    } else if ui.button("Abort").clicked() {
                        job.cancel();
                    }
                });
            return;
        }

        let Some(err) = self.error.clone() else { return };
        let mut dismiss = false;
        egui::Window::new("Channel Classification failed")
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
}
