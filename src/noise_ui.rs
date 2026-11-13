//! The settings part of the "Noise Suppression" window.

use crate::noise::{NoiseSuppression, Wavelet};

/// What the user asked for in the settings window this frame.
#[derive(PartialEq)]
pub enum Action {
    None,
    /// use the draft as it is
    Apply,
    /// switch every step off right away (draft and applied); values are kept
    DisableAll,
}

/// Contents of the "Noise Suppression" window. Edits `draft`; `applied` is only used to
/// show whether there are unapplied edits and whether anything is on.
pub fn draw_settings(ui: &mut egui::Ui, draft: &mut NoiseSuppression, applied: &NoiseSuppression) -> Action {
    ui.label(
        egui::RichText::new(
            "Visual filter: changes the heatmap and the waveform view only. \
             Steps run from top to bottom.",
        )
        .small()
        .weak(),
    );

    // only the number is editable; units and other text are labels next to it
    let pct = |ui: &mut egui::Ui, v: &mut f32, unit: &str| {
        ui.horizontal(|ui| {
            let mut p = *v * 100.0;
            let r = ui.add(egui::DragValue::new(&mut p).range(0.0..=100.0).speed(1.0));
            if r.changed() {
                *v = p / 100.0;
            }
            ui.label(unit);
            r
        })
        .inner
    };
    let uv = |ui: &mut egui::Ui, v: &mut f32| {
        ui.horizontal(|ui| {
            ui.label("±");
            let r = ui.add(egui::DragValue::new(v).range(0.0..=10_000.0).speed(0.5));
            ui.label("µV");
            r
        })
        .inner
    };

    ui.separator();
    ui.checkbox(&mut draft.wavelet_enabled, "Wavelet denoising").on_hover_text(
        "Each channel is decomposed with an orthogonal wavelet transform. Detail coefficients \
         smaller than k × the noise level of their scale (median |d| / 0.6745) are removed \
         (hard) or all are shrunk toward 0 by that amount (soft), then the signal is rebuilt.",
    );
    ui.add_enabled_ui(draft.wavelet_enabled, |ui| {
        egui::Grid::new("noise_wavelet_grid").num_columns(2).show(ui, |ui| {
            ui.label("Wavelet:");
            egui::ComboBox::from_id_salt("noise_wavelet_combo")
                .selected_text(draft.wavelet.label())
                .show_ui(ui, |ui| {
                    for w in [Wavelet::Haar, Wavelet::Db4, Wavelet::Sym4] {
                        ui.selectable_value(&mut draft.wavelet, w, w.label());
                    }
                });
            ui.end_row();
            ui.label("Levels:");
            ui.add(egui::DragValue::new(&mut draft.wavelet_levels).range(1..=10).speed(0.05))
                .on_hover_text("Level j covers roughly fs/2^(j+1) to fs/2^j. At 30 kHz, 4 levels reach down to ~940 Hz.");
            ui.end_row();
            ui.label("Threshold:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut draft.wavelet_k).range(0.0..=20.0).speed(0.05));
                ui.label("× σ");
            });
            ui.end_row();
            ui.label("Mode:");
            ui.horizontal(|ui| {
                ui.radio_value(&mut draft.wavelet_soft, true, "Soft");
                ui.radio_value(&mut draft.wavelet_soft, false, "Hard");
            });
            ui.end_row();
        });
    });

    ui.separator();
    ui.checkbox(&mut draft.isolated_enabled, "Attenuate isolated events").on_hover_text(
        "An event (zero crossing to zero crossing) whose peak reaches the event level is \
         kept only if it spans enough adjacent channels that each reach the given \
         fraction of its peak (same polarity, within the time tolerance). Otherwise \
         it is attenuated.",
    );
    ui.add_enabled_ui(draft.isolated_enabled, |ui| {
        egui::Grid::new("noise_isolated_grid").num_columns(2).show(ui, |ui| {
            ui.label("Event level:");
            uv(ui, &mut draft.event_level_uv);
            ui.end_row();
            ui.label("Min. adjacent channels:");
            ui.add(egui::DragValue::new(&mut draft.min_channels).range(1..=32).speed(0.1))
                .on_hover_text("Including the event's own channel. 1 keeps every event.");
            ui.end_row();
            ui.label("Neighbour amplitude:");
            pct(ui, &mut draft.neighbour_frac, "% of peak");
            ui.end_row();
            ui.label("Time tolerance:");
            ui.horizontal(|ui| {
                ui.label("±");
                ui.add(egui::DragValue::new(&mut draft.time_slack_ms).range(0.0..=5.0).speed(0.01))
                    .on_hover_text("How far a neighbour's peak may be from the event's peak.");
                ui.label("ms");
            });
            ui.end_row();
            ui.label("Attenuation:");
            pct(ui, &mut draft.attenuation, "%").on_hover_text("100 % removes the event.");
            ui.end_row();
        });
    });

    ui.separator();
    ui.checkbox(&mut draft.sigmoid_enabled, "Soft-knee noise gate").on_hover_text(
        "Scales every sample by a smooth S-shaped (logistic) function of its amplitude: \
         weak signals are pushed toward 0, strong ones keep (or are raised to) the max gain. \
         Also known as a downward expander; unlike a hard gate it has no jump at the threshold.",
    );
    ui.add_enabled_ui(draft.sigmoid_enabled, |ui| {
        egui::Grid::new("noise_sigmoid_grid").num_columns(2).show(ui, |ui| {
            ui.label("Centre:");
            uv(ui, &mut draft.sigmoid_center_uv)
                .on_hover_text("Amplitude at which the gain is half the max gain.");
            ui.end_row();
            ui.label("Width:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut draft.sigmoid_width_uv).range(0.1..=1000.0).speed(0.1))
                    .on_hover_text("Sharpness of the transition: the gain goes from 12 % to 88 % of the max gain within centre ± 2 widths.");
                ui.label("µV");
            });
            ui.end_row();
            ui.label("Max gain:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut draft.sigmoid_max_gain).range(0.1..=20.0).speed(0.01))
                    .on_hover_text("Gain for strong signals. Above 1 has the same effect as narrowing the colour range; in %ile colour mode the range comes from the unfiltered signal, so strong signals saturate.");
                ui.label("×");
            });
            ui.end_row();
        });
    });

    ui.separator();
    let mut action = Action::None;
    ui.horizontal(|ui| {
        let enter = crate::app::enter_applies(ui) && draft != applied;
        if ui.add_enabled(draft != applied, egui::Button::new("Apply")).clicked() || enter {
            action = Action::Apply;
        }
        if ui
            .add_enabled(draft.active() || applied.active(), egui::Button::new("Disable all"))
            .on_hover_text("Switches every filter off and redraws right away. The values are kept.")
            .clicked()
        {
            action = Action::DisableAll;
        }
        if ui.button("Reset to defaults").on_hover_text("Resets the values; the checkboxes stay.").clicked() {
            *draft = NoiseSuppression {
                wavelet_enabled: draft.wavelet_enabled,
                isolated_enabled: draft.isolated_enabled,
                sigmoid_enabled: draft.sigmoid_enabled,
                ..Default::default()
            };
        }
        if draft != applied {
            ui.label(egui::RichText::new("not applied yet").small().weak());
        }
    });
    action
}
