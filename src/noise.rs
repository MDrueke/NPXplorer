//! Noise suppression — a visual filter. It runs on the preprocessed samples of the
//! current view only (heatmap and waveform view) and never touches the worker
//! buffer, so firing rate, PSTH and spectrum keep the unfiltered signal.
//!
//! Steps, in this order (each optional):
//! 1. Wavelet shrinkage: per channel, the detail coefficients of an orthogonal wavelet
//!    transform are shrunk by k × their level's noise σ (median |d| / 0.6745).
//! 2. Isolated events: an event is a run of same-sign samples (zero crossing to zero
//!    crossing) whose peak reaches the event level. It counts as a real signal only if
//!    it sits inside `min_channels` physically adjacent channels that all reach
//!    `neighbour_frac` of its peak (same polarity, within ±`time_slack_ms`) — a
//!    morphological opening along depth. Events without that support are attenuated.
//! 3. Soft-knee noise gate: every sample is scaled by a smooth (logistic) function of
//!    its amplitude, ~0 well below the centre level, ~max gain well above it — a noise
//!    gate / downward expander with a soft knee.

use rayon::prelude::*;

use crate::data::DisplayRow;
use crate::preprocess::contiguous_depth_blocks;

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Wavelet {
    Haar,
    Db4,
    Sym4,
}

impl Wavelet {
    pub fn label(self) -> &'static str {
        match self {
            Wavelet::Haar => "Haar",
            Wavelet::Db4 => "Daubechies 4 (db4)",
            Wavelet::Sym4 => "Symlet 4 (sym4)",
        }
    }

    /// orthonormal lowpass decomposition filter (PyWavelets `dec_lo`)
    fn lowpass(self) -> &'static [f64] {
        match self {
            Wavelet::Haar => &[std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2],
            Wavelet::Db4 => &[
                -0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114,
                -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523,
            ],
            Wavelet::Sym4 => &[
                -0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161,
                0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427,
            ],
        }
    }
}

#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct NoiseSuppression {
    pub wavelet_enabled: bool,
    pub wavelet: Wavelet,
    /// decomposition levels (capped by the window length)
    pub wavelet_levels: usize,
    /// detail coefficients are shrunk by this many noise σ of their level
    pub wavelet_k: f32,
    /// soft (shrink toward 0) or hard (zero below the threshold) coefficient threshold
    pub wavelet_soft: bool,

    pub isolated_enabled: bool,
    /// an event's peak must reach ±this (µV) to be judged at all
    pub event_level_uv: f32,
    /// number of adjacent channels (including the event's own) the event must span
    pub min_channels: usize,
    /// those channels must reach this fraction of the event's peak amplitude
    pub neighbour_frac: f32,
    /// neighbours' peaks may be this far (ms) from the event's peak
    pub time_slack_ms: f32,
    /// fraction removed from unsupported events (0.8 = scaled to 20 %)
    pub attenuation: f32,

    /// soft-knee noise gate (named `sigmoid_*` in the saved preferences)
    pub sigmoid_enabled: bool,
    /// amplitude (µV) at which the gain is halfway between 0 and the max gain
    pub sigmoid_center_uv: f32,
    /// width (µV) of the transition; the gain goes from 12 % to 88 % within ±2 widths
    pub sigmoid_width_uv: f32,
    /// gain for amplitudes far above the centre
    pub sigmoid_max_gain: f32,
}

impl Default for NoiseSuppression {
    fn default() -> Self {
        NoiseSuppression {
            wavelet_enabled: false,
            wavelet: Wavelet::Sym4,
            wavelet_levels: 2,
            wavelet_k: 3.0,
            wavelet_soft: true,
            isolated_enabled: false,
            event_level_uv: 15.0,
            min_channels: 3,
            neighbour_frac: 0.9,
            time_slack_ms: 0.3,
            attenuation: 0.5,
            sigmoid_enabled: false,
            sigmoid_center_uv: 20.0,
            sigmoid_width_uv: 5.0,
            sigmoid_max_gain: 1.0,
        }
    }
}

impl NoiseSuppression {
    pub fn active(&self) -> bool {
        self.wavelet_enabled || self.isolated_enabled || self.sigmoid_enabled
    }
}

/// Samples of context filtered on each side of the view, so events and wavelet
/// support that straddle the view's edges are judged as in the middle of the data.
pub fn margin_samples(fs: f64) -> usize {
    (0.01 * fs).round() as usize
}

/// Filter `data` (`[n_data_rows][n_samp]`, rows in `display_rows`' data order) in
/// place. Each shank is handled on its own.
pub fn apply(data: &mut [f32], n_samp: usize, display_rows: &[DisplayRow], ns: &NoiseSuppression, fs: f64) {
    if n_samp == 0 || !ns.active() {
        return;
    }
    let data_rows: Vec<&DisplayRow> = display_rows.iter().filter(|r| matches!(r, DisplayRow::Data { .. })).collect();
    debug_assert_eq!(data.len(), data_rows.len() * n_samp);
    let shank = |r: &DisplayRow| if let DisplayRow::Data { shank, .. } = r { *shank } else { 0 };

    if ns.wavelet_enabled {
        let h = ns.wavelet.lowpass();
        data.par_chunks_mut(n_samp).for_each(|x| wavelet_denoise(x, h, ns.wavelet_levels, ns.wavelet_k, ns.wavelet_soft));
    }
    if ns.isolated_enabled {
        // runs of rows on the same shank, in data order; adjacency within a shank
        // follows physical depth and stops at gaps, like destripe
        let mut start = 0;
        while start < data_rows.len() {
            let s = shank(data_rows[start]);
            let end = (start..data_rows.len()).find(|&i| shank(data_rows[i]) != s).unwrap_or(data_rows.len());
            let shank_data = &mut data[start * n_samp..end * n_samp];
            for block in contiguous_depth_blocks(&data_rows[start..end]) {
                attenuate_isolated(shank_data, n_samp, &block, ns, fs);
            }
            start = end;
        }
    }
    if ns.sigmoid_enabled {
        let gain = sigmoid_gain(ns);
        data.par_iter_mut().for_each(|v| *v *= gain(v.abs()));
    }
}

/// Gain as a function of amplitude: a logistic curve around the centre, rescaled so
/// that it is exactly 0 at amplitude 0 and tends to the max gain.
fn sigmoid_gain(ns: &NoiseSuppression) -> impl Fn(f32) -> f32 + Sync {
    let (c, w) = (ns.sigmoid_center_uv.max(0.0), ns.sigmoid_width_uv.max(1e-3));
    let logistic = move |a: f32| 1.0 / (1.0 + (-(a - c) / w).exp());
    let g0 = logistic(0.0);
    let scale = ns.sigmoid_max_gain.max(0.0) / (1.0 - g0);
    move |a| (logistic(a) - g0) * scale
}

/// Calls `f(start, end, peak_t, peak, positive)` for every run of same-sign samples
/// `[start, end)` whose largest |x| (`peak`, at `peak_t`) reaches `level`.
fn for_each_event(x: &[f32], level: f32, mut f: impl FnMut(usize, usize, usize, f32, bool)) {
    let n = x.len();
    let mut t = 0;
    while t < n {
        if x[t] == 0.0 {
            t += 1;
            continue;
        }
        let pos = x[t] > 0.0;
        let start = t;
        let (mut peak_t, mut peak) = (t, 0.0f32);
        while t < n && x[t] != 0.0 && (x[t] > 0.0) == pos {
            let a = x[t].abs();
            if a > peak {
                peak = a;
                peak_t = t;
            }
            t += 1;
        }
        if peak >= level {
            f(start, t, peak_t, peak, pos);
        }
    }
}

/// `block`: row indices into `data`, ordered by depth, all physically adjacent.
fn attenuate_isolated(data: &mut [f32], n_samp: usize, block: &[usize], ns: &NoiseSuppression, fs: f64) {
    let n = block.len();
    if n == 0 {
        return;
    }
    let win = ns.min_channels.clamp(1, n);
    let slack = (ns.time_slack_ms as f64 * 1e-3 * fs).round().max(0.0) as usize;
    let frac = ns.neighbour_frac;

    // decide on the unmodified data first, so no row sees an already attenuated neighbour
    let src: &[f32] = data;
    let row = |i: usize| &src[block[i] * n_samp..(block[i] + 1) * n_samp];

    // largest same-polarity value of block row `j` within ±slack of `t`
    let env = |j: usize, t: usize, pos: bool| {
        let x = row(j);
        let lo = t.saturating_sub(slack);
        let hi = (t + slack + 1).min(n_samp);
        x[lo..hi].iter().fold(0.0f32, |m, &v| m.max(if pos { v } else { -v }))
    };

    let spans: Vec<Vec<(usize, usize)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            // windows of `win` channels that contain `i`: starts k in [k_lo, k_hi]
            let k_lo = (i + 1).saturating_sub(win);
            let k_hi = i.min(n - win);
            let mut envs = vec![0.0f32; k_hi + win - k_lo];
            let mut out = Vec::new();
            for_each_event(row(i), ns.event_level_uv.max(0.0), |start, end, peak_t, peak, pos| {
                for (e, j) in envs.iter_mut().zip(k_lo..) {
                    *e = if j == i { peak } else { env(j, peak_t, pos) };
                }
                let need = frac * peak;
                let supported = (0..=k_hi - k_lo).any(|k| envs[k..k + win].iter().all(|&e| e >= need));
                if !supported {
                    out.push((start, end));
                }
            });
            out
        })
        .collect();

    let gain = 1.0 - ns.attenuation.clamp(0.0, 1.0);
    let mut by_row: Vec<Option<&Vec<(usize, usize)>>> = vec![None; data.len() / n_samp];
    for (&r, s) in block.iter().zip(&spans) {
        if !s.is_empty() {
            by_row[r] = Some(s);
        }
    }
    data.par_chunks_mut(n_samp).zip(by_row.par_iter()).for_each(|(x, s)| {
        for &(a, b) in s.iter().copied().flatten() {
            x[a..b].iter_mut().for_each(|v| *v *= gain);
        }
    });
}

/// Wavelet shrinkage of one channel: periodic orthogonal DWT (the signal is first
/// extended by reflection to a multiple of 2^levels), detail coefficients thresholded
/// at k × their level's σ = median |d| / 0.6745, then the inverse transform.
fn wavelet_denoise(x: &mut [f32], h: &[f64], levels: usize, k: f32, soft: bool) {
    let n = x.len();
    let taps = h.len();
    // every level must keep at least `taps` coefficients
    let mut levels = levels;
    while levels > 0 && n >> levels < taps {
        levels -= 1;
    }
    if levels == 0 {
        return;
    }
    let m = 1usize << levels;
    let n_pad = n.div_ceil(m) * m;
    let mut c: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    for i in 0..n_pad - n {
        c.push(x[n - 1 - (i % n)] as f64);
    }
    // highpass g[j] = (-1)^j h[taps-1-j]
    let g: Vec<f64> = (0..taps).map(|j| if j % 2 == 0 { 1.0 } else { -1.0 } * h[taps - 1 - j]).collect();

    let mut details: Vec<Vec<f64>> = Vec::with_capacity(levels);
    for _ in 0..levels {
        let len = c.len();
        let half = len / 2;
        let (mut a, mut d) = (vec![0.0; half], vec![0.0; half]);
        for i in 0..half {
            let (mut sa, mut sd) = (0.0, 0.0);
            let s0 = 2 * i;
            if s0 + taps <= len {
                // no wrap-around: plain slice, no modulo in the hot loop
                for ((&hj, &gj), &v) in h.iter().zip(&g).zip(&c[s0..s0 + taps]) {
                    sa += hj * v;
                    sd += gj * v;
                }
            } else {
                for j in 0..taps {
                    let v = c[(s0 + j) % len];
                    sa += h[j] * v;
                    sd += g[j] * v;
                }
            }
            a[i] = sa;
            d[i] = sd;
        }
        details.push(d);
        c = a;
    }

    for d in &mut details {
        let mut mags: Vec<f64> = d.iter().map(|v| v.abs()).collect();
        let mid = mags.len() / 2;
        let (_, med, _) = mags.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let t = k as f64 * *med / 0.6745;
        for v in d.iter_mut() {
            *v = if soft {
                v.signum() * (v.abs() - t).max(0.0)
            } else if v.abs() < t {
                0.0
            } else {
                *v
            };
        }
    }

    for d in details.iter().rev() {
        let half = c.len();
        let len = half * 2;
        let mut up = vec![0.0; len];
        for i in 0..half {
            let (ci, di, s0) = (c[i], d[i], 2 * i);
            if s0 + taps <= len {
                for ((&hj, &gj), u) in h.iter().zip(&g).zip(&mut up[s0..s0 + taps]) {
                    *u += hj * ci + gj * di;
                }
            } else {
                for j in 0..taps {
                    up[(s0 + j) % len] += h[j] * ci + g[j] * di;
                }
            }
        }
        c = up;
    }
    for (o, v) in x.iter_mut().zip(&c) {
        *o = *v as f32;
    }
}

/// What the user asked for in the settings window this frame.
#[derive(PartialEq)]
pub enum Action {
    None,
    /// use the draft as it is
    Apply,
    /// switch every step off right away (draft and applied); values are kept
    DisableAll,
}

impl NoiseSuppression {
    pub fn disable_all(&mut self) {
        self.wavelet_enabled = false;
        self.isolated_enabled = false;
        self.sigmoid_enabled = false;
    }
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
        if ui.add_enabled(draft != applied, egui::Button::new("Apply")).clicked() {
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

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 30_000.0;
    const N: usize = 200;

    fn rows(n: usize) -> Vec<DisplayRow> {
        (0..n)
            .map(|i| DisplayRow::Data { data_idx: i, channels: vec![i], first_ch: i, x_um: 0.0, y_um: 20.0 * i as f32, shank: 0 })
            .collect()
    }

    /// a negative spike of `amp` µV centred at `t0` (5 samples wide) on row `r`
    fn spike(data: &mut [f32], r: usize, t0: usize, amp: f32) {
        for (k, w) in [0.3, 0.7, 1.0, 0.7, 0.3].iter().enumerate() {
            data[r * N + t0 - 2 + k] = -amp * w;
        }
    }

    fn run(data: &mut [f32], n_rows: usize, ns: &NoiseSuppression) {
        apply(data, N, &rows(n_rows), ns, FS);
    }

    fn iso() -> NoiseSuppression {
        NoiseSuppression {
            isolated_enabled: true,
            event_level_uv: 10.0,
            min_channels: 3,
            neighbour_frac: 0.8,
            attenuation: 0.8,
            ..Default::default()
        }
    }

    #[test]
    fn single_channel_event_is_attenuated() {
        let mut d = vec![0.0f32; 8 * N];
        spike(&mut d, 4, 100, 100.0);
        run(&mut d, 8, &iso());
        assert!((d[4 * N + 100] + 20.0).abs() < 1e-4, "peak {}", d[4 * N + 100]);
        assert!((d[4 * N + 98] + 6.0).abs() < 1e-4, "whole event is scaled");
    }

    #[test]
    fn event_spanning_three_channels_is_kept_despite_offset() {
        let mut d = vec![0.0f32; 8 * N];
        spike(&mut d, 3, 103, 85.0); // peaks a few samples apart (within 0.3 ms = 9 samples)
        spike(&mut d, 4, 100, 100.0);
        spike(&mut d, 5, 98, 90.0);
        run(&mut d, 8, &iso());
        assert_eq!(d[4 * N + 100], -100.0);
        assert_eq!(d[3 * N + 103], -85.0);
        assert_eq!(d[5 * N + 98], -90.0);
    }

    #[test]
    fn weak_or_opposite_neighbours_do_not_support() {
        let mut d = vec![0.0f32; 8 * N];
        spike(&mut d, 3, 100, 50.0); // only 50 % of the peak
        spike(&mut d, 4, 100, 100.0);
        spike(&mut d, 5, 100, -100.0); // positive: other polarity
        run(&mut d, 8, &iso());
        assert!((d[4 * N + 100] + 20.0).abs() < 1e-4);
    }

    #[test]
    fn shanks_are_independent() {
        // the same spike on rows 3..=5, but row 4 is on another shank
        let mut d = vec![0.0f32; 8 * N];
        for r in 3..=5 {
            spike(&mut d, r, 100, 100.0);
        }
        let mut rows = rows(8);
        for (i, r) in rows.iter_mut().enumerate() {
            if let DisplayRow::Data { shank, .. } = r {
                *shank = if i >= 4 { 1 } else { 0 };
            }
        }
        apply(&mut d, N, &rows, &iso(), FS);
        assert!((d[3 * N + 100] + 20.0).abs() < 1e-4, "row 3 has only one same-shank neighbour");
    }

    #[test]
    fn soft_knee_gate_shape() {
        let ns = NoiseSuppression { sigmoid_enabled: true, ..Default::default() };
        let g = sigmoid_gain(&ns);
        assert_eq!(g(0.0), 0.0);
        assert!(g(10.0) < 0.15 && g(10.0) > 0.0);
        assert!((g(20.0) - 0.5).abs() < 0.01);
        assert!(g(50.0) > 0.99 && g(50.0) <= 1.0);
        let mut d = vec![0.0f32; N];
        d[5] = -100.0;
        d[6] = 5.0;
        run(&mut d, 1, &NoiseSuppression { sigmoid_max_gain: 2.0, ..ns });
        assert!((d[5] + 200.0).abs() < 1e-2, "strong: {}", d[5]);
        assert!(d[6] > 0.0 && d[6] < 0.5, "weak: {}", d[6]); // gain ~0.06 at 5 µV
    }

    #[test]
    fn wavelet_reconstructs_exactly_without_threshold() {
        for w in [Wavelet::Haar, Wavelet::Db4, Wavelet::Sym4] {
            // odd length: exercises the reflection padding
            let x: Vec<f32> = (0..1001).map(|i| ((i as f32) * 0.37).sin() * 50.0 + (i % 7) as f32).collect();
            let mut y = x.clone();
            wavelet_denoise(&mut y, w.lowpass(), 5, 0.0, true);
            let err = x.iter().zip(&y).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(err < 1e-3, "{w:?}: max error {err}");
        }
    }

    #[test]
    fn wavelet_reduces_noise_and_keeps_a_large_spike() {
        let mut seed = 1u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32
        };
        let n = 4096;
        let mut x: Vec<f32> = (0..n)
            .map(|_| 7.0 * (-2.0 * rnd().max(1e-9).ln()).sqrt() * (std::f32::consts::TAU * rnd()).cos())
            .collect();
        for (k, w) in [0.3, 0.7, 1.0, 0.7, 0.3].iter().enumerate() {
            x[2000 + k] -= 150.0 * w;
        }
        let rms = |v: &[f32]| (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32).sqrt();
        let before = rms(&x[..1900]);
        wavelet_denoise(&mut x, Wavelet::Sym4.lowpass(), 4, 3.0, true);
        assert!(rms(&x[..1900]) < 0.5 * before, "noise rms {before} -> {}", rms(&x[..1900]));
        assert!(x[2002] < -60.0, "spike peak {}", x[2002]);
    }
}
