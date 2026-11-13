//! The central panel: the heatmap (or the single-channel waveform view) with its
//! overlays, legend and readouts, the caches it is drawn from, and the mouse and
//! keyboard input that moves the view.

use super::*;
use crate::render::HeatmapStyle;
use crate::rows::SampleWindow;

/// Noise-filtered copy of the view; `key` = (buffer data pointer, buffer first sample,
/// buffer length, view first sample, view length).
pub(super) struct NoiseView {
    /// (view window generation, view first sample, view length)
    pub(super) key: (u64, usize, usize),
    pub(super) settings: crate::noise::NoiseSuppression,
    pub(super) data: Arc<Vec<f32>>,
    pub(super) first: usize,
    pub(super) n: usize,
}

/// The view drawn this frame and the preprocessed data around it, taken from
/// `view_window` once per frame, so the drawing code can borrow `self` freely.
pub(super) struct Frame {
    pub(super) view_first: usize,
    pub(super) view_n: usize,
    /// the view window; its data may be for older preprocessing settings
    pub(super) buf: Option<ViewWindow>,
    /// the window's data is for the current preprocessing settings
    pub(super) matches_cfg: bool,
    /// what the heatmap and waveform view draw: the window's data, or with noise
    /// suppression on a filtered copy of the view; `[row][src_n]` from `src_first`
    pub(super) src_data: Option<Arc<Vec<f32>>>,
    pub(super) src_first: usize,
    pub(super) src_n: usize,
}

/// Everything the heatmap texture is drawn from. The texture is redrawn exactly when
/// this changes, so no setting has to remember to invalidate it.
#[derive(Clone, PartialEq)]
pub(super) struct HeatmapKey {
    view: (usize, usize),
    size: [usize; 2],
    /// generation of the view window drawn
    window: u64,
    rows: (usize, usize),
    cfg: PreprocConfig,
    noise: crate::noise::NoiseSuppression,
    scale: crate::render::ColorScale,
    peak_pooling: bool,
    colormap: ColorMapChoice,
}

#[derive(Default)]
pub(super) struct HeatmapCache {
    pub(super) texture: Option<TextureHandle>,
    pub(super) pixel_buf: Vec<u8>,
    pub(super) key: Option<HeatmapKey>,
}

/// Everything the firing-rate projection is computed from.
#[derive(Clone, PartialEq)]
pub(super) struct ProjectionKey {
    view: (usize, usize),
    window: u64,
    rows: (usize, usize),
    threshold: f32,
    sigma: f32,
    cfg: PreprocConfig,
}

/// Firing-rate overlay: smoothed threshold-crossing count per visible row.
#[derive(Default)]
pub(super) struct ProjectionCache {
    pub(super) sums: Vec<f32>,
    pub(super) key: Option<ProjectionKey>,
}

/// Threshold crossings per row of `rows`, in samples `[offset, offset + n)` of each
/// row of `data` (`[row][stride]`, rows by `data_idx`): samples below `threshold`
/// (µV), at most one per `refractory` samples. Then smoothed across rows with a 7-tap
/// Gaussian of σ = `sigma` rows (normalised at the ends).
pub(super) fn spike_projection(
    data: &[f32],
    stride: usize,
    (offset, n): (usize, usize),
    rows: &[DisplayRow],
    threshold: f32,
    refractory: usize,
    sigma: f32,
) -> Vec<f32> {
    use rayon::prelude::*;
    let mut sums = vec![0.0f32; rows.len()];
    sums.par_iter_mut().zip(rows).for_each(|(count, row)| {
        let DisplayRow::Data { data_idx, .. } = row else { return };
        let base = data_idx * stride + offset;
        if n == 0 || base + n > data.len() {
            return;
        }
        let mut spikes = 0.0f32;
        let mut last_spike: Option<usize> = None;
        for (t, &v) in data[base..base + n].iter().enumerate() {
            if v < threshold && last_spike.is_none_or(|last| t - last > refractory) {
                spikes += 1.0;
                last_spike = Some(t);
            }
        }
        *count = spikes;
    });

    const RADIUS: usize = 3;
    let sigma = sigma.max(0.01);
    let kernel: [f32; 2 * RADIUS + 1] = std::array::from_fn(|j| {
        let x = j as f32 - RADIUS as f32;
        (-x * x / (2.0 * sigma * sigma)).exp()
    });
    (0..sums.len())
        .map(|i| {
            let (mut v, mut weight_sum) = (0.0, 0.0);
            for (j, &k) in kernel.iter().enumerate() {
                if let Some(&s) = (i + j).checked_sub(RADIUS).and_then(|idx| sums.get(idx)) {
                    v += s * k;
                    weight_sum += k;
                }
            }
            if weight_sum > 0.0 { v / weight_sum } else { 0.0 }
        })
        .collect()
}

/// Channels of the bottom and top data rows inside the screen-space selection `sel` on
/// a heatmap drawn in `rect` with rows `first_row..=last_row`; at least two rows when
/// the heatmap has them. None if the selection holds no data row.
pub(super) fn zoom_selection(
    display_rows: &[DisplayRow],
    first_row: usize,
    last_row: usize,
    rect: egui::Rect,
    sel: egui::Rect,
) -> Option<(usize, usize)> {
    let n_rows = last_row - first_row + 1;
    let row_at = |y: f32| {
        let frac = ((y - rect.top()) / rect.height()).clamp(0.0, 1.0);
        last_row
            .saturating_sub((frac as f64 * n_rows as f64) as usize)
            .clamp(first_row, last_row)
    };
    let data_rows: Vec<(usize, usize)> = (first_row..=last_row)
        .filter_map(|r| match &display_rows[r] {
            DisplayRow::Data { first_ch, .. } => Some((r, *first_ch)),
            _ => None,
        })
        .collect();
    let (lo, hi) = (row_at(sel.bottom()), row_at(sel.top()));
    let inside: Vec<usize> = (0..data_rows.len())
        .filter(|&i| (lo..=hi).contains(&data_rows[i].0))
        .collect();
    let (mut b, mut t) = (*inside.first()?, *inside.last()?);
    if b == t {
        if t + 1 < data_rows.len() {
            t += 1;
        } else {
            b = b.saturating_sub(1);
        }
    }
    Some((data_rows[b].1, data_rows[t].1))
}

/// "Esc to close zoom" at `pos`: solid text on a semi-transparent box, both switching
/// with the colormap (dark box / white text, or the reverse on Cool-Warm's light
/// background).
pub(super) fn draw_zoom_notice(painter: &egui::Painter, pos: egui::Pos2, cmap: &ColorMapChoice) {
    let [fr, fg, fb] = cmap.spec().heatmap_fg;
    let fg_color = egui::Color32::from_rgb(fr, fg, fb);
    let [br, bg, bb] = cmap.spec().label_bg;
    let galley = painter.layout_no_wrap("Esc to close zoom".to_string(), egui::FontId::proportional(13.0), fg_color);
    painter.rect_filled(
        galley.rect.translate(pos.to_vec2()).expand2(egui::vec2(6.0, 4.0)),
        4.0,
        egui::Color32::from_rgba_unmultiplied(br, bg, bb, 160),
    );
    painter.galley(pos, galley, fg_color);
}

/// The hover readout (channel, time, voltage) at the bottom left of `rect`.
pub(super) fn draw_readout(painter: &egui::Painter, rect: egui::Rect, label: String) {
    let color = egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200);
    let galley = painter.layout_no_wrap(label, egui::FontId::proportional(12.0), color);
    let text_pos = rect.left_bottom() + Vec2::new(6.0, -6.0 - galley.rect.height());
    let bg_rect = galley.rect.translate(text_pos.to_vec2()).expand(4.0);
    let [r, g, b] = crate::render::C_ZERO;
    painter.rect_filled(bg_rect, 2.0, egui::Color32::from_rgba_unmultiplied(r, g, b, 200));
    painter.galley(text_pos, galley, color);
}

impl NPXplorerApp {
    /// The samples to draw as `(data, first_sample, n_samp)`: the worker buffer, or
    /// with noise suppression on, a filtered copy of the view plus a margin. The copy
    /// is cached until the view, the buffer or the settings change.
    pub(super) fn display_source(&mut self, w: &ViewWindow, view_first: usize, view_n: usize) -> (Option<Arc<Vec<f32>>>, usize, usize) {
        let (data, rows, buf_first, buf_n_samp) = (&w.data, &w.display_rows, w.first_sample, w.n_samp);
        let unfiltered = (Some(Arc::clone(data)), buf_first, buf_n_samp);
        if !self.noise.active() || buf_n_samp == 0 {
            self.noise_view = None;
            return unfiltered;
        }
        let key = (w.generation, view_first, view_n);
        if let Some(v) = &self.noise_view {
            if v.key == key && v.settings == self.noise {
                return (Some(Arc::clone(&v.data)), v.first, v.n);
            }
        }
        let fs = self.meta.sample_rate;
        let margin = crate::noise::margin_samples(fs);
        let lo = view_first.saturating_sub(margin).max(buf_first);
        let hi = (view_first + view_n + margin).min(buf_first + buf_n_samp);
        if lo >= hi {
            return unfiltered;
        }
        let n = hi - lo;
        let off = lo - buf_first;
        let n_rows = data.len() / buf_n_samp;
        let mut out = vec![0.0f32; n_rows * n];
        {
            use rayon::prelude::*;
            out.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
                row.copy_from_slice(&data[r * buf_n_samp + off..r * buf_n_samp + off + n]);
            });
        }
        crate::noise::apply(&mut out, n, rows, &self.noise, fs);
        let out = Arc::new(out);
        self.noise_view = Some(NoiseView { key, settings: self.noise.clone(), data: Arc::clone(&out), first: lo, n });
        (Some(out), lo, n)
    }

    /// Replaces the heatmap with a line plot of one channel's voltage over the
    /// current time window. Y-axis range is ±`waveform_y_range_uv` (Alt+scroll to
    /// adjust); line color follows the active colormap's accent color.
    pub(super) fn draw_waveform_view(&mut self, ui: &mut Ui, ch: usize, frame: &Frame) {
        let (matches_cfg, view_first, view_n) = (frame.matches_cfg, frame.view_first, frame.view_n);
        // the samples drawn (noise-filtered when that is on) and the rows they belong to
        let (buf_first, buf_n_samp, buf_data) = (frame.src_first, frame.src_n, &frame.src_data);
        let buf_display_rows = &frame.buf.as_ref().map(|w| Arc::clone(&w.display_rows));
        let rect = ui.available_rect_before_wrap();
        let resp = ui.interact(rect, ui.id().with("waveform_plot"), egui::Sense::click_and_drag());
        let hover = resp
            .hover_pos()
            .or_else(|| self.zoom_drag_start.and_then(|_| ui.input(|i| i.pointer.interact_pos())))
            .filter(|_| self.capture.is_none());
        let painter = ui.painter_at(rect);

        // vertical axis: ±half_range around the centre voltage
        let half_range = self.waveform_y_range_uv.max(self.waveform_min_range());
        let center_uv = self.waveform_y_center_uv;
        let mid_y = rect.center().y;
        let half_h = rect.height() * 0.5 - 4.0;
        let y_of = |v: f32| mid_y - ((v - center_uv) / half_range).clamp(-1.0, 1.0) * half_h;
        let uv_at = |y: f32| center_uv + (mid_y - y) / half_h * half_range;
        let (view_start_s, view_dur_s) = (self.view_start_s, self.view_dur_s);
        let t_at = |x: f32| view_start_s + ((x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64 * view_dur_s;

        // rectangle zoom: left-drag selects a time and voltage range
        if resp.drag_started_by(egui::PointerButton::Primary) {
            self.zoom_drag_start = ui.input(|i| i.pointer.press_origin());
        }
        let mut drag_sel = None;
        if let Some(start) = self.zoom_drag_start {
            let end = ui.input(|i| i.pointer.interact_pos()).unwrap_or(start);
            let sel = egui::Rect::from_two_pos(start, end).intersect(rect);
            if resp.drag_stopped() || !ui.input(|i| i.pointer.primary_down()) {
                self.zoom_drag_start = None;
                if sel.width() >= ZOOM_MIN_DRAG_PX && sel.height() >= ZOOM_MIN_DRAG_PX {
                    let (t0, t1) = (t_at(sel.left()), t_at(sel.right()));
                    let (v_lo, v_hi) = (uv_at(sel.bottom()), uv_at(sel.top()));
                    // nested zooms keep the view from before the first one
                    if self.waveform_zoom.is_none() {
                        self.waveform_zoom = Some(WaveformZoom {
                            prev_start_s: self.view_start_s,
                            prev_dur_s: self.view_dur_s,
                            prev_y_range_uv: self.waveform_y_range_uv,
                        });
                    }
                    let total_s = self.meta.n_samples as f64 / self.meta.sample_rate;
                    self.view_dur_s = (t1 - t0).max(0.01);
                    self.view_start_s = t0.clamp(0.0, (total_s - self.view_dur_s).max(0.0));
                    self.window_dur_str = format!("{:.3}", self.view_dur_s);
                    self.waveform_y_center_uv = (v_lo + v_hi) / 2.0;
                    self.waveform_y_range_uv = ((v_hi - v_lo) / 2.0).max(self.waveform_min_range());
                    ui.ctx().request_repaint();
                }
            } else {
                drag_sel = Some(sel);
            }
        }
        painter.rect_filled(
            rect,
            0.0,
            egui::Color32::from_rgb(
                crate::render::C_ZERO[0],
                crate::render::C_ZERO[1],
                crate::render::C_ZERO[2],
            ),
        );
        self.events.draw_overlay(
            &painter,
            rect,
            self.view_start_s,
            self.view_dur_s,
            &self.colormap_choice,
        );

        let row_data: Option<(Arc<Vec<f32>>, usize)> = if matches_cfg {
            buf_display_rows
                .as_ref()
                .zip(buf_data.as_ref())
                .and_then(|(rows, data)| {
                    rows.iter()
                        .find_map(|r| match r {
                            DisplayRow::Data {
                                data_idx, first_ch, ..
                            } if *first_ch + 1 == ch => Some(*data_idx),
                            _ => None,
                        })
                        .map(|data_idx| (Arc::clone(data), data_idx))
                })
        } else {
            None
        };

        match row_data {
            Some((data, data_idx)) => {
                let row_base = data_idx * buf_n_samp;
                let ov_start = view_first.max(buf_first);
                let ov_end = (view_first + view_n).min(buf_first + buf_n_samp);

                if ov_start < ov_end && row_base + buf_n_samp <= data.len() {
                    let lo = ov_start - buf_first;
                    let hi = ov_end - buf_first;
                    let samples = &data[row_base + lo..row_base + hi];
                    let n = samples.len();

                    if n >= 2 {
                        let [ar, ag, ab] = self.colormap_choice.spec().accent;
                        let line_color = egui::Color32::from_rgb(ar, ag, ab);

                        // base line width — tune WAVEFORM_LINE_BASE_WIDTH below; it
                        // scales gently with the panel height so it isn't too thin on
                        // a large window or too thick on a small one
                        const WAVEFORM_LINE_BASE_WIDTH: f32 = 1.0;
                        let line_width =
                            WAVEFORM_LINE_BASE_WIDTH * (rect.height() / 400.0).clamp(0.5, 2.5);

                        let pts: Vec<egui::Pos2> = samples
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| {
                                let x = rect.left() + (i as f32 / (n - 1) as f32) * rect.width();
                                egui::pos2(x, y_of(v))
                            })
                            .collect();

                        // 0 µV line, while it is in the shown range
                        if (center_uv - 0.0).abs() <= half_range {
                            let y0 = y_of(0.0);
                            painter.line_segment(
                                [egui::pos2(rect.left(), y0), egui::pos2(rect.right(), y0)],
                                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(80)),
                            );
                        }
                        // hovered sample: marked on the trace, values in the readout
                        let hovered = hover.map(|p| {
                            let frac = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                            let i = (frac * (n - 1) as f32).round() as usize;
                            (i, pts[i])
                        });
                        painter.add(egui::Shape::line(
                            pts,
                            egui::Stroke::new(line_width, line_color),
                        ));

                        let hint = if self.capture.is_some() { "" } else { " (Alt+scroll to rescale, drag to zoom)" };
                        let (unit, p) = (self.unit(), if self.display_is_csd { 2 } else { 0 });
                        let range = if center_uv == 0.0 {
                            format!("±{half_range:.p$} {unit}")
                        } else {
                            format!("{:.p1$} to {:.p1$} {unit}", center_uv - half_range, center_uv + half_range, p1 = p + 1)
                        };
                        painter.text(
                            egui::pos2(rect.left() + 6.0, rect.top() + 4.0),
                            egui::Align2::LEFT_TOP,
                            format!("{}  ·  {range}{hint}", self.meta.channel_id(ch - 1)),
                            egui::FontId::proportional(13.0),
                            egui::Color32::from_gray(200),
                        );

                        let id = self.meta.channel_id(ch - 1);
                        match (drag_sel, hovered) {
                            // while dragging a zoom rectangle: the ranges it covers
                            (Some(sel), _) => draw_readout(
                                &painter,
                                rect,
                                format!(
                                    "{id}  t = {:.4}–{:.4} s  {:.p$} to {:.p$} {unit}",
                                    t_at(sel.left()),
                                    t_at(sel.right()),
                                    uv_at(sel.bottom()),
                                    uv_at(sel.top()),
                                    p = p + 1
                                ),
                            ),
                            (None, Some((i, pt))) => {
                                painter.circle_filled(pt, line_width + 2.0, line_color);
                                let t = (ov_start + i) as f64 / self.meta.sample_rate;
                                draw_readout(&painter, rect, format!("{id}  t = {t:.4} s  {:.p$} {unit}", samples[i], p = p + 1));
                            }
                            (None, None) => {}
                        }
                    }
                } else {
                    Self::draw_centered_message(&painter, rect, "⏳ Loading…");
                }
            }
            None => {
                let msg = if matches_cfg {
                    format!("Channel {} isn't currently displayed (removed, or outside the loaded buffer).", self.meta.channel_id(ch - 1))
                } else {
                    "⏳ Loading…".to_string()
                };
                Self::draw_centered_message(&painter, rect, &msg);
            }
        }

        if self.capture.is_some() {
            return;
        }
        if let Some(sel) = drag_sel {
            painter.rect_filled(sel, 0.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 3));
            painter.rect_stroke(sel, 0.0, egui::Stroke::new(1.0_f32, egui::Color32::WHITE), egui::StrokeKind::Inside);
        }
        if self.waveform_zoom.is_some() {
            draw_zoom_notice(&painter, rect.left_top() + egui::vec2(10.0, 28.0), &self.colormap_choice);
        }
        let close_rect = egui::Rect::from_min_size(
            rect.right_top() + egui::vec2(-34.0, 6.0),
            egui::vec2(28.0, 24.0),
        );
        if ui
            .put(close_rect, egui::Button::new("✖"))
            .on_hover_text("Back to heatmap (Esc)")
            .clicked()
        {
            self.leave_waveform_zoom();
            self.waveform_channel = None;
        }
    }

    pub(super) fn draw_central_panel(&mut self, ctx: &egui::Context, ui: &mut Ui, alt_held: bool) {
        self.plot_rect = Some(ui.available_rect_before_wrap());
        let avail = ui.available_size();
        // power spectrum panel: a fixed 20% strip to the right of the main
        // heatmap, once a result exists and the user hasn't hidden it
        let spec_visible = self.spectrum.show_overlay
            && self.waveform_channel.is_none()
            && self.spectrum.result.as_ref().is_some_and(|c| c.matches(&self.preproc_cfg));
        let heat_w = if spec_visible { (avail.x * 0.8).max(1.0) } else { avail.x };
        let (pw, ph) = (heat_w as usize, avail.y as usize);
        if pw < 2 || ph < 2 {
            return;
        }

        let frame = self.prepare_frame(ctx);
        self.update_heatmap(ctx, &frame, [pw, ph]);
        self.update_spike_projection(&frame);
        self.post_plan(&frame);
        self.update_live_spectrum(ctx, &frame);

        if let Some(ch) = self.waveform_channel {
            self.draw_waveform_view(ui, ch, &frame);
        } else {
            self.draw_heatmap_view(ctx, ui, &frame, egui::vec2(heat_w, avail.y), avail.x, alt_held);
        }
    }

    /// The view in samples, the view window kept around it, and what is drawn from it.
    pub(super) fn prepare_frame(&mut self, ctx: &egui::Context) -> Frame {
        let fs = self.meta.sample_rate;
        let view_first = (self.view_start_s * fs) as usize;
        let view_n = (self.view_dur_s * fs) as usize;
        if view_first != self.last_view_first {
            self.scroll_dir = if view_first > self.last_view_first { 1 } else { -1 };
            self.last_view_first = view_first;
        }

        let pad = (fs as usize).max(view_n);
        crate::worker::update_view_window(&self.worker_state, &mut self.view_window, view_first, view_n, pad, &self.preproc_cfg);
        let buf = self.view_window.clone();
        let expected_end = view_first + view_n.min(self.meta.n_samples.saturating_sub(view_first));
        let matches_view = buf.as_ref().is_some_and(|w| w.first_sample <= view_first && expected_end <= w.first_sample + w.n_samp);
        let matches_cfg = buf.as_ref().is_some_and(|w| w.cfg == self.preproc_cfg);

        // what the heatmap and waveform view draw (noise-filtered or not)
        let (src_data, src_first, src_n) = match &buf {
            Some(w) if matches_cfg => self.display_source(w, view_first, view_n),
            Some(w) => (Some(Arc::clone(&w.data)), w.first_sample, w.n_samp),
            None => (None, 0, 0),
        };

        // the worker repaints when a block arrives; this only catches up after
        // a missed one while the view isn't covered yet
        if !matches_view || !matches_cfg {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        Frame { view_first, view_n, buf, matches_cfg, src_data, src_first, src_n }
    }

    /// The heatmap's colour scale. With peak pooling the percentile is taken from the
    /// pooled values on screen: pooled extremes sit well above the raw samples'
    /// percentiles, so a raw-sample range would saturate the map.
    pub(super) fn color_scale(&mut self, w: &ViewWindow, frame: &Frame) -> crate::render::ColorScale {
        use crate::render::ColorScale;
        let floor = if self.display_is_csd { 1e-4 } else { 1.0 };
        match self.color_mode {
            ColorMode::Percentile if self.peak_pooling => ColorScale::ViewPercentile(self.color_pct),
            ColorMode::Percentile => {
                // over ±1 s around the view, or the view if it is longer
                let center = frame.view_first + frame.view_n / 2;
                let half = (frame.view_n / 2).max(self.meta.sample_rate as usize);
                let key = (w.generation, center.saturating_sub(half), center + half, self.color_pct.to_bits());
                let v = match self.pct_cache {
                    Some((k, v)) if k == key => v,
                    _ => {
                        let v = crate::worker::window_percentile(w, key.1, key.2, self.color_pct).unwrap_or(250.0);
                        self.pct_cache = Some((key, v));
                        v
                    }
                };
                ColorScale::Fixed(v.max(floor))
            }
            ColorMode::Voltage => ColorScale::Fixed(self.color_uv.max(floor)),
        }
    }

    /// Redraw the heatmap texture when what it shows has changed. Whatever time overlap
    /// the window has with the view is drawn (the rest background-filled), so the view
    /// stays live while the worker catches up.
    pub(super) fn update_heatmap(&mut self, ctx: &egui::Context, frame: &Frame, size: [usize; 2]) {
        let Some(w) = frame.buf.as_ref().filter(|_| frame.matches_cfg && frame.view_n > 0) else { return };
        let scale = self.color_scale(w, frame);
        let rows = view_rows(&w.display_rows, self.zoom.as_ref());
        let key = HeatmapKey {
            view: (frame.view_first, frame.view_n),
            size,
            window: w.generation,
            rows,
            cfg: self.preproc_cfg.clone(),
            noise: self.noise.clone(),
            scale,
            peak_pooling: self.peak_pooling,
            colormap: self.colormap_choice.clone(),
        };
        let cache = &mut self.heatmap;
        if cache.texture.is_some() && cache.key.as_ref() == Some(&key) {
            return;
        }
        let shown_rows = if w.display_rows.is_empty() { &w.display_rows[..] } else { &w.display_rows[rows.0..=rows.1] };
        let src = SampleWindow {
            data: frame.src_data.as_deref().map_or(&w.data[..], |d| &d[..]),
            first: frame.src_first,
            n: frame.src_n,
        };
        let style = HeatmapStyle { scale, peak_pooling: self.peak_pooling, colormap: &self.colormap_choice };
        build_heatmap_into(&mut cache.pixel_buf, &src, shown_rows, (frame.view_first, frame.view_n), size, &style);
        let img = egui::ColorImage::from_rgba_unmultiplied(size, &cache.pixel_buf);
        // update the existing GPU texture in place while the size is unchanged (every
        // scroll frame) instead of allocating a new one
        let same_size = cache.key.as_ref().is_some_and(|k| k.size == size);
        match &mut cache.texture {
            Some(tex) if same_size => tex.set(img, TextureOptions::NEAREST),
            _ => cache.texture = Some(ctx.load_texture("heatmap", img, TextureOptions::NEAREST)),
        }
        cache.key = Some(key);
    }

    /// The heatmap texture, if it shows data for the current preprocessing settings
    /// (until then the view shows "Loading…").
    pub(super) fn current_heatmap(&self) -> Option<TextureHandle> {
        let key = self.heatmap.key.as_ref()?;
        (key.cfg == self.preproc_cfg).then(|| self.heatmap.texture.clone()).flatten()
    }

    /// Recompute the firing-rate projection when what it is computed from has changed.
    /// It counts in the preprocessed window data (not the noise-filtered copy), over
    /// whatever overlap the window has with the view.
    pub(super) fn update_spike_projection(&mut self, frame: &Frame) {
        let Some(w) = frame.buf.as_ref().filter(|_| frame.matches_cfg && frame.view_n > 0) else { return };
        // a µV spike threshold means nothing on CSD values
        if self.preproc_cfg.spatial_filter == SpatialFilter::Csd {
            self.projection = ProjectionCache::default();
            return;
        }
        if !self.show_firing_rate_overlay {
            return;
        }
        let rows = view_rows(&w.display_rows, self.zoom.as_ref());
        let key = ProjectionKey {
            view: (frame.view_first, frame.view_n),
            window: w.generation,
            rows,
            threshold: self.spike_threshold,
            sigma: self.spike_smoothing_sigma,
            cfg: self.preproc_cfg.clone(),
        };
        if self.projection.key.as_ref() == Some(&key) {
            return;
        }
        let ov_start = frame.view_first.max(w.first_sample);
        let ov_end = (frame.view_first + frame.view_n).min(w.first_sample + w.n_samp);
        let overlap = if ov_start < ov_end { (ov_start - w.first_sample, ov_end - ov_start) } else { (0, 0) };
        let refractory = (1.5 * self.meta.sample_rate as f32 / 1000.0) as usize;
        self.projection.sums = spike_projection(
            &w.data,
            w.n_samp,
            overlap,
            &w.display_rows[rows.0..=rows.1],
            self.spike_threshold,
            refractory,
            self.spike_smoothing_sigma,
        );
        self.projection.key = Some(key);
    }

    /// Tell the worker what to preprocess next.
    pub(super) fn post_plan(&mut self, frame: &Frame) {
        // heavy jobs get the CPU to themselves; background fill waits for them
        let busy = self.psth.job.is_some()
            || self.classify.is_running()
            || self.spectrum.is_computing()
            || self.export_ui.is_running();
        crate::worker::post_plan(&self.worker_state, &self.worker_cancel, Plan {
            cfg: self.preproc_cfg.clone(),
            view_first: frame.view_first,
            view_n: frame.view_n,
            direction: self.scroll_dir,
            around_samp: (self.around_s * self.meta.sample_rate) as usize,
            background: self.background_fill && !busy,
            budget_pct: self.mem_budget_pct,
            reserve_bytes: (self.mem_reserve_mb * 1e6) as u64,
        });
    }

    /// Power spectrum, current-view scope.
    pub(super) fn update_live_spectrum(&mut self, ctx: &egui::Context, frame: &Frame) {
        let (view_first, view_n, fs) = (frame.view_first, frame.view_n, self.meta.sample_rate);
        // power spectrum, current-view scope: recomputed once "Calculate" has been
        // pressed, whenever the result no longer matches the view/settings — but
        // only after the view has stopped moving for crate::spectrum_ui::SETTLE, so scrolling
        // isn't slowed down by an FFT of every channel on each frame. Skipped while
        // the panel is hidden or the waveform view replaces the heatmap.
        if (view_first, view_n) != self.spectrum.view_seen {
            self.spectrum.view_seen = (view_first, view_n);
            self.spectrum.view_changed_at = std::time::Instant::now();
        }
        if self.spectrum.want_live
            && self.spectrum.show_overlay
            && self.waveform_channel.is_none()
            && self.spectrum.time_scope == crate::spectrum::SpectrumTimeScope::CurrentView
        {
            let stale = self.spectrum.result.as_ref().is_none_or(|c| {
                c.view != Some((view_first, view_n))
                    || c.source != self.spectrum.source
                    || !c.matches(&self.preproc_cfg)
            });
            let still_for = self.spectrum.view_changed_at.elapsed();
            if stale && still_for < crate::spectrum_ui::SETTLE {
                ctx.request_repaint_after(crate::spectrum_ui::SETTLE - still_for);
            } else if stale {
                let new_result = match self.spectrum.source {
                    crate::spectrum::SpectrumSource::Raw => {
                        let full_rows = self.meta.build_display_rows(
                            self.preproc_cfg.avg_depths,
                            &self.preproc_cfg.removed_channels,
                            self.preproc_cfg.channel_order,
                            self.preproc_cfg.shank_order,
                        );
                        crate::spectrum::compute_psd_raw_current_view(
                            &self.raw, &self.meta, &full_rows, view_first, view_n,
                        )
                    }
                    // the buffer's rows are indexed by this same config's layout
                    crate::spectrum::SpectrumSource::Preprocessed => {
                        match &frame.buf {
                            Some(w) if frame.matches_cfg => crate::spectrum::compute_psd_preprocessed_current_view(
                                &SampleWindow { data: &w.data, first: w.first_sample, n: w.n_samp },
                                &w.display_rows,
                                (view_first, view_n),
                                fs,
                            ),
                            _ => None,
                        }
                    }
                };
                if let Some(result) = new_result {
                    self.spectrum.result = Some(Arc::new(crate::spectrum::ComputedSpectrum {
                        result,
                        cfg: self.preproc_cfg.clone(),
                        source: self.spectrum.source,
                        view: Some((view_first, view_n)),
                    }));
                }
            }
        }
    }

    /// The heatmap and everything on it: overlays, legend, hover readout, scale bar.
    pub(super) fn draw_heatmap_view(&mut self, ctx: &egui::Context, ui: &mut Ui, frame: &Frame, heat_size: Vec2, full_width: f32, alt_held: bool) {
        let shot = self.shot_includes();
        let Some(tex) = self.current_heatmap() else {
            ui.painter().text(
                ui.clip_rect().center(),
                egui::Align2::CENTER_CENTER,
                "⏳ Loading…",
                egui::FontId::proportional(18.0),
                egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200),
            );
            return;
        };
        let resp = ui.add(egui::Image::new(&tex).fit_to_exact_size(heat_size).sense(egui::Sense::click_and_drag()));
        self.events.draw_overlay(&ui.painter_at(resp.rect), resp.rect, self.view_start_s, self.view_dur_s, &self.colormap_choice);

        if heat_size.x < full_width {
            let strip = [(full_width - resp.rect.width()).max(1.0) as usize, heat_size.y as usize];
            self.draw_spectrum_strip(ctx, ui, frame, resp.rect, strip);
        }

        if let Some(display_rows) = frame.buf.as_ref().map(|w| Arc::clone(&w.display_rows)).filter(|r| !r.is_empty()) {
            let rows = view_rows(&display_rows, self.zoom.as_ref());
            // channel-selection lines and atlas borders reach across the spectrum strip
            let full_right = resp.rect.left() + full_width;
            self.handle_heatmap_pointer(ctx, ui, &resp, &display_rows, rows, alt_held);
            self.draw_heatmap_overlays(ui, &resp, full_right, &display_rows, rows, shot);
            self.draw_legend(ctx, ui, &resp);
        }
        self.draw_hover_readout(ctx, ui, &resp, frame);
        self.draw_plot_decorations(ui, &resp, shot);
    }

    /// The spectrum strip right of the heatmap, its rows aligned with the heatmap's
    /// (including the zoom).
    pub(super) fn draw_spectrum_strip(&mut self, ctx: &egui::Context, ui: &Ui, frame: &Frame, heat: egui::Rect, size: [usize; 2]) {
        let full_rows: Arc<Vec<DisplayRow>> = match &frame.buf {
            Some(w) if frame.matches_cfg => Arc::clone(&w.display_rows),
            _ => Arc::new(self.meta.build_display_rows(
                self.preproc_cfg.avg_depths,
                &self.preproc_cfg.removed_channels,
                self.preproc_cfg.channel_order,
                self.preproc_cfg.shank_order,
            )),
        };
        let rows = view_rows(&full_rows, self.zoom.as_ref());
        self.spectrum.draw_strip(ctx, ui, heat, size, &full_rows, rows);
    }

    /// Clicks (channel selection with Alt, context menu) and the rectangle zoom.
    pub(super) fn handle_heatmap_pointer(
        &mut self,
        ctx: &egui::Context,
        ui: &Ui,
        resp: &egui::Response,
        display_rows: &[DisplayRow],
        (first_row, last_row): (usize, usize),
        alt_held: bool,
    ) {
        let n_rows = last_row.saturating_sub(first_row) + 1;
        // click detection — channel selection now requires Alt (plain
        // left-click is a no-op; plain right-click opens the context menu)
        let mut click_pos = None;
        let mut is_left_click = false;
        let mut is_right_click = false;
        let mut is_context_click = false;

        if resp.clicked() {
            click_pos = resp.interact_pointer_pos().or_else(|| ctx.input(|i| i.pointer.interact_pos()));
            if alt_held {
                is_left_click = true;
            }
        }
        if resp.secondary_clicked() {
            click_pos = resp.interact_pointer_pos().or_else(|| ctx.input(|i| i.pointer.interact_pos()));
            if alt_held {
                is_right_click = true;
            } else {
                is_context_click = true;
            }
        }

        // rectangle zoom: plain left-drag (Alt+drag is left to the atlas
        // borders). Atlas border grab zones sit on top of the heatmap, so a
        // drag starting on a border never reaches here.
        if resp.drag_started_by(egui::PointerButton::Primary) && !alt_held {
            self.zoom_drag_start = ctx.input(|i| i.pointer.press_origin());
        }
        if let Some(start) = self.zoom_drag_start {
            let end = ctx.input(|i| i.pointer.interact_pos()).unwrap_or(start);
            let sel = egui::Rect::from_two_pos(start, end).intersect(resp.rect);
            if resp.drag_stopped() || !ctx.input(|i| i.pointer.primary_down()) {
                self.zoom_drag_start = None;
                let chans = zoom_selection(display_rows, first_row, last_row, resp.rect, sel);
                if sel.width() >= ZOOM_MIN_DRAG_PX && sel.height() >= ZOOM_MIN_DRAG_PX {
                    if let Some((ch_bottom, ch_top)) = chans {
                        let w = resp.rect.width() as f64;
                        let t0 = self.view_start_s + (sel.left() - resp.rect.left()) as f64 / w * self.view_dur_s;
                        let t1 = self.view_start_s + (sel.right() - resp.rect.left()) as f64 / w * self.view_dur_s;
                        // nested zooms keep the view from before the first one
                        let (prev_start_s, prev_dur_s) = self
                            .zoom
                            .as_ref()
                            .map_or((self.view_start_s, self.view_dur_s), |z| (z.prev_start_s, z.prev_dur_s));
                        let total_s = self.meta.n_samples as f64 / self.meta.sample_rate;
                        self.view_dur_s = (t1 - t0).max(0.01);
                        self.view_start_s = t0.clamp(0.0, (total_s - self.view_dur_s).max(0.0));
                        self.window_dur_str = format!("{:.3}", self.view_dur_s);
                        self.zoom = Some(Zoom { ch_bottom, ch_top, prev_start_s, prev_dur_s });
                        ctx.request_repaint();
                    }
                }
            } else {
                ui.painter().rect_filled(sel, 0.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 3));
                ui.painter().rect_stroke(
                    sel,
                    0.0,
                    egui::Stroke::new(1.0_f32, egui::Color32::WHITE),
                    egui::StrokeKind::Inside,
                );
            }
        }

        // handle clicks
        if let Some(pos) = click_pos {
            let frac_y = ((pos.y - resp.rect.top()) / resp.rect.height()).clamp(0.0, 1.0);
            let disp_idx = last_row.saturating_sub(
                (frac_y as f64 * n_rows as f64) as usize
            ).clamp(first_row, last_row);

            if let DisplayRow::Data { first_ch, .. } = &display_rows[disp_idx] {
                let ch = *first_ch + 1;
                if is_left_click {
                    self.selected_channel_1 = Some(ch);
                }
                if is_right_click {
                    self.selected_channel_2 = Some(ch);
                }
                if is_context_click {
                    self.context_menu_channel = Some(ch);
                    self.context_menu_pos = Some(pos);
                }
            }
        }
    }

    /// Channel lines, shank labels, firing rate, classification, atlas and spikes.
    pub(super) fn draw_heatmap_overlays(
        &mut self,
        ui: &mut Ui,
        resp: &egui::Response,
        full_right: f32,
        display_rows: &[DisplayRow],
        (first_row, last_row): (usize, usize),
        shot: ShotOverlays,
    ) {
        let n_rows = last_row.saturating_sub(first_row) + 1;
        // draw channel marker lines
        let draw_line = |ch_to_draw: usize, color: egui::Color32| {
            for (r, row) in (first_row..).zip(&display_rows[first_row..=last_row]) {
                if let DisplayRow::Data { first_ch, .. } = row {
                    if *first_ch + 1 == ch_to_draw {
                        let frac_y = (last_row - r) as f32 / n_rows as f32 + (0.5 / n_rows as f32);
                        let y = resp.rect.top() + frac_y * resp.rect.height();
                        ui.painter().line_segment(
                            [egui::pos2(resp.rect.left(), y), egui::pos2(full_right, y)],
                            egui::Stroke::new(2.0_f32, color)
                        );
                        break;
                    }
                }
            }
        };

        let [fr, fg, fb] = self.colormap_choice.spec().heatmap_fg;
        if shot.selection {
            if let Some(ch1) = self.selected_channel_1 {
                draw_line(ch1, egui::Color32::from_rgba_unmultiplied(fr, fg, fb, 128));
            }
            if let Some(ch2) = self.selected_channel_2 {
                draw_line(ch2, egui::Color32::from_rgba_unmultiplied(255, 182, 23, 128));
            }
        }
        if let Some(ctx_ch) = self.context_menu_channel.filter(|_| self.capture.is_none()) {
            draw_line(ctx_ch, egui::Color32::from_rgba_unmultiplied(fr, fg, fb, 100));
        }

        // label each shank's section with "shank N", at the top-right
        // corner of its (possibly partial) visible span — only when more
        // than one shank is actually visible
        let mut shank_tops: Vec<(u32, usize)> = Vec::new(); // (shank, topmost visible row idx)
        for (r, row) in (first_row..).zip(&display_rows[first_row..=last_row]) {
            if let DisplayRow::Data { shank, .. } = row {
                match shank_tops.last_mut() {
                    Some((s, top)) if *s == *shank => *top = r,
                    _ => shank_tops.push((*shank, r)),
                }
            }
        }
        if shank_tops.len() > 1 {
            for (shank, top_idx) in shank_tops {
                let frac_top = (last_row - top_idx) as f32 / n_rows as f32;
                let y = resp.rect.top() + frac_top * resp.rect.height();
                ui.painter().text(
                    egui::pos2(resp.rect.right() - 6.0, y + 2.0),
                    egui::Align2::RIGHT_TOP,
                    format!("shank {shank}"),
                    egui::FontId::proportional(12.0),
                    egui::Color32::from_rgba_unmultiplied(fr, fg, fb, 200),
                );
            }
        }

        // draw projection overlay
        if self.show_firing_rate_overlay && !self.projection.sums.is_empty() {
            // scale with window duration and threshold (baseline: -20 µV → 1x),
            // plus a user-configurable multiplier (spike_overlay_scale, default 1)
            let threshold_scale = self.spike_threshold.abs() / 20.0;
            let spike_scale_factor = threshold_scale * (0.5 / self.view_dur_s) as f32 * self.spike_overlay_scale;

            // per-map alpha is tuned for this overlay's many-triangle accumulation,
            // so it is a separate field from the shared accent RGB in colormap.rs
            let overlay_alpha = self.colormap_choice.spec().overlay_alpha;
            let [pr, pg, pb] = self.colormap_choice.spec().accent;
            let color = egui::Color32::from_rgba_unmultiplied(pr, pg, pb, overlay_alpha);

            let min_x = resp.rect.left();
            let max_x = resp.rect.right();
            let top_y = resp.rect.top();
            let h = resp.rect.height();
            let row_h = h / n_rows as f32;

            let mut mesh = egui::epaint::Mesh::default();

            for (i, &count) in self.projection.sums.iter().enumerate() {
                let x = min_x + count * spike_scale_factor;
                let x = x.min(max_x);

                let y = top_y + h - (i as f32 + 0.5) * row_h;

                let idx_base = mesh.vertices.len() as u32;
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: egui::pos2(min_x, y),
                    uv: egui::epaint::WHITE_UV,
                    color,
                });
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: egui::pos2(x, y),
                    uv: egui::epaint::WHITE_UV,
                    color,
                });

                if i > 0 {
                    mesh.indices.push(idx_base - 2);
                    mesh.indices.push(idx_base - 1);
                    mesh.indices.push(idx_base);

                    mesh.indices.push(idx_base - 1);
                    mesh.indices.push(idx_base + 1);
                    mesh.indices.push(idx_base);
                }
            }

            if !mesh.is_empty() {
                ui.painter().add(egui::Shape::mesh(mesh));
            }
        }

        // channel classification overlay: horizontal stripe per
        // non-good row (worst label wins when avg_depths merges
        // several raw channels into one row: dead > noisy > outside)
        if self.classify.show_overlay {
            if let Some(labels) = &self.classify.labels {
                let top_y = resp.rect.top();
                let h = resp.rect.height();
                let row_h = h / n_rows as f32;
                let min_x = resp.rect.left();
                let max_x = resp.rect.right();

                for (r, row) in (first_row..).zip(&display_rows[first_row..=last_row]) {
                    if let DisplayRow::Data { channels, .. } = row {
                        let worst = channels
                            .iter()
                            .filter_map(|&ch| labels.get(ch).copied())
                            .max_by_key(|&l| match l { 1 => 3, 2 => 2, 3 => 1, _ => 0 });
                        if let Some(l) = worst {
                            if l != 0 {
                                let frac_top = (last_row - r) as f32 / n_rows as f32;
                                let y0 = top_y + frac_top * h;
                                let rect = egui::Rect::from_min_size(
                                    egui::pos2(min_x, y0),
                                    egui::vec2(max_x - min_x, row_h),
                                );
                                ui.painter().rect_filled(rect, 0.0, classification_color(l, CLASSIFICATION_OVERLAY_ALPHA));
                            }
                        }
                    }
                }
            }
        }

        // atlas region borders + labels, on top of the firing-rate overlay
        let shown = RowsOnScreen { rows: display_rows, first: first_row, last: last_row, rect: resp.rect };
        self.atlas.draw_overlay(ui, &self.meta, shown, full_right, self.preproc_cfg.channel_order, &self.colormap_choice);

        // spike-sorted spikes, on top of the other overlays
        self.spikes.draw_overlay(&ui.painter_at(resp.rect), shown, (self.view_start_s, self.view_dur_s), &self.colormap_choice);
    }

    /// Legend box: one Show/Hide row per overlay that has been computed this session.
    pub(super) fn draw_legend(&mut self, ctx: &egui::Context, ui: &mut Ui, resp: &egui::Response) {
        // legend box: one Show/Hide row per overlay that has been computed
        // this session (Events, Atlas, firing rate, power spectrum, channel
        // classification), pinned to the heatmap's bottom-right corner just
        // above the scale bar. Laid out inside the heatmap panel itself (not
        // a floating Area), so every window the user opens stays on top of it.
        if self.capture.is_none()
            && (self.classify.labels.is_some()
                || self.events.has_data()
                || self.spikes.has_data()
                || self.atlas.has_data()
                || !self.projection.sums.is_empty()
                || self.spectrum.result.is_some())
        {
            // anchored by its bottom-right corner; the scale bar's top
            // edge sits 30 px above the heatmap bottom. The box's size is
            // only known after layout, so last frame's size places it and
            // a size change (legend shown/hidden) re-runs the layout pass.
            let size_id = egui::Id::new("classify_box_size");
            let prev_size: egui::Vec2 =
                ctx.data(|d| d.get_temp(size_id)).unwrap_or_default();
            let anchor = egui::pos2(resp.rect.right() - 8.0, resp.rect.bottom() - 38.0);
            let mut box_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(egui::Rect::from_min_max(anchor - prev_size, anchor))
                    .layout(egui::Layout::top_down(egui::Align::Min)),
            );
            box_ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            box_ui.visuals_mut().override_text_color = Some(egui::Color32::WHITE);
            let bg = egui::Color32::from_rgba_unmultiplied(
                crate::render::C_ZERO[0],
                crate::render::C_ZERO[1],
                crate::render::C_ZERO[2],
                77,
            );
            let box_size = egui::Frame::new()
                .fill(bg)
                .corner_radius(8.0)
                .inner_margin(8.0)
                .show(&mut box_ui, |ui| {
                    ui.vertical(|ui| {
                        let legend_row = |ui: &mut Ui, color: egui::Color32, text: &str| {
                            ui.horizontal(|ui| {
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(12.0, 12.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(rect, 2.0, color);
                                ui.label(text);
                            });
                        };
                        let mut any_above = false;

                        if self.events.has_data() {
                            if self.events.show_overlay {
                                let [r, g, b] = self.colormap_choice.spec().accent;
                                legend_row(ui, egui::Color32::from_rgb(r, g, b), "Events");
                            }
                            let label = if self.events.show_overlay { "Hide events" } else { "Show events" };
                            ui.toggle_value(&mut self.events.show_overlay, label);
                            any_above = true;
                        }

                        if self.spikes.has_data() {
                            if any_above {
                                ui.add_space(4.0);
                            }
                            if self.spikes.show_overlay {
                                let spec = self.colormap_choice.spec();
                                ui.horizontal(|ui| {
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(12.0, 12.0),
                                        egui::Sense::hover(),
                                    );
                                    let [r, g, b] = spec.spike_outline;
                                    ui.painter().circle(
                                        rect.center(),
                                        3.5,
                                        egui::Color32::from_rgb(spec.spike_fill[0], spec.spike_fill[1], spec.spike_fill[2]),
                                        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(r, g, b)),
                                    );
                                    ui.label(format!("Spikes ({} units)", self.spikes.n_units()));
                                });
                                if let Some(n) = self.spikes.too_many() {
                                    ui.label(format!("{n} in view: zoom in to show them"));
                                }
                                if let Some(note) = self.spikes.note() {
                                    ui.label(egui::RichText::new(note).small().color(egui::Color32::LIGHT_GRAY));
                                }
                            }
                            let label = if self.spikes.show_overlay { "Hide spikes" } else { "Show spikes" };
                            ui.toggle_value(&mut self.spikes.show_overlay, label);
                            any_above = true;
                        }

                        if self.atlas.has_data() {
                            if any_above {
                                ui.add_space(4.0);
                            }
                            if self.atlas.show_overlay {
                                let [r, g, b] = self.colormap_choice.spec().atlas;
                                legend_row(ui, egui::Color32::from_rgb(r, g, b), "Atlas");
                            }
                            let label = if self.atlas.show_overlay { "Hide Atlas" } else { "Show Atlas" };
                            ui.toggle_value(&mut self.atlas.show_overlay, label);
                            any_above = true;
                        }

                        if !self.projection.sums.is_empty() {
                            if any_above {
                                ui.add_space(4.0);
                            }
                            if self.show_firing_rate_overlay {
                                let [r, g, b] = self.colormap_choice.spec().accent;
                                legend_row(ui, egui::Color32::from_rgb(r, g, b), "Firing rate");
                            }
                            let label = if self.show_firing_rate_overlay {
                                "Hide firing rate"
                            } else {
                                "Show firing rate"
                            };
                            ui.toggle_value(&mut self.show_firing_rate_overlay, label);
                            any_above = true;
                        }

                        if self.spectrum.result.is_some() {
                            if any_above {
                                ui.add_space(4.0);
                            }
                            if self.spectrum.show_overlay {
                                // swatch: the spectrum palette, low to high power
                                ui.horizontal(|ui| {
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(12.0, 12.0),
                                        egui::Sense::hover(),
                                    );
                                    for i in 0..12 {
                                        let [r, g, b] = crate::render::spectrum_color(i as f32 / 11.0);
                                        let x = rect.left() + i as f32;
                                        ui.painter().rect_filled(
                                            egui::Rect::from_x_y_ranges(x..=x + 1.0, rect.y_range()),
                                            0.0,
                                            egui::Color32::from_rgb(r, g, b),
                                        );
                                    }
                                    ui.label("Power spectrum");
                                });
                            }
                            let label =
                                if self.spectrum.show_overlay { "Hide Spectra" } else { "Show Spectra" };
                            ui.toggle_value(&mut self.spectrum.show_overlay, label);
                            any_above = true;
                        }

                        if self.classify.labels.is_some() {
                            if any_above {
                                ui.add_space(4.0);
                            }
                            if self.classify.show_overlay {
                                legend_row(ui, classification_color(1, 255), "Dead");
                                legend_row(ui, classification_color(2, 255), "Noisy");
                                legend_row(ui, classification_color(3, 255), "Out of brain");
                            }
                            if self.classify.show_overlay || any_above {
                                ui.add_space(4.0);
                            }
                            let label = if self.classify.show_overlay {
                                "Hide Chan Classification"
                            } else {
                                "Show Chan Classification"
                            };
                            ui.toggle_value(&mut self.classify.show_overlay, label);
                        }
                    });
                })
                .response
                .rect
                .size();
            if (box_size - prev_size).length() > 0.5 {
                ctx.data_mut(|d| d.insert_temp(size_id, box_size));
                ctx.request_discard("classification box resized");
            }
        }
    }

    /// Channel, time and value under the pointer.
    pub(super) fn draw_hover_readout(&self, ctx: &egui::Context, ui: &Ui, resp: &egui::Response, frame: &Frame) {
        // hover overlay: ch / time / voltage
        let hover_pos = resp.hover_pos().or_else(|| {
            if resp.dragged() || resp.is_pointer_button_down_on() {
                ctx.input(|i| i.pointer.interact_pos())
            } else {
                None
            }
        }).filter(|_| self.capture.is_none());

        if let Some(pos) = hover_pos {
            if let Some(display_rows) = frame.buf.as_ref().map(|w| &w.display_rows).filter(|r| !r.is_empty()) {
                let (first_row, last_row) = view_rows(display_rows, self.zoom.as_ref());
                let n_rows = last_row.saturating_sub(first_row) + 1;

                let frac_y = ((pos.y - resp.rect.top()) / resp.rect.height()).clamp(0.0, 1.0);
                let disp_idx = last_row.saturating_sub(
                    (frac_y as f64 * n_rows as f64) as usize
                ).clamp(first_row, last_row);

                let ch_str = match &display_rows[disp_idx] {
                    DisplayRow::Data { first_ch, .. } => {
                        let id = self.meta.channel_id(*first_ch);
                        match self.atlas.channel_region_label(*first_ch) {
                            Some(region) => format!("{id}  {region}  "),
                            None => format!("{id}  "),
                        }
                    }
                    DisplayRow::IntraShankGap => "Channel gap  ".to_string(),
                    DisplayRow::ShankBoundary => "Shank gap  ".to_string(),
                };

                let frac_x = ((pos.x - resp.rect.left()) / resp.rect.width()).clamp(0.0, 1.0);
                let t = self.view_start_s + frac_x as f64 * self.view_dur_s;

                // voltage readout from snapshot data
                let voltage_uv: Option<f32> = if let Some(DisplayRow::Data { data_idx, .. }) = display_rows.get(disp_idx) {
                    // same samples as drawn (noise-filtered when that is on)
                    if let Some(data) = &frame.src_data {
                        let t_sample = (t * self.meta.sample_rate) as usize;
                        if t_sample >= frame.src_first {
                            let off = t_sample - frame.src_first;
                            let idx = data_idx * frame.src_n + off;
                            if off < frame.src_n && idx < data.len() {
                                Some(data[idx])
                            } else { None }
                        } else { None }
                    } else { None }
                } else { None };

                let p = if self.display_is_csd { 3 } else { 1 };
                let volt_str = voltage_uv.map(|v| format!("  {:.p$} {}", v, self.unit())).unwrap_or_default();
                let mut label = format!("{}t = {:.4} s{}", ch_str, t, volt_str);
                // while dragging a zoom rectangle: the channel and time ranges it covers
                if let Some(start) = self.zoom_drag_start {
                    let sel = egui::Rect::from_two_pos(start, pos).intersect(resp.rect);
                    let t_at = |x: f32| {
                        self.view_start_s
                            + ((x - resp.rect.left()) / resp.rect.width()) as f64 * self.view_dur_s
                    };
                    let chans = zoom_selection(display_rows, first_row, last_row, resp.rect, sel)
                        .map(|(b, t)| format!("{}–{}  ", self.meta.channel_id(b), self.meta.channel_id(t)))
                        .unwrap_or_default();
                    label = format!("{chans}t = {:.4}–{:.4} s", t_at(sel.left()), t_at(sel.right()));
                }

                draw_readout(ui.painter(), resp.rect, label);
            }
        }
    }

    /// Scale bar and zoom notice.
    pub(super) fn draw_plot_decorations(&self, ui: &Ui, resp: &egui::Response, shot: ShotOverlays) {
        // scale bar overlay (10% of view_dur_s) bottom right
        let [fr, fg, fb] = self.colormap_choice.spec().heatmap_fg;
        let fg_color = egui::Color32::from_rgb(fr, fg, fb);
        if shot.scale_bar {
            let scale_bar_frac = 0.1;
            let scale_bar_w = resp.rect.width() * scale_bar_frac;
            let scale_bar_h = 4.0;
            let bar_min = resp.rect.right_bottom() - egui::vec2(scale_bar_w + 20.0, 30.0);
            let bar_rect = egui::Rect::from_min_size(bar_min, egui::vec2(scale_bar_w, scale_bar_h));
            ui.painter().rect_filled(bar_rect, 0.0, fg_color);

            let dur_ms = self.view_dur_s * (scale_bar_frac as f64) * 1000.0;
            ui.painter().text(
                bar_rect.right_bottom() + egui::vec2(0.0, 5.0),
                egui::Align2::RIGHT_TOP,
                format!("{:.0} ms", dur_ms),
                egui::FontId::proportional(14.0),
                fg_color,
            );
        }

        // zoom notice, top left
        if self.zoom.is_some() && self.capture.is_none() {
            draw_zoom_notice(ui.painter(), resp.rect.left_top() + egui::vec2(10.0, 10.0), &self.colormap_choice);
        }
    }

    /// Mouse wheel over the plot: scrolls the view in time (Alt: changes the colour
    /// scale, or the waveform view's voltage range). Left / right arrows (A / D) step
    /// the view by half its length.
    pub(super) fn handle_view_input(&mut self, ctx: &egui::Context, alt_held: bool) {
        // mouse-wheel scroll — 5% of window per tick; Alt+scroll instead adjusts the
        // color scale (or, in single-channel waveform view, the waveform's y-axis range)
        // wheel over a window (Atlas Registration, Preferences, PSTH, ...) scrolls only
        // that window; panels, incl. the heatmap, live on the Background layer
        let pointer_over_window = ctx
            .input(|i| i.pointer.hover_pos())
            .and_then(|p| ctx.layer_id_at(p))
            .is_some_and(|layer| layer.order != egui::Order::Background);
        let ticks = if pointer_over_window {
            0.0
        } else {
            ctx.input(|i| {
                i.events
                    .iter()
                    .filter_map(|e| match e {
                        egui::Event::MouseWheel { delta, .. } => Some(delta.y.signum()),
                        _ => None,
                    })
                    .sum::<f32>()
            })
        };
        if ticks != 0.0 && alt_held {
            // inverted relative to a plain sum of tick signs: scrolling up now
            // decreases the value (mirrors the "zoom out" feel of scroll-up elsewhere)
            if self.waveform_channel.is_some() {
                // proportional steps, so the scale stays usable when zoomed to a few µV
                self.waveform_y_range_uv = (self.waveform_y_range_uv * 1.05f32.powf(-ticks))
                    .clamp(self.waveform_min_range(), 2000.0);
            } else if self.color_mode == ColorMode::Percentile {
                let (min, step) = if self.display_is_csd { (CSD_PCT_MIN, 0.5) } else { (95.0, 0.1) };
                self.color_pct = (self.color_pct - ticks * step).clamp(min, 100.0);
                self.color_pct_str = format!("{:.2}", self.color_pct);
            } else if self.display_is_csd {
                self.color_uv = (self.color_uv * 1.1f32.powf(-ticks)).clamp(CSD_SCALE_RANGE.0, CSD_SCALE_RANGE.1);
            } else {
                self.color_uv = (self.color_uv - ticks * 5.0).clamp(10.0, 300.0);
                self.color_uv_str = format!("{:.0}", self.color_uv);
            }
        } else if ticks != 0.0 {
            let fs = self.meta.sample_rate;
            let total_s = self.meta.n_samples as f64 / fs;
            let max_start = (total_s - self.view_dur_s).max(0.0);
            let pct = if self.scroll_speed_fine { 0.05 } else { 0.30 };
            let step = ticks as f64 * self.view_dur_s * pct;
            self.view_start_s = (self.view_start_s + step).clamp(0.0, max_start);
        }

        // keyboard scroll (not while typing into a text field)
        let typing = ctx.wants_keyboard_input();
        ctx.input(|i| {
            if typing {
                return;
            }
            let fs = self.meta.sample_rate;
            let total_s = self.meta.n_samples as f64 / fs;
            let step = self.view_dur_s * 0.5;
            let max_start = (total_s - self.view_dur_s).max(0.0);
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::D) {
                self.view_start_s = (self.view_start_s + step).min(max_start);
            }
            if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::A) {
                self.view_start_s = (self.view_start_s - step).max(0.0);
            }
        });
    }
}
