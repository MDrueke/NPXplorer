use crate::data::DisplayRow;

pub const C_ZERO: [u8; 3] = [0x26, 0x29, 0x30]; // #262930 grey

/// Single representative accent color per colormap. This is the one place to edit
/// when tuning colors that need to track the active colormap outside the heatmap
/// itself — e.g. the nav bar's view/buffer markers and the spike projection overlay.
pub fn colormap_accent(cmap: &crate::app::ColorMapChoice) -> [u8; 3] {
    match cmap {
        crate::app::ColorMapChoice::YellowMagenta => [250, 234, 130],
        crate::app::ColorMapChoice::RedBlue => [204, 103, 230],
        crate::app::ColorMapChoice::OrangeBlue => [242, 171, 126],
        crate::app::ColorMapChoice::IceFire => [166, 217, 237],
        crate::app::ColorMapChoice::Vanimo => [202, 237, 166],
        crate::app::ColorMapChoice::GreyScale => [255, 255, 255],
    }
}

/// Opacity (0-255) of the nav bar's "preprocessed buffer extent" shading.
/// 26 ≈ 10% — the one place to tune this.
pub const BUFFER_EXTENT_ALPHA: u8 = 10;

/// Opacity (0-255) of the nav bar's "currently displayed view" marker.
pub const VIEW_MARKER_ALPHA: u8 = 180;

#[inline]
fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t) as u8,
    ]
}

#[inline]
fn interpolate_stops(t: f32, stops: &[[u8; 3]]) -> [u8; 3] {
    let n = stops.len() - 1;
    let scaled_t = t * n as f32;
    let idx = scaled_t.floor() as usize;
    if idx >= n {
        return stops[n];
    }
    let local_t = scaled_t - idx as f32;
    lerp_rgb(stops[idx], stops[idx + 1], local_t)
}

#[inline]
pub fn voltage_to_rgba(v: f32, vmax: f32, cmap: &crate::app::ColorMapChoice) -> [u8; 4] {
    let t = (v / vmax).clamp(-1.0, 1.0); // -1..1

    let [r, g, b] = match cmap {
        crate::app::ColorMapChoice::YellowMagenta => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        C_ZERO,
                        [0x44, 0x2a, 0x4a],
                        [0x5d, 0x33, 0x66],
                        [0x7b, 0x26, 0x8c],
                        [0x93, 0x04, 0xb0],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x33, 0x31, 0x26],
                        [0x3d, 0x39, 0x1f],
                        [0x52, 0x4b, 0x1e],
                        [0x75, 0x6a, 0x1e],
                        [0xa3, 0x90, 0x12],
                        [0xff, 0xdf, 0x12],
                    ],
                )
            }
        }
        crate::app::ColorMapChoice::RedBlue => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        C_ZERO,
                        [0x2e, 0x30, 0x42],
                        [0x25, 0x2c, 0x61],
                        [0x24, 0x34, 0xb3],
                        [0x2c, 0x43, 0xf5],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x40, 0x2c, 0x2b],
                        [0x61, 0x2f, 0x2c],
                        [0x9e, 0x32, 0x2b],
                        [0xf5, 0x43, 0x36],
                    ],
                )
            }
        }
        crate::app::ColorMapChoice::OrangeBlue => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        C_ZERO,
                        [0x29, 0x3b, 0x54],
                        [0x31, 0x54, 0x85],
                        [0x2d, 0x6f, 0xc4],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x4a, 0x29, 0x22],
                        [0x75, 0x36, 0x28],
                        [0xd1, 0x42, 0x21],
                    ],
                )
            }
        }
        crate::app::ColorMapChoice::IceFire => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        C_ZERO,
                        [0x39, 0x32, 0x47],
                        [0x39, 0x29, 0x5c],
                        [0x46, 0x27, 0x8a],
                        [0x20, 0x5f, 0x9e],
                        [0x71, 0xb5, 0xbd],
                        [0x93, 0xcf, 0xc9],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x40, 0x31, 0x30],
                        [0x4d, 0x2f, 0x2d],
                        [0x5e, 0x29, 0x25],
                        [0x8a, 0x24, 0x1d],
                        [0xba, 0x4f, 0x22],
                        [0xd9, 0xa2, 0x73],
                    ],
                )
            }
        }
        crate::app::ColorMapChoice::Vanimo => {
            if t >= 0.0 {
                interpolate_stops(
                    t,
                    &[
                        C_ZERO,
                        [0x2e, 0x36, 0x27],
                        [0x3c, 0x52, 0x27],
                        [0x56, 0x8a, 0x22],
                        [0x8d, 0xed, 0x2d],
                    ],
                )
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x43, 0x31, 0x47],
                        [0x66, 0x35, 0x73],
                        [0xb9, 0x4e, 0xd4],
                    ],
                )
            }
        }
        crate::app::ColorMapChoice::GreyScale => {
            if t >= 0.0 {
                interpolate_stops(t, &[C_ZERO, [0x00, 0x00, 0x00]])
            } else {
                interpolate_stops(
                    -t,
                    &[
                        C_ZERO,
                        [0x30, 0x30, 0x30],
                        [0x50, 0x50, 0x50],
                        [0x60, 0x60, 0x60],
                        [0xd0, 0xd0, 0xd0],
                    ],
                )
            }
        }
    };
    [r, g, b, 255]
}

// ---------------------------------------------------------------------------
// Heatmap renderer
//
// `display_rows` — the full ordered list of rows to render (Data + Gap variants).
//   Data rows carry a `data_idx` into the flat `data` buffer.
//   Gap rows are rendered as a solid light-grey stripe.
//
// `first_row_idx` / `last_row_idx` — indices into `display_rows` to render
//   (the channel-range selection from the UI sliders).
// ---------------------------------------------------------------------------

pub fn build_heatmap_into(
    out: &mut Vec<u8>,
    data: &[f32],
    display_rows: &[DisplayRow],
    first_row_idx: usize, // first display_row index to render
    last_row_idx: usize,  // last display_row index to render (inclusive)
    data_stride: usize,   // n_samp in the buffer
    buf_first: usize,     // absolute first sample covered by the buffer
    buf_n_samp: usize,    // number of samples covered by the buffer
    view_first: usize,    // absolute first sample of the requested view
    n_view: usize,        // number of samples in the requested view
    pixel_w: usize,
    pixel_h: usize,
    vmax: f32,
    cmap: &crate::app::ColorMapChoice,
) {
    use rayon::prelude::*;

    let total = pixel_w * pixel_h * 4;
    out.resize(total, 0);

    if pixel_w == 0 || pixel_h == 0 || n_view == 0 {
        return;
    }

    let first = first_row_idx.min(display_rows.len().saturating_sub(1));
    let last = last_row_idx.min(display_rows.len().saturating_sub(1));
    let visible = &display_rows[first..=last];
    let n_rows = visible.len();
    if n_rows == 0 {
        return;
    }

    let row_bytes = pixel_w * 4;
    let buf_end = buf_first + buf_n_samp;

    out.par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(py, row)| {
            // map pixel row → display row (ch_last at top, ch_first at bottom)
            let disp_idx = n_rows
                .saturating_sub(1)
                .saturating_sub((py * n_rows) / pixel_h);
            let disp_idx = disp_idx.min(n_rows - 1);

            match &visible[disp_idx] {
                DisplayRow::IntraShankGap => {
                    // Background grey with dotted grey line
                    for (px_idx, px) in row.chunks_exact_mut(4).enumerate() {
                        if (px_idx / 4) % 2 == 0 {
                            px[0] = 0x60;
                            px[1] = 0x60;
                            px[2] = 0x60;
                            px[3] = 255; // grey dot
                        } else {
                            px[0] = C_ZERO[0];
                            px[1] = C_ZERO[1];
                            px[2] = C_ZERO[2];
                            px[3] = 255;
                        }
                    }
                }
                DisplayRow::ShankBoundary => {
                    // solid white inter-shank separator
                    for px in row.chunks_exact_mut(4) {
                        px[0] = 255;
                        px[1] = 255;
                        px[2] = 255;
                        px[3] = 255;
                    }
                }
                DisplayRow::Data { data_idx, .. } => {
                    let row_base = data_idx * data_stride;
                    if row_base + data_stride > data.len() {
                        // row doesn't exist in this buffer at all — background fill
                        for px in row.chunks_exact_mut(4) {
                            px[0] = C_ZERO[0];
                            px[1] = C_ZERO[1];
                            px[2] = C_ZERO[2];
                            px[3] = 255;
                        }
                        return;
                    }
                    let ch_data = &data[row_base..row_base + data_stride];
                    for (px_col, px) in row.chunks_exact_mut(4).enumerate() {
                        let t0 = (px_col * n_view) / pixel_w;
                        let t1 = (((px_col + 1) * n_view) / pixel_w).min(n_view);
                        let (has_range, abs_lo, abs_hi) = if t1 > t0 {
                            (true, view_first + t0, view_first + t1)
                        } else if t0 < n_view {
                            (true, view_first + t0, view_first + t0 + 1)
                        } else {
                            (false, 0, 0)
                        };
                        // background fill wherever the buffer hasn't been preprocessed yet
                        // (e.g. still extending, or a jump landed ahead of what's ready)
                        let valid = has_range && abs_lo >= buf_first && abs_hi <= buf_end;
                        if !valid {
                            px[0] = C_ZERO[0];
                            px[1] = C_ZERO[1];
                            px[2] = C_ZERO[2];
                            px[3] = 255;
                            continue;
                        }
                        let buf_lo = abs_lo - buf_first;
                        let buf_hi = abs_hi - buf_first;
                        let v = if buf_hi > buf_lo {
                            ch_data[buf_lo..buf_hi].iter().copied().sum::<f32>()
                                / (buf_hi - buf_lo) as f32
                        } else {
                            ch_data[buf_lo]
                        };
                        let rgba = voltage_to_rgba(v, vmax, cmap);
                        px[0] = rgba[0];
                        px[1] = rgba[1];
                        px[2] = rgba[2];
                        px[3] = rgba[3];
                    }
                }
            }
        });
}
