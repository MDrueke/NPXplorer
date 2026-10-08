use anyhow::{Result, bail};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering, AtomicUsize};

use crate::data::{DisplayRow, Meta, RawData};
use crate::preprocess::{Filters, PreprocConfig, SpatialFilter, preprocess};
use crate::worker::compute_thread_count;

// ---------------------------------------------------------------------------
// Encoding-robust text reading
// ---------------------------------------------------------------------------

/// Read a text file without assuming UTF-8. Handles UTF-8 (with/without BOM) and
/// UTF-16 LE/BE (with BOM); anything else that isn't valid UTF-8 is decoded as
/// Latin-1 (ISO-8859-1), which maps every byte to a char and so never fails. This
/// keeps event files exported from Excel/MATLAB/Python on any platform readable
/// without adding an encoding dependency.
pub fn read_text_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("could not read {}: {e}", path.display()))?;

    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Ok(String::from_utf8_lossy(&bytes[3..]).into_owned());
    }
    if bytes.starts_with(&[0xFF, 0xFE]) {
        return Ok(decode_utf16(&bytes[2..], false));
    }
    if bytes.starts_with(&[0xFE, 0xFF]) {
        return Ok(decode_utf16(&bytes[2..], true));
    }
    match String::from_utf8(bytes) {
        Ok(s) => Ok(s),
        // not valid UTF-8: fall back to Latin-1 (each byte -> code point)
        Err(e) => Ok(e.into_bytes().iter().map(|&b| b as char).collect()),
    }
}

fn decode_utf16(bytes: &[u8], big_endian: bool) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| if big_endian { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) })
        .collect();
    String::from_utf16_lossy(&units)
}

/// Split a data line into fields. Uses commas if present (CSV/TSV-with-commas),
/// otherwise falls back to any-whitespace splitting.
fn split_fields(line: &str) -> Vec<&str> {
    if line.contains(',') {
        line.split(',').map(|f| f.trim()).collect()
    } else {
        line.split_whitespace().collect()
    }
}

// ---------------------------------------------------------------------------
// Layout file
// ---------------------------------------------------------------------------

/// Describes where event onset (and optionally offset) times live in an event file.
/// Determined by a layout file whose lines mirror the event file's structure:
/// leading lines with no `o` token are header rows to skip; the first line that
/// contains one or more `o` tokens marks which column(s) hold the onset times, and
/// `f` tokens on that line the offset times (paired with the onsets in order).
/// Trailing `x` markers beyond the actual number of columns are ignored.
#[derive(Clone, Debug)]
pub struct EventLayout {
    pub n_header_rows: usize,
    pub onset_cols: Vec<usize>,
    pub offset_cols: Vec<usize>,
}

impl EventLayout {
    pub fn parse(text: &str) -> Result<Self> {
        let mut n_header_rows = 0usize;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue; // ignore blank lines and comments in the layout
            }
            let tokens = split_fields(trimmed);
            let cols_marked = |m: &str| -> Vec<usize> {
                tokens
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.trim().eq_ignore_ascii_case(m))
                    .map(|(i, _)| i)
                    .collect()
            };
            let onset_cols = cols_marked("o");
            if onset_cols.is_empty() {
                // a header / ignored line
                n_header_rows += 1;
                continue;
            }
            let offset_cols = cols_marked("f");
            if !offset_cols.is_empty() && offset_cols.len() != onset_cols.len() {
                bail!(
                    "the layout marks {} onset column(s) ('o') but {} offset column(s) ('f'). \
                     Mark one offset column per onset column, or none.",
                    onset_cols.len(),
                    offset_cols.len()
                );
            }
            return Ok(EventLayout { n_header_rows, onset_cols, offset_cols });
        }
        bail!(
            "the layout file contains no 'o' marker, so it does not say which column \
             holds the event onset times. Mark the onset column with 'o' (e.g. 'o,x,x')."
        );
    }
}

// ---------------------------------------------------------------------------
// Loading event times
// ---------------------------------------------------------------------------

/// Read event onset times (seconds) from `event_path`, using `layout` to locate
/// the onset column and skip header rows. Errors describe exactly how the layout
/// disagrees with the file rather than panicking.
pub fn load_event_times(event_path: &Path, layout: &EventLayout) -> Result<Vec<f64>> {
    load_columns(event_path, layout.n_header_rows, &layout.onset_cols, "onset")
}

/// Read event offset times (seconds) from the `f` columns, in the same order as
/// the onsets from `load_event_times`. `None` if the layout marks no offset column.
pub fn load_event_offsets(event_path: &Path, layout: &EventLayout) -> Result<Option<Vec<f64>>> {
    if layout.offset_cols.is_empty() {
        return Ok(None);
    }
    load_columns(event_path, layout.n_header_rows, &layout.offset_cols, "offset").map(Some)
}

fn load_columns(event_path: &Path, n_header_rows: usize, cols: &[usize], what: &str) -> Result<Vec<f64>> {
    let text = read_text_file(event_path)?;
    let event_name = event_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();

    let max_col = *cols.iter().max().unwrap_or(&0);
    let mut times = Vec::new();

    // human-facing row numbers count every line so they match a text editor
    for (line_no, line) in text.lines().enumerate().skip(n_header_rows) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue; // tolerate blank/trailing lines in the data
        }
        let fields = split_fields(trimmed);
        if max_col >= fields.len() {
            bail!(
                "layout does not match '{event_name}': the layout marks column {} as the \
                 event {what} time, but row {} has only {} column(s). Check the number of \
                 header rows and the {what} column in the layout file.",
                max_col + 1,
                line_no + 1,
                fields.len()
            );
        }
        for &c in cols {
            let tok = fields[c];
            match tok.parse::<f64>() {
                Ok(v) => times.push(v),
                Err(_) => bail!(
                    "could not read a number from '{event_name}': the value '{tok}' in row {}, \
                     column {} is not a valid event {what} time. The layout may mark the wrong \
                     column, or the header-row count may be off.",
                    line_no + 1,
                    c + 1
                ),
            }
        }
    }

    if times.is_empty() {
        bail!(
            "no event times were found in '{event_name}' after skipping {} header row(s).",
            n_header_rows
        );
    }
    Ok(times)
}

/// Text of the default layout file in `config/` (the built-in default if it is missing).
pub fn default_layout_text() -> String {
    read_text_file(&default_layout_path()).unwrap_or_else(|_| DEFAULT_LAYOUT.to_string())
}

/// `None` for text equal to the default, so a recording follows later changes to the
/// default instead of keeping a copy.
pub fn layout_to_save(text: &str) -> Option<String> {
    let norm = |s: &str| s.replace("\r\n", "\n").trim_end().to_string();
    (norm(text) != norm(&default_layout_text())).then(|| text.to_string())
}

// ---------------------------------------------------------------------------
// PSTH computation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PsthParams {
    pub start_ms: f64,
    pub end_ms: f64,
}

pub struct PsthResult {
    /// the display rows (Data + Gap/Boundary) the PSTH was computed for; Data rows'
    /// `data_idx` indexes `data`
    pub display_rows: Vec<DisplayRow>,
    pub n_win: usize,  // number of time samples per row
    pub data: Vec<f32>, // n_rows * n_win, row-major (µV, averaged over events)
    pub avg_trace: Vec<f32>, // n_win, mean across the Data rows
    pub start_ms: f64,
    pub dt_ms: f64,
    pub n_used: usize,
    pub n_skipped: usize,
    /// sorted |data| values, for percentile-based color scaling
    pub abs_sorted: Vec<f32>,
}

impl PsthResult {
    /// vmax for a given percentile (95..=100) of |averaged data|.
    pub fn vmax_percentile(&self, pct: f32) -> f32 {
        if self.abs_sorted.is_empty() {
            return 1.0;
        }
        let f = (pct / 100.0).clamp(0.0, 1.0);
        let idx = ((self.abs_sorted.len() as f32 - 1.0) * f).round() as usize;
        self.abs_sorted[idx.min(self.abs_sorted.len() - 1)].max(1e-6)
    }
}

/// Compute the peri-stimulus average of the preprocessed signal. For each event,
/// a window (plus filter-settle padding) is read from disk, depth-averaged and
/// preprocessed exactly as the main view, then the aligned segment is accumulated.
/// Without a spatial filter every step is linear, so the raw windows are averaged
/// first and the average is preprocessed once, which gives the same result.
/// Events whose full padded window falls outside the recording are skipped.
/// `progress_total` is set to the number of events used once known, and `progress`
/// counts them as they are processed.
pub fn compute_psth(
    raw: &Arc<RawData>,
    meta: &Meta,
    cfg: &PreprocConfig,
    event_times_s: &[f64],
    params: &PsthParams,
    cancel: &AtomicBool,
    progress: &AtomicUsize,
    progress_total: &AtomicUsize,
) -> Result<PsthResult> {
    let fs = meta.sample_rate;
    if params.end_ms <= params.start_ms {
        bail!("PSTH window end ({} ms) must be greater than start ({} ms).", params.end_ms, params.start_ms);
    }

    let w_start = (params.start_ms / 1000.0 * fs).round() as i64;
    let w_end = (params.end_ms / 1000.0 * fs).round() as i64;
    let n_win = (w_end - w_start) as usize;
    if n_win == 0 {
        bail!("PSTH window is too short to contain a single sample at {} Hz.", fs);
    }
    // settle margin on both sides, as long as the enabled filters need: the 300 Hz
    // highpass decays below 1e-4 within ~10 ms, destripe's AGC also averages over
    // ±0.05 s (destripe always includes the highpass). DC removal without the highpass
    // keeps 0.15 s, as its baseline is the mean of the read chunk; with the highpass the
    // DC step has no effect (the filter removes any constant exactly).
    let pad_s: f64 = if cfg.spatial_filter == SpatialFilter::Destripe {
        0.08
    } else if cfg.highpass {
        0.02
    } else if cfg.dc_removal {
        0.15
    } else {
        0.0
    };
    // narrow notches ring for longer than any of the above
    let pad_s = pad_s.max(crate::notch::settle_s(cfg));
    let pad = (pad_s * fs).round() as i64;

    let display_rows = Arc::new(meta.build_display_rows(cfg.avg_depths, &cfg.removed_channels, cfg.channel_order, cfg.shank_order));
    let n_rows = display_rows.iter().filter(|r| matches!(r, DisplayRow::Data { .. })).count();
    if n_rows == 0 {
        bail!("no channels left to average (all channels are removed).");
    }

    // events whose full padded window fits inside the recording
    let n_samples = meta.n_samples as i64;
    let valid: Vec<i64> = event_times_s
        .iter()
        .map(|t| (t * fs).round() as i64)
        .filter(|&onset| {
            let read_first = onset + w_start - pad;
            read_first >= 0 && read_first + n_win as i64 + 2 * pad <= n_samples
        })
        .collect();
    let n_used = valid.len();
    let n_skipped = event_times_s.len() - n_used;
    if n_used == 0 {
        bail!("all {} events fall too close to the recording edges for the chosen window.", event_times_s.len());
    }
    progress_total.store(n_used, Ordering::Relaxed);

    let filt = Filters::new(cfg);
    let n_threads = compute_thread_count();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| anyhow::anyhow!("thread pool: {e}"))?;

    let pad = pad as usize;
    let read_n = n_win + 2 * pad;
    // median CMR and destripe's AGC are not linear, so they need per-event preprocessing
    let linear = cfg.spatial_filter == SpatialFilter::Off;
    // linear: accumulate the whole padded window; otherwise only the aligned segment
    let (seg_off, seg_n) = if linear { (0, read_n) } else { (pad, n_win) };
    let acc_len = n_rows * seg_n;
    // one accumulator per chunk of events, so memory stays bounded by the thread count
    let chunk = n_used.div_ceil(n_threads).max(1);

    let sum = pool.install(|| {
        use rayon::prelude::*;
        valid
            .par_chunks(chunk)
            .map(|onsets| {
                let mut acc = vec![0.0f64; acc_len];
                for &onset in onsets {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    let read_first = (onset + w_start) as usize - pad;
                    let mut data = raw.read_rows(read_first, read_n, meta, &display_rows, cfg.phase_shift);
                    if !linear {
                        preprocess(&mut data, read_n, cfg, &filt, cancel, &display_rows, None);
                    }
                    debug_assert_eq!(data.len(), n_rows * read_n);
                    for r in 0..n_rows {
                        let src = &data[r * read_n + seg_off..r * read_n + seg_off + seg_n];
                        for (d, &v) in acc[r * seg_n..(r + 1) * seg_n].iter_mut().zip(src) {
                            *d += v as f64;
                        }
                    }
                    progress.fetch_add(1, Ordering::Relaxed);
                }
                acc
            })
            .reduce(
                || vec![0.0f64; acc_len],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x += *y;
                    }
                    a
                },
            )
    });

    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }

    let inv = 1.0 / n_used as f64;
    let mut mean: Vec<f32> = sum.iter().map(|&s| (s * inv) as f32).collect();
    let data: Vec<f32> = if linear {
        preprocess(&mut mean, read_n, cfg, &filt, cancel, &display_rows, None);
        (0..n_rows)
            .flat_map(|r| mean[r * read_n + pad..r * read_n + pad + n_win].iter().copied())
            .collect()
    } else {
        mean
    };

    // average across rows at each time sample
    let mut avg_trace = vec![0.0f32; n_win];
    for r in 0..n_rows {
        let row = &data[r * n_win..(r + 1) * n_win];
        for t in 0..n_win {
            avg_trace[t] += row[t];
        }
    }
    let rinv = 1.0 / n_rows as f32;
    for v in &mut avg_trace {
        *v *= rinv;
    }

    let mut abs_sorted: Vec<f32> = data.iter().map(|v| v.abs()).collect();
    abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Ok(PsthResult {
        display_rows: (*display_rows).clone(),
        n_win,
        data,
        avg_trace,
        start_ms: params.start_ms,
        dt_ms: 1000.0 / fs,
        n_used,
        n_skipped,
        abs_sorted,
    })
}

// ---------------------------------------------------------------------------
// Config paths
// ---------------------------------------------------------------------------

/// `config/` directory next to the executable.
pub fn config_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            return dir.join("config");
        }
    }
    PathBuf::from("config")
}

pub fn default_layout_path() -> PathBuf {
    config_dir().join("events_file_layout.csv")
}

const DEFAULT_LAYOUT: &str = "\
# This file tells NPXplorer how to read an event-times file.
# Lines here mirror the structure of that file, one line each (comment lines
# like this one are ignored and don't count). Lines with no 'o' are header
# rows in the event file, to be skipped. The first line containing 'o'
# marks which comma-separated column(s) hold the onset times (in seconds),
# 'f' the offset times (optional, used by the Events window); 'x' marks a column to ignore.
header
o
";

/// Write the default layout file into `config/` if it does not exist yet.
pub fn ensure_default_layout() {
    let path = default_layout_path();
    if !path.is_file() {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&path, DEFAULT_LAYOUT);
    }
}

// ---------------------------------------------------------------------------
// Minimal PNG writer (8-bit RGBA), using flate2 for the zlib IDAT stream so we
// don't pull in an image-encoding dependency.
// ---------------------------------------------------------------------------

pub fn save_png(path: &Path, width: usize, height: usize, rgba: &[u8]) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("cannot export an empty image");
    }
    if rgba.len() < width * height * 4 {
        bail!("pixel buffer too small for {width}x{height} RGBA image");
    }

    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, RGBA, deflate, no filter, no interlace
    write_chunk(&mut out, b"IHDR", &ihdr);

    // filter each scanline with filter type 0 (None)
    let mut raw = Vec::with_capacity(height * (1 + width * 4));
    for y in 0..height {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * width * 4..(y + 1) * width * 4]);
    }
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(&raw)?;
    let compressed = enc.finish()?;
    write_chunk(&mut out, b"IDAT", &compressed);
    write_chunk(&mut out, b"IEND", &[]);

    std::fs::write(path, out)
        .map_err(|e| anyhow::anyhow!("could not write {}: {e}", path.display()))?;
    Ok(())
}

fn write_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(tag);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}
