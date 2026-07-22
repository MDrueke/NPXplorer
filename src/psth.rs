use anyhow::{Result, bail};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::data::{DisplayRow, Meta, RawData};
use crate::preprocess::{Filters, PreprocConfig, preprocess};
use crate::worker::{average_depth_rows, compute_thread_count};

// ---------------------------------------------------------------------------
// Encoding-robust text reading
// ---------------------------------------------------------------------------

/// Read a text file without assuming UTF-8. Handles UTF-8 (with/without BOM) and
/// UTF-16 LE/BE (with BOM); anything else that isn't valid UTF-8 is decoded as
/// Latin-1 (ISO-8859-1), which maps every byte to a char and so never fails. This
/// keeps stimulus files exported from Excel/MATLAB/Python on any platform readable
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

/// Describes where stimulus onset times live in a stimulus file.
/// Determined by a layout file whose lines mirror the stim file's structure:
/// leading lines with no `o` token are header rows to skip; the first line that
/// contains one or more `o` tokens marks which column(s) hold the onset times.
/// Trailing `x` markers beyond the actual number of columns are ignored.
#[derive(Clone, Debug)]
pub struct StimLayout {
    pub n_header_rows: usize,
    pub onset_cols: Vec<usize>,
}

impl StimLayout {
    pub fn parse(text: &str) -> Result<Self> {
        let mut n_header_rows = 0usize;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue; // ignore blank lines in the layout
            }
            let tokens = split_fields(trimmed);
            let onset_cols: Vec<usize> = tokens
                .iter()
                .enumerate()
                .filter(|(_, t)| t.trim().eq_ignore_ascii_case("o"))
                .map(|(i, _)| i)
                .collect();
            if onset_cols.is_empty() {
                // a header / ignored line
                n_header_rows += 1;
            } else {
                return Ok(StimLayout { n_header_rows, onset_cols });
            }
        }
        bail!(
            "the layout file contains no 'o' marker, so it does not say which column \
             holds the stimulus onset times. Mark the onset column with 'o' (e.g. 'o,x,x')."
        );
    }
}

// ---------------------------------------------------------------------------
// Loading stimulus times
// ---------------------------------------------------------------------------

/// Read stimulus onset times (seconds) from `stim_path`, using `layout` to locate
/// the onset column and skip header rows. Errors describe exactly how the layout
/// disagrees with the file rather than panicking.
pub fn load_stim_times(stim_path: &Path, layout: &StimLayout) -> Result<Vec<f64>> {
    let text = read_text_file(stim_path)?;
    let stim_name = stim_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();

    let onset_col = *layout.onset_cols.iter().max().unwrap_or(&0);
    let mut times = Vec::new();

    // human-facing row numbers count every line so they match a text editor
    for (line_no, line) in text.lines().enumerate().skip(layout.n_header_rows) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue; // tolerate blank/trailing lines in the data
        }
        let fields = split_fields(trimmed);
        if onset_col >= fields.len() {
            bail!(
                "layout does not match '{stim_name}': the layout marks column {} as the \
                 stimulus onset time, but row {} has only {} column(s). Check the number of \
                 header rows and the onset column in the layout file.",
                onset_col + 1,
                line_no + 1,
                fields.len()
            );
        }
        for &c in &layout.onset_cols {
            let tok = fields[c];
            match tok.parse::<f64>() {
                Ok(v) => times.push(v),
                Err(_) => bail!(
                    "could not read a number from '{stim_name}': the value '{tok}' in row {}, \
                     column {} is not a valid stimulus onset time. The layout may mark the wrong \
                     column, or the header-row count may be off.",
                    line_no + 1,
                    c + 1
                ),
            }
        }
    }

    if times.is_empty() {
        bail!(
            "no stimulus times were found in '{stim_name}' after skipping {} header row(s).",
            layout.n_header_rows
        );
    }
    Ok(times)
}

/// Resolve the layout for a chosen stim file: a `stims_file_layout.csv` sitting next
/// to the stim file takes precedence, otherwise fall back to the default in `config/`.
pub fn resolve_layout(stim_path: &Path, default_layout_path: &Path) -> Result<StimLayout> {
    let sidecar = stim_path
        .parent()
        .map(|d| d.join("stims_file_layout.csv"))
        .filter(|p| p.is_file());
    let layout_path = sidecar.as_deref().unwrap_or(default_layout_path);
    if !layout_path.is_file() {
        bail!(
            "no layout file found: expected 'stims_file_layout.csv' next to the stimulus file \
             or a default at {}.",
            default_layout_path.display()
        );
    }
    let text = read_text_file(layout_path)?;
    StimLayout::parse(&text)
}

// ---------------------------------------------------------------------------
// PSTH computation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PsthParams {
    pub ch_first: usize,
    pub ch_last: usize,
    pub start_ms: f64,
    pub end_ms: f64,
}

pub struct PsthResult {
    /// selected display rows (Data + Gap/Boundary), Data rows remapped so `data_idx`
    /// is 0-based within this subset and indexes `data`
    pub display_rows: Vec<DisplayRow>,
    pub n_win: usize,  // number of time samples per row
    pub data: Vec<f32>, // n_rows * n_win, row-major (µV, averaged over stimuli)
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

/// Compute the peri-stimulus average of the preprocessed signal. For each stimulus,
/// a window (plus filter-settle padding) is read from disk, depth-averaged and
/// preprocessed exactly as the main view, then the aligned segment is accumulated.
/// Stimuli whose full padded window falls outside the recording are skipped.
pub fn compute_psth(
    raw: &Arc<RawData>,
    meta: &Meta,
    cfg: &PreprocConfig,
    stim_times_s: &[f64],
    params: &PsthParams,
    cancel: &AtomicBool,
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
    // zero-phase (filtfilt) highpass and the destripe AGC need settle margin on both
    // sides; 0.15 s is comfortably longer than either transient
    let pad = (0.15 * fs).round() as i64;

    let display_rows_full = Arc::new(meta.build_display_rows(cfg.avg_depths));
    let data_rows: Vec<usize> = display_rows_full
        .iter()
        .enumerate()
        .filter(|(_, r)| matches!(r, DisplayRow::Data { .. }))
        .map(|(i, _)| i)
        .collect();
    let n_data_rows_full = data_rows.len();

    // display-row index range covering the selected channels (same rule as the main view)
    let (first_disp, last_disp) = select_display_range(&display_rows_full, params.ch_first, params.ch_last);

    // map to the contiguous data_idx range within the full buffer, and build the
    // selected display-row subset with remapped data_idx
    let mut sel_display_rows = Vec::new();
    let mut sel_data_idxs = Vec::new(); // full-buffer data_idx for each selected Data row
    for row in &display_rows_full[first_disp..=last_disp] {
        match row {
            DisplayRow::Data { data_idx, channels, first_ch, x_um, y_um, shank } => {
                let new_idx = sel_data_idxs.len();
                sel_data_idxs.push(*data_idx);
                sel_display_rows.push(DisplayRow::Data {
                    data_idx: new_idx,
                    channels: channels.clone(),
                    first_ch: *first_ch,
                    x_um: *x_um,
                    y_um: *y_um,
                    shank: *shank,
                });
            }
            other => sel_display_rows.push(other.clone()),
        }
    }
    let n_rows = sel_data_idxs.len();
    if n_rows == 0 {
        bail!("the selected channel range contains no channels.");
    }

    // stimuli whose full padded window fits inside the recording
    let n_samples = meta.n_samples as i64;
    let valid: Vec<i64> = stim_times_s
        .iter()
        .map(|t| (t * fs).round() as i64)
        .filter(|&onset| {
            let read_first = onset + w_start - pad;
            read_first >= 0 && read_first + n_win as i64 + 2 * pad <= n_samples
        })
        .collect();
    let n_used = valid.len();
    let n_skipped = stim_times_s.len() - n_used;
    if n_used == 0 {
        bail!("all {} stimuli fall too close to the recording edges for the chosen window.", stim_times_s.len());
    }

    let filt = Filters::new(cfg);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(compute_thread_count())
        .build()
        .map_err(|e| anyhow::anyhow!("thread pool: {e}"))?;

    let sum = pool.install(|| {
        use rayon::prelude::*;
        valid
            .par_iter()
            .fold(
                || vec![0.0f64; n_rows * n_win],
                |mut acc, &onset| {
                    if cancel.load(Ordering::Relaxed) {
                        return acc;
                    }
                    let read_first = (onset + w_start - pad) as usize;
                    let read_n = n_win + 2 * pad as usize;
                    let raw_chunk = raw.read_chunk_uv(read_first, read_n, meta);
                    let mut data = average_depth_rows(&raw_chunk, read_n, &display_rows_full);
                    preprocess(&mut data, read_n, cfg, &filt, cancel, &display_rows_full);
                    debug_assert_eq!(data.len(), n_data_rows_full * read_n);
                    // extract the aligned [pad .. pad+n_win] segment for the selected rows
                    for (r, &full_idx) in sel_data_idxs.iter().enumerate() {
                        let src = full_idx * read_n + pad as usize;
                        let dst = r * n_win;
                        let src_row = &data[src..src + n_win];
                        let dst_row = &mut acc[dst..dst + n_win];
                        for t in 0..n_win {
                            dst_row[t] += src_row[t] as f64;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0f64; n_rows * n_win],
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
    let data: Vec<f32> = sum.iter().map(|&s| (s * inv) as f32).collect();

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
        display_rows: sel_display_rows,
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

/// display-row index range (inclusive) covering channels [ch_first, ch_last],
/// mirroring `NPXplorerApp::visible_row_range`.
fn select_display_range(display_rows: &[DisplayRow], ch_first: usize, ch_last: usize) -> (usize, usize) {
    let mut first_idx = 0usize;
    let mut last_idx = display_rows.len().saturating_sub(1);
    let mut found_first = false;
    for (i, row) in display_rows.iter().enumerate() {
        if let DisplayRow::Data { first_ch, .. } = row {
            if !found_first && *first_ch >= ch_first {
                first_idx = i;
                found_first = true;
            }
            if *first_ch <= ch_last {
                last_idx = i;
            }
        }
    }
    if last_idx < first_idx {
        last_idx = first_idx;
    }
    (first_idx, last_idx)
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
    config_dir().join("stims_file_layout.csv")
}

const DEFAULT_LAYOUT: &str = "header\no\n";

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
