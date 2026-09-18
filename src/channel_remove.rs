use anyhow::{Result, bail};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::psth::read_text_file;

/// Split a data line into fields, same convention as the PSTH stim-file reader.
fn split_fields(line: &str) -> Vec<&str> {
    if line.contains(',') {
        line.split(',').map(|f| f.trim()).collect()
    } else {
        line.split_whitespace().collect()
    }
}

/// A single channel number or an inclusive range ("40-50"), 1-based to match the UI.
/// Returns 0-based channel indices.
fn parse_token(tok: &str) -> Result<Vec<usize>> {
    let tok = tok.trim();
    if let Some((lo, hi)) = tok.split_once('-') {
        let lo: usize = lo.trim().parse()
            .map_err(|_| anyhow::anyhow!("'{tok}' is not a valid channel range"))?;
        let hi: usize = hi.trim().parse()
            .map_err(|_| anyhow::anyhow!("'{tok}' is not a valid channel range"))?;
        if lo == 0 || hi < lo {
            bail!("'{tok}' is not a valid channel range (channels are numbered from 1)");
        }
        Ok((lo..=hi).map(|c| c - 1).collect())
    } else {
        let c: usize = tok.parse().map_err(|_| anyhow::anyhow!("'{tok}' is not a valid channel number"))?;
        if c == 0 {
            bail!("channel numbers start at 1");
        }
        Ok(vec![c - 1])
    }
}

/// Parse a manually-entered list like "3,17,40-50" into 0-based channel indices.
pub fn parse_channel_list(text: &str) -> Result<BTreeSet<usize>> {
    let mut out = BTreeSet::new();
    for tok in text.split(|c: char| c == ',' || c.is_whitespace()).filter(|t| !t.is_empty()) {
        for ch in parse_token(tok)? {
            out.insert(ch);
        }
    }
    Ok(out)
}

/// Render a set of 0-based channel indices back into a compact 1-based list,
/// collapsing consecutive runs into ranges (e.g. "3,17,40-50").
pub fn format_channel_list(channels: &BTreeSet<usize>) -> String {
    let mut parts = Vec::new();
    let mut iter = channels.iter().copied().peekable();
    while let Some(start) = iter.next() {
        let mut end = start;
        while iter.peek() == Some(&(end + 1)) {
            end = iter.next().unwrap();
        }
        if end == start {
            parts.push(format!("{}", start + 1));
        } else {
            parts.push(format!("{}-{}", start + 1, end + 1));
        }
    }
    parts.join(",")
}

// ---------------------------------------------------------------------------
// Layout file — same convention as PSTH's stims_file_layout.csv, but the
// marker letter is 'c' (for "channel") instead of 'o'.
// ---------------------------------------------------------------------------

/// Describes where channel numbers to remove live in a list file. Same convention
/// as PSTH's stims_file_layout.csv: leading lines with no `o` token are header rows
/// to skip; the first line containing one or more `o` tokens marks which column(s)
/// hold channel numbers (or ranges).
#[derive(Clone, Debug)]
pub struct ChannelRemoveLayout {
    pub n_header_rows: usize,
    pub channel_cols: Vec<usize>,
}

impl ChannelRemoveLayout {
    pub fn parse(text: &str) -> Result<Self> {
        let mut n_header_rows = 0usize;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue; // ignore blank lines and comments in the layout
            }
            let tokens = split_fields(trimmed);
            let channel_cols: Vec<usize> = tokens
                .iter()
                .enumerate()
                .filter(|(_, t)| t.trim().eq_ignore_ascii_case("o"))
                .map(|(i, _)| i)
                .collect();
            if channel_cols.is_empty() {
                n_header_rows += 1;
            } else {
                return Ok(ChannelRemoveLayout { n_header_rows, channel_cols });
            }
        }
        bail!(
            "the layout file contains no 'o' marker, so it does not say which column \
             holds the channel numbers. Mark the channel column with 'o' (e.g. 'o,x,x')."
        );
    }
}

/// Read channel numbers/ranges from `list_path`, using `layout` to locate the marked
/// column(s) and skip header rows. Returns 0-based channel indices.
pub fn load_removed_channels(list_path: &Path, layout: &ChannelRemoveLayout) -> Result<BTreeSet<usize>> {
    let text = read_text_file(list_path)?;
    let name = list_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();

    let mut out = BTreeSet::new();
    for (line_no, line) in text.lines().enumerate().skip(layout.n_header_rows) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields = split_fields(trimmed);
        for &c in &layout.channel_cols {
            if c >= fields.len() {
                bail!(
                    "layout does not match '{name}': the layout marks column {} for channel \
                     numbers, but row {} has only {} column(s). Check the number of header \
                     rows and the channel column in the layout file.",
                    c + 1,
                    line_no + 1,
                    fields.len()
                );
            }
            let tok = fields[c];
            match parse_token(tok) {
                Ok(chs) => out.extend(chs),
                Err(_) => bail!(
                    "could not read a channel number from '{name}': the value '{tok}' in row {}, \
                     column {} is not a valid channel number or range.",
                    line_no + 1,
                    c + 1
                ),
            }
        }
    }

    if out.is_empty() {
        bail!("no channel numbers were found in '{name}' after skipping {} header row(s).", layout.n_header_rows);
    }
    Ok(out)
}

/// Resolve the layout for a chosen channel-list file: a `channel_remove_layout.csv`
/// sitting next to it takes precedence, otherwise fall back to the default in `config/`.
pub fn resolve_layout(list_path: &Path, default_layout_path: &Path) -> Result<ChannelRemoveLayout> {
    let sidecar = list_path
        .parent()
        .map(|d| d.join("channel_remove_layout.csv"))
        .filter(|p| p.is_file());
    let layout_path = sidecar.as_deref().unwrap_or(default_layout_path);
    if !layout_path.is_file() {
        bail!(
            "no layout file found: expected 'channel_remove_layout.csv' next to the channel \
             list file or a default at {}.",
            default_layout_path.display()
        );
    }
    let text = read_text_file(layout_path)?;
    ChannelRemoveLayout::parse(&text)
}

pub fn default_layout_path() -> PathBuf {
    crate::psth::config_dir().join("channel_remove_layout.csv")
}

const DEFAULT_LAYOUT: &str = "\
# This file tells NPXplorer how to read a channel-numbers-to-remove file.
# Lines here mirror the structure of that file, one line each (comment lines
# like this one are ignored and don't count). Lines with no 'o' are header
# rows in the channel-list file, to be skipped. The first line containing 'o'
# marks which comma-separated column(s) hold the channel numbers (e.g. '5' or
# a range like '40-50'); 'x' marks a column to ignore.
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
