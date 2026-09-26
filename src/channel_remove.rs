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

/// Split a channel ID into its text prefix and trailing number, e.g. "AP12" ->
/// ("AP", Some(12)); IDs without a trailing number give None.
fn split_id(id: &str) -> (&str, Option<u64>) {
    let digits = id.len() - id.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    let (prefix, num) = id.split_at(id.len() - digits);
    (prefix, num.parse().ok())
}

/// Resolve one entry to 0-based channel indices, matching against the channel IDs
/// from the meta file (`ids`, one per channel in file order):
/// - an exact ID ("AP12", case-insensitive) selects that channel;
/// - a bare number ("12") selects the channel(s) whose ID ends in that number;
/// - a range ("AP40-AP50" or "40-50") selects every channel whose ID number lies in
///   it (restricted to the given prefix, if any).
fn parse_token(tok: &str, ids: &[String]) -> Result<Vec<usize>> {
    let tok = tok.trim();
    if let Some(i) = ids.iter().position(|id| id.eq_ignore_ascii_case(tok)) {
        return Ok(vec![i]);
    }
    let bound = |s: &str| -> Result<(String, u64)> {
        let (prefix, num) = split_id(s.trim());
        let num = num.ok_or_else(|| anyhow::anyhow!("'{tok}' is not a channel ID or range"))?;
        Ok((prefix.to_ascii_lowercase(), num))
    };
    let ((p_lo, lo), (p_hi, hi)) = match tok.split_once('-') {
        Some((a, b)) => (bound(a)?, bound(b)?),
        None => (bound(tok)?, bound(tok)?),
    };
    if hi < lo {
        bail!("'{tok}' is not a valid channel range");
    }
    let matches: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, id)| {
            let (prefix, num) = split_id(id);
            let prefix = prefix.to_ascii_lowercase();
            num.is_some_and(|n| n >= lo && n <= hi)
                && (p_lo.is_empty() || p_lo == prefix)
                && (p_hi.is_empty() || p_hi == prefix)
        })
        .map(|(i, _)| i)
        .collect();
    if matches.is_empty() {
        bail!("'{tok}' does not match any channel ID of this recording");
    }
    Ok(matches)
}

/// Parse a manually-entered list like "AP3,AP17,AP40-AP50" into 0-based channel indices.
pub fn parse_channel_list(text: &str, ids: &[String]) -> Result<BTreeSet<usize>> {
    let mut out = BTreeSet::new();
    for tok in text.split(|c: char| c == ',' || c.is_whitespace()).filter(|t| !t.is_empty()) {
        for ch in parse_token(tok, ids)? {
            out.insert(ch);
        }
    }
    Ok(out)
}

/// Render a set of 0-based channel indices back into a compact list of channel IDs,
/// collapsing runs of consecutive IDs into ranges (e.g. "AP3,AP17,AP40-AP50").
pub fn format_channel_list(channels: &BTreeSet<usize>, ids: &[String]) -> String {
    let id = |i: usize| ids.get(i).map(|s| s.as_str()).unwrap_or("?");
    let follows = |a: usize, b: usize| {
        let ((pa, na), (pb, nb)) = (split_id(id(a)), split_id(id(b)));
        pa == pb && matches!((na, nb), (Some(x), Some(y)) if y == x + 1)
    };
    let mut parts = Vec::new();
    let mut iter = channels.iter().copied().peekable();
    while let Some(start) = iter.next() {
        let mut end = start;
        while let Some(&next) = iter.peek() {
            if !follows(end, next) {
                break;
            }
            end = next;
            iter.next();
        }
        if end == start {
            parts.push(id(start).to_string());
        } else {
            parts.push(format!("{}-{}", id(start), id(end)));
        }
    }
    parts.join(",")
}

/// Example entry text for this recording's IDs, e.g. "AP3,AP17,AP40-AP50".
pub fn example_list(ids: &[String]) -> String {
    let prefix = ids.first().map(|id| split_id(id).0).unwrap_or("");
    format!("{prefix}3,{prefix}17,{prefix}40-{prefix}50")
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

/// Read channel IDs/ranges from `list_path`, using `layout` to locate the marked
/// column(s) and skip header rows. Returns 0-based channel indices.
pub fn load_removed_channels(list_path: &Path, layout: &ChannelRemoveLayout, ids: &[String]) -> Result<BTreeSet<usize>> {
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
            match parse_token(tok, ids) {
                Ok(chs) => out.extend(chs),
                Err(_) => bail!(
                    "could not read a channel from '{name}': the value '{tok}' in row {}, \
                     column {} does not match any channel ID or range of this recording.",
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
# marks which comma-separated column(s) hold the channels, as channel IDs
# (e.g. 'AP5', a bare number like '5', or a range like 'AP40-AP50'); 'x' marks
# a column to ignore.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(prefix: &str, n: usize) -> Vec<String> {
        (0..n).map(|i| format!("{prefix}{i}")).collect()
    }

    #[test]
    fn parse_ids_numbers_and_ranges() {
        let ids = ids("AP", 60);
        let set = parse_channel_list("AP3, 17 ap40-AP42,50-51", &ids).unwrap();
        assert_eq!(set.into_iter().collect::<Vec<_>>(), vec![3, 17, 40, 41, 42, 50, 51]);
        assert!(parse_channel_list("LF3", &ids).is_err());
        assert!(parse_channel_list("AP99", &ids).is_err());
    }

    #[test]
    fn format_round_trip() {
        let ids = ids("AP", 60);
        let set: BTreeSet<usize> = [3, 17, 40, 41, 42].into_iter().collect();
        let text = format_channel_list(&set, &ids);
        assert_eq!(text, "AP3,AP17,AP40-AP42");
        assert_eq!(parse_channel_list(&text, &ids).unwrap(), set);
    }

    #[test]
    fn subset_saved_channels() {
        // file holds only some hardware channels: IDs, not positions, are matched
        let ids: Vec<String> = ["AP10", "AP11", "AP20"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_channel_list("20", &ids).unwrap().into_iter().collect::<Vec<_>>(), vec![2]);
        let all: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
        assert_eq!(format_channel_list(&all, &ids), "AP10-AP11,AP20");
    }
}
