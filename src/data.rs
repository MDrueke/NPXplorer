use anyhow::{Context, Result, bail};
use std::path::Path;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct ChannelGeom {
    pub x_um: f32,
    pub y_um: f32,
    pub shank: u32,
}

/// One row in the display (after depth-averaging).
/// `data_idx` is the row index into the PreprocBuffer data array (None for gaps).
#[derive(Clone, Debug)]
pub enum DisplayRow {
    Data { data_idx: usize, channels: Vec<usize>, first_ch: usize, x_um: f32, y_um: f32 },
    Gap,
}

#[derive(Clone, Debug)]
pub struct Meta {
    pub n_saved_chans: usize,
    pub n_ap_chans: usize,
    pub sample_rate: f64,
    pub n_samples: usize,
    pub uv_per_bit: f32,
    pub im_dat_prb_type: u32,
    pub channel_geom: Vec<ChannelGeom>,
}

impl Meta {
    pub fn from_file(meta_path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(meta_path)
            .with_context(|| format!("reading meta file: {}", meta_path.display()))?;

        let mut n_saved_chans: Option<usize> = None;
        let mut sample_rate: Option<f64> = None;
        let mut file_size_bytes: Option<u64> = None;
        let mut ai_range_max: f64 = 0.6;
        let mut ap_gain: f64 = 500.0;
        let mut max_int: f64 = 512.0;
        let mut n_ap: Option<usize> = None;
        let mut n_lf: Option<usize> = None;
        let mut n_sy: Option<usize> = None;
        let mut geom_str: Option<String> = None;
        let mut im_dat_prb_type: Option<u32> = None;

        for line in text.lines() {
            let line = line.trim_end_matches('\r');
            if let Some((key, val)) = line.split_once('=') {
                match key {
                    "nSavedChans" => n_saved_chans = val.parse().ok(),
                    "imSampRate" => sample_rate = val.parse().ok(),
                    "fileSizeBytes" => file_size_bytes = val.parse().ok(),
                    "imAiRangeMax" => ai_range_max = val.parse().unwrap_or(0.6),
                    "imChan0apGain" => ap_gain = val.parse().unwrap_or(500.0),
                    "imMaxInt" => max_int = val.parse().unwrap_or(512.0),
                    "snsApLfSy" => {
                        let parts: Vec<&str> = val.split(',').collect();
                        if parts.len() >= 3 {
                            n_ap = parts[0].parse().ok();
                            n_lf = parts[1].parse().ok();
                            n_sy = parts[2].parse().ok();
                        }
                    }
                    "imDatPrb_type" => im_dat_prb_type = val.parse().ok(),
                    // both ~snsGeomMap (new) and snsGeomMap (no tilde) variants
                    k if k == "~snsGeomMap" || k == "snsGeomMap" => {
                        geom_str = Some(val.to_string());
                    }
                    _ => {}
                }
            }
        }

        let n_saved_chans = n_saved_chans.context("missing nSavedChans")?;
        let sample_rate = sample_rate.context("missing imSampRate")?;
        let file_size_bytes = file_size_bytes.context("missing fileSizeBytes")?;

        let n_ap = n_ap.unwrap_or(n_saved_chans.saturating_sub(1));
        let _n_sy = n_sy.unwrap_or(1);
        let _ = n_lf;

        let n_ap_chans = n_ap;
        let n_samples = (file_size_bytes / (n_saved_chans as u64 * 2)) as usize;
        let uv_per_bit = (ai_range_max / max_int / ap_gain * 1e6) as f32;
        let channel_geom = parse_geom_map(geom_str.as_deref(), n_ap_chans);

        Ok(Meta {
            n_saved_chans,
            n_ap_chans,
            sample_rate,
            n_samples,
            uv_per_bit,
            im_dat_prb_type: im_dat_prb_type.unwrap_or(0),
            channel_geom,
        })
    }

    /// Compute the typical vertical pitch (µm) per shank from the geometry.
    /// Returns the minimum positive y-difference between channels on the same shank.
    fn typical_pitch_per_shank(&self) -> HashMap<u32, f32> {
        let mut by_shank: HashMap<u32, Vec<f32>> = HashMap::new();
        for g in &self.channel_geom {
            by_shank.entry(g.shank).or_default().push(g.y_um);
        }
        let mut result = HashMap::new();
        for (shank, mut ys) in by_shank {
            ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ys.dedup_by(|a, b| (*a - *b).abs() < 0.1);
            let min_diff = ys.windows(2)
                .filter_map(|w| {
                    let d = w[1] - w[0];
                    if d > 0.1 { Some(d) } else { None }
                })
                .fold(f32::INFINITY, f32::min);
            result.insert(shank, if min_diff.is_finite() { min_diff } else { 20.0 });
        }
        result
    }

    /// Build the ordered list of display rows for rendering.
    ///
    /// If `avg_depths` is true, channels at the same (shank, y_um) are averaged into one row.
    /// Gap rows are inserted wherever the vertical distance between consecutive rows
    /// exceeds 1.5× the typical pitch for that shank.
    pub fn build_display_rows(&self, avg_depths: bool) -> Vec<DisplayRow> {
        let pitch_map = self.typical_pitch_per_shank();

        // collect (shank, y_um, channel_idx) tuples
        let mut entries: Vec<(u32, f32, usize)> = self.channel_geom.iter()
            .enumerate()
            .map(|(i, g)| (g.shank, g.y_um, i))
            .collect();
        // sort by shank, then y ascending
        entries.sort_by(|a, b| {
            a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        // group by (shank, y_um)
        let mut groups: Vec<(u32, f32, Vec<usize>)> = Vec::new();
        for (shank, y, ch) in entries {
            if let Some(last) = groups.last_mut() {
                if last.0 == shank && (last.1 - y).abs() < 0.5 && avg_depths {
                    last.2.push(ch);
                    continue;
                }
            }
            groups.push((shank, y, vec![ch]));
        }

        // if not avg_depths, each channel is its own group (already the case above since we skip merging)
        // Re-sort group channels so first_ch is the smallest index
        for (_, _, chs) in &mut groups {
            chs.sort_unstable();
        }

        // build display rows with gap detection
        let mut rows: Vec<DisplayRow> = Vec::new();
        let mut data_idx = 0usize;
        let mut prev: Option<(u32, f32)> = None; // (shank, y)

        for (shank, y, channels) in &groups {
            let pitch = *pitch_map.get(shank).unwrap_or(&20.0);

            if let Some((prev_shank, prev_y)) = prev {
                let gap = if *shank != prev_shank {
                    // different shank: always insert a gap
                    true
                } else {
                    // same shank: gap if spacing > 1.5× pitch
                    (y - prev_y) > pitch * 1.5
                };
                if gap {
                    rows.push(DisplayRow::Gap);
                }
            }

            let first_ch = *channels.first().unwrap();
            let x_um = self.channel_geom[first_ch].x_um;
            rows.push(DisplayRow::Data {
                data_idx,
                channels: channels.clone(),
                first_ch,
                x_um,
                y_um: *y,
            });
            data_idx += 1;
            prev = Some((*shank, *y));
        }

        rows
    }
}

fn parse_geom_map(s: Option<&str>, n_ap: usize) -> Vec<ChannelGeom> {
    let default = || {
        (0..n_ap)
            .map(|i| ChannelGeom { x_um: 0.0, y_um: i as f32 * 20.0, shank: 0 })
            .collect()
    };

    let s = match s {
        Some(s) => s,
        None => return default(),
    };

    // entries are parenthesised: (shank:x_um:y_um:used)
    let mut geoms: Vec<(usize, ChannelGeom)> = Vec::new();
    let mut ch_idx: usize = 0;
    for token in s.split(')') {
        let token = token.trim_start_matches('(');
        if token.is_empty() {
            continue;
        }
        let parts: Vec<&str> = token.split(':').collect();
        if parts.len() == 4 {
            // shank:x:y:used
            let shank: u32 = parts[0].parse().unwrap_or(0);
            let x: f32 = parts[1].parse().unwrap_or(0.0);
            let y: f32 = parts[2].parse().unwrap_or(0.0);
            geoms.push((ch_idx, ChannelGeom { x_um: x, y_um: y, shank }));
            ch_idx += 1;
        }
        // else: header token like "(NP1000,1,0,70)" — skip
    }

    if geoms.is_empty() {
        return default();
    }

    let mut out = vec![ChannelGeom { x_um: 0.0, y_um: 0.0, shank: 0 }; n_ap];
    for (i, g) in geoms.into_iter().take(n_ap) {
        out[i] = g;
    }
    out
}

// ---------------------------------------------------------------------------
// Raw data access
// ---------------------------------------------------------------------------

pub enum RawData {
    Uncompressed(memmap2::Mmap),
    Compressed(crate::mtscomp::MtscompReader),
}

impl RawData {
    /// Return a flat Vec<f32> in µV, layout: [n_ap][n_samp].
    pub fn read_chunk_uv(
        &self,
        first_sample: usize,
        n_samp: usize,
        meta: &Meta,
    ) -> Vec<f32> {
        let n_ch = meta.n_saved_chans;
        let n_ap = meta.n_ap_chans;
        let scale = meta.uv_per_bit;
        let n_samp = n_samp.min(meta.n_samples.saturating_sub(first_sample));

        match self {
            RawData::Uncompressed(mmap) => {
                let raw: &[i16] = bytemuck::cast_slice(mmap.as_ref());
                let start = (first_sample * n_ch).min(raw.len());
                let end = ((first_sample + n_samp) * n_ch).min(raw.len());
                let src = &raw[start..end];

                let mut out = vec![0.0f32; n_ap * n_samp];
                use rayon::prelude::*;
                out.par_chunks_mut(n_samp)
                    .enumerate()
                    .for_each(|(ch, row)| {
                        for t in 0..n_samp {
                            let idx = t * n_ch + ch;
                            row[t] = if idx < src.len() {
                                src[idx] as f32 * scale
                            } else {
                                0.0
                            };
                        }
                    });
                out
            }
            RawData::Compressed(reader) => {
                // Find overlapping chunks
                let mut out = vec![0.0f32; n_ap * n_samp];
                let end_sample = first_sample + n_samp;
                
                // Identify which chunks we need
                let chunk_bounds = &reader.meta.chunk_bounds;
                let mut start_idx = 0;
                while start_idx < chunk_bounds.len() - 1 && chunk_bounds[start_idx + 1] <= first_sample {
                    start_idx += 1;
                }
                
                let mut current_idx = start_idx;
                let mut out_offset = 0;

                while current_idx < chunk_bounds.len() - 1 && out_offset < n_samp {
                    let chunk_start = chunk_bounds[current_idx];
                    let chunk_end = chunk_bounds[current_idx + 1];

                    // Compute overlap
                    let overlap_start = chunk_start.max(first_sample);
                    let overlap_end = chunk_end.min(end_sample);
                    
                    if overlap_start < overlap_end {
                        let overlap_len = overlap_end - overlap_start;
                        let src_offset = overlap_start - chunk_start;

                        if let Ok(decompressed) = reader.decompress_chunk(current_idx) {
                            // Decompressed is in C-order: [time * n_channels + ch]
                            use rayon::prelude::*;
                            // Better: process channel by channel
                            out.par_chunks_mut(n_samp).enumerate().for_each(|(ch, row_dst)| {
                                for t in 0..overlap_len {
                                    let src_idx = (src_offset + t) * n_ch + ch;
                                    if src_idx < decompressed.len() {
                                        row_dst[out_offset + t] = decompressed[src_idx] as f32 * scale;
                                    }
                                }
                            });
                        }
                        out_offset += overlap_len;
                    }
                    current_idx += 1;
                }

                out
            }
        }
    }
}

pub fn open_data(bin_path: &Path, meta: &Meta) -> Result<(RawData, usize)> {
    if bin_path.extension().and_then(|s| s.to_str()) == Some("cbin") {
        let ch_path = bin_path.with_extension("ch");
        if !ch_path.exists() {
            bail!("Metadata file {} not found for {}", ch_path.display(), bin_path.display());
        }
        let mts_meta = crate::mtscomp::MtscompMeta::from_file(&ch_path)?;
        let reader = crate::mtscomp::MtscompReader::new(bin_path, mts_meta)?;
        Ok((RawData::Compressed(reader), meta.n_samples))
    } else {
        let file = std::fs::File::open(bin_path)
            .with_context(|| format!("opening {}", bin_path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // verify alignment
        if mmap.as_ptr() as usize % 2 != 0 {
            bail!("mmap pointer is not 2-byte aligned");
        }
        Ok((RawData::Uncompressed(mmap), meta.n_samples))
    }
}
