use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::probe::{self, MuxFamily, ProbeSpec};

/// AP-band sample rate the ADC cycle timing is defined against.
const AP_RATE_HZ: f64 = 30_000.0;

/// Factor converting a delay in AP sample periods into samples of a stream at `fs`:
/// 1 for an AP / full-band stream (whatever its calibrated rate, e.g. 29999.84 Hz),
/// fs / 30 kHz for an LF stream.
fn stream_delay_scale(fs: f64) -> f32 {
    if fs > 10_000.0 { 1.0 } else { (fs / AP_RATE_HZ) as f32 }
}

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
    Data { data_idx: usize, channels: Vec<usize>, first_ch: usize, x_um: f32, y_um: f32, shank: u32 },
    IntraShankGap,
    ShankBoundary,
}

/// Rows are considered physically separated (a gap is drawn, and spatial filters stop)
/// when consecutive rows on a shank are further apart than this many pitches.
pub const GAP_PITCH_FACTOR: f32 = 1.5;

/// How channels are ordered top-to-bottom within a shank in the display.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum ChannelOrder {
    /// Raw hardware channel number, ascending from bottom to top.
    Id,
    /// Physical depth (y_um), ascending from bottom to top — the deepest channel
    /// on each shank is at the bottom.
    Depth,
}

impl Default for ChannelOrder {
    fn default() -> Self {
        ChannelOrder::Depth
    }
}

/// How shanks are ordered left-to-right in the display.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum ShankOrder {
    /// Raw shank index from the meta file, ascending.
    Id,
    /// Physical x position (mean over the shank's electrodes), ascending — left to right.
    XCoord,
}

impl Default for ShankOrder {
    fn default() -> Self {
        ShankOrder::Id
    }
}

#[derive(Clone, Debug)]
pub struct Meta {
    pub n_saved_chans: usize,
    pub n_ap_chans: usize,
    pub sample_rate: f64,
    pub n_samples: usize,
    /// µV per raw integer, one per signal channel in file order
    pub uv_per_bit: Vec<f32>,
    pub im_dat_prb_type: u32,
    pub channel_geom: Vec<ChannelGeom>,
    /// channel identifier per saved signal channel, as named by the acquisition
    /// software (SpikeGLX `~snsChanMap`, e.g. "AP12"; Open Ephys `channel_name`, e.g.
    /// "CH13") — shown everywhere a channel is identified, and used in exported files
    pub channel_ids: Vec<String>,
    /// ADC sampling delay per signal channel, in samples of this stream (0 = sampled
    /// first in its ADC cycle). Corrected by the "Phase Shift" preprocessing step.
    pub sample_shift: Vec<f32>,
    /// values the metadata did not provide and that had to be assumed — shown in the
    /// status bar so the user knows what to trust
    pub warnings: Vec<String>,
    /// channels that carry no neural signal (on-shank reference sites): SpikeGLX marks
    /// them `used=0` in its geometry map, Open Ephys lists no position for them. They
    /// are pre-filled into "Remove channels" when the recording is opened.
    pub reference_channels: BTreeSet<usize>,
}

impl Meta {
    /// Identifier of the channel at 0-based position `idx` in the data file.
    pub fn channel_id(&self, idx: usize) -> &str {
        self.channel_ids.get(idx).map(|s| s.as_str()).unwrap_or("?")
    }

    /// Detect the acquisition format from the data file's location and load metadata
    /// accordingly. SpikeGLX is identified by a sibling `.meta` file; Open Ephys is
    /// identified by a `structure.oebin` found in an ancestor directory (with a
    /// `settings.xml` further up, at the Record Node level).
    pub fn from_data_path(bin_path: &Path) -> Result<Self> {
        // for mtscomp-compressed data the true sample count lives in the .ch file: the
        // compressed file's size says nothing about it
        let cbin_n_samples = if is_cbin(bin_path) {
            let ch_path = bin_path.with_extension("ch");
            let m = crate::mtscomp::MtscompMeta::from_file(&ch_path)?;
            Some(m.chunk_bounds.last().copied().context("empty chunk_bounds in .ch file")?)
        } else {
            None
        };

        let meta_path = bin_path.with_extension("meta");
        if meta_path.is_file() {
            return Self::from_file(&meta_path, cbin_n_samples);
        }
        if let Some((oebin_path, settings_path)) = find_open_ephys_meta(bin_path) {
            return Self::from_open_ephys(bin_path, &oebin_path, &settings_path, cbin_n_samples);
        }
        bail!(
            "no metadata found for {}: expected a sibling .meta file (SpikeGLX) \
             or a structure.oebin in a parent directory (Open Ephys)",
            bin_path.display()
        );
    }

    /// `cbin_n_samples`: sample count from the mtscomp `.ch` file, when the data is
    /// compressed (overrides the size-derived count).
    pub fn from_file(meta_path: &Path, cbin_n_samples: Option<usize>) -> Result<Self> {
        let text = std::fs::read_to_string(meta_path)
            .with_context(|| format!("reading meta file: {}", meta_path.display()))?;

        let mut n_saved_chans: Option<usize> = None;
        let mut sample_rate: Option<f64> = None;
        let mut file_size_bytes: Option<u64> = None;
        let mut ai_range_max: Option<f64> = None;
        let mut chan0_ap_gain: Option<f64> = None;
        let mut chan0_lf_gain: Option<f64> = None;
        let mut max_int: Option<f64> = None;
        let mut n_ap: Option<usize> = None;
        let mut n_lf: Option<usize> = None;
        let mut n_sy: Option<usize> = None;
        let mut geom_str: Option<String> = None;
        let mut shank_map_str: Option<String> = None;
        let mut imro_str: Option<String> = None;
        let mut mux_str: Option<String> = None;
        let mut im_dat_prb_type: Option<u32> = None;
        let mut chan_map_str: Option<String> = None;

        for line in text.lines() {
            let line = line.trim_end_matches('\r');
            if let Some((key, val)) = line.split_once('=') {
                match key.trim_start_matches('~') {
                    "nSavedChans" => n_saved_chans = val.parse().ok(),
                    "imSampRate" => sample_rate = val.parse().ok(),
                    "fileSizeBytes" => file_size_bytes = val.parse().ok(),
                    "imAiRangeMax" => ai_range_max = val.parse().ok(),
                    "imChan0apGain" => chan0_ap_gain = val.parse().ok(),
                    "imChan0lfGain" => chan0_lf_gain = val.parse().ok(),
                    "imMaxInt" => max_int = val.parse().ok(),
                    "snsApLfSy" => {
                        let parts: Vec<&str> = val.split(',').collect();
                        if parts.len() >= 3 {
                            n_ap = parts[0].parse().ok();
                            n_lf = parts[1].parse().ok();
                            n_sy = parts[2].parse().ok();
                        }
                    }
                    "imDatPrb_type" => im_dat_prb_type = val.parse().ok(),
                    "snsGeomMap" => geom_str = Some(val.to_string()),
                    "snsShankMap" => shank_map_str = Some(val.to_string()),
                    "snsChanMap" => chan_map_str = Some(val.to_string()),
                    "imroTbl" => imro_str = Some(val.to_string()),
                    "muxTbl" => mux_str = Some(val.to_string()),
                    _ => {}
                }
            }
        }

        let n_saved_chans = n_saved_chans.context("missing nSavedChans")?;
        let sample_rate = sample_rate.context("missing imSampRate")?;
        let file_size_bytes = file_size_bytes.context("missing fileSizeBytes")?;
        let im_dat_prb_type = im_dat_prb_type.unwrap_or(0);
        let spec = probe::spec_for_type(im_dat_prb_type);
        let mut warnings = Vec::new();

        // snsApLfSy reports counts for both bands, e.g. (384,0,1) in an .ap.meta and
        // (0,384,1) in the sibling .lf.meta — only the band saved in *this* file is
        // nonzero. Sum them to get the number of signal (non-sync) channels present here,
        // falling back to nSavedChans - nSy if the field is missing entirely.
        let n_sy = n_sy.unwrap_or(1);
        let is_lf_band = n_lf.unwrap_or(0) > 0 && n_ap.unwrap_or(0) == 0;
        let n_signal_chans = match (n_ap, n_lf) {
            (Some(a), Some(l)) if a + l > 0 => a + l,
            _ => n_saved_chans.saturating_sub(n_sy),
        };
        let n_ap_chans = n_signal_chans;

        let size_n_samples = (file_size_bytes / (n_saved_chans as u64 * 2)) as usize;
        let n_samples = match cbin_n_samples {
            Some(n) => n.min(size_n_samples.max(1)),
            None => size_n_samples,
        };

        let channel_ids = parse_chan_map(chan_map_str.as_deref(), n_ap_chans);
        // acquisition channel number of each saved channel (the number in its name),
        // which is what the imro and mux tables are indexed by
        let chan_nums: Vec<usize> = channel_ids
            .iter()
            .enumerate()
            .map(|(i, id)| trailing_number(id).unwrap_or(i))
            .collect();

        // --- gain and µV scale ---------------------------------------------------
        let imro = parse_imro_table(imro_str.as_deref());
        let fixed_gain = spec.as_ref().and_then(|s| if is_lf_band { s.fixed_lf_gain } else { s.fixed_ap_gain });
        let chan0_gain = if is_lf_band { chan0_lf_gain } else { chan0_ap_gain };
        let mut gain_assumed = false;
        let gains: Vec<f64> = chan_nums
            .iter()
            .map(|&cn| {
                let from_imro = imro.as_ref().and_then(|t| {
                    t.per_channel.get(&cn).copied().or(t.header).map(|(ap, lf)| if is_lf_band { lf } else { ap })
                });
                from_imro.or(chan0_gain).or(fixed_gain).unwrap_or_else(|| {
                    gain_assumed = true;
                    if is_lf_band { 250.0 } else { 500.0 }
                })
            })
            .collect();
        if gain_assumed {
            warnings.push(format!(
                "gain not in metadata (probe type {im_dat_prb_type} unknown): assuming {} — µV values may be off",
                if is_lf_band { 250 } else { 500 }
            ));
        }
        let max_int = max_int.or_else(|| spec.as_ref().map(|s| (1u64 << (s.adc_bits - 1)) as f64)).unwrap_or_else(|| {
            warnings.push("ADC resolution not in metadata: assuming 10 bits".into());
            512.0
        });
        let ai_range_max = ai_range_max.or_else(|| spec.as_ref().map(|s| s.ai_range_max_v)).unwrap_or_else(|| {
            warnings.push("ADC range not in metadata: assuming ±0.6 V".into());
            0.6
        });
        let uv_per_bit: Vec<f32> = gains.iter().map(|g| (ai_range_max / max_int / g * 1e6) as f32).collect();

        // --- geometry ------------------------------------------------------------
        let nominal = || (default_geom(n_ap_chans), BTreeSet::new());
        let (channel_geom, reference_channels) = match parse_geom_map(geom_str.as_deref(), n_ap_chans) {
            Some(g) => g,
            None => match (shank_map_str.as_deref(), &spec) {
                (Some(sm), Some(s)) => match parse_shank_map(sm, s, n_ap_chans) {
                    Some(g) => g,
                    None => {
                        warnings.push("could not read the shank map: channel positions are nominal (20 µm single column)".into());
                        nominal()
                    }
                },
                (Some(sm), None) => {
                    warnings.push(format!(
                        "probe type {im_dat_prb_type} unknown, no geometry map: channel positions are nominal (20 µm single column)"
                    ));
                    // the used flags are still valid without knowing the pitch
                    let unused = parse_shank_map(sm, &probe::spec_for_type(0).unwrap(), n_ap_chans)
                        .map(|(_, u)| u)
                        .unwrap_or_default();
                    (default_geom(n_ap_chans), unused)
                }
                (None, _) => {
                    warnings.push("no geometry in metadata: channel positions are nominal (20 µm single column)".into());
                    nominal()
                }
            },
        };

        // metadata without marked reference sites: those of the probe type
        let reference_channels = if reference_channels.is_empty() && spec.is_some() {
            reference_sites_of_type(im_dat_prb_type, &chan_nums)
        } else {
            reference_channels
        };

        // --- ADC sampling delays -------------------------------------------------
        let mux_family = spec.as_ref().map(|s| s.mux).unwrap_or(MuxFamily::None);
        let shift_fraction: Vec<f32> = match mux_str.as_deref().and_then(|m| parse_mux_table(m, mux_family)) {
            Some(by_chan) => chan_nums.iter().map(|cn| by_chan.get(cn).copied().unwrap_or(0.0)).collect(),
            None => chan_nums.iter().map(|&cn| mux_family.sample_shift_fraction(cn)).collect(),
        };
        let stream_scale = stream_delay_scale(sample_rate);
        let sample_shift = shift_fraction.iter().map(|f| f * stream_scale).collect();

        Ok(Meta {
            n_saved_chans,
            n_ap_chans,
            sample_rate,
            n_samples,
            uv_per_bit,
            im_dat_prb_type,
            channel_geom,
            channel_ids,
            sample_shift,
            warnings,
            reference_channels,
        })
    }

    /// Load metadata for an Open Ephys binary-format recording. `bin_path` is the
    /// chosen `continuous.dat` (or a compressed `.cbin` in its place); `oebin_path` and
    /// `settings_path` are the associated `structure.oebin` / `settings.xml`.
    pub fn from_open_ephys(
        bin_path: &Path,
        oebin_path: &Path,
        settings_path: &Path,
        cbin_n_samples: Option<usize>,
    ) -> Result<Self> {
        let oebin_text = std::fs::read_to_string(oebin_path)
            .with_context(|| format!("reading {}", oebin_path.display()))?;
        let oebin: serde_json::Value = serde_json::from_str(&oebin_text)
            .with_context(|| format!("parsing {}", oebin_path.display()))?;

        // the stream is identified by the name of the directory the data file lives in,
        // e.g. ".../continuous/Neuropix-PXI-100.1/continuous.dat" -> "Neuropix-PXI-100.0"
        let stream_dir_name = bin_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .context("could not determine stream folder name from data path")?;

        let continuous = oebin
            .get("continuous")
            .and_then(|v| v.as_array())
            .context("structure.oebin missing 'continuous' array")?;

        let stream = continuous
            .iter()
            .find(|c| {
                c.get("folder_name")
                    .and_then(|v| v.as_str())
                    .map(|f| f.trim_end_matches('/') == stream_dir_name)
                    .unwrap_or(false)
            })
            .with_context(|| format!("no stream '{}' found in structure.oebin", stream_dir_name))?;

        let sample_rate = stream
            .get("sample_rate")
            .and_then(|v| v.as_f64())
            .context("missing sample_rate in structure.oebin stream")?;
        let num_channels = stream
            .get("num_channels")
            .and_then(|v| v.as_u64())
            .context("missing num_channels in structure.oebin stream")? as usize;
        // identifies which PROCESSOR block in settings.xml this stream came from
        let source_processor_id = stream
            .get("source_processor_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let channels = stream
            .get("channels")
            .and_then(|v| v.as_array())
            .context("missing channels array in structure.oebin stream")?;

        // bit_volts is already a direct µV-per-bit scale (unlike SpikeGLX's gain formula);
        // key name varies across GUI versions (bit_volts vs bitVolts)
        let mut uv_per_bit = Vec::with_capacity(channels.len());
        let mut channel_ids = Vec::with_capacity(num_channels);
        for (i, ch) in channels.iter().enumerate() {
            channel_ids.push(
                ch.get("channel_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| i.to_string()),
            );
            let bv = ch
                .get("bit_volts")
                .or_else(|| ch.get("bitVolts"))
                .and_then(|v| v.as_f64())
                .context("channel missing bit_volts/bitVolts in structure.oebin")?;
            uv_per_bit.push(bv as f32);
        }
        if uv_per_bit.is_empty() {
            bail!("empty channels array in structure.oebin");
        }
        let last_bv = *uv_per_bit.last().unwrap();
        uv_per_bit.resize(num_channels, last_bv);

        let file_size_bytes = std::fs::metadata(bin_path)
            .with_context(|| format!("stat {}", bin_path.display()))?
            .len();
        let n_samples = cbin_n_samples.unwrap_or((file_size_bytes / (num_channels as u64 * 2)) as usize);

        let settings_text = std::fs::read_to_string(settings_path)
            .with_context(|| format!("reading {}", settings_path.display()))?;
        let (channel_geom, part_number, reference_channels) =
            parse_open_ephys_geometry(&settings_text, source_processor_id, num_channels)?;

        let mut warnings = Vec::new();
        let spec = probe::spec_for_part_number(&part_number);
        let (im_dat_prb_type, mux_family) = match &spec {
            Some(s) => (probe_type_of_spec(s), s.mux),
            None => {
                warnings.push(format!(
                    "probe part number '{part_number}' unknown: no ADC sampling-delay correction available"
                ));
                (0, MuxFamily::None)
            }
        };
        let stream_scale = stream_delay_scale(sample_rate);
        let sample_shift = (0..num_channels)
            .map(|ch| mux_family.sample_shift_fraction(ch) * stream_scale)
            .collect();
        // settings.xml without missing positions: the probe type's reference sites
        let reference_channels = if reference_channels.is_empty() && spec.is_some() {
            reference_sites_of_type(im_dat_prb_type, &(0..num_channels).collect::<Vec<_>>())
        } else {
            reference_channels
        };

        Ok(Meta {
            n_saved_chans: num_channels,
            n_ap_chans: num_channels,
            sample_rate,
            n_samples,
            uv_per_bit,
            im_dat_prb_type,
            channel_geom,
            channel_ids: (0..num_channels)
                .map(|i| channel_ids.get(i).cloned().unwrap_or_else(|| i.to_string()))
                .collect(),
            sample_shift,
            warnings,
            reference_channels,
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

    /// Rank shanks for left-to-right display order. `ShankOrder::Id` ranks by the raw
    /// shank index; `ShankOrder::XCoord` ranks by each shank's mean x_um, ascending.
    fn shank_rank_map(&self, order: ShankOrder) -> HashMap<u32, u32> {
        let mut shanks: Vec<u32> = self.channel_geom.iter().map(|g| g.shank).collect();
        shanks.sort_unstable();
        shanks.dedup();

        match order {
            ShankOrder::Id => shanks.iter().map(|&s| (s, s)).collect(),
            ShankOrder::XCoord => {
                let mut mean_x: Vec<(u32, f32)> = shanks.iter().map(|&s| {
                    let xs: Vec<f32> = self.channel_geom.iter()
                        .filter(|g| g.shank == s)
                        .map(|g| g.x_um)
                        .collect();
                    (s, xs.iter().sum::<f32>() / xs.len() as f32)
                }).collect();
                mean_x.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                mean_x.into_iter().enumerate().map(|(rank, (s, _))| (s, rank as u32)).collect()
            }
        }
    }

    /// Build the ordered list of display rows for rendering.
    ///
    /// If `avg_depths` is true, channels at the same (shank, y_um) are averaged into one row.
    /// `removed` holds 0-based channel indices to exclude entirely, as if they were
    /// never on the probe (they take no part in depth averaging or any spatial filter).
    /// `channel_order`/`shank_order` control the display order (see their docs); depth
    /// grouping for `avg_depths` always uses true physical proximity regardless of the
    /// chosen display order, so averaging stays correct even when ordering by ID.
    /// Gap rows are inserted wherever the vertical distance between consecutive rows
    /// exceeds `GAP_PITCH_FACTOR`× the typical pitch for that shank — only meaningful
    /// (and only done) when `channel_order` is `Depth`, since row adjacency isn't
    /// spatial otherwise.
    pub fn build_display_rows(
        &self,
        avg_depths: bool,
        removed: &BTreeSet<usize>,
        channel_order: ChannelOrder,
        shank_order: ShankOrder,
    ) -> Vec<DisplayRow> {
        let pitch_map = self.typical_pitch_per_shank();

        // collect (shank, y_um, channel_idx) tuples
        let mut entries: Vec<(u32, f32, usize)> = self.channel_geom.iter()
            .enumerate()
            .filter(|(i, _)| !removed.contains(i))
            .map(|(i, g)| (g.shank, g.y_um, i))
            .collect();
        // sort by shank, then y ascending — always by true depth here, so that
        // same-depth channels end up adjacent and get merged correctly below,
        // independent of the final display order chosen
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

        // Re-sort group channels so first_ch is the smallest index
        for (_, _, chs) in &mut groups {
            chs.sort_unstable();
        }

        // reorder the (already correctly-grouped) rows for display
        let shank_rank = self.shank_rank_map(shank_order);
        groups.sort_by(|a, b| {
            let rank_a = shank_rank.get(&a.0).copied().unwrap_or(a.0);
            let rank_b = shank_rank.get(&b.0).copied().unwrap_or(b.0);
            let key_a = match channel_order { ChannelOrder::Depth => a.1, ChannelOrder::Id => a.2[0] as f32 };
            let key_b = match channel_order { ChannelOrder::Depth => b.1, ChannelOrder::Id => b.2[0] as f32 };
            rank_a.cmp(&rank_b).then(key_a.partial_cmp(&key_b).unwrap_or(std::cmp::Ordering::Equal))
        });

        // build display rows with gap detection
        let mut rows: Vec<DisplayRow> = Vec::new();
        let mut data_idx = 0usize;
        let mut prev: Option<(u32, f32)> = None; // (shank, y)

        for (shank, y, channels) in &groups {
            let pitch = *pitch_map.get(shank).unwrap_or(&20.0);

            if let Some((prev_shank, prev_y)) = prev {
                if *shank != prev_shank {
                    // different shank: always insert a ShankBoundary
                    rows.push(DisplayRow::ShankBoundary);
                } else if channel_order == ChannelOrder::Depth && (y - prev_y) > pitch * GAP_PITCH_FACTOR {
                    // same shank: gap if spacing > 1.5× pitch (Depth order only — row
                    // adjacency in ID order doesn't correspond to physical spacing)
                    rows.push(DisplayRow::IntraShankGap);
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
                shank: *shank,
            });
            data_idx += 1;
            prev = Some((*shank, *y));
        }

        rows
    }

    /// Fractional-delay kernels for the "Phase Shift" step, one per distinct delay.
    pub fn shift_kernels(&self) -> ShiftKernels {
        ShiftKernels::new(&self.sample_shift)
    }
}

fn is_cbin(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()) == Some("cbin")
}

/// A representative `imDatPrb_type` for a spec looked up by part number.
fn probe_type_of_spec(spec: &ProbeSpec) -> u32 {
    match spec.part_number {
        "NP1000" => 0,
        "NP2000" => 21,
        "NP2010" => 24,
        pn => pn.trim_start_matches("NP").parse().unwrap_or(0),
    }
}

/// Indices of the saved channels whose acquisition channel number (`chan_nums`) is a
/// reference site of the probe type.
fn reference_sites_of_type(prb_type: u32, chan_nums: &[usize]) -> BTreeSet<usize> {
    let refs = probe::reference_channel_numbers(prb_type);
    let set: BTreeSet<usize> = (0..chan_nums.len()).filter(|&i| refs.contains(&chan_nums[i])).collect();
    // never all of them (a recording of the reference channel alone)
    if set.len() < chan_nums.len() { set } else { BTreeSet::new() }
}

/// Trailing decimal number of a channel name ("AP12" -> 12, "LF7" -> 7, "CH3" -> 3).
fn trailing_number(id: &str) -> Option<usize> {
    let digits = id.len() - id.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 {
        return None;
    }
    id[id.len() - digits..].parse().ok()
}

/// Channel names from SpikeGLX's `~snsChanMap`, e.g. "(384,384,1)(AP0;0:0)(AP1;1:1)…",
/// which lists the saved channels in file order as `name;acquisition index:order`.
/// Falls back to the 0-based file position for any channel the map doesn't cover.
fn parse_chan_map(s: Option<&str>, n_ap: usize) -> Vec<String> {
    let mut ids: Vec<String> = s
        .unwrap_or("")
        .split(')')
        .map(|t| t.trim_start_matches('('))
        .filter_map(|t| t.split_once(';').map(|(name, _)| name.trim().to_string()))
        .filter(|name| !name.is_empty())
        .take(n_ap)
        .collect();
    while ids.len() < n_ap {
        ids.push(ids.len().to_string());
    }
    ids
}

fn default_geom(n_ap: usize) -> Vec<ChannelGeom> {
    (0..n_ap)
        .map(|i| ChannelGeom { x_um: 0.0, y_um: i as f32 * 20.0, shank: 0 })
        .collect()
}

/// `~snsGeomMap`: header `(part,nShank,shankPitch,shankWidth)` then `(shank:x:y:used)`
/// per saved channel. Returns the positions and the channels flagged `used=0`;
/// `None` when the map is absent or holds no entries.
fn parse_geom_map(s: Option<&str>, n_ap: usize) -> Option<(Vec<ChannelGeom>, BTreeSet<usize>)> {
    let s = s?;
    let mut out = vec![ChannelGeom { x_um: 0.0, y_um: 0.0, shank: 0 }; n_ap];
    let mut unused = BTreeSet::new();
    let mut ch_idx = 0usize;
    for token in s.split(')') {
        let token = token.trim_start_matches('(');
        if token.is_empty() {
            continue;
        }
        let parts: Vec<&str> = token.split(':').collect();
        if parts.len() == 4 {
            if ch_idx < n_ap {
                out[ch_idx] = ChannelGeom {
                    shank: parts[0].parse().unwrap_or(0),
                    x_um: parts[1].parse().unwrap_or(0.0),
                    y_um: parts[2].parse().unwrap_or(0.0),
                };
                if parts[3].trim() == "0" {
                    unused.insert(ch_idx);
                }
            }
            ch_idx += 1;
        }
        // else: header token like "(NP1000,1,0,70)" — skip
    }
    (ch_idx > 0).then_some((out, unused))
}

/// `~snsShankMap` (SpikeGLX before 20230202): header `(nShank,nCol,nRow)` then
/// `(shank:col:row:used)` per saved channel. Positions follow from the probe's pitch.
fn parse_shank_map(s: &str, spec: &ProbeSpec, n_ap: usize) -> Option<(Vec<ChannelGeom>, BTreeSet<usize>)> {
    let mut out = vec![ChannelGeom { x_um: 0.0, y_um: 0.0, shank: 0 }; n_ap];
    let mut unused = BTreeSet::new();
    let mut ch_idx = 0usize;
    for token in s.split(')') {
        let token = token.trim_start_matches('(');
        if token.is_empty() {
            continue;
        }
        let parts: Vec<&str> = token.split(':').collect();
        if parts.len() == 4 {
            if ch_idx < n_ap {
                let shank: u32 = parts[0].parse().unwrap_or(0);
                let col: f32 = parts[1].parse().unwrap_or(0.0);
                let row: u32 = parts[2].parse().unwrap_or(0);
                let x0 = if row % 2 == 0 { spec.even_row_x0_um } else { spec.odd_row_x0_um };
                out[ch_idx] = ChannelGeom {
                    shank,
                    x_um: x0 + col * spec.pitch_h_um,
                    y_um: row as f32 * spec.pitch_v_um,
                };
                if parts[3].trim() == "0" {
                    unused.insert(ch_idx);
                }
            }
            ch_idx += 1;
        }
    }
    (ch_idx > 0).then_some((out, unused))
}

/// Gains from `~imroTbl`. Per-channel entries exist for the NP 1.0 family
/// (`(chan bank ref apGain lfGain hpf)`); NP1110 carries one gain pair in its header
/// (`(NP1110,colMode,ref,apGain,lfGain,hpf)`); NP 2.0 formats have no gain field.
struct ImroGains {
    per_channel: HashMap<usize, (f64, f64)>,
    header: Option<(f64, f64)>,
}

fn parse_imro_table(s: Option<&str>) -> Option<ImroGains> {
    let s = s?;
    let mut tokens = s.split(')').map(|t| t.trim_start_matches('(')).filter(|t| !t.is_empty());
    let header = tokens.next()?;
    let hdr: Vec<&str> = header.split(',').map(|t| t.trim()).collect();
    let header_gains = if hdr.len() == 6 {
        hdr[3].parse::<f64>().ok().zip(hdr[4].parse::<f64>().ok())
    } else {
        None
    };
    let mut per_channel = HashMap::new();
    for entry in tokens {
        let f: Vec<&str> = entry.split_whitespace().collect();
        if f.len() == 6 {
            if let (Ok(ch), Ok(ap), Ok(lf)) = (f[0].parse::<usize>(), f[3].parse::<f64>(), f[4].parse::<f64>()) {
                per_channel.insert(ch, (ap, lf));
            }
        }
    }
    if per_channel.is_empty() && header_gains.is_none() {
        return None;
    }
    Some(ImroGains { per_channel, header: header_gains })
}

/// `~muxTbl`: header `(nADC,nGrp)` then `nGrp` groups, each listing the `nADC`
/// channels digitised together; group `g` is sampled `g` cycles into the sample period.
/// Returns the delay of each acquisition channel as a fraction of the AP period.
fn parse_mux_table(s: &str, family: MuxFamily) -> Option<HashMap<usize, f32>> {
    let mut tokens = s.split(')').map(|t| t.trim_start_matches('(')).filter(|t| !t.trim().is_empty());
    let header = tokens.next()?;
    let (_n_adc, n_grp) = header.split_once(',')?;
    let n_grp: usize = n_grp.trim().parse().ok()?;
    if n_grp == 0 {
        return None;
    }
    let n_cycles = family.n_cycles(n_grp) as f32;
    let mut out = HashMap::new();
    for (g, group) in tokens.enumerate() {
        for ch in group.split_whitespace().filter_map(|c| c.parse::<usize>().ok()) {
            out.insert(ch, g as f32 / n_cycles);
        }
    }
    (!out.is_empty()).then_some(out)
}

// ---------------------------------------------------------------------------
// Open Ephys support
// ---------------------------------------------------------------------------

/// Search ancestor directories of `bin_path` for `structure.oebin`, then continue
/// searching upward from there for `settings.xml` (which lives at the Record Node
/// level, above `structure.oebin`'s experiment/recording level).
pub(crate) fn find_open_ephys_meta(bin_path: &Path) -> Option<(PathBuf, PathBuf)> {
    let ancestors: Vec<&Path> = bin_path.ancestors().collect();

    let oebin_idx = ancestors.iter().position(|dir| dir.join("structure.oebin").is_file())?;
    let oebin_path = ancestors[oebin_idx].join("structure.oebin");

    for dir in &ancestors[oebin_idx..] {
        let candidate = dir.join("settings.xml");
        if candidate.is_file() {
            return Some((oebin_path, candidate));
        }
    }
    None
}

/// Parse channel geometry and the probe part number from an Open Ephys `settings.xml`.
///
/// Positions come from `<ELECTRODE_XPOS>`/`<ELECTRODE_YPOS>` attributes (named `CH{n}`)
/// inside the `<NP_PROBE>` element nested under the `<PROCESSOR NodeId="{node_id}">`
/// block. Some channels (e.g. NP 1.0's internal reference site) have no listed
/// position — these fall back to the nearest channel index that does.
///
/// Shank is inferred from clustering x-positions: NP 1.0 / single-shank NP 2.0 columns
/// are tens of µm apart, while distinct shanks on multi-shank NP 2.0 probes are ~250 µm
/// apart, so a gap threshold separates them. This has not been tested against a real
/// multi-shank Open Ephys recording — see README.
fn parse_open_ephys_geometry(xml: &str, node_id: u32, n_ap: usize) -> Result<(Vec<ChannelGeom>, String, BTreeSet<usize>)> {
    let doc = roxmltree::Document::parse(xml).context("parsing settings.xml")?;

    let processor = doc
        .descendants()
        .find(|n| {
            n.has_tag_name("PROCESSOR")
                && n.attribute("NodeId").and_then(|s| s.parse::<u32>().ok()) == Some(node_id)
        })
        .with_context(|| format!("no PROCESSOR with NodeId={} in settings.xml", node_id))?;

    let np_probe = processor
        .descendants()
        .find(|n| n.has_tag_name("NP_PROBE"))
        .context("no NP_PROBE found for this processor — only Neuropixels streams are supported")?;

    let part_number = np_probe.attribute("probe_part_number").unwrap_or("").to_string();

    let xpos_node = np_probe
        .descendants()
        .find(|n| n.has_tag_name("ELECTRODE_XPOS"))
        .context("missing ELECTRODE_XPOS in settings.xml")?;
    let ypos_node = np_probe
        .descendants()
        .find(|n| n.has_tag_name("ELECTRODE_YPOS"))
        .context("missing ELECTRODE_YPOS in settings.xml")?;

    let read_ch_attrs = |node: roxmltree::Node| -> HashMap<usize, f32> {
        node.attributes()
            .filter_map(|attr| {
                let idx = attr.name().strip_prefix("CH")?.parse::<usize>().ok()?;
                let val = attr.value().parse::<f32>().ok()?;
                Some((idx, val))
            })
            .collect()
    };
    let xs = read_ch_attrs(xpos_node);
    let ys = read_ch_attrs(ypos_node);

    // cluster x-positions into shanks: sort unique values, start a new shank whenever
    // consecutive values are further apart than a single-shank column spacing
    const SHANK_GAP_THRESHOLD_UM: f32 = 100.0;
    let mut unique_x: Vec<f32> = xs.values().copied().collect();
    unique_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_x.dedup_by(|a, b| (*a - *b).abs() < 1.0);
    let mut cluster_starts: Vec<f32> = Vec::new();
    for &x in &unique_x {
        if cluster_starts.last().map_or(true, |&last| x - last > SHANK_GAP_THRESHOLD_UM) {
            cluster_starts.push(x);
        }
    }
    let shank_for_x = |x: f32| -> u32 {
        cluster_starts.iter().rposition(|&start| x + 0.5 >= start).unwrap_or(0) as u32
    };

    let mut channel_geom = vec![ChannelGeom { x_um: 0.0, y_um: 0.0, shank: 0 }; n_ap];
    // channels without a listed position are reference sites (NP 1.0: channel 191) —
    // only meaningful if the probe lists positions for most channels at all
    let mut no_position = BTreeSet::new();
    for i in 0..n_ap {
        let (x, y) = match (xs.get(&i), ys.get(&i)) {
            (Some(&x), Some(&y)) => (x, y),
            _ => {
                no_position.insert(i);
                // fallback for channels with no listed position (e.g. NP 1.0's
                // internal reference site): use the nearest channel that has one
                (1..n_ap)
                    .find_map(|d| {
                        i.checked_sub(d).and_then(|lo| xs.get(&lo).zip(ys.get(&lo)))
                            .or_else(|| (i + d < n_ap).then(|| ()).and_then(|_| xs.get(&(i + d)).zip(ys.get(&(i + d)))))
                    })
                    .map(|(&x, &y)| (x, y))
                    .unwrap_or((0.0, 0.0))
            }
        };
        channel_geom[i] = ChannelGeom { x_um: x, y_um: y, shank: shank_for_x(x) };
    }

    if no_position.len() * 2 > n_ap {
        no_position.clear(); // positions largely missing: not a reference-site pattern
    }
    Ok((channel_geom, part_number, no_position))
}

// ---------------------------------------------------------------------------
// Fractional-delay kernels (ADC sampling-delay correction)
// ---------------------------------------------------------------------------

/// Taps of the windowed-sinc fractional-delay filter.
pub const SHIFT_TAPS: usize = 16;
/// Output sample `t` is computed from source samples `t - (SHIFT_TAPS - 1 - SHIFT_LEAD) ..= t + SHIFT_LEAD`.
const SHIFT_LEAD: isize = 7;
/// Source samples needed beyond the requested range on either side.
pub const SHIFT_HALO: usize = SHIFT_TAPS;

/// Lanczos-windowed sinc kernels that delay a signal by a fraction of a sample
/// (`y[t] = x[t - d]`), one per distinct delay, plus the kernel index of every channel
/// (`None` = no delay needed).
pub struct ShiftKernels {
    pub kernels: Vec<[f32; SHIFT_TAPS]>,
    pub channel_kernel: Vec<Option<usize>>,
}

impl ShiftKernels {
    pub fn new(sample_shift: &[f32]) -> Self {
        let mut kernels: Vec<[f32; SHIFT_TAPS]> = Vec::new();
        let mut delays: Vec<f32> = Vec::new();
        let channel_kernel = sample_shift
            .iter()
            .map(|&d| {
                if d.abs() < 1e-4 {
                    return None;
                }
                if let Some(i) = delays.iter().position(|&k| (k - d).abs() < 1e-4) {
                    return Some(i);
                }
                delays.push(d);
                kernels.push(Self::kernel(d));
                Some(kernels.len() - 1)
            })
            .collect();
        Self { kernels, channel_kernel }
    }

    /// Kernel for delay `d` (samples): tap `k` weights source sample `t + SHIFT_LEAD - k`.
    fn kernel(d: f32) -> [f32; SHIFT_TAPS] {
        let a = (SHIFT_TAPS / 2) as f64;
        let sinc = |u: f64| if u.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * u).sin() / (std::f64::consts::PI * u) };
        let mut h = [0.0f32; SHIFT_TAPS];
        let mut sum = 0.0f64;
        for k in 0..SHIFT_TAPS {
            let u = k as f64 - SHIFT_LEAD as f64 - d as f64;
            let w = if u.abs() < a { sinc(u) * sinc(u / a) } else { 0.0 };
            h[k] = w as f32;
            sum += w;
        }
        for v in &mut h {
            *v = (*v as f64 / sum) as f32;
        }
        h
    }
}

// ---------------------------------------------------------------------------
// Raw data access
// ---------------------------------------------------------------------------

pub enum RawData {
    Uncompressed(memmap2::Mmap),
    Compressed(crate::mtscomp::MtscompReader),
}

/// Time samples converted per parallel work item: a block of interleaved source
/// samples small enough to stay in cache while every row reads from it.
const BLOCK_T: usize = 1024;

/// Raw mutable pointer wrapper for cross-thread access in rayon.
/// SAFETY: each parallel item writes only its own, disjoint time range.
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Convert interleaved i16 samples (`src[s * n_ch + ch]`) into per-row f32 traces:
/// `out[r * n_samp + t]` is the mean over `rows[r]`'s channels of `scale[ch] * x[ch]`
/// at source sample `src_offset + t`, with the per-channel fractional delay applied
/// first when `kernels` is given. Samples beyond the source are zero.
fn convert_rows(
    src: &[i16],
    n_ch: usize,
    src_offset: usize,
    n_samp: usize,
    rows: &[&[usize]],
    scale: &[f32],
    kernels: Option<&ShiftKernels>,
    out: &mut [f32],
) {
    use rayon::prelude::*;
    debug_assert_eq!(out.len(), rows.len() * n_samp);
    if n_samp == 0 || rows.is_empty() || n_ch == 0 {
        return;
    }
    let src_len = src.len() / n_ch;
    let inv: Vec<f32> = rows.iter().map(|r| 1.0 / r.len().max(1) as f32).collect();
    let out_ptr = SendPtr(out.as_mut_ptr());
    let n_blocks = (n_samp + BLOCK_T - 1) / BLOCK_T;

    // kernels reversed, so the delay is a plain dot product with a contiguous window
    // of past samples: y[t] = Σ_j hr[j] · x[t + LEAD - (TAPS-1) + j]
    let rev: Vec<[f32; SHIFT_TAPS]> = kernels
        .map(|k| k.kernels.iter().map(|h| { let mut r = *h; r.reverse(); r }).collect())
        .unwrap_or_default();
    let chan_kernel = |ch: usize| kernels.and_then(|k| k.channel_kernel.get(ch).copied().flatten());

    (0..n_blocks).into_par_iter().for_each(|b| {
        let op = out_ptr.0;
        let _ = &out_ptr;
        let t0 = b * BLOCK_T;
        let t1 = ((b + 1) * BLOCK_T).min(n_samp);
        let s0 = src_offset + t0; // first source sample of the block
        let s1 = (src_offset + t1).min(src_len); // end of the in-range source samples
        // zero the block (covers samples beyond the source)
        for r in 0..rows.len() {
            unsafe { std::ptr::write_bytes(op.add(r * n_samp + t0), 0, t1 - t0) };
        }
        if s0 >= s1 {
            return;
        }

        if kernels.is_none() {
            // no delay: read each source sample once, straight into the rows
            for s in s0..s1 {
                let t = s - src_offset;
                let base = s * n_ch;
                for (r, chans) in rows.iter().enumerate() {
                    let mut acc = 0.0f32;
                    for &ch in chans.iter() {
                        acc += src[base + ch] as f32 * scale[ch];
                    }
                    unsafe { *op.add(r * n_samp + t) = acc * inv[r] };
                }
            }
            return;
        }

        // with delay: transpose block + halo into contiguous per-channel scratch,
        // then filter each channel with a contiguous dot product and accumulate
        let h_lo = s0.saturating_sub(SHIFT_HALO);
        let h_hi = (s1 + SHIFT_HALO).min(src_len);
        let w = h_hi - h_lo;
        SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            sc.resize(n_ch * w, 0.0);
            for s in h_lo..h_hi {
                let base = s * n_ch;
                let j = s - h_lo;
                for ch in 0..n_ch {
                    sc[ch * w + j] = src[base + ch] as f32;
                }
            }
            for (r, chans) in rows.iter().enumerate() {
                let dst = unsafe { std::slice::from_raw_parts_mut(op.add(r * n_samp + t0), t1 - t0) };
                for &ch in chans.iter() {
                    let x = &sc[ch * w..(ch + 1) * w];
                    let g = scale[ch] * inv[r];
                    match chan_kernel(ch) {
                        None => {
                            for s in s0..s1 {
                                dst[s - s0] += x[s - h_lo] * g;
                            }
                        }
                        Some(ki) => {
                            let hr = &rev[ki];
                            let lead = SHIFT_LEAD as usize;
                            let back = SHIFT_TAPS - 1 - lead;
                            for s in s0..s1 {
                                let v = if s >= back && s + lead < src_len {
                                    // interior: contiguous window, vectorizable
                                    let lo = s - back - h_lo;
                                    let win = &x[lo..lo + SHIFT_TAPS];
                                    let mut a = 0.0f32;
                                    for j in 0..SHIFT_TAPS {
                                        a += hr[j] * win[j];
                                    }
                                    a
                                } else {
                                    // recording edges: repeat the first / last sample
                                    let mut a = 0.0f32;
                                    for j in 0..SHIFT_TAPS {
                                        let si = (s as isize - back as isize + j as isize)
                                            .clamp(0, src_len as isize - 1) as usize;
                                        a += hr[j] * x[si - h_lo];
                                    }
                                    a
                                };
                                dst[s - s0] += v * g;
                            }
                        }
                    }
                }
            }
        });
    });
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
}

impl RawData {
    /// Samples `[first, first + n_samp)` of every signal channel in µV, layout
    /// `[n_ap][n_samp]`, without delay correction (used by channel classification).
    pub fn read_chunk_uv(&self, first_sample: usize, n_samp: usize, meta: &Meta) -> Vec<f32> {
        let singles: Vec<Vec<usize>> = (0..meta.n_ap_chans).map(|c| vec![c]).collect();
        let rows: Vec<&[usize]> = singles.iter().map(|v| v.as_slice()).collect();
        self.read_into_rows(first_sample, n_samp, meta, &rows, None)
    }

    /// Samples `[first, first + n_samp)` averaged into the given display rows (µV),
    /// layout `[n_data_rows][n_samp]`; with `phase_shift`, each channel's ADC sampling
    /// delay is corrected before averaging.
    pub fn read_rows(
        &self,
        first_sample: usize,
        n_samp: usize,
        meta: &Meta,
        display_rows: &[DisplayRow],
        phase_shift: bool,
    ) -> Vec<f32> {
        let rows: Vec<&[usize]> = display_rows
            .iter()
            .filter_map(|r| match r {
                DisplayRow::Data { channels, .. } => Some(channels.as_slice()),
                _ => None,
            })
            .collect();
        let kernels = phase_shift.then(|| meta.shift_kernels());
        self.read_into_rows(first_sample, n_samp, meta, &rows, kernels.as_ref())
    }

    fn read_into_rows(
        &self,
        first_sample: usize,
        n_samp: usize,
        meta: &Meta,
        rows: &[&[usize]],
        kernels: Option<&ShiftKernels>,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows.len() * n_samp];
        if n_samp == 0 || rows.is_empty() {
            return out;
        }
        let n_ch = meta.n_saved_chans;
        match self {
            RawData::Uncompressed(mmap) => {
                let src = mmap_as_i16(mmap);
                convert_rows(src, n_ch, first_sample, n_samp, rows, &meta.uv_per_bit, kernels, &mut out);
            }
            RawData::Compressed(reader) => {
                // the range plus the filter halo, decompressed into one interleaved buffer
                let want_lo = first_sample.saturating_sub(SHIFT_HALO);
                let want_hi = first_sample + n_samp + SHIFT_HALO;
                let Some((src, gather_start)) = gather_compressed(reader, want_lo, want_hi) else {
                    return out;
                };
                let src_offset = first_sample.saturating_sub(gather_start);
                convert_rows(&src, n_ch, src_offset, n_samp, rows, &meta.uv_per_bit, kernels, &mut out);
            }
        }
        out
    }

    /// Raw samples `[first, first + n_samp)` of the file channels `chans`, layout
    /// `[n_samp][chans.len()]`; samples beyond the recording are 0. Used to copy
    /// channels unchanged (the SpikeGLX sync channel) into an export.
    pub fn read_i16(&self, first_sample: usize, n_samp: usize, meta: &Meta, chans: &[usize]) -> Vec<i16> {
        let k = chans.len();
        let mut out = vec![0i16; n_samp * k];
        if n_samp == 0 || k == 0 {
            return out;
        }
        let n_ch = meta.n_saved_chans;
        let mut copy = |src: &[i16], src_offset: usize| {
            let src_len = src.len() / n_ch;
            for t in 0..n_samp.min(src_len.saturating_sub(src_offset)) {
                let base = (src_offset + t) * n_ch;
                for (j, &c) in chans.iter().enumerate() {
                    out[t * k + j] = src[base + c];
                }
            }
        };
        match self {
            RawData::Uncompressed(mmap) => copy(mmap_as_i16(mmap), first_sample),
            RawData::Compressed(reader) => {
                if let Some((src, gather_start)) = gather_compressed(reader, first_sample, first_sample + n_samp) {
                    copy(&src, first_sample - gather_start);
                }
            }
        }
        out
    }
}

/// Decompress every chunk overlapping samples `[want_lo, want_hi)` (in parallel) and
/// stitch them into one interleaved buffer; returns it with its first sample, or
/// `None` if the range lies outside the recording.
fn gather_compressed(reader: &crate::mtscomp::MtscompReader, want_lo: usize, want_hi: usize) -> Option<(Vec<i16>, usize)> {
    use rayon::prelude::*;
    let bounds = &reader.meta.chunk_bounds;
    let n_ch = reader.meta.n_channels;
    if bounds.len() < 2 {
        return None;
    }
    let want_hi = want_hi.min(*bounds.last().unwrap());
    let n_chunks = bounds.len() - 1;
    let c_lo = bounds.partition_point(|&b| b <= want_lo).saturating_sub(1).min(n_chunks - 1);
    let c_hi = bounds.partition_point(|&b| b < want_hi).min(n_chunks); // exclusive
    if c_lo >= c_hi {
        return None;
    }
    let gather_start = bounds[c_lo];
    let gather_len = bounds[c_hi] - gather_start;
    let mut src = vec![0i16; gather_len * n_ch];
    let pieces: Vec<Option<Arc<Vec<i16>>>> = (c_lo..c_hi).into_par_iter().map(|c| reader.chunk(c).ok()).collect();
    for (i, piece) in pieces.into_iter().enumerate() {
        if let Some(p) = piece {
            let off = (bounds[c_lo + i] - gather_start) * n_ch;
            let len = p.len().min(src.len() - off);
            src[off..off + len].copy_from_slice(&p[..len]);
        }
    }
    Some((src, gather_start))
}

/// The mapped file as i16 samples; a trailing odd byte (a file still being written)
/// is ignored rather than causing a panic.
fn mmap_as_i16(mmap: &memmap2::Mmap) -> &[i16] {
    let bytes = mmap.as_ref();
    let even = &bytes[..bytes.len() & !1];
    bytemuck::try_cast_slice(even).unwrap_or(&[])
}

pub fn open_data(bin_path: &Path, meta: &Meta) -> Result<RawData> {
    if is_cbin(bin_path) {
        let ch_path = bin_path.with_extension("ch");
        if !ch_path.exists() {
            bail!("Metadata file {} not found for {}", ch_path.display(), bin_path.display());
        }
        let mts_meta = crate::mtscomp::MtscompMeta::from_file(&ch_path)?;
        if mts_meta.n_channels != meta.n_saved_chans {
            bail!(
                "{} lists {} channels but the recording metadata says {}",
                ch_path.display(),
                mts_meta.n_channels,
                meta.n_saved_chans
            );
        }
        if mts_meta.chunk_bounds.len() < 2 || mts_meta.chunk_offsets.len() < mts_meta.chunk_bounds.len() {
            bail!("{} has no chunks", ch_path.display());
        }
        let reader = crate::mtscomp::MtscompReader::new(bin_path, mts_meta)?;
        Ok(RawData::Compressed(reader))
    } else {
        let file = std::fs::File::open(bin_path)
            .with_context(|| format!("opening {}", bin_path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // verify alignment
        if mmap.as_ptr() as usize % 2 != 0 {
            bail!("mmap pointer is not 2-byte aligned");
        }
        Ok(RawData::Uncompressed(mmap))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn chan_map_names_in_file_order() {
        // saved subset (AP5, AP6, AP200) + sync channel, as SpikeGLX writes it
        let map = "(384,384,1)(AP5;5:5)(AP6;6:6)(AP200;200:200)(SY0;768:768)";
        assert_eq!(parse_chan_map(Some(map), 3), vec!["AP5", "AP6", "AP200"]);
        // missing map: fall back to file positions
        assert_eq!(parse_chan_map(None, 2), vec!["0", "1"]);
        assert_eq!(trailing_number("AP12"), Some(12));
        assert_eq!(trailing_number("LF0"), Some(0));
        assert_eq!(trailing_number("SY"), None);
    }

    #[test]
    fn imro_gains() {
        let t = parse_imro_table(Some("(0,384)(0 0 0 500 250 1)(1 0 0 1000 250 1)")).unwrap();
        assert_eq!(t.per_channel[&0], (500.0, 250.0));
        assert_eq!(t.per_channel[&1], (1000.0, 250.0));
        // NP2: no gain field
        assert!(parse_imro_table(Some("(21,384)(0 1 0 0)(1 1 0 1)")).is_none());
        // NP1110: gains in the header
        let t = parse_imro_table(Some("(NP1110,2,0,500,250,1)(0 0 0)(1 0 0)")).unwrap();
        assert_eq!(t.header, Some((500.0, 250.0)));
        assert!(t.per_channel.is_empty());
    }

    #[test]
    fn mux_table_delays() {
        let m = parse_mux_table("(32,12)(0 1 24 25)(2 3 26 27)(4 5 28 29)", MuxFamily::Np1).unwrap();
        assert_eq!(m[&0], 0.0);
        assert_eq!(m[&25], 0.0);
        assert!((m[&2] - 1.0 / 13.0).abs() < 1e-7);
        assert!((m[&29] - 2.0 / 13.0).abs() < 1e-7);
        let m = parse_mux_table("(24,16)(0 1 32 33)(2 3 34 35)", MuxFamily::Np2).unwrap();
        assert!((m[&34] - 1.0 / 16.0).abs() < 1e-7);
    }

    #[test]
    fn shank_map_geometry_matches_geom_map() {
        // NP 1.0 checkerboard: the ~snsGeomMap example from the SpikeGLX docs
        let spec = probe::spec_for_type(0).unwrap();
        let (g, unused) = parse_shank_map("(1,2,480)(0:0:0:1)(0:1:0:1)(0:0:1:1)(0:1:1:0)", &spec, 4).unwrap();
        assert_eq!(unused.into_iter().collect::<Vec<_>>(), vec![3]);
        let (expect, unused) = parse_geom_map(Some("(NP1000,1,0,70)(0:27:0:1)(0:59:0:1)(0:11:20:0)(0:43:20:1)"), 4).unwrap();
        assert_eq!(unused.into_iter().collect::<Vec<_>>(), vec![2]);
        for (a, b) in g.iter().zip(&expect) {
            assert_eq!((a.x_um, a.y_um, a.shank), (b.x_um, b.y_um, b.shank));
        }
        assert!(parse_geom_map(Some("(NP1000,1,0,70)"), 4).is_none());
    }

    #[test]
    fn fractional_delay_kernels() {
        // zero delay -> no kernel; delay 1 -> exact one-sample shift; sum is 1
        let k = ShiftKernels::new(&[0.0, 0.5, 1.0, 0.5]);
        assert_eq!(k.channel_kernel, vec![None, Some(0), Some(1), Some(0)]);
        assert_eq!(k.kernels.len(), 2);
        let one = &k.kernels[1];
        assert!((one[(SHIFT_LEAD + 1) as usize] - 1.0).abs() < 1e-6);
        assert!(one.iter().enumerate().all(|(i, &v)| i == (SHIFT_LEAD + 1) as usize || v.abs() < 1e-6));
        let half = &k.kernels[0];
        assert!((half.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((half[SHIFT_LEAD as usize] - half[SHIFT_LEAD as usize + 1]).abs() < 1e-6);
    }

    /// Old-style (pre-2023) NP 2.0 single-shank meta: no gain/maxInt fields, no
    /// ~snsGeomMap, only ~snsShankMap — everything must come from the probe table.
    fn old_np2_meta(n_ch: usize, n_samples: usize) -> String {
        let n_ap = n_ch - 1;
        let mut chan_map = format!("({n_ap},0,1)");
        let mut shank_map = format!("(1,2,{})", n_ap / 2 + 1);
        for c in 0..n_ap {
            chan_map += &format!("(AP{c};{c}:{c})");
            shank_map += &format!("(0:{}:{}:1)", c % 2, c / 2);
        }
        chan_map += &format!("(SY0;{n_ap}:{n_ap})");
        format!(
            "nSavedChans={n_ch}\nimSampRate=30000\nfileSizeBytes={}\nimAiRangeMax=0.62\n\
             snsApLfSy={n_ap},0,1\nimDatPrb_type=21\n~snsChanMap={chan_map}\n~snsShankMap={shank_map}\n",
            n_samples * n_ch * 2
        )
    }

    /// Interleaved i16 test signal: channel c at sample s holds (s % 100) * 10 + c.
    fn test_signal(n_ch: usize, n_samples: usize) -> Vec<i16> {
        (0..n_samples * n_ch).map(|i| ((i / n_ch) % 100 * 10 + i % n_ch) as i16).collect()
    }

    /// Write `sig` as an mtscomp .cbin/.ch pair (time diff on; `fortran` = store each
    /// chunk channel-major, mtscomp's default).
    pub(crate) fn write_cbin(path: &Path, sig: &[i16], n_ch: usize, chunk_samples: usize, fortran: bool) {
        use std::io::Write;
        let n_samples = sig.len() / n_ch;
        let mut bounds = vec![0usize];
        let mut offsets = vec![0u64];
        let mut out = Vec::new();
        let mut s0 = 0;
        while s0 < n_samples {
            let s1 = (s0 + chunk_samples).min(n_samples);
            let d = |s: usize, c: usize| {
                let v = sig[s * n_ch + c];
                if s == s0 { v } else { v.wrapping_sub(sig[(s - 1) * n_ch + c]) }
            };
            let mut diff = Vec::with_capacity((s1 - s0) * n_ch);
            if fortran {
                for c in 0..n_ch { for s in s0..s1 { diff.push(d(s, c)); } }
            } else {
                for s in s0..s1 { for c in 0..n_ch { diff.push(d(s, c)); } }
            }
            let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
            enc.write_all(bytemuck::cast_slice(&diff)).unwrap();
            out.extend(enc.finish().unwrap());
            bounds.push(s1);
            offsets.push(out.len() as u64);
            s0 = s1;
        }
        std::fs::write(path, &out).unwrap();
        let ch = serde_json::json!({
            "chunk_bounds": bounds, "chunk_offsets": offsets, "chunk_order": if fortran { "F" } else { "C" },
            "do_spatial_diff": false, "do_time_diff": true, "dtype": "int16", "n_channels": n_ch,
            "sample_rate": 30000.0
        });
        std::fs::write(path.with_extension("ch"), ch.to_string()).unwrap();
    }

    #[test]
    fn spikeglx_end_to_end_uncompressed_and_compressed() {
        let dir = std::env::temp_dir().join(format!("npx_data_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (n_ch, n_samples) = (5usize, 2500usize);
        let sig = test_signal(n_ch, n_samples);

        let bin = dir.join("rec_g0_t0.imec0.ap.bin");
        std::fs::write(&bin, bytemuck::cast_slice::<i16, u8>(&sig)).unwrap();
        std::fs::write(bin.with_extension("meta"), old_np2_meta(n_ch, n_samples)).unwrap();
        let cbin = dir.join("rec_g0_t0.imec0.ap.cbin");
        write_cbin(&cbin, &sig, n_ch, 1000, false);
        // the cbin's own meta must not report the true size (mtscomp never rewrites it,
        // but a size-derived count from the compressed file would be far too small)
        std::fs::write(cbin.with_extension("meta"), old_np2_meta(n_ch, n_samples)).unwrap();

        let meta = Meta::from_data_path(&bin).unwrap();
        assert_eq!(meta.n_samples, n_samples);
        assert_eq!(meta.n_ap_chans, 4);
        assert_eq!(meta.channel_ids, vec!["AP0", "AP1", "AP2", "AP3"]);
        // NP 2.0 (type 21): gain 80, 14-bit ADC -> 0.62 / 8192 / 80 V per bit
        let expect_uv = (0.62 / 8192.0 / 80.0 * 1e6) as f32;
        assert!(meta.uv_per_bit.iter().all(|&v| (v - expect_uv).abs() < 1e-4), "{:?}", meta.uv_per_bit);
        assert!(meta.warnings.is_empty(), "{:?}", meta.warnings);
        // geometry from the shank map: two columns 32 µm apart, rows 15 µm apart
        let g: Vec<(f32, f32)> = meta.channel_geom.iter().map(|g| (g.x_um, g.y_um)).collect();
        assert_eq!(g, vec![(27.0, 0.0), (59.0, 0.0), (27.0, 15.0), (59.0, 15.0)]);
        // NP2 mux: channels 0,1 sampled first, 2,3 one cycle (1/16 sample) later
        assert_eq!(meta.sample_shift[0], 0.0);
        assert!((meta.sample_shift[2] - 1.0 / 16.0).abs() < 1e-6);

        let raw = open_data(&bin, &meta).unwrap();
        let rows = meta.build_display_rows(true, &BTreeSet::new(), ChannelOrder::Depth, ShankOrder::Id);
        assert_eq!(rows.len(), 2); // two depth-averaged rows
        let data = raw.read_rows(1990, 20, &meta, &rows, false);
        // row 0 = mean of channels 0,1 at samples 1990.. ; sample 1990 % 100 = 90 -> 900.5
        assert!((data[0] - 900.5 * expect_uv).abs() < 1e-2, "{}", data[0]);
        assert!((data[20] - 902.5 * expect_uv).abs() < 1e-2);

        // the compressed copy: sample count from the .ch, identical samples, also when
        // the read spans a chunk boundary and when phase shift needs the halo
        let cmeta = Meta::from_data_path(&cbin).unwrap();
        assert_eq!(cmeta.n_samples, n_samples);
        let craw = open_data(&cbin, &cmeta).unwrap();
        let cdata = craw.read_rows(1990, 20, &cmeta, &rows, false);
        for (a, b) in data.iter().zip(&cdata) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        let shifted = raw.read_rows(1990, 20, &meta, &rows, true);
        let cshifted = craw.read_rows(1990, 20, &cmeta, &rows, true);
        for (a, b) in shifted.iter().zip(&cshifted) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        // channels 0,1 have no delay: row 0 unchanged by the phase shift
        assert!((shifted[5] - data[5]).abs() < 1e-4);
        // reading past the end yields zeros, not a panic
        let tail = raw.read_chunk_uv(n_samples - 3, 10, &meta);
        assert_eq!(tail.len(), 4 * 10);
        assert_eq!(tail[3], 0.0);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn convert_rows_averages_scales_and_delays() {
        // 3 channels interleaved, 8 samples; channel c at sample s holds 10*s + c
        let n_ch = 3;
        let n_s = 8;
        let src: Vec<i16> = (0..n_s * n_ch).map(|i| (10 * (i / n_ch) + i % n_ch) as i16).collect();
        let rows: Vec<&[usize]> = vec![&[0, 1], &[2]];
        let scale = [1.0, 1.0, 2.0];
        let mut out = vec![0.0; 2 * 4];
        convert_rows(&src, n_ch, 2, 4, &rows, &scale, None, &mut out);
        assert_eq!(out, vec![20.5, 30.5, 40.5, 50.5, 44.0, 64.0, 84.0, 104.0]);
        // delay channel 2 by one sample: row 1 becomes the previous sample's value
        let k = ShiftKernels::new(&[0.0, 0.0, 1.0]);
        convert_rows(&src, n_ch, 2, 4, &rows, &scale, Some(&k), &mut out);
        assert!((out[4] - 24.0).abs() < 1e-3 && (out[5] - 44.0).abs() < 1e-3);
        // past the end -> zeros
        convert_rows(&src, n_ch, 6, 4, &rows, &scale, None, &mut out);
        assert_eq!(&out[2..4], &[0.0, 0.0]);
    }
}
