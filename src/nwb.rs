// Recordings in Neurodata Without Borders files (`.nwb`, HDF5): an `ElectricalSeries` in
// `/acquisition` holds the samples as a [n_samples][n_channels] int16 dataset (usually in
// gzip-compressed chunks) with a factor to volts, and points into the electrodes table
// in `/general/extracellular_ephys` for the per-channel values.
//
// The file says little about the probe: electrode positions only if the table has the
// `rel_x`/`rel_y` columns, the ADC sampling delay only with an `inter_sample_shift` column.

use anyhow::{bail, Context, Result};
use hdf5_pure::AttrValue;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::data::{default_geom, ChannelGeom, ChunkSource, Meta};
use crate::mtscomp::ChunkCache;

const ELECTRODES: &str = "general/extracellular_ephys/electrodes";

pub fn is_nwb(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("nwb")) && path.is_file()
}

fn open(path: &Path) -> Result<hdf5_pure::File> {
    hdf5_pure::File::open_streaming(path).map_err(|e| anyhow::anyhow!("opening {}: {e}", path.display()))
}

fn attr_f64(attrs: &HashMap<String, AttrValue>, name: &str) -> Option<f64> {
    Some(match attrs.get(name)? {
        AttrValue::F64(v) => *v,
        AttrValue::F32(v) => *v as f64,
        AttrValue::I64(v) => *v as f64,
        AttrValue::I32(v) => *v as f64,
        AttrValue::U64(v) => *v as f64,
        AttrValue::U32(v) => *v as f64,
        AttrValue::F64Array(v) if v.len() == 1 => v[0],
        _ => return None,
    })
}

/// The series to show and what the file says about it.
struct Series {
    /// e.g. "acquisition/ElectricalSeriesAP"
    path: String,
    rate: f64,
    n_samples: usize,
    n_ch: usize,
    /// rows of the dataset per stored chunk (a decompression unit)
    chunk_rows: usize,
    /// names of the other electrical series in the file
    others: Vec<String>,
}

fn find_series(file: &hdf5_pure::File, path: &Path) -> Result<Series> {
    let err = |e: hdf5_pure::Error| anyhow::anyhow!("{}: {e}", path.display());
    let acquisition = file.group("acquisition").map_err(err)?;
    let mut found: Vec<Series> = Vec::new();
    for name in acquisition.groups().map_err(err)? {
        let gpath = format!("acquisition/{name}");
        let Ok(group) = file.group(&gpath) else { continue };
        let is_series = group
            .attrs()
            .is_ok_and(|a| matches!(a.get("neurodata_type"), Some(AttrValue::String(t) | AttrValue::VarLenString(t)) if t == "ElectricalSeries"));
        let Ok(data) = file.dataset(&format!("{gpath}/data")) else { continue };
        let shape = data.shape().map_err(err)?;
        if !is_series || shape.len() != 2 {
            continue;
        }
        // regular sampling: the rate; else the first two timestamps
        let rate = file
            .dataset(&format!("{gpath}/starting_time"))
            .ok()
            .and_then(|d| d.attrs().ok())
            .and_then(|a| attr_f64(&a, "rate"))
            .or_else(|| {
                let t = file.dataset(&format!("{gpath}/timestamps")).ok()?.read_f64_rows(0, 2).ok()?;
                (t.len() == 2 && t[1] > t[0]).then(|| 1.0 / (t[1] - t[0]))
            });
        let Some(rate) = rate else { continue };
        let (n_samples, n_ch) = (shape[0] as usize, shape[1] as usize);
        let chunk_rows = match data.chunk_shape().map_err(err)? {
            Some(c) if !c.is_empty() && c[0] > 0 => c[0] as usize,
            // not chunked: read a second at a time
            _ => (rate.round() as usize).max(1),
        };
        found.push(Series { path: gpath, rate, n_samples, n_ch, chunk_rows, others: Vec::new() });
    }
    // the AP band, if the file holds several
    found.sort_by(|a, b| b.rate.total_cmp(&a.rate));
    let names: Vec<String> = found.iter().skip(1).map(|s| s.path.trim_start_matches("acquisition/").to_string()).collect();
    let mut first = found.into_iter().next().with_context(|| format!("{}: no ElectricalSeries in /acquisition", path.display()))?;
    first.others = names;
    Ok(first)
}

/// Metadata of the recording in the NWB file `path`.
pub fn load_meta(path: &Path) -> Result<Meta> {
    let file = open(path)?;
    let series = find_series(&file, path)?;
    let n_ch = series.n_ch;
    let name = series.path.trim_start_matches("acquisition/");
    let mut warnings = Vec::new();
    if !series.others.is_empty() {
        warnings.push(format!("the file holds several electrical series: showing {name}, not {}", series.others.join(", ")));
    }

    let data = file.dataset(&format!("{}/data", series.path)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let attrs = data.attrs().unwrap_or_default();
    let volts = attr_f64(&attrs, "conversion").unwrap_or(1.0);
    if attr_f64(&attrs, "offset").is_some_and(|o| o != 0.0) {
        warnings.push("the series' offset is not 0 and is ignored".to_string());
    }
    let per_channel = file
        .dataset(&format!("{}/channel_conversion", series.path))
        .ok()
        .and_then(|d| d.read_f64().ok())
        .filter(|v| v.len() == n_ch);
    let uv_per_bit: Vec<f32> = (0..n_ch).map(|i| (volts * 1e6 * per_channel.as_ref().map_or(1.0, |c| c[i])) as f32).collect();

    // rows of the electrodes table, one per channel of the series
    let rows: Vec<usize> = file
        .dataset(&format!("{}/electrodes", series.path))
        .ok()
        .and_then(|d| d.read_f64().ok())
        .filter(|v| v.len() == n_ch)
        .map(|v| v.iter().map(|&r| r as usize).collect())
        .unwrap_or_else(|| (0..n_ch).collect());
    let pick = |v: Vec<f64>| -> Option<Vec<f64>> { rows.iter().map(|&r| v.get(r).copied()).collect() };
    let column = |name: &str| file.dataset(&format!("{ELECTRODES}/{name}")).ok().and_then(|d| d.read_f64().ok()).and_then(pick);
    let text_column = |name: &str| -> Option<Vec<String>> {
        let all = file.dataset(&format!("{ELECTRODES}/{name}")).ok()?.read_string().ok()?;
        rows.iter().map(|&r| all.get(r).cloned()).collect()
    };

    let channel_ids: Vec<String> = text_column("channel_name")
        .or_else(|| text_column("electrode_name"))
        .map(|names| names.iter().map(|n| n.rsplit('#').next().unwrap_or(n).to_string()).collect())
        .unwrap_or_else(|| rows.iter().map(|r| r.to_string()).collect());

    let channel_geom: Vec<ChannelGeom> = match (column("rel_x"), column("rel_y")) {
        (Some(x), Some(y)) => {
            // electrode groups in order of appearance are the shanks
            let groups = text_column("group_name").unwrap_or_default();
            let mut seen: Vec<&String> = Vec::new();
            (0..n_ch)
                .map(|i| {
                    let shank = groups.get(i).map_or(0, |g| {
                        seen.iter().position(|s| *s == g).unwrap_or_else(|| {
                            seen.push(g);
                            seen.len() - 1
                        })
                    });
                    ChannelGeom { x_um: x[i] as f32, y_um: y[i] as f32, shank: shank as u32 }
                })
                .collect()
        }
        _ => {
            warnings.push("no electrode positions (rel_x, rel_y) in the NWB file: positions assumed (one column, 20 µm apart)".to_string());
            default_geom(n_ch)
        }
    };
    let sample_shift: Vec<f32> = match column("inter_sample_shift") {
        Some(s) => s.iter().map(|&v| v as f32).collect(),
        None => {
            warnings.push("no inter_sample_shift in the NWB file: no ADC sampling-delay correction available".to_string());
            vec![0.0; n_ch]
        }
    };

    Ok(Meta {
        n_saved_chans: n_ch,
        n_ap_chans: n_ch,
        sample_rate: series.rate,
        n_samples: series.n_samples,
        uv_per_bit,
        max_int: 32768.0,
        im_dat_prb_type: 0,
        channel_geom,
        channel_ids,
        sample_shift,
        warnings,
        reference_channels: BTreeSet::new(),
    })
}

/// The samples of an NWB recording, one stored chunk (along time) at a time.
pub struct NwbReader {
    path: PathBuf,
    data_path: String,
    n_ch: usize,
    bounds: Vec<usize>,
    cache: Mutex<ChunkCache>,
}

impl NwbReader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = open(path)?;
        let series = find_series(&file, path)?;
        let data_path = format!("{}/data", series.path);
        let dtype = file.dataset(&data_path).and_then(|d| d.dtype()).map_err(|e| anyhow::anyhow!("{e}"))?;
        if !matches!(dtype, hdf5_pure::DType::I16) {
            bail!("{}: samples of type {dtype:?} are not supported, only 16-bit integers", path.display());
        }
        let mut bounds: Vec<usize> = (0..series.n_samples).step_by(series.chunk_rows).collect();
        bounds.push(series.n_samples);
        Ok(Self { path: path.to_path_buf(), data_path, n_ch: series.n_ch, bounds, cache: Mutex::new(ChunkCache::new()) })
    }
}

impl ChunkSource for NwbReader {
    fn n_channels(&self) -> usize {
        self.n_ch
    }

    fn chunk_bounds(&self) -> &[usize] {
        &self.bounds
    }

    fn chunk(&self, idx: usize) -> Result<Arc<Vec<i16>>> {
        if let Some(c) = self.cache.lock().unwrap().get(idx) {
            return Ok(c);
        }
        if idx + 1 >= self.bounds.len() {
            bail!("chunk index out of bounds: {idx}");
        }
        let (lo, hi) = (self.bounds[idx], self.bounds[idx + 1]);
        // a handle per read: opening only parses the file's first block, and reads of
        // different chunks then run in parallel
        let file = open(&self.path)?;
        let bytes = file
            .dataset(&self.data_path)
            .and_then(|d| d.read_raw_rows(lo as u64, (hi - lo) as u64))
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", self.path.display()))?;
        if bytes.len() != (hi - lo) * self.n_ch * 2 {
            bail!("{}: read {} bytes, expected {}", self.path.display(), bytes.len(), (hi - lo) * self.n_ch * 2);
        }
        let data: Arc<Vec<i16>> = Arc::new(bytes.as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b)).collect());
        self.cache.lock().unwrap().insert(idx, Arc::clone(&data));
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NPX_NWB=<file.nwb> with `<file>.ref.bin` next to it (the series as raw int16):
    ///   NPX_NWB=... cargo test --release nwb_file -- --ignored --nocapture
    #[test]
    #[ignore]
    fn nwb_file_matches_reference() {
        let Ok(p) = std::env::var("NPX_NWB") else { return eprintln!("skipping: NPX_NWB not set") };
        let path = PathBuf::from(p);
        let meta = Meta::from_data_path(&path).unwrap();
        println!(
            "{} channels, {} samples at {} Hz, {} µV/bit, first ids {:?}, warnings {:?}",
            meta.n_ap_chans, meta.n_samples, meta.sample_rate, meta.uv_per_bit[0], &meta.channel_ids[..3], meta.warnings
        );
        let raw = crate::data::open_data(&path, &meta).unwrap();
        let want = std::fs::read(path.with_extension("ref.bin")).unwrap();
        let n_ch = meta.n_saved_chans;
        assert_eq!(want.len(), meta.n_samples * n_ch * 2);
        let chans: Vec<usize> = (0..n_ch).collect();
        let t0 = std::time::Instant::now();
        for first in (0..meta.n_samples).step_by(50_001) {
            let n = 50_001.min(meta.n_samples - first);
            let got = raw.read_i16(first, n, &meta, &chans);
            let same = got.iter().zip(want[first * n_ch * 2..].as_chunks::<2>().0).all(|(g, &w)| *g == i16::from_le_bytes(w));
            assert!(same, "samples from {first} differ");
        }
        println!("all samples equal, {:.2} s", t0.elapsed().as_secs_f64());
    }
}
