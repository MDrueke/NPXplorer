// Recordings saved by SpikeInterface as a Zarr v2 folder (`recording.save(format="zarr")`):
// `traces_seg0` holds the samples as a chunked, compressed [n_samples][n_channels] int16
// array, `channel_ids` and `properties/*` the per-channel values, and `.zattrs` the sample
// rate and (from SpikeInterface 0.10x on) the probe as a probeinterface description.
//
// Compressors read: Blosc (LZ4, Zstd, Zlib, BloscLZ), Zstd, GZip, Zlib, LZ4, WavPack
// (`wavpack-numcodecs`) and FLAC (`flac-numcodecs`), with an optional Delta filter.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::data::{default_geom, probe_type_of_spec, stream_delay_scale, ChannelGeom, ChunkSource, Meta};
use crate::mtscomp::ChunkCache;
use crate::probe;

const TRACES: &str = "traces_seg0";

/// The Zarr folder `path` points to or into (the folder itself, a file in it such as
/// `.zattrs`, or its `traces_seg0`), if it holds a SpikeInterface recording.
pub fn root_of(path: &Path) -> Option<PathBuf> {
    path.ancestors().take(3).find(|d| d.join(TRACES).join(".zarray").is_file()).map(Path::to_path_buf)
}

// ---------------------------------------------------------------------------
// Arrays
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Codec {
    None,
    Blosc,
    Zstd,
    Gzip,
    Zlib,
    Lz4,
    WavPack,
    Flac,
}

#[derive(Clone, Debug, PartialEq)]
enum Filter {
    /// numcodecs `Delta`: differences along the flattened chunk
    Delta,
    /// numcodecs `VLenUTF8`: an object array of strings
    VlenUtf8,
}

pub struct ZarrArray {
    dir: PathBuf,
    pub shape: Vec<usize>,
    chunks: Vec<usize>,
    /// numpy type string, e.g. "<i2"
    pub dtype: String,
    fortran_order: bool,
    separator: char,
    fill: f64,
    codec: Codec,
    filters: Vec<Filter>,
    /// WavPack in hybrid mode (`bps` set): the stored data is not the original
    lossy: bool,
}

impl ZarrArray {
    pub fn open(dir: &Path) -> Result<Self> {
        let meta_path = dir.join(".zarray");
        if !meta_path.is_file() && dir.join("zarr.json").is_file() {
            bail!("{} is a Zarr v3 array; only Zarr v2 is supported", dir.display());
        }
        let text = std::fs::read_to_string(&meta_path).with_context(|| format!("reading {}", meta_path.display()))?;
        let j: Value = serde_json::from_str(&sanitize_json(&text)).with_context(|| format!("parsing {}", meta_path.display()))?;
        let dims = |key: &str| -> Result<Vec<usize>> {
            j.get(key)
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
                .with_context(|| format!("{}: missing '{key}'", meta_path.display()))
        };
        let shape = dims("shape")?;
        let chunks = dims("chunks")?;
        if shape.len() != chunks.len() || chunks.iter().any(|&c| c == 0) {
            bail!("{}: shape {shape:?} and chunks {chunks:?} do not match", meta_path.display());
        }
        let dtype = j.get("dtype").and_then(Value::as_str).context("structured Zarr dtypes are not supported")?.to_string();
        let codec = match j.get("compressor") {
            None | Some(Value::Null) => Codec::None,
            Some(c) => match c.get("id").and_then(Value::as_str).unwrap_or("") {
                "blosc" => Codec::Blosc,
                "zstd" => Codec::Zstd,
                "gzip" => Codec::Gzip,
                "zlib" => Codec::Zlib,
                "lz4" => Codec::Lz4,
                "wavpack" => Codec::WavPack,
                "flac" => Codec::Flac,
                id => bail!("{}: compressor '{id}' is not supported", dir.display()),
            },
        };
        let mut filters = Vec::new();
        for f in j.get("filters").and_then(Value::as_array).map(|a| a.as_slice()).unwrap_or(&[]) {
            filters.push(match f.get("id").and_then(Value::as_str).unwrap_or("") {
                "delta" => {
                    let same = |k: &str| f.get(k).and_then(Value::as_str).is_none_or(|t| t == dtype);
                    if !same("dtype") || !same("astype") {
                        bail!("{}: a Delta filter that changes the data type is not supported", dir.display());
                    }
                    Filter::Delta
                }
                "vlen-utf8" => Filter::VlenUtf8,
                id => bail!("{}: filter '{id}' is not supported", dir.display()),
            });
        }
        let lossy = codec == Codec::WavPack
            && j.get("compressor").and_then(|c| c.get("bps")).and_then(Value::as_f64).is_some_and(|b| b > 0.0);
        Ok(Self {
            lossy,
            dir: dir.to_path_buf(),
            shape,
            chunks,
            dtype,
            fortran_order: j.get("order").and_then(Value::as_str) == Some("F"),
            separator: if j.get("dimension_separator").and_then(Value::as_str) == Some("/") { '/' } else { '.' },
            fill: j.get("fill_value").and_then(Value::as_f64).unwrap_or(0.0),
            codec,
            filters,
        })
    }

    /// Bytes per element; 0 for variable-length (object) arrays.
    fn item_size(&self) -> usize {
        let digits = self.dtype.trim_start_matches(|c: char| !c.is_ascii_digit());
        let n: usize = digits.parse().unwrap_or(0);
        match self.dtype.chars().nth(1) {
            Some('U') => n * 4,
            Some('O') => 0,
            _ => n,
        }
    }

    /// Decoded contents of the chunk at grid position `idx` (always a whole chunk, also
    /// at the edge of the array); `None` if the chunk was never written (all fill value).
    fn chunk_bytes(&self, idx: &[usize]) -> Result<Option<Vec<u8>>> {
        let name: Vec<String> = idx.iter().map(|i| i.to_string()).collect();
        let path = self.dir.join(name.join(&self.separator.to_string()));
        let raw = match std::fs::read(&path) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        self.decode(&raw).with_context(|| format!("decoding {}", path.display())).map(Some)
    }

    fn decode(&self, raw: &[u8]) -> Result<Vec<u8>> {
        let mut out = match self.codec {
            Codec::None => raw.to_vec(),
            Codec::Blosc => crate::blosc::decompress(raw)?,
            Codec::Zstd => {
                let mut v = Vec::new();
                ruzstd::decoding::StreamingDecoder::new(raw).map_err(|e| anyhow::anyhow!("Zstd: {e}"))?.read_to_end(&mut v)?;
                v
            }
            Codec::Gzip => {
                let mut v = Vec::new();
                flate2::read::MultiGzDecoder::new(raw).read_to_end(&mut v)?;
                v
            }
            Codec::Zlib => {
                let mut v = Vec::new();
                flate2::read::ZlibDecoder::new(raw).read_to_end(&mut v)?;
                v
            }
            // numcodecs: the uncompressed size (u32), then one LZ4 block
            Codec::Lz4 => lz4_flex::block::decompress_size_prepended(raw).map_err(|e| anyhow::anyhow!("LZ4: {e}"))?,
            Codec::WavPack => self.samples_to_bytes(decode_wavpack(raw)?)?,
            Codec::Flac => self.samples_to_bytes(decode_flac(raw)?)?,
        };
        for f in self.filters.iter().rev() {
            if *f == Filter::Delta {
                undo_delta(&mut out, self.item_size())?;
            }
        }
        Ok(out)
    }

    /// Audio codecs return samples: store them back as the array's integer type.
    fn samples_to_bytes(&self, samples: Vec<i32>) -> Result<Vec<u8>> {
        match self.dtype.as_str() {
            "<i2" => Ok(samples.iter().flat_map(|&s| (s as i16).to_le_bytes()).collect()),
            "<i4" => Ok(samples.iter().flat_map(|&s| s.to_le_bytes()).collect()),
            "|i1" => Ok(samples.iter().map(|&s| s as i8 as u8).collect()),
            t => bail!("WavPack/FLAC data of type '{t}' is not supported"),
        }
    }

    /// The whole array in C order (for the small per-channel arrays).
    fn read_all(&self) -> Result<Vec<u8>> {
        let item = self.item_size();
        if item == 0 || self.shape.len() > 2 {
            bail!("{}: arrays of type '{}' with {} dimensions are not supported", self.dir.display(), self.dtype, self.shape.len());
        }
        let (rows, cols) = (self.shape.first().copied().unwrap_or(1), self.shape.get(1).copied().unwrap_or(1));
        let (crows, ccols) = (self.chunks.first().copied().unwrap_or(1), self.chunks.get(1).copied().unwrap_or(1));
        let mut out = vec![0u8; rows * cols * item];
        for ci in 0..rows.div_ceil(crows) {
            for cj in 0..cols.div_ceil(ccols) {
                let idx = if self.shape.len() == 2 { vec![ci, cj] } else { vec![ci] };
                let Some(chunk) = self.chunk_bytes(&idx)? else { continue };
                if chunk.len() < crows * ccols * item {
                    bail!("{}: chunk {idx:?} is too short", self.dir.display());
                }
                for r in 0..crows.min(rows - ci * crows) {
                    for c in 0..ccols.min(cols - cj * ccols) {
                        let src = if self.fortran_order { c * crows + r } else { r * ccols + c } * item;
                        let dst = ((ci * crows + r) * cols + cj * ccols + c) * item;
                        out[dst..dst + item].copy_from_slice(&chunk[src..src + item]);
                    }
                }
            }
        }
        Ok(out)
    }

    /// All elements as numbers (integer, unsigned, float or bool arrays).
    pub fn read_f64(&self) -> Result<Vec<f64>> {
        let bytes = self.read_all()?;
        let item = self.item_size();
        let big = self.dtype.starts_with('>');
        let kind = self.dtype.chars().nth(1).unwrap_or('?');
        bytes
            .chunks_exact(item)
            .map(|b| {
                let mut le = [0u8; 8];
                le[..item.min(8)].copy_from_slice(&b[..item.min(8)]);
                if big {
                    le[..item].reverse();
                }
                Ok(match (kind, item) {
                    ('f', 8) => f64::from_le_bytes(le),
                    ('f', 4) => f32::from_le_bytes([le[0], le[1], le[2], le[3]]) as f64,
                    ('i', 1) => le[0] as i8 as f64,
                    ('i', 2) => i16::from_le_bytes([le[0], le[1]]) as f64,
                    ('i', 4) => i32::from_le_bytes([le[0], le[1], le[2], le[3]]) as f64,
                    ('i', 8) => i64::from_le_bytes(le) as f64,
                    ('u' | 'b', 1..=8) => u64::from_le_bytes(le) as f64,
                    _ => bail!("{}: type '{}' is not numeric", self.dir.display(), self.dtype),
                })
            })
            .collect()
    }

    /// All elements as text: fixed-width (`<U`, `|S`), variable-length UTF-8, or numbers.
    pub fn read_strings(&self) -> Result<Vec<String>> {
        match self.dtype.chars().nth(1) {
            Some('U') => {
                let item = self.item_size();
                let big = self.dtype.starts_with('>');
                Ok(self
                    .read_all()?
                    .chunks_exact(item.max(4))
                    .map(|b| {
                        b.chunks_exact(4)
                            .map(|c| if big { u32::from_be_bytes([c[0], c[1], c[2], c[3]]) } else { u32::from_le_bytes([c[0], c[1], c[2], c[3]]) })
                            .take_while(|&c| c != 0)
                            .filter_map(char::from_u32)
                            .collect()
                    })
                    .collect())
            }
            Some('S') => Ok(self
                .read_all()?
                .chunks_exact(self.item_size().max(1))
                .map(|b| String::from_utf8_lossy(b).trim_end_matches('\0').to_string())
                .collect()),
            Some('O') if self.filters.contains(&Filter::VlenUtf8) && self.shape.len() == 1 => {
                let mut out = Vec::new();
                for ci in 0..self.shape[0].div_ceil(self.chunks[0]) {
                    let chunk = self.chunk_bytes(&[ci])?.unwrap_or_default();
                    let mut pos = 4; // item count
                    while pos + 4 <= chunk.len() && out.len() < self.shape[0] {
                        let n = u32::from_le_bytes([chunk[pos], chunk[pos + 1], chunk[pos + 2], chunk[pos + 3]]) as usize;
                        let s = chunk.get(pos + 4..pos + 4 + n).context("corrupt string array")?;
                        out.push(String::from_utf8_lossy(s).into_owned());
                        pos += 4 + n;
                    }
                }
                out.resize(self.shape[0], String::new());
                Ok(out)
            }
            _ => Ok(self.read_f64()?.iter().map(|v| if v.fract() == 0.0 { format!("{}", *v as i64) } else { v.to_string() }).collect()),
        }
    }
}

/// numcodecs `Delta`: element 0 as it is, then the difference to the previous element.
fn undo_delta(bytes: &mut [u8], item: usize) -> Result<()> {
    match item {
        1 => {
            for i in 1..bytes.len() {
                bytes[i] = bytes[i].wrapping_add(bytes[i - 1]);
            }
        }
        2 => {
            let mut acc = 0i16;
            for b in bytes.chunks_exact_mut(2) {
                acc = acc.wrapping_add(i16::from_le_bytes([b[0], b[1]]));
                b.copy_from_slice(&acc.to_le_bytes());
            }
        }
        4 => {
            let mut acc = 0i32;
            for b in bytes.chunks_exact_mut(4) {
                acc = acc.wrapping_add(i32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                b.copy_from_slice(&acc.to_le_bytes());
            }
        }
        n => bail!("Delta filter on {n}-byte elements is not supported"),
    }
    Ok(())
}

/// A chunk written by `wavpack-numcodecs`: a complete WavPack file whose channels are
/// the columns of the chunk (or one channel holding the flattened chunk). Returns the
/// samples interleaved, which is the chunk in C order either way. A file with more than
/// two channels is a sequence of mono and stereo blocks per stretch of samples.
fn decode_wavpack(raw: &[u8]) -> Result<Vec<i32>> {
    let err = |e: oxideav_wavpack::Error| anyhow::anyhow!("WavPack: {e}");
    let mut out = Vec::new();
    // channels of the stretch being collected, one Vec per channel
    let mut set: Vec<Vec<i32>> = Vec::new();
    let flush = |set: &mut Vec<Vec<i32>>, out: &mut Vec<i32>| -> Result<()> {
        let n = set.first().map_or(0, Vec::len);
        if set.iter().any(|c| c.len() != n) {
            bail!("WavPack: channels of one block set differ in length");
        }
        out.reserve(n * set.len());
        for t in 0..n {
            out.extend(set.iter().map(|c| c[t]));
        }
        set.clear();
        Ok(())
    };
    let mut rest = raw;
    while rest.len() >= 32 {
        let (block, tail) = oxideav_wavpack::parse_block(rest).map_err(err)?;
        rest = tail;
        let n = block.header.block_samples as usize;
        if n == 0 {
            continue; // metadata only
        }
        if block.header.flags.is_first_block() {
            flush(&mut set, &mut out)?;
        }
        let samples = block.decode_member_samples().map_err(err)?;
        let n_ch = samples.len() / n;
        if n_ch == 0 || samples.len() != n * n_ch {
            bail!("WavPack: block of {n} samples decoded to {} values", samples.len());
        }
        for c in 0..n_ch {
            set.push(samples.iter().skip(c).step_by(n_ch).copied().collect());
        }
    }
    flush(&mut set, &mut out)?;
    Ok(out)
}

/// A chunk written by `flac-numcodecs`: a complete FLAC file with one channel (the
/// flattened chunk) or two (a two-column chunk). Interleaved, that is the chunk in C order.
fn decode_flac(raw: &[u8]) -> Result<Vec<i32>> {
    let mut reader = claxon::FlacReader::new(std::io::Cursor::new(raw)).map_err(|e| anyhow::anyhow!("FLAC: {e}"))?;
    reader.samples().collect::<std::result::Result<Vec<i32>, _>>().map_err(|e| anyhow::anyhow!("FLAC: {e}"))
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

/// The samples of a Zarr recording, one time chunk at a time.
pub struct ZarrReader {
    traces: ZarrArray,
    bounds: Vec<usize>,
    cache: Mutex<ChunkCache>,
}

impl ZarrReader {
    pub fn open(root: &Path) -> Result<Self> {
        let traces = ZarrArray::open(&root.join(TRACES))?;
        if traces.shape.len() != 2 {
            bail!("{}: expected a [samples][channels] array, found shape {:?}", traces.dir.display(), traces.shape);
        }
        if traces.dtype != "<i2" {
            bail!(
                "{}: samples of type '{}' are not supported, only 16-bit integers ('<i2'). \
                 Save the recording without scaling or filtering it.",
                traces.dir.display(),
                traces.dtype
            );
        }
        let (n, step) = (traces.shape[0], traces.chunks[0]);
        let mut bounds: Vec<usize> = (0..n).step_by(step).collect();
        bounds.push(n);
        let reader = Self { traces, bounds, cache: Mutex::new(ChunkCache::new()) };
        if n > 0 {
            // an unreadable recording fails here, with the reason, instead of showing zeros
            reader.decode_chunk(0)?;
        }
        Ok(reader)
    }

    fn decode_chunk(&self, idx: usize) -> Result<Vec<i16>> {
        let n_ch = self.traces.shape[1];
        let (ct, cw) = (self.traces.chunks[0], self.traces.chunks[1]);
        let nt = self.bounds[idx + 1] - self.bounds[idx];
        let mut out = vec![self.traces.fill as i16; nt * n_ch];
        for j in 0..n_ch.div_ceil(cw) {
            let Some(bytes) = self.traces.chunk_bytes(&[idx, j])? else { continue };
            if bytes.len() < ct * cw * 2 {
                bail!("chunk {idx}.{j} holds {} bytes, expected {}", bytes.len(), ct * cw * 2);
            }
            let w = cw.min(n_ch - j * cw);
            for t in 0..nt {
                let row = &mut out[t * n_ch + j * cw..t * n_ch + j * cw + w];
                for (c, v) in row.iter_mut().enumerate() {
                    let at = if self.traces.fortran_order { c * ct + t } else { t * cw + c } * 2;
                    *v = i16::from_le_bytes([bytes[at], bytes[at + 1]]);
                }
            }
        }
        Ok(out)
    }
}

impl ChunkSource for ZarrReader {
    fn n_channels(&self) -> usize {
        self.traces.shape[1]
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
        let data = Arc::new(self.decode_chunk(idx)?);
        self.cache.lock().unwrap().insert(idx, Arc::clone(&data));
        Ok(data)
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Python writes NaN and Infinity as bare words, which is not JSON: replace them
/// (outside strings) with null.
fn sanitize_json(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let (mut in_str, mut escaped) = (false, false);
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if in_str {
            in_str = escaped || c != '"';
            escaped = !escaped && c == '\\';
        } else if c == '"' {
            in_str = true;
        } else if let Some(word) = ["-Infinity", "Infinity", "NaN"].iter().find(|w| rest.starts_with(**w)) {
            out.push_str("null");
            rest = &rest[word.len()..];
            continue;
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

fn property(root: &Path, name: &str) -> Option<ZarrArray> {
    let dir = root.join("properties").join(name);
    dir.join(".zarray").is_file().then(|| ZarrArray::open(&dir).ok()).flatten()
}

/// Positions, shanks, sampling order and part number from a probeinterface description.
struct ProbeInfo {
    /// per recording channel; `None` = no contact (a reference or unconnected channel)
    geom: Vec<Option<ChannelGeom>>,
    adc_order: Vec<Option<f64>>,
    part_number: Option<String>,
    channels_per_adc: Option<usize>,
}

fn parse_probegroup(pg: &Value, n_ch: usize) -> Option<ProbeInfo> {
    let probes = pg.get("probes")?.as_array()?;
    let mut info = ProbeInfo { geom: vec![None; n_ch], adc_order: vec![None; n_ch], part_number: None, channels_per_adc: None };
    let mut shank_base = 0u32;
    for p in probes {
        let pos = p.get("contact_positions")?.as_array()?;
        let dev = p.get("device_channel_indices").and_then(Value::as_array);
        let shanks = p.get("shank_ids").and_then(Value::as_array);
        let order = p.get("contact_annotations").and_then(|a| a.get("adc_sample_order")).and_then(Value::as_array);
        let ann = p.get("annotations");
        if info.part_number.is_none() {
            info.part_number = ann.and_then(|a| a.get("part_number").or_else(|| a.get("model_name"))).and_then(Value::as_str).map(str::to_string);
            info.channels_per_adc = ann.and_then(|a| a.get("num_channels_per_adc")).and_then(Value::as_u64).map(|v| v as usize);
        }
        let mut max_shank = 0u32;
        for (i, xy) in pos.iter().enumerate() {
            // without device indices, contact i is channel i (counted over all probes)
            let ch = match dev {
                Some(d) => d.get(i).and_then(Value::as_i64).unwrap_or(-1),
                None => i as i64,
            };
            if ch < 0 || ch as usize >= n_ch {
                continue;
            }
            let shank = shanks
                .and_then(|s| s.get(i))
                .and_then(|s| s.as_str().map(|t| t.trim().parse::<u32>().unwrap_or(0)).or(s.as_u64().map(|v| v as u32)))
                .unwrap_or(0);
            max_shank = max_shank.max(shank);
            let coord = |k: usize| xy.get(k).and_then(Value::as_f64).unwrap_or(0.0) as f32;
            info.geom[ch as usize] = Some(ChannelGeom { x_um: coord(0), y_um: coord(1), shank: shank_base + shank });
            info.adc_order[ch as usize] = order.and_then(|o| o.get(i)).and_then(Value::as_f64);
        }
        shank_base += max_shank + 1;
    }
    info.geom.iter().any(Option::is_some).then_some(info)
}

/// Metadata of the recording in the Zarr folder `root`.
pub fn load_meta(root: &Path) -> Result<Meta> {
    let attrs_path = root.join(".zattrs");
    let attrs_text = std::fs::read_to_string(&attrs_path).with_context(|| format!("reading {}", attrs_path.display()))?;
    let attrs: Value = serde_json::from_str(&sanitize_json(&attrs_text)).with_context(|| format!("parsing {}", attrs_path.display()))?;
    let sample_rate = attrs
        .get("sampling_frequency")
        .and_then(Value::as_f64)
        .with_context(|| format!("{}: no 'sampling_frequency' (not a SpikeInterface recording?)", attrs_path.display()))?;

    let traces = ZarrArray::open(&root.join(TRACES))?;
    if traces.shape.len() != 2 {
        bail!("{}: expected a [samples][channels] array, found shape {:?}", root.join(TRACES).display(), traces.shape);
    }
    let (n_samples, n_ch) = (traces.shape[0], traces.shape[1]);
    let mut warnings = Vec::new();
    if traces.lossy {
        warnings.push("lossy WavPack: decoded values can differ by a few ADC counts from the WavPack reference decoder".to_string());
    }
    if attrs.get("num_segments").and_then(Value::as_u64).unwrap_or(1) > 1 {
        warnings.push("the Zarr folder holds several segments: only the first is shown".to_string());
    }

    // channel names: SpikeInterface ids such as "CH13" or "imec0.ap#AP12"
    let ids = ZarrArray::open(&root.join("channel_ids")).and_then(|a| a.read_strings()).unwrap_or_default();
    let channel_ids: Vec<String> = (0..n_ch)
        .map(|i| match ids.get(i) {
            Some(id) if !id.is_empty() => id.rsplit('#').next().unwrap_or(id).to_string(),
            _ => i.to_string(),
        })
        .collect();

    let per_channel = |name: &str| property(root, name).and_then(|a| a.read_f64().ok()).filter(|v| v.len() >= n_ch);
    let uv_per_bit: Vec<f32> = match per_channel("gain_to_uV") {
        Some(g) => g[..n_ch].iter().map(|&v| v as f32).collect(),
        None => {
            warnings.push("no gain_to_uV in the Zarr folder: values are raw ADC counts, not µV".to_string());
            vec![1.0; n_ch]
        }
    };
    if per_channel("offset_to_uV").is_some_and(|o| o[..n_ch].iter().any(|&v| v != 0.0)) {
        warnings.push("the Zarr folder's offset_to_uV is not 0 and is ignored".to_string());
    }

    // geometry: the stored probe, else the location/group properties of older versions
    let probe_info = attrs.get("probegroup").and_then(|pg| parse_probegroup(pg, n_ch));
    let mut reference_channels = BTreeSet::new();
    let channel_geom: Vec<ChannelGeom> = if let Some(info) = &probe_info {
        // a channel without a contact carries no neural signal; it is drawn at the end
        let top = info.geom.iter().flatten().map(|g| g.y_um).fold(0.0f32, f32::max);
        let mut extra = 0.0;
        info.geom
            .iter()
            .enumerate()
            .map(|(i, g)| {
                g.clone().unwrap_or_else(|| {
                    reference_channels.insert(i);
                    extra += 20.0;
                    ChannelGeom { x_um: 0.0, y_um: top + extra, shank: 0 }
                })
            })
            .collect()
    } else if let Some(loc) = property(root, "location").and_then(|a| a.read_f64().ok()).filter(|v| v.len() >= n_ch * 2) {
        let dims = loc.len() / n_ch;
        let group = per_channel("group");
        // groups are shanks only if they are numbered from 0
        let shank = |i: usize| group.as_ref().map(|g| g[i]).filter(|&v| v >= 0.0 && v < 64.0).map_or(0, |v| v as u32);
        (0..n_ch).map(|i| ChannelGeom { x_um: loc[i * dims] as f32, y_um: loc[i * dims + 1] as f32, shank: shank(i) }).collect()
    } else {
        warnings.push("no probe in the Zarr folder: electrode positions assumed (one column, 20 µm apart)".to_string());
        default_geom(n_ch)
    };
    if reference_channels.len() == n_ch {
        reference_channels.clear();
    }

    let spec = probe_info.as_ref().and_then(|p| p.part_number.as_deref()).and_then(probe::spec_for_part_number);
    let im_dat_prb_type = spec.as_ref().map_or(0, probe_type_of_spec);

    // ADC sampling delay: SpikeInterface's inter_sample_shift (a fraction of this
    // stream's sample period), else from the probe's sampling order
    let sample_shift: Vec<f32> = if let Some(s) = per_channel("inter_sample_shift") {
        s[..n_ch].iter().map(|&v| v as f32).collect()
    } else if let (Some(info), Some(spec)) = (&probe_info, &spec) {
        let cycles = spec.mux.n_cycles(info.channels_per_adc.unwrap_or(0));
        let scale = stream_delay_scale(sample_rate);
        if cycles > 1.0 && info.adc_order.iter().any(Option::is_some) && spec.mux != probe::MuxFamily::None {
            info.adc_order.iter().map(|o| (o.unwrap_or(0.0) / cycles) as f32 * scale).collect()
        } else {
            warnings.push("no ADC sampling order in the Zarr folder: no sampling-delay correction available".to_string());
            vec![0.0; n_ch]
        }
    } else {
        warnings.push("no probe type in the Zarr folder: no ADC sampling-delay correction available".to_string());
        vec![0.0; n_ch]
    };

    Ok(Meta {
        n_saved_chans: n_ch,
        n_ap_chans: n_ch,
        sample_rate,
        n_samples,
        uv_per_bit,
        max_int: 32768.0,
        im_dat_prb_type,
        channel_geom,
        channel_ids,
        sample_shift,
        warnings,
        reference_channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_with_python_nan() {
        let s = sanitize_json(r#"{"a": NaN, "b": "NaN \" Infinity", "c": [-Infinity, 1.5, Infinity]}"#);
        let v: Value = serde_json::from_str(&s).unwrap();
        assert!(v["a"].is_null() && v["c"][0].is_null() && v["c"][2].is_null());
        assert_eq!(v["b"], "NaN \" Infinity");
        assert_eq!(v["c"][1], 1.5);
    }

    #[test]
    fn delta_filter() {
        let vals: [i16; 5] = [100, -3, 7, 32767, -32768];
        let mut enc: Vec<u8> = Vec::new();
        let mut prev = 0i16;
        for v in vals {
            enc.extend_from_slice(&v.wrapping_sub(prev).to_le_bytes());
            prev = v;
        }
        undo_delta(&mut enc, 2).unwrap();
        let back: Vec<i16> = enc.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
        assert_eq!(back, vals);
    }

    /// Every `<name>.zarr` in the folder NPX_ZARR_DIR, read through `open_data` and
    /// compared with `<name>.ref.bin` (the same recording as raw int16, as decoded by
    /// SpikeInterface). `<name>` containing "lossy" allows a difference of NPX_ZARR_TOL
    /// counts (default 0).
    ///   NPX_ZARR_DIR=... cargo test --release zarr_folders -- --ignored --nocapture
    #[test]
    #[ignore]
    fn zarr_folders_match_reference() {
        let Ok(dir) = std::env::var("NPX_ZARR_DIR") else { return eprintln!("skipping: NPX_ZARR_DIR not set") };
        let tol: i32 = std::env::var("NPX_ZARR_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut names: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "zarr")).collect();
        names.sort();
        let mut failed = Vec::new();
        for root in names {
            let name = root.file_stem().unwrap().to_string_lossy().into_owned();
            let t0 = std::time::Instant::now();
            let result = (|| -> Result<(usize, i32, usize)> {
                let meta = Meta::from_data_path(&root)?;
                let raw = crate::data::open_data(&root, &meta)?;
                let want = std::fs::read(root.with_extension("ref.bin"))?;
                let n_ch = meta.n_saved_chans;
                if want.len() != meta.n_samples * n_ch * 2 {
                    bail!("reference holds {} bytes, expected {}", want.len(), meta.n_samples * n_ch * 2);
                }
                let chans: Vec<usize> = (0..n_ch).collect();
                let (mut max_diff, mut n_diff) = (0i32, 0usize);
                // odd block length, so reads cross chunk borders
                for first in (0..meta.n_samples).step_by(7001) {
                    let n = 7001.min(meta.n_samples - first);
                    let got = raw.read_i16(first, n, &meta, &chans);
                    for (g, w) in got.iter().zip(want[first * n_ch * 2..].chunks_exact(2)) {
                        let d = (*g as i32 - i16::from_le_bytes([w[0], w[1]]) as i32).abs();
                        max_diff = max_diff.max(d);
                        n_diff += (d != 0) as usize;
                    }
                }
                Ok((meta.n_samples * n_ch, max_diff, n_diff))
            })();
            match result {
                Ok((n, max_diff, n_diff)) => {
                    let allowed = if name.contains("lossy") { tol } else { 0 };
                    println!("{name}: {n} values, {n_diff} differ, max difference {max_diff}, {:.2} s", t0.elapsed().as_secs_f64());
                    if max_diff > allowed {
                        failed.push(name);
                    }
                }
                Err(e) => {
                    println!("{name}: ERROR {e:#}");
                    failed.push(name);
                }
            }
        }
        assert!(failed.is_empty(), "failed: {failed:?}");
    }

    /// A small Zarr recording written by hand: uncompressed, two channel chunks, a
    /// missing chunk, a probe with one unconnected channel.
    #[test]
    fn hand_written_recording() {
        let root = std::env::temp_dir().join(format!("npx_zarr_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let traces = root.join(TRACES);
        std::fs::create_dir_all(&traces).unwrap();
        std::fs::create_dir_all(root.join("properties/gain_to_uV")).unwrap();
        std::fs::create_dir_all(root.join("channel_ids")).unwrap();
        let (n_t, n_ch) = (5usize, 3usize);
        std::fs::write(
            root.join(".zattrs"),
            r#"{"sampling_frequency": 30000.0, "num_segments": 1, "probegroup": {"probes": [{
                "contact_positions": [[16.0, 0.0], [48.0, 20.0]], "device_channel_indices": [0, 2],
                "shank_ids": ["0", "0"], "annotations": {"part_number": "NP1000", "num_channels_per_adc": 12},
                "contact_annotations": {"adc_sample_order": [0, 3]}}]}}"#,
        )
        .unwrap();
        std::fs::write(
            traces.join(".zarray"),
            format!(r#"{{"shape": [{n_t}, {n_ch}], "chunks": [2, 2], "dtype": "<i2", "compressor": null, "filters": null, "order": "C", "fill_value": 0, "zarr_format": 2}}"#),
        )
        .unwrap();
        let value = |t: usize, c: usize| (t * 10 + c) as i16;
        for ct in 0..3 {
            for cc in 0..2 {
                if (ct, cc) == (1, 1) {
                    continue; // never written: fill value
                }
                let mut b = Vec::new();
                for t in 0..2 {
                    for c in 0..2 {
                        b.extend_from_slice(&value(ct * 2 + t, cc * 2 + c).to_le_bytes());
                    }
                }
                std::fs::write(traces.join(format!("{ct}.{cc}")), b).unwrap();
            }
        }
        std::fs::write(
            root.join("properties/gain_to_uV/.zarray"),
            r#"{"shape": [3], "chunks": [3], "dtype": "<f8", "compressor": null, "filters": null, "order": "C", "fill_value": 0.0, "zarr_format": 2}"#,
        )
        .unwrap();
        std::fs::write(root.join("properties/gain_to_uV/0"), [2.34f64, 2.34, 2.34].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        std::fs::write(
            root.join("channel_ids/.zarray"),
            r#"{"shape": [3], "chunks": [3], "dtype": "<U4", "compressor": null, "filters": null, "order": "C", "fill_value": "", "zarr_format": 2}"#,
        )
        .unwrap();
        let mut ids = Vec::new();
        for name in ["AP0", "AP1", "s#C2"] {
            let mut cp: Vec<u32> = name.chars().map(|c| c as u32).collect();
            cp.resize(4, 0);
            ids.extend(cp.iter().flat_map(|c| c.to_le_bytes()));
        }
        std::fs::write(root.join("channel_ids/0"), ids).unwrap();

        assert_eq!(root_of(&root.join(".zattrs")).as_deref(), Some(root.as_path()));
        assert_eq!(root_of(&traces).as_deref(), Some(root.as_path()));
        let meta = load_meta(&root).unwrap();
        assert_eq!((meta.n_samples, meta.n_ap_chans, meta.sample_rate), (n_t, n_ch, 30000.0));
        assert_eq!(meta.channel_ids, ["AP0", "AP1", "C2"]);
        assert_eq!(meta.uv_per_bit, [2.34f32; 3]);
        assert_eq!(meta.reference_channels, BTreeSet::from([1]));
        assert_eq!((meta.channel_geom[2].x_um, meta.channel_geom[2].y_um), (48.0, 20.0));
        assert!((meta.sample_shift[2] - 3.0 / 13.0).abs() < 1e-6 && meta.sample_shift[0] == 0.0);

        let reader = ZarrReader::open(&root).unwrap();
        assert_eq!(reader.chunk_bounds(), [0, 2, 4, 5]);
        for (ci, w) in reader.chunk_bounds().windows(2).enumerate() {
            let chunk = reader.chunk(ci).unwrap();
            assert_eq!(chunk.len(), (w[1] - w[0]) * n_ch);
            for t in w[0]..w[1] {
                for c in 0..n_ch {
                    let want = if ci == 1 && c >= 2 { 0 } else { value(t, c) };
                    assert_eq!(chunk[(t - w[0]) * n_ch + c], want, "t={t} c={c}");
                }
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}
