use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use flate2::read::ZlibDecoder;
use std::sync::Mutex;

#[derive(Deserialize, Debug, Clone)]
pub struct MtscompMeta {
    pub chunk_bounds: Vec<usize>,
    pub chunk_offsets: Vec<u64>,
    pub chunk_order: String,
    pub do_spatial_diff: bool,
    pub do_time_diff: bool,
    pub dtype: String,
    pub n_channels: usize,
}

impl MtscompMeta {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading mtscomp metadata file: {}", path.display()))?;
        let meta: MtscompMeta = serde_json::from_str(&content)?;
        if meta.dtype != "int16" {
            bail!("Unsupported mtscomp dtype: {}", meta.dtype);
        }
        Ok(meta)
    }
}

pub struct MtscompReader {
    pub meta: MtscompMeta,
    file: Mutex<File>,
}

impl MtscompReader {
    pub fn new(cbin_path: &Path, meta: MtscompMeta) -> Result<Self> {
        let file = File::open(cbin_path)
            .with_context(|| format!("opening cbin file: {}", cbin_path.display()))?;
        Ok(Self {
            meta,
            file: Mutex::new(file),
        })
    }

    pub fn decompress_chunk(&self, chunk_idx: usize) -> Result<Vec<i16>> {
        if chunk_idx >= self.meta.chunk_bounds.len().saturating_sub(1) {
            bail!("Chunk index out of bounds: {}", chunk_idx);
        }

        let start_sample = self.meta.chunk_bounds[chunk_idx];
        let end_sample = self.meta.chunk_bounds[chunk_idx + 1];
        let n_samples_chunk = end_sample - start_sample;
        let n_channels = self.meta.n_channels;
        let expected_items = n_samples_chunk * n_channels;

        let start_offset = self.meta.chunk_offsets[chunk_idx];
        let end_offset = self.meta.chunk_offsets[chunk_idx + 1];
        let comp_len = (end_offset - start_offset) as usize;

        let mut comp_buf = vec![0u8; comp_len];
        {
            let mut file = self.file.lock().unwrap();
            file.seek(SeekFrom::Start(start_offset))?;
            file.read_exact(&mut comp_buf)?;
        }

        let mut decoder = ZlibDecoder::new(&comp_buf[..]);
        let mut decomp_buf = Vec::with_capacity(expected_items * 2);
        decoder.read_to_end(&mut decomp_buf)?;

        if decomp_buf.len() != expected_items * 2 {
            bail!("Decompressed size mismatch: got {}, expected {}", decomp_buf.len(), expected_items * 2);
        }

        // convert bytes to i16
        let raw_i16: &[i16] = bytemuck::cast_slice(&decomp_buf);

        let mut out = vec![0i16; expected_items];

        // 1. Un-transpose if necessary (F-order -> C-order)
        if self.meta.chunk_order == "F" {
            for ch in 0..n_channels {
                for t in 0..n_samples_chunk {
                    out[t * n_channels + ch] = raw_i16[ch * n_samples_chunk + t];
                }
            }
        } else {
            out.copy_from_slice(raw_i16);
        }

        // 2. Reverse spatial diff (axis 1)
        if self.meta.do_spatial_diff {
            for t in 0..n_samples_chunk {
                let row_start = t * n_channels;
                let mut acc = 0i16;
                for ch in 0..n_channels {
                    acc = acc.wrapping_add(out[row_start + ch]);
                    out[row_start + ch] = acc;
                }
            }
        }

        // 3. Reverse time diff (axis 0)
        if self.meta.do_time_diff {
            for ch in 0..n_channels {
                let mut acc = 0i16;
                for t in 0..n_samples_chunk {
                    let idx = t * n_channels + ch;
                    acc = acc.wrapping_add(out[idx]);
                    out[idx] = acc;
                }
            }
        }

        Ok(out)
    }
}
