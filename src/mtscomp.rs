use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use flate2::read::ZlibDecoder;

/// Memory budget for decompressed chunks kept around for reuse (bytes). A 1 s chunk
/// of a 385-channel AP file is ~23 MB, so this holds ~10 of them — enough for the
/// chunks around the current view plus those shared by nearby PSTH windows.
const CACHE_BUDGET_BYTES: usize = 256 * 1024 * 1024;

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

/// Small LRU of decompressed chunks (most recently used at the back).
struct ChunkCache {
    entries: VecDeque<(usize, Arc<Vec<i16>>)>,
    bytes: usize,
}

impl ChunkCache {
    fn get(&mut self, idx: usize) -> Option<Arc<Vec<i16>>> {
        let pos = self.entries.iter().position(|(i, _)| *i == idx)?;
        let entry = self.entries.remove(pos)?;
        let data = Arc::clone(&entry.1);
        self.entries.push_back(entry);
        Some(data)
    }

    fn insert(&mut self, idx: usize, data: Arc<Vec<i16>>) {
        if self.entries.iter().any(|(i, _)| *i == idx) {
            return; // another thread decompressed the same chunk meanwhile
        }
        let size = data.len() * 2;
        if size > CACHE_BUDGET_BYTES {
            return;
        }
        while self.bytes + size > CACHE_BUDGET_BYTES {
            match self.entries.pop_front() {
                Some((_, old)) => self.bytes -= old.len() * 2,
                None => break,
            }
        }
        self.bytes += size;
        self.entries.push_back((idx, data));
    }
}

pub struct MtscompReader {
    pub meta: MtscompMeta,
    file: File,
    cache: Mutex<ChunkCache>,
}

/// Read exactly `buf.len()` bytes at `offset` without moving a shared file cursor, so
/// several threads can read different chunks at once.
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0usize;
        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
        Ok(())
    }
}

impl MtscompReader {
    pub fn new(cbin_path: &Path, meta: MtscompMeta) -> Result<Self> {
        let file = File::open(cbin_path)
            .with_context(|| format!("opening cbin file: {}", cbin_path.display()))?;
        Ok(Self {
            meta,
            file,
            cache: Mutex::new(ChunkCache { entries: VecDeque::new(), bytes: 0 }),
        })
    }

    /// Decompressed chunk `chunk_idx` (interleaved `[t][ch]`), from the cache when it
    /// was used recently.
    pub fn chunk(&self, chunk_idx: usize) -> Result<Arc<Vec<i16>>> {
        if let Some(c) = self.cache.lock().unwrap().get(chunk_idx) {
            return Ok(c);
        }
        // decompress without holding the lock, so other chunks proceed in parallel
        let data = Arc::new(self.decompress_chunk(chunk_idx)?);
        self.cache.lock().unwrap().insert(chunk_idx, Arc::clone(&data));
        Ok(data)
    }

    pub fn decompress_chunk(&self, chunk_idx: usize) -> Result<Vec<i16>> {
        if chunk_idx >= self.meta.chunk_bounds.len().saturating_sub(1)
            || chunk_idx + 1 >= self.meta.chunk_offsets.len()
        {
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
        read_exact_at(&self.file, &mut comp_buf, start_offset)?;

        let mut decoder = ZlibDecoder::new(&comp_buf[..]);
        let mut decomp_buf = Vec::with_capacity(expected_items * 2);
        decoder.read_to_end(&mut decomp_buf)?;

        if decomp_buf.len() != expected_items * 2 {
            bail!("Decompressed size mismatch: got {}, expected {}", decomp_buf.len(), expected_items * 2);
        }

        // bytes -> i16 (the Vec<u8> may not be 2-byte aligned, so copy via LE decode)
        let raw_i16: Vec<i16> = decomp_buf
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();

        // 1. Un-transpose if necessary (F-order -> C-order)
        let mut out = if self.meta.chunk_order == "F" {
            let mut out = vec![0i16; expected_items];
            for ch in 0..n_channels {
                for t in 0..n_samples_chunk {
                    out[t * n_channels + ch] = raw_i16[ch * n_samples_chunk + t];
                }
            }
            out
        } else {
            raw_i16
        };

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

        // 3. Reverse time diff (axis 0) — row by row, so the inner loop is contiguous
        if self.meta.do_time_diff && n_samples_chunk > 1 {
            for t in 1..n_samples_chunk {
                let (prev, cur) = out.split_at_mut(t * n_channels);
                let prev = &prev[(t - 1) * n_channels..];
                for (c, p) in cur[..n_channels].iter_mut().zip(prev) {
                    *c = c.wrapping_add(*p);
                }
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_evicts_oldest_and_refreshes_on_hit() {
        let mut c = ChunkCache { entries: VecDeque::new(), bytes: 0 };
        let chunk = |v: i16| Arc::new(vec![v; CACHE_BUDGET_BYTES / 2 / 3]); // 3 fit
        c.insert(0, chunk(0));
        c.insert(1, chunk(1));
        c.insert(2, chunk(2));
        assert!(c.get(0).is_some()); // 0 is now most recent
        c.insert(3, chunk(3)); // evicts 1
        assert!(c.get(1).is_none());
        assert!(c.get(0).is_some() && c.get(2).is_some() && c.get(3).is_some());
        assert!(c.bytes <= CACHE_BUDGET_BYTES);
    }
}
