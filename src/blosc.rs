// Decoder for Blosc 1 buffers, the default compressor of Zarr v2 arrays (numcodecs
// `Blosc`). A buffer is a 16-byte header, a table of block offsets, and blocks that are
// compressed one by one with LZ4, Zstd, Zlib or BloscLZ after an optional byte or bit
// shuffle. Snappy (dropped from Blosc long ago) is not supported.

use anyhow::{bail, Context, Result};
use std::io::Read;

const HEADER_LEN: usize = 16;
const FLAG_SHUFFLE: u8 = 0x01;
const FLAG_MEMCPYED: u8 = 0x02;
const FLAG_BITSHUFFLE: u8 = 0x04;
const FLAG_DONT_SPLIT: u8 = 0x10;
// a block is only split into one stream per byte of the type when it is big enough
const MAX_SPLITS: usize = 16;
const MIN_BUFFERSIZE: usize = 128;
const BLOSCLZ_MAX_DISTANCE: usize = 8191;

fn le_u32(b: &[u8], at: usize) -> Result<usize> {
    let s = b.get(at..at + 4).context("Blosc buffer is truncated")?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
}

/// Decompress one Blosc buffer.
pub fn decompress(src: &[u8]) -> Result<Vec<u8>> {
    if src.len() < HEADER_LEN {
        bail!("Blosc buffer is shorter than its header");
    }
    let flags = src[2];
    let typesize = (src[3] as usize).max(1);
    let nbytes = le_u32(src, 4)?;
    let blocksize = le_u32(src, 8)?;
    let cbytes = le_u32(src, 12)?;
    if cbytes > src.len() {
        bail!("Blosc buffer is truncated ({} of {cbytes} bytes)", src.len());
    }
    if flags & FLAG_MEMCPYED != 0 {
        return Ok(src.get(HEADER_LEN..HEADER_LEN + nbytes).context("Blosc buffer is truncated")?.to_vec());
    }
    if nbytes == 0 {
        return Ok(Vec::new());
    }
    if blocksize == 0 {
        bail!("Blosc buffer has block size 0");
    }
    let format = flags >> 5;
    let mut out = vec![0u8; nbytes];
    let mut tmp = vec![0u8; blocksize];
    for (b, dest) in out.chunks_mut(blocksize).enumerate() {
        let bsize = dest.len();
        let leftover = bsize != blocksize;
        let shuffled = (flags & FLAG_SHUFFLE != 0 && typesize > 1) || (flags & FLAG_BITSHUFFLE != 0);
        let mut pos = le_u32(src, HEADER_LEN + 4 * b)?;
        let split = flags & FLAG_DONT_SPLIT == 0
            && typesize <= MAX_SPLITS
            && bsize / typesize >= MIN_BUFFERSIZE
            && !leftover;
        let n_splits = if split { typesize } else { 1 };
        let neblock = bsize / n_splits;
        let target: &mut [u8] = if shuffled { &mut tmp[..bsize] } else { &mut dest[..] };
        for part in target.chunks_mut(neblock) {
            let clen = le_u32(src, pos)?;
            pos += 4;
            let comp = src.get(pos..pos + clen).context("Blosc block is truncated")?;
            pos += clen;
            if clen == part.len() {
                part.copy_from_slice(comp);
                continue;
            }
            let n = match format {
                0 => blosclz_decompress(comp, part)?,
                1 => lz4_flex::block::decompress_into(comp, part).map_err(|e| anyhow::anyhow!("LZ4: {e}"))?,
                3 => read_full(flate2::read::ZlibDecoder::new(comp), part)?,
                4 => {
                    let dec = ruzstd::decoding::StreamingDecoder::new(comp).map_err(|e| anyhow::anyhow!("Zstd: {e}"))?;
                    read_full(dec, part)?
                }
                2 => bail!("Blosc buffers compressed with Snappy are not supported"),
                f => bail!("unknown Blosc compressor format {f}"),
            };
            if n != part.len() {
                bail!("Blosc block decompressed to {n} bytes, expected {}", part.len());
            }
        }
        if flags & FLAG_BITSHUFFLE != 0 {
            bit_unshuffle(typesize, &tmp[..bsize], dest);
        } else if shuffled {
            byte_unshuffle(typesize, &tmp[..bsize], dest);
        }
    }
    Ok(out)
}

/// Fill `out` from `rd`; returns the number of bytes read (short if the stream ends).
fn read_full(mut rd: impl Read, out: &mut [u8]) -> Result<usize> {
    let mut n = 0;
    while n < out.len() {
        match rd.read(&mut out[n..])? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

/// Undo Blosc's byte shuffle: `src` holds byte 0 of every element, then byte 1, ...;
/// bytes beyond the last whole element are stored unchanged.
fn byte_unshuffle(typesize: usize, src: &[u8], dest: &mut [u8]) {
    let n = src.len() / typesize;
    for (j, plane) in src[..n * typesize].chunks_exact(n.max(1)).enumerate().take(typesize) {
        for (i, &v) in plane.iter().enumerate() {
            dest[i * typesize + j] = v;
        }
    }
    dest[n * typesize..].copy_from_slice(&src[n * typesize..]);
}

/// Undo Blosc's bit shuffle (the bitshuffle library's `bshuf_trans_bit_elem`): for byte
/// `j` of the element and bit `k`, `src` holds a row of `n/8` bytes with that bit of
/// every element, least significant bit first. Only a multiple of 8 elements is
/// shuffled; the rest of the block is stored unchanged.
fn bit_unshuffle(typesize: usize, src: &[u8], dest: &mut [u8]) {
    let n = (src.len() / typesize) & !7;
    let row = n / 8;
    dest[..n * typesize].fill(0);
    for j in 0..typesize {
        for k in 0..8 {
            let bits = &src[(j * 8 + k) * row..(j * 8 + k + 1) * row];
            for (i, &byte) in bits.iter().enumerate() {
                if byte == 0 {
                    continue;
                }
                for b in 0..8 {
                    dest[(i * 8 + b) * typesize + j] |= ((byte >> b) & 1) << k;
                }
            }
        }
    }
    dest[n * typesize..].copy_from_slice(&src[n * typesize..]);
}

/// BloscLZ (a FastLZ variant): literal runs and back references with a 13-bit distance,
/// extended to 16 bits for far matches.
fn blosclz_decompress(input: &[u8], out: &mut [u8]) -> Result<usize> {
    let bad = || anyhow::anyhow!("corrupt BloscLZ block");
    let mut ip = 0usize;
    let mut op = 0usize;
    let next = |ip: &mut usize| -> Result<usize> {
        let v = *input.get(*ip).ok_or_else(bad)? as usize;
        *ip += 1;
        Ok(v)
    };
    let mut ctrl = next(&mut ip)? & 31;
    loop {
        if ctrl >= 32 {
            let mut len = (ctrl >> 5) - 1;
            let mut ofs = (ctrl & 31) << 8;
            if len == 6 {
                loop {
                    let code = next(&mut ip)?;
                    len += code;
                    if code != 255 {
                        break;
                    }
                }
            }
            let code = next(&mut ip)?;
            len += 3;
            ofs += code;
            if code == 255 && ofs == (31 << 8) + 255 {
                ofs = (next(&mut ip)? << 8) + next(&mut ip)? + BLOSCLZ_MAX_DISTANCE;
            }
            let dist = ofs + 1;
            if dist > op || op + len > out.len() {
                return Err(bad());
            }
            // byte by byte: the source may overlap the bytes being written
            for i in op..op + len {
                out[i] = out[i - dist];
            }
            op += len;
        } else {
            let n = ctrl + 1;
            let lit = input.get(ip..ip + n).ok_or_else(bad)?;
            out.get_mut(op..op + n).ok_or_else(bad)?.copy_from_slice(lit);
            ip += n;
            op += n;
        }
        if ip >= input.len() {
            break;
        }
        ctrl = next(&mut ip)?;
    }
    Ok(op)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffles_round_trip() {
        let data: Vec<u8> = (0..70u32).map(|i| (i * 37 % 251) as u8).collect();
        // byte shuffle of 2-byte elements
        let n = data.len() / 2;
        let mut sh = vec![0u8; data.len()];
        for i in 0..n {
            sh[i] = data[2 * i];
            sh[n + i] = data[2 * i + 1];
        }
        let mut back = vec![0u8; data.len()];
        byte_unshuffle(2, &sh, &mut back);
        assert_eq!(back, data);
        // bit shuffle: 35 elements -> 32 shuffled, 3 stored unchanged
        let m = 32;
        let mut bs = data.clone();
        bs[..m * 2].fill(0);
        for i in 0..m {
            for j in 0..2 {
                for k in 0..8 {
                    let bit = (data[i * 2 + j] >> k) & 1;
                    bs[(j * 8 + k) * (m / 8) + i / 8] |= bit << (i % 8);
                }
            }
        }
        let mut back = vec![0u8; data.len()];
        bit_unshuffle(2, &bs, &mut back);
        assert_eq!(back, data);
    }

    #[test]
    fn memcpyed_buffer() {
        let mut buf = vec![2, 1, FLAG_MEMCPYED, 2, 4, 0, 0, 0, 4, 0, 0, 0, 20, 0, 0, 0];
        buf.extend_from_slice(&[9, 8, 7, 6]);
        assert_eq!(decompress(&buf).unwrap(), vec![9, 8, 7, 6]);
    }
}
