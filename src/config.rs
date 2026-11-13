//! Where NPXplorer keeps its own files, and reading the text files users give it.

use std::path::{Path, PathBuf};

use anyhow::Result;

/// `config/` next to the executable: preferences and the file-layout descriptions.
pub fn config_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            return dir.join("config");
        }
    }
    PathBuf::from("config")
}

// ---------------------------------------------------------------------------
// Encoding-robust text reading
// ---------------------------------------------------------------------------

/// Read a text file without assuming UTF-8. Handles UTF-8 (with/without BOM) and
/// UTF-16 LE/BE (with BOM); anything else that isn't valid UTF-8 is decoded as
/// Latin-1 (ISO-8859-1), which maps every byte to a char and so never fails. This
/// keeps event files exported from Excel/MATLAB/Python on any platform readable
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
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| if big_endian { u16::from_be_bytes(c) } else { u16::from_le_bytes(c) })
        .collect();
    String::from_utf16_lossy(&units)
}
