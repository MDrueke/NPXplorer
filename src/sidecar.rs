//! The recording's settings file, `<recording>.npxplorer.toml` next to its data,
//! shared by its AP and LF files (the app's settings types are in settings.rs).

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Band {
    Ap,
    Lf,
}

impl Band {
    /// LF streams are sampled at 2.5 kHz, AP streams at 30 kHz (SpikeGLX and Open
    /// Ephys alike).
    pub fn of_sample_rate(fs: f64) -> Self {
        if fs < 10_000.0 {
            Band::Lf
        } else {
            Band::Ap
        }
    }

    /// Name of the band's section in the file.
    pub fn key(self) -> &'static str {
        match self {
            Band::Ap => "ap",
            Band::Lf => "lf",
        }
    }
}

/// `<recording><suffix>` next to the data file; the AP and LF files of a SpikeGLX
/// recording share it.
pub fn recording_sidecar(bin_path: &Path, suffix: &str) -> PathBuf {
    let stem = bin_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let base = stem
        .strip_suffix(".ap")
        .or_else(|| stem.strip_suffix(".lf"))
        .unwrap_or(&stem)
        .to_string();
    bin_path.with_file_name(format!("{base}{suffix}"))
}

/// `<recording>.npxplorer.toml` next to the data file, shared by its AP and LF files.
pub fn path(bin_path: &Path) -> PathBuf {
    recording_sidecar(bin_path, ".npxplorer.toml")
}

/// The file's contents as a TOML table (empty if there is no file or it can't be read).
pub fn load_table(bin_path: &Path) -> toml::Table {
    std::fs::read_to_string(path(bin_path))
        .ok()
        .and_then(|t| t.parse::<toml::Table>().ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_name_shared_between_bands() {
        let a = recording_sidecar(Path::new("/d/rec_g0_t0.imec0.ap.bin"), ".x.toml");
        let b = recording_sidecar(Path::new("/d/rec_g0_t0.imec0.lf.bin"), ".x.toml");
        assert_eq!(a, b);
        assert_eq!(a, Path::new("/d/rec_g0_t0.imec0.x.toml"));
    }
}
