//! Per-recording settings: everything set in the main window and the tool windows,
//! saved in one file next to the recording, `<recording>.npxplorer.toml`, and applied
//! again when the recording is opened. The AP and LF files of a SpikeGLX recording
//! share the file: band-specific settings (preprocessing, colour scale, view, ...) live
//! in its `[ap]` / `[lf]` sections, the channel removal, stimulus and atlas settings are
//! shared.
//!
//! Values missing from the file (older versions, hand edits) keep the value the app
//! started with — the global preferences, i.e. the settings last used anywhere.
//!
//! Earlier versions wrote three files instead (`.npx_atlas.toml`, `.npx_stim.toml`,
//! `<data file>.npx_notch.toml`) plus `stims_file_layout.csv` in the data folder. They
//! are read when the new file lacks their section; the first three are deleted once
//! their content has been written to the new file.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::app::ColorMode;
use crate::colormap::ColorMapChoice;
use crate::atlas::{Insertion, RegionEdit};
use crate::preprocess::SpatialFilter;
use crate::spectrum::{SpectrumNormalization, SpectrumScaling, SpectrumSource, SpectrumTimeScope};

#[derive(Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RecordingSettings {
    /// 0-based channels excluded from display and computation; `None` = never saved,
    /// the probe's reference sites are removed then
    #[serde(skip_serializing_if = "Option::is_none")]
    pub removed_channels: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stim: Option<StimSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atlas: Option<AtlasSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ap: Option<BandSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lf: Option<BandSettings>,
}

/// Stimulus file and the settings of the TTL and PSTH windows.
#[derive(Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StimSettings {
    /// stimulus file last loaded in the TTL or PSTH window
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stim_file: Option<PathBuf>,
    /// stim-file format; `None` = the default in `config/`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout: Option<String>,
    pub ttl: crate::ttl::TtlSettings,
    pub psth: crate::ttl::PsthSettings,
}

#[derive(Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AtlasSettings {
    /// overlay shown when the recording was closed: the atlas is loaded and the probe
    /// registered again on opening
    pub show_overlay: bool,
    /// regions with fewer channels are not drawn on their own
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_region_channels: Option<usize>,
    pub insertion: Insertion,
    /// corrections made by dragging region borders
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub region_edits: Vec<RegionEdit>,
}

/// Settings of one band (AP or LF).
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BandSettings {
    /// displayed time window (the one from before the zoom, when zoomed)
    pub view_start_s: f64,
    pub view_dur_s: f64,
    pub scroll_fine: bool,
    pub color_mode: ColorMode,
    pub color_pct: f32,
    pub color_uv: f32,
    pub colormap: ColorMapChoice,
    pub peak_pooling: bool,
    /// ± range of the single-channel waveform view
    pub waveform_y_range_uv: f32,
    pub preproc: PreprocSettings,
    pub firing_rate: FiringRateSettings,
    pub classification: ClassificationSettings,
    pub spectrum: SpectrumSettings,
    /// applied noise-suppression display filters
    pub noise: crate::noise::NoiseSuppression,
    /// settings of the noise-peak scan (Notch filters)
    pub notch_scan: crate::notch::ScanSettings,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocSettings {
    pub dc_removal: bool,
    pub phase_shift: bool,
    pub highpass: bool,
    pub spatial_filter: SpatialFilter,
    pub avg_depths: bool,
    pub channel_order: crate::data::ChannelOrder,
    pub shank_order: crate::data::ShankOrder,
    pub notch_enabled: bool,
    pub notches: Vec<crate::notch::Notch>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct FiringRateSettings {
    pub show: bool,
    pub threshold_uv: f32,
    pub overlay_scale: f32,
    pub smoothing_sigma: f32,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationSettings {
    pub n_chunks: usize,
    pub outside_rule: crate::channel_classify::OutsideRule,
    pub show_overlay: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct SpectrumSettings {
    pub time_scope: SpectrumTimeScope,
    pub source: SpectrumSource,
    pub scaling: SpectrumScaling,
    pub normalization: SpectrumNormalization,
    pub n_chunks: usize,
    pub freq_restrict: bool,
    pub freq_min_hz: f64,
    pub freq_max_hz: f64,
    pub time_restrict: bool,
    pub time_start_s: f64,
    pub time_end_s: f64,
    pub show_overlay: bool,
}

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

impl RecordingSettings {
    pub fn band_mut(&mut self, band: Band) -> &mut Option<BandSettings> {
        match band {
            Band::Ap => &mut self.ap,
            Band::Lf => &mut self.lf,
        }
    }
}

/// `<recording>.npxplorer.toml` next to the data file, shared by its AP and LF files.
pub fn path(bin_path: &Path) -> PathBuf {
    crate::atlas::recording_sidecar(bin_path, ".npxplorer.toml")
}

/// The file's contents as a TOML table (empty if there is no file or it can't be read).
pub fn load_table(bin_path: &Path) -> toml::Table {
    std::fs::read_to_string(path(bin_path))
        .ok()
        .and_then(|t| t.parse::<toml::Table>().ok())
        .unwrap_or_default()
}

/// `base` with every value present in `file` replaced by the file's: values missing
/// from the file (or of the wrong type) keep the base value.
pub fn overlay<T: Serialize + DeserializeOwned + Clone>(base: &T, file: Option<&toml::Value>) -> T {
    let (Some(toml::Value::Table(file)), Ok(toml::Value::Table(mut merged))) = (file, toml::Value::try_from(base))
    else {
        return base.clone();
    };
    // key by key, so a value that doesn't fit (wrong type, a removed option) only
    // loses itself
    for (k, fv) in file {
        let mut trial = merged.clone();
        match trial.get_mut(k) {
            Some(slot) => merge(slot, fv),
            None => {
                trial.insert(k.clone(), fv.clone());
            }
        }
        if toml::Value::Table(trial.clone()).try_into::<T>().is_ok() {
            merged = trial;
        }
    }
    toml::Value::Table(merged).try_into().unwrap_or_else(|_| base.clone())
}

fn merge(base: &mut toml::Value, file: &toml::Value) {
    match (base, file) {
        (toml::Value::Table(b), toml::Value::Table(f)) => {
            for (k, fv) in f {
                match b.get_mut(k) {
                    Some(bv) => merge(bv, fv),
                    None => {
                        b.insert(k.clone(), fv.clone());
                    }
                }
            }
        }
        (b, f) => *b = f.clone(),
    }
}

/// Write the file: this band's section and the shared sections from `settings`; the
/// other band's section is kept as it is in the file.
pub fn save(bin_path: &Path, band: Band, settings: &RecordingSettings) -> Result<()> {
    let path = path(bin_path);
    let mut table = load_table(bin_path);
    let mine = toml::Table::try_from(settings)?;
    for k in ["removed_channels", "stim", "atlas", band.key()] {
        match mine.get(k) {
            Some(v) => table.insert(k.to_string(), v.clone()),
            None => table.remove(k),
        };
    }
    let mut table = toml::Value::Table(table);
    tidy_floats(&mut table);
    let text = format!(
        "# NPXplorer settings for this recording, restored when it is opened again.\n\
         # Shared by its AP and LF files; [ap] and [lf] hold the per-band settings.\n\n{}",
        toml::to_string(&table)?
    );
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    remove_legacy_files(bin_path);
    Ok(())
}

/// f32 settings pass through `toml::Value` as f64 and would be written as e.g.
/// 0.8999999761581421: write them as the f32's shortest form (0.9) instead.
fn tidy_floats(v: &mut toml::Value) {
    match v {
        toml::Value::Float(f) if (*f as f32) as f64 == *f => {
            if let Ok(short) = (*f as f32).to_string().parse::<f64>() {
                *f = short;
            }
        }
        toml::Value::Table(t) => t.iter_mut().for_each(|(_, v)| tidy_floats(v)),
        toml::Value::Array(a) => a.iter_mut().for_each(tidy_floats),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Files of earlier versions
// ---------------------------------------------------------------------------

/// Atlas settings from a `.npx_atlas.toml` of an earlier version.
pub fn legacy_atlas(bin_path: &Path) -> Option<AtlasSettings> {
    #[derive(Deserialize)]
    struct File {
        #[serde(flatten)]
        insertion: Insertion,
        #[serde(default)]
        region_edits: Vec<RegionEdit>,
    }
    let text = std::fs::read_to_string(legacy_atlas_path(bin_path)).ok()?;
    let f: File = toml::from_str(&text).ok()?;
    Some(AtlasSettings {
        show_overlay: false,
        min_region_channels: None,
        insertion: f.insertion,
        region_edits: f.region_edits,
    })
}

/// Stimulus settings from a `.npx_stim.toml` and the folder's `stims_file_layout.csv`
/// of earlier versions.
pub fn legacy_stim(bin_path: &Path) -> Option<StimSettings> {
    #[derive(Deserialize, Default)]
    #[serde(default)]
    struct File {
        stim_file: Option<PathBuf>,
        ttl: crate::ttl::TtlSettings,
        psth: crate::ttl::PsthSettings,
    }
    let f: Option<File> = std::fs::read_to_string(legacy_stim_path(bin_path))
        .ok()
        .and_then(|t| toml::from_str(&t).ok());
    let layout = crate::psth::read_text_file(&crate::psth::layout_sidecar_path(bin_path)).ok();
    if f.is_none() && layout.is_none() {
        return None;
    }
    let f = f.unwrap_or_default();
    Some(StimSettings { stim_file: f.stim_file, layout, ttl: f.ttl, psth: f.psth })
}

/// Notches from a `<data file>.npx_notch.toml` of an earlier version.
pub fn legacy_notches(bin_path: &Path) -> Option<Vec<crate::notch::Notch>> {
    #[derive(Deserialize)]
    struct File {
        #[serde(default)]
        notches: Vec<crate::notch::Notch>,
    }
    let text = std::fs::read_to_string(legacy_notch_path(bin_path)).ok()?;
    toml::from_str::<File>(&text).ok().map(|f| f.notches)
}

fn legacy_atlas_path(bin_path: &Path) -> PathBuf {
    crate::atlas::recording_sidecar(bin_path, ".npx_atlas.toml")
}

fn legacy_stim_path(bin_path: &Path) -> PathBuf {
    crate::atlas::recording_sidecar(bin_path, ".npx_stim.toml")
}

fn legacy_notch_path(bin_path: &Path) -> PathBuf {
    let stem = bin_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    bin_path.with_file_name(format!("{stem}.npx_notch.toml"))
}

/// Delete the per-recording files of earlier versions once the new file holds their
/// content. The folder's `stims_file_layout.csv` stays: other recordings may use it.
fn remove_legacy_files(bin_path: &Path) {
    for p in [legacy_atlas_path(bin_path), legacy_stim_path(bin_path), legacy_notch_path(bin_path)] {
        if p.is_file() {
            let _ = std::fs::remove_file(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
    struct Inner {
        a: f64,
        b: bool,
    }
    #[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
    struct Outer {
        x: u32,
        inner: Inner,
    }

    #[test]
    fn overlay_keeps_missing_values() {
        let base = Outer { x: 1, inner: Inner { a: 2.0, b: false } };
        let file: toml::Value = toml::from_str("[inner]\nb = true\n").unwrap();
        assert_eq!(overlay(&base, Some(&file)), Outer { x: 1, inner: Inner { a: 2.0, b: true } });
        assert_eq!(overlay(&base, None), base);
    }

    #[test]
    fn overlay_skips_values_of_the_wrong_type() {
        let base = Outer { x: 1, inner: Inner { a: 2.0, b: false } };
        let file: toml::Value = toml::from_str("x = \"seven\"\n[inner]\nb = true\n").unwrap();
        assert_eq!(overlay(&base, Some(&file)), Outer { x: 1, inner: Inner { a: 2.0, b: true } });
    }

    #[test]
    fn band_sections_are_kept_and_shared_ones_replaced() {
        let dir = std::env::temp_dir().join(format!("npx_settings_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ap = dir.join("rec_g0_t0.imec0.ap.bin");
        let lf = dir.join("rec_g0_t0.imec0.lf.bin");
        assert_eq!(path(&ap), path(&lf));

        let atlas = |depth: f64| AtlasSettings {
            insertion: Insertion { depth_mm: depth, ..Default::default() },
            region_edits: vec![RegionEdit { shank: 0, from_um: 10.0, to_um: 30.0, region: Some(315) }],
            ..Default::default()
        };
        let s_ap = RecordingSettings {
            removed_channels: Some(vec![191]),
            atlas: Some(atlas(3.0)),
            stim: Some(StimSettings { layout: Some("header\no".into()), ..Default::default() }),
            ..Default::default()
        };
        save(&ap, Band::Ap, &s_ap).unwrap();
        // an [ap] section, written by hand, must survive a save from the LF file
        let mut text = std::fs::read_to_string(path(&ap)).unwrap();
        text.push_str("\n[ap]\nview_dur_s = 0.25\n");
        std::fs::write(path(&ap), text).unwrap();

        let s_lf = RecordingSettings { removed_channels: Some(vec![]), atlas: Some(atlas(2.5)), ..Default::default() };
        save(&lf, Band::Lf, &s_lf).unwrap();
        let t = load_table(&ap);
        assert_eq!(t["removed_channels"].as_array().unwrap().len(), 0);
        assert_eq!(t["ap"]["view_dur_s"].as_float(), Some(0.25));
        assert!(t.get("stim").is_none());
        let a: AtlasSettings = t["atlas"].clone().try_into().unwrap();
        assert!(a == atlas(2.5));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f32_values_are_written_short() {
        let mut v = toml::Value::try_from(Inner { a: 0.9f32 as f64, b: true }).unwrap();
        tidy_floats(&mut v);
        assert_eq!(v["a"].as_float(), Some(0.9));
        // f64 values that aren't f32s stay as they are
        let mut v = toml::Value::Float(0.05);
        tidy_floats(&mut v);
        assert_eq!(v.as_float(), Some(0.05));
    }

    #[test]
    fn atlas_round_trip() {
        let a = AtlasSettings {
            show_overlay: true,
            min_region_channels: Some(2),
            insertion: Insertion { ap_mm: -2.5, depth_mm: 3.2, level: Some(5), ..Default::default() },
            region_edits: vec![
                RegionEdit { shank: 0, from_um: 100.0, to_um: 140.0, region: Some(385) },
                RegionEdit { shank: 1, from_um: 0.0, to_um: 20.0, region: None },
            ],
        };
        let s = RecordingSettings { atlas: Some(a), ..Default::default() };
        let text = toml::to_string_pretty(&s).unwrap();
        assert!(toml::from_str::<RecordingSettings>(&text).unwrap() == s);
    }

    #[test]
    fn legacy_files_are_read_and_removed() {
        let dir = std::env::temp_dir().join(format!("npx_settings_legacy_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("rec_g0_t0.imec0.ap.bin");
        std::fs::write(legacy_atlas_path(&bin), "ap_mm = -2.0\ndepth_mm = 3.5\n").unwrap();
        std::fs::write(legacy_stim_path(&bin), "stim_file = \"/d/stims.csv\"\n[ttl]\nopacity_pct = 30.0\n").unwrap();
        std::fs::write(legacy_notch_path(&bin), "[[notches]]\nfreq_hz = 50.0\nbw_hz = 1.0\n").unwrap();

        let a = legacy_atlas(&bin).unwrap();
        assert_eq!((a.insertion.ap_mm, a.insertion.depth_mm), (-2.0, 3.5));
        let s = legacy_stim(&bin).unwrap();
        assert_eq!(s.stim_file.as_deref(), Some(Path::new("/d/stims.csv")));
        assert_eq!(s.ttl.opacity_pct, 30.0);
        assert_eq!(legacy_notches(&bin).unwrap().len(), 1);

        save(&bin, Band::Ap, &RecordingSettings::default()).unwrap();
        assert!(legacy_atlas(&bin).is_none() && legacy_notches(&bin).is_none());
        assert!(!legacy_stim_path(&bin).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
