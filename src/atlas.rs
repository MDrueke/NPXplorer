// Allen CCF atlas registration: find the brain region under every recorded electrode
// from the probe's insertion coordinates. The geometry (CCF -> bregma transform,
// brain-surface entry, probe rotation) is a port of the MATLAB Neuropixels Trajectory
// Explorer (petersaj/neuropixels_trajectory_explorer, `neuropixels_trajectory_explorer.m`),
// so the same inputs give the same regions as NTE. Reads NTE's atlas files:
// `annotation_volume_10um_by_index.npy` (memory-mapped) + `structure_tree_safe_2017.csv`.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::data::Meta;

pub const ANNOTATION_FILE: &str = "annotation_volume_10um_by_index.npy";
pub const STRUCTURE_TREE_FILE: &str = "structure_tree_safe_2017.csv";

/// Average stereotaxic bregma-lambda distance (mm) the CCF corresponds to (NTE's value).
pub const REFERENCE_BREGMA_LAMBDA_MM: f64 = 4.1;
pub const DEFAULT_TIP_OFFSET_UM: f64 = 195.0;
/// Physical length of a Neuropixels shank — only used for sanity warnings.
pub const SHANK_LENGTH_UM: f64 = 10_000.0;
/// Center-to-center spacing of shanks on multi-shank probes.
const SHANK_PITCH_UM: f64 = 250.0;
/// Step size (mm) when sampling along a trajectory — NTE uses 1 µm.
const TRAJECTORY_STEP_MM: f64 = 0.001;
/// Step size (µm) of the per-shank region table samples.
pub const TABLE_STEP_UM: f64 = 10.0;

/// Progress reported by a background load/registration job, in per-mille.
pub const PROGRESS_TOTAL: usize = 1000;

// ---------------------------------------------------------------------------
// Structure tree
// ---------------------------------------------------------------------------

pub struct Structure {
    /// Allen structure id
    pub id: u32,
    pub acronym: String,
    pub name: String,
    /// row index of the parent structure in `StructureTree::rows`
    pub parent: Option<usize>,
    /// hierarchy depth (root = 0)
    pub depth: u32,
}

pub struct StructureTree {
    pub rows: Vec<Structure>,
    pub max_depth: u32,
    row_of: HashMap<u32, usize>,
}

impl StructureTree {
    pub fn parse(text: &str) -> Result<Self> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = split_csv_line(lines.next().context("structure tree CSV is empty")?);
        let col = |name: &str| header.iter().position(|h| h.trim() == name);
        let c_id = col("id").context("structure tree CSV has no 'id' column")?;
        let c_acr = col("acronym").context("structure tree CSV has no 'acronym' column")?;
        let c_parent = col("parent_structure_id")
            .context("structure tree CSV has no 'parent_structure_id' column")?;
        let c_depth = col("depth").context("structure tree CSV has no 'depth' column")?;
        let c_name = col("safe_name")
            .or_else(|| col("name"))
            .context("structure tree CSV has no 'safe_name'/'name' column")?;

        let mut ids = Vec::new();
        let mut parent_ids = Vec::new();
        let mut rows = Vec::new();
        for (i, line) in lines.enumerate() {
            let f = split_csv_line(line);
            let get = |c: usize| f.get(c).map(|s| s.trim()).unwrap_or("");
            let id: u32 = get(c_id)
                .parse()
                .with_context(|| format!("structure tree row {}: bad id '{}'", i + 1, get(c_id)))?;
            ids.push(id);
            parent_ids.push(get(c_parent).parse::<u32>().ok());
            rows.push(Structure {
                id,
                acronym: get(c_acr).to_string(),
                name: get(c_name).to_string(),
                parent: None,
                depth: get(c_depth).parse().unwrap_or(0),
            });
        }
        if rows.is_empty() {
            bail!("structure tree CSV has no rows");
        }
        let row_of: HashMap<u32, usize> = ids.iter().enumerate().map(|(r, &id)| (id, r)).collect();
        for (r, pid) in parent_ids.into_iter().enumerate() {
            rows[r].parent = pid.and_then(|p| row_of.get(&p).copied());
        }
        let max_depth = rows.iter().map(|s| s.depth).max().unwrap_or(0);
        Ok(Self { rows, max_depth, row_of })
    }

    /// Row of the structure with the given Allen id.
    pub fn row_of_id(&self, id: u32) -> Option<usize> {
        self.row_of.get(&id).copied()
    }

    /// Coarsen `row` to the given hierarchy level: walk up to the ancestor at that depth
    /// (structures already at or above it are returned unchanged). `None` = finest.
    pub fn at_level(&self, row: usize, level: Option<u32>) -> usize {
        let Some(level) = level else { return row };
        let mut r = row;
        while self.rows[r].depth > level {
            match self.rows[r].parent {
                Some(p) => r = p,
                None => break,
            }
        }
        r
    }
}

/// Split one CSV line on commas, honoring double-quoted fields ("a, b" and "" escapes).
fn split_csv_line(line: &str) -> Vec<String> {
    let line = line.trim_end_matches('\r');
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

// ---------------------------------------------------------------------------
// Annotation volume (.npy, memory-mapped)
// ---------------------------------------------------------------------------

enum VolumeData {
    Mapped(memmap2::Mmap),
    #[cfg(test)]
    Owned(Vec<u8>),
}

impl VolumeData {
    fn bytes(&self) -> &[u8] {
        match self {
            VolumeData::Mapped(m) => m.as_ref(),
            #[cfg(test)]
            VolumeData::Owned(v) => v,
        }
    }
}

/// u16 annotation volume indexed [AP][DV][ML] (NTE's layout), holding 1-based row
/// numbers into the structure tree; 1 (root) and 0 mean outside the brain.
pub struct AnnotationVolume {
    data: VolumeData,
    offset: usize,
    /// (AP, DV, ML)
    pub shape: [usize; 3],
    fortran_order: bool,
    /// test-only procedural volume, so tests don't need to allocate a real-sized one
    #[cfg(test)]
    synthetic: Option<fn(usize, usize, usize) -> u16>,
}

struct NpyHeader {
    data_offset: usize,
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
}

fn parse_npy_header(bytes: &[u8]) -> Result<NpyHeader> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not a .npy file (bad magic)");
    }
    let major = bytes[6];
    let (header_len, start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                bail!("truncated .npy header");
            }
            (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
        }
        v => bail!("unsupported .npy version {v}"),
    };
    let end = start + header_len;
    if bytes.len() < end {
        bail!("truncated .npy header");
    }
    let header = std::str::from_utf8(&bytes[start..end]).context(".npy header is not text")?;

    let value_after = |key: &str| -> Result<&str> {
        let k = format!("'{key}'");
        let i = header.find(&k).with_context(|| format!(".npy header missing {k}"))?;
        let rest = &header[i + k.len()..];
        let colon = rest.find(':').context("malformed .npy header")?;
        Ok(rest[colon + 1..].trim_start())
    };
    let descr = {
        let v = value_after("descr")?;
        let q = v.chars().next().context("malformed descr")?;
        let inner = &v[1..];
        inner[..inner.find(q).context("malformed descr")?].to_string()
    };
    let fortran_order = value_after("fortran_order")?.starts_with("True");
    let shape = {
        let v = value_after("shape")?;
        let open = v.find('(').context("malformed shape")?;
        let close = v.find(')').context("malformed shape")?;
        v[open + 1..close]
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().context("malformed shape entry"))
            .collect::<Result<Vec<_>>>()?
    };
    Ok(NpyHeader { data_offset: end, descr, fortran_order, shape })
}

impl AnnotationVolume {
    fn from_bytes(data: VolumeData) -> Result<Self> {
        let h = parse_npy_header(data.bytes())?;
        if !matches!(h.descr.as_str(), "<u2" | "<i2" | "=u2" | "|u2") {
            bail!("annotation volume has dtype '{}', expected 16-bit ('<u2')", h.descr);
        }
        if h.shape.len() != 3 {
            bail!("annotation volume has {} dimensions, expected 3", h.shape.len());
        }
        let shape = [h.shape[0], h.shape[1], h.shape[2]];
        let n = shape[0] * shape[1] * shape[2];
        if data.bytes().len() < h.data_offset + 2 * n {
            bail!("annotation volume file is truncated");
        }
        Ok(Self {
            data,
            offset: h.data_offset,
            shape,
            fortran_order: h.fortran_order,
            #[cfg(test)]
            synthetic: None,
        })
    }

    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_bytes(VolumeData::Mapped(mmap)).with_context(|| format!("reading {}", path.display()))
    }

    /// Raw value at 0-based voxel (ap, dv, ml), or None outside the volume.
    pub fn get(&self, ap: usize, dv: usize, ml: usize) -> Option<u16> {
        let [n_ap, n_dv, n_ml] = self.shape;
        if ap >= n_ap || dv >= n_dv || ml >= n_ml {
            return None;
        }
        #[cfg(test)]
        if let Some(f) = self.synthetic {
            return Some(f(ap, dv, ml));
        }
        let idx = if self.fortran_order {
            ap + n_ap * (dv + n_dv * ml)
        } else {
            (ap * n_dv + dv) * n_ml + ml
        };
        let b = self.data.bytes();
        let o = self.offset + 2 * idx;
        Some(u16::from_le_bytes([b[o], b[o + 1]]))
    }
}

// ---------------------------------------------------------------------------
// Atlas = volume + tree + CCF<->bregma transform
// ---------------------------------------------------------------------------

pub struct Atlas {
    pub dir: PathBuf,
    pub volume: AnnotationVolume,
    pub tree: StructureTree,
}

impl Atlas {
    pub fn load(dir: &Path, cancel: &AtomicBool, progress: &AtomicUsize) -> Result<Option<Self>> {
        let tree_path = dir.join(STRUCTURE_TREE_FILE);
        let vol_path = dir.join(ANNOTATION_FILE);
        let missing: Vec<&str> = [(STRUCTURE_TREE_FILE, &tree_path), (ANNOTATION_FILE, &vol_path)]
            .iter()
            .filter(|(_, p)| !p.is_file())
            .map(|(n, _)| *n)
            .collect();
        if !missing.is_empty() {
            bail!("atlas file(s) not found in {}: {}", dir.display(), missing.join(", "));
        }
        let text = crate::psth::read_text_file(&tree_path)?;
        let tree = StructureTree::parse(&text).with_context(|| format!("reading {}", tree_path.display()))?;
        progress.store(PROGRESS_TOTAL / 10, Ordering::Relaxed);
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let volume = AnnotationVolume::open(&vol_path)?;
        progress.store(PROGRESS_TOTAL * 3 / 10, Ordering::Relaxed);
        Ok(Some(Self { dir: dir.to_path_buf(), volume, tree }))
    }

    /// Structure-tree row at a bregma-relative point, `None` outside the brain.
    pub fn region_at(&self, p: [f64; 3], tf: &CcfTransform) -> Option<usize> {
        let c = tf.bregma_to_ccf(p); // 1-based voxel coords [ML, AP, DV]
        let idx = |v: f64| -> Option<usize> {
            let r = v.round();
            if r >= 1.0 { Some(r as usize - 1) } else { None }
        };
        let (ml, ap, dv) = (idx(c[0])?, idx(c[1])?, idx(c[2])?);
        let v = self.volume.get(ap, dv, ml)? as usize;
        if v <= 1 || v > self.tree.rows.len() {
            return None;
        }
        Some(v - 1)
    }
}

/// NTE's CCF -> bregma transform: bregma-relative [ML, AP, DV] in mm (AP + anterior,
/// DV + ventral) from 1-based 10 µm CCF voxel coordinates [ML, AP, DV]:
/// translate to bregma, anisotropic "Toronto MRI" scale, 5° nose-up AP tilt, then an
/// isotropic per-mouse scale (bregma-lambda distance / 4.1 mm).
pub struct CcfTransform {
    mouse_scale: f64,
}

const BREGMA_CCF: [f64; 3] = [570.5, 520.0, 44.0]; // [ML, AP, DV] voxels
const CCF_SCALE: [f64; 3] = [0.952 / 100.0, -1.031 / 100.0, 0.885 / 100.0];
const AP_TILT_DEG: f64 = 5.0;

impl CcfTransform {
    pub fn new(bregma_lambda_mm: f64) -> Self {
        Self { mouse_scale: bregma_lambda_mm / REFERENCE_BREGMA_LAMBDA_MM }
    }

    #[allow(dead_code)] // inverse direction; used by tests
    pub fn ccf_to_bregma(&self, c: [f64; 3]) -> [f64; 3] {
        let s = [
            (c[0] - BREGMA_CCF[0]) * CCF_SCALE[0],
            (c[1] - BREGMA_CCF[1]) * CCF_SCALE[1],
            (c[2] - BREGMA_CCF[2]) * CCF_SCALE[2],
        ];
        let (sn, cs) = AP_TILT_DEG.to_radians().sin_cos();
        // row vector times NTE's rotation matrix
        let r = [s[0], s[1] * cs + s[2] * sn, -s[1] * sn + s[2] * cs];
        r.map(|v| v * self.mouse_scale)
    }

    pub fn bregma_to_ccf(&self, p: [f64; 3]) -> [f64; 3] {
        let q = p.map(|v| v / self.mouse_scale);
        let (sn, cs) = AP_TILT_DEG.to_radians().sin_cos();
        let s = [q[0], q[1] * cs - q[2] * sn, q[1] * sn + q[2] * cs];
        [
            s[0] / CCF_SCALE[0] + BREGMA_CCF[0],
            s[1] / CCF_SCALE[1] + BREGMA_CCF[1],
            s[2] / CCF_SCALE[2] + BREGMA_CCF[2],
        ]
    }
}

// ---------------------------------------------------------------------------
// Insertion parameters, angle conventions, sidecar
// ---------------------------------------------------------------------------

/// How the insertion angles are entered in the UI. Stored values are always
/// azimuth/elevation/rotation; the other convention is converted on entry.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, Default)]
pub enum AngleConvention {
    /// azimuth from the lambda->bregma axis, elevation from horizontal, rotation
    #[default]
    AzimuthElevation,
    /// polar angle from vertical, azimuth from +ML (counter-clockwise seen from above), roll
    PolarAzimuth,
}

/// polar θ / azimuth φ -> azimuth / elevation (degrees)
pub fn polar_to_az_el(theta: f64, phi: f64) -> (f64, f64) {
    ((270.0 - phi).rem_euclid(360.0), 90.0 - theta)
}

/// azimuth / elevation -> polar θ / azimuth φ (degrees)
pub fn az_el_to_polar(az: f64, el: f64) -> (f64, f64) {
    (90.0 - el, (270.0 - az).rem_euclid(360.0))
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Insertion {
    /// mm from bregma, + anterior
    pub ap_mm: f64,
    /// mm from bregma, + right
    pub ml_mm: f64,
    /// degrees, relative to the lambda -> bregma axis
    pub azimuth_deg: f64,
    /// degrees from horizontal (90 = vertical)
    pub elevation_deg: f64,
    /// degrees, rotation of the probe around its own axis
    pub rotation_deg: f64,
    /// mm along the probe axis from the brain-surface entry point to the tip
    pub depth_mm: f64,
    /// µm from the probe tip to the first row of electrodes
    pub tip_offset_um: f64,
    pub bregma_lambda_mm: f64,
    pub angle_convention: AngleConvention,
    /// hierarchy level to coarsen regions to; None = finest (incl. cortical layers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<u32>,
}

impl Default for Insertion {
    fn default() -> Self {
        Self {
            ap_mm: 0.0,
            ml_mm: 0.0,
            azimuth_deg: 0.0,
            elevation_deg: 90.0,
            rotation_deg: 0.0,
            depth_mm: 3.0,
            tip_offset_um: DEFAULT_TIP_OFFSET_UM,
            bregma_lambda_mm: REFERENCE_BREGMA_LAMBDA_MM,
            angle_convention: AngleConvention::AzimuthElevation,
            level: None,
        }
    }
}

/// A correction made by dragging a region border on the heatmap: on `shank`, the
/// stretch of the trajectory from `from_um` to `to_um` (µm along the probe axis below
/// the brain-surface entry point) belongs to `region`. Stored in brain coordinates, so
/// edits move along with the atlas borders when the insertion depth changes.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RegionEdit {
    pub shank: u32,
    pub from_um: f64,
    pub to_um: f64,
    /// Allen structure id; absent = outside the brain
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<u32>,
}

/// The edit covering depth `d_um` on `shank`, if any.
pub fn edit_at(edits: &[RegionEdit], shank: u32, d_um: f64) -> Option<&RegionEdit> {
    edits.iter().rev().find(|e| e.shank == shank && e.from_um <= d_um && d_um < e.to_um)
}

/// Record `new`, replacing whatever edits overlap its range; with `keep == false` the
/// range just reverts to the atlas (overlapping edits are cut out, nothing is added).
/// Neighboring edits of the same region are merged.
pub fn upsert_edit(edits: &mut Vec<RegionEdit>, new: RegionEdit, keep: bool) {
    let mut out: Vec<RegionEdit> = Vec::with_capacity(edits.len() + 2);
    for e in edits.drain(..) {
        if e.shank != new.shank || e.to_um <= new.from_um || e.from_um >= new.to_um {
            out.push(e);
            continue;
        }
        if e.from_um < new.from_um {
            out.push(RegionEdit { to_um: new.from_um, ..e.clone() });
        }
        if e.to_um > new.to_um {
            out.push(RegionEdit { from_um: new.to_um, ..e });
        }
    }
    if keep {
        out.push(new);
    }
    out.sort_by(|a, b| a.shank.cmp(&b.shank).then(a.from_um.total_cmp(&b.from_um)));
    for e in out {
        match edits.last_mut() {
            Some(l) if l.shank == e.shank && l.region == e.region && (l.to_um - e.from_um).abs() < 1e-6 => {
                l.to_um = e.to_um
            }
            _ => edits.push(e),
        }
    }
}

/// Depth (µm along the probe axis below the brain-surface entry point) of an
/// electrode at probe position `y_um`, for the given insertion.
pub fn brain_depth_um(ins: &Insertion, y_um: f32) -> f64 {
    ins.depth_mm * 1000.0 - (ins.tip_offset_um + y_um as f64)
}

/// Half the vertical electrode pitch per shank (µm) — the extent of one row of
/// electrodes along the probe, used for the range an edited row covers.
pub fn half_pitch_per_shank(meta: &Meta) -> HashMap<u32, f64> {
    let mut ys: HashMap<u32, Vec<f32>> = HashMap::new();
    for g in &meta.channel_geom {
        ys.entry(g.shank).or_default().push(g.y_um);
    }
    ys.into_iter()
        .map(|(s, mut v)| {
            v.sort_by(|a, b| a.total_cmp(b));
            let pitch = v.windows(2).map(|w| (w[1] - w[0]) as f64).filter(|&d| d > 0.1).fold(f64::INFINITY, f64::min);
            (s, if pitch.is_finite() { pitch / 2.0 } else { 10.0 })
        })
        .collect()
}

/// Contents of the sidecar: the insertion plus the edits made by dragging borders.
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct SidecarFile {
    #[serde(flatten)]
    insertion: Insertion,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    region_edits: Vec<RegionEdit>,
}

/// Sidecar file next to the recording holding its insertion coordinates. The AP and
/// LF files of one SpikeGLX recording share it (".ap"/".lf" is stripped).
pub fn sidecar_path(bin_path: &Path) -> PathBuf {
    let stem = bin_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let base = stem
        .strip_suffix(".ap")
        .or_else(|| stem.strip_suffix(".lf"))
        .unwrap_or(&stem)
        .to_string();
    bin_path.with_file_name(format!("{base}.npx_atlas.toml"))
}

pub fn load_sidecar(bin_path: &Path) -> Option<(Insertion, Vec<RegionEdit>)> {
    let text = std::fs::read_to_string(sidecar_path(bin_path)).ok()?;
    let f: SidecarFile = toml::from_str(&text).ok()?;
    Some((f.insertion, f.region_edits))
}

pub fn save_sidecar(bin_path: &Path, ins: &Insertion, region_edits: &[RegionEdit]) -> Result<()> {
    let path = sidecar_path(bin_path);
    let f = SidecarFile { insertion: ins.clone(), region_edits: region_edits.to_vec() };
    let text = toml::to_string_pretty(&f)?;
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
}

/// Path of the per-channel region table: `<data file stem>_regions.csv` next to it.
pub fn regions_csv_path(bin_path: &Path) -> PathBuf {
    let stem = bin_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    bin_path.with_file_name(format!("{stem}_regions.csv"))
}

/// Write (channel ID, region acronym) rows; returns the path written.
pub fn save_regions_csv(bin_path: &Path, rows: &[(String, String)]) -> Result<PathBuf> {
    let path = regions_csv_path(bin_path);
    let mut text = String::from("channel,region\n");
    for (ch, region) in rows {
        text.push_str(&format!("{ch},{region}\n"));
    }
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// Probe geometry
// ---------------------------------------------------------------------------

type Mat3 = [[f64; 3]; 3];

fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

fn mat_vec(a: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
}

fn rot_z(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn rot_x(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|v| v * s)
}

/// Unit vector of the insertion direction (top -> tip) in bregma space [ML, AP, DV].
pub fn insertion_direction(azimuth_deg: f64, elevation_deg: f64) -> [f64; 3] {
    let (az, el) = (azimuth_deg.to_radians(), elevation_deg.to_radians());
    [el.cos() * az.sin(), el.cos() * az.cos(), el.sin()]
}

/// NTE's probe rotation (`update_probe_position`): maps probe-local coordinates
/// [across shanks, 0, along shank towards the tip] to bregma space.
pub fn probe_rotation(azimuth_deg: f64, elevation_deg: f64, rotation_deg: f64) -> Mat3 {
    // cart2sph of the direction: azimuth = 90° - az, elevation = el
    let d = insertion_direction(azimuth_deg, elevation_deg);
    let traj_az = d[1].atan2(d[0]);
    let traj_el = d[2].atan2((d[0] * d[0] + d[1] * d[1]).sqrt());
    let r_shank = rot_z(-rotation_deg.to_radians());
    let r_elev = rot_x(std::f64::consts::FRAC_PI_2 - traj_el);
    let r_az = rot_z(traj_az + std::f64::consts::FRAC_PI_2);
    mat_mul(&mat_mul(&r_az, &r_elev), &r_shank)
}

/// Lateral position (µm, across shanks) of every channel relative to the reference
/// shank's midline. SpikeGLX gives x within each shank, Open Ephys gives x across the
/// whole probe — detected from how far apart the per-shank mean x values are.
fn lateral_positions_um(meta: &Meta) -> Vec<f64> {
    let geom = &meta.channel_geom;
    let mut per_shank: HashMap<u32, (f64, usize)> = HashMap::new();
    for g in geom {
        let e = per_shank.entry(g.shank).or_insert((0.0, 0));
        e.0 += g.x_um as f64;
        e.1 += 1;
    }
    let means: HashMap<u32, f64> = per_shank.iter().map(|(&s, &(sum, n))| (s, sum / n as f64)).collect();
    let (lo, hi) = means.values().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &m| (lo.min(m), hi.max(m)));
    if hi - lo > 100.0 {
        // x already spans the probe: reference = midline of the leftmost shank
        geom.iter().map(|g| g.x_um as f64 - lo).collect()
    } else {
        let center = geom.iter().map(|g| g.x_um as f64).sum::<f64>() / geom.len().max(1) as f64;
        geom.iter()
            .map(|g| g.shank as f64 * SHANK_PITCH_UM + g.x_um as f64 - center)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Regions along one shank, sampled every `TABLE_STEP_UM` from the brain-surface
/// entry depth (0) down to the tip.
pub struct ShankSamples {
    pub shank: u32,
    /// finest-level region per sample, None = outside the brain
    pub regions: Vec<Option<usize>>,
    /// depth range (µm from the entry point, along the axis) covered by recorded channels
    pub recorded_um: Option<(f64, f64)>,
}

pub struct Registration {
    /// finest-level region per raw channel index, None = outside the brain
    pub channel_regions: Vec<Option<usize>>,
    pub shanks: Vec<ShankSamples>,
    /// brain-surface entry point (bregma mm [ML, AP, DV])
    pub entry: [f64; 3],
    /// probe tip of the reference shank (bregma mm [ML, AP, DV])
    pub tip: [f64; 3],
    pub warnings: Vec<String>,
}

/// Find where a line first enters the brain (NTE: `update_probe_areas_coordinates`).
/// Returns the line parameter (mm) of the first in-brain sample.
fn first_in_brain(atlas: &Atlas, tf: &CcfTransform, origin: [f64; 3], dir: [f64; 3], t0: f64, t1: f64) -> Option<f64> {
    let n = ((t1 - t0) / TRAJECTORY_STEP_MM).ceil() as usize;
    (0..=n)
        .map(|i| t0 + i as f64 * TRAJECTORY_STEP_MM)
        .find(|&t| atlas.region_at(add(origin, scale(dir, t)), tf).is_some())
}

pub fn register(
    atlas: &Atlas,
    meta: &Meta,
    ins: &Insertion,
    cancel: &AtomicBool,
    progress: &AtomicUsize,
) -> Result<Option<Registration>> {
    let tf = CcfTransform::new(ins.bregma_lambda_mm);
    let base = PROGRESS_TOTAL * 3 / 10;
    let set_progress = |frac: f64| {
        progress.store(base + ((PROGRESS_TOTAL - base) as f64 * frac) as usize, Ordering::Relaxed)
    };

    // 1. brain surface straight below the entered AP/ML (NTE: set_probe_entry)
    let top = [ins.ml_mm, ins.ap_mm, -1.0];
    let down = [0.0, 0.0, 1.0];
    let Some(t_surf) = first_in_brain(atlas, &tf, top, down, 0.0, 7.0) else {
        bail!("no brain found below AP {:.2} / ML {:.2} mm", ins.ap_mm, ins.ml_mm);
    };
    let surface = add(top, scale(down, t_surf));
    set_progress(0.2);
    if cancel.load(Ordering::Relaxed) {
        return Ok(None);
    }

    // 2. entry point along the (angled) trajectory through that surface point
    let dir = insertion_direction(ins.azimuth_deg, ins.elevation_deg);
    let reach = 15.0; // mm, longer than the atlas diagonal
    let t_entry = first_in_brain(atlas, &tf, surface, dir, -reach, reach)
        .context("the trajectory does not enter the brain")?;
    let entry = add(surface, scale(dir, t_entry));
    let tip = add(entry, scale(dir, ins.depth_mm));
    set_progress(0.4);
    if cancel.load(Ordering::Relaxed) {
        return Ok(None);
    }

    // 3. region under every channel
    let rot = probe_rotation(ins.azimuth_deg, ins.elevation_deg, ins.rotation_deg);
    let lateral = lateral_positions_um(meta);
    let along: Vec<f64> = meta.channel_geom.iter().map(|g| ins.tip_offset_um + g.y_um as f64).collect();
    let channel_regions: Vec<Option<usize>> = (0..meta.channel_geom.len())
        .map(|ch| {
            let local = [lateral[ch] / 1000.0, 0.0, -along[ch] / 1000.0];
            atlas.region_at(add(tip, mat_vec(&rot, local)), &tf)
        })
        .collect();
    set_progress(0.6);
    if cancel.load(Ordering::Relaxed) {
        return Ok(None);
    }

    // 4. per-shank samples for the region table (shank line = mean lateral offset)
    let mut shank_ids: Vec<u32> = meta.channel_geom.iter().map(|g| g.shank).collect();
    shank_ids.sort_unstable();
    shank_ids.dedup();
    let depth_um = ins.depth_mm * 1000.0;
    let n_samples = (depth_um.max(0.0) / TABLE_STEP_UM).floor() as usize + 1;
    let mut shanks = Vec::new();
    for (i, &s) in shank_ids.iter().enumerate() {
        let chans: Vec<usize> = (0..meta.channel_geom.len()).filter(|&c| meta.channel_geom[c].shank == s).collect();
        let lat_um = chans.iter().map(|&c| lateral[c]).sum::<f64>() / chans.len().max(1) as f64;
        let regions = (0..n_samples)
            .map(|k| {
                let d_um = k as f64 * TABLE_STEP_UM;
                let local = [lat_um / 1000.0, 0.0, -(depth_um - d_um) / 1000.0];
                atlas.region_at(add(tip, mat_vec(&rot, local)), &tf)
            })
            .collect();
        let recorded_um = chans.iter().fold(None, |acc: Option<(f64, f64)>, &c| {
            let d = depth_um - along[c];
            Some(acc.map_or((d, d), |(lo, hi)| (lo.min(d), hi.max(d))))
        });
        shanks.push(ShankSamples { shank: s, regions, recorded_um });
        set_progress(0.6 + 0.4 * (i + 1) as f64 / shank_ids.len() as f64);
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
    }

    let mut warnings = Vec::new();
    if depth_um > SHANK_LENGTH_UM {
        warnings.push(format!(
            "insertion depth ({:.2} mm) exceeds the {:.0} mm shank length",
            ins.depth_mm,
            SHANK_LENGTH_UM / 1000.0
        ));
    }
    if along.iter().any(|&a| a > SHANK_LENGTH_UM) {
        warnings.push("some electrodes would lie beyond the end of the shank — check the tip offset".into());
    }
    let outside = channel_regions.iter().filter(|r| r.is_none()).count();
    if outside == channel_regions.len() {
        warnings.push("all channels are outside the brain".into());
    } else if outside > 0 {
        warnings.push(format!("{outside} channel(s) outside the brain"));
    }

    progress.store(PROGRESS_TOTAL, Ordering::Relaxed);
    Ok(Some(Registration { channel_regions, shanks, entry, tip, warnings }))
}

/// One contiguous stretch of a region along a shank, for the region table.
pub struct Segment {
    pub region: Option<usize>,
    pub from_um: f64,
    pub to_um: f64,
    pub recorded: bool,
}

/// Merge a shank's samples into segments at the given hierarchy level.
pub fn shank_segments(samples: &ShankSamples, tree: &StructureTree, level: Option<u32>) -> Vec<Segment> {
    let mut segs: Vec<Segment> = Vec::new();
    for (k, r) in samples.regions.iter().enumerate() {
        let r = r.map(|r| tree.at_level(r, level));
        let d = k as f64 * TABLE_STEP_UM;
        match segs.last_mut() {
            Some(s) if s.region == r => s.to_um = d + TABLE_STEP_UM,
            _ => segs.push(Segment { region: r, from_um: d, to_um: d + TABLE_STEP_UM, recorded: false }),
        }
    }
    if let (Some(last), Some(n)) = (segs.last_mut(), samples.regions.len().checked_sub(1)) {
        last.to_um = last.to_um.min(n as f64 * TABLE_STEP_UM);
    }
    if let Some((lo, hi)) = samples.recorded_um {
        for s in &mut segs {
            s.recorded = s.from_um <= hi && s.to_um >= lo;
        }
    }
    segs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_bytes(shape: [usize; 3], fortran: bool, f: impl Fn(usize, usize, usize) -> u16) -> Vec<u8> {
        let header = format!(
            "{{'descr': '<u2', 'fortran_order': {}, 'shape': ({}, {}, {}), }}",
            if fortran { "True" } else { "False" },
            shape[0],
            shape[1],
            shape[2]
        );
        let mut h = header.into_bytes();
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(b' ');
        }
        h.push(b'\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        out.extend_from_slice(&h);
        let [a, b, c] = shape;
        let mut push = |i: usize, j: usize, k: usize| out.extend_from_slice(&f(i, j, k).to_le_bytes());
        if fortran {
            for k in 0..c { for j in 0..b { for i in 0..a { push(i, j, k) } } }
        } else {
            for i in 0..a { for j in 0..b { for k in 0..c { push(i, j, k) } } }
        }
        out
    }

    const TREE_CSV: &str = "\
id,atlas_id,name,acronym,st_level,ontology_id,hemisphere_id,weight,parent_structure_id,depth,graph_id,graph_order,structure_id_path,color_hex_triplet,neuro_name_structure_id,neuro_name_structure_id_path,failed,sphinx_id,structure_name_facet,failed_facet,safe_name
997,-1,root,root,,1,3,8690,,0,1,0,/997/,FFFFFF,,,f,1,385153371,734881840,root
315,,Isocortex,Isocortex,,1,3,8690,997,1,1,1,/997/315/,70FF71,,,f,2,1,1,Isocortex
385,,\"Primary visual area\",VISp,,1,3,8690,315,2,1,2,/997/315/385/,08858C,,,f,3,1,1,\"Primary visual area\"
593,,\"Primary visual area, layer 1\",VISp1,,1,3,8690,385,3,1,3,/997/315/385/593/,08858C,,,f,4,1,1,\"Primary visual area, layer 1\"
721,,\"Primary visual area, layer 4\",VISp4,,1,3,8690,385,3,1,4,/997/315/385/721/,08858C,,,f,5,1,1,\"Primary visual area, layer 4\"
";

    #[test]
    fn csv_quotes_and_hierarchy() {
        let t = StructureTree::parse(TREE_CSV).unwrap();
        assert_eq!(t.rows.len(), 5);
        assert_eq!(t.rows[3].name, "Primary visual area, layer 1");
        assert_eq!(t.rows[3].acronym, "VISp1");
        assert_eq!(t.rows[3].parent, Some(2));
        assert_eq!(t.at_level(3, None), 3);
        assert_eq!(t.at_level(3, Some(2)), 2);
        assert_eq!(t.at_level(3, Some(1)), 1);
        assert_eq!(t.at_level(1, Some(2)), 1);
    }

    #[test]
    fn npy_c_and_fortran_order() {
        let f = |i: usize, j: usize, k: usize| (i * 100 + j * 10 + k) as u16;
        for fortran in [false, true] {
            let vol = AnnotationVolume::from_bytes(VolumeData::Owned(npy_bytes([3, 4, 5], fortran, f))).unwrap();
            assert_eq!(vol.shape, [3, 4, 5]);
            assert_eq!(vol.get(2, 3, 4), Some(234));
            assert_eq!(vol.get(1, 0, 2), Some(102));
            assert_eq!(vol.get(3, 0, 0), None);
        }
    }

    #[test]
    fn transform_round_trip_and_bregma() {
        let tf = CcfTransform::new(4.5);
        let b = tf.ccf_to_bregma(BREGMA_CCF);
        assert!(b.iter().all(|v| v.abs() < 1e-12));
        let p = [1.3, -2.1, 3.7];
        let back = tf.ccf_to_bregma(tf.bregma_to_ccf(p));
        for i in 0..3 {
            assert!((back[i] - p[i]).abs() < 1e-9);
        }
        // moving posterior in the CCF (higher AP index) is negative AP in bregma space
        assert!(CcfTransform::new(4.1).ccf_to_bregma([570.5, 620.0, 44.0])[1] < 0.0);
    }

    #[test]
    fn angle_conventions() {
        // vertical
        let d = insertion_direction(0.0, 90.0);
        assert!(d[0].abs() < 1e-12 && d[1].abs() < 1e-12 && (d[2] - 1.0).abs() < 1e-12);
        // azimuth 0 tilts the tip anteriorly, 90 tilts it to +ML
        assert!(insertion_direction(0.0, 45.0)[1] > 0.0);
        assert!(insertion_direction(90.0, 45.0)[0] > 0.0);
        for (az, el) in [(0.0, 90.0), (37.0, 55.0), (270.0, 80.0)] {
            let (t, p) = az_el_to_polar(az, el);
            let (az2, el2) = polar_to_az_el(t, p);
            let diff = (az2 - az).rem_euclid(360.0);
            assert!(diff < 1e-9 || diff > 360.0 - 1e-9);
            assert!((el2 - el).abs() < 1e-9);
        }
    }

    #[test]
    fn rotation_matches_direction_and_rolls_shanks() {
        for (az, el, rot) in [(0.0, 90.0, 0.0), (30.0, 60.0, 0.0), (200.0, 75.0, 45.0)] {
            let r = probe_rotation(az, el, rot);
            let along = mat_vec(&r, [0.0, 0.0, 1.0]);
            let d = insertion_direction(az, el);
            for i in 0..3 {
                assert!((along[i] - d[i]).abs() < 1e-9);
            }
        }
        // vertical probe, no rotation: shanks spread along ML; 90° rotation -> along AP
        let x0 = mat_vec(&probe_rotation(0.0, 90.0, 0.0), [1.0, 0.0, 0.0]);
        assert!((x0[0].abs() - 1.0).abs() < 1e-9 && x0[1].abs() < 1e-9);
        let x90 = mat_vec(&probe_rotation(0.0, 90.0, 90.0), [1.0, 0.0, 0.0]);
        assert!(x90[0].abs() < 1e-9 && (x90[1].abs() - 1.0).abs() < 1e-9);
    }

    /// Synthetic atlas: brain below CCF DV voxel 100, layer 1 above DV 150, layer 4 below.
    fn synthetic_atlas() -> Atlas {
        fn f(_ap: usize, dv: usize, _ml: usize) -> u16 {
            if dv < 100 { 1 } else if dv < 150 { 4 } else { 5 }
        }
        Atlas {
            dir: PathBuf::new(),
            volume: AnnotationVolume {
                data: VolumeData::Owned(Vec::new()),
                offset: 0,
                shape: [1320, 800, 1140],
                fortran_order: false,
                synthetic: Some(f),
            },
            tree: StructureTree::parse(TREE_CSV).unwrap(),
        }
    }

    fn vertical_meta() -> Meta {
        Meta {
            n_saved_chans: 3,
            n_ap_chans: 3,
            sample_rate: 30000.0,
            n_samples: 0,
            uv_per_bit: vec![1.0; 3],
            im_dat_prb_type: 0,
            sample_shift: vec![0.0; 3],
            warnings: Vec::new(),
            reference_channels: Default::default(),
            channel_ids: vec!["AP0".into(), "AP1".into(), "AP2".into()],
            channel_geom: vec![
                crate::data::ChannelGeom { x_um: 0.0, y_um: 0.0, shank: 0 },
                crate::data::ChannelGeom { x_um: 0.0, y_um: 400.0, shank: 0 },
                crate::data::ChannelGeom { x_um: 0.0, y_um: 3000.0, shank: 0 },
            ],
        }
    }

    #[test]
    fn vertical_registration() {
        let atlas = synthetic_atlas();
        let meta = vertical_meta();
        let ins = Insertion { depth_mm: 1.0, tip_offset_um: 200.0, ..Default::default() };
        let reg = register(&atlas, &meta, &ins, &AtomicBool::new(false), &AtomicUsize::new(0))
            .unwrap()
            .unwrap();
        let tf = CcfTransform::new(ins.bregma_lambda_mm);
        // entry is at the first in-brain voxel (DV index 100 -> 1-based 101, rounded)
        let dv_entry = tf.bregma_to_ccf(reg.entry)[2];
        assert!((dv_entry - 100.5).abs() < 0.2, "entry DV voxel {dv_entry}");
        // layer 1 spans 50 voxels * 8.85 µm ≈ 0.44 mm below the surface. Channel 0 is
        // 0.8 mm deep (layer 4), channel 1 is 0.4 mm deep (layer 1), channel 2 is above the brain.
        assert_eq!(reg.channel_regions, vec![Some(4), Some(3), None]);
        let segs = shank_segments(&reg.shanks[0], &atlas.tree, None);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].region, Some(3));
        assert!((segs[0].to_um - 440.0).abs() <= 20.0, "boundary at {}", segs[0].to_um);
        let coarse = shank_segments(&reg.shanks[0], &atlas.tree, Some(2));
        assert_eq!(coarse.len(), 1);
        assert_eq!(coarse[0].region, Some(2));
    }

    #[test]
    fn lateral_offsets_both_geometry_styles() {
        let g = |x: f32, shank: u32| crate::data::ChannelGeom { x_um: x, y_um: 0.0, shank };
        let mut meta = vertical_meta();
        // SpikeGLX: x within each shank (0/32), shanks 250 µm apart
        meta.channel_geom = vec![g(0.0, 0), g(32.0, 0), g(0.0, 1), g(32.0, 1)];
        assert_eq!(lateral_positions_um(&meta), vec![-16.0, 16.0, 234.0, 266.0]);
        // Open Ephys: x across the whole probe
        meta.channel_geom = vec![g(0.0, 0), g(32.0, 0), g(250.0, 1), g(282.0, 1)];
        assert_eq!(lateral_positions_um(&meta), vec![-16.0, 16.0, 234.0, 266.0]);
    }

    #[test]
    fn regions_csv() {
        let dir = std::env::temp_dir().join(format!("npx_regions_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("rec_g0_t0.imec0.ap.cbin");
        let path = save_regions_csv(&bin, &[("AP0".into(), "VISp4".into()), ("AP1".into(), "outside".into())]).unwrap();
        assert_eq!(path, dir.join("rec_g0_t0.imec0.ap_regions.csv"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "channel,region\nAP0,VISp4\nAP1,outside\n");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sidecar_round_trip_with_edits() {
        let dir = std::env::temp_dir().join(format!("npx_sidecar_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("rec.imec0.ap.bin");
        let ins = Insertion { ap_mm: -2.5, depth_mm: 3.2, ..Default::default() };
        let edits = vec![
            RegionEdit { shank: 0, from_um: 100.0, to_um: 140.0, region: Some(385) },
            RegionEdit { shank: 1, from_um: 0.0, to_um: 20.0, region: None },
        ];
        save_sidecar(&bin, &ins, &edits).unwrap();
        let (ins2, edits2) = load_sidecar(&bin).unwrap();
        assert_eq!(ins2, ins);
        assert_eq!(edits2, edits);
        // a level is kept too, and a sidecar without edits still loads
        let ins = Insertion { level: Some(5), ..ins };
        save_sidecar(&bin, &ins, &[]).unwrap();
        assert_eq!(load_sidecar(&bin).unwrap(), (ins, vec![]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn upsert_cuts_replaces_and_merges() {
        let e = |from_um: f64, to_um: f64, region: Option<u32>| RegionEdit { shank: 0, from_um, to_um, region };
        let mut edits = vec![e(0.0, 100.0, Some(1))];
        // overwrite the middle with another region: splits the old edit
        upsert_edit(&mut edits, e(40.0, 60.0, Some(2)), true);
        assert_eq!(edits, vec![e(0.0, 40.0, Some(1)), e(40.0, 60.0, Some(2)), e(60.0, 100.0, Some(1))]);
        // give it back the first region: everything merges again
        upsert_edit(&mut edits, e(40.0, 60.0, Some(1)), true);
        assert_eq!(edits, vec![e(0.0, 100.0, Some(1))]);
        // revert a stretch to the atlas: cut out, nothing added
        upsert_edit(&mut edits, e(80.0, 120.0, Some(9)), false);
        assert_eq!(edits, vec![e(0.0, 80.0, Some(1))]);
        assert_eq!(edit_at(&edits, 0, 79.9).map(|x| x.region), Some(Some(1)));
        assert!(edit_at(&edits, 0, 80.0).is_none());
        assert!(edit_at(&edits, 1, 10.0).is_none());
    }

    #[test]
    fn edits_follow_depth_changes() {
        // an electrode's brain depth shifts one-for-one with the insertion depth, so a
        // range stored in brain depth stays attached to the same stretch of brain
        let ins = Insertion { depth_mm: 3.0, tip_offset_um: 195.0, ..Default::default() };
        assert_eq!(brain_depth_um(&ins, 405.0), 2400.0);
        let deeper = Insertion { depth_mm: 3.1, ..ins.clone() };
        assert!((brain_depth_um(&deeper, 505.0) - 2400.0).abs() < 1e-9);
    }

    #[test]
    fn sidecar_name_shared_between_bands() {
        let a = sidecar_path(Path::new("/d/rec_g0_t0.imec0.ap.bin"));
        let b = sidecar_path(Path::new("/d/rec_g0_t0.imec0.lf.bin"));
        assert_eq!(a, b);
        assert_eq!(a, Path::new("/d/rec_g0_t0.imec0.npx_atlas.toml"));
    }
}
