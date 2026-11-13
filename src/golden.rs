//! Golden files for the bit-exact regression tests on real recordings (compiled into
//! the tests of both the library and the app).

use std::path::PathBuf;

/// Golden files for bit-exact regression checks: `check` stores an output the first
/// time (or always with `NPX_GOLDEN_WRITE=1`) and compares it on later runs.
pub struct Golden {
    dir: PathBuf,
    write: bool,
    mismatches: Vec<String>,
}

impl Golden {
    /// From `NPX_GOLDEN_DIR`; `None` (test skipped) if it isn't set.
    pub fn from_env() -> Option<Self> {
        let Some(dir) = std::env::var_os("NPX_GOLDEN_DIR").map(PathBuf::from) else {
            eprintln!("skipping: NPX_GOLDEN_DIR not set");
            return None;
        };
        std::fs::create_dir_all(&dir).unwrap();
        Some(Golden { dir, write: std::env::var_os("NPX_GOLDEN_WRITE").is_some(), mismatches: Vec::new() })
    }

    pub fn dir(&self) -> &std::path::Path {
        &self.dir
    }

    /// `.f32` outputs report the largest difference when they differ.
    pub fn check(&mut self, name: &str, bytes: &[u8]) {
        let path = self.dir.join(name);
        if self.write || !path.is_file() {
            std::fs::write(&path, bytes).unwrap();
            println!("  {name}: stored ({} bytes)", bytes.len());
            return;
        }
        let want = std::fs::read(&path).unwrap();
        if want == bytes {
            println!("  {name}: identical");
            return;
        }
        let diff = if want.len() == bytes.len() && name.ends_with(".f32") {
            let a: &[f32] = bytemuck::cast_slice(&want);
            let b: &[f32] = bytemuck::cast_slice(bytes);
            let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            format!("max |diff| {d:e}")
        } else if want.len() == bytes.len() {
            format!("{} of {} bytes differ", want.iter().zip(bytes).filter(|(a, b)| a != b).count(), want.len())
        } else {
            format!("{} vs {} bytes", want.len(), bytes.len())
        };
        println!("  {name}: DIFFERENT ({diff})");
        self.mismatches.push(name.to_string());
    }

    /// Fails the test if any output differed.
    pub fn finish(self) {
        assert!(self.mismatches.is_empty(), "outputs differ from the golden files: {:?}", self.mismatches);
    }
}
