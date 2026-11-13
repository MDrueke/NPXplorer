//! Debug logging (`--debug`): `file_log!` appends timestamped lines to debug.log next
//! to the executable, and does nothing otherwise.

/// Set once at startup by `--debug`.
pub static DEBUG_LOGGING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Logging macro that only writes when --debug is active.
/// Writes to debug.log next to the executable with timestamps.
#[macro_export]
macro_rules! file_log {
    ($($arg:tt)*) => {
        if $crate::log::DEBUG_LOGGING.load(std::sync::atomic::Ordering::Relaxed) {
            let log_path = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("debug.log")))
                .unwrap_or_else(|| std::path::PathBuf::from("debug.log"));
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log_path) {
                use std::io::Write;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                let _ = write!(f, "[{:.3}] ", now);
                let _ = writeln!(f, $($arg)*);
            }
        }
    };
}
