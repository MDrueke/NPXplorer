//! App icon, embedded in the binary.
//!
//! - Window / taskbar icon (Windows, X11): `window_icon()`, from icon/logo.png.
//! - Windows .exe icon (Explorer, shortcuts): icon/logo.ico, embedded by build.rs.
//! - Wayland: compositors don't take an icon from the window; they look up the
//!   `.desktop` file named after the window's app id. `install_desktop_entry()`
//!   writes that file and the icon to ~/.local/share on startup.

/// Window app id (Wayland) / WM_CLASS (X11); must match the .desktop file name.
pub const APP_ID: &str = "npxplorer";

const LOGO_PNG: &[u8] = include_bytes!("../icon/logo.png");

pub fn window_icon() -> egui::IconData {
    eframe::icon_data::from_png_bytes(LOGO_PNG).expect("embedded icon/logo.png is a valid PNG")
}

/// Write (or refresh) the user-level .desktop entry and icon, so Wayland compositors,
/// docks and app launchers show the logo. Only writes files whose content changed;
/// failures are logged and otherwise ignored, since the app works without them.
#[cfg(target_os = "linux")]
pub fn install_desktop_entry() {
    use std::path::PathBuf;

    let Some(data_home) = std::env::var_os("XDG_DATA_HOME")
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    else {
        return;
    };
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    for (path, result) in install_into(&data_home, &exe) {
        if let Err(e) = result {
            crate::file_log!("desktop integration: writing {} failed: {e}", path.display());
        }
    }
}

/// Write the icon and .desktop entry for `exe` below `data_home`; returns each file's
/// path and write result.
#[cfg(target_os = "linux")]
fn install_into(
    data_home: &std::path::Path,
    exe: &std::path::Path,
) -> Vec<(std::path::PathBuf, std::io::Result<()>)> {
    fn write_if_changed(path: &std::path::Path, content: &[u8]) -> std::io::Result<()> {
        if std::fs::read(path).is_ok_and(|old| old == content) {
            return Ok(());
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, content)
    }

    // Exec value: quoted, with the characters the desktop-entry spec reserves
    // inside quotes escaped; a backslash is unescaped twice (string value, then
    // quoting), so it needs four
    let exe_quoted = exe
        .to_string_lossy()
        .chars()
        .fold(String::new(), |mut s, c| {
            match c {
                '\\' => s.push_str("\\\\\\\\"),
                '"' | '`' | '$' => {
                    s.push('\\');
                    s.push(c);
                }
                _ => s.push(c),
            }
            s
        });
    let desktop = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=NPXplorer\n\
         Comment=Neuropixels recording explorer\n\
         Exec=\"{exe_quoted}\" %f\n\
         Icon={APP_ID}\n\
         Terminal=false\n\
         Categories=Science;\n\
         StartupWMClass={APP_ID}\n"
    );

    let icon_path = data_home.join(format!("icons/hicolor/512x512/apps/{APP_ID}.png"));
    let desktop_path = data_home.join(format!("applications/{APP_ID}.desktop"));
    [(icon_path, LOGO_PNG), (desktop_path, desktop.as_bytes())]
        .into_iter()
        .map(|(path, content)| {
            let result = write_if_changed(&path, content);
            (path, result)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_icon_decodes() {
        let icon = window_icon();
        assert!(icon.width > 0 && icon.rgba.len() == (icon.width * icon.height * 4) as usize);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn desktop_entry_written_and_idempotent() {
        let dir = std::env::temp_dir().join(format!("npxplorer_icon_test_{}", std::process::id()));
        let exe = std::path::Path::new("/opt/my apps/NPX\\$plorer");
        for (path, r) in install_into(&dir, exe) {
            r.unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        }
        let desktop = std::fs::read_to_string(dir.join("applications/npxplorer.desktop")).unwrap();
        assert!(desktop.contains("Exec=\"/opt/my apps/NPX\\\\\\\\\\$plorer\" %f"), "{desktop}");
        assert_eq!(std::fs::read(dir.join("icons/hicolor/512x512/apps/npxplorer.png")).unwrap(), LOGO_PNG);
        // second run leaves the files alone
        let mtime = |p: &str| std::fs::metadata(dir.join(p)).unwrap().modified().unwrap();
        let before = mtime("applications/npxplorer.desktop");
        install_into(&dir, exe);
        assert_eq!(mtime("applications/npxplorer.desktop"), before);
        let status = std::process::Command::new("desktop-file-validate")
            .arg(dir.join("applications/npxplorer.desktop"))
            .status();
        if let Ok(status) = status {
            assert!(status.success(), "desktop-file-validate rejected the entry");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
