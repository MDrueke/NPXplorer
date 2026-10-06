//! Embeds icon/logo.ico into the Windows executable, so Explorer, shortcuts and the
//! taskbar show the app icon before the window opens. No-op for other targets.

fn main() {
    println!("cargo:rerun-if-changed=icon/logo.ico");
    // build scripts run on the host: check the target, so cross-compiling from
    // Linux to Windows embeds the icon too (via mingw's windres)
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("icon/logo.ico");
        res.compile().expect("embedding the Windows icon resource failed");
    }
}
