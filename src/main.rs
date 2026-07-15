pub static DEBUG_LOGGING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Logging macro that only writes when --debug is active.
/// Writes to debug.log next to the executable with timestamps.
#[macro_export]
macro_rules! file_log {
    ($($arg:tt)*) => {
        if $crate::DEBUG_LOGGING.load(std::sync::atomic::Ordering::Relaxed) {
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

mod app;
mod data;
mod geometry;
mod mtscomp;
mod preprocess;
mod render;
mod worker;

use clap::Parser;
use std::path::PathBuf;
use std::sync::mpsc;

/// Spawn the native file picker on a background thread. `rfd::FileDialog::pick_file`
/// blocks until the user responds; calling it directly inside egui's `update()` stalls
/// winit's event loop, which on Linux (GTK/portal-backed dialogs) can prevent the
/// dialog itself from ever gaining focus. Running it off-thread and polling the
/// channel keeps the UI event loop pumping while the dialog is open.
fn spawn_file_picker(last_dir: Option<PathBuf>) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut dlg = rfd::FileDialog::new()
            .add_filter("Recordings", &["bin", "cbin", "dat"])
            .add_filter("SpikeGLX uncompressed", &["bin"])
            .add_filter("Compressed (SpikeGLX or Open Ephys)", &["cbin"])
            .add_filter("Open Ephys uncompressed", &["dat"]);
        if let Some(dir) = last_dir {
            dlg = dlg.set_directory(dir);
        }
        let _ = tx.send(dlg.pick_file());
    });
    rx
}

#[derive(Parser)]
#[command(name = "npxplorer", about = "Neuropixels raw data viewer")]
struct Args {
    /// Path to a SpikeGLX .ap.bin/.ap.cbin file, or an Open Ephys continuous.dat/.cbin
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Enable debug logging to debug.log next to the executable
    #[arg(long)]
    debug: bool,
}

enum AppState {
    Empty,
    Loaded(app::NPXplorerApp),
}

struct MainApp {
    state: AppState,
    last_dir: Option<PathBuf>,
    error_msg: Option<String>,
    pending_pick: Option<mpsc::Receiver<Option<PathBuf>>>,
}

impl MainApp {
    fn new(ctx: &egui::Context, bin_path: Option<PathBuf>) -> Self {
        let mut visuals = egui::Visuals::dark();
        let bg_color = egui::Color32::from_rgb(
            crate::render::C_ZERO[0],
            crate::render::C_ZERO[1],
            crate::render::C_ZERO[2],
        );
        visuals.panel_fill = bg_color;
        visuals.window_fill = bg_color;
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(0x28, 0x28, 0x28);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0x38, 0x38, 0x38);
        visuals.extreme_bg_color = visuals.widgets.inactive.bg_fill;
        ctx.set_visuals(visuals);
        ctx.options_mut(|o| o.theme_preference = egui::ThemePreference::Dark);

        let last_dir = app::Preferences::load()
            .and_then(|p| p.last_dir)
            .map(PathBuf::from)
            .or_else(|| {
                bin_path
                    .as_ref()
                    .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            });

        let mut error_msg = None;
        let state = if let Some(path) = bin_path {
            match app::NPXplorerApp::new(ctx, path) {
                Ok(a) => AppState::Loaded(a),
                Err(e) => {
                    error_msg = Some(format!("{e}"));
                    AppState::Empty
                }
            }
        } else {
            AppState::Empty
        };
        Self {
            state,
            last_dir,
            error_msg,
            pending_pick: None,
        }
    }

    fn custom_title_bar(&self, ctx: &egui::Context) {
        let title_bar_height = 24.0;
        let title_bar_color = egui::Color32::from_rgb(
            crate::render::C_ZERO[0],
            crate::render::C_ZERO[1],
            crate::render::C_ZERO[2],
        );

        egui::TopBottomPanel::top("custom_title_bar")
            .frame(egui::Frame::NONE.fill(title_bar_color))
            .exact_height(title_bar_height)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Drag area
                    let drag_rect = ui.available_rect_before_wrap();
                    let response = ui.interact(
                        drag_rect,
                        ui.id().with("drag_title_bar"),
                        egui::Sense::click_and_drag(),
                    );
                    if response.drag_started() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                    }
                    if response.double_clicked() {
                        let is_maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                        ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!is_maximized));
                    }

                    // Title text
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("NPXplorer v0.5")
                                .color(egui::Color32::WHITE)
                                .size(14.0),
                        );
                    });

                    // Window controls
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(4.0);
                        ui.spacing_mut().item_spacing.x = 0.0;
                        ui.style_mut().visuals.widgets.inactive.bg_fill =
                            egui::Color32::TRANSPARENT;

                        if ui.button(" 🗙 ").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ui.button(" 🗖 ").clicked() {
                            let is_maximized =
                                ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!is_maximized));
                        }
                        if ui.button(" 🗕 ").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                        }
                    });
                });
            });
    }
}

impl eframe::App for MainApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.custom_title_bar(ctx);
        let mut file_to_open = None;

        // poll any in-flight file picker rather than blocking on it
        if let Some(rx) = &self.pending_pick {
            match rx.try_recv() {
                Ok(picked) => {
                    self.pending_pick = None;
                    if let Some(path) = picked {
                        file_to_open = Some(path);
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pending_pick = None;
                }
            }
        }

        match &mut self.state {
            AppState::Empty => {
                egui::TopBottomPanel::top("empty_top_bar").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.menu_button("File", |ui| {
                            if ui.button("Open").clicked() {
                                if self.pending_pick.is_none() {
                                    self.pending_pick =
                                        Some(spawn_file_picker(self.last_dir.clone()));
                                }
                                ui.close_menu();
                            }
                        });
                    });
                });

                egui::CentralPanel::default().show(ctx, |ui| {
                    let rect = ui.available_rect_before_wrap();
                    let response = ui
                        .interact(rect, ui.id().with("open_file_prompt"), egui::Sense::click())
                        .on_hover_cursor(egui::CursorIcon::PointingHand);
                    let text_color = if response.hovered() {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::GRAY
                    };
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Open a file to start",
                        egui::FontId::proportional(16.0),
                        text_color,
                    );
                    if let Some(err) = &self.error_msg {
                        ui.painter().text(
                            rect.center() + egui::vec2(0.0, 28.0),
                            egui::Align2::CENTER_CENTER,
                            format!("Error: {err}"),
                            egui::FontId::proportional(13.0),
                            egui::Color32::from_rgb(0xff, 0x66, 0x66),
                        );
                    }
                    if response.clicked() && self.pending_pick.is_none() {
                        self.pending_pick = Some(spawn_file_picker(self.last_dir.clone()));
                    }
                });
            }
            AppState::Loaded(app) => {
                app.update(ctx);
                if app.file_dialog_request {
                    app.file_dialog_request = false;
                    if self.pending_pick.is_none() {
                        self.pending_pick = Some(spawn_file_picker(self.last_dir.clone()));
                    }
                }
            }
        }

        if let Some(path) = file_to_open {
            self.last_dir = path.parent().map(|p| p.to_path_buf());
            match app::NPXplorerApp::new(ctx, path) {
                Ok(a) => {
                    self.state = AppState::Loaded(a);
                    self.error_msg = None;
                }
                Err(e) => {
                    self.error_msg = Some(format!("{e}"));
                    self.state = AppState::Empty;
                }
            }
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let AppState::Loaded(app) = &self.state {
            app.save_prefs();
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.debug {
        DEBUG_LOGGING.store(true, std::sync::atomic::Ordering::Relaxed);
        file_log!("=== NPXplorer debug logging started ===");
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NPXplorer v0.2")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 500.0])
            .with_decorations(false),
        ..Default::default()
    };

    eframe::run_native(
        "NPXplorer v0.2",
        options,
        Box::new(move |cc| Ok(Box::new(MainApp::new(&cc.egui_ctx, args.file)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;

    Ok(())
}
