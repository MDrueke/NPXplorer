mod app;
mod data;
mod geometry;
mod preprocess;
mod render;
mod worker;
mod mtscomp;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rawviewer", about = "Neuropixels raw data viewer")]
struct Args {
    /// Path to the .ap.bin SpikeGLX file
    #[arg(short, long)]
    file: Option<PathBuf>,
}

enum AppState {
    Empty,
    Loaded(app::RawViewerApp),
}

struct MainApp {
    state: AppState,
}

impl MainApp {
    fn new(ctx: &egui::Context, bin_path: Option<PathBuf>) -> Self {
        let mut visuals = egui::Visuals::dark();
        let bg_color = egui::Color32::from_rgb(0x18, 0x18, 0x18);
        visuals.panel_fill = bg_color;
        visuals.window_fill = bg_color;
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(0x28, 0x28, 0x28); // slightly lighter for hover
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0x38, 0x38, 0x38);  // slightly lighter for active
        visuals.extreme_bg_color = visuals.widgets.inactive.bg_fill; // make text edits match sliders
        ctx.set_visuals(visuals);
        let state = if let Some(path) = bin_path {
            match app::RawViewerApp::new(ctx, path) {
                Ok(a) => AppState::Loaded(a),
                Err(e) => {
                    eprintln!("Error opening file: {e}");
                    AppState::Empty
                }
            }
        } else {
            AppState::Empty
        };
        Self { state }
    }
}

impl eframe::App for MainApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut file_to_open = None;

        match &mut self.state {
            AppState::Empty => {
                egui::TopBottomPanel::top("empty_top_bar").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.menu_button("File", |ui| {
                            if ui.button("Open").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("Recordings", &["bin", "cbin"])
                                    .add_filter("Uncompressed", &["bin"])
                                    .add_filter("Compressed", &["cbin"])
                                    .pick_file() {
                                    file_to_open = Some(path);
                                }
                                ui.close_menu();
                            }
                        });
                    });
                });

                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.centered_and_justified(|ui| {
                        if ui.add(egui::Button::new("open a file to start").frame(false)).clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Recordings", &["bin", "cbin"])
                                .add_filter("Uncompressed", &["bin"])
                                .add_filter("Compressed", &["cbin"])
                                .pick_file() {
                                file_to_open = Some(path);
                            }
                        }
                    });
                });
            }
            AppState::Loaded(app) => {
                app.update(ctx);
                if app.file_dialog_request {
                    app.file_dialog_request = false;
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Recordings", &["bin", "cbin"])
                        .add_filter("Uncompressed", &["bin"])
                        .add_filter("Compressed", &["cbin"])
                        .pick_file() {
                        file_to_open = Some(path);
                    }
                }
            }
        }

        if let Some(path) = file_to_open {
            match app::RawViewerApp::new(ctx, path) {
                Ok(a) => self.state = AppState::Loaded(a),
                Err(e) => eprintln!("Error opening file: {e}"),
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

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("RawViewer")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "RawViewer",
        options,
        Box::new(move |cc| {
            Ok(Box::new(MainApp::new(&cc.egui_ctx, args.file)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;

    Ok(())
}
