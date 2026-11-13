use super::*;

/// A short synthetic NP 1.0 recording without a geometry map (so the reference
/// site comes from the probe type); returns the AP and LF data files.
fn write_recording(dir: &std::path::Path) -> (PathBuf, PathBuf) {
    std::fs::create_dir_all(dir).unwrap();
    let mut out = Vec::new();
    for (band, fs, n_samp, ap_lf) in [("ap", 30000.0, 9000usize, "384,0,1"), ("lf", 2500.0, 750, "0,384,1")] {
        let bin = dir.join(format!("rec_g0_t0.imec0.{band}.bin"));
        let n_ch = 385;
        let data: Vec<u8> = (0..n_samp * n_ch).flat_map(|i| (((i * 7) % 200) as i16 - 100).to_le_bytes()).collect();
        std::fs::write(&bin, &data).unwrap();
        std::fs::write(
            bin.with_extension("meta"),
            format!(
                "nSavedChans={n_ch}\nimSampRate={fs}\nfileSizeBytes={}\nsnsApLfSy={ap_lf}\nimDatPrb_type=0\n",
                data.len()
            ),
        )
        .unwrap();
        out.push(bin);
    }
    (out[0].clone(), out[1].clone())
}

#[test]
fn waveform_zoom_returns_to_the_previous_view() {
    let dir = std::env::temp_dir().join(format!("npx_app_wzoom_{}", std::process::id()));
    let (ap, _) = write_recording(&dir);
    let mut app = NPXplorerApp::new(&egui::Context::default(), ap).unwrap();
    app.waveform_channel = Some(5);
    let before = (app.view_start_s, app.view_dur_s, app.waveform_y_range_uv);
    app.waveform_zoom = Some(WaveformZoom { prev_start_s: before.0, prev_dur_s: before.1, prev_y_range_uv: before.2 });
    (app.view_start_s, app.view_dur_s, app.waveform_y_range_uv, app.waveform_y_center_uv) = (0.1, 0.02, 12.0, -30.0);
    // the zoom isn't what is saved
    let s = app.band_settings();
    assert_eq!((s.view_start_s, s.view_dur_s, s.waveform_y_range_uv), before);

    assert!(app.leave_waveform_zoom());
    assert_eq!((app.view_start_s, app.view_dur_s, app.waveform_y_range_uv), before);
    assert_eq!(app.waveform_y_center_uv, 0.0);
    assert!(!app.leave_waveform_zoom());
    drop(app);
    let _ = std::fs::remove_dir_all(&dir);
}

// --- headless render regression test --------------------------------------------
// The helpers below are the only place the test touches the app's render state.

/// What the UI handlers do after changing a setting that affects the heatmap:
/// nothing, the heatmap and projection are keyed on everything they are drawn from.
fn invalidate(_: &mut NPXplorerApp) {}

/// The heatmap pixels and firing-rate projection last drawn, and whether a texture exists.
fn drawn(app: &NPXplorerApp) -> (&[u8], &[f32], bool) {
    (&app.heatmap.pixel_buf, &app.projection.sums, app.heatmap.texture.is_some())
}

fn jobs_running(app: &NPXplorerApp) -> bool {
    app.classify.is_running() || app.spectrum.is_computing()
}

fn start_classify(app: &mut NPXplorerApp, ctx: &egui::Context) {
    app.classify.n_chunks = 5;
    app.classify.dispatch(ctx, &app.raw, &app.meta, &app.preproc_cfg.removed_channels);
}

fn labels(app: &NPXplorerApp) -> Option<&[u8]> {
    app.classify.labels.as_deref().map(|v| v.as_slice())
}

fn start_spectrum(app: &mut NPXplorerApp, ctx: &egui::Context) {
    app.spectrum.n_chunks = 5;
    app.spectrum.time_scope = crate::spectrum::SpectrumTimeScope::WholeRecordingChunks;
    app.spectrum.dispatch(ctx, &app.raw, &app.meta, &app.preproc_cfg);
}

fn spectrum_pixels(app: &NPXplorerApp) -> &[u8] {
    &app.spectrum.pixel_buf
}

/// Bit-exact regression check of what the heatmap view draws on a real recording:
/// runs the app headless through a series of view and setting changes and compares
/// the heatmap pixels (and the firing-rate projection, when shown) after each with
/// golden files (`realdata_tests::Golden`).
///   NPX_GOLDEN_BIN=<.ap.bin|.ap.cbin> NPX_GOLDEN_DIR=<dir> \
///   cargo test --release realdata_render_golden -- --ignored --nocapture
#[test]
#[ignore]
fn realdata_render_golden() {
    use std::hash::{Hash, Hasher};
    let Some(bin) = std::env::var_os("NPX_GOLDEN_BIN").map(PathBuf::from).filter(|p| p.is_file()) else {
        eprintln!("skipping: NPX_GOLDEN_BIN not set or not a file");
        return;
    };
    let Some(mut golden) = crate::golden::Golden::from_env() else { return };
    // the app writes its settings file next to the recording: open the recording
    // through links in a new scratch folder (nothing is deleted)
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    let rec_dir = golden.dir().join(format!("render_rec_{stamp}"));
    std::fs::create_dir_all(&rec_dir).unwrap();
    for ext in ["meta", "ch"] {
        let f = bin.with_extension(ext);
        if f.is_file() {
            std::os::unix::fs::symlink(&f, rec_dir.join(f.file_name().unwrap())).unwrap();
        }
    }
    let link = rec_dir.join(bin.file_name().unwrap());
    std::os::unix::fs::symlink(&bin, &link).unwrap();

    let ctx = egui::Context::default();
    let mut app = NPXplorerApp::new(&ctx, link).unwrap();
    app.background_fill = false;
    app.around_s = 2.0;
    let total_s = app.meta.n_samples as f64 / app.meta.sample_rate;
    app.view_start_s = (total_s / 3.0).min(600.0);
    app.view_dur_s = 1.0;
    app.preproc_cfg.spatial_filter = SpatialFilter::GlobalCmr;
    app.color_mode = ColorMode::Percentile;
    app.peak_pooling = false;
    app.show_firing_rate_overlay = false;
    app.colormap_choice = ColorMapChoice::ALL[0].clone();
    // everything else that the preferences may have set and that affects the drawing
    app.color_pct = 99.0;
    app.color_uv = 50.0;
    app.spike_threshold = -20.0;
    app.spike_smoothing_sigma = 1.5;
    app.preproc_cfg.channel_order = ChannelOrder::Depth;
    app.preproc_cfg.shank_order = ShankOrder::Id;
    app.preproc_cfg.removed_channels = app.meta.reference_channels.clone();
    app.preproc_cfg.highpass = true;
    app.preproc_cfg.phase_shift = true;
    app.preproc_cfg.avg_depths = false;
    app.preproc_cfg.notch_enabled = false;
    app.noise = crate::noise::NoiseSuppression::default();

    let mut size = egui::vec2(1600.0, 1000.0);
    // run frames until the drawn view stops changing with the worker idle
    let settle = |app: &mut NPXplorerApp, size: egui::Vec2| {
        let input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            ..Default::default()
        };
        let mut last = None;
        let mut stable = 0;
        for _ in 0..3000 {
            let _ = ctx.run(input(), |ctx| app.update(ctx));
            let (px, proj, has_tex) = drawn(app);
            let mut h = std::collections::hash_map::DefaultHasher::new();
            px.hash(&mut h);
            proj.iter().for_each(|v| v.to_bits().hash(&mut h));
            app.view_window.as_ref().map(|w| (w.first_sample, w.n_samp)).hash(&mut h);
            let sig = h.finish();
            let idle = !crate::worker::computing_foreground(&app.worker_state) && !jobs_running(app);
            stable = if has_tex && idle && last == Some(sig) { stable + 1 } else { 0 };
            if stable >= 5 {
                return;
            }
            last = Some(sig);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        panic!("the view never settled");
    };

    type Step = fn(&mut NPXplorerApp, &mut egui::Vec2, &egui::Context);
    let steps: [(&str, Step); 14] = [
        ("base", |_, _, _| {}),
        ("pooling", |a, _, _| a.peak_pooling = true),
        ("voltage", |a, _, _| {
            a.peak_pooling = false;
            a.color_mode = ColorMode::Voltage;
            a.color_uv = 80.0;
        }),
        ("colormap", |a, _, _| a.colormap_choice = ColorMapChoice::ALL[1].clone()),
        ("projection", |a, _, _| {
            a.show_firing_rate_overlay = true;
            a.spike_threshold = -40.0;
        }),
        ("threshold", |a, _, _| a.spike_threshold = -60.0),
        ("scroll", |a, _, _| a.view_start_s += 0.3),
        ("noise", |a, _, _| a.noise.sigmoid_enabled = true),
        ("zoom", |a, _, _| {
            a.zoom = Some(Zoom { ch_bottom: 100, ch_top: 200, prev_start_s: a.view_start_s, prev_dur_s: a.view_dur_s })
        }),
        ("destripe", |a, _, _| {
            a.zoom = None;
            a.preproc_cfg.spatial_filter = SpatialFilter::Destripe;
        }),
        ("resize", |_, size, _| *size = egui::vec2(1200.0, 800.0)),
        ("percentile", |a, _, _| {
            a.noise.sigmoid_enabled = false;
            a.color_mode = ColorMode::Percentile;
            a.colormap_choice = ColorMapChoice::ALL[0].clone();
        }),
        ("classify", |a, _, ctx| start_classify(a, ctx)),
        ("spectrum", |a, _, ctx| start_spectrum(a, ctx)),
    ];
    for (name, step) in steps {
        step(&mut app, &mut size, &ctx);
        invalidate(&mut app);
        settle(&mut app, size);
        let (px, proj, _) = drawn(&app);
        golden.check(&format!("render_{name}.rgba"), px);
        if app.show_firing_rate_overlay {
            golden.check(&format!("render_{name}_proj.f32"), bytemuck::cast_slice(proj));
        }
        if name == "classify" {
            golden.check("render_classify_labels.u8", labels(&app).expect("classification finished"));
        }
        if name == "spectrum" {
            golden.check("render_spectrum_strip.rgba", spectrum_pixels(&app));
        }
    }
    drop(app);
    golden.finish();
}

#[test]
fn settings_are_restored_on_reopen() {
    let dir = std::env::temp_dir().join(format!("npx_app_settings_{}", std::process::id()));
    let (ap, lf) = write_recording(&dir);
    let ctx = egui::Context::default();

    // first open: the reference site is removed and the settings file written
    let mut app = NPXplorerApp::new(&ctx, ap.clone()).unwrap();
    assert_eq!(app.preproc_cfg.removed_channels, BTreeSet::from([191]));
    assert!(crate::settings::path(&ap).is_file());

    app.preproc_cfg.removed_channels = BTreeSet::from([3, 191]);
    app.preproc_cfg.spatial_filter = SpatialFilter::Destripe;
    app.preproc_cfg.notches = vec![crate::notch::Notch { freq_hz: 50.0, bw_hz: 1.0 }];
    app.preproc_cfg.notch_enabled = true;
    app.view_dur_s = 0.1;
    app.view_start_s = 0.05;
    app.color_mode = ColorMode::Voltage;
    app.color_uv = 77.0;
    app.colormap_choice = ColorMapChoice::Vanimo;
    app.waveform_y_range_uv = 333.0;
    app.noise.sigmoid_enabled = true;
    app.spectrum.n_chunks = 42;
    app.classify.show_overlay = true;
    app.event_layout_text = "header\no,f\n".to_string();
    app.psth.color_pct = 97.5;
    drop(app);

    let app = NPXplorerApp::new(&ctx, ap.clone()).unwrap();
    assert_eq!(app.preproc_cfg.removed_channels, BTreeSet::from([3, 191]));
    assert_eq!(app.preproc_cfg.spatial_filter, SpatialFilter::Destripe);
    assert!(app.preproc_cfg.highpass && app.preproc_cfg.notch_enabled);
    assert_eq!(app.preproc_cfg.notches.len(), 1);
    assert_eq!((app.view_dur_s, app.view_start_s), (0.1, 0.05));
    assert!(app.color_mode == ColorMode::Voltage && app.colormap_choice == ColorMapChoice::Vanimo);
    assert_eq!((app.color_uv, app.waveform_y_range_uv), (77.0, 333.0));
    assert!(app.noise.sigmoid_enabled && app.noise_draft.sigmoid_enabled);
    assert_eq!(app.spectrum.n_chunks, 42);
    assert!(app.classify.show_overlay);
    assert_eq!(app.event_layout_text, "header\no,f\n");
    assert_eq!(app.psth.color_pct, 97.5);
    drop(app);

    // the LF file shares the removed channels and event settings, but has its
    // own band settings (from the preferences, not the AP band's)
    let app = NPXplorerApp::new(&ctx, lf.clone()).unwrap();
    assert_eq!(app.band, crate::settings::Band::Lf);
    assert_eq!(app.preproc_cfg.removed_channels, BTreeSet::from([3, 191]));
    assert_eq!(app.event_layout_text, "header\no,f\n");
    assert!(app.preproc_cfg.notches.is_empty());
    drop(app);
    let t = crate::settings::load_table(&ap);
    assert!(t.contains_key("ap") && t.contains_key("lf"));

    let _ = std::fs::remove_dir_all(&dir);
}
