# NPXplorer v0.1 beta

A lightweight viewer for raw Neuropixels electrophysiology data. Renders voltage traces as a heatmap mapped to the physical probe geometry in real time.

**This is beta software.** It has not been extensively tested across all recording configurations and probe types. Expect rough edges.

![App Screenshot](ex_screenshot.png)

## Requirements

- **SpikeGLX format only.** Supports `.ap.bin` (uncompressed) and `.ap.cbin` (mtscomp-compressed, requires `.ch` metadata alongside).
- The corresponding `.ap.meta` file must be in the same directory as the data file — this is where probe geometry, gain, and sample rate are read from.
- Tested with NP 1.0, NP 2.0 single-shank, and NP 2.0 multi-shank probes. Other probe types may work but are untested.

## Usage

Launch with a file:
```bash
npxplorer --file /path/to/data.ap.bin
```

Or launch empty and use the file picker:
```bash
npxplorer
```

### Navigation

- **Scroll wheel** moves the view forward/backward in time. The step size is 5% (Fine) or 30% (Coarse) of the current window width.
- **Arrow keys** or **A/D** jump half a window at a time.
- **Click on the navigation bar** at the bottom to jump to any point in the recording.
- **"Jump to (s)"** field lets you type an exact time in seconds.

### Selecting channels

- **Left-click** on the heatmap to select a channel (white marker line).
- **Right-click** to select a second channel (orange marker line).
- When both are selected, the vertical distance (Δ µm) between them is displayed.

### Preprocessing options

All filters run in real time on the displayed chunk. The pipeline order is fixed:

1. **DC removal** — subtracts the per-channel mean.
2. **Phase shift correction** — compensates the ADC multiplexing delay across channel groups using linear interpolation. The group size depends on the probe type.
3. **300 Hz highpass** — 3rd-order Butterworth, applied forward-backward (zero phase). Automatically enabled and locked on when Destripe is selected.
4. **Spatial filter** — choose one:
   - **Off** — no spatial filtering.
   - **Global CMR** — subtracts the median across all channels at each time point.
   - **Local CMR** — subtracts the median of channels within a 100–400 µm annulus around each channel. Uses Euclidean distance from probe geometry.
   - **Destripe** — IBL-style kfilt: AGC normalization → spatial highpass (0.01 Wn Butterworth, forward-backward) → rescale. Includes mirror padding at probe edges.
5. **Avg depths** — when enabled, channels at the same depth on the same shank are averaged into a single display row.

For multi-shank probes, all spatial filters operate independently per shank.

### Spike projection overlay

A semi-transparent bar overlay on the left side of the heatmap shows threshold crossings per channel, computed over the visible time window. This gives a quick approximation of firing rate by depth.

- **Threshold** — configurable in Preferences (default: −40 µV). A 1.5 ms refractory period is enforced between counted crossings.
- **Scaling** — the overlay width scales with both the time window duration and the threshold. At −20 µV the overlay has a baseline size; at −40 µV it's 2× as wide, at −80 µV it's 4×, etc.

### Color scale

- **Percentile mode** — the color range is set by a percentile of the absolute voltage distribution in the current buffer (default: 99th percentile, adjustable 95–100%).
- **±µV mode** — manually set the symmetric color range (10–300 µV).
- **Colormaps** — six options available in Preferences: Ice-Fire (default), Yellow-Magenta, Red-Blue, Orange-Blue, Vanimo, and Greyscale.

## Configuration

Preferences are saved to `npxplorer_prefs.toml` in the same directory as the executable. This includes preprocessing settings, colormap, color scale mode, spike threshold, window duration, and the last opened directory.

## Building from source

Requires Rust (stable). Build with:
```bash
cargo build --release
```

The binary will be at `target/release/npxplorer`. Debug builds are too slow for real-time rendering of full Neuropixels data.

## Known limitations

- SpikeGLX `.ap.bin`/`.ap.cbin` only. No OpenEphys support.
- Compressed `.cbin` files are a bit slower to navigate since each chunk must be decompressed on the fly.
