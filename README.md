# NPXplorer v0.7.0

A viewer for raw Neuropixels recordings. Shows the voltage of all channels as a heatmap laid out by the probe geometry.

- Fast scrolling through the whole recording, with preprocessing (DC, phase shift, highpass, CMR, destripe) applied on the fly
- Single-channel waveform view
- Firing rate overlay
- Power spectrum overlay (per-channel PSD)
- Channel removal
- Noise suppression display filters (wavelet, isolated events, soft-knee noise gate)
- IBL-style channel classification (dead, noisy, outside of brain)
- Atlas registration with the Allen mouse brain atlas
- Peri-stimulus averages (PSTH)
- Event overlay (stimuli, TTL pulses, behavioural events)
- Screenshots of the plot area
- Export of the preprocessed recording (SpikeGLX or Open Ephys format) for spike sorting

**This is beta software.** It has not been tested with all recording configurations and probe types.

![App Screenshot](ex_screenshot.png)

## Supported data

- SpikeGLX (`.ap.bin`, `.lf.bin`) with the matching `.meta` file next to it.
- Open Ephys binary format (`continuous.dat`); `structure.oebin` and `settings.xml` are found automatically.
- mtscomp-compressed files (`.cbin` with its `.ch` file).
- Tested with NP 1.0, NP 2.0 single-shank and NP 2.0 multi-shank probes (SpikeGLX), and a single-shank NP 1.0 recording from Open Ephys.

## Usage

Channels are named as in the recording's metadata (e.g. `AP12` for SpikeGLX, `CH13` for Open Ephys), everywhere in the app and in exported files. The status bar lists anything the metadata did not provide (e.g. an unknown probe type, whose gain and electrode positions are then assumed), so you know when µV values or depths are nominal.

### Navigation

- **Scroll wheel**: move in time (step size set by **Fine**/**Coarse**).
- **Arrow keys** or **A/D**: jump half a window.
- **Navigation bar** (bottom): click to jump. The solid marker shows the displayed window, the shaded area the part that is already preprocessed.
- **Left-drag** on the heatmap: draw a rectangle to zoom into its channels and time range (the readout at the bottom left shows both while dragging). You can zoom again inside the zoomed view.
- **Esc**: closes the topmost open window, one per press, aborting what it is calculating (in a progress window, Esc is Abort; while exporting, it aborts the export); with no window open, it leaves the waveform view's zoom, then the waveform view, then the heatmap's zoom (returning to the full view and the previous time window).

### Selecting channels

- **Alt + left-click** a row to select a channel, **Alt + right-click** to select a second one. With two selected, their vertical distance (Δ µm) is shown in the top right.
- **Right-click** a row for a menu: **View waveform** shows that channel's trace (Alt + scroll changes its vertical scale; the readout at the bottom left shows the time and voltage of the hovered point; left-drag a rectangle to zoom into its time and voltage range, Esc returns to the view before the zoom; Esc or ✖ returns to the heatmap), **Remove channel** excludes it.

### Removing channels

**Remove channels…** excludes channels from the display and from all processing. Enter channel IDs or ranges separated by commas (e.g. `AP3,AP17,AP40-AP50`); a bare number (`17`, `40-50`) matches the channels whose ID ends in that number. **Load from file…** reads the list from a file, using a layout file like the PSTH one (see below): `channel_remove_layout.csv` next to the list file, or the default in `config/`.

When a recording is opened for the first time, the list starts with the probe's reference sites, which carry no neural signal, and they are removed before any preprocessing: those SpikeGLX marks as unused in its geometry map, and those Open Ephys lists without a position. If the metadata marks none, they are taken from the probe type (channel 191 on NP 1.0; NP 2.0 has none). **Reset** includes them again. The list is saved with the recording's settings.

### Preprocessing

- **DC**: remove each channel's mean.
- **Phase Shift**: correct the sampling delay between channels. Channels sharing an ADC are digitised one after another within each sample period; the delay of every channel is taken from the recording's `~muxTbl` (SpikeGLX) or from the probe type, and removed with a windowed-sinc fractional-delay filter on each raw channel before any averaging.
- **Notch (n)**: the notch filters set up in the Noise Suppression window (shown once there are any). Applied after DC removal, before the highpass.
- **300 Hz HP**: zero-phase highpass filter (always on with Destripe).
- **Spatial**: **Global CMR** (subtract the median of all channels of the shank), **Local CMR** (median of nearby channels), or **Destripe** (IBL-style). Destripe's spatial filter runs along physical depth over each stretch of neighbouring electrodes — it never mixes channels across a gap in the layout (drawn as a dotted line) or across shanks, whatever the display order.
- **Avg adjacent chans**: average channels at the same depth into one row.

### Noise suppression

**Noise Suppression** opens a window with filters for the display: they change the heatmap, the waveform view and the hovered µV value, not the firing rate, PSTH or spectrum. Changes take effect when **Apply** is pressed; **Disable all** switches every filter off at once (the values are kept). The button is highlighted while a filter is on; the applied settings are saved with the recording's settings. In **%ile** colour mode (without peak pooling) the colour range still comes from the unfiltered signal, so filtered and unfiltered views can be compared directly. The filters run in this order:

- **Wavelet denoising**: each channel is split into scales with a wavelet (Haar, db4 or sym4; **Levels**, default 2). At each scale, coefficients below **Threshold** × that scale's noise level (default 3 × σ, σ = median |coefficient| / 0.6745) are removed (**Hard**) or all are shrunk toward 0 by it (**Soft**, default).
- **Attenuate isolated events**: an event is a stretch of the signal between two zero crossings whose peak reaches **Event level** (default ±15 µV). It is kept if it spans at least **Min. adjacent channels** (default 3, counting its own) that are neighbours on the probe, each reaching **Neighbour amplitude** (default 90 %) of its peak with the same polarity within **Time tolerance** (default ±0.3 ms). Otherwise the whole event is reduced by **Attenuation** (default 50 %; 100 % removes it). Like Destripe, channels across a gap in the layout are not neighbours.
- **Soft-knee noise gate**: scales every sample by a smooth S-shaped (logistic) function of its amplitude: 0 at 0 µV, half of **Max gain** at **Centre** (default 20 µV), reaching **Max gain** (default 1×) for strong signals. **Width** (default 5 µV) sets how sharp the transition is (12 % to 88 % within Centre ± 2 widths). Unlike a hard threshold it has no jump. A max gain above 1 is the same as narrowing the colour range; in **%ile** mode strong signals then saturate.

**Notch filters** (lower part of the window) are part of the preprocessing: they change the data everywhere, not just the display. **Find noise peaks** samples **Chunks to sample** (default 100) chunks of about 1 s across the recording, applies the current preprocessing (without notches), and averages each channel's power spectrum over the chunks. Narrow peaks are found on every channel's own spectrum, so noise that reaches the channels at different times, or only some of them, is found too; CMR and Destripe cannot remove such noise. A progress bar shows the scan, **Abort** stops it. A plot shows each shank's mean spectrum. The table lists a frequency when it is a peak on at least **Min. channels** of one shank's channels (default 30 %) and the peak is narrow: at least **Min. prominence** (default 6 dB) above the smoothed spectrum, at most **Max. width** wide (default 3 Hz), and present in at least **Min. presence** of the chunks (default 25 %; estimated from how much the peak's power varies between chunks). Broad peaks are usually brain oscillations, and peaks that come and go (e.g. stimulation artifacts) are no case for a filter on the whole recording. Evenly spaced lines (at least 3, e.g. a carrier with mains sidebands every 50 Hz) are shown as one row; its checkbox adds a notch for each line, and hovering lists them. Tick a peak to add a notch, or add one by hand; **Apply notches** activates them on all channels. The notches are saved with the recording's settings (separately for AP and LF) and restored when the file is reopened. **Notch (n)** in the preprocessing bar switches them off and on.

Voltages are scaled per channel from the metadata: SpikeGLX gains from `~imroTbl` (or `imChan0apGain`, or the fixed gain of the probe type), Open Ephys `bit_volts`.

### Color scale

- **%ile**: the color range follows a percentile of the displayed voltages; **±µV**: a fixed range. **Alt + scroll** adjusts either.
- **Peak pooling** (Preferences, off by default): when a pixel column covers many samples (long windows), it shows the sample with the largest magnitude instead of the mean, so spikes keep their amplitude at any window length. In **%ile** mode the colour range then follows the values on screen. Off gives a smoother, mean-based picture (better for LFP).

### Firing rate overlay

A bar on the left of the heatmap shows the number of threshold crossings per channel in the displayed window. Threshold, scale and depth smoothing are set in Preferences.

### Channel classification

Marks dead (magenta), noisy (red) and out-of-brain (green) channels with a colored stripe. Each shank is classified on its own with the channels in depth order; removed channels are skipped. **Outside of brain** (Preferences) chooses how the brain surface is found: **Adaptive** (default; IBL's adaptive mode, finds weaker surfaces, e.g. in LFP data) or **Fixed threshold** (IBL's default, stricter). **Chunks to sample** in Preferences sets how many snippets of the recording are used (more is more reliable but slower).

### Atlas registration

Draws the borders between brain regions of the Allen mouse atlas (CCF 2017, 10 µm) on the heatmap, from the probe's insertion coordinates. It needs a folder with `annotation_10.nrrd` from the [Allen Institute](https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/) and the structure ontology saved as `query.csv` from the [Allen API](https://api.brain-map.org/api/v2/data/query.csv?criteria=model::Structure,rma::criteria,[ontology_id$eq1],rma::options[order$eq%27structures.graph_order%27][num_rows$eqall]). On the first load, the `.nrrd` is converted once into `annotation_10_npxplorer.npy` (2.4 GB) and `annotation_10_npxplorer_ids.txt` in the same folder, which takes a while and needs write access to the folder.

- **Insertion**: AP and ML in mm from bregma (AP positive = anterior, ML positive = right). Angles as **Azimuth / elevation / rotation** (azimuth from the lambda→bregma axis, elevation from horizontal) or **Polar / azimuth / roll** (polar angle from vertical, azimuth from +ML). Rotation/roll turns the probe around its axis, which matters for multi-shank probes.
- **Depth**: from the brain surface to the tip, along the probe; the tip offset is the distance from the tip to the first electrode row (default 195 µm).
- **Bregma–lambda distance** (default 4.1 mm) scales the atlas to the animal.

Hover a region label for its full name; the hovered channel's region also appears in the readout at the bottom left. With channels ordered by ID, borders and labels are hidden. Regions with fewer channels than set in **Skip drawing regions with less than … channels** (default 4; 0 draws all) are not drawn on their own: their channels are split between the neighboring regions. The readout and **Save** still use the exact regions. To match the borders to the data:

- **Alt + drag** any border to move all borders together (changes the depth).
- **Drag** a single border to move just that one (it stops at its neighbors). Dragged borders are kept when the depth changes, and only **Reset borders** undoes them.

**Save** writes `<recording>_regions.csv` next to the data file, with each channel's ID and region. The coordinates and dragged borders are saved with the recording's settings (shared by the AP and LF files). If the overlay was shown when the recording was closed, the atlas is loaded and the borders are drawn again when it is opened.

The tool also works with the reformatted version of the atlas that the [Neuropixels Trajectory Explorer](https://github.com/petersaj/neuropixels_trajectory_explorer) uses (`annotation_volume_10um_by_index.npy` and `structure_tree_safe_2017.csv`, [download](https://osf.io/fv7ed/)). It has the same voxels and borders, loads without the conversion, and gives the same regions as that tool for the same inputs.

### PSTH (peri-stimulus average)

Averages the signal around event onsets for all channels that are not removed, using the main window's preprocessing, and shows traces of up to two selected channels (left-/right-click the heatmap), the mean over all channels, and a heatmap. Nothing is computed until **Apply/Compute** is pressed. **Alt + scroll** over the window adjusts the color scale; the traces are scaled to fit when computed and then zoom along with it. Without a spatial filter, the raw windows are averaged first and the average is preprocessed once (identical result, since every step is then linear), which is much faster.

The event file (onset times in seconds) is read according to a layout. Its lines mirror the event file: lines without an `o` are header rows to skip, and the first line with an `o` marks the column(s) holding the onset times (`f` = offset times, used by the Events window; `x` = ignore). For example

```
header
o,f,x
```

skips one header row, reads onsets from the first column and offsets from the second. Lines starting with `#` are comments. The layout is shown in the **File format** field of the PSTH and Events windows. Edits are saved with the recording's settings and used by both windows; the default in `config/` is never changed. **Reset to default** restores the default.

### Events

Shades the events on the heatmap and the waveform view, in the color of the firing rate overlay, and lists them as **Events** in the legend. The event file is read with the same layout as the PSTH. Offsets come from the `f` column(s); without one, each area is **Duration (ms)** long. **Emphasize on/offset** draws a thin line at each onset and offset.

The event file last loaded in the Events or PSTH window is saved with the recording's settings, together with the Events and PSTH settings, and filled in when the recording is opened again (it is not loaded until **Load** or **Apply/Compute** is pressed).

### Power Spectrum

**Power Spectrum** opens a settings window for computing each channel's power spectral density (Welch's method) and showing it as a second heatmap to the right of the main one: one row per channel, aligned with the main heatmap's rows (including the current zoom), frequency on a log x axis with four tick labels, colour = power.

- **Time scope**: **Current view window** follows the view as you scroll or zoom, once calculated; **Whole recording (chunks)** (default) averages evenly-spaced snippets across the whole recording — **Chunks to sample** (default 100) sets how many.
- **Source**: raw voltage, or the currently preprocessed signal (current-view scope only; whole-recording mode always uses raw).
- **Scaling**: linear power or **dB** (default); **Colour range**: **Global** (default, one shared range across channels) or **Per-channel**.
- **Restrict frequency range** limits the displayed/coloured band to a chosen Hz range; off shows the full band up to Nyquist. In whole-recording mode, **Restrict time window** takes the chunks only from a chosen part of the recording.

**Calculate** starts it. Whole-recording mode shows a progress bar and can be aborted; current-view mode recomputes once scrolling has stopped. Once computed, **Show Spectra**/**Hide Spectra** in the legend toggles the overlay (as do the other overlays' Show/Hide entries there, once they have something to show).

### Screenshot

**Screenshot** (top right) opens a window listing the overlays calculated so far (events, atlas regions, firing rate, power spectrum, channel classification), plus the selected-channel lines and the scale bar; tick the ones to include. **Save…** asks for a file name and saves the plot area (the heatmap with the spectrum panel, or the waveform view) as a PNG at screen resolution, without windows, the legend or the hover readout. The window closes once the image is saved.

### Export preprocessed data

**File → Export preprocessed data…** writes a copy of the recording with the current preprocessing applied, in the format of the original: a SpikeGLX `.bin` with its `.meta`, or an Open Ephys session folder (`settings.xml`, `structure.oebin`, `continuous.dat`, the timestamps cut to the exported range, and the events). All channels that are not removed are exported, in their original order, under their original names and at their original µV per bit; the SpikeGLX sync channel is copied unchanged.

The window offers:

- **Time range** (**From**/**To**, s): **Whole recording** or **Current view**. The exported file starts at **From**; SpikeGLX's `firstSample` is shifted accordingly, Open Ephys timestamps keep their values, so the export stays aligned with other streams.
- **Bad channels**, from the channel classification (it runs first if it hasn't): dead and noisy channels can each be kept, removed or interpolated from their neighbours; channels outside the brain can be removed. Check the classification overlay first: the outside-of-brain detection can be wrong.
- **Average same-depth channels**, off by default. It changes the probe geometry and breaks spike sorters and most other analyses.
- **Event blanking**: sets the data around each event of the **Events** window to 0 (**Blank**) or bridges it with a straight line (**Interpolate**), in a window from **Start** to **End** (ms, relative to the onset). **Around offsets** adds a second window around each offset; without an offset column in the event file, the offset is the onset plus **Event length**. Blanking runs before notch, highpass and spatial filter, so they don't spread the artifact.
- **Write probe file for Kilosort 4**, and **Open export when done**.

**Export…** opens a save dialog with a proposed name: the original's with `_preprocessed` and the time range added (e.g. `run_preprocessed_120-300s_g0_t0.imec0.ap.bin`); the original can't be overwritten. While the export runs, the window is frozen and greyed out and all CPU cores are used; **Abort** stops it and removes what was written. Noise suppression is a display filter and is not exported.

Next to the export, NPXplorer writes:

- `<name>.preprocessing.toml` (Open Ephys: `preprocessing.toml` in the session folder): source, time range, channels exported, removed and interpolated, and every step applied with its parameters, in order.
- `<name>.npxplorer.toml`: settings for opening the export in NPXplorer, with preprocessing switched off (it is in the data now) and the atlas registration carried over.
- `<name>.events.csv`: the events in the exported range, with times relative to its start (if an event file is loaded).
- `<name>.kilosort4_probe.json`: channel positions and shanks in Kilosort 4's probe format (if ticked).

## Configuration

Every setting made for a recording, in the main window and in the tool windows (preprocessing, colour scale, time window, removed channels, noise suppression and notches, overlays, power spectrum, classification, events, PSTH, atlas registration), is saved in `<recording>.npxplorer.toml` next to the data file and applied when the recording is opened again. The AP and LF files of a recording share it: settings that differ between the bands are kept separately, the removed channels, event and atlas settings are shared. Overlays that take a computation (power spectrum, channel classification, PSTH) are not recomputed on opening; their settings are restored. The file is written about a second after a change and when the recording is closed. Files written by earlier versions (`.npx_atlas.toml`, `.npx_notch.toml`) are read and replaced by it.

Preferences are saved in `config/npxplorer_prefs.toml` next to the executable, together with the default layout files for PSTH and Remove channels. They hold the settings that are not specific to a recording (atlas folder, buffer and memory limits, recent files) and the last-used settings, which a recording opened for the first time starts with.

The **Buffer** section in Preferences controls how much of the recording is preprocessed ahead of time and the memory limits at which the buffer stops growing.

## Known limitations

- `.cbin` files are slower to navigate, since they are decompressed on the fly.
- Open Ephys: only the first probe of a `settings.xml` processor is used; multi-shank probes and multi-experiment/multi-recording sessions are untested.
- Open Ephys export: removing channels rewrites the probe's channel lists in `settings.xml`; this has not been checked with SpikeInterface/probeinterface yet.
- SpikeGLX recordings older than 20230202 have no `~snsGeomMap`; electrode positions are then rebuilt from `~snsShankMap` and the probe type's pitch. Unknown probe types fall back to a nominal single column at 20 µm and a warning in the status bar.

## Command line

Build with `cargo build --release` (Rust stable). `--file <path>` opens a recording directly, `--debug` writes `debug.log` next to the executable.
