# RawViewer

RawViewer is a lightweight, high-performance application for the visualization and exploration of raw Neuropixels electrophysiology data.

## Features and Limitations

- **File Support**: Currently restricted to data acquired via SpikeGLX (`.ap.bin` format). Support for OpenEphys formats is planned. Natively supports and automatically decompresses `mtscomp` compressed data (`.cbin` + `.ch` metadata).
- **Metadata Requirement**: The corresponding `.ap.meta` file must be present in the same directory as the binary data file for spatial and hardware layout decoding.
- **Hardware Support**: Currently supports single-shank Neuropixels probes. Multi-shank support is planned for future releases.

## Functionality

RawViewer provides real-time rendering of high-density voltage traces mapped to the physical geometry of the probe. Key functionalities include:
- **Spatial Preprocessing**: Apply real-time Destripe, Global CMR, and Local CMR spatial filters.
- **Temporal Filtering**: Optional 300 Hz high-pass filtering and DC offset removal.
- **Spike Projection Overlay**: An integrated overlay visualizes threshold crossings (default $\le -40$ $\mu$V) computed dynamically over the visible temporal window, incorporating a 1.5 ms refractory period.
- **Geometry Inspection**: Calculate exact vertical distances ($\Delta$ $\mu$m) between selected channels by left-clicking and right-clicking on the heatmap.

## Usage

RawViewer can be executed directly from the command line.

To launch the application with a file pre-loaded:
```bash
rawviewer --file /path/to/data.ap.bin
```

To launch the empty application and use the native file picker:
```bash
rawviewer
```

Navigate the temporal domain using the scroll wheel (configurable between fine and coarse steps) and adjust the temporal window size using the toolbar. Colormaps, spike projection thresholds, and other visual settings can be accessed via the Preferences menu.
