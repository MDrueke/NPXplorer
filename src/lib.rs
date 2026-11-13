//! NPXplorer's core: reading Neuropixels recordings (SpikeGLX and Open Ephys, raw or
//! compressed with mtscomp, Zarr, NWB), preprocessing, the analyses (power spectrum,
//! PSTH, channel classification, notch scan, noise suppression), export, and the
//! pixel rendering of the heatmaps. It holds no GUI code: the app (main.rs and the
//! modules it declares) is built on it.

pub mod atlas;
pub mod blosc;
pub mod channel_classify;
pub mod channel_remove;
pub mod colormap;
pub mod config;
pub mod control;
pub mod data;
pub mod export;
#[macro_use]
pub mod log;
pub mod mtscomp;
pub mod noise;
pub mod notch;
pub mod nwb;
pub mod preprocess;
pub mod probe;
pub mod psth;
pub mod render;
pub mod rows;
pub mod sidecar;
pub mod spectrum;
pub mod spikes;
pub mod stats;
pub mod worker;
pub mod zarr;

#[cfg(test)]
mod golden;
#[cfg(test)]
mod realdata_tests;
