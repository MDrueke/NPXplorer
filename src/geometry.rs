/// Probe geometry helpers
use crate::data::ChannelGeom;

/// Euclidean distance in µm between two channels
#[allow(dead_code)]
pub fn dist_um(a: &ChannelGeom, b: &ChannelGeom) -> f32 {
    let dx = a.x_um - b.x_um;
    let dy = a.y_um - b.y_um;
    (dx * dx + dy * dy).sqrt()
}
