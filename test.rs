#[derive(Clone, PartialEq, Debug)]
pub enum SpatialFilter {
    Off,
    GlobalCmr,
    LocalCmr,
    Destripe,
}
#[derive(Clone, PartialEq, Debug)]
pub struct PreprocConfig {
    pub dc_removal: bool,
    pub phase_shift: bool,
    pub highpass: bool,
    pub spatial_filter: SpatialFilter,
    pub avg_depths: bool,
    pub sample_rate: f64,
    pub im_dat_prb_type: u32,
}
fn main() {
    let a = PreprocConfig {
        dc_removal: true,
        phase_shift: false,
        highpass: true,
        spatial_filter: SpatialFilter::Destripe,
        avg_depths: true,
        sample_rate: 30000.0,
        im_dat_prb_type: 2013,
    };
    let b = a.clone();
    println!("a == b: {}", a == b);
}
