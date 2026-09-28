//! Per-probe-type constants, used wherever the recording's metadata does not state a
//! value itself: ADC resolution and input range, fixed gains of the NP 2.0 family,
//! electrode pitch (to rebuild a geometry from an old `~snsShankMap`), and the ADC
//! multiplexing layout (to derive the per-channel sampling delay).
//!
//! Values are taken from probeinterface's `neuropixels_probe_features.json` (table
//! version 1.8) and SpikeGLX's metadata documentation.

/// How channels are multiplexed onto ADCs, which fixes the per-channel sampling delay.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MuxFamily {
    /// 32 ADCs × 12 channels, 13 cycles per sample period (one cycle serves the LF
    /// band): ADC = floor(ch/24)·2 + ch%2, cycle = (ch % 24) / 2.
    Np1,
    /// 16 channels per ADC, 16 cycles: ADC = floor(ch/32)·2 + ch%2, cycle = (ch % 32) / 2.
    Np2,
    /// Unknown or irregular table (e.g. NP1200); no delay is applied.
    None,
}

impl MuxFamily {
    /// Sampling delay of channel `ch`, as a fraction of the AP sample period.
    pub fn sample_shift_fraction(self, ch: usize) -> f32 {
        match self {
            MuxFamily::Np1 => ((ch % 24) / 2) as f32 / 13.0,
            MuxFamily::Np2 => ((ch % 32) / 2) as f32 / 16.0,
            MuxFamily::None => 0.0,
        }
    }

    /// Number of ADC cycles per AP sample period for a mux table with `n_groups`
    /// channel groups (NP1 adds one cycle for the LF band).
    pub fn n_cycles(self, n_groups: usize) -> f64 {
        match self {
            MuxFamily::Np1 => n_groups as f64 + 1.0,
            _ => n_groups as f64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProbeSpec {
    pub part_number: &'static str,
    /// ADC resolution (bits); `imMaxInt` = 2^(bits-1)
    pub adc_bits: u32,
    /// `imAiRangeMax` (V) = half the ADC's peak-to-peak input range
    pub ai_range_max_v: f64,
    /// fixed AP / LF gain; `None` = selectable per channel in the imro table
    pub fixed_ap_gain: Option<f64>,
    pub fixed_lf_gain: Option<f64>,
    pub mux: MuxFamily,
    /// horizontal / vertical electrode pitch (µm)
    pub pitch_h_um: f32,
    pub pitch_v_um: f32,
    /// x of column 0 on even / odd rows (µm from the left shank edge) — differ on
    /// staggered layouts (NP 1.0 checkerboard)
    pub even_row_x0_um: f32,
    pub odd_row_x0_um: f32,
}

const fn np1(part_number: &'static str, pitch_h: f32, pitch_v: f32, even_x0: f32, odd_x0: f32) -> ProbeSpec {
    ProbeSpec {
        part_number,
        adc_bits: 10,
        ai_range_max_v: 0.6,
        fixed_ap_gain: None,
        fixed_lf_gain: None,
        mux: MuxFamily::Np1,
        pitch_h_um: pitch_h,
        pitch_v_um: pitch_v,
        even_row_x0_um: even_x0,
        odd_row_x0_um: odd_x0,
    }
}

const fn np2(part_number: &'static str, adc_bits: u32, ai_range_max_v: f64, gain: f64) -> ProbeSpec {
    ProbeSpec {
        part_number,
        adc_bits,
        ai_range_max_v,
        fixed_ap_gain: Some(gain),
        fixed_lf_gain: Some(1.0),
        mux: MuxFamily::Np2,
        pitch_h_um: 32.0,
        pitch_v_um: 15.0,
        even_row_x0_um: 27.0,
        odd_row_x0_um: 27.0,
    }
}

/// Constants for a SpikeGLX `imDatPrb_type`. (Multi-shank probes: 24, 2013, 2014,
/// 2020, 2021, 3020, 3022 — shank pitch 250 µm; SpikeGLX gives x within each shank.)
pub fn spec_for_type(prb_type: u32) -> Option<ProbeSpec> {
    Some(match prb_type {
        0 => np1("NP1000", 32.0, 20.0, 27.0, 11.0),
        1020 => np1("NP1020", 87.0, 20.0, 27.0, 11.0),
        1030 => np1("NP1030", 87.0, 20.0, 27.0, 11.0),
        1100 => np1("NP1100", 6.0, 6.0, 14.0, 14.0),
        1110 => np1("NP1110", 6.0, 6.0, 15.5, 15.5),
        1120 => np1("NP1120", 4.5, 4.5, 6.75, 6.75),
        1121 => np1("NP1121", 0.0, 3.0, 6.25, 6.25),
        1122 => np1("NP1122", 3.0, 3.0, 12.5, 12.5),
        1123 => np1("NP1123", 4.5, 4.5, 10.25, 10.25),
        1200 => ProbeSpec { mux: MuxFamily::None, ..np1("NP1200", 32.0, 31.0, 56.5, 36.5) },
        1300 => np1("NP1300", 48.0, 20.0, 11.0, 11.0),
        21 => np2("NP2000", 14, 0.5, 80.0),
        24 => np2("NP2010", 14, 0.5, 80.0),
        2003 => np2("NP2003", 12, 0.62, 100.0),
        2004 => np2("NP2004", 12, 0.62, 100.0),
        2013 => np2("NP2013", 12, 0.62, 100.0),
        2014 => np2("NP2014", 12, 0.62, 100.0),
        2020 => np2("NP2020", 12, 0.62, 100.0),
        2021 => np2("NP2021", 12, 0.62, 100.0),
        3010 => np2("NP3010", 12, 0.67, 100.0),
        3020 => np2("NP3020", 12, 0.67, 100.0),
        3022 => np2("NP3022", 12, 0.67, 100.0),
        _ => return None,
    })
}

/// Constants for a probe part number as written by Open Ephys / Imec, e.g.
/// "PRB_1_4_0480_1" (NP 1.0), "PRB2_1_2_0640_0" (NP 2.0 single shank),
/// "PRB2_4_2_0640_0" (NP 2.0 four shank), "NP1100", "NP2013".
pub fn spec_for_part_number(pn: &str) -> Option<ProbeSpec> {
    let pn = pn.trim();
    if let Some(num) = pn.strip_prefix("NP").and_then(|s| s[..s.len().min(4)].parse::<u32>().ok()) {
        let t = match num {
            1000 => 0,
            2000 => 21,
            2010 => 24,
            n => n,
        };
        if let Some(s) = spec_for_type(t) {
            return Some(s);
        }
    }
    if pn.starts_with("PRB2_4") {
        return spec_for_type(24);
    }
    if pn.starts_with("PRB2") {
        return spec_for_type(21);
    }
    if pn.starts_with("PRB_1") || pn.starts_with("PRB1") {
        return spec_for_type(0);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mux_formulas_match_reference_tables() {
        // IBL adc_shifts / SpikeGLX ~muxTbl: cycle 0 = channels 0,1,24,25,…; cycle 1 = 2,3,26,27,…
        let f = MuxFamily::Np1;
        assert_eq!(f.sample_shift_fraction(0), 0.0);
        assert_eq!(f.sample_shift_fraction(1), 0.0);
        assert_eq!(f.sample_shift_fraction(24), 0.0);
        assert!((f.sample_shift_fraction(2) - 1.0 / 13.0).abs() < 1e-7);
        assert!((f.sample_shift_fraction(23) - 11.0 / 13.0).abs() < 1e-7);
        let g = MuxFamily::Np2;
        assert_eq!(g.sample_shift_fraction(33), 0.0);
        assert!((g.sample_shift_fraction(31) - 15.0 / 16.0).abs() < 1e-7);
    }

    #[test]
    fn part_numbers() {
        assert_eq!(spec_for_part_number("PRB_1_4_0480_1").unwrap().part_number, "NP1000");
        assert_eq!(spec_for_part_number("PRB2_1_2_0640_0").unwrap().part_number, "NP2000");
        assert_eq!(spec_for_part_number("PRB2_4_2_0640_0").unwrap().part_number, "NP2010");
        assert_eq!(spec_for_part_number("NP2013").unwrap().fixed_ap_gain, Some(100.0));
        assert!(spec_for_part_number("foo").is_none());
    }
}
