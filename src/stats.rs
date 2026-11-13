//! Order statistics on float slices. Comparisons use the IEEE total order, so a NaN
//! in the data gives a defined result instead of an inconsistent comparator.

use std::cmp::Ordering;

use rayon::prelude::*;

/// Floats with a total order (`f32::total_cmp`, `f64::total_cmp`).
pub trait TotalOrd: Copy {
    fn cmp_total(&self, other: &Self) -> Ordering;
}

impl TotalOrd for f32 {
    fn cmp_total(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl TotalOrd for f64 {
    fn cmp_total(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

/// The element of rank `len / 2`: the median for odd lengths, the upper of the two
/// middle values for even ones. Reorders `v`; `None` if it is empty.
pub fn upper_median<T: TotalOrd>(v: &mut [T]) -> Option<T> {
    if v.is_empty() {
        return None;
    }
    let mid = v.len() / 2;
    Some(*v.select_nth_unstable_by(mid, T::cmp_total).1)
}

/// The median as numpy computes it: for even lengths the mean of the two middle
/// values. Uses `v` as scratch space (its contents are unspecified afterwards);
/// `None` if it is empty.
///
/// Selects on integer keys that sort like `f32::total_cmp`: the same result, but a
/// key comparison is a single integer compare, which makes the CMR filters (one
/// median per row and sample) about 15 % faster than comparing with `total_cmp`.
pub fn median(v: &mut [f32]) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    let keys: &mut [i32] = bytemuck::cast_slice_mut(v);
    for k in keys.iter_mut() {
        *k = total_order_key(*k);
    }
    let half = keys.len() / 2;
    let upper = from_total_order_key(*keys.select_nth_unstable(half).1);
    if keys.len() % 2 == 1 {
        return Some(upper);
    }
    // after the selection, keys[..half] holds the values below the upper middle one
    let lower = from_total_order_key(*keys[..half].select_nth_unstable(half - 1).1);
    Some((upper + lower) / 2.0)
}

/// Maps the bits of an `f32` (as `i32`) to a key whose integer order is
/// `f32::total_cmp`'s order (the transform `total_cmp` itself uses).
fn total_order_key(bits: i32) -> i32 {
    bits ^ ((((bits >> 31) as u32) >> 1) as i32)
}

/// Inverse of `total_order_key`: the transform is its own inverse.
fn from_total_order_key(key: i32) -> f32 {
    f32::from_bits(total_order_key(key) as u32)
}

/// Time samples per parallel work item of `median_trace`. A column gather touches one
/// cache line per row, and the next 15 samples reuse those lines.
const TRACE_BLOCK_T: usize = 512;

/// The median across rows at every sample of `data` (`[n_rows][n_samp]`): the
/// common-mode signal that CMR subtracts. All zeros if there are no rows.
pub fn median_trace(data: &[f32], n_rows: usize, n_samp: usize) -> Vec<f32> {
    assert_eq!(data.len(), n_rows * n_samp, "data does not match the rows");
    let mut trace = vec![0.0f32; n_samp];
    if n_rows == 0 {
        return trace;
    }
    trace.par_chunks_mut(TRACE_BLOCK_T).enumerate().for_each_init(Vec::new, |col, (b, out)| {
        // consecutive samples read the same cache lines of every row
        let t0 = b * TRACE_BLOCK_T;
        for (i, m) in out.iter_mut().enumerate() {
            let t = t0 + i;
            col.clear();
            col.extend((0..n_rows).map(|r| data[r * n_samp + t]));
            *m = median(col).expect("n_rows > 0");
        }
    });
    trace
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_odd_and_even_lengths() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&mut [7.0]), Some(7.0));
        assert_eq!(median(&mut []), None);
    }

    #[test]
    fn upper_median_takes_the_upper_middle_value() {
        assert_eq!(upper_median(&mut [4.0f32, 1.0, 3.0, 2.0]), Some(3.0));
        assert_eq!(upper_median(&mut [5.0f64, 1.0, 3.0]), Some(3.0));
        assert_eq!(upper_median::<f64>(&mut []), None);
    }

    #[test]
    fn median_orders_like_total_cmp() {
        let vals = [-3.5f32, -0.0, 0.0, 1e-30, -1e30, 7.25, f32::INFINITY, f32::NEG_INFINITY, 2.0, -2.0];
        for n in 1..=vals.len() {
            let mut sorted = vals[..n].to_vec();
            sorted.sort_by(f32::total_cmp);
            let want = if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 };
            let got = median(&mut vals[..n].to_vec()).unwrap();
            assert_eq!(got.to_bits(), want.to_bits(), "n = {n}");
        }
    }

    #[test]
    fn median_trace_takes_the_median_of_each_column() {
        // 3 rows x 1000 samples, so the trace spans two blocks
        let n = 1000;
        let data: Vec<f32> = (0..3 * n).map(|i| ((i * 7919) % 101) as f32).collect();
        let trace = median_trace(&data, 3, n);
        for t in 0..n {
            let mut col = [data[t], data[n + t], data[2 * n + t]];
            col.sort_by(f32::total_cmp);
            assert_eq!(trace[t], col[1], "t = {t}");
        }
    }

    #[test]
    fn nan_does_not_break_the_order() {
        // NaN sorts above every number in the total order
        assert_eq!(median(&mut [f32::NAN, 1.0, 2.0, 3.0, 4.0]), Some(3.0));
    }
}
