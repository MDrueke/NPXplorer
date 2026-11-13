//! Row-major `[row][sample]` buffers, split for parallel work along time.
//!
//! The spatial filters (CMR, destripe) and the channel-to-row conversion combine all
//! rows at each time sample, so they run in parallel over blocks of time. These helpers
//! give each parallel item its own time block of every row as ordinary `&mut` slices.
//! The borrow checker then proves that the items are disjoint, so no `unsafe` is needed.

/// A block of a row-major buffer: `data[row * n + i]` is sample `first + i` (an
/// absolute sample index in the recording) of row `row`.
#[derive(Clone, Copy)]
pub struct SampleWindow<'a> {
    pub data: &'a [f32],
    pub first: usize,
    pub n: usize,
}

/// The rows of `data` (`[n_rows][n_samp]`) listed in `which`, in that order.
///
/// Panics if `n_samp` is 0, if `data` is not a whole number of rows, or if `which`
/// lists a row twice or one that is out of range.
pub fn select_rows_mut<'a>(data: &'a mut [f32], n_samp: usize, which: &[usize]) -> Vec<&'a mut [f32]> {
    assert!(n_samp > 0 && data.len().is_multiple_of(n_samp), "data is not a whole number of rows");
    let mut all: Vec<Option<&mut [f32]>> = data.chunks_exact_mut(n_samp).map(Some).collect();
    which
        .iter()
        .map(|&r| all[r].take().expect("row selected twice"))
        .collect()
}

/// `rows` cut into time blocks of `block` samples: entry `b` holds block `b` of every
/// row, in the order given (the last block may be shorter).
///
/// Panics if `block` is 0 or the rows differ in length.
pub fn time_blocks_mut<'a>(rows: impl IntoIterator<Item = &'a mut [f32]>, block: usize) -> Vec<Vec<&'a mut [f32]>> {
    assert!(block > 0, "block length must be positive");
    let mut out: Vec<Vec<&'a mut [f32]>> = Vec::new();
    let mut len = None;
    for row in rows {
        assert_eq!(*len.get_or_insert(row.len()), row.len(), "rows differ in length");
        for (b, piece) in row.chunks_mut(block).enumerate() {
            if b == out.len() {
                out.push(Vec::new());
            }
            out[b].push(piece);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_blocks_cover_every_row_once() {
        let mut data: Vec<f32> = (0..10).map(|v| v as f32).collect(); // 2 rows x 5
        let blocks = time_blocks_mut(data.as_chunks_mut::<5>().0.iter_mut().map(|r| &mut r[..]), 2);
        let shape: Vec<Vec<&[f32]>> = blocks.iter().map(|b| b.iter().map(|r| &r[..]).collect()).collect();
        assert_eq!(
            shape,
            vec![
                vec![&[0.0, 1.0][..], &[5.0, 6.0][..]],
                vec![&[2.0, 3.0][..], &[7.0, 8.0][..]],
                vec![&[4.0][..], &[9.0][..]],
            ]
        );
    }

    #[test]
    fn select_rows_keeps_the_given_order() {
        let mut data: Vec<f32> = (0..6).map(|v| v as f32).collect(); // 3 rows x 2
        let rows = select_rows_mut(&mut data, 2, &[2, 0]);
        assert_eq!(rows.iter().map(|r| r[0]).collect::<Vec<_>>(), vec![4.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "row selected twice")]
    fn select_rows_rejects_a_repeated_row() {
        let mut data = vec![0.0f32; 4];
        let _ = select_rows_mut(&mut data, 2, &[1, 1]);
    }
}
