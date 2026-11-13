//! Cancelling a long computation and following its progress.

use std::sync::atomic::{AtomicBool, AtomicUsize};

/// The caller's cancel flag and progress counters, lent to a computation for the
/// duration of the call (a background job lends its own, an export its own).
#[derive(Clone, Copy)]
pub struct Control<'a> {
    /// set by the caller; the computation stops at its next check
    pub cancel: &'a AtomicBool,
    /// units of work done
    pub done: &'a AtomicUsize,
    /// units of work in total, for computations that only know it once they run
    pub total: Option<&'a AtomicUsize>,
}
