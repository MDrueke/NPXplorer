//! Computations that run on their own thread while the UI keeps drawing: channel
//! classification, the power spectrum, the PSTH and the notch scan. The UI owns an
//! `Option<BackgroundJob<T>>`, polls it every frame and shows its progress.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

/// How often a running job's progress is redrawn.
const PROGRESS_REPAINT: Duration = Duration::from_millis(80);

/// Shared between a job's thread and the UI.
#[derive(Default)]
pub struct JobControl {
    /// set by the UI; the work stops at its next check
    pub cancel: AtomicBool,
    /// units of work done and to do, for the progress bar (the work may update `total`)
    pub done: AtomicUsize,
    pub total: AtomicUsize,
}

impl JobControl {
    /// These flags, lent to a computation.
    pub fn control(&self) -> crate::control::Control<'_> {
        crate::control::Control { cancel: &self.cancel, done: &self.done, total: Some(&self.total) }
    }
}

/// How a job ended.
pub enum Outcome<T> {
    Done(T),
    Failed(String),
    /// cancelled by the UI; whatever the work returned is dropped
    Cancelled,
}

pub struct BackgroundJob<T> {
    rx: mpsc::Receiver<Result<T, String>>,
    control: Arc<JobControl>,
}

impl<T: Send + 'static> BackgroundJob<T> {
    /// Run `work` on a new thread. `total` is the initial progress total.
    pub fn spawn<F>(ctx: &egui::Context, total: usize, work: F) -> Self
    where
        F: FnOnce(&JobControl) -> Result<T, String> + Send + 'static,
    {
        let control = Arc::new(JobControl { total: AtomicUsize::new(total), ..Default::default() });
        let (tx, rx) = mpsc::channel();
        let thread_control = Arc::clone(&control);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(work(&thread_control));
            ctx.request_repaint();
        });
        BackgroundJob { rx, control }
    }
}

impl<T> BackgroundJob<T> {
    /// The outcome once the work has finished; `None` while it runs (a repaint is then
    /// scheduled so the progress stays current).
    pub fn poll(&self, ctx: &egui::Context) -> Option<Outcome<T>> {
        let res = match self.rx.try_recv() {
            Ok(res) => res,
            Err(mpsc::TryRecvError::Empty) => {
                ctx.request_repaint_after(PROGRESS_REPAINT);
                return None;
            }
            Err(mpsc::TryRecvError::Disconnected) => Err("the computation stopped unexpectedly".to_string()),
        };
        Some(if self.is_cancelling() {
            Outcome::Cancelled
        } else {
            match res {
                Ok(v) => Outcome::Done(v),
                Err(e) => Outcome::Failed(e),
            }
        })
    }

    pub fn cancel(&self) {
        self.control.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelling(&self) -> bool {
        self.control.cancel.load(Ordering::Relaxed)
    }

    /// (done, total) units of work.
    pub fn progress(&self) -> (usize, usize) {
        (self.control.done.load(Ordering::Relaxed), self.control.total.load(Ordering::Relaxed))
    }
}

/// A job that is replaced or dropped stops computing.
impl<T> Drop for BackgroundJob<T> {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait<T>(job: &BackgroundJob<T>, ctx: &egui::Context) -> Outcome<T> {
        loop {
            if let Some(o) = job.poll(ctx) {
                return o;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_job_reports_its_result_and_progress() {
        let ctx = egui::Context::default();
        let job = BackgroundJob::spawn(&ctx, 3, |c| {
            c.done.store(3, Ordering::Relaxed);
            Ok(42)
        });
        assert!(matches!(wait(&job, &ctx), Outcome::Done(42)));
        assert_eq!(job.progress(), (3, 3));
    }

    #[test]
    fn a_cancelled_job_reports_cancelled_whatever_it_returns() {
        let ctx = egui::Context::default();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let job = BackgroundJob::spawn(&ctx, 1, move |_| {
            go_rx.recv().unwrap();
            Ok(1)
        });
        job.cancel();
        go_tx.send(()).unwrap();
        assert!(matches!(wait(&job, &ctx), Outcome::Cancelled));
    }

    #[test]
    fn a_failed_job_reports_its_error() {
        let ctx = egui::Context::default();
        let job: BackgroundJob<()> = BackgroundJob::spawn(&ctx, 1, |_| Err("no events".to_string()));
        assert!(matches!(wait(&job, &ctx), Outcome::Failed(e) if e == "no events"));
    }
}
