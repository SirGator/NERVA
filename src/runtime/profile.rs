//! Opt-in, bounded host measurements. Values never feed simulation decisions.

use std::time::Duration;

/// Fixed histogram ranges; zero is separate from small non-empty batches.
pub const HISTOGRAM_LABELS: [&str; 6] = ["0", "1–8", "9–32", "33–128", "129–512", "513+"];

/// Constant-memory distribution of non-negative integer observations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Histogram {
    /// Observation counts in HISTOGRAM_LABELS order.
    pub bins: [u64; 6],
    /// Number of recorded observations.
    pub samples: u64,
    /// Sum of observed values, saturating on overflow.
    pub total: u64,
    /// Largest recorded value.
    pub max: u64,
    /// Exact count of one-element observations, separate from the 1–8 bin.
    pub singletons: u64,
}

impl Histogram {
    pub(crate) fn observe(&mut self, value: usize) {
        let value = u64::try_from(value).unwrap_or(u64::MAX);
        let bin = match value {
            0 => 0,
            1..=8 => 1,
            9..=32 => 2,
            33..=128 => 3,
            129..=512 => 4,
            _ => 5,
        };
        self.bins[bin] = self.bins[bin].saturating_add(1);
        self.samples = self.samples.saturating_add(1);
        self.total = self.total.saturating_add(value);
        self.max = self.max.max(value);
        self.singletons = self.singletons.saturating_add(u64::from(value == 1));
    }

    fn merge(&mut self, other: &Self) {
        for (left, right) in self.bins.iter_mut().zip(other.bins) {
            *left = left.saturating_add(right);
        }
        self.samples = self.samples.saturating_add(other.samples);
        self.total = self.total.saturating_add(other.total);
        self.max = self.max.max(other.max);
        self.singletons = self.singletons.saturating_add(other.singletons);
    }
}

/// Host phases. Batch includes all stages; PreStdp is nested within Arrivals.
/// PropagationPlan and PropagationEnqueue are nested within Propagation.
/// SchedulerDequeue is outside Batch; it also records empty/failed attempts.
/// Other phases are disjoint. Compute includes synchronous device transfers
/// when a backend performs them. Export/publication are timed separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum ProfilePhase {
    Batch,
    AdvanceExport,
    AdvanceCompute,
    AdvancePublish,
    Arrivals,
    PreStdp,
    IntegratePrepare,
    IntegrateCompute,
    IntegratePublish,
    PostStdp,
    Utility,
    Maintenance,
    Propagation,
    Prediction,
    PropagationPlan,
    PropagationEnqueue,
    SchedulerDequeue,
}

impl ProfilePhase {
    /// Stable phase order for reports.
    pub const ALL: [Self; 17] = [
        Self::Batch,
        Self::AdvanceExport,
        Self::AdvanceCompute,
        Self::AdvancePublish,
        Self::Arrivals,
        Self::PreStdp,
        Self::IntegratePrepare,
        Self::IntegrateCompute,
        Self::IntegratePublish,
        Self::PostStdp,
        Self::Utility,
        Self::Maintenance,
        Self::Propagation,
        Self::Prediction,
        Self::PropagationPlan,
        Self::PropagationEnqueue,
        Self::SchedulerDequeue,
    ];
}

/// Host wall time, including both successful and failed phase attempts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhaseProfile {
    pub calls: u64,
    pub failures: u64,
    pub total: Duration,
    pub max: Duration,
}

/// Aggregate diagnostics, independent of simulated time and replay records.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeProfile {
    pub batches_started: u64,
    pub batches_completed: u64,
    pub batches_failed: u64,
    /// All queued events in each successfully dequeued exact-timestamp batch,
    /// before runtime validation (including maintenance and external inputs).
    pub scheduler_batch_size: Histogram,
    /// Requested row counts, including attempts that later fail.
    pub advance_batch_size: Histogram,
    pub integrate_batch_size: Histogram,
    /// One sample per integration row, including empty intrinsic checks.
    pub inputs_per_neuron: Histogram,
    /// One sample per dequeued timestamp, before runtime validation.
    pub synaptic_arrivals_per_timestamp: Histogram,
    /// One sample per successfully completed timestamp.
    pub spikes_per_timestamp: Histogram,
    /// All outgoing candidates visited, including disabled connections.
    pub outgoing_synapses_per_spike: Histogram,
    phases: [PhaseProfile; ProfilePhase::ALL.len()],
}

impl RuntimeProfile {
    pub fn phase(&self, phase: ProfilePhase) -> &PhaseProfile {
        &self.phases[phase as usize]
    }

    pub(crate) fn record_phase(&mut self, phase: ProfilePhase, elapsed: Duration, failed: bool) {
        let stats = &mut self.phases[phase as usize];
        stats.calls = stats.calls.saturating_add(1);
        stats.failures = stats.failures.saturating_add(u64::from(failed));
        stats.total = stats.total.saturating_add(elapsed);
        stats.max = stats.max.max(elapsed);
    }

    /// Combines independent runs without storing per-event samples.
    pub fn merge(&mut self, other: &Self) {
        self.batches_started = self.batches_started.saturating_add(other.batches_started);
        self.batches_completed = self
            .batches_completed
            .saturating_add(other.batches_completed);
        self.batches_failed = self.batches_failed.saturating_add(other.batches_failed);
        for (left, right) in [
            (&mut self.scheduler_batch_size, &other.scheduler_batch_size),
            (&mut self.advance_batch_size, &other.advance_batch_size),
            (&mut self.integrate_batch_size, &other.integrate_batch_size),
            (&mut self.inputs_per_neuron, &other.inputs_per_neuron),
            (
                &mut self.synaptic_arrivals_per_timestamp,
                &other.synaptic_arrivals_per_timestamp,
            ),
            (&mut self.spikes_per_timestamp, &other.spikes_per_timestamp),
            (
                &mut self.outgoing_synapses_per_spike,
                &other.outgoing_synapses_per_spike,
            ),
        ] {
            left.merge(right);
        }
        for (left, right) in self.phases.iter_mut().zip(&other.phases) {
            left.calls = left.calls.saturating_add(right.calls);
            left.failures = left.failures.saturating_add(right.failures);
            left.total = left.total.saturating_add(right.total);
            left.max = left.max.max(right.max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_ranges_are_disjoint_at_every_boundary() {
        let mut histogram = Histogram::default();
        for value in [0, 1, 8, 9, 32, 33, 128, 129, 512, 513] {
            histogram.observe(value);
        }
        assert_eq!(histogram.bins, [1, 2, 2, 2, 2, 1]);
        assert_eq!(histogram.samples, 10);
        assert_eq!(histogram.max, 513);
        assert_eq!(histogram.total, 1365);
        assert_eq!(histogram.singletons, 1);
    }
}
