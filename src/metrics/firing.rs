//! Per-neuron firing measurements.

use std::collections::{BTreeMap, BTreeSet};

use crate::core::{NeuronId, SimTime, Spike};

#[cfg(feature = "diagnostics")]
use crate::debug::Observer;

/// Spike counts and timestamps collected without network access.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FiringMetrics {
    spikes: BTreeMap<NeuronId, Vec<SimTime>>,
}

impl FiringMetrics {
    /// Adds an immutable spike record.
    pub fn observe(&mut self, spike: Spike) {
        let times = self.spikes.entry(spike.neuron_id).or_default();
        let insertion_index = times.partition_point(|&time| time <= spike.time);
        times.insert(insertion_index, spike.time);
    }

    /// Number of observed spikes for one neuron.
    pub fn spike_count(&self, neuron: NeuronId) -> usize {
        self.spikes.get(&neuron).map_or(0, Vec::len)
    }

    /// Observed timestamps for one neuron.
    pub fn spike_times(&self, neuron: NeuronId) -> &[SimTime] {
        self.spikes.get(&neuron).map_or(&[], Vec::as_slice)
    }

    /// Mean rate in hertz over an explicit measurement interval.
    pub fn rate_hz(&self, neuron: NeuronId, duration_us: u64) -> f32 {
        if duration_us == 0 {
            return 0.0;
        }
        self.spike_count(neuron) as f32 * 1_000_000.0 / duration_us as f32
    }

    /// Fraction of the supplied population with no recorded spikes.
    pub fn silent_fraction(&self, neurons: impl IntoIterator<Item = NeuronId>) -> f32 {
        let neurons: BTreeSet<_> = neurons.into_iter().collect();
        if neurons.is_empty() {
            return 0.0;
        }
        let silent = neurons
            .iter()
            .filter(|&&neuron| self.spike_count(neuron) == 0)
            .count();
        silent as f32 / neurons.len() as f32
    }

    /// Total number of spikes across all observed neurons.
    pub fn total_spike_count(&self) -> usize {
        self.spikes.values().map(Vec::len).sum()
    }

    /// Number of neurons for which at least one spike was observed.
    pub fn active_neuron_count(&self) -> usize {
        self.spikes.len()
    }

    /// Mean per-neuron rate over an explicit population and interval.
    pub fn mean_rate_hz(
        &self,
        neurons: impl IntoIterator<Item = NeuronId>,
        duration_us: u64,
    ) -> f32 {
        let neurons: BTreeSet<_> = neurons.into_iter().collect();
        if neurons.is_empty() || duration_us == 0 {
            return 0.0;
        }
        let spike_count: usize = neurons.iter().map(|&neuron| self.spike_count(neuron)).sum();
        spike_count as f32 * 1_000_000.0 / duration_us as f32 / neurons.len() as f32
    }

    /// Iterates over active neurons in stable ID order.
    pub fn iter(&self) -> impl Iterator<Item = (NeuronId, &[SimTime])> {
        self.spikes
            .iter()
            .map(|(&neuron, times)| (neuron, times.as_slice()))
    }
}

#[cfg(feature = "diagnostics")]
impl Observer<Spike> for FiringMetrics {
    fn observe(&mut self, spike: &Spike) {
        FiringMetrics::observe(self, *spike);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_times_chronologically_and_iterates_by_neuron_id() {
        let mut metrics = FiringMetrics::default();
        metrics.observe(Spike::new(NeuronId(2), SimTime(20)));
        metrics.observe(Spike::new(NeuronId(1), SimTime(30)));
        metrics.observe(Spike::new(NeuronId(2), SimTime(10)));

        assert_eq!(
            metrics.spike_times(NeuronId(2)),
            &[SimTime(10), SimTime(20)]
        );
        assert_eq!(
            metrics.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            [NeuronId(1), NeuronId(2)]
        );
        assert_eq!(metrics.total_spike_count(), 3);
        assert_eq!(metrics.active_neuron_count(), 2);
    }

    #[test]
    fn rates_and_silent_fraction_use_unique_population_ids() {
        let mut metrics = FiringMetrics::default();
        metrics.observe(Spike::new(NeuronId(1), SimTime(10)));
        metrics.observe(Spike::new(NeuronId(1), SimTime(20)));

        assert_eq!(metrics.rate_hz(NeuronId(1), 1_000_000), 2.0);
        assert_eq!(
            metrics.mean_rate_hz([NeuronId(1), NeuronId(2)], 1_000_000),
            1.0
        );
        assert_eq!(
            metrics.silent_fraction([NeuronId(1), NeuronId(2), NeuronId(2)]),
            0.5
        );
    }

    #[test]
    fn empty_or_zero_duration_measurements_are_zero() {
        let metrics = FiringMetrics::default();

        assert_eq!(metrics.rate_hz(NeuronId(1), 0), 0.0);
        assert_eq!(metrics.mean_rate_hz([], 10), 0.0);
        assert_eq!(metrics.silent_fraction([]), 0.0);
    }
}
