//! Convenient aggregation of independent read-only metric families.

use crate::{
    core::{Network, Spike},
    environment::Pattern,
    runtime::ObservationEvent,
};

#[cfg(feature = "diagnostics")]
use crate::debug::Observer;

use super::{FiringMetrics, SequenceMetrics, WeightMetrics};

/// Metrics state updated only from copied observations and weight samples.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetricsCollector {
    /// Per-neuron firing data.
    pub firing: FiringMetrics,
    /// Latest sampled weight distribution.
    pub weights: WeightMetrics,
    /// Latest classified probe sequence.
    pub sequence: SequenceMetrics,
}

impl MetricsCollector {
    /// Records one immutable spike.
    pub fn observe_spike(&mut self, spike: Spike) {
        self.firing.observe(spike);
    }

    /// Consumes the metric-relevant part of one immutable runtime observation.
    pub fn observe_event(&mut self, event: &ObservationEvent) {
        if let ObservationEvent::SpikeEmitted(spike) = event {
            self.observe_spike(*spike);
        }
    }

    /// Replaces the derived weight snapshot.
    pub fn sample_weights(
        &mut self,
        weights: impl IntoIterator<Item = f32>,
        min_weight: f32,
        max_weight: f32,
    ) {
        self.weights = WeightMetrics::from_weights(weights, min_weight, max_weight);
    }

    /// Samples current magnitudes through the network's immutable API.
    pub fn sample_network_weights(&mut self, network: &Network, min_weight: f32, max_weight: f32) {
        self.sample_weights(
            network.synapses().map(|synapse| synapse.weight().get()),
            min_weight,
            max_weight,
        );
    }

    /// Replaces the classified A-only probe result.
    pub fn classify_probe(
        &mut self,
        observed: impl IntoIterator<Item = (crate::core::SimTime, Pattern)>,
    ) {
        self.sequence = SequenceMetrics::from_probe(observed);
    }
}

#[cfg(feature = "diagnostics")]
impl Observer<ObservationEvent> for MetricsCollector {
    fn observe(&mut self, event: &ObservationEvent) {
        self.observe_event(event);
    }
}

#[cfg(feature = "diagnostics")]
impl Observer<Spike> for MetricsCollector {
    fn observe(&mut self, spike: &Spike) {
        self.observe_spike(*spike);
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{NeuronId, SimTime};

    use super::*;

    #[test]
    fn runtime_observer_collects_spikes_and_ignores_unrelated_events() {
        let mut metrics = MetricsCollector::default();
        metrics.observe_event(&ObservationEvent::ExternalInput {
            time: SimTime(1),
            target: NeuronId(3),
            amplitude: 0.5,
        });
        metrics.observe_event(&ObservationEvent::SpikeEmitted(Spike::new(
            NeuronId(3),
            SimTime(2),
        )));

        assert_eq!(metrics.firing.spike_count(NeuronId(3)), 1);
        assert_eq!(metrics.firing.spike_times(NeuronId(3)), &[SimTime(2)]);
    }

    #[test]
    fn collector_replaces_weight_and_sequence_samples() {
        let mut metrics = MetricsCollector::default();
        metrics.sample_weights([0.0, 0.5, 1.0], 0.0, 1.0);
        metrics.classify_probe([
            (SimTime(3), Pattern::B),
            (SimTime(4), Pattern::C),
            (SimTime(5), Pattern::D),
        ]);

        assert_eq!(metrics.weights.count, 3);
        assert_eq!(metrics.weights.at_min_fraction, 1.0 / 3.0);
        assert_eq!(metrics.weights.at_max_fraction, 1.0 / 3.0);
        assert_eq!(metrics.sequence.transition_hits, 3);
    }
}
