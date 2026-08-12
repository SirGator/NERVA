//! Runtime-facing boundary for local plasticity.

use crate::core::{Neuron, Polarity, SimTime, Synapse};

/// A local learning rule notified by the event runtime.
///
/// Implementations may inspect only the affected synapse, its postsynaptic
/// neuron, and incoming synapses of a neuron that just fired. The core does not
/// depend on this trait.
pub trait PlasticityRule {
    /// Whether this concrete local rule accepts one affected synapse.
    ///
    /// The runtime supplies presynaptic polarity but owns no learning policy.
    /// Pair-STDP uses this to admit only plastic internal excitation; a future
    /// inhibitory rule can choose a different local eligibility predicate.
    fn accepts(&self, synapse: &Synapse, presynaptic_polarity: Polarity) -> bool;

    /// Handles delivery of one presynaptic spike at a synapse.
    fn on_pre_arrival(&mut self, synapse: &mut Synapse, post: &Neuron, time: SimTime);

    /// Handles a postsynaptic spike for the neuron's incoming synapses.
    fn on_post_spike(&mut self, neuron: &Neuron, incoming: &mut [Synapse], time: SimTime);
}

/// A rule that deliberately leaves all local state unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoPlasticity;

impl PlasticityRule for NoPlasticity {
    fn accepts(&self, _synapse: &Synapse, _presynaptic_polarity: Polarity) -> bool {
        false
    }

    fn on_pre_arrival(&mut self, _synapse: &mut Synapse, _post: &Neuron, _time: SimTime) {}

    fn on_post_spike(&mut self, _neuron: &Neuron, _incoming: &mut [Synapse], _time: SimTime) {}
}
