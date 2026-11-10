//! Runtime-facing boundary for local plasticity.

use crate::core::{Neuron, Polarity, SimTime, Synapse};
use crate::primitives::Weight;

/// Read-only prediction of one delayed local update, before f32 rounding/bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelayedLearningUpdate {
    pub apply_weight: bool,
    pub decayed_tag: f32,
    pub tag_integral_seconds: f64,
    pub unrounded_delta: f64,
    /// Preview of the exact f32 value published by this integration.
    pub next_weight: Weight,
    /// Signed unpublished mass before and after this time integration.
    pub residual_before: f64,
    pub residual_after: f64,
    /// Signed mass rejected by weight bounds; never retained as hidden debt.
    pub clipped_delta: f64,
    /// Signed difference after accounting for actual, residual and clipped mass.
    pub rounding_error: f64,
}

/// A local learning rule notified by the event runtime.
///
/// Implementations may inspect only the affected synapse, its postsynaptic
/// neuron, and incoming synapses of a neuron that just fired. The core does not
/// depend on this trait.
pub trait PlasticityRule {
    /// Whether delayed local state needs advancement even without new pairs.
    /// Neutral rules retain the original event path and incur no graph scan.
    fn needs_time_updates(&self) -> bool {
        false
    }

    /// Advances a synapse-local delayed learning state against its current
    /// postsynaptic receptor signal. Disabled weight learning still lets time
    /// pass, so a frozen interval is never learned retroactively on resume.
    fn advance_time(
        &mut self,
        _synapse: &mut Synapse,
        _post: &Neuron,
        _time: SimTime,
        _enabled: bool,
    ) {
    }

    /// Optional read-only local arithmetic for diagnostic observation.
    /// This must not advance traces or mutate the rule.
    fn delayed_update_diagnostics(
        &self,
        _synapse: &Synapse,
        _post: &Neuron,
        _time: SimTime,
        _enabled: bool,
    ) -> Option<DelayedLearningUpdate> {
        None
    }

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
