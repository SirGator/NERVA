use crate::common::{NeuronId, SynapseId};

use super::{Plasticity, SynapseParams, SynapseState};

/// Sign of a synapse's contribution to the target neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynapticEffect {
    /// Raises the target neuron's integrated input.
    Excitatory,
    /// Lowers the target neuron's integrated input.
    Inhibitory,
}

/// Directed connection between two neurons.
pub struct Synapse {
    /// Stable identifier of this synapse.
    pub id: SynapseId,

    /// Neuron that emits into this synapse.
    pub pre_neuron: NeuronId,
    /// Neuron receiving this synapse's output.
    pub post_neuron: NeuronId,

    /// Whether the synapse is excitatory or inhibitory.
    pub effect: SynapticEffect,

    /// Fixed transmission and weight bounds.
    pub params: SynapseParams,
    /// Mutable transmission state.
    pub state: SynapseState,

    /// Rule and mutable state governing local plasticity.
    pub plasticity: Plasticity,
}
