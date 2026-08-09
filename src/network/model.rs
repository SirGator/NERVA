use std::collections::HashMap;

use crate::{
    area::BrainArea,
    common::{AreaId, NeuronId, SynapseId},
    neuron::Neuron,
    synapse::Synapse,
};

/// Aggregate storage for the network's neurons, synapses, and areas.
pub struct NeuralNetwork {
    /// Neurons indexed by their stable identifiers.
    pub neurons: HashMap<NeuronId, Neuron>,
    /// Synapses indexed by their stable identifiers.
    pub synapses: HashMap<SynapseId, Synapse>,
    /// Areas indexed by their stable identifiers.
    pub areas: HashMap<AreaId, BrainArea>,
}
