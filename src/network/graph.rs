use std::collections::HashMap;

use crate::common::{NeuronId, SynapseId};

/// Directed adjacency lists for synapses in a neural network.
pub struct NetworkGraph {
    /// Synapses originating at each neuron.
    pub outgoing_synapses: HashMap<NeuronId, Vec<SynapseId>>,

    /// Synapses terminating at each neuron.
    pub incoming_synapses: HashMap<NeuronId, Vec<SynapseId>>,
}
