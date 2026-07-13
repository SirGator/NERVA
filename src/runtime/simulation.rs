use synapsen::SynapseParams;
use neuron::NeuronState;

pub struct SimulationParams {
    pub current_tick: u64, // current simulation tick
    
}
pub struct Network {
    pub neurons: Vec<NeuronState>, // list of neurons in the network
    pub synapses: Vec<SynapseParams>, // list of synapses in the network
    pub connection_matrix: Vec<NeuronState::neuronID<SynapseParams::synapseID>>>, // connection matrix representing the connections between neurons and synapses
}
