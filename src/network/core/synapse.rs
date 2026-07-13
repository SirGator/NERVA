use super::ids::NeuronId;
use super::plasticity::stdp::{StdpParams, StdpState};

pub struct SynapseParams {
    pub source: NeuronId, // the owned neuron
    pub target: NeuronId,
    pub initial_weight: f32, // initial weight of the synapse
    pub weight_min: f32, // minimum weight of the synapse
    pub weight_max: f32, // maximum weight of the synapse
    pub plasticity: StdpParams, // plasticity parameters for the synapse

}

pub struct SynapseState{
    pub weight: f32,
    pub stdp: StdpState,
}