
#[derive(Debug, Clone)]
pub struct NeuronParams {
    
    pub potential_decay: f32, // rate at which potential decays
    pub threshold: f32, // threshold for firing an action potential
    pub reset_potential: f32, // potential to reset to after a spike
    pub potential_min: f32, // minimum potential of the neuron
    pub potential_max: f32, // maximum potential of the neuron
    pub refractory_ticks: u32, // how many ticks the neuron is refractory after a spike
    pub neuron_type: NeuronType, // type of the neuron (excitatory or inhibitory)

}



#[derive(Debug, Clone)]
pub struct NeuronState {
    pub potential: f32, // current potential of the neuron
    pub refractory: u32, // remaining ticks in refractory period
    pub spike: bool, // spike status of the neuron

}


#[derive(Debug, Clone)]
pub enum NeuronType {
    Excitatory,
    Inhibitory,
}
