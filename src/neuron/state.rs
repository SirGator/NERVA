use crate::common::SimTimeUs;

/// Mutable simulation state of one neuron.
#[derive(Clone, Debug)]
pub struct NeuronState {
    /// Current membrane potential.
    pub membrane_potential: f32,

    /// Timestamp at which continuous state was last decayed or integrated.
    pub last_update_at: SimTimeUs,
    /// Earliest timestamp at which the neuron can fire again.
    pub refractory_until: SimTimeUs,
    /// Timestamp of the most recently emitted spike, if any.
    pub last_spike_at: Option<SimTimeUs>,

    /// Decaying trace of recent spikes.
    pub activity_trace: f32,
    /// Decaying threshold-reducing adaptation term.
    pub adaptation: f32,

    /// Total number of spikes emitted since construction.
    pub spike_count: u64,
}

impl NeuronState {
    /// Creates the initial resting state at `start_time`.
    pub fn new(start_time: SimTimeUs, resting_potential: f32) -> Self {
        Self {
            membrane_potential: resting_potential,
            last_update_at: start_time,
            refractory_until: start_time,
            last_spike_at: None,
            activity_trace: 0.0,
            adaptation: 0.0,
            spike_count: 0,
        }
    }
}
