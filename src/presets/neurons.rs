use crate::neuron::NeuronParams;

/// Returns normalized parameters for an excitatory pyramidal neuron.
pub fn pyramidal_params() -> NeuronParams {
    NeuronParams {
        resting_potential: 0.0,
        reset_potential: 0.0,
        threshold: 1.0,
        membrane_tau_us: 20_000.0,
        refractory_period_us: 2_000,
        activity_trace_tau_us: 100_000.0,
        adaptation_increment: 0.1,
        adaptation_tau_us: 200_000.0,
        noise_strength: 0.0,
    }
}

/// Returns normalized parameters for a fast-spiking inhibitory interneuron.
pub fn fast_spiking_params() -> NeuronParams {
    NeuronParams {
        resting_potential: 0.0,
        reset_potential: 0.0,
        threshold: 0.8,
        membrane_tau_us: 10_000.0,
        refractory_period_us: 1_000,
        activity_trace_tau_us: 50_000.0,
        adaptation_increment: 0.0,
        adaptation_tau_us: 100_000.0,
        noise_strength: 0.0,
    }
}
