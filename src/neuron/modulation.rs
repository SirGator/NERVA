/// Change to a neuron's behavior caused by one modulator.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModulatorResponse {
    /// Additive change to the input gain.
    pub input_gain_delta: f32,
    /// Additive shift of the firing threshold.
    pub threshold_shift: f32,
    /// Additive change to the adaptation gain.
    pub adaptation_gain_delta: f32,
}

/// Sensitivity of a neuron to each supported neuromodulator.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeuronModulatorSensitivity {
    /// Response to dopamine.
    pub dopamine: ModulatorResponse,
    /// Response to acetylcholine.
    pub acetylcholine: ModulatorResponse,
    /// Response to noradrenaline.
    pub noradrenaline: ModulatorResponse,
    /// Response to serotonin.
    pub serotonin: ModulatorResponse,
}
