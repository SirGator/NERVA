//! Time evolution of a single neuron.

use crate::common::SimTimeUs;

use super::{Neuron, NeuronRuntimeError};

impl Neuron {
    /// Decays continuous state forward to `requested` without applying external input.
    ///
    /// Returns an error if the requested time precedes the last update.
    pub fn advance_to(&mut self, requested: SimTimeUs) -> Result<(), NeuronRuntimeError> {
        let current = self.state.last_update_at;
        let elapsed_us = requested
            .duration_since(current)
            .ok_or(NeuronRuntimeError::TimeWentBackwards { current, requested })?;

        if elapsed_us == 0 {
            return Ok(());
        }

        self.state.membrane_potential = decay_towards(
            self.state.membrane_potential,
            self.params.resting_potential,
            elapsed_us,
            self.params.membrane_tau_us,
        );
        self.state.activity_trace = decay_to_zero(
            self.state.activity_trace,
            elapsed_us,
            self.params.activity_trace_tau_us,
        );
        self.state.adaptation = decay_to_zero(
            self.state.adaptation,
            elapsed_us,
            self.params.adaptation_tau_us,
        );
        self.state.last_update_at = requested;

        Ok(())
    }

    /// Applies one aggregated input batch and returns whether it emitted a spike.
    ///
    /// Inputs arriving during the refractory period are ignored after time decay.
    pub fn integrate_input_batch(
        &mut self,
        arrives_at: SimTimeUs,
        summed_input: f32,
        noise_sample: f32,
    ) -> Result<bool, NeuronRuntimeError> {
        if !summed_input.is_finite() {
            return Err(NeuronRuntimeError::NonFiniteInput);
        }

        if !noise_sample.is_finite() {
            return Err(NeuronRuntimeError::NonFiniteNoise);
        }

        self.advance_to(arrives_at)?;

        if arrives_at < self.state.refractory_until {
            return Ok(false);
        }

        self.state.membrane_potential += summed_input + noise_sample;

        let effective_threshold =
            self.params.threshold + self.state.adaptation;

        if self.state.membrane_potential >= effective_threshold {
            self.emit_spike(arrives_at);
            return Ok(true);
        }

        Ok(false)
    }

    /// Resets state and records a spike at `emitted_at`.
    pub fn emit_spike(&mut self, emitted_at: SimTimeUs) {
        self.state.membrane_potential = self.params.reset_potential;
        self.state.refractory_until =
            emitted_at.saturating_add_us(self.params.refractory_period_us);
        self.state.last_spike_at = Some(emitted_at);
        self.state.activity_trace += 1.0;
        self.state.adaptation += self.params.adaptation_increment;
        self.state.spike_count = self.state.spike_count.saturating_add(1);
    }
}

/// Exponentially decays `value` toward an equilibrium over elapsed simulation time.
fn decay_towards(value: f32, equilibrium: f32, elapsed_us: u64, tau_us: f32) -> f32 {
    equilibrium + (value - equilibrium) * decay_factor(elapsed_us, tau_us)
}

/// Exponentially decays `value` toward zero over elapsed simulation time.
fn decay_to_zero(value: f32, elapsed_us: u64, tau_us: f32) -> f32 {
    value * decay_factor(elapsed_us, tau_us)
}

/// Computes the exponential decay factor for `elapsed_us` and a time constant.
fn decay_factor(elapsed_us: u64, tau_us: f32) -> f32 {
    (-(elapsed_us as f32) / tau_us).exp()
}

#[cfg(test)]
mod tests {
    use crate::{
        common::{AreaId, NeuronId, PopulationId, SimTimeUs},
        neuron::{
            CellType, Neuron, NeuronAddress, NeuronModulatorSensitivity, NeuronParams, NeuronRole,
            NeuronRuntimeError, NeuronSpec, PrimaryTransmitter,
        },
    };

    fn neuron() -> Neuron {
        let spec = NeuronSpec {
            id: NeuronId(1),
            address: NeuronAddress {
                area: AreaId(1),
                layer: None,
                population: PopulationId(1),
                local_index: 0,
            },
            role: NeuronRole::LocalProcessing,
            cell_type: CellType::Generic,
            transmitter: PrimaryTransmitter::Glutamate,
            params: NeuronParams {
                resting_potential: 0.0,
                reset_potential: 0.0,
                threshold: 1.0,
                membrane_tau_us: 20_000.0,
                refractory_period_us: 2_000,
                activity_trace_tau_us: 100_000.0,
                adaptation_increment: 0.1,
                adaptation_tau_us: 200_000.0,
                noise_strength: 0.0,
            },
            modulator_sensitivity: NeuronModulatorSensitivity::default(),
        };

        Neuron::new(spec, SimTimeUs::ZERO).expect("valid test neuron")
    }

    #[test]
    fn input_at_threshold_emits_a_spike() {
        let mut neuron = neuron();

        assert!(
            neuron
                .integrate_input_batch(SimTimeUs(10), 1.0, 0.0)
                .expect("finite input")
        );
        assert_eq!(neuron.state().spike_count, 1);
        assert_eq!(neuron.state().last_spike_at, Some(SimTimeUs(10)));
        assert_eq!(neuron.state().membrane_potential, 0.0);
    }

    #[test]
    fn cannot_advance_backwards_in_time() {
        let mut neuron = neuron();
        neuron.advance_to(SimTimeUs(10)).expect("forward time");

        assert_eq!(
            neuron.advance_to(SimTimeUs(9)),
            Err(NeuronRuntimeError::TimeWentBackwards {
                current: SimTimeUs(10),
                requested: SimTimeUs(9),
            })
        );
    }
}









