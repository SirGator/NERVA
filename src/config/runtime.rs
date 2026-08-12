//! LIF-neuron and technical runtime configuration.

use super::{ConfigError, finite, positive};

/// Immutable parameters shared by one class of leaky integrate-and-fire cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuronConfig {
    /// Passive membrane equilibrium.
    pub resting_potential: f32,
    /// Potential restored immediately after emitting a spike.
    pub reset_potential: f32,
    /// Initial firing threshold.
    pub threshold: f32,
    /// Passive membrane time constant in microseconds.
    pub membrane_tau_us: f32,
    /// Duration after a spike during which input cannot trigger another spike.
    pub refractory_period_us: u64,
    /// Time constant of the neuron's local activity trace in microseconds.
    pub activity_trace_tau_us: f32,
}

impl NeuronConfig {
    /// Validates all LIF parameters.
    pub fn validate(&self) -> Result<(), ConfigError> {
        finite(self.resting_potential, "neuron.resting_potential")?;
        finite(self.reset_potential, "neuron.reset_potential")?;
        finite(self.threshold, "neuron.threshold")?;
        positive(self.membrane_tau_us, "neuron.membrane_tau_us")?;
        positive(self.activity_trace_tau_us, "neuron.activity_trace_tau_us")?;

        if self.threshold <= self.resting_potential {
            return Err(ConfigError::ThresholdNotAboveRestingPotential);
        }
        if self.reset_potential >= self.threshold {
            return Err(ConfigError::ResetNotBelowThreshold);
        }

        Ok(())
    }
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            resting_potential: -65.0,
            reset_potential: -70.0,
            threshold: -50.0,
            membrane_tau_us: 20_000.0,
            refractory_period_us: 2_000,
            activity_trace_tau_us: 100_000.0,
        }
    }
}

/// Technical event-processing safeguards; these do not define simulation
/// dynamics or introduce a global tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// Maximum number of events accepted in one exact-timestamp batch.
    pub max_events_per_batch: usize,
}

impl RuntimeConfig {
    /// Rejects safeguards that could never process an event.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_events_per_batch == 0 {
            Err(ConfigError::ZeroValue {
                field: "runtime.max_events_per_batch",
            })
        } else {
            Ok(())
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_events_per_batch: 1_000_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_positive_membrane_time_constant() {
        let config = NeuronConfig {
            membrane_tau_us: 0.0,
            ..NeuronConfig::default()
        };

        assert!(matches!(
            config.validate(),
            Err(ConfigError::NonPositive {
                field: "neuron.membrane_tau_us",
                ..
            })
        ));
    }

    #[test]
    fn rejects_threshold_at_resting_potential() {
        let config = NeuronConfig {
            threshold: NeuronConfig::default().resting_potential,
            ..NeuronConfig::default()
        };

        assert_eq!(
            config.validate(),
            Err(ConfigError::ThresholdNotAboveRestingPotential)
        );
    }

    #[test]
    fn refractory_period_may_be_zero() {
        let config = NeuronConfig {
            refractory_period_us: 0,
            ..NeuronConfig::default()
        };

        assert_eq!(config.validate(), Ok(()));
    }
}
