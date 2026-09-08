//! LIF-neuron and technical runtime configuration.

use super::{ConfigError, finite, non_negative, positive};

/// Continuous intrinsic capabilities shared by one neuron's local dynamics.
///
/// The gains form a continuous capability vector rather than selecting a
/// discrete neuron type. A zero gain disables only that capability. Drive
/// gains are currents in potential units per second; threshold adaptation is
/// measured directly in potential units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntrinsicDynamicsConfig {
    /// Constant autonomous current added to the homeostatic intrinsic current.
    pub intrinsic_drive: f32,
    /// Positive current added after each spike.
    pub burst_gain: f32,
    /// Exponential time constant of the burst current in microseconds.
    pub burst_tau_us: f32,
    /// Negative current added after each spike.
    pub adaptation_gain: f32,
    /// Exponential time constant of the adaptation current in microseconds.
    pub adaptation_tau_us: f32,
    /// Increase of the local firing threshold after each spike.
    pub threshold_adaptation_gain: f32,
    /// Exponential time constant of threshold adaptation in microseconds.
    pub threshold_adaptation_tau_us: f32,
    /// Positive after-current created by inhibitory input magnitude.
    pub rebound_gain: f32,
    /// Exponential time constant of the rebound current in microseconds.
    pub rebound_tau_us: f32,
}

impl IntrinsicDynamicsConfig {
    /// Validates every capability independently of whether its gain is zero.
    pub fn validate(&self) -> Result<(), ConfigError> {
        finite(self.intrinsic_drive, "neuron.intrinsic.intrinsic_drive")?;
        non_negative(self.burst_gain, "neuron.intrinsic.burst_gain")?;
        positive(self.burst_tau_us, "neuron.intrinsic.burst_tau_us")?;
        non_negative(self.adaptation_gain, "neuron.intrinsic.adaptation_gain")?;
        positive(self.adaptation_tau_us, "neuron.intrinsic.adaptation_tau_us")?;
        non_negative(
            self.threshold_adaptation_gain,
            "neuron.intrinsic.threshold_adaptation_gain",
        )?;
        positive(
            self.threshold_adaptation_tau_us,
            "neuron.intrinsic.threshold_adaptation_tau_us",
        )?;
        non_negative(self.rebound_gain, "neuron.intrinsic.rebound_gain")?;
        positive(self.rebound_tau_us, "neuron.intrinsic.rebound_tau_us")?;
        Ok(())
    }

    /// Whether any capability can alter the default LIF trajectory.
    pub fn is_active(&self) -> bool {
        self.intrinsic_drive != 0.0
            || self.burst_gain != 0.0
            || self.adaptation_gain != 0.0
            || self.threshold_adaptation_gain != 0.0
            || self.rebound_gain != 0.0
    }
}

impl Default for IntrinsicDynamicsConfig {
    fn default() -> Self {
        Self {
            intrinsic_drive: 0.0,
            burst_gain: 0.0,
            burst_tau_us: 20_000.0,
            adaptation_gain: 0.0,
            adaptation_tau_us: 100_000.0,
            threshold_adaptation_gain: 0.0,
            threshold_adaptation_tau_us: 100_000.0,
            rebound_gain: 0.0,
            rebound_tau_us: 20_000.0,
        }
    }
}

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
    /// Optional continuous intrinsic capabilities of this cell.
    pub intrinsic: IntrinsicDynamicsConfig,
}

impl NeuronConfig {
    /// Validates all LIF parameters.
    pub fn validate(&self) -> Result<(), ConfigError> {
        finite(self.resting_potential, "neuron.resting_potential")?;
        finite(self.reset_potential, "neuron.reset_potential")?;
        finite(self.threshold, "neuron.threshold")?;
        positive(self.membrane_tau_us, "neuron.membrane_tau_us")?;
        positive(self.activity_trace_tau_us, "neuron.activity_trace_tau_us")?;
        self.intrinsic.validate()?;

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
            intrinsic: IntrinsicDynamicsConfig::default(),
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

    #[test]
    fn validates_intrinsic_capabilities_even_when_the_gain_is_zero() {
        let config = NeuronConfig {
            intrinsic: IntrinsicDynamicsConfig {
                burst_tau_us: 0.0,
                ..IntrinsicDynamicsConfig::default()
            },
            ..NeuronConfig::default()
        };

        assert!(matches!(
            config.validate(),
            Err(ConfigError::NonPositive {
                field: "neuron.intrinsic.burst_tau_us",
                ..
            })
        ));
    }
}
