//! Validated, immutable configuration for DSVLM simulations.
//!
//! Configuration types deliberately contain no runtime or learning state. Call
//! [`crate::config::DsvlmConfig::validate`] once before constructing a network and retain the
//! value as the reproducible description of the experiment.

use std::{error::Error, fmt};

/// Parameters of local plasticity and homeostasis.
pub mod learning;
/// Network size, connectivity, propagation, and weight bounds.
pub mod network;
/// Neuron dynamics and technical runtime limits.
pub mod runtime;

pub use learning::{HomeostasisConfig, LearningConfig};
pub use network::NetworkConfig;
pub use runtime::{NeuronConfig, RuntimeConfig};

/// A complete validated configuration for one deterministic experiment.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DsvlmConfig {
    /// Static topology-generation and propagation parameters.
    pub network: NetworkConfig,
    /// Parameters shared by the LIF neurons created for the experiment.
    pub neuron: NeuronConfig,
    /// Technical safeguards for event processing.
    pub runtime: RuntimeConfig,
    /// Local STDP and homeostasis parameters.
    pub learning: LearningConfig,
}

impl DsvlmConfig {
    /// Validates every slice and the bounds shared by network construction and
    /// learning.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.network.validate()?;
        self.neuron.validate()?;
        self.runtime.validate()?;
        self.learning.validate()?;

        if self.learning.min_weight < self.network.min_weight
            || self.learning.max_weight > self.network.max_weight
        {
            return Err(ConfigError::IncompatibleWeightBounds {
                network_min: self.network.min_weight,
                network_max: self.network.max_weight,
                learning_min: self.learning.min_weight,
                learning_max: self.learning.max_weight,
            });
        }

        let minimum_valid_threshold = self
            .neuron
            .resting_potential
            .max(self.neuron.reset_potential);
        if self.learning.homeostasis.min_threshold <= minimum_valid_threshold {
            return Err(ConfigError::HomeostasisThresholdNotAboveCellBaseline {
                min_threshold: self.learning.homeostasis.min_threshold,
                resting_potential: self.neuron.resting_potential,
                reset_potential: self.neuron.reset_potential,
            });
        }

        Ok(())
    }
}

/// Why a configuration cannot describe a valid simulation.
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigError {
    /// A floating-point field is NaN or infinite.
    NonFinite {
        /// Fully qualified field name.
        field: &'static str,
        /// Rejected value.
        value: f32,
    },
    /// A field that must be strictly positive is zero or negative.
    NonPositive {
        /// Fully qualified field name.
        field: &'static str,
        /// Rejected value.
        value: f32,
    },
    /// A field that must be non-negative is negative.
    Negative {
        /// Fully qualified field name.
        field: &'static str,
        /// Rejected value.
        value: f32,
    },
    /// A probability is outside the inclusive range `0..=1`.
    ProbabilityOutOfRange {
        /// Fully qualified field name.
        field: &'static str,
        /// Rejected value.
        value: f32,
    },
    /// The lower endpoint of a range exceeds its upper endpoint.
    InvalidRange {
        /// Fully qualified lower-bound field name.
        min_field: &'static str,
        /// Lower endpoint.
        min: f32,
        /// Fully qualified upper-bound field name.
        max_field: &'static str,
        /// Upper endpoint.
        max: f32,
    },
    /// A count or discrete duration that must be non-zero is zero.
    ZeroValue {
        /// Fully qualified field name.
        field: &'static str,
    },
    /// No neuron was requested for the generated network.
    EmptyNetwork,
    /// Excitatory and inhibitory population counts do not fit in `usize`.
    PopulationSizeOverflow,
    /// The learning bounds are not contained in the network's hard bounds.
    IncompatibleWeightBounds {
        /// Network-wide minimum magnitude.
        network_min: f32,
        /// Network-wide maximum magnitude.
        network_max: f32,
        /// Minimum requested by the learning rule.
        learning_min: f32,
        /// Maximum requested by the learning rule.
        learning_max: f32,
    },
    /// The resting potential is not strictly below the firing threshold.
    ThresholdNotAboveRestingPotential,
    /// The reset potential is not strictly below the firing threshold.
    ResetNotBelowThreshold,
    /// The lowest homeostatic threshold would violate the neuron's LIF ordering.
    HomeostasisThresholdNotAboveCellBaseline {
        /// Configured inclusive lower homeostatic bound.
        min_threshold: f32,
        /// Cell resting potential.
        resting_potential: f32,
        /// Cell reset potential.
        reset_potential: f32,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { field, value } => {
                write!(formatter, "{field} must be finite, got {value}")
            }
            Self::NonPositive { field, value } => {
                write!(formatter, "{field} must be greater than zero, got {value}")
            }
            Self::Negative { field, value } => {
                write!(formatter, "{field} must not be negative, got {value}")
            }
            Self::ProbabilityOutOfRange { field, value } => write!(
                formatter,
                "{field} must be between zero and one inclusive, got {value}"
            ),
            Self::InvalidRange {
                min_field,
                min,
                max_field,
                max,
            } => write!(
                formatter,
                "{min_field} ({min}) must not exceed {max_field} ({max})"
            ),
            Self::ZeroValue { field } => write!(formatter, "{field} must not be zero"),
            Self::EmptyNetwork => {
                formatter.write_str("the network must contain at least one neuron")
            }
            Self::PopulationSizeOverflow => {
                formatter.write_str("the total neuron population does not fit in usize")
            }
            Self::IncompatibleWeightBounds {
                network_min,
                network_max,
                learning_min,
                learning_max,
            } => write!(
                formatter,
                "learning weight bounds [{learning_min}, {learning_max}] must lie within network bounds [{network_min}, {network_max}]"
            ),
            Self::ThresholdNotAboveRestingPotential => {
                formatter.write_str("threshold must be greater than resting potential")
            }
            Self::ResetNotBelowThreshold => {
                formatter.write_str("reset potential must be lower than threshold")
            }
            Self::HomeostasisThresholdNotAboveCellBaseline {
                min_threshold,
                resting_potential,
                reset_potential,
            } => write!(
                formatter,
                "homeostasis minimum threshold ({min_threshold}) must be above resting ({resting_potential}) and reset ({reset_potential}) potentials"
            ),
        }
    }
}

impl Error for ConfigError {}

pub(crate) fn finite(value: f32, field: &'static str) -> Result<(), ConfigError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ConfigError::NonFinite { field, value })
    }
}

pub(crate) fn positive(value: f32, field: &'static str) -> Result<(), ConfigError> {
    finite(value, field)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ConfigError::NonPositive { field, value })
    }
}

pub(crate) fn non_negative(value: f32, field: &'static str) -> Result<(), ConfigError> {
    finite(value, field)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ConfigError::Negative { field, value })
    }
}

pub(crate) fn ordered_range(
    min: f32,
    min_field: &'static str,
    max: f32,
    max_field: &'static str,
) -> Result<(), ConfigError> {
    finite(min, min_field)?;
    finite(max, max_field)?;
    if min <= max {
        Ok(())
    } else {
        Err(ConfigError::InvalidRange {
            min_field,
            min,
            max_field,
            max,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_configuration_is_valid() {
        assert_eq!(DsvlmConfig::default().validate(), Ok(()));
    }

    #[test]
    fn aggregate_rejects_learning_bounds_outside_hard_network_bounds() {
        let mut config = DsvlmConfig::default();
        config.learning.max_weight = config.network.max_weight + 1.0;

        assert!(matches!(
            config.validate(),
            Err(ConfigError::IncompatibleWeightBounds { .. })
        ));
    }

    #[test]
    fn aggregate_rejects_homeostasis_threshold_that_core_would_reject() {
        let mut config = DsvlmConfig::default();
        config.learning.homeostasis.min_threshold = config.neuron.resting_potential;

        assert!(matches!(
            config.validate(),
            Err(ConfigError::HomeostasisThresholdNotAboveCellBaseline { .. })
        ));
    }
}
