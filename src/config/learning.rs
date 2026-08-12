//! Pair-STDP and local homeostasis configuration.

use super::{ConfigError, non_negative, ordered_range, positive};

/// Parameters for local threshold homeostasis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HomeostasisConfig {
    /// Whether local threshold maintenance events are active.
    pub enabled: bool,
    /// Per-neuron target firing rate in hertz.
    pub target_rate_hz: f32,
    /// Threshold change per unit local rate error.
    pub adjustment_rate: f32,
    /// Inclusive lower threshold bound.
    pub min_threshold: f32,
    /// Inclusive upper threshold bound.
    pub max_threshold: f32,
}

impl HomeostasisConfig {
    /// Validates local-only homeostasis parameters.
    pub fn validate(&self) -> Result<(), ConfigError> {
        non_negative(self.target_rate_hz, "homeostasis.target_rate_hz")?;
        non_negative(self.adjustment_rate, "homeostasis.adjustment_rate")?;
        ordered_range(
            self.min_threshold,
            "homeostasis.min_threshold",
            self.max_threshold,
            "homeostasis.max_threshold",
        )?;

        Ok(())
    }
}

impl Default for HomeostasisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_rate_hz: 5.0,
            adjustment_rate: 0.01,
            min_threshold: -60.0,
            max_threshold: -40.0,
        }
    }
}

/// Local pair-STDP configuration and its hard weight bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearningConfig {
    /// Whether synaptic learning is active.
    pub enabled: bool,
    /// Potentiation amplitude for a causal pre-before-post pair.
    pub a_plus: f32,
    /// Depression magnitude for an anti-causal post-before-pre pair.
    pub a_minus: f32,
    /// Potentiation trace time constant in microseconds.
    pub tau_plus_us: f32,
    /// Depression trace time constant in microseconds.
    pub tau_minus_us: f32,
    /// Maximum temporal separation considered a spike pair.
    pub stdp_window_us: u64,
    /// Inclusive lower non-negative weight bound.
    pub min_weight: f32,
    /// Inclusive upper non-negative weight bound.
    pub max_weight: f32,
    /// Optional local threshold stabilization parameters.
    pub homeostasis: HomeostasisConfig,
}

impl LearningConfig {
    /// Validates STDP rates, time constants, window, bounds, and homeostasis.
    pub fn validate(&self) -> Result<(), ConfigError> {
        non_negative(self.a_plus, "learning.a_plus")?;
        non_negative(self.a_minus, "learning.a_minus")?;
        positive(self.tau_plus_us, "learning.tau_plus_us")?;
        positive(self.tau_minus_us, "learning.tau_minus_us")?;
        if self.stdp_window_us == 0 {
            return Err(ConfigError::ZeroValue {
                field: "learning.stdp_window_us",
            });
        }

        non_negative(self.min_weight, "learning.min_weight")?;
        non_negative(self.max_weight, "learning.max_weight")?;
        ordered_range(
            self.min_weight,
            "learning.min_weight",
            self.max_weight,
            "learning.max_weight",
        )?;
        self.homeostasis.validate()?;

        Ok(())
    }
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus_us: 20_000.0,
            tau_minus_us: 20_000.0,
            stdp_window_us: 100_000,
            min_weight: 0.0,
            max_weight: 1.0,
            homeostasis: HomeostasisConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_negative_learning_rate() {
        let config = LearningConfig {
            a_minus: -0.1,
            ..LearningConfig::default()
        };

        assert!(matches!(
            config.validate(),
            Err(ConfigError::Negative {
                field: "learning.a_minus",
                ..
            })
        ));
    }

    #[test]
    fn rejects_zero_stdp_window() {
        let config = LearningConfig {
            stdp_window_us: 0,
            ..LearningConfig::default()
        };

        assert_eq!(
            config.validate(),
            Err(ConfigError::ZeroValue {
                field: "learning.stdp_window_us"
            })
        );
    }

    #[test]
    fn disabled_homeostasis_still_has_reproducible_valid_parameters() {
        let config = HomeostasisConfig {
            enabled: false,
            adjustment_rate: f32::NAN,
            ..HomeostasisConfig::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn validates_finite_threshold_range() {
        let config = HomeostasisConfig {
            min_threshold: f32::NEG_INFINITY,
            ..HomeostasisConfig::default()
        };

        assert!(matches!(
            config.validate(),
            Err(ConfigError::NonFinite { .. })
        ));
    }
}
