//! Pair-STDP and local homeostasis configuration.

use super::{ConfigError, non_negative, ordered_range, positive};

/// Parameters for local cellular and structural homeostasis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HomeostasisConfig {
    /// Whether local maintenance events are active.
    pub enabled: bool,
    /// Time between two maintenance events of the same neuron, in microseconds.
    ///
    /// Each neuron owns and reschedules its own deadline. This is a local slow
    /// clock, not a population-wide simulation tick.
    pub update_interval_us: u64,
    /// Per-neuron target firing rate in hertz.
    pub target_rate_hz: f32,
    /// Minimum locally observed input magnitude per second that counts as
    /// adequate drive for intrinsic excitability regulation.
    pub target_input_rate: f32,
    /// Intrinsic-current change per second and per hertz of firing-rate error.
    ///
    /// The current itself is measured in potential units per second. Applying
    /// this rate with elapsed simulation time keeps regulation independent of
    /// how often unrelated events happen to touch a neuron.
    pub intrinsic_adjustment_rate: f32,
    /// Structural-drive change per second and per missing input-rate unit.
    ///
    /// Structural drive is a local request signal for a future growth/pruning
    /// slice; it does not mutate topology in the M0 core.
    pub structural_adjustment_rate: f32,
    /// Inclusive lower intrinsic-current bound.
    pub min_intrinsic_current: f32,
    /// Inclusive upper intrinsic-current bound.
    pub max_intrinsic_current: f32,
    /// Inclusive lower structural-drive bound.
    pub min_structural_drive: f32,
    /// Inclusive upper structural-drive bound.
    pub max_structural_drive: f32,
}

impl HomeostasisConfig {
    /// Validates local-only homeostasis parameters.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.update_interval_us == 0 {
            return Err(ConfigError::ZeroValue {
                field: "homeostasis.update_interval_us",
            });
        }
        non_negative(self.target_rate_hz, "homeostasis.target_rate_hz")?;
        non_negative(self.target_input_rate, "homeostasis.target_input_rate")?;
        non_negative(
            self.intrinsic_adjustment_rate,
            "homeostasis.intrinsic_adjustment_rate",
        )?;
        non_negative(
            self.structural_adjustment_rate,
            "homeostasis.structural_adjustment_rate",
        )?;
        ordered_range(
            self.min_intrinsic_current,
            "homeostasis.min_intrinsic_current",
            self.max_intrinsic_current,
            "homeostasis.max_intrinsic_current",
        )?;
        ordered_range(
            self.min_structural_drive,
            "homeostasis.min_structural_drive",
            self.max_structural_drive,
            "homeostasis.max_structural_drive",
        )?;

        Ok(())
    }
}

impl Default for HomeostasisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            update_interval_us: 100_000,
            target_rate_hz: 5.0,
            target_input_rate: 1.0,
            intrinsic_adjustment_rate: 0.001,
            structural_adjustment_rate: 0.01,
            min_intrinsic_current: -1_000.0,
            max_intrinsic_current: 1_000.0,
            min_structural_drive: 0.0,
            max_structural_drive: 1_000.0,
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
    /// Optional local cellular and structural homeostasis parameters.
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
            intrinsic_adjustment_rate: f32::NAN,
            ..HomeostasisConfig::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn validates_finite_intrinsic_current_range() {
        let config = HomeostasisConfig {
            min_intrinsic_current: f32::NEG_INFINITY,
            ..HomeostasisConfig::default()
        };

        assert!(matches!(
            config.validate(),
            Err(ConfigError::NonFinite { .. })
        ));
    }

    #[test]
    fn rejects_a_zero_local_maintenance_interval() {
        let config = HomeostasisConfig {
            update_interval_us: 0,
            ..HomeostasisConfig::default()
        };

        assert_eq!(
            config.validate(),
            Err(ConfigError::ZeroValue {
                field: "homeostasis.update_interval_us"
            })
        );
    }
}
