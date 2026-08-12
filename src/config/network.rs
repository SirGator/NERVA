//! Static network-generation and propagation configuration.

use super::{ConfigError, finite, non_negative, ordered_range, positive};

/// Parameters from which the initial sparse directed graph is generated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetworkConfig {
    /// Number of excitatory neurons.
    pub excitatory_neurons: usize,
    /// Number of inhibitory neurons.
    pub inhibitory_neurons: usize,
    /// Probability of creating each eligible directed connection.
    pub connection_probability: f32,
    /// Seed controlling every stochastic construction decision.
    pub seed: u64,
    /// Smallest permitted non-negative synaptic weight magnitude.
    pub min_weight: f32,
    /// Largest permitted non-negative synaptic weight magnitude.
    pub max_weight: f32,
    /// Axonal conduction velocity in position units per microsecond.
    pub conduction_velocity: f32,
    /// Spatial attenuation length `lambda`, in position units.
    pub distance_decay_length: f32,
}

impl NetworkConfig {
    /// Validates topology, weight, and propagation invariants.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.excitatory_neurons == 0 && self.inhibitory_neurons == 0 {
            return Err(ConfigError::EmptyNetwork);
        }
        self.excitatory_neurons
            .checked_add(self.inhibitory_neurons)
            .ok_or(ConfigError::PopulationSizeOverflow)?;

        finite(
            self.connection_probability,
            "network.connection_probability",
        )?;
        if !(0.0..=1.0).contains(&self.connection_probability) {
            return Err(ConfigError::ProbabilityOutOfRange {
                field: "network.connection_probability",
                value: self.connection_probability,
            });
        }

        non_negative(self.min_weight, "network.min_weight")?;
        non_negative(self.max_weight, "network.max_weight")?;
        ordered_range(
            self.min_weight,
            "network.min_weight",
            self.max_weight,
            "network.max_weight",
        )?;
        positive(self.conduction_velocity, "network.conduction_velocity")?;
        positive(self.distance_decay_length, "network.distance_decay_length")?;

        Ok(())
    }

    /// Checked total population size.
    pub fn neuron_count(&self) -> Result<usize, ConfigError> {
        self.excitatory_neurons
            .checked_add(self.inhibitory_neurons)
            .ok_or(ConfigError::PopulationSizeOverflow)
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            excitatory_neurons: 80,
            inhibitory_neurons: 20,
            connection_probability: 0.1,
            seed: 0,
            min_weight: 0.0,
            max_weight: 1.0,
            conduction_velocity: 1.0,
            distance_decay_length: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_network() {
        let config = NetworkConfig {
            excitatory_neurons: 0,
            inhibitory_neurons: 0,
            ..NetworkConfig::default()
        };

        assert_eq!(config.validate(), Err(ConfigError::EmptyNetwork));
    }

    #[test]
    fn rejects_invalid_probability_even_when_nan() {
        for probability in [-0.1, 1.1, f32::NAN] {
            let config = NetworkConfig {
                connection_probability: probability,
                ..NetworkConfig::default()
            };
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn rejects_zero_conduction_velocity() {
        let config = NetworkConfig {
            conduction_velocity: 0.0,
            ..NetworkConfig::default()
        };

        assert!(matches!(
            config.validate(),
            Err(ConfigError::NonPositive {
                field: "network.conduction_velocity",
                ..
            })
        ));
    }

    #[test]
    fn rejects_reversed_weight_bounds() {
        let config = NetworkConfig {
            min_weight: 2.0,
            max_weight: 1.0,
            ..NetworkConfig::default()
        };

        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidRange { .. })
        ));
    }

    #[test]
    fn rejects_population_size_overflow() {
        let config = NetworkConfig {
            excitatory_neurons: usize::MAX,
            inhibitory_neurons: 1,
            ..NetworkConfig::default()
        };

        assert_eq!(config.validate(), Err(ConfigError::PopulationSizeOverflow));
        assert_eq!(
            config.neuron_count(),
            Err(ConfigError::PopulationSizeOverflow)
        );
    }
}
