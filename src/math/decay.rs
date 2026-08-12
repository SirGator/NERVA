//! Closed-form exponential decay between precisely timestamped events.

use std::{error::Error, fmt};

/// Computes `exp(-elapsed_us / tau_us)`.
///
/// Configurations validate `tau_us` before simulation. For callers handling
/// unchecked inputs, [`try_decay_factor`] provides an explicit error.
pub fn decay_factor(elapsed_us: u64, tau_us: f32) -> f32 {
    try_decay_factor(elapsed_us, tau_us).unwrap_or(f32::NAN)
}

/// Checked form of [`decay_factor`].
pub fn try_decay_factor(elapsed_us: u64, tau_us: f32) -> Result<f32, DecayError> {
    validate_positive(tau_us, "tau_us")?;
    Ok((-(elapsed_us as f32) / tau_us).exp())
}

/// Analytically decays `value` toward zero.
pub fn decay_to_zero(value: f32, elapsed_us: u64, tau_us: f32) -> f32 {
    value * decay_factor(elapsed_us, tau_us)
}

/// Analytically decays `value` toward `equilibrium`.
///
/// This is the LIF membrane equation
/// `equilibrium + (value - equilibrium) * exp(-dt/tau)`.
pub fn decay_towards(value: f32, equilibrium: f32, elapsed_us: u64, tau_us: f32) -> f32 {
    equilibrium + (value - equilibrium) * decay_factor(elapsed_us, tau_us)
}

/// Spatial attenuation `exp(-distance / decay_length)`.
pub fn distance_attenuation(distance: f32, decay_length: f32) -> f32 {
    try_distance_attenuation(distance, decay_length).unwrap_or(f32::NAN)
}

/// Checked form of [`distance_attenuation`].
pub fn try_distance_attenuation(distance: f32, decay_length: f32) -> Result<f32, DecayError> {
    validate_non_negative(distance, "distance")?;
    validate_positive(decay_length, "decay_length")?;
    Ok((-distance / decay_length).exp())
}

fn validate_positive(value: f32, parameter: &'static str) -> Result<(), DecayError> {
    if !value.is_finite() {
        return Err(DecayError::NonFinite { parameter, value });
    }
    if value <= 0.0 {
        return Err(DecayError::NonPositive { parameter, value });
    }
    Ok(())
}

fn validate_non_negative(value: f32, parameter: &'static str) -> Result<(), DecayError> {
    if !value.is_finite() {
        return Err(DecayError::NonFinite { parameter, value });
    }
    if value < 0.0 {
        return Err(DecayError::Negative { parameter, value });
    }
    Ok(())
}

/// Invalid input to an exponential decay function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecayError {
    /// A value is NaN or infinite.
    NonFinite {
        /// Parameter name.
        parameter: &'static str,
        /// Rejected value.
        value: f32,
    },
    /// A required time or length constant is not strictly positive.
    NonPositive {
        /// Parameter name.
        parameter: &'static str,
        /// Rejected value.
        value: f32,
    },
    /// A distance is negative.
    Negative {
        /// Parameter name.
        parameter: &'static str,
        /// Rejected value.
        value: f32,
    },
}

impl fmt::Display for DecayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { parameter, value } => {
                write!(formatter, "{parameter} must be finite, got {value}")
            }
            Self::NonPositive { parameter, value } => {
                write!(
                    formatter,
                    "{parameter} must be greater than zero, got {value}"
                )
            }
            Self::Negative { parameter, value } => {
                write!(formatter, "{parameter} must not be negative, got {value}")
            }
        }
    }
}

impl Error for DecayError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f32, right: f32) {
        assert!((left - right).abs() <= 1.0e-6, "{left} != {right}");
    }

    #[test]
    fn no_elapsed_time_means_no_decay() {
        assert_eq!(decay_factor(0, 10.0), 1.0);
        assert_eq!(decay_to_zero(7.0, 0, 10.0), 7.0);
    }

    #[test]
    fn one_time_constant_decays_by_e() {
        close(decay_factor(20_000, 20_000.0), (-1.0_f32).exp());
    }

    #[test]
    fn lif_decay_moves_toward_resting_potential() {
        let decayed = decay_towards(-50.0, -70.0, 20_000, 20_000.0);
        close(decayed, -70.0 + 20.0 / std::f32::consts::E);
    }

    #[test]
    fn attenuation_is_one_at_zero_distance() {
        assert_eq!(distance_attenuation(0.0, 5.0), 1.0);
    }

    #[test]
    fn checked_functions_reject_unphysical_inputs() {
        assert!(matches!(
            try_decay_factor(1, 0.0),
            Err(DecayError::NonPositive { .. })
        ));
        assert!(matches!(
            try_distance_attenuation(-1.0, 1.0),
            Err(DecayError::Negative { .. })
        ));
    }
}
