//! Synapse-local release dynamics; no clock is scheduled by this state.
use super::SimTime;
use std::{error::Error, fmt};

/// Bounded Tsodyks-style resource depression and facilitation.
/// This is an abstract release model, not a fitted biological parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShortTermConfig {
    /// Baseline available release utilization, in `(0, 1]`.
    pub utilization: f32,
    /// Fraction of remaining utilization recruited by each spike, in `[0, 1]`.
    pub facilitation_increment: f32,
    /// Recovery time constant in microseconds.
    pub recovery_tau_us: f32,
    /// Facilitation relaxation time constant in microseconds.
    pub facilitation_tau_us: f32,
}

impl ShortTermConfig {
    /// Rejects nonfinite values, invalid fractions, and nonpositive time constants.
    pub fn validate(self) -> Result<(), ShortTermError> {
        if !self.utilization.is_finite()
            || self.utilization <= 0.
            || self.utilization > 1.
            || !self.facilitation_increment.is_finite()
            || !(0. ..=1.).contains(&self.facilitation_increment)
            || !self.recovery_tau_us.is_finite()
            || self.recovery_tau_us <= 0.
            || !self.facilitation_tau_us.is_finite()
            || self.facilitation_tau_us <= 0.
        {
            return Err(ShortTermError::InvalidConfig);
        }
        Ok(())
    }
}

/// Release state is owned by its individual synapse and copied with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShortTermState {
    /// Resource fraction remaining after the last release.
    pub resources: f32,
    /// Facilitated utilization after the last release.
    pub utilization: f32,
    /// Last presynaptic emission, or none for a fresh connection.
    pub updated_at: Option<SimTime>,
}

impl ShortTermState {
    /// Starts with fully recovered resources and baseline utilization.
    pub fn new(config: ShortTermConfig) -> Result<Self, ShortTermError> {
        config.validate()?;
        Ok(Self {
            resources: 1.,
            utilization: config.utilization,
            updated_at: None,
        })
    }

    /// Validates externally constructed state before any transition.
    pub fn validate(self) -> Result<(), ShortTermError> {
        if self.resources.is_finite()
            && (0. ..=1.).contains(&self.resources)
            && self.utilization.is_finite()
            && (0. ..=1.).contains(&self.utilization)
        {
            Ok(())
        } else {
            Err(ShortTermError::InvalidState)
        }
    }

    /// Analytically recovers between presynaptic emissions, then releases.
    /// Equal timestamps describe distinct releases in insertion order.
    pub fn release(&mut self, config: ShortTermConfig, at: SimTime) -> Result<f32, ShortTermError> {
        config.validate()?;
        self.validate()?;
        let elapsed = match self.updated_at {
            Some(previous) => at
                .duration_since(previous)
                .ok_or(ShortTermError::TimeWentBackwards)?,
            None => 0,
        };
        let resources = 1.
            - (1. - self.resources)
                * (-(elapsed as f64) / f64::from(config.recovery_tau_us)).exp() as f32;
        let relaxed = config.utilization
            + (self.utilization - config.utilization)
                * (-(elapsed as f64) / f64::from(config.facilitation_tau_us)).exp() as f32;
        let utilization = relaxed + config.facilitation_increment * (1. - relaxed);
        let released = (resources * utilization).clamp(0., 1.);
        self.resources = (resources - released).clamp(0., 1.);
        self.utilization = utilization.clamp(0., 1.);
        self.updated_at = Some(at);
        Ok(released)
    }
}

/// Invalid release parameters or reverse time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortTermError {
    /// Fractions/time constants do not satisfy the documented finite bounds.
    InvalidConfig,
    /// Resource/utilization fractions must be finite and in [0,1].
    InvalidState,
    /// A release preceded this connection's last release.
    TimeWentBackwards,
}
impl fmt::Display for ShortTermError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidConfig => "invalid short-term release configuration",
            Self::InvalidState => "invalid short-term release state",
            Self::TimeWentBackwards => "short-term release time went backwards",
        })
    }
}
impl Error for ShortTermError {}

#[cfg(test)]
mod tests {
    use super::*;
    fn facilitating() -> ShortTermConfig {
        ShortTermConfig {
            utilization: 0.05,
            facilitation_increment: 0.15,
            recovery_tau_us: 100_000.,
            facilitation_tau_us: 500_000.,
        }
    }
    #[test]
    fn facilitating_train_recovers_and_reverse_time_is_atomic() {
        let cfg = facilitating();
        let mut state = ShortTermState::new(cfg).unwrap();
        let first = state.release(cfg, SimTime(100)).unwrap();
        let second = state.release(cfg, SimTime(10_100)).unwrap();
        assert!(second > first, "facilitation must change actual release");
        let before = state;
        assert!(state.release(cfg, SimTime(1)).is_err());
        assert_eq!(state, before);
        let recovered = state.release(cfg, SimTime(20_000_000)).unwrap();
        assert!((first - recovered).abs() < 1e-6);
    }
    #[test]
    fn repeated_releases_are_bounded_and_resources_depress() {
        let cfg = ShortTermConfig {
            utilization: 0.8,
            facilitation_increment: 0.,
            ..facilitating()
        };
        let mut state = ShortTermState::new(cfg).unwrap();
        let first = state.release(cfg, SimTime(0)).unwrap();
        for _ in 0..100 {
            let value = state.release(cfg, SimTime(0)).unwrap();
            assert!((0. ..=first).contains(&value));
        }
        assert!(state.resources < 1e-6);
    }
    #[test]
    fn invalid_configuration_is_rejected() {
        for value in [f32::NAN, 0., -1., 1.1] {
            assert!(
                ShortTermConfig {
                    utilization: value,
                    ..facilitating()
                }
                .validate()
                .is_err()
            );
        }
        for value in [f32::NAN, 0., -1.] {
            assert!(
                ShortTermConfig {
                    recovery_tau_us: value,
                    ..facilitating()
                }
                .validate()
                .is_err()
            );
        }
    }
}
