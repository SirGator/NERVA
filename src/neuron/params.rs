use super::error::NeuronConfigError;

/// Static parameters controlling a neuron's electrical behavior.
#[derive(Clone, Copy, Debug)]
pub struct NeuronParams {
    /// Baseline membrane potential in the selected potential scale.
    pub resting_potential: f32,
    /// Membrane potential restored after a spike.
    pub reset_potential: f32,
    /// Membrane potential at which the neuron emits a spike.
    pub threshold: f32,

    /// Time constant of passive membrane-potential decay in microseconds.
    pub membrane_tau_us: f32,
    /// Duration of the post-spike refractory period in microseconds.
    pub refractory_period_us: u64,

    /// Time constant of the activity trace in microseconds.
    pub activity_trace_tau_us: f32,

    /// Adaptation added after each spike.
    pub adaptation_increment: f32,
    /// Time constant of adaptation decay in microseconds.
    pub adaptation_tau_us: f32,

    /// Magnitude of external noise expected by the dynamics layer.
    pub noise_strength: f32,
}

impl NeuronParams {
    /// Verifies that all parameter values satisfy the neuron's invariants.
    pub fn validate(&self) -> Result<(), NeuronConfigError> {
        validate_finite(self.resting_potential, "resting_potential")?;
        validate_finite(self.reset_potential, "reset_potential")?;
        validate_finite(self.threshold, "threshold")?;
        validate_positive(self.membrane_tau_us, "membrane_tau_us")?;
        validate_positive(self.activity_trace_tau_us, "activity_trace_tau_us")?;
        validate_finite(self.adaptation_increment, "adaptation_increment")?;
        validate_non_negative(self.adaptation_increment, "adaptation_increment")?;
        validate_positive(self.adaptation_tau_us, "adaptation_tau_us")?;
        validate_finite(self.noise_strength, "noise_strength")?;
        validate_non_negative(self.noise_strength, "noise_strength")?;

        if self.threshold <= self.resting_potential {
            return Err(NeuronConfigError::ThresholdNotAboveRestingPotential);
        }

        if self.reset_potential >= self.threshold {
            return Err(NeuronConfigError::ResetNotBelowThreshold);
        }

        Ok(())
    }
}

/// Validates that a floating-point parameter is finite.
fn validate_finite(value: f32, name: &'static str) -> Result<(), NeuronConfigError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(NeuronConfigError::NonFiniteParameter(name))
    }
}

/// Validates that a floating-point parameter is finite and strictly positive.
fn validate_positive(value: f32, name: &'static str) -> Result<(), NeuronConfigError> {
    validate_finite(value, name)?;

    if value > 0.0 {
        Ok(())
    } else {
        Err(NeuronConfigError::NonPositiveParameter(name))
    }
}

/// Validates that a floating-point parameter is non-negative.
fn validate_non_negative(value: f32, name: &'static str) -> Result<(), NeuronConfigError> {
    if value >= 0.0 {
        Ok(())
    } else {
        Err(NeuronConfigError::NegativeParameter(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_params() -> NeuronParams {
        NeuronParams {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            membrane_tau_us: 20_000.0,
            refractory_period_us: 2_000,
            activity_trace_tau_us: 100_000.0,
            adaptation_increment: 0.1,
            adaptation_tau_us: 200_000.0,
            noise_strength: 0.0,
        }
    }

    #[test]
    fn rejects_a_threshold_at_resting_potential() {
        let mut params = valid_params();
        params.threshold = 0.0;

        assert_eq!(
            params.validate(),
            Err(NeuronConfigError::ThresholdNotAboveRestingPotential)
        );
    }
}
