//! Weight-bound handling shared by local plasticity rules.

/// Validation failure for an inclusive synaptic weight interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightBoundsError {
    /// The lower bound is not finite.
    NonFiniteMinimum,
    /// The upper bound is not finite.
    NonFiniteMaximum,
    /// M0 stores weights as non-negative magnitudes.
    NegativeMinimum,
    /// The inclusive lower bound is greater than the upper bound.
    MinimumExceedsMaximum,
}

/// Inclusive bounds for a non-negative synaptic weight magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightBounds {
    min: f32,
    max: f32,
}

impl WeightBounds {
    /// Creates validated bounds.
    pub fn new(min: f32, max: f32) -> Result<Self, WeightBoundsError> {
        if !min.is_finite() {
            return Err(WeightBoundsError::NonFiniteMinimum);
        }
        if !max.is_finite() {
            return Err(WeightBoundsError::NonFiniteMaximum);
        }
        if min < 0.0 {
            return Err(WeightBoundsError::NegativeMinimum);
        }
        if min > max {
            return Err(WeightBoundsError::MinimumExceedsMaximum);
        }

        Ok(Self { min, max })
    }

    /// Returns the inclusive lower bound.
    pub fn min(self) -> f32 {
        self.min
    }

    /// Returns the inclusive upper bound.
    pub fn max(self) -> f32 {
        self.max
    }

    /// Restricts a finite weight magnitude to the configured interval.
    pub fn clamp(self, weight: f32) -> f32 {
        weight.clamp(self.min, self.max)
    }

    /// Applies a finite delta and clamps the result without allowing a sign flip.
    ///
    /// NERVA-M0 represents excitation or inhibition through the presynaptic
    /// neuron's polarity. A synaptic weight is therefore always a non-negative
    /// magnitude.
    pub fn apply_delta(self, weight: f32, delta: f32) -> f32 {
        debug_assert!(weight.is_finite(), "weight must be finite");
        debug_assert!(delta.is_finite(), "weight delta must be finite");
        self.clamp(weight + delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_magnitude_bounds() {
        assert_eq!(
            WeightBounds::new(-0.1, 1.0),
            Err(WeightBoundsError::NegativeMinimum)
        );
        assert_eq!(
            WeightBounds::new(2.0, 1.0),
            Err(WeightBoundsError::MinimumExceedsMaximum)
        );
    }

    #[test]
    fn clamps_potentiation_and_depression() {
        let bounds = WeightBounds::new(0.2, 0.8).expect("valid bounds");

        assert_eq!(bounds.apply_delta(0.7, 0.3), 0.8);
        assert_eq!(bounds.apply_delta(0.3, -0.4), 0.2);
    }
}
