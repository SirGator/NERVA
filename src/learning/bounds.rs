//! Weight-bound handling shared by local plasticity rules.

use crate::primitives::Weight;

/// Validation failure for an inclusive synaptic weight interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightBoundsError {
    /// The lower bound is not finite.
    NonFiniteMinimum,
    /// The upper bound is not finite.
    NonFiniteMaximum,
    /// A bound is negative; weights are non-negative magnitudes.
    NegativeBound,
    /// The inclusive lower bound is greater than the upper bound.
    MinimumExceedsMaximum,
}

/// Inclusive bounds for a non-negative synaptic weight magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightBounds {
    min: Weight,
    max: Weight,
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
        if min < 0.0 || max < 0.0 {
            return Err(WeightBoundsError::NegativeBound);
        }
        if min > max {
            return Err(WeightBoundsError::MinimumExceedsMaximum);
        }

        Ok(Self {
            min: Weight::new(min).expect("bounds were validated above"),
            max: Weight::new(max).expect("bounds were validated above"),
        })
    }

    /// Returns the inclusive lower bound.
    pub const fn min(self) -> Weight {
        self.min
    }

    /// Returns the inclusive upper bound.
    pub const fn max(self) -> Weight {
        self.max
    }

    /// Restricts a finite weight magnitude to the configured interval.
    pub fn clamp(self, weight: Weight) -> Weight {
        weight.clamp_to(self.min, self.max)
    }

    /// Applies a finite signed delta and clamps the result without allowing a
    /// sign flip.
    ///
    /// NERVA-M0 represents excitation or inhibition through the presynaptic
    /// neuron's polarity. A synaptic weight is therefore always a non-negative
    /// magnitude. The delta path saturates on both sides, so finite operands
    /// cannot overflow the magnitude representation.
    pub fn apply_delta(self, weight: Weight, delta: f32) -> Weight {
        debug_assert!(delta.is_finite(), "weight delta must be finite");
        self.clamp(weight.saturating_add_delta(delta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(min: f32, max: f32) -> Result<WeightBounds, WeightBoundsError> {
        WeightBounds::new(min, max)
    }

    #[test]
    fn rejects_invalid_magnitude_bounds() {
        assert_eq!(bounds(-0.1, 1.0), Err(WeightBoundsError::NegativeBound));
        assert_eq!(
            bounds(2.0, 1.0),
            Err(WeightBoundsError::MinimumExceedsMaximum)
        );
        assert!(matches!(
            bounds(f32::NAN, 1.0),
            Err(WeightBoundsError::NonFiniteMinimum)
        ));
        assert!(matches!(
            bounds(0.0, f32::INFINITY),
            Err(WeightBoundsError::NonFiniteMaximum)
        ));
    }

    #[test]
    fn clamps_potentiation_and_depression() {
        let bounds = bounds(0.2, 0.8).expect("valid bounds");
        let base = Weight::new(0.7).unwrap();

        assert_eq!(bounds.apply_delta(base, 0.3), Weight::new(0.8).unwrap());
        assert_eq!(
            bounds.apply_delta(Weight::new(0.3).unwrap(), -0.4),
            Weight::new(0.2).unwrap()
        );
    }

    #[test]
    fn extreme_deltas_saturate_instead_of_overflowing() {
        let bounds = bounds(0.0, f32::MAX).expect("valid bounds");
        let max = Weight::MAX;

        assert_eq!(bounds.apply_delta(max, f32::MAX), Weight::MAX);
        assert_eq!(
            bounds.apply_delta(Weight::ZERO, f64::from(f32::MIN) as f32),
            Weight::ZERO
        );
    }
}
