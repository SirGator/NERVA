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

    /// Applies a finite signed delta and keeps the result inside the interval
    /// without a sign flip.
    ///
    /// NERVA-M0 represents excitation or inhibition through the presynaptic
    /// neuron's polarity. A synaptic weight is therefore always a non-negative
    /// magnitude. The delta path saturates on both sides, so finite operands
    /// cannot overflow the magnitude representation.
    ///
    /// The step is **soft**: instead of clipping every step beyond a bound
    /// onto the same value, the step is damped continuously towards the bound.
    /// Two different learning pressures therefore remain distinguishable, the
    /// response has no jump at the bound, and a bound is approached only
    /// asymptotically.
    pub fn apply_delta(self, weight: Weight, delta: f32) -> Weight {
        debug_assert!(delta.is_finite(), "weight delta must be finite");
        self.soft_update(weight, delta)
    }

    /// Applies a finite signed delta with a soft landing at the bounds.
    ///
    /// The realised change is `room · (1 - e^{-|δ|/room})`, where `room` is the
    /// remaining distance to the bound in the direction of the step. Without
    /// remaining distance the weight rests at the bound.
    pub fn soft_update(self, weight: Weight, delta: f32) -> Weight {
        let current = weight.get();
        let next = if delta >= 0.0 {
            let room = self.max.get() - current;
            if room <= 0.0 {
                self.max.get()
            } else {
                current + soft_step(room, delta)
            }
        } else {
            let room = current - self.min.get();
            if room <= 0.0 {
                self.min.get()
            } else {
                current - soft_step(room, -delta)
            }
        };
        // The rounding of the soft step may leave the result a fraction below
        // the bound. The bounds stay exact.
        Weight::from_valid(next).clamp_to(self.min, self.max)
    }
}

/// Damps a step of `magnitude` towards the remaining `room`.
///
/// The result lies in `[0, room)` and grows strictly with the magnitude.
fn soft_step(room: f32, magnitude: f32) -> f32 {
    room * -(-(magnitude / room)).exp_m1()
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
    fn soft_landing_keeps_the_weight_inside_the_bounds() {
        let bounds = bounds(0.2, 0.8).expect("valid bounds");
        let base = Weight::new(0.7).unwrap();

        let up = bounds.apply_delta(base, 0.3);
        assert!(up.get() > base.get());
        assert!(up.get() < 0.8);

        let down = bounds.apply_delta(Weight::new(0.3).unwrap(), -0.4);
        assert!(down.get() < 0.3);
        assert!(down.get() > 0.2);
    }

    #[test]
    fn soft_landing_distinguishes_learning_pressures() {
        // The former hard clamp mapped every step beyond the bound onto the
        // same value. The soft landing keeps the pressures apart.
        let bounds = bounds(0.0, 1.0).expect("valid bounds");
        let base = Weight::new(0.5).unwrap();

        let small = bounds.apply_delta(base, 0.2);
        let large = bounds.apply_delta(base, 2.0);
        assert!(small.get() < large.get());
        assert!(large.get() < 1.0);
    }

    #[test]
    fn soft_landing_is_monotone_and_leaves_frozen_weights_alone() {
        let bounds = bounds(0.0, 1.0).expect("valid bounds");
        let base = Weight::new(0.5).unwrap();

        assert_eq!(bounds.apply_delta(base, 0.0), base);

        let mut previous = base.get();
        for step in [0.01_f32, 0.05, 0.1, 0.25, 0.5] {
            let next = bounds.apply_delta(base, step).get();
            assert!(next > previous);
            previous = next;
        }
    }

    #[test]
    fn extreme_deltas_saturate_instead_of_overflowing() {
        let bounds = bounds(0.0, f32::MAX).expect("valid bounds");
        let max = Weight::MAX;

        let up = bounds.apply_delta(max, f32::MAX);
        assert!(up.get().is_finite());
        assert!(up.get() <= f32::MAX);

        let down = bounds.apply_delta(Weight::ZERO, f64::from(f32::MIN) as f32);
        assert!(down.get().is_finite());
        assert!(down.get() >= 0.0);
    }
}
