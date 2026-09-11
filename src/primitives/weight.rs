//! Strongly typed non-negative synaptic weight magnitudes.

use std::ops::{Add, Sub};

/// Validation failure for a weight magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightError {
    /// The value is NaN or infinite.
    NonFinite(f32),
    /// Weights are non-negative magnitudes; the sign comes from polarity.
    Negative(f32),
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite(value) => {
                write!(formatter, "weight magnitude must be finite, got {value}")
            }
            Self::Negative(value) => write!(
                formatter,
                "weight magnitude must be non-negative, got {value}"
            ),
        }
    }
}

impl std::error::Error for WeightError {}

/// Non-negative synaptic connection strength.
///
/// The inner value is private and every constructor validates, so a `Weight`
/// is always finite and `>= 0`. The excitatory or inhibitory sign of a
/// contribution is never part of a weight; it is derived from the
/// presynaptic neuron's polarity when a [`crate::primitives::SignalStrength`]
/// amplitude is formed.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Weight(f32);

impl Weight {
    /// The smallest representable magnitude.
    pub const ZERO: Self = Self(0.0);

    /// The largest representable magnitude.
    pub const MAX: Self = Self(f32::MAX);

    /// Creates a validated non-negative, finite magnitude.
    pub const fn new(value: f32) -> Result<Self, WeightError> {
        // `const fn` requires const-callable checks; NaN and infinity are
        // rejected, as is any negative value.
        if value.is_nan() || value.is_infinite() {
            Err(WeightError::NonFinite(value))
        } else if value < 0.0 {
            Err(WeightError::Negative(value))
        } else {
            Ok(Self(value))
        }
    }

    /// Creates a magnitude from a value already known to be a valid,
    /// non-negative, finite magnitude.
    ///
    /// This is a trusted constructor for internal arithmetic results that are
    /// saturated or clamped into range by construction. Public API surfaces
    /// must use [`Self::new`].
    pub(crate) const fn from_valid(value: f32) -> Self {
        Self(value)
    }

    /// Returns the validated scalar representation.
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Whether the scalar can safely participate in simulation arithmetic.
    pub const fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Restricts a candidate magnitude to inclusive bounds.
    ///
    /// Negative candidates clamp to `min`, mirroring how a local learning rule
    /// treats a proposed step below the configured lower bound.
    pub fn clamp_candidate(self, candidate: f32, min: Self, max: Self) -> Self {
        Self::from_valid(candidate.clamp(min.0, max.0))
    }

    /// Restricts the magnitude to inclusive bounds.
    pub fn clamp_to(self, min: Self, max: Self) -> Self {
        Self::from_valid(self.0.clamp(min.0, max.0))
    }

    /// Saturating sum: finite operands cannot overflow in either direction.
    ///
    /// The `f64` intermediate makes `f32::MAX + f32::MAX` saturate to
    /// `Self::MAX` instead of producing infinity, and any negative
    /// intermediate result saturates to zero because a weight is a magnitude.
    pub fn saturating_add(self, other: Self) -> Self {
        let sum = f64::from(self.0) + f64::from(other.0);
        Self::from_valid((sum.max(0.0).min(f64::from(f32::MAX)) as f32).max(0.0))
    }

    /// Saturating difference: finite operands cannot overflow in either
    /// direction and the result cannot become negative.
    pub fn saturating_sub(self, other: Self) -> Self {
        let difference = f64::from(self.0) - f64::from(other.0);
        Self::from_valid((difference.max(0.0).min(f64::from(f32::MAX)) as f32).max(0.0))
    }

    /// Applies a finite signed delta and saturates on both sides.
    ///
    /// Learning rules express an update as a magnitude plus a signed delta;
    /// the result remains a valid magnitude without any sign flip.
    pub fn saturating_add_delta(self, delta: f32) -> Self {
        let sum = f64::from(self.0) + f64::from(delta);
        Self::from_valid((sum.max(0.0).min(f64::from(f32::MAX)) as f32).max(0.0))
    }
}

impl From<Weight> for f32 {
    fn from(value: Weight) -> Self {
        value.0
    }
}

impl Add for Weight {
    type Output = Weight;

    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl Sub for Weight {
    type Output = Weight;

    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_rejects_invalid_magnitudes() {
        assert_eq!(Weight::new(-1.0), Err(WeightError::Negative(-1.0)));
        assert!(matches!(
            Weight::new(f32::NAN),
            Err(WeightError::NonFinite(_))
        ));
        assert!(matches!(
            Weight::new(f32::INFINITY),
            Err(WeightError::NonFinite(_))
        ));
        assert!(matches!(
            Weight::new(f32::NEG_INFINITY),
            Err(WeightError::NonFinite(_))
        ));
        assert_eq!(Weight::new(0.0).map(Weight::get), Ok(0.0));
        assert_eq!(Weight::new(f32::MAX).map(Weight::get), Ok(f32::MAX));
    }

    #[test]
    fn saturating_arithmetic_never_produces_non_finite_or_negative_values() {
        let max = Weight::MAX;
        let zero = Weight::ZERO;

        assert_eq!(max.saturating_add(max), Weight::MAX);
        assert_eq!(zero.saturating_sub(max), Weight::ZERO);
        assert_eq!(
            zero.saturating_add_delta(f64::from(f32::MIN) as f32),
            Weight::ZERO
        );
        assert_eq!(
            max.saturating_add_delta(f64::from(f32::MIN) as f32),
            Weight::ZERO
        );
        assert_eq!(max.saturating_add_delta(f32::MAX), Weight::MAX);
        assert_eq!(
            Weight::new(0.25).map(|weight| weight + Weight::new(0.5).unwrap()),
            Weight::new(0.75)
        );
    }
}
