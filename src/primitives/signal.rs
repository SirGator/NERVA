//! Strongly typed scalar values used by signals and local environments.

use std::ops::{Add, Neg};

macro_rules! define_scalar {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
        pub struct $name(pub f32);

        impl $name {
            /// Creates the typed scalar.
            pub const fn new(value: f32) -> Self {
                Self(value)
            }

            /// Returns the scalar representation.
            pub const fn get(self) -> f32 {
                self.0
            }

            /// Whether the scalar can safely participate in simulation arithmetic.
            pub const fn is_finite(self) -> bool {
                self.0.is_finite()
            }
        }

        impl From<f32> for $name {
            fn from(value: f32) -> Self {
                Self(value)
            }
        }

        impl From<$name> for f32 {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

define_scalar!(
    SignalStrength,
    "Signed magnitude carried by a neural signal. Finite; its sign comes from the presynaptic polarity."
);
define_scalar!(Activity, "Locally measured neural activity.");
define_scalar!(Distance, "Spatial distance in the configured length scale.");
define_scalar!(
    EnergyCost,
    "Resource or maintenance cost in the configured energy scale."
);

impl SignalStrength {
    /// Saturating sum: finite operands cannot overflow into infinity in
    /// either direction.
    pub fn saturating_add(self, other: Self) -> Self {
        let sum = f64::from(self.0) + f64::from(other.0);
        Self((sum.clamp(f64::from(f32::MIN), f64::from(f32::MAX))) as f32)
    }

    /// The absolute contribution magnitude, mirroring how the runtime derives
    /// the total input magnitude from a signed amplitude.
    pub fn magnitude(self) -> Self {
        Self(self.0.abs())
    }

    /// Whether this signal inhibits (is strictly negative).
    pub fn is_inhibitory(self) -> bool {
        self.0 < 0.0
    }
}

impl Add for SignalStrength {
    type Output = SignalStrength;

    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl Neg for SignalStrength {
    type Output = SignalStrength;

    fn neg(self) -> Self {
        Self(-self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_domains_preserve_their_values() {
        assert_eq!(SignalStrength::new(0.5).get(), 0.5);
        assert_eq!(Activity::new(0.25).get(), 0.25);
        assert_eq!(Distance::new(3.0).get(), 3.0);
        assert_eq!(EnergyCost::new(1.5).get(), 1.5);
    }

    #[test]
    fn signed_strength_saturates_on_both_sides() {
        assert_eq!(
            SignalStrength::new(f32::MAX).saturating_add(SignalStrength::new(f32::MAX)),
            SignalStrength::new(f32::MAX)
        );
        assert_eq!(
            SignalStrength::new(f32::MIN).saturating_add(SignalStrength::new(f32::MIN)),
            SignalStrength::new(f32::MIN)
        );
        assert_eq!(
            SignalStrength::new(-0.5).saturating_add(SignalStrength::new(0.25)),
            SignalStrength::new(-0.25)
        );
        assert_eq!(
            SignalStrength::new(-2.0).magnitude(),
            SignalStrength::new(2.0)
        );
        assert!(SignalStrength::new(-0.1).is_inhibitory());
        assert!(!SignalStrength::new(0.1).is_inhibitory());
    }
}
