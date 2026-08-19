//! Strongly typed scalar values used by signals and local environments.

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
            pub fn is_finite(self) -> bool {
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
    "Magnitude carried by a neutral neural signal."
);
define_scalar!(Activity, "Locally measured neural activity.");
define_scalar!(Distance, "Spatial distance in the configured length scale.");
define_scalar!(
    EnergyCost,
    "Resource or maintenance cost in the configured energy scale."
);

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
}
