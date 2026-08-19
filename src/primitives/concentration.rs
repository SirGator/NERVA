//! Strongly typed modulator concentrations.

/// Local concentration of a modulatory signal.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Concentration(pub f32);

impl Concentration {
    /// Creates a concentration value.
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

impl From<f32> for Concentration {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<Concentration> for f32 {
    fn from(value: Concentration) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concentration_round_trips_as_a_distinct_domain() {
        let concentration = Concentration::new(0.2);

        assert_eq!(f32::from(concentration), 0.2);
        assert!(concentration.is_finite());
    }
}
