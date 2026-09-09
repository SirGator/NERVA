//! Strongly typed electrical values.

/// Membrane potential in the simulation's configured potential scale.
///
/// Domain wrappers prevent accidentally passing a threshold where a membrane
/// potential is required:
///
/// ```compile_fail
/// use nerva::primitives::{Potential, Threshold};
///
/// fn accepts_potential(_: Potential) {}
///
/// accepts_potential(Threshold(-50.0));
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Potential(pub f32);

impl Potential {
    /// Creates a membrane-potential value.
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

impl From<f32> for Potential {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<Potential> for f32 {
    fn from(value: Potential) -> Self {
        value.0
    }
}

/// Firing threshold in the simulation's configured potential scale.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Threshold(pub f32);

impl Threshold {
    /// Creates a firing-threshold value.
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

impl From<f32> for Threshold {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<Threshold> for f32 {
    fn from(value: Threshold) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn potential_and_threshold_remain_distinct_values() {
        let potential = Potential::new(-65.0);
        let threshold = Threshold::new(-50.0);

        assert_eq!(potential.get(), -65.0);
        assert_eq!(threshold.get(), -50.0);
        assert!(potential.is_finite());
        assert!(threshold.is_finite());
    }
}
