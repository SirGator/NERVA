//! Strongly typed synaptic weight values.

/// Synaptic connection strength before signal-type semantics are applied.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Weight(pub f32);

impl Weight {
    /// Creates a weight value.
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

impl From<f32> for Weight {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<Weight> for f32 {
    fn from(value: Weight) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_round_trips_without_adding_signal_semantics() {
        let weight = Weight::new(0.75);

        assert_eq!(f32::from(weight), 0.75);
        assert!(weight.is_finite());
    }
}
