//! Three-dimensional geometry without network identity semantics.

use std::{error::Error, fmt};

use super::Distance;

/// Position in the simulation's configured three-dimensional coordinate space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Position3 {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Z coordinate.
    pub z: f32,
}

impl Position3 {
    /// Coordinate-space origin.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Compatibility name used by the event-driven M0 model.
    pub const ORIGIN: Self = Self::ZERO;

    /// Creates a position. Use [`Self::validate`] for untrusted coordinates.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Creates a position only when every coordinate is finite.
    pub fn try_new(x: f32, y: f32, z: f32) -> Result<Self, PositionError> {
        let position = Self::new(x, y, z);
        position.validate()?;
        Ok(position)
    }

    /// Verifies that Euclidean operations on this position are well-defined.
    pub fn validate(self) -> Result<(), PositionError> {
        for (coordinate, value) in [("x", self.x), ("y", self.y), ("z", self.z)] {
            if !value.is_finite() {
                return Err(PositionError::NonFiniteCoordinate { coordinate, value });
            }
        }
        Ok(())
    }

    /// Whether all coordinates can safely participate in geometry operations.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Squared Euclidean distance, useful for relative comparisons.
    pub fn squared_distance_to(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx.mul_add(dx, dy.mul_add(dy, dz * dz))
    }

    /// Euclidean distance to another position.
    pub fn distance_to(self, other: Self) -> f32 {
        self.squared_distance_to(other).sqrt()
    }

    /// Euclidean distance wrapped in the strongly typed distance domain.
    pub fn typed_distance_to(self, other: Self) -> Distance {
        Distance(self.distance_to(other))
    }

    /// Alias for [`Self::distance_to`] convenient in functional pipelines.
    pub fn distance(self, other: Self) -> f32 {
        self.distance_to(other)
    }
}

/// Compatibility name retained for the existing M0 public API.
pub use Position3 as Position3D;

/// Invalid geometric input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionError {
    /// One Cartesian coordinate is NaN or infinite.
    NonFiniteCoordinate {
        /// Coordinate name (`x`, `y`, or `z`).
        coordinate: &'static str,
        /// Rejected value.
        value: f32,
    },
}

impl fmt::Display for PositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteCoordinate { coordinate, value } => {
                write!(
                    formatter,
                    "position coordinate {coordinate} must be finite, got {value}"
                )
            }
        }
    }
}

impl Error for PositionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_euclidean_distance_without_assigning_identity() {
        let origin = Position3::ZERO;
        let point = Position3::new(2.0, 3.0, 6.0);

        assert_eq!(origin.distance_to(point), 7.0);
        assert_eq!(origin.typed_distance_to(point), Distance(7.0));
        assert!(point.is_finite());
    }

    #[test]
    fn rejects_non_finite_coordinates() {
        assert!(matches!(
            Position3::try_new(f32::NAN, 0.0, 0.0),
            Err(PositionError::NonFiniteCoordinate {
                coordinate: "x",
                ..
            })
        ));
    }
}
