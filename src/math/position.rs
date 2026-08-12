//! Three-dimensional positions are geometry, never neuron identity.

use std::{error::Error, fmt};

/// Cartesian position in arbitrary but consistent spatial units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Position3D {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Z coordinate.
    pub z: f32,
}

impl Position3D {
    /// Coordinate origin.
    pub const ORIGIN: Self = Self::new(0.0, 0.0, 0.0);

    /// Creates a position. Use [`Self::validate`] when accepting untrusted
    /// coordinates; keeping this constructor infallible makes generated layouts
    /// concise.
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

    /// Whether all coordinates are finite.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Squared Euclidean distance, useful when only relative distance matters.
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

    /// Alias for [`Self::distance_to`] convenient in functional pipelines.
    pub fn distance(self, other: Self) -> f32 {
        self.distance_to(other)
    }
}

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

/// Converts geometric length and conduction velocity into an integer delay.
///
/// The result rounds up so an impulse never arrives earlier than the continuous
/// travel time. `minimum_delay_us` protects the runtime from undefined zero-delay
/// recurrence when two cells share a position.
pub fn conduction_delay_us(
    distance: f32,
    velocity_units_per_us: f32,
    minimum_delay_us: u64,
) -> Result<u64, ConductionDelayError> {
    if !distance.is_finite() || distance < 0.0 {
        return Err(ConductionDelayError::InvalidDistance(distance));
    }
    if !velocity_units_per_us.is_finite() || velocity_units_per_us <= 0.0 {
        return Err(ConductionDelayError::InvalidVelocity(velocity_units_per_us));
    }
    if minimum_delay_us == 0 {
        return Err(ConductionDelayError::ZeroMinimumDelay);
    }

    // Parameters are specified as `f32`; perform the quotient in that same
    // precision so a representable ratio such as `1.0 / 0.0001 == 10_000.0`
    // is not perturbed by first widening the already-rounded operands.
    let continuous = distance / velocity_units_per_us;
    if !continuous.is_finite() || f64::from(continuous) > u64::MAX as f64 {
        return Err(ConductionDelayError::UnrepresentableDelay);
    }
    Ok((continuous.ceil() as u64).max(minimum_delay_us))
}

/// Invalid geometry-to-delay conversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConductionDelayError {
    /// Distance must be finite and non-negative.
    InvalidDistance(f32),
    /// Velocity must be finite and strictly positive.
    InvalidVelocity(f32),
    /// Runtime causality requires at least one microsecond.
    ZeroMinimumDelay,
    /// The calculated delay does not fit in [`u64`].
    UnrepresentableDelay,
}

impl fmt::Display for ConductionDelayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDistance(value) => {
                write!(
                    formatter,
                    "distance must be finite and non-negative, got {value}"
                )
            }
            Self::InvalidVelocity(value) => {
                write!(
                    formatter,
                    "conduction velocity must be finite and positive, got {value}"
                )
            }
            Self::ZeroMinimumDelay => {
                formatter.write_str("minimum conduction delay must be at least one microsecond")
            }
            Self::UnrepresentableDelay => {
                formatter.write_str("conduction delay does not fit in simulation time")
            }
        }
    }
}

impl Error for ConductionDelayError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_three_dimensional_euclidean_distance() {
        let a = Position3D::new(1.0, 2.0, 3.0);
        let b = Position3D::new(4.0, 6.0, 3.0);

        assert_eq!(a.distance_to(b), 5.0);
        assert_eq!(a.distance_to(b), b.distance_to(a));
    }

    #[test]
    fn a_position_is_not_an_identity() {
        let first = Position3D::ORIGIN;
        let second = Position3D::ORIGIN;

        assert_eq!(first.distance_to(second), 0.0);
        // Equal geometry is intentionally allowed for distinct neurons. IDs are
        // the sole identity mechanism in `core`.
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_non_finite_coordinates() {
        assert!(matches!(
            Position3D::try_new(f32::NAN, 0.0, 0.0),
            Err(PositionError::NonFiniteCoordinate {
                coordinate: "x",
                ..
            })
        ));
    }

    #[test]
    fn conduction_delay_uses_geometry_velocity_and_ceil_rounding() {
        assert_eq!(conduction_delay_us(1.0, 0.0001, 1), Ok(10_000));
        assert_eq!(conduction_delay_us(0.25, 0.1, 1), Ok(3));
        assert_eq!(conduction_delay_us(0.0, 1.0, 1), Ok(1));
    }

    #[test]
    fn conduction_delay_rejects_unphysical_inputs() {
        assert!(matches!(
            conduction_delay_us(-1.0, 1.0, 1),
            Err(ConductionDelayError::InvalidDistance(_))
        ));
        assert!(matches!(
            conduction_delay_us(1.0, 0.0, 1),
            Err(ConductionDelayError::InvalidVelocity(_))
        ));
        assert_eq!(
            conduction_delay_us(1.0, 1.0, 0),
            Err(ConductionDelayError::ZeroMinimumDelay)
        );
    }
}
