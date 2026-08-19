//! Three-dimensional positions are geometry, never neuron identity.

use std::{error::Error, fmt};

pub use crate::primitives::{Position3D, PositionError};

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
