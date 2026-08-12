//! Stateless geometry and analytical decay functions.

/// Analytical temporal and spatial exponential decay.
pub mod decay;
/// Three-dimensional geometry used by neurons and propagation.
pub mod position;

pub use decay::{
    DecayError, decay_factor, decay_to_zero, decay_towards, distance_attenuation, try_decay_factor,
    try_distance_attenuation,
};
pub use position::{ConductionDelayError, Position3D, PositionError, conduction_delay_us};
