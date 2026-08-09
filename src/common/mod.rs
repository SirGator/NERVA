//! Common identifiers and time primitives.

/// Identifier wrapper types used across the model.
pub mod id;
/// Monotonic simulation time representation.
pub mod time;

pub use id::{AreaId, LayerId, NeuronId, PopulationId, SynapseId};
pub use time::SimTimeUs;
