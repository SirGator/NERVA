//! Synapse models, state, transmission, and plasticity.

/// Synapse identity and connectivity model.
pub mod model;
/// Fixed synapse parameters.
pub mod params;
/// Local synaptic learning rules.
pub mod plasticity;
/// Mutable synapse state.
pub mod state;
/// Synaptic signal propagation.
pub mod transmission;

pub use model::{Synapse, SynapticEffect};
pub use params::SynapseParams;
pub use plasticity::Plasticity;
pub use state::SynapseState;
