//! Area models and modulation.

/// Area identity, membership, and state.
pub mod model;
/// Area-local neuromodulator levels.
pub mod modulation;

pub use model::BrainArea;
pub use modulation::ModulatorLevels;
