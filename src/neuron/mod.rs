//! Neuron models, state, and dynamics.

/// Continuous-time evolution of one neuron.
pub mod dynamics;
/// Configuration and runtime errors.
pub mod error;
/// Neuron construction and read-only accessors.
pub mod model;
/// Neuromodulator response types.
pub mod modulation;
/// Fixed neuron parameters and validation.
pub mod params;
/// Mutable simulation state.
pub mod state;
/// Roles, locations, cell types, and transmitters.
pub mod types;

pub use error::{NeuronConfigError, NeuronRuntimeError};
pub use model::{Neuron, NeuronSpec};
pub use modulation::{ModulatorResponse, NeuronModulatorSensitivity};
pub use params::NeuronParams;
pub use state::NeuronState;
pub use types::{CellType, NeuronAddress, NeuronRole, PrimaryTransmitter};
