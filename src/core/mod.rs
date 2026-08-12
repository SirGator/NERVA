//! State model for neurons, synapses, spikes, and the sparse directed graph.
//!
//! The core has no scheduler and knows no concrete learning rule. It exposes
//! validated state transitions that the runtime may compose deterministically.

/// Timestamped input events and simulation time.
pub mod event;
/// Stable identity wrapper types.
pub mod ids;
/// Sparse directed graph and synchronized adjacency indices.
pub mod network;
/// Leaky integrate-and-fire neuron state.
pub mod neuron;
/// Emitted spike records.
pub mod spike;
/// Directed synaptic connection state.
pub mod synapse;

pub use event::{Event, EventKind, SimTime, SimTimeError};
pub use ids::{NeuronId, SynapseId};
pub use network::{Network, NetworkError};
pub use neuron::{Neuron, NeuronError, NeuronRole, Polarity};
pub use spike::Spike;
pub use synapse::{Synapse, SynapseError};
