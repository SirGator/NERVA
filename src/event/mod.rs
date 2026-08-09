//! Events and event scheduling.

/// Deterministic scheduled-event storage.
pub mod queue;
/// Event payload and batching types.
pub mod types;

pub use queue::EventQueue;
pub use types::{EventPayload, NeuronInputBatch, ScheduledEvent, SpikeEvent, SynapticInputEvent};
