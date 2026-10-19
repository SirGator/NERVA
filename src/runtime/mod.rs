//! Deterministic event orchestration with a separate numerical compute backend.

/// Complete equal-timestamp event batches.
pub mod event_batch;
/// Runtime-only recurring-maintenance booking registry.
mod homeostasis_booking;
/// Runtime-only autonomous-spike booking registry.
mod intrinsic_booking;
/// Opt-in distributions and host wall-clock measurements.
pub mod profile;
/// Delayed and distance-attenuated spike propagation.
pub mod propagation;
/// Technical `(timestamp, insertion_sequence)` scheduler.
pub mod scheduler;
/// Network execution, learning hooks, observations, and replay log.
pub mod simulation;

pub use event_batch::{EventBatch, ScheduledEvent};
pub use profile::{HISTOGRAM_LABELS, Histogram, PhaseProfile, ProfilePhase, RuntimeProfile};
pub use propagation::{PlannedTransmission, PropagationError, plan_spike_propagation};
pub use scheduler::{EventScheduler, SchedulerError};
pub use simulation::{
    BatchReport, EventLog, ObservationEvent, RunReport, Simulation, SimulationError,
};
