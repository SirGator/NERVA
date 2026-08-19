//! Strongly typed, policy-free values shared by all architectural layers.
//!
//! Primitives carry identity, time, physical values, and geometry. They
//! deliberately contain no neuron, synapse, network, plasticity, event, or
//! runtime policy.
//!
//! The typed scalar domains are a prepared migration API. The active M0 data
//! path still uses some raw `f32` values and will migrate one complete path at
//! a time so partially typed state cannot split invariants across layers.

/// Scalar modulator concentrations.
pub mod concentration;
/// Stable identifiers for independently stored domain objects.
pub mod ids;
/// Three-dimensional spatial positions.
pub mod position;
/// Membrane-potential and firing-threshold values.
pub mod potential;
/// Signal, activity, distance, and cost values.
pub mod signal;
/// Exact microsecond simulation timestamps.
pub mod time;
/// Synaptic weight values.
pub mod weight;

pub use concentration::Concentration;
pub use ids::{ActuatorId, ModulatorId, NeuronId, RegionId, SensorId, SynapseId, SystemId};
pub use position::{Position3, Position3D, PositionError};
pub use potential::{Potential, Threshold};
pub use signal::{Activity, Distance, EnergyCost, SignalStrength};
pub use time::{SimTime, SimTimeError};
pub use weight::Weight;
