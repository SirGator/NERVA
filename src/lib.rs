//! NERVA — Neural Emergent Reactive & Versatile Architectur.
//!
//! An embeddable Rust library for deterministic, event-driven spatial spiking
//! networks. The host application owns the network setup, input, execution
//! horizon, and interpretation of observations.
//!
//! The core only represents local neuronal and synaptic state. Simulation,
//! plasticity, external connections, experiments and diagnostics live in
//! separate slices so observing an experiment can never change its outcome.
//!
//! # Getting started
//!
//! Construct a [`core::Network`], choose a [`learning::PlasticityRule`], and
//! execute timestamped inputs with [`runtime::Simulation`]:
//!
//! ```
//! use nerva::{
//!     config::{NeuronConfig, RuntimeConfig},
//!     core::{Network, Neuron, NeuronId, Polarity, SimTime},
//!     learning::NoPlasticity,
//!     math::Position3D,
//!     runtime::Simulation,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut network = Network::new();
//! network.add_neuron(Neuron::new(
//!     NeuronId(1),
//!     Position3D::ORIGIN,
//!     Polarity::Excitatory,
//!     None,
//!     NeuronConfig::default(),
//!     SimTime::ZERO,
//! )?)?;
//!
//! let mut simulation = Simulation::new(
//!     network, NoPlasticity, RuntimeConfig::default(), 10.0,
//! )?;
//! simulation.schedule_external_input(SimTime::ZERO, NeuronId(1), 20.0)?;
//! let report = simulation.run_until(SimTime(1_000))?;
//! assert_eq!(report.spikes_emitted, 1);
//! # Ok(())
//! # }
//! ```
//!
//! Configure local state before constructing the simulation. Use an explicit
//! horizon with [`runtime::Simulation::run_until`] for autonomous firing or
//! recurring homeostasis. Read results through [`runtime::Simulation::network`]
//! and [`runtime::Simulation::event_log`].
//!
//! The [`experiment`] module supplies reference studies built on these same
//! APIs. Optional features expose `diagnostics` (snapshots and metrics),
//! `visualization` (CSV/JSON export), and the reserved `development` boundary.

/// Validated, immutable simulation parameters.
pub mod config;
/// Neurons, synapses, spikes, IDs and sparse graph state.
pub mod core;
/// Neutral environment interfaces and deterministic test worlds.
pub mod environment;
/// Reproducible experiment assembly and M0 comparisons.
pub mod experiment;
/// Local synaptic and cellular plasticity rules.
pub mod learning;
/// Pure position, distance and decay helpers.
pub mod math;
/// Fixed nerve fibers, bundles, mappings and routing.
pub mod nerves;
/// Policy-free identities, values, geometry, and exact simulation time.
pub mod primitives;
/// Stable sensory and motor connection points.
pub mod roots;
/// Deterministic event scheduling and simulation.
pub mod runtime;
/// Translation between external values and spike trains.
pub mod transduction;

/// Local development mechanisms reserved for post-M0 work.
#[cfg(feature = "development")]
pub mod development;

/// Read-only event logs and state snapshots.
#[cfg(feature = "diagnostics")]
pub mod debug;

#[cfg(not(feature = "diagnostics"))]
#[allow(dead_code, unused_imports)]
mod debug;

/// Read-only firing, weight and sequence metrics.
#[cfg(feature = "diagnostics")]
pub mod metrics;

// Experiment orchestration derives its result record through the same pure
// metric functions. Keep them crate-private in the lean build and expose them
// publicly only when diagnostics are requested.
#[cfg(not(feature = "diagnostics"))]
#[allow(dead_code, unused_imports)]
mod metrics;

/// Neutral CSV/JSON state export.
#[cfg(feature = "visualization")]
pub mod visualization;
