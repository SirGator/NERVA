//! DSVLM-M0: a deterministic, event-driven spatial spiking-network library.
//!
//! The core only represents local neuronal and synaptic state. Simulation,
//! plasticity, external connections, experiments and diagnostics live in
//! separate slices so observing an experiment can never change its outcome.

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
