//! Core library for the Dynamic Self-organising Vector Learning Model (DSVLM).

/// Regional network state and neuromodulators.
pub mod area;
/// Shared identifiers and simulation time.
pub mod common;
/// Deterministic, timestamped events.
pub mod event;
/// Connections to the outside world and between large subsystems.
pub mod nerves;
/// Graph-level network representation and execution.
pub mod network;
/// Single-neuron configuration, state, and dynamics.
pub mod neuron;
/// Ready-to-use model parameter sets.
pub mod presets;
/// Synaptic transmission and plasticity.
pub mod synapse;
