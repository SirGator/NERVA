//! Local synaptic and cellular plasticity.
//!
//! Learning observes only affected neurons, synapses, their local traces, and
//! event timestamps. It has no dependency on environments, roots, metrics, or
//! a global objective.

/// Hard bounds for non-negative synaptic weight magnitudes.
pub mod bounds;
/// Per-neuron cellular and structural homeostasis.
pub mod homeostasis;
/// Pair-based spike-timing-dependent plasticity.
pub mod pair_stdp;
/// Runtime-facing local plasticity trait.
pub mod rule;
/// Analytically decaying local traces.
pub mod traces;

pub use bounds::{WeightBounds, WeightBoundsError};
pub use homeostasis::{HomeostasisError, HomeostaticChange, LocalHomeostasis};
pub use pair_stdp::{PairStdp, PairStdpError};
pub use rule::{NoPlasticity, PlasticityRule};
pub use traces::{DecayingTrace, TraceError};
