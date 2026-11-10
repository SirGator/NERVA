//! Local synaptic and cellular plasticity.
//!
//! Learning observes only affected neurons, synapses, their local traces, and
//! event timestamps. It has no dependency on environments, roots, metrics, or
//! a global objective.

/// Hard bounds for non-negative synaptic weight magnitudes.
pub mod bounds;
/// Optional fixed-budget competition among excitatory incoming synapses.
pub mod competition;
/// Excitatory causal utility rule.
pub mod excitatory_utility;
/// Per-neuron cellular and structural homeostasis.
pub mod homeostasis;
/// Pair-based spike-timing-dependent plasticity.
pub mod pair_stdp;
/// Runtime-facing local plasticity trait.
pub mod rule;
/// Analytically decaying local traces.
pub mod traces;
/// Synapse-local structural utility evidence and memory boundary.
pub mod utility;
/// Opt-in compensated publication of synaptic learning mass.
pub mod weight_update;

pub use bounds::{WeightBounds, WeightBoundsError};
pub use competition::{CompetitionError, CompetitionEvent, CompetitionPool, IncomingCompetition};
pub use excitatory_utility::ExcitatoryCausalUtility;
pub use homeostasis::{HomeostasisError, HomeostaticChange, LocalHomeostasis};
pub use pair_stdp::{PairStdp, PairStdpError, ThreeFactorConfig};
pub use rule::{DelayedLearningUpdate, NoPlasticity, PlasticityRule};
pub use traces::{DecayingTrace, TraceError};
pub use utility::{NoUtility, UtilityContext, UtilityDynamicsConfig, UtilityRule, UtilityTrigger};
pub use weight_update::WeightUpdateMode;
