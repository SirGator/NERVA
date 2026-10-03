//! Local structural-plasticity mechanisms.
//!
//! M1.1 starts deliberately small: a neuron's local structural drive is an
//! **incoming growth demand** signal, and M1.2/M1.3 may form at most one
//! incoming synapse per target cell and explicit development step. There is
//! no global objective, population statistic, field, or topology command here.
//!
//! ## Direction
//!
//! A positive structural drive `D_i^{in,+}` means neuron *i* needs more
//! **incoming** input. Formation therefore creates `source → i` (incoming),
//! not `i → source` (outgoing). A separate outgoing-growth drive may be
//! introduced later for axonal growth. Negative structural drive values
//! carry no pruning semantics; they are clamped to zero by
//! [`structural_drive::IncomingGrowthDrive`].
//!
//! ## Atomicity
//!
//! [`plan::DevelopmentPlan`] builds all topology changes against an unmodified
//! network, validates them, and commits them atomically. This prevents a
//! partial failure from leaving half-applied mutations and stops earlier
//! formations from contaminating the candidate search of later neurons in the
//! same step.
//!
//! ## Pruning
//!
//! Pruning is synapse-local: each synapse carries a smoothed utility estimate
//! that decays toward zero between evidence events with time constant `τ_U`.
//! A synapse whose utility stays below a threshold for long enough is pruned.
//! (Exponential decay of utility between evidence events lands in M1.6.)
//! This is not derived from the postsynaptic neuron feeling "over-supplied";
//! the connection itself is evaluated.

/// Finds and scores nearby, not-yet-connected local formation candidates.
pub mod candidate_search;
/// Forms one selected local connection through the live runtime boundary.
pub mod formation;
/// Atomic plan-then-commit topology changes for development steps.
pub mod plan;
/// Synapse-local pruning based on a smoothed utility criterion.
pub mod pruning;
/// Splits a signed local structural signal into growth and pruning requests.
pub mod structural_drive;

pub use candidate_search::{
    Candidate, CandidateSearchConfig, CandidateSearchError, local_candidates,
};
pub use formation::{CreatedSynapse, FormationConfig, FormationError};
pub use plan::{DevelopmentPlan, DevelopmentPlanError, PlannedFormation, PlannedPruning};
pub use pruning::{PruningConfig, PruningController, PruningError};
pub use structural_drive::{IncomingGrowthDrive, StructuralDriveError};
