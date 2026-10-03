//! M1.3 formation configuration and result types.
//!
//! A neuron with positive incoming-growth drive `D_i^{in,+}` needs more input.
//! Formation finds a nearby candidate source `j` and creates `j → i`, bringing
//! new input to the undersupplied neuron. The target's structural drive is
//! reduced after a successful formation so repeated steps do not accumulate
//! synapses indefinitely.
//!
//! ## API
//!
//! The single supported development path is:
//!
//! ```text
//! DevelopmentPlan::build()
//!     ↓
//! PruningController::populate_plan()
//!     ↓
//! DevelopmentPlan::validate()
//!     ↓
//! DevelopmentPlan::commit()
//! ```
//!
//! The old per-neuron `FormationController::step()` loop was removed because it
//! committed each synapse immediately without an atomic transaction. Use
//! `DevelopmentPlan` for all topology changes.

use std::{error::Error, fmt};

use crate::{
    core::{NeuronId, SynapseError, SynapseId},
    primitives::Weight,
};

use super::CandidateSearchConfig;

/// Immutable parameters for the M1.3 formation primitive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FormationConfig {
    /// Strict lower bound on a target cell's local incoming-growth request.
    pub growth_threshold: f32,
    /// Weight magnitude of each newly formed connection.
    pub initial_weight: Weight,
    /// Propagation delay of each newly formed connection in microseconds.
    pub delay_us: u64,
    /// Whether the normal local learning rule may subsequently alter it.
    pub plastic: bool,
    /// Amount of structural drive consumed by each successful formation.
    ///
    /// After a new incoming synapse is created, the target neuron's
    /// `structural_drive` is reduced by this amount. This prevents repeated
    /// development steps from accumulating connections indefinitely.
    pub drive_consumption: f32,
    /// Local geometric and activity score parameters.
    pub candidate_search: CandidateSearchConfig,
}

impl FormationConfig {
    /// Validates formation and candidate-search parameters before mutation.
    pub fn validate(&self) -> Result<(), FormationError> {
        if !self.growth_threshold.is_finite() || self.growth_threshold < 0.0 {
            return Err(FormationError::InvalidGrowthThreshold(
                self.growth_threshold,
            ));
        }
        if !self.drive_consumption.is_finite() || self.drive_consumption < 0.0 {
            return Err(FormationError::InvalidDriveConsumption(
                self.drive_consumption,
            ));
        }
        if self.delay_us == 0 {
            return Err(FormationError::ZeroDelay);
        }
        self.candidate_search
            .validate()
            .map_err(FormationError::Search)
    }
}

impl Default for FormationConfig {
    fn default() -> Self {
        Self {
            growth_threshold: 1.0,
            initial_weight: Weight::new(0.1).expect("constant default weight is valid"),
            delay_us: 1,
            plastic: true,
            drive_consumption: 1.0,
            candidate_search: CandidateSearchConfig::default(),
        }
    }
}

/// One synapse that was added by a development step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CreatedSynapse {
    /// Newly allocated stable connection identity.
    pub synapse_id: SynapseId,
    /// Locally selected presynaptic source neuron.
    pub source: NeuronId,
    /// Neuron whose local incoming-growth request initiated the connection.
    pub target: NeuronId,
    /// Score that selected the source.
    pub score: f32,
}

/// Formation configuration, search, or runtime mutation failed.
#[derive(Clone, Debug, PartialEq)]
pub enum FormationError {
    /// The strict growth threshold was negative, NaN, or infinite.
    InvalidGrowthThreshold(f32),
    /// The drive consumption was negative, NaN, or infinite.
    InvalidDriveConsumption(f32),
    /// New synapses must retain the runtime's positive-delay causality rule.
    ZeroDelay,
    /// Candidate scoring configuration was invalid.
    Search(super::CandidateSearchError),
    /// A neuron had an invalid local structural drive.
    StructuralDrive(super::StructuralDriveError),
    /// Constructing the synapse violated a core invariant.
    Synapse(SynapseError),
    /// The live runtime rejected the topology mutation.
    Simulation(crate::runtime::SimulationError),
    /// No unused `SynapseId` remains at or above the cursor.
    NoSynapseIdsAvailable,
}

impl From<super::StructuralDriveError> for FormationError {
    fn from(error: super::StructuralDriveError) -> Self {
        Self::StructuralDrive(error)
    }
}

impl From<super::CandidateSearchError> for FormationError {
    fn from(error: super::CandidateSearchError) -> Self {
        Self::Search(error)
    }
}

impl fmt::Display for FormationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGrowthThreshold(value) => write!(
                formatter,
                "growth threshold must be finite and non-negative, got {value}"
            ),
            Self::InvalidDriveConsumption(value) => write!(
                formatter,
                "drive consumption must be finite and non-negative, got {value}"
            ),
            Self::ZeroDelay => {
                formatter.write_str("formation delay must be at least one microsecond")
            }
            Self::Search(error) => write!(formatter, "invalid formation candidate search: {error}"),
            Self::StructuralDrive(error) => {
                write!(formatter, "invalid local structural drive: {error}")
            }
            Self::Synapse(error) => write!(formatter, "cannot construct formed synapse: {error}"),
            Self::Simulation(error) => {
                write!(formatter, "runtime rejected formed synapse: {error}")
            }
            Self::NoSynapseIdsAvailable => {
                formatter.write_str("no synapse identities remain for formation")
            }
        }
    }
}

impl Error for FormationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Search(error) => Some(error),
            Self::StructuralDrive(error) => Some(error),
            Self::Synapse(error) => Some(error),
            Self::Simulation(error) => Some(error),
            _ => None,
        }
    }
}
