//! Atomic development plan: plan all topology changes first, validate, then
//! commit in one transaction.
//!
//! Without a plan, a per-neuron formation loop commits each synapse
//! immediately. A failure at neuron *n* leaves neurons 1..n-1 already
//! modified, and earlier formations contaminate the candidate search of
//! later neurons in the same step. The `DevelopmentPlan` fixes both:
//!
//! 1. All candidate searches run against the same unmodified network.
//! 2. All formations and prunings are applied in one atomic commit. If any
//!    step fails, the network is unchanged.

use std::{error::Error, fmt};

use crate::{
    core::{NetworkError, NeuronId, Synapse, SynapseError, SynapseId},
    learning::PlasticityRule,
    runtime::{Simulation, SimulationError},
};

use super::{
    CandidateSearchError, FormationConfig, FormationError, StructuralDriveError,
    formation::CreatedSynapse, local_candidates, structural_drive::IncomingGrowthDrive,
};

/// One planned incoming synapse formation, ready for atomic commit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedFormation {
    /// Neuron whose growth request initiated this formation.
    pub target: NeuronId,
    /// Prospective presynaptic source.
    pub source: NeuronId,
    /// Score that selected the source.
    pub score: f32,
}

/// One planned synapse pruning, ready for atomic commit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedPruning {
    /// Synapse to remove.
    pub synapse_id: SynapseId,
    /// Postsynaptic neuron that owns the pruning decision.
    pub target: NeuronId,
}

/// A complete set of topology changes produced by one development step.
///
/// Built against an unmodified network. All changes are validated before any
/// mutation occurs. [`Self::commit`] applies them atomically through the live
/// runtime.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DevelopmentPlan {
    /// New incoming synapses to form.
    pub formations: Vec<PlannedFormation>,
    /// Existing synapses to prune.
    pub prunings: Vec<PlannedPruning>,
}

impl DevelopmentPlan {
    /// Builds a plan from the current network state without mutating it.
    ///
    /// Each neuron with growth above the threshold contributes at most one
    /// formation candidate. All searches see the same network topology.
    /// Synapse IDs are not allocated yet; that happens during commit.
    pub fn build<R: PlasticityRule>(
        simulation: &Simulation<R>,
        formation_config: &FormationConfig,
    ) -> Result<Self, DevelopmentPlanError> {
        formation_config.validate()?;
        let now = simulation.current_time();
        let network = simulation.network();
        let target_ids: Vec<_> = network.neuron_ids().collect();
        let mut formations = Vec::new();

        for target_id in target_ids {
            let target_neuron = network
                .neuron(target_id)
                .expect("neuron ID came from the authoritative network");
            if IncomingGrowthDrive::from_neuron(target_neuron)?.demand
                <= formation_config.growth_threshold
            {
                continue;
            }
            let Some(candidate) =
                local_candidates(network, target_id, now, &formation_config.candidate_search)?
                    .into_iter()
                    .next()
            else {
                continue;
            };
            formations.push(PlannedFormation {
                target: target_id,
                source: candidate.source,
                score: candidate.score,
            });
        }

        Ok(Self {
            formations,
            prunings: Vec::new(),
        })
    }

    /// Returns `true` if the plan would change no topology.
    pub fn is_empty(&self) -> bool {
        self.formations.is_empty() && self.prunings.is_empty()
    }

    /// Number of individual topology changes in this plan.
    pub fn len(&self) -> usize {
        self.formations.len() + self.prunings.len()
    }

    /// Validates the plan against the current live network.
    ///
    /// This checks that all formation sources and targets exist, no duplicate
    /// or already-existing connections are planned, and all pruning targets
    /// refer to existing synapses. No mutation occurs.
    pub fn validate<R: PlasticityRule>(
        &self,
        simulation: &Simulation<R>,
    ) -> Result<(), DevelopmentPlanError> {
        let network = simulation.network();

        let mut planned_formations: Vec<(NeuronId, NeuronId)> = Vec::new();
        for formation in &self.formations {
            if network.neuron(formation.target).is_none() {
                return Err(DevelopmentPlanError::UnknownTarget(formation.target));
            }
            if network.neuron(formation.source).is_none() {
                return Err(DevelopmentPlanError::UnknownSource(formation.source));
            }
            if formation.source == formation.target {
                return Err(DevelopmentPlanError::SelfConnection(formation.target));
            }
            let already_connected =
                network
                    .incoming_synapse_ids(formation.target)
                    .iter()
                    .any(|id| {
                        network.synapse(*id).expect("adjacency ID is valid").pre()
                            == formation.source
                    });
            if already_connected {
                return Err(DevelopmentPlanError::ConnectionAlreadyExists {
                    source: formation.source,
                    target: formation.target,
                });
            }
            let key = (formation.target, formation.source);
            if planned_formations.contains(&key) {
                return Err(DevelopmentPlanError::DuplicateFormation {
                    source: formation.source,
                    target: formation.target,
                });
            }
            planned_formations.push(key);
        }

        let mut planned_prunings: Vec<SynapseId> = Vec::new();
        for pruning in &self.prunings {
            if network.synapse(pruning.synapse_id).is_none() {
                return Err(DevelopmentPlanError::UnknownSynapse(pruning.synapse_id));
            }
            if planned_prunings.contains(&pruning.synapse_id) {
                return Err(DevelopmentPlanError::DuplicatePruning(pruning.synapse_id));
            }
            planned_prunings.push(pruning.synapse_id);
        }

        Ok(())
    }

    /// Atomically commits all planned topology changes to the live runtime.
    ///
    /// All formations and prunings are applied inside one outer runtime
    /// transaction. If any single mutation fails, the runtime restores the
    /// complete pre-commit state (network, scheduler, bookings) and no
    /// partial topology change is retained.
    pub fn commit<R: PlasticityRule>(
        &self,
        simulation: &mut Simulation<R>,
        formation_config: &FormationConfig,
        first_synapse_id: SynapseId,
    ) -> Result<Vec<CreatedSynapse>, DevelopmentPlanError> {
        formation_config.validate()?;
        self.validate(simulation)?;

        let mut next_id = first_synapse_id;
        let mut created = Vec::new();

        let result = simulation.transaction(|sim| {
            for formation in &self.formations {
                let synapse_id = allocate_fresh_id(sim, next_id)?;
                next_id = SynapseId(
                    synapse_id
                        .get()
                        .checked_add(1)
                        .ok_or(SimulationError::NoSynapseIdsAvailable)?,
                );

                let synapse = Synapse::new(
                    synapse_id,
                    formation.source,
                    formation.target,
                    formation_config.initial_weight,
                    formation_config.delay_us,
                    formation_config.plastic,
                )
                .map_err(|e| SimulationError::InvalidNetwork(NetworkError::InvalidSynapse(e)))?;
                sim.add_synapse_inner(synapse)?;

                let drive_reduction = formation_config.drive_consumption;
                sim.update_neuron_inner(formation.target, |neuron| {
                    let current = neuron.structural_drive();
                    neuron.set_structural_drive_clamped(
                        current - drive_reduction,
                        -1_000_000.0,
                        1_000_000.0,
                    )?;
                    Ok(())
                })?;

                created.push(CreatedSynapse {
                    synapse_id,
                    source: formation.source,
                    target: formation.target,
                    score: formation.score,
                });
            }

            for pruning in &self.prunings {
                sim.remove_synapse_inner(pruning.synapse_id)?;
            }

            Ok(())
        });

        match result {
            Ok(()) => Ok(created),
            Err(sim_error) => {
                created.clear();
                Err(DevelopmentPlanError::Simulation(sim_error))
            }
        }
    }
}

fn allocate_fresh_id<R: PlasticityRule>(
    simulation: &Simulation<R>,
    start: SynapseId,
) -> Result<SynapseId, SimulationError> {
    let mut raw = start.get();
    loop {
        let id = SynapseId(raw);
        if simulation.network().synapse(id).is_none() {
            return Ok(id);
        }
        raw = raw
            .checked_add(1)
            .ok_or(SimulationError::NoSynapseIdsAvailable)?;
    }
}

/// Plan construction, validation, or commit failed.
#[derive(Clone, Debug, PartialEq)]
pub enum DevelopmentPlanError {
    /// Formation configuration was invalid.
    Formation(FormationError),
    /// Candidate search configuration was invalid.
    Search(CandidateSearchError),
    /// A neuron had an invalid local structural drive.
    StructuralDrive(StructuralDriveError),
    /// A planned target neuron does not exist.
    UnknownTarget(NeuronId),
    /// A planned source neuron does not exist.
    UnknownSource(NeuronId),
    /// A planned self-connection was rejected.
    SelfConnection(NeuronId),
    /// A planned connection already exists.
    ConnectionAlreadyExists { source: NeuronId, target: NeuronId },
    /// The same formation was planned twice.
    DuplicateFormation { source: NeuronId, target: NeuronId },
    /// A planned pruning target synapse does not exist.
    UnknownSynapse(SynapseId),
    /// The same synapse was planned for pruning twice.
    DuplicatePruning(SynapseId),
    /// Constructing a planned synapse violated a core invariant.
    Synapse(SynapseError),
    /// The live runtime rejected a topology mutation.
    Simulation(SimulationError),
    /// No unused `SynapseId` remains.
    NoSynapseIdsAvailable,
}

impl From<FormationError> for DevelopmentPlanError {
    fn from(error: FormationError) -> Self {
        Self::Formation(error)
    }
}

impl From<CandidateSearchError> for DevelopmentPlanError {
    fn from(error: CandidateSearchError) -> Self {
        Self::Search(error)
    }
}

impl From<StructuralDriveError> for DevelopmentPlanError {
    fn from(error: StructuralDriveError) -> Self {
        Self::StructuralDrive(error)
    }
}

impl fmt::Display for DevelopmentPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Formation(error) => write!(formatter, "invalid formation config: {error}"),
            Self::Search(error) => write!(formatter, "invalid candidate search: {error}"),
            Self::StructuralDrive(error) => {
                write!(formatter, "invalid local structural drive: {error}")
            }
            Self::UnknownTarget(id) => write!(formatter, "planned target {id} does not exist"),
            Self::UnknownSource(id) => write!(formatter, "planned source {id} does not exist"),
            Self::SelfConnection(id) => {
                write!(formatter, "planned self-connection to {id} rejected")
            }
            Self::ConnectionAlreadyExists { source, target } => {
                write!(formatter, "connection {source} → {target} already exists")
            }
            Self::DuplicateFormation { source, target } => {
                write!(formatter, "formation {source} → {target} was planned twice")
            }
            Self::UnknownSynapse(id) => {
                write!(
                    formatter,
                    "planned pruning target synapse {id} does not exist"
                )
            }
            Self::DuplicatePruning(id) => {
                write!(formatter, "synapse {id} was planned for pruning twice")
            }
            Self::Synapse(error) => {
                write!(formatter, "cannot construct planned synapse: {error}")
            }
            Self::Simulation(error) => {
                write!(
                    formatter,
                    "runtime rejected planned topology change: {error}"
                )
            }
            Self::NoSynapseIdsAvailable => {
                formatter.write_str("no synapse identities remain for development plan")
            }
        }
    }
}

impl Error for DevelopmentPlanError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Formation(error) => Some(error),
            Self::Search(error) => Some(error),
            Self::StructuralDrive(error) => Some(error),
            Self::Synapse(error) => Some(error),
            Self::Simulation(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{NeuronConfig, RuntimeConfig},
        core::{Network, Neuron, NeuronId, Polarity, SimTime, SynapseId},
        development::CandidateSearchConfig,
        learning::NoPlasticity,
        math::Position3D,
    };

    fn neuron(id: u64, x: f32, drive: f32) -> Neuron {
        let mut neuron = Neuron::new(
            NeuronId(id),
            Position3D::new(x, 0.0, 0.0),
            Polarity::Excitatory,
            None,
            NeuronConfig::default(),
            SimTime::ZERO,
        )
        .unwrap();
        neuron
            .set_structural_drive_clamped(drive, -10.0, 10.0)
            .unwrap();
        neuron
    }

    fn setup_simulation() -> Simulation<NoPlasticity> {
        let mut network = Network::new();
        network.add_neuron(neuron(1, 0.0, 2.0)).unwrap();
        network.add_neuron(neuron(2, 1.0, 0.0)).unwrap();
        let mut simulation =
            Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0).unwrap();
        for id in [1, 2] {
            simulation
                .schedule_external_input(SimTime(10), NeuronId(id), 100.0)
                .unwrap();
        }
        simulation.run_until(SimTime(20)).unwrap();
        simulation
    }

    #[test]
    fn plan_build_does_not_mutate_network() {
        let simulation = setup_simulation();
        let config = FormationConfig {
            growth_threshold: 1.0,
            candidate_search: CandidateSearchConfig {
                radius: 2.0,
                activity_weight: 0.0,
                temporal_weight: 1.0,
                distance_weight: 0.0,
                temporal_tau_us: 1.0,
                recency_tau_us: 1_000_000.0,
                min_candidate_score: 0.0,
            },
            ..Default::default()
        };

        let synapse_count_before = simulation.network().synapse_count();
        let plan = DevelopmentPlan::build(&simulation, &config).unwrap();

        assert_eq!(simulation.network().synapse_count(), synapse_count_before);
        assert_eq!(plan.formations.len(), 1);
        assert_eq!(plan.formations[0].target, NeuronId(1));
        assert_eq!(plan.formations[0].source, NeuronId(2));
    }

    #[test]
    fn plan_commit_applies_all_formations() {
        let mut simulation = setup_simulation();
        let config = FormationConfig {
            growth_threshold: 1.0,
            drive_consumption: 1.0,
            candidate_search: CandidateSearchConfig {
                radius: 2.0,
                activity_weight: 0.0,
                temporal_weight: 1.0,
                distance_weight: 0.0,
                temporal_tau_us: 1.0,
                recency_tau_us: 1_000_000.0,
                min_candidate_score: 0.0,
            },
            ..Default::default()
        };

        let plan = DevelopmentPlan::build(&simulation, &config).unwrap();
        let created = plan
            .commit(&mut simulation, &config, SynapseId(10))
            .unwrap();

        assert_eq!(created.len(), 1);
        assert_eq!(simulation.network().synapse_count(), 1);
        let synapse = simulation.network().synapse(SynapseId(10)).unwrap();
        assert_eq!(synapse.pre(), NeuronId(2));
        assert_eq!(synapse.post(), NeuronId(1));
    }

    #[test]
    fn plan_commit_reduces_structural_drive() {
        let mut simulation = setup_simulation();
        let config = FormationConfig {
            growth_threshold: 1.0,
            drive_consumption: 1.5,
            candidate_search: CandidateSearchConfig {
                radius: 2.0,
                activity_weight: 0.0,
                temporal_weight: 1.0,
                distance_weight: 0.0,
                temporal_tau_us: 1.0,
                recency_tau_us: 1_000_000.0,
                min_candidate_score: 0.0,
            },
            ..Default::default()
        };

        let plan = DevelopmentPlan::build(&simulation, &config).unwrap();
        plan.commit(&mut simulation, &config, SynapseId(10))
            .unwrap();

        let drive = simulation
            .network()
            .neuron(NeuronId(1))
            .unwrap()
            .structural_drive();
        assert!(
            (drive - 0.5).abs() < 1e-5,
            "drive should be 2.0 - 1.5 = 0.5, got {drive}"
        );
    }

    #[test]
    fn empty_plan_commits_without_changes() {
        let mut simulation = setup_simulation();
        let config = FormationConfig {
            growth_threshold: 100.0,
            ..Default::default()
        };

        let plan = DevelopmentPlan::build(&simulation, &config).unwrap();
        assert!(plan.is_empty());
        let created = plan
            .commit(&mut simulation, &config, SynapseId(10))
            .unwrap();
        assert!(created.is_empty());
        assert_eq!(simulation.network().synapse_count(), 0);
    }
}
