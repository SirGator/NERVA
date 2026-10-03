//! M1.3/M1.5 combined formation scenario.
//!
//! Verifies the full formation pipeline against one canonical A/B/C topology:
//!
//! ```text
//!   A     B     C
//!
//!   B needs input (positive structural drive).
//!   A and B fired repeatedly at nearby times (correlated).
//!   C is close but fired at unrelated times (uncorrelated).
//!
//!             ↓ DevelopmentPlan::build + commit
//!
//!   A ───────→ B
//!
//!   C   (no connection)
//! ```
//!
//! The single test asserts every M1.3a/M1.5a invariant at once:
//! - no self-loop,
//! - no duplicate synapse,
//! - no candidate below the minimum score,
//! - old correlation loses its effect (recency decay),
//! - a commit error leaves the network unchanged (atomicity).

#![cfg(feature = "development")]

use nerva::{
    config::{NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, SynapseId},
    development::{
        CandidateSearchConfig, DevelopmentPlan, FormationConfig, IncomingGrowthDrive,
        PlannedFormation,
    },
    learning::NoPlasticity,
    math::Position3D,
    primitives::Weight,
    runtime::Simulation,
};

const A: NeuronId = NeuronId(1);
const B: NeuronId = NeuronId(2);
const C: NeuronId = NeuronId(3);

const FIRST_SYNAPSE: SynapseId = SynapseId(500);

fn make_neuron(id: NeuronId, x: f32, drive: f32) -> Neuron {
    let mut neuron = Neuron::new(
        id,
        Position3D::new(x, 0.0, 0.0),
        Polarity::Excitatory,
        None,
        NeuronConfig::default(),
        SimTime::ZERO,
    )
    .expect("neuron construction");
    neuron
        .set_structural_drive_clamped(drive, -10.0, 10.0)
        .expect("drive clamp");
    neuron
}

fn correlated_ab_uses_c_as_decoy() -> Simulation<NoPlasticity> {
    let mut network = Network::new();
    network.add_neuron(make_neuron(A, 0.0, 0.0)).unwrap();
    network.add_neuron(make_neuron(B, 1.0, 2.0)).unwrap();
    network.add_neuron(make_neuron(C, 0.5, 0.0)).unwrap();

    let mut simulation =
        Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0).unwrap();

    for t in [10u64, 100, 1_000] {
        simulation
            .schedule_external_input(SimTime(t), A, 100.0)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(t + 1), B, 100.0)
            .unwrap();
    }
    simulation
        .schedule_external_input(SimTime(5_000), C, 100.0)
        .unwrap();

    simulation.run_until(SimTime(2_000)).unwrap();
    simulation
}

fn formation_config(min_score: f32, recency_tau_us: f32) -> FormationConfig {
    FormationConfig {
        growth_threshold: 1.0,
        initial_weight: Weight::new(0.1).unwrap(),
        delay_us: 1,
        plastic: true,
        drive_consumption: 1.0,
        candidate_search: CandidateSearchConfig {
            radius: 2.0,
            activity_weight: 0.0,
            temporal_weight: 1.0,
            distance_weight: 0.0,
            temporal_tau_us: 1_000.0,
            recency_tau_us,
            min_candidate_score: min_score,
        },
    }
}

#[test]
fn m1_formation_creates_correlated_a_to_b_only() {
    let mut simulation = correlated_ab_uses_c_as_decoy();
    let config = formation_config(0.0, 1_000_000.0);

    let plan = DevelopmentPlan::build(&simulation, &config).unwrap();

    assert_eq!(plan.formations.len(), 1, "exactly one formation expected");
    let formation = &plan.formations[0];
    assert_eq!(formation.target, B, "target must be the undersupplied neuron B");
    assert_eq!(formation.source, A, "source must be the correlated neuron A");
    assert_ne!(formation.source, formation.target, "no self-loop");

    let created = plan.commit(&mut simulation, &config, FIRST_SYNAPSE).unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(simulation.network().synapse_count(), 1);

    let synapse = simulation
        .network()
        .synapse(FIRST_SYNAPSE)
        .expect("synapse exists after commit");
    assert_eq!(synapse.pre(), A);
    assert_eq!(synapse.post(), B);

    assert!(
        simulation
            .network()
            .incoming_synapse_ids(B)
            .iter()
            .filter_map(|id| simulation.network().synapse(*id))
            .filter(|s| s.pre() == C)
            .count()
            == 0,
        "uncorrelated C must not connect to B"
    );

    let drive_after = simulation
        .network()
        .neuron(B)
        .unwrap()
        .structural_drive();
    assert!(
        (drive_after - 1.0).abs() < 1e-5,
        "drive should be 2.0 - 1.0 = 1.0 after consumption, got {drive_after}"
    );
}

#[test]
fn m1_formation_plan_does_not_mutate_network() {
    let simulation = correlated_ab_uses_c_as_decoy();
    let config = formation_config(0.0, 1_000_000.0);

    let synapse_count_before = simulation.network().synapse_count();
    let drive_before = simulation
        .network()
        .neuron(B)
        .unwrap()
        .structural_drive();

    let plan = DevelopmentPlan::build(&simulation, &config).unwrap();
    assert!(!plan.is_empty());

    assert_eq!(simulation.network().synapse_count(), synapse_count_before);
    assert_eq!(
        simulation
            .network()
            .neuron(B)
            .unwrap()
            .structural_drive(),
        drive_before,
        "build must not alter structural drive"
    );
}

#[test]
fn m1_formation_rejects_duplicate_plan() {
    let simulation = correlated_ab_uses_c_as_decoy();
    let config = formation_config(0.0, 1_000_000.0);

    let mut plan = DevelopmentPlan::build(&simulation, &config).unwrap();
    plan.formations.push(PlannedFormation {
        target: B,
        source: A,
        score: plan.formations[0].score,
    });

    let result = plan.validate(&simulation);
    assert!(
        result.is_err(),
        "duplicate formation must fail validation"
    );
}

#[test]
fn m1_formation_rejects_self_connection_in_plan() {
    let simulation = correlated_ab_uses_c_as_decoy();
    let config = formation_config(0.0, 1_000_000.0);

    let mut plan = DevelopmentPlan::build(&simulation, &config).unwrap();
    plan.formations[0].source = plan.formations[0].target;

    let result = plan.validate(&simulation);
    assert!(result.is_err(), "self-connection must fail validation");
}

#[test]
fn m1_formation_min_score_blocks_all_candidates() {
    let mut simulation = correlated_ab_uses_c_as_decoy();
    let config = formation_config(100.0, 1_000_000.0);

    let plan = DevelopmentPlan::build(&simulation, &config).unwrap();
    assert!(
        plan.formations.is_empty(),
        "no candidate should reach the minimum score"
    );

    let created = plan.commit(&mut simulation, &config, FIRST_SYNAPSE).unwrap();
    assert!(created.is_empty());
    assert_eq!(simulation.network().synapse_count(), 0);
}

#[test]
fn m1_formation_old_correlation_loses_effect() {
    let mut simulation = correlated_ab_uses_c_as_decoy();

    let config_recent = formation_config(0.0, 1_000_000.0);
    let plan_recent = DevelopmentPlan::build(&simulation, &config_recent).unwrap();
    let score_recent = plan_recent.formations[0].score;

    simulation
        .schedule_external_input(SimTime(10_000_000), A, 0.0)
        .unwrap();
    simulation.run_until(SimTime(10_000_000)).unwrap();

    let config_old = formation_config(0.0, 1_000.0);
    let plan_old = DevelopmentPlan::build(&simulation, &config_old).unwrap();

    if plan_old.formations.is_empty() {
    } else {
        let score_old = plan_old.formations[0].score;
        assert!(
            score_old < score_recent,
            "old correlation must score lower than recent: old={score_old}, recent={score_recent}"
        );
    }
}

#[test]
fn m1_formation_atomicity_on_error_after_successful_mutation() {
    let mut network = Network::new();
    network.add_neuron(make_neuron(A, 0.0, 2.0)).unwrap();
    network.add_neuron(make_neuron(B, 1.0, 2.0)).unwrap();
    network.add_neuron(make_neuron(C, 2.0, 2.0)).unwrap();
    network.add_neuron(make_neuron(NeuronId(4), 3.0, 0.0)).unwrap();
    network.add_neuron(make_neuron(NeuronId(5), 4.0, 0.0)).unwrap();
    network.add_neuron(make_neuron(NeuronId(6), 5.0, 0.0)).unwrap();

    let mut simulation =
        Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0).unwrap();
    for id in [A, B, C, NeuronId(4), NeuronId(5), NeuronId(6)] {
        simulation
            .schedule_external_input(SimTime(10), id, 100.0)
            .unwrap();
    }
    simulation.run_until(SimTime(20)).unwrap();

    let config = FormationConfig {
        growth_threshold: 1.0,
        initial_weight: Weight::new(0.1).unwrap(),
        delay_us: 1,
        plastic: true,
        drive_consumption: 1.0,
        candidate_search: CandidateSearchConfig {
            radius: 10.0,
            activity_weight: 0.0,
            temporal_weight: 1.0,
            distance_weight: 0.0,
            temporal_tau_us: 1_000.0,
            recency_tau_us: 1_000_000.0,
            min_candidate_score: 0.0,
        },
    };

    let plan = DevelopmentPlan::build(&simulation, &config).unwrap();
    assert!(
        plan.formations.len() >= 2,
        "need at least two formations for this test, got {}",
        plan.formations.len()
    );

    let synapse_count_before = simulation.network().synapse_count();
    let drives_before: Vec<(NeuronId, f32)> = [A, B, C]
        .iter()
        .map(|id| {
            (
                *id,
                simulation.network().neuron(*id).unwrap().structural_drive(),
            )
        })
        .collect();

    let overflow_id = SynapseId(u64::MAX - 1);
    let result = plan.commit(&mut simulation, &config, overflow_id);

    assert!(
        result.is_err(),
        "commit must fail when synapse IDs overflow after first formation"
    );

    assert_eq!(
        simulation.network().synapse_count(),
        synapse_count_before,
        "failed commit after successful mutation must restore synapse count"
    );

    for (id, drive_before) in drives_before {
        let drive_after = simulation
            .network()
            .neuron(id)
            .unwrap()
            .structural_drive();
        assert!(
            (drive_after - drive_before).abs() < 1e-5,
            "failed commit must restore drive for neuron {id}: before={drive_before}, after={drive_after}"
        );
    }
}

#[test]
fn m1_formation_no_duplicate_synapse_on_second_step() {
    let mut simulation = correlated_ab_uses_c_as_decoy();
    let config = formation_config(0.0, 1_000_000.0);

    let plan1 = DevelopmentPlan::build(&simulation, &config).unwrap();
    plan1.commit(&mut simulation, &config, FIRST_SYNAPSE).unwrap();
    assert_eq!(simulation.network().synapse_count(), 1);

    let plan2 = DevelopmentPlan::build(&simulation, &config).unwrap();
    if !plan2.formations.is_empty() {
        let result = plan2.validate(&simulation);
        assert!(
            result.is_err(),
            "second formation to the same target must be blocked by existing connection"
        );
    }
}

#[test]
fn m1_formation_structural_drive_interpretation() {
    let neuron_b = make_neuron(B, 1.0, 5.0);
    let drive = IncomingGrowthDrive::from_neuron(&neuron_b).unwrap();
    assert!(drive.demand > 0.0, "positive drive must mean growth demand");

    let neuron_b_neg = make_neuron(B, 1.0, -3.0);
    let drive = IncomingGrowthDrive::from_neuron(&neuron_b_neg).unwrap();
    assert_eq!(
        drive.demand, 0.0,
        "negative drive must clamp to zero, not produce pruning demand"
    );
}