//! M1.6a end-to-end test: Formation → Activity → Utility → Retention/Pruning.
//!
//! Verifies the complete self-organisation cycle for an excitatory connection:
//!
//! 1. A neuron with positive structural drive triggers formation of a new
//!    incoming synapse.
//! 2. The new synapse carries spikes, the postsynaptic neuron fires, and the
//!    `ExcitatoryCausalUtility` rule accumulates positive utility evidence.
//! 3. The utility memory stays above the pruning threshold, so the synapse is
//!    **retained**.
//! 4. A second synapse that never carries useful activity decays below the
//!    threshold and is **pruned** after `T_min`.
//!
//! This is the first test that exercises the full
//! `Formation ↔ Activity ↔ Utility ↔ Pruning` loop.

#![cfg(feature = "development")]

use nerva::{
    config::{NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    development::{DevelopmentPlan, FormationConfig, PruningConfig, PruningController},
    learning::{ExcitatoryCausalUtility, NoPlasticity, UtilityDynamicsConfig},
    math::Position3D,
    primitives::Weight,
    runtime::Simulation,
};

const TARGET: NeuronId = NeuronId(1);
const USEFUL_SOURCE: NeuronId = NeuronId(2);
const SILENT_SOURCE: NeuronId = NeuronId(3);

const USEFUL_SYNAPSE: SynapseId = SynapseId(100);
const SILENT_SYNAPSE: SynapseId = SynapseId(101);

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

fn build_network() -> Network {
    let mut network = Network::new();
    // Target neuron with high structural drive → will request growth.
    network.add_neuron(make_neuron(TARGET, 0.0, 2.0)).unwrap();
    // Two excitatory sources nearby.
    network.add_neuron(make_neuron(USEFUL_SOURCE, 0.5, 0.0)).unwrap();
    network.add_neuron(make_neuron(SILENT_SOURCE, 1.0, 0.0)).unwrap();
    network
}

fn build_simulation(network: Network) -> Simulation<NoPlasticity> {
    Simulation::with_utility_rule(
        network,
        NoPlasticity,
        Box::new(ExcitatoryCausalUtility::new()),
        RuntimeConfig::default(),
        1.0,
    )
    .and_then(|sim| {
        sim.with_utility_dynamics(UtilityDynamicsConfig {
            eligibility_tau_us: 20_000.0,
            eta: 0.2,
            utility_tau_us: 500_000.0,
        })
    })
    .expect("simulation with utility rule")
}

fn formation_config() -> FormationConfig {
    FormationConfig {
        growth_threshold: 1.0,
        initial_weight: Weight::new(0.5).unwrap(),
        delay_us: 1,
        plastic: true,
        drive_consumption: 1.0,
        ..Default::default()
    }
}

fn pruning_config() -> PruningConfig {
    PruningConfig {
        pruning_threshold: 0.05,
        min_below_duration_us: 1_000_000,
        prune_non_plastic: false,
    }
}

/// Drive the useful source and the target together so the useful synapse
/// accumulates utility evidence. Leave the silent source alone.
fn drive_useful_activity(simulation: &mut Simulation<NoPlasticity>) {
    // A burst of correlated spikes: USEFUL_SOURCE fires, then TARGET gets
    // external input so it fires too. Repeat several times.
    for t in [100u64, 500, 900, 1_300, 1_700] {
        simulation
            .schedule_external_input(SimTime(t), USEFUL_SOURCE, 100.0)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(t + 50), TARGET, 100.0)
            .unwrap();
    }
    simulation.run_until(SimTime(2_000)).unwrap();
}

#[test]
fn useful_synapse_accumulates_utility_and_is_retained() {
    let mut network = build_network();
    // Manually add both synapses so we can track them by stable IDs.
    network
        .add_synapse(
            Synapse::new(USEFUL_SYNAPSE, USEFUL_SOURCE, TARGET, Weight::new(0.5).unwrap(), 1, true)
                .unwrap(),
        )
        .unwrap();
    network
        .add_synapse(
            Synapse::new(SILENT_SYNAPSE, SILENT_SOURCE, TARGET, Weight::new(0.5).unwrap(), 1, true)
                .unwrap(),
        )
        .unwrap();

    let mut simulation = build_simulation(network);
    drive_useful_activity(&mut simulation);

    // The useful synapse should have accumulated positive utility.
    let useful = simulation.network().synapse(USEFUL_SYNAPSE).unwrap();
    let useful_u = useful.utility();
    assert!(
        useful_u > 0.0,
        "useful synapse should have positive utility, got {useful_u}"
    );
    assert!(
        useful.utility_eligibility() > 0.0,
        "useful synapse should have positive eligibility trace"
    );

    // The silent synapse should still have zero utility (no evidence ever).
    let silent = simulation.network().synapse(SILENT_SYNAPSE).unwrap();
    assert_eq!(
        silent.utility(),
        0.0,
        "silent synapse should have zero utility"
    );
    assert_eq!(
        silent.utility_eligibility(),
        0.0,
        "silent synapse should have zero eligibility"
    );

    // Observe pruning state: the useful synapse must not be a candidate.
    let mut controller = PruningController::new(pruning_config()).unwrap();
    controller.observe(&simulation).unwrap();

    let useful_since = controller.below_threshold_since(USEFUL_SYNAPSE);
    assert_eq!(
        useful_since, None,
        "useful synapse is above threshold → no pruning timer"
    );

    // The silent synapse has U=0 < threshold → timer should start.
    let silent_since = controller.below_threshold_since(SILENT_SYNAPSE);
    assert!(
        silent_since.is_some(),
        "silent synapse is below threshold → timer should start"
    );
}

#[test]
fn silent_synapse_is_pruned_after_min_duration() {
    let mut network = build_network();
    network
        .add_synapse(
            Synapse::new(USEFUL_SYNAPSE, USEFUL_SOURCE, TARGET, Weight::new(0.5).unwrap(), 1, true)
                .unwrap(),
        )
        .unwrap();
    network
        .add_synapse(
            Synapse::new(SILENT_SYNAPSE, SILENT_SOURCE, TARGET, Weight::new(0.5).unwrap(), 1, true)
                .unwrap(),
        )
        .unwrap();

    let mut simulation = build_simulation(network);

    // Drive useful activity to keep the useful synapse alive.
    drive_useful_activity(&mut simulation);

    // Observe once right after activity to start the timer on the silent
    // synapse (which has U=0 < threshold).
    let mut controller = PruningController::new(pruning_config()).unwrap();
    controller.observe(&simulation).unwrap();
    let silent_timer = controller.below_threshold_since(SILENT_SYNAPSE);
    assert!(silent_timer.is_some(), "silent synapse timer should start");

    // Now let a long silent period pass: both synapses decay, but the useful
    // one had high utility. Run well past T_min = 1_000_000 us.
    simulation
        .schedule_external_input(SimTime(5_000_000), TARGET, 0.0)
        .unwrap();
    simulation.run_until(SimTime(5_000_000)).unwrap();

    controller.observe(&simulation).unwrap();
    let candidates = controller.candidates(&simulation).unwrap();

    let silent_candidate = candidates.iter().find(|c| c.synapse_id == SILENT_SYNAPSE);
    assert!(
        silent_candidate.is_some(),
        "silent synapse should be a pruning candidate after T_min"
    );

    // The useful synapse may or may not be a candidate depending on how much
    // utility it accumulated. The key assertion is that the silent one is
    // definitely pruned. Verify via a DevelopmentPlan commit.
    use nerva::development::PlannedPruning;
    let mut plan = DevelopmentPlan::default();
    plan.prunings.push(PlannedPruning {
        synapse_id: SILENT_SYNAPSE,
        target: TARGET,
    });
    plan.validate(&simulation).unwrap();
    plan.commit(&mut simulation, &formation_config(), SynapseId(200)).unwrap();

    assert!(
        simulation.network().synapse(SILENT_SYNAPSE).is_none(),
        "silent synapse must be removed after commit"
    );
    assert!(
        simulation.network().synapse(USEFUL_SYNAPSE).is_some(),
        "useful synapse must still exist"
    );
}

#[test]
fn formation_then_activity_then_retention_full_cycle() {
    // Full cycle: start with NO synapses, form one via development, drive it,
    // and confirm it survives a pruning pass.
    let network = build_network();
    let mut simulation = Simulation::with_utility_rule(
        network,
        NoPlasticity,
        Box::new(ExcitatoryCausalUtility::new()),
        RuntimeConfig::default(),
        1.0,
    )
    .and_then(|sim| sim.with_utility_dynamics(UtilityDynamicsConfig {
        eligibility_tau_us: 20_000.0,
        eta: 0.2,
        utility_tau_us: 500_000.0,
    }))
    .unwrap();

    // Initial activity so the candidate search has spike correlation data.
    for t in [10u64, 100, 1_000] {
        simulation
            .schedule_external_input(SimTime(t), USEFUL_SOURCE, 100.0)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(t + 1), TARGET, 100.0)
            .unwrap();
    }
    simulation.run_until(SimTime(2_000)).unwrap();

    // Form a synapse via the development plan.
    let config = formation_config();
    let plan = DevelopmentPlan::build(&simulation, &config).unwrap();
    assert!(
        !plan.formations.is_empty(),
        "development plan should form at least one synapse"
    );
    let created = plan.commit(&mut simulation, &config, SynapseId(200)).unwrap();
    assert_eq!(created.len(), 1);
    let formed_id = created[0].synapse_id;
    assert_eq!(created[0].target, TARGET);

    // Now drive activity through the formed synapse: USEFUL_SOURCE fires,
    // signal travels through the new synapse (raising e_ij), and TARGET is
    // externally driven to fire shortly after so the PostSpike trigger
    // produces utility evidence while e_ij is still elevated.
    for t in [3_000u64, 4_000, 5_000, 6_000, 7_000] {
        simulation
            .schedule_external_input(SimTime(t), USEFUL_SOURCE, 100.0)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(t + 10), TARGET, 100.0)
            .unwrap();
    }
    simulation.run_until(SimTime(10_000)).unwrap();

    // The formed synapse should have accumulated positive utility.
    let formed = simulation.network().synapse(formed_id).unwrap();
    assert!(
        formed.utility() > 0.0,
        "formed synapse should have positive utility after activity, got {}",
        formed.utility()
    );

    // Run a pruning observation: the formed synapse must NOT be a candidate.
    let mut controller = PruningController::new(pruning_config()).unwrap();
    controller.observe(&simulation).unwrap();
    let candidates = controller.candidates(&simulation).unwrap();
    assert!(
        !candidates.iter().any(|c| c.synapse_id == formed_id),
        "active formed synapse must not be a pruning candidate"
    );
}

#[test]
fn non_plastic_synapse_still_accumulates_structural_utility() {
    let mut network = build_network();
    network
        .add_synapse(
            Synapse::new(
                USEFUL_SYNAPSE,
                USEFUL_SOURCE,
                TARGET,
                Weight::new(0.5).unwrap(),
                1,
                false,
            )
            .unwrap(),
        )
        .unwrap();

    let mut simulation = build_simulation(network);
    drive_useful_activity(&mut simulation);

    let synapse = simulation.network().synapse(USEFUL_SYNAPSE).unwrap();
    assert!(!synapse.is_plastic());
    assert!(
        synapse.utility_eligibility() > 0.0,
        "non-plastic synapse should still accumulate structural eligibility"
    );
    assert!(
        synapse.utility() > 0.0,
        "non-plastic synapse should still accumulate structural utility"
    );
}

#[test]
fn inhibitory_synapse_is_exempt_from_utility_based_pruning() {
    // ExcitatoryCausalUtility cannot judge inhibitory synapses. Their
    // utility memory therefore stays at zero — but zero means "unknown",
    // not "bad", so the pruning controller must not track them.
    let mut network = Network::new();
    network
        .add_neuron(
            Neuron::new(
                TARGET,
                Position3D::ORIGIN,
                Polarity::Excitatory,
                None,
                NeuronConfig::default(),
                SimTime::ZERO,
            )
            .unwrap(),
        )
        .unwrap();
    network
        .add_neuron(
            Neuron::new(
                USEFUL_SOURCE,
                Position3D::new(1.0, 0.0, 0.0),
                Polarity::Inhibitory,
                None,
                NeuronConfig::default(),
                SimTime::ZERO,
            )
            .unwrap(),
        )
        .unwrap();
    network
        .add_synapse(
            Synapse::new(
                USEFUL_SYNAPSE,
                USEFUL_SOURCE,
                TARGET,
                Weight::new(0.5).unwrap(),
                1,
                true,
            )
            .unwrap(),
        )
        .unwrap();

    let mut simulation = build_simulation(network);
    // Long silent period: no activity at all.
    simulation
        .schedule_external_input(SimTime(5_000_000), TARGET, 0.0)
        .unwrap();
    simulation.run_until(SimTime(5_000_000)).unwrap();

    let mut controller = PruningController::new(pruning_config()).unwrap();
    controller.observe(&simulation).unwrap();

    assert_eq!(
        controller.below_threshold_since(USEFUL_SYNAPSE),
        None,
        "inhibitory synapse must not be tracked: its zero utility means unknown, not bad"
    );
    let candidates = controller.candidates(&simulation).unwrap();
    assert!(
        candidates.iter().all(|c| c.synapse_id != USEFUL_SYNAPSE),
        "inhibitory synapse must never become a pruning candidate"
    );
}

#[test]
fn excitatory_synapse_without_activity_becomes_pruning_candidate() {
    // The counterpart: an excitatory synapse IS supported, so a long silent
    // period with zero utility makes it a legitimate pruning candidate.
    let mut network = Network::new();
    network
        .add_neuron(
            Neuron::new(
                TARGET,
                Position3D::ORIGIN,
                Polarity::Excitatory,
                None,
                NeuronConfig::default(),
                SimTime::ZERO,
            )
            .unwrap(),
        )
        .unwrap();
    network
        .add_neuron(
            Neuron::new(
                USEFUL_SOURCE,
                Position3D::new(1.0, 0.0, 0.0),
                Polarity::Excitatory,
                None,
                NeuronConfig::default(),
                SimTime::ZERO,
            )
            .unwrap(),
        )
        .unwrap();
    network
        .add_synapse(
            Synapse::new(
                USEFUL_SYNAPSE,
                USEFUL_SOURCE,
                TARGET,
                Weight::new(0.5).unwrap(),
                1,
                true,
            )
            .unwrap(),
        )
        .unwrap();

    let mut simulation = build_simulation(network);
    simulation
        .schedule_external_input(SimTime(1_000), TARGET, 0.0)
        .unwrap();
    simulation.run_until(SimTime(1_000)).unwrap();

    // First observation starts the timer (never-updated utility is
    // conservatively "below since now").
    let mut controller = PruningController::new(pruning_config()).unwrap();
    controller.observe(&simulation).unwrap();
    assert!(
        controller.below_threshold_since(USEFUL_SYNAPSE).is_some(),
        "supported excitatory synapse with zero utility must be tracked"
    );

    // Let T_min pass, then observe again: now the synapse is a candidate.
    simulation
        .schedule_external_input(SimTime(5_000_000), TARGET, 0.0)
        .unwrap();
    simulation.run_until(SimTime(5_000_000)).unwrap();
    controller.observe(&simulation).unwrap();

    let candidates = controller.candidates(&simulation).unwrap();
    assert!(
        candidates.iter().any(|c| c.synapse_id == USEFUL_SYNAPSE),
        "supported excitatory synapse must become a pruning candidate after T_min"
    );
}

#[test]
fn no_utility_rule_exempts_every_synapse_from_pruning() {
    // NoUtility supports no polarity at all: with it active, no synapse may
    // be pruned for "low utility", because no utility statement exists.
    let mut network = Network::new();
    network
        .add_neuron(
            Neuron::new(
                TARGET,
                Position3D::ORIGIN,
                Polarity::Excitatory,
                None,
                NeuronConfig::default(),
                SimTime::ZERO,
            )
            .unwrap(),
        )
        .unwrap();
    network
        .add_neuron(
            Neuron::new(
                USEFUL_SOURCE,
                Position3D::new(1.0, 0.0, 0.0),
                Polarity::Excitatory,
                None,
                NeuronConfig::default(),
                SimTime::ZERO,
            )
            .unwrap(),
        )
        .unwrap();
    network
        .add_synapse(
            Synapse::new(
                USEFUL_SYNAPSE,
                USEFUL_SOURCE,
                TARGET,
                Weight::new(0.5).unwrap(),
                1,
                true,
            )
            .unwrap(),
        )
        .unwrap();

    let mut simulation = Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0)
        .unwrap();
    simulation
        .schedule_external_input(SimTime(5_000_000), TARGET, 0.0)
        .unwrap();
    simulation.run_until(SimTime(5_000_000)).unwrap();

    let mut controller = PruningController::new(pruning_config()).unwrap();
    controller.observe(&simulation).unwrap();

    assert_eq!(
        controller.below_threshold_since(USEFUL_SYNAPSE),
        None,
        "NoUtility must exempt every synapse from utility-based pruning"
    );
    let candidates = controller.candidates(&simulation).unwrap();
    assert!(
        candidates.is_empty(),
        "no pruning candidates may exist while NoUtility is active"
    );
}