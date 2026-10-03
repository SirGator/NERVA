//! M1 completion test: topological self-organisation.
//!
//! Verifies that NERVA can, without any external direction:
//!
//! 1. Form a new connection B → X because B's activity correlates with X.
//! 2. Prune an existing connection A → X because it never carries useful
//!    activity.
//! 3. Prune an existing connection C → X for the same reason.
//!
//! The test uses the full `Formation ↔ Activity ↔ Utility ↔ Pruning` loop.
//! No connection is explicitly added or removed by the test after the initial
//! setup; all topology changes come from `DevelopmentPlan::build`,
//! `PruningController::candidates`, and `DevelopmentPlan::commit`.
//!
//! ## Scenario
//!
//! ```text
//! Initial topology:
//!   A ─────→ X     (exists, but A fires at unrelated times → useless)
//!   B              (fires correlated with X, no connection yet)
//!   C ─────→ X     (exists, but C fires at unrelated times → useless)
//!
//! After autonomous activity + development:
//!   A ─────→ X     (pruned: low utility)
//!   B ─────→ X     (formed: correlated activity → high utility)
//!   C ─────→ X     (pruned: low utility)
//! ```

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

const X: NeuronId = NeuronId(1);
const A: NeuronId = NeuronId(2);
const B: NeuronId = NeuronId(3);
const C: NeuronId = NeuronId(4);

const A_TO_X: SynapseId = SynapseId(100);
const C_TO_X: SynapseId = SynapseId(101);

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
    // X is the target, with a small structural drive so it requests growth.
    network.add_neuron(make_neuron(X, 0.0, 2.0)).unwrap();
    // A, B, C are potential sources.
    network.add_neuron(make_neuron(A, 0.5, 0.0)).unwrap();
    network.add_neuron(make_neuron(B, 1.0, 0.0)).unwrap();
    network.add_neuron(make_neuron(C, 1.5, 0.0)).unwrap();

    // Pre-existing A → X and C → X connections (will be pruned).
    network
        .add_synapse(
            Synapse::new(A_TO_X, A, X, Weight::new(0.5).unwrap(), 1, true).unwrap(),
        )
        .unwrap();
    network
        .add_synapse(
            Synapse::new(C_TO_X, C, X, Weight::new(0.5).unwrap(), 1, true).unwrap(),
        )
        .unwrap();
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
            utility_tau_us: 5_000_000.0,
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
        pruning_threshold: 0.02,
        min_below_duration_us: 2_000_000,
        prune_non_plastic: false,
    }
}

/// Drive B and X together (correlated), and A and C at unrelated times.
///
/// A and C fire at times that are deliberately **not** shortly before X's
/// spikes. Since the excitatory causal utility rule credits a synapse when
/// the postsynaptic neuron fires while `e_ij` is still elevated, A and C must
/// fire far enough from X's spikes that their eligibility traces have decayed
/// to near-zero by the time X fires.
fn drive_correlated_activity(simulation: &mut Simulation<NoPlasticity>) {
    // B and X fire close together → B's signal arrives at X, X fires,
    // eligibility on B → X synapse rises, utility accumulates.
    // But B has no synapse to X yet, so we also drive X externally.
    for t in [100u64, 500, 1_000, 2_000, 3_000, 4_000] {
        // B fires (no synapse yet, so this just makes B's activity trace rise).
        simulation
            .schedule_external_input(SimTime(t), B, 100.0)
            .unwrap();
        // X fires shortly after B.
        simulation
            .schedule_external_input(SimTime(t + 20), X, 100.0)
            .unwrap();
    }
    // A fires at unrelated times — far from X's spikes. X fires at
    // 100, 500, 1000, 2000, 3000, 4000. A fires at 300, 700, 1500, 2500,
    // 5000 — these are 200+ us away from any X spike. With τ_e = 20_000,
    // the eligibility trace decays by exp(-200/20000) ≈ 0.99 — still high.
    // We need A to fire much earlier or much later. Let's fire A at times
    // that are 10_000 us away from any X spike so the trace fully decays.
    for t in [10_000u64, 20_000, 30_000, 50_000, 90_000] {
        simulation
            .schedule_external_input(SimTime(t), A, 100.0)
            .unwrap();
    }
    // C fires at unrelated times — also far from X's spikes.
    for t in [15_000u64, 25_000, 35_000, 60_000, 80_000] {
        simulation
            .schedule_external_input(SimTime(t), C, 100.0)
            .unwrap();
    }
    simulation.run_until(SimTime(100_000)).unwrap();
}

#[test]
fn m1_self_organisation_forms_useful_prunes_useless() {
    let network = build_network();
    let mut simulation = build_simulation(network);

    // Phase 1: drive correlated activity (B↔X) and uncorrelated (A, C).
    drive_correlated_activity(&mut simulation);

    // Phase 2: Form new connections via development.
    let plan = DevelopmentPlan::build(&simulation, &formation_config()).unwrap();
    let created = plan
        .commit(&mut simulation, &formation_config(), SynapseId(200))
        .unwrap();

    // A formation should have been created. B is the best candidate because
    // B's activity trace correlates with X's.
    assert!(
        !created.is_empty(),
        "development should form at least one synapse"
    );
    let formed_id = created[0].synapse_id;
    assert_eq!(
        created[0].source, B,
        "B should be selected as the best correlated source"
    );
    assert_eq!(created[0].target, X);

    // Phase 3: Drive more correlated activity through the new B → X synapse
    // so it accumulates utility.
    for t in [200_000u64, 300_000, 400_000, 500_000] {
        simulation
            .schedule_external_input(SimTime(t), B, 100.0)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(t + 20), X, 100.0)
            .unwrap();
    }
    simulation.run_until(SimTime(600_000)).unwrap();

    // Phase 4: Check utility. B → X should have positive utility.
    // A → X and C → X should have zero utility (their arrivals never
    // coincided with X firing in a useful way).
    let b_synapse = simulation.network().synapse(formed_id).unwrap();
    assert!(
        b_synapse.utility() > 0.0,
        "B → X should have positive utility, got {}",
        b_synapse.utility()
    );

    let a_synapse = simulation.network().synapse(A_TO_X);
    let c_synapse = simulation.network().synapse(C_TO_X);
    if let Some(a) = a_synapse {
        assert!(
            a.utility() < 0.02,
            "A → X should have low utility, got {}",
            a.utility()
        );
    }
    if let Some(c) = c_synapse {
        assert!(
            c.utility() < 0.02,
            "C → X should have low utility, got {}",
            c.utility()
        );
    }

    // Phase 5: Run a long silent period so A → X and C → X decay below
    // threshold and become pruning candidates. B → X should survive.
    // First observe to start timers on low-utility synapses.
    let mut controller = PruningController::new(pruning_config()).unwrap();
    controller.observe(&simulation).unwrap();

    // Continue running past T_min = 2_000_000 us. With τ_U = 5_000_000,
    // B → X's utility after 3s of silence is still well above threshold.
    simulation
        .schedule_external_input(SimTime(3_600_000), X, 0.0)
        .unwrap();
    simulation.run_until(SimTime(3_600_000)).unwrap();

    controller.observe(&simulation).unwrap();
    let candidates = controller.candidates(&simulation).unwrap();
    // A → X and C → X should be candidates; B → X should NOT.
    let a_is_candidate = candidates.iter().any(|c| c.synapse_id == A_TO_X);
    let c_is_candidate = candidates.iter().any(|c| c.synapse_id == C_TO_X);
    let b_is_candidate = candidates.iter().any(|c| c.synapse_id == formed_id);

    assert!(
        a_is_candidate,
        "A → X should be a pruning candidate (low utility)"
    );
    assert!(
        c_is_candidate,
        "C → X should be a pruning candidate (low utility)"
    );
    assert!(
        !b_is_candidate,
        "B → X should NOT be a pruning candidate (high utility)"
    );

    // Phase 6: Commit the pruning plan. A → X and C → X are removed.
    let mut prune_plan = DevelopmentPlan::default();
    for candidate in &candidates {
        prune_plan.prunings.push(nerva::development::PlannedPruning {
            synapse_id: candidate.synapse_id,
            target: candidate.target,
        });
    }
    prune_plan.validate(&simulation).unwrap();
    prune_plan
        .commit(&mut simulation, &formation_config(), SynapseId(300))
        .unwrap();

    // Final topology: B → X exists, A → X and C → X are gone.
    assert!(
        simulation.network().synapse(A_TO_X).is_none(),
        "A → X must be pruned"
    );
    assert!(
        simulation.network().synapse(C_TO_X).is_none(),
        "C → X must be pruned"
    );
    assert!(
        simulation.network().synapse(formed_id).is_some(),
        "B → X must survive"
    );
}