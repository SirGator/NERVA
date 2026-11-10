use crate::{
    config::{LearningConfig, NeuronConfig, RuntimeConfig},
    core::{Event, EventKind, Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    learning::{CompetitionEvent, PairStdp},
    math::Position3D,
    primitives::Weight,
    runtime::{ObservationEvent, Simulation},
};

fn runtime() -> Simulation<PairStdp> {
    let mut net = Network::new();
    for id in 1..=4 {
        net.add_neuron(
            Neuron::new(
                NeuronId(id),
                Position3D::ORIGIN,
                Polarity::Excitatory,
                None,
                NeuronConfig {
                    resting_potential: 0.0,
                    reset_potential: 0.0,
                    threshold: 1.0,
                    membrane_tau_us: 1000.0,
                    refractory_period_us: 100,
                    activity_trace_tau_us: 1000.0,
                    intrinsic: Default::default(),
                },
                SimTime::ZERO,
            )
            .unwrap(),
        )
        .unwrap();
    }
    for (id, pre, post, w) in [(1, 1, 3, 0.8), (2, 2, 3, 0.4), (3, 3, 4, 0.5)] {
        net.add_synapse(
            Synapse::new(
                SynapseId(id),
                NeuronId(pre),
                NeuronId(post),
                Weight::new(w).unwrap(),
                1,
                true,
            )
            .unwrap(),
        )
        .unwrap();
    }
    let rule = PairStdp::new(&LearningConfig {
        a_plus: 0.3,
        a_minus: 0.2,
        tau_plus_us: 100.0,
        tau_minus_us: 100.0,
        stdp_window_us: 1000,
        max_weight: 2.0,
        ..Default::default()
    });
    Simulation::new(net, rule, RuntimeConfig::default(), 1.0)
        .unwrap()
        .with_incoming_competition(&[NeuronId(1), NeuronId(2), NeuronId(3)])
        .unwrap()
}
fn arrival(sim: &mut Simulation<PairStdp>, id: u64, time: u64) {
    sim.schedule_event(Event::new(
        SimTime(time),
        EventKind::SynapticArrival {
            synapse_id: SynapseId(id),
            target: NeuronId(3),
            amplitude: 0.0,
        },
    ))
    .unwrap();
}
fn weights(sim: &Simulation<PairStdp>) -> Vec<f32> {
    sim.network().synapses().map(|s| s.weight().get()).collect()
}
#[test]
fn simultaneous_hooks_normalize_once_before_emission_and_preserve_queued_amplitudes() {
    let mut sim = runtime();
    let initial = weights(&sim);
    arrival(&mut sim, 1, 10);
    arrival(&mut sim, 2, 10);
    sim.schedule_external_input(SimTime(10), NeuronId(3), 1.1)
        .unwrap();
    sim.schedule_external_input(SimTime(10), NeuronId(1), 1.1)
        .unwrap();
    sim.run_until(SimTime(10)).unwrap();
    let current = weights(&sim);
    assert!(
        (f64::from(current[0]) + f64::from(current[1])
            - f64::from(initial[0])
            - f64::from(initial[1]))
        .abs()
            < 1e-6
    );
    assert_eq!(current[2], initial[2], "outside-population edge changed");
    let records = sim.drain_competition_events();
    assert_eq!(
        records
            .iter()
            .filter(|e| matches!(e, CompetitionEvent::Normalized { .. }))
            .count(),
        1
    );
    match records.last().unwrap() {
        CompetitionEvent::Normalized { before, .. } => {
            assert!((before[0] - 1.1).abs() < 1e-6);
            assert!((before[1] - 0.7).abs() < 1e-6);
        }
        _ => panic!("normalization did not follow both hooks"),
    }
    // Freezing and a diagnostic weight edit cannot rewrite an already emitted input.
    sim.set_learning_enabled(false);
    sim.update_synapse(SynapseId(1), |s| s.set_weight(Weight::new(0.1).unwrap()))
        .unwrap();
    sim.run_until(SimTime(11)).unwrap();
    assert!(sim.event_log().iter().any(|e| matches!(e,
        ObservationEvent::SynapticArrival { time,synapse_id,amplitude,.. }
        if *time==SimTime(11) && *synapse_id==SynapseId(1) && *amplitude==current[0])));
    assert!(sim.drain_competition_events().is_empty());
}
#[test]
fn ltd_without_post_spike_normalizes_and_freeze_prevents_all_changes() {
    let mut sim = runtime();
    sim.schedule_external_input(SimTime(5), NeuronId(3), 1.1)
        .unwrap();
    sim.run_until(SimTime(5)).unwrap();
    sim.drain_competition_events();
    let initial = weights(&sim);
    arrival(&mut sim, 1, 10);
    sim.run_until(SimTime(10)).unwrap();
    let after = weights(&sim);
    assert!(after[0] < initial[0] && after[1] > initial[1]);
    assert!((after[0] + after[1] - initial[0] - initial[1]).abs() < 1e-6);
    assert_eq!(
        sim.drain_competition_events()
            .iter()
            .filter(|e| matches!(e, CompetitionEvent::Normalized { .. }))
            .count(),
        1
    );
    sim.set_learning_enabled(false);
    arrival(&mut sim, 1, 20);
    sim.run_until(SimTime(20)).unwrap();
    assert_eq!(weights(&sim), after);
    assert!(sim.drain_competition_events().is_empty());
}
#[test]
fn changed_membership_fails_and_competition_state_participates_in_replay_checks() {
    let mut a = runtime();
    let b = runtime();
    assert!(a.same_replay_state(&b));
    a.update_synapse(SynapseId(1), |s| {
        s.set_enabled(false);
        Ok(())
    })
    .unwrap();
    a.schedule_external_input(SimTime(10), NeuronId(3), 0.0)
        .unwrap();
    assert!(a.run_until(SimTime(10)).is_err());
    assert!(!a.same_replay_state(&b));
}
