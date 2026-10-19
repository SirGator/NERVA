use nerva::{
    config::{LearningConfig, NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    experiment::{M0Experiment, M0ExperimentConfig, M0Group},
    learning::PairStdp,
    math::Position3D,
    primitives::Weight,
    runtime::{ProfilePhase, Simulation},
};

fn simulation(profile: bool) -> Simulation<PairStdp> {
    let mut network = Network::new();
    for id in [1, 2] {
        network
            .add_neuron(
                Neuron::new(
                    NeuronId(id),
                    Position3D::ORIGIN,
                    Polarity::Excitatory,
                    None,
                    NeuronConfig {
                        resting_potential: 0.0,
                        reset_potential: 0.0,
                        threshold: 1.0,
                        refractory_period_us: 0,
                        ..Default::default()
                    },
                    SimTime::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
    }
    network
        .add_synapse(
            Synapse::new(
                SynapseId(1),
                NeuronId(1),
                NeuronId(2),
                Weight::new(0.5).unwrap(),
                5,
                true,
            )
            .unwrap(),
        )
        .unwrap();
    let mut simulation = Simulation::new(
        network,
        PairStdp::new(&LearningConfig::default()),
        RuntimeConfig::default(),
        1.0,
    )
    .unwrap();
    if profile {
        simulation.enable_profiling();
    }
    for (at, target, amplitude) in [(0, 1, 1.1), (5, 2, 0.6), (20, 1, 0.0)] {
        simulation
            .schedule_external_input(SimTime(at), NeuronId(target), amplitude)
            .unwrap();
    }
    simulation
}

#[test]
fn profiling_counts_real_work_and_does_not_change_replay() {
    let mut measured = simulation(true);
    let mut reference = simulation(false);
    assert_eq!(measured.run().unwrap(), reference.run().unwrap());
    assert_eq!(measured.network(), reference.network());
    assert_eq!(measured.event_log(), reference.event_log());
    assert_eq!(
        measured.pending_event_count(),
        reference.pending_event_count()
    );
    assert!(reference.profile().is_none());
    let profile = measured.profile().unwrap();
    assert_eq!(
        (
            profile.batches_started,
            profile.batches_completed,
            profile.batches_failed
        ),
        (3, 3, 0)
    );
    assert_eq!(profile.advance_batch_size.bins, [0, 3, 0, 0, 0, 0]);
    assert_eq!(profile.scheduler_batch_size.bins, [0, 3, 0, 0, 0, 0]);
    assert_eq!(profile.scheduler_batch_size.total, 4);
    assert_eq!(profile.scheduler_batch_size.singletons, 2);
    // run() makes one final empty dequeue attempt.
    assert_eq!(profile.phase(ProfilePhase::SchedulerDequeue).calls, 4);
    assert_eq!(profile.phase(ProfilePhase::SchedulerDequeue).failures, 0);
    assert_eq!(profile.integrate_batch_size.total, 3);
    assert_eq!(profile.inputs_per_neuron.total, 4);
    assert_eq!(profile.inputs_per_neuron.samples, 3);
    assert_eq!(
        profile.synaptic_arrivals_per_timestamp.bins,
        [2, 1, 0, 0, 0, 0]
    );
    assert_eq!(profile.spikes_per_timestamp.bins, [1, 2, 0, 0, 0, 0]);
    assert_eq!(profile.outgoing_synapses_per_spike.bins, [1, 1, 0, 0, 0, 0]);
    assert_eq!(profile.phase(ProfilePhase::PreStdp).calls, 1);
    assert_eq!(profile.phase(ProfilePhase::AdvanceCompute).calls, 3);
    assert_eq!(profile.phase(ProfilePhase::PropagationPlan).calls, 2);
    assert_eq!(profile.phase(ProfilePhase::PropagationEnqueue).calls, 2);
    assert!(
        profile.phase(ProfilePhase::Propagation).total
            >= profile.phase(ProfilePhase::PropagationPlan).total
                + profile.phase(ProfilePhase::PropagationEnqueue).total
    );
    let profile = measured.take_profile().unwrap();
    assert!(measured.profile().is_none());
    assert!(measured.step().unwrap().is_none());
    let mut merged = profile.clone();
    merged.merge(&profile);
    assert_eq!(merged.batches_started, 6);
    assert_eq!(merged.inputs_per_neuron.total, 8);
    assert_eq!(merged.scheduler_batch_size.total, 8);
    assert_eq!(merged.scheduler_batch_size.singletons, 4);
    assert_eq!(merged.phase(ProfilePhase::SchedulerDequeue).calls, 8);
    assert_eq!(merged.phase(ProfilePhase::PropagationPlan).calls, 4);
    assert_eq!(merged.phase(ProfilePhase::PropagationEnqueue).calls, 4);
    assert_eq!(
        merged.phase(ProfilePhase::AdvanceCompute).total,
        profile
            .phase(ProfilePhase::AdvanceCompute)
            .total
            .saturating_mul(2)
    );
}

#[test]
fn scheduler_limit_failure_is_measured_without_consuming_a_batch() {
    let mut network = Network::new();
    network
        .add_neuron(
            Neuron::new(
                NeuronId(1),
                Position3D::ORIGIN,
                Polarity::Excitatory,
                None,
                NeuronConfig::default(),
                SimTime::ZERO,
            )
            .unwrap(),
        )
        .unwrap();
    let mut simulation = Simulation::new(
        network,
        PairStdp::new(&LearningConfig::default()),
        RuntimeConfig {
            max_events_per_batch: 1,
        },
        1.0,
    )
    .unwrap()
    .with_profiling();
    for _ in 0..2 {
        simulation
            .schedule_external_input(SimTime(10), NeuronId(1), 0.1)
            .unwrap();
    }
    assert!(simulation.step().is_err());
    assert_eq!(simulation.pending_event_count(), 2);
    let profile = simulation.profile().unwrap();
    assert_eq!(profile.phase(ProfilePhase::SchedulerDequeue).calls, 1);
    assert_eq!(profile.phase(ProfilePhase::SchedulerDequeue).failures, 1);
    assert_eq!(profile.scheduler_batch_size.samples, 0);
    assert_eq!(profile.batches_started, 0);
}

#[test]
fn failed_attempts_remain_visible_without_completed_spike_samples() {
    let mut simulation = simulation(true);
    for _ in 0..2 {
        simulation
            .schedule_external_input(SimTime(1), NeuronId(1), f32::MAX)
            .unwrap();
    }
    assert!(simulation.run().is_err());
    let profile = simulation.profile().unwrap();
    assert_eq!(
        (
            profile.batches_started,
            profile.batches_completed,
            profile.batches_failed
        ),
        (2, 1, 1)
    );
    assert_eq!(profile.phase(ProfilePhase::IntegrateCompute).failures, 1);
    assert_eq!(profile.spikes_per_timestamp.samples, 1);
}

#[test]
fn profiled_m0_keeps_deterministic_results_and_includes_both_probes() {
    for group in [
        M0Group::OrderedLearning,
        M0Group::OrderedFixed,
        M0Group::RandomLearning,
        M0Group::OrderedHomeostasis,
    ] {
        let experiment = M0Experiment::new(M0ExperimentConfig::default(), group).unwrap();
        let reference = experiment.run().unwrap();
        let (measured, profile) = experiment.run_profiled().unwrap();
        assert_eq!(measured, reference);
        assert_eq!(profile.batches_failed, 0);
        assert_eq!(
            profile.advance_batch_size.samples,
            profile.batches_completed
        );
        assert!(profile.batches_completed > 0);
    }
}
