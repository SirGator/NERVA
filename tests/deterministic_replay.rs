use nerva::{
    config::{LearningConfig, NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    learning::{DecayingTrace, PairStdp},
    math::Position3D,
    primitives::Weight,
    runtime::{EventLog, ObservationEvent, RunReport, Simulation},
};

const REPLAY_SEED: u64 = 0xd571_5eed;

struct ReplayOutcome {
    report: RunReport,
    network: Network,
    post_traces: Vec<Option<DecayingTrace>>,
    homeostasis_enabled: bool,
    pending_events: usize,
    event_log: EventLog,
}

fn neuron(id: u64) -> Neuron {
    Neuron::new(
        NeuronId(id),
        Position3D::ORIGIN,
        Polarity::Excitatory,
        None,
        NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            membrane_tau_us: 500.0,
            refractory_period_us: 0,
            activity_trace_tau_us: 5_000.0,
            intrinsic: Default::default(),
        },
        SimTime::ZERO,
    )
    .expect("valid replay neuron")
}

fn replay(seed: u64, reverse_synapse_insertion: bool) -> ReplayOutcome {
    let mut network = Network::new();
    for id in 1..=3 {
        network.add_neuron(neuron(id)).expect("unique neuron");
    }

    let mut synapses = vec![
        Synapse::new(
            SynapseId(11),
            NeuronId(1),
            NeuronId(2),
            Weight::new(0.55).unwrap(),
            7,
            true,
        )
        .expect("valid first synapse"),
        Synapse::new(
            SynapseId(12),
            NeuronId(2),
            NeuronId(3),
            Weight::new(0.65).unwrap(),
            5,
            true,
        )
        .expect("valid second synapse"),
        Synapse::new(
            SynapseId(13),
            NeuronId(3),
            NeuronId(1),
            Weight::new(0.20).unwrap(),
            9,
            false,
        )
        .expect("valid fixed synapse"),
    ];
    if reverse_synapse_insertion {
        synapses.reverse();
    }
    for synapse in synapses {
        network.add_synapse(synapse).expect("unique synapse");
    }

    let learning = LearningConfig {
        enabled: true,
        a_plus: 0.08,
        a_minus: 0.09,
        tau_plus_us: 100.0,
        tau_minus_us: 120.0,
        stdp_window_us: 200,
        min_weight: 0.1,
        max_weight: 0.9,
        ..LearningConfig::default()
    };
    let rule = PairStdp::try_from_config(&learning).expect("valid learning parameters");
    let mut simulation = Simulation::new(network, rule, RuntimeConfig::default(), 1.0)
        .expect("valid replay runtime");

    // The seed affects timestamps, while scheduling deliberately does not use
    // chronological insertion order. Replaying the same seed must nevertheless
    // produce the same exact batches and floating-point state.
    let offset = seed % 5;
    for (time, target, amplitude) in [
        (20 + offset, NeuronId(3), 0.4),
        (9 + offset, NeuronId(2), 0.5),
        (offset, NeuronId(1), 0.65),
        (16 + offset, NeuronId(3), 0.4),
        (offset, NeuronId(1), 0.45),
    ] {
        simulation
            .schedule_external_input(SimTime(time), target, amplitude)
            .expect("valid replay stimulus");
    }

    let report = simulation.run().expect("replay succeeds");
    let pending_events = simulation.pending_event_count();
    let (network, rule, homeostasis, event_log) = simulation.into_parts();
    let post_traces = [NeuronId(1), NeuronId(2), NeuronId(3)]
        .into_iter()
        .map(|id| rule.post_trace(id).copied())
        .collect();

    ReplayOutcome {
        report,
        network,
        post_traces,
        homeostasis_enabled: homeostasis.is_enabled(),
        pending_events,
        event_log,
    }
}

#[test]
fn same_seed_replays_identical_event_log_and_complete_end_state() {
    let first = replay(REPLAY_SEED, false);
    let second = replay(REPLAY_SEED, true);

    assert!(
        first
            .event_log
            .events()
            .iter()
            .any(|event| matches!(event, ObservationEvent::WeightChanged { .. }))
    );
    assert_eq!(first.report, second.report);
    assert_eq!(first.event_log, second.event_log);
    assert_eq!(first.network, second.network);
    assert_eq!(first.post_traces, second.post_traces);
    assert_eq!(first.homeostasis_enabled, second.homeostasis_enabled);
    assert_eq!(first.pending_events, 0);
    assert_eq!(second.pending_events, 0);
    first
        .network
        .validate_indices()
        .expect("first replay preserves graph indices");
    second
        .network
        .validate_indices()
        .expect("second replay preserves graph indices");
}
