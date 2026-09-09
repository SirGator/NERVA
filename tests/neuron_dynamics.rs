use nerva::{
    config::{IntrinsicDynamicsConfig, NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime},
    learning::NoPlasticity,
    math::Position3D,
    runtime::{ObservationEvent, Simulation},
};

fn neuron_config(intrinsic: IntrinsicDynamicsConfig) -> NeuronConfig {
    NeuronConfig {
        resting_potential: 0.0,
        reset_potential: 0.0,
        threshold: 1.0,
        membrane_tau_us: 1_000.0,
        refractory_period_us: 0,
        activity_trace_tau_us: 10_000.0,
        intrinsic,
    }
}

fn neuron() -> Neuron {
    Neuron::new(
        NeuronId(1),
        Position3D::ORIGIN,
        Polarity::Excitatory,
        None,
        neuron_config(IntrinsicDynamicsConfig::default()),
        SimTime::ZERO,
    )
    .expect("valid integration-test neuron")
}

fn simulation() -> Simulation<NoPlasticity> {
    let mut network = Network::new();
    network.add_neuron(neuron()).expect("unique neuron");
    Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0).expect("valid runtime")
}

fn simulation_with_intrinsic(intrinsic: IntrinsicDynamicsConfig) -> Simulation<NoPlasticity> {
    let mut network = Network::new();
    network
        .add_neuron(
            Neuron::new(
                NeuronId(1),
                Position3D::ORIGIN,
                Polarity::Excitatory,
                None,
                neuron_config(intrinsic),
                SimTime::ZERO,
            )
            .unwrap(),
        )
        .unwrap();
    Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0).unwrap()
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1.0e-6,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn simultaneous_excitation_and_inhibition_are_integrated_atomically() {
    // If these two inputs were integrated in insertion order, +1.1 would emit a
    // spike before -0.2 arrived. Exact-timestamp batching must instead apply
    // their net +0.9 once, independently of insertion order.
    for amplitudes in [[1.1, -0.2], [-0.2, 1.1]] {
        let mut simulation = simulation();
        for amplitude in amplitudes {
            simulation
                .schedule_external_input(SimTime(25), NeuronId(1), amplitude)
                .expect("known target and finite input");
        }

        let report = simulation.run().expect("simulation succeeds");
        let neuron = simulation
            .network()
            .neuron(NeuronId(1))
            .expect("neuron remains present");

        assert_eq!(report.batches_processed, 1);
        assert_eq!(report.events_processed, 2);
        assert_eq!(report.spikes_emitted, 0);
        assert_eq!(neuron.spike_count(), 0);
        assert_close(neuron.membrane_potential(), 0.9);
        assert_close(neuron.input_trace(), 1.3);
        assert!(
            !simulation
                .event_log()
                .iter()
                .any(|event| matches!(event, ObservationEvent::SpikeEmitted(_)))
        );
    }
}

#[test]
fn sparse_runtime_events_follow_the_analytical_lif_solution() {
    let mut simulation = simulation();
    simulation
        .schedule_external_input(SimTime::ZERO, NeuronId(1), 0.8)
        .expect("valid first input");
    simulation
        .schedule_external_input(SimTime(1_000), NeuronId(1), 0.5)
        .expect("valid second input");

    let report = simulation.run().expect("simulation succeeds");
    let neuron = simulation
        .network()
        .neuron(NeuronId(1))
        .expect("neuron remains present");
    let expected = 0.8 * (-1.0_f32).exp() + 0.5;

    assert_eq!(report.batches_processed, 2);
    assert_eq!(report.spikes_emitted, 0);
    assert_eq!(neuron.last_update(), SimTime(1_000));
    assert_close(neuron.membrane_potential(), expected);
}

#[test]
fn burst_emits_an_autonomous_follow_up_spike() {
    let mut simulation = simulation_with_intrinsic(IntrinsicDynamicsConfig {
        burst_gain: 2_000.0,
        burst_tau_us: 20_000.0,
        ..IntrinsicDynamicsConfig::default()
    });
    simulation
        .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
        .unwrap();

    let report = simulation.run_until(SimTime(2_000)).unwrap();
    let spike_times: Vec<_> = simulation
        .event_log()
        .iter()
        .filter_map(|event| match event {
            ObservationEvent::SpikeEmitted(spike) => Some(spike.time),
            _ => None,
        })
        .collect();

    assert!(report.spikes_emitted >= 2, "spikes: {spike_times:?}");
    assert_eq!(spike_times[0], SimTime::ZERO);
    assert!(spike_times[1] > SimTime::ZERO);
}

#[test]
fn balanced_input_preserves_rebound_and_is_insertion_order_independent() {
    let intrinsic = IntrinsicDynamicsConfig {
        rebound_gain: 2_000.0,
        rebound_tau_us: 20_000.0,
        ..IntrinsicDynamicsConfig::default()
    };
    let mut spike_runs = Vec::new();
    for amplitudes in [[1.0, -1.0], [-1.0, 1.0]] {
        let mut simulation = simulation_with_intrinsic(intrinsic);
        for amplitude in amplitudes {
            simulation
                .schedule_external_input(SimTime::ZERO, NeuronId(1), amplitude)
                .unwrap();
        }
        simulation.run_until(SimTime(5_000)).unwrap();
        spike_runs.push(
            simulation
                .event_log()
                .iter()
                .filter_map(|event| match event {
                    ObservationEvent::SpikeEmitted(spike) => Some(spike.time),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        );
    }

    assert!(!spike_runs[0].is_empty());
    assert_eq!(spike_runs[0], spike_runs[1]);
}

#[test]
fn configured_intrinsic_drive_is_scheduled_at_runtime_construction() {
    let mut simulation = simulation_with_intrinsic(IntrinsicDynamicsConfig {
        intrinsic_drive: 2_000.0,
        ..IntrinsicDynamicsConfig::default()
    });

    assert_eq!(simulation.pending_event_count(), 1);
    let report = simulation.run_until(SimTime(1_000)).unwrap();
    assert_eq!(report.spikes_emitted, 1);
}
