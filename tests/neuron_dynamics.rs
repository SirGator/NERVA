use dsvlm_rust::{
    config::{NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime},
    learning::NoPlasticity,
    math::Position3D,
    runtime::{ObservationEvent, Simulation},
};

fn neuron() -> Neuron {
    Neuron::new(
        NeuronId(1),
        Position3D::ORIGIN,
        Polarity::Excitatory,
        None,
        NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            membrane_tau_us: 1_000.0,
            refractory_period_us: 0,
            activity_trace_tau_us: 10_000.0,
        },
        SimTime::ZERO,
    )
    .expect("valid integration-test neuron")
}

fn simulation() -> Simulation<NoPlasticity> {
    let mut network = Network::new();
    network.add_neuron(neuron()).expect("unique neuron");
    Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0).expect("valid runtime")
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
