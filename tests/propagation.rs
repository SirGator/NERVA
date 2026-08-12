use dsvlm_rust::{
    config::{NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    learning::NoPlasticity,
    math::Position3D,
    runtime::{ObservationEvent, Simulation},
};

fn neuron(id: u64, x: f32, threshold: f32) -> Neuron {
    Neuron::new(
        NeuronId(id),
        Position3D::new(x, 0.0, 0.0),
        Polarity::Excitatory,
        None,
        NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold,
            membrane_tau_us: 10_000.0,
            refractory_period_us: 0,
            activity_trace_tau_us: 10_000.0,
        },
        SimTime::ZERO,
    )
    .expect("valid integration-test neuron")
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1.0e-6,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn emitted_spike_arrives_only_after_delay_with_distance_attenuation() {
    let source = NeuronId(1);
    let target = NeuronId(2);
    let synapse_id = SynapseId(9);
    let mut network = Network::new();
    network
        .add_neuron(neuron(source.get(), 0.0, 1.0))
        .expect("unique source");
    network
        .add_neuron(neuron(target.get(), 2.0, 2.0))
        .expect("unique target");
    network
        .add_synapse(
            Synapse::new(synapse_id, source, target, 2.0, 7, false).expect("valid connection"),
        )
        .expect("unique synapse");

    let mut simulation = Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 2.0)
        .expect("valid runtime");
    simulation
        .schedule_external_input(SimTime(11), source, 1.0)
        .expect("valid source stimulus");

    let before_arrival = simulation
        .run_until(SimTime(17))
        .expect("source spike succeeds");
    assert_eq!(before_arrival.spikes_emitted, 1);
    assert_eq!(simulation.pending_event_count(), 1);
    assert_eq!(
        simulation
            .network()
            .neuron(target)
            .expect("target exists")
            .membrane_potential(),
        0.0
    );

    let arrival_report = simulation.run_until(SimTime(18)).expect("arrival succeeds");
    let expected_amplitude = 2.0 * (-1.0_f32).exp();

    assert_eq!(arrival_report.batches_processed, 1);
    assert_eq!(arrival_report.events_processed, 1);
    assert_eq!(arrival_report.spikes_emitted, 0);
    assert_eq!(simulation.pending_event_count(), 0);
    assert_close(
        simulation
            .network()
            .neuron(target)
            .expect("target exists")
            .membrane_potential(),
        expected_amplitude,
    );
    assert_eq!(
        simulation
            .network()
            .synapse(synapse_id)
            .expect("synapse exists")
            .transmission_count(),
        1
    );

    let arrival = simulation.event_log().iter().find_map(|event| match event {
        ObservationEvent::SynapticArrival {
            time,
            synapse_id: observed_synapse,
            source: observed_source,
            target: observed_target,
            amplitude,
        } => Some((
            *time,
            *observed_synapse,
            *observed_source,
            *observed_target,
            *amplitude,
        )),
        _ => None,
    });
    let (time, observed_synapse, observed_source, observed_target, amplitude) =
        arrival.expect("propagation must be observable");
    assert_eq!(time, SimTime(18));
    assert_eq!(observed_synapse, synapse_id);
    assert_eq!(observed_source, source);
    assert_eq!(observed_target, target);
    assert_close(amplitude, expected_amplitude);
}
