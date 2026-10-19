use nerva::{
    compute::{ComputeBackend, ComputeError, CpuCompute, DenseNeuronBatch, FlatNeuronInputs},
    config::{IntrinsicDynamicsConfig, NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronError, NeuronId, NeuronRole, Polarity, SimTime, Spike},
    learning::NoPlasticity,
    math::Position3D,
    runtime::{ObservationEvent, Simulation, SimulationError},
};

fn cell(id: u64, intrinsic: bool) -> Neuron {
    Neuron::new(
        NeuronId(id),
        Position3D::new(id as f32, 2.0, 3.0),
        Polarity::Inhibitory,
        Some(NeuronRole::Motor),
        NeuronConfig {
            resting_potential: 0.0,
            reset_potential: -0.1,
            threshold: 1.0,
            membrane_tau_us: 1_000.0,
            refractory_period_us: 0,
            activity_trace_tau_us: 5_000.0,
            intrinsic: if intrinsic {
                IntrinsicDynamicsConfig {
                    intrinsic_drive: 120.0,
                    burst_gain: 200.0,
                    adaptation_gain: 100.0,
                    threshold_adaptation_gain: 0.05,
                    rebound_gain: 100.0,
                    ..Default::default()
                }
            } else {
                Default::default()
            },
        },
        SimTime::ZERO,
    )
    .unwrap()
}

fn network() -> Network {
    let mut network = Network::new();
    for id in [91, 2, 17] {
        network.add_neuron(cell(id, false)).unwrap();
    }
    network
}

#[test]
fn numerical_round_trip_preserves_all_state_without_changing_metadata() {
    let mut original = cell(17, true);
    original.set_threshold(1.2).unwrap();
    original.set_intrinsic_current(-5.0).unwrap();
    original
        .set_structural_drive_clamped(4.0, 0.0, 10.0)
        .unwrap();
    original
        .integrate_input_with_components(SimTime(100), 1.4, 2.0, 0.3)
        .unwrap();
    original.advance_to(SimTime(150)).unwrap();
    original.record_homeostasis_update(SimTime(150)).unwrap();
    let state = original.numerical_state();
    assert_eq!(state.last_spike_time, 100);
    assert_eq!(state.last_spike_valid, 1);
    assert!(state.burst_drive > 0.0 && state.adaptation_drive > 0.0 && state.rebound_drive > 0.0);
    assert!(
        state.threshold_adaptation > 0.0 && state.activity_trace > 0.0 && state.input_trace > 0.0
    );
    let mut restored = cell(17, true);
    restored.apply_numerical_state(state).unwrap();
    // Neuron equality covers every original private field, including metadata.
    assert_eq!(restored, original);
    assert_eq!(restored.numerical_params().threshold, 1.0);
    assert_eq!(restored.base_threshold(), 1.2);
    assert_eq!(restored.numerical_params().rebound_gain, 100.0);

    let mut changed = restored.numerical_state();
    changed.last_spike_valid = 2;
    assert!(restored.apply_numerical_state(changed).is_err());
    assert_eq!(restored, original);
    changed = restored.numerical_state();
    changed.last_update = SimTime(149);
    assert!(matches!(
        restored.apply_numerical_state(changed),
        Err(NeuronError::TimeWentBackwards { .. })
    ));
    assert_eq!(restored, original);
}

#[test]
fn optional_spikes_and_large_timestamps_round_trip_without_sentinels_or_rounding() {
    for at in [SimTime::ZERO, SimTime((1_u64 << 53) + 7), SimTime(u64::MAX)] {
        let mut network = Network::new();
        network.add_neuron(cell(17, false)).unwrap();
        assert_eq!(
            network
                .neuron(NeuronId(17))
                .unwrap()
                .numerical_state()
                .last_spike_valid,
            0
        );
        let mut state = DenseNeuronBatch::from_network(&network, &[NeuronId(17)]).unwrap();
        CpuCompute.advance_neurons(&mut state, at).unwrap();
        state.write_back(&mut network).unwrap();
        let mut state = DenseNeuronBatch::from_network(&network, &[NeuronId(17)]).unwrap();
        let mut inputs = FlatNeuronInputs::new(vec![NeuronId(17)], vec![0, 1], vec![1.1]).unwrap();
        assert_eq!(
            CpuCompute
                .integrate_neurons(&mut state, &mut inputs, at)
                .unwrap(),
            vec![Spike::new(NeuronId(17), at)]
        );
        assert_eq!(state.states()[0].last_spike_valid, 1);
        assert_eq!(state.states()[0].last_spike_time, at.as_micros());
        state.write_back(&mut network).unwrap();
        assert_eq!(network.neuron(NeuronId(17)).unwrap().last_spike(), Some(at));
    }
}

#[test]
fn flat_inputs_validate_layout_and_preserve_empty_autonomous_rows() {
    let mut flat = FlatNeuronInputs::from_rows([
        (NeuronId(4), vec![0.2, 0.7, -0.3]),
        (NeuronId(8), vec![]),
        (NeuronId(15), vec![0.8, 0.2]),
    ])
    .unwrap();
    assert_eq!(flat.neuron_ids(), &[NeuronId(4), NeuronId(8), NeuronId(15)]);
    assert_eq!(flat.offsets(), &[0, 3, 3, 5]);
    assert_eq!(flat.amplitudes(), &[0.2, 0.7, -0.3, 0.8, 0.2]);
    assert_eq!(flat.row(1), Some([].as_slice()));
    flat.row_mut(0).unwrap().reverse();
    assert_eq!(flat.row(0), Some([-0.3, 0.7, 0.2].as_slice()));
    assert_eq!(flat.row(usize::MAX), None);
    assert_eq!(flat.row_mut(3), None);
    assert_eq!(FlatNeuronInputs::from_rows([]).unwrap().offsets(), &[0]);
    for (ids, offsets, values) in [
        (vec![NeuronId(2)], vec![0], vec![]),
        (vec![NeuronId(2)], vec![1, 1], vec![1.0]),
        (vec![NeuronId(2)], vec![0, 2], vec![1.0]),
        (vec![NeuronId(2), NeuronId(17)], vec![0, 2, 1], vec![1.0]),
        (
            vec![NeuronId(2), NeuronId(17)],
            vec![0, u64::MAX, 1],
            vec![1.0],
        ),
        (vec![NeuronId(2), NeuronId(2)], vec![0, 0, 0], vec![]),
        (vec![NeuronId(17), NeuronId(2)], vec![0, 0, 0], vec![]),
    ] {
        assert!(matches!(
            FlatNeuronInputs::new(ids, offsets, values),
            Err(ComputeError::InvalidBatch { .. })
        ));
    }
}

#[test]
fn invalid_or_stale_dense_results_never_overwrite_authoritative_state() {
    let mut network = network();
    assert!(DenseNeuronBatch::from_network(&network, &[NeuronId(17), NeuronId(2)]).is_err());
    assert!(DenseNeuronBatch::from_network(&network, &[NeuronId(2), NeuronId(2)]).is_err());
    assert!(matches!(
        DenseNeuronBatch::from_network(&network, &[NeuronId(999)]),
        Err(ComputeError::UnknownNeuron(NeuronId(999)))
    ));
    let original = network.clone();
    let mut batch = DenseNeuronBatch::from_network(&network, &[NeuronId(2), NeuronId(17)]).unwrap();
    batch.states_mut()[0].membrane_potential = 0.5;
    batch.states_mut()[1].activity_trace = f32::NAN;
    assert!(batch.write_back(&mut network).is_err());
    assert_eq!(network, original);

    // Signed zero is part of the CPU bit pattern and must count as a live edit.
    let batch = DenseNeuronBatch::from_network(&network, &[NeuronId(2)]).unwrap();
    network
        .neuron_mut(NeuronId(2))
        .unwrap()
        .set_intrinsic_current(-0.0)
        .unwrap();
    assert!(matches!(
        batch.write_back(&mut network),
        Err(ComputeError::InvalidBatch { .. })
    ));
    assert_eq!(
        network
            .neuron(NeuronId(2))
            .unwrap()
            .intrinsic_current()
            .to_bits(),
        (-0.0_f32).to_bits()
    );

    let batch = DenseNeuronBatch::from_network(&network, &[NeuronId(2), NeuronId(17)]).unwrap();
    network.remove_neuron(NeuronId(17)).unwrap();
    let before = network.clone();
    assert!(matches!(
        batch.write_back(&mut network),
        Err(ComputeError::UnknownNeuron(NeuronId(17)))
    ));
    assert_eq!(network, before);

    let batch = DenseNeuronBatch::from_network(&network, &[NeuronId(2)]).unwrap();
    network.remove_neuron(NeuronId(2)).unwrap();
    network.add_neuron(cell(2, true)).unwrap();
    let before = network.clone();
    assert!(matches!(
        batch.write_back(&mut network),
        Err(ComputeError::InvalidBatch { .. })
    ));
    assert_eq!(network, before);
}

#[test]
fn compute_rejects_mismatched_input_ids_without_changing_dense_state() {
    let network = network();
    let mut state = DenseNeuronBatch::from_network(&network, &[NeuronId(2)]).unwrap();
    let original = state.clone();
    let mut inputs = FlatNeuronInputs::from_rows([(NeuronId(17), vec![1.1])]).unwrap();
    assert!(matches!(
        CpuCompute.integrate_neurons(&mut state, &mut inputs, SimTime(10)),
        Err(ComputeError::InvalidBatch { .. })
    ));
    assert_eq!(state, original);
}

#[test]
fn dense_failure_publishes_the_same_partial_cells_before_poisoning() {
    let mut simulation =
        Simulation::new(network(), NoPlasticity, RuntimeConfig::default(), 1.0).unwrap();
    for (id, input) in [(2, 1.1), (17, f32::MAX), (17, f32::MAX), (91, 1.1)] {
        simulation
            .schedule_external_input(SimTime(10), NeuronId(id), input)
            .unwrap();
    }
    assert!(matches!(
        simulation.step(),
        Err(SimulationError::SimulationPoisoned { .. })
    ));
    assert_eq!(
        simulation.poison_reason(),
        Some(&SimulationError::NonFiniteSummedInput {
            target: NeuronId(17)
        })
    );
    assert_eq!(
        simulation
            .network()
            .neuron(NeuronId(2))
            .unwrap()
            .spike_count(),
        1
    );
    for id in [17, 91] {
        let neuron = simulation.network().neuron(NeuronId(id)).unwrap();
        assert_eq!(neuron.last_update(), SimTime(10));
        assert_eq!(neuron.spike_count(), 0);
        assert_eq!(neuron.input_trace(), 0.0);
    }
    // The original runtime logged spikes only after the entire integration phase.
    assert!(
        !simulation
            .event_log()
            .iter()
            .any(|event| matches!(event, ObservationEvent::SpikeEmitted(_)))
    );

    let mut network = network();
    network
        .neuron_mut(NeuronId(17))
        .unwrap()
        .advance_to(SimTime(100))
        .unwrap();
    let mut simulation =
        Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0).unwrap();
    for id in [2, 17] {
        simulation
            .schedule_external_input(SimTime(10), NeuronId(id), 1.1)
            .unwrap();
    }
    assert!(simulation.step().is_err());
    assert_eq!(
        simulation
            .network()
            .neuron(NeuronId(2))
            .unwrap()
            .last_update(),
        SimTime(10)
    );
    assert_eq!(
        simulation
            .network()
            .neuron(NeuronId(17))
            .unwrap()
            .last_update(),
        SimTime(100)
    );
    assert!(simulation.event_log().is_empty());
}

struct InvalidResult;

impl ComputeBackend for InvalidResult {
    fn advance_neurons(
        &mut self,
        state: &mut DenseNeuronBatch,
        _time: SimTime,
    ) -> Result<(), ComputeError> {
        state.states_mut()[0].activity_trace = f32::NAN;
        Ok(())
    }
    fn integrate_neurons(
        &mut self,
        _state: &mut DenseNeuronBatch,
        _inputs: &mut FlatNeuronInputs,
        _time: SimTime,
    ) -> Result<Vec<Spike>, ComputeError> {
        panic!("invalid advancement must stop before integration")
    }
}

#[test]
fn invalid_successful_backend_result_is_rejected_before_publication() {
    let network = network();
    let original = network.clone();
    let mut simulation = Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0)
        .unwrap()
        .with_compute_backend(InvalidResult);
    simulation
        .schedule_external_input(SimTime(10), NeuronId(2), 1.1)
        .unwrap();
    assert!(simulation.step().is_err());
    assert_eq!(
        simulation.poison_reason(),
        Some(&SimulationError::Neuron(
            NeuronError::InvalidNumericalState {
                field: "activity_trace"
            }
        ))
    );
    assert_eq!(simulation.network(), &original);
}
