#![cfg(feature = "cuda")]

use nerva::{
    compute::{
        ComputeBackend, ComputeNeuronState, CpuCompute, CudaCompute, DenseNeuronBatch,
        FlatNeuronInputs,
        cuda::{ABS_TOLERANCE, REL_TOLERANCE},
    },
    config::{IntrinsicDynamicsConfig, LearningConfig, NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    learning::{ExcitatoryCausalUtility, PairStdp},
    math::Position3D,
    primitives::Weight,
    runtime::Simulation,
};

fn floats(s: &ComputeNeuronState) -> [f32; 10] {
    [
        s.membrane_potential,
        s.activity_trace,
        s.input_trace,
        s.threshold,
        s.burst_drive,
        s.adaptation_drive,
        s.rebound_drive,
        s.threshold_adaptation,
        s.intrinsic_current,
        s.structural_drive,
    ]
}

fn close(cpu: &[ComputeNeuronState], gpu: &[ComputeNeuronState]) -> f32 {
    assert_eq!(cpu.len(), gpu.len());
    let mut max_error = 0.0_f32;
    for (row, (cpu, gpu)) in cpu.iter().zip(gpu).enumerate() {
        assert_eq!(
            (
                cpu.last_update,
                cpu.refractory_until,
                cpu.last_spike_time,
                cpu.last_spike_valid,
                cpu.spike_count,
                cpu.last_homeostasis_update
            ),
            (
                gpu.last_update,
                gpu.refractory_until,
                gpu.last_spike_time,
                gpu.last_spike_valid,
                gpu.spike_count,
                gpu.last_homeostasis_update
            ),
            "discrete state row {row}"
        );
        for (column, (a, b)) in floats(cpu).into_iter().zip(floats(gpu)).enumerate() {
            assert!(a.is_finite() && b.is_finite());
            let error = (a - b).abs();
            max_error = max_error.max(error);
            assert!(
                error <= ABS_TOLERANCE + REL_TOLERANCE * a.abs().max(b.abs()),
                "row={row} field={column} CPU={a} GPU={b} error={error}"
            );
        }
    }
    max_error
}

struct Random(u64);
impl Random {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / 16777216.0
    }
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored --nocapture"]
fn thousands_of_seeded_states_match_cpu_and_replay_on_cuda() {
    let mut gpu =
        CudaCompute::new(0).expect("GPU validation requires working CUDA, never skip silently");
    println!("device={}", gpu.device_name());
    let start = SimTime((1_u64 << 53) + 17);
    let at = start.checked_add_us(777).unwrap();
    let mut random = Random(0xc0da_5eed);
    let mut network = Network::new();
    let mut rows = Vec::new();
    for id in 0..8_192_u64 {
        let tau = 200.0 + random.next() * 20_000.0;
        let config = NeuronConfig {
            resting_potential: 0.0,
            reset_potential: -0.1,
            threshold: 1.0 + random.next(),
            membrane_tau_us: tau,
            activity_trace_tau_us: 500.0 + random.next() * 100_000.0,
            refractory_period_us: (random.next() * 100.0) as u64,
            intrinsic: if id % 2 == 0 {
                IntrinsicDynamicsConfig {
                    intrinsic_drive: random.next() * 200.0,
                    burst_gain: random.next() * 200.0,
                    adaptation_gain: random.next() * 100.0,
                    threshold_adaptation_gain: random.next() * 0.2,
                    rebound_gain: random.next() * 100.0,
                    burst_tau_us: if id % 3 == 0 { tau } else { tau * 1.7 },
                    adaptation_tau_us: tau * 0.7,
                    rebound_tau_us: tau * 2.0,
                    ..Default::default()
                }
            } else {
                Default::default()
            },
        };
        let mut cell = Neuron::new(
            NeuronId(id * 17 + 2),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            None,
            config,
            start,
        )
        .unwrap();
        let mut state = cell.numerical_state();
        state.membrane_potential = random.next() * 1.5 - 0.5;
        state.threshold += 0.1;
        state.activity_trace = random.next() * 3.0;
        state.input_trace = random.next() * 5.0;
        state.burst_drive = random.next() * 100.0;
        state.adaptation_drive = random.next() * 50.0;
        state.rebound_drive = random.next() * 80.0;
        state.threshold_adaptation = random.next() * 0.2;
        state.intrinsic_current = random.next() * 20.0 - 10.0;
        state.structural_drive = random.next();
        if id % 4 == 0 {
            state.refractory_until = at.checked_add_us(100).unwrap();
        }
        if id % 3 == 0 {
            state.last_spike_valid = 1;
            state.last_spike_time = start.as_micros();
            state.spike_count = 3;
        }
        cell.apply_numerical_state(state).unwrap();
        let amplitudes = if id % 257 == 0 && id % 2 != 0 {
            vec![f32::MAX, -f32::MAX, 1.0]
        } else {
            (0..id % 6).map(|_| random.next() * 1.2 - 0.4).collect()
        };
        rows.push((cell.id(), amplitudes));
        network.add_neuron(cell).unwrap();
    }
    let ids: Vec<_> = network.neuron_ids().collect();
    let original = DenseNeuronBatch::from_network(&network, &ids).unwrap();
    let mut cpu = original.clone();
    let mut cuda = original.clone();
    let mut replay = original;
    CpuCompute.advance_neurons(&mut cpu, at).unwrap();
    gpu.advance_neurons(&mut cuda, at).unwrap();
    gpu.advance_neurons(&mut replay, at).unwrap();
    println!(
        "advance rows=8192 max_abs_error={}",
        close(cpu.states(), cuda.states())
    );
    for (a, b) in cuda.states().iter().zip(replay.states()) {
        assert_eq!(floats(a).map(f32::to_bits), floats(b).map(f32::to_bits));
    }
    let mut inputs = FlatNeuronInputs::from_rows(rows).unwrap();
    let mut gpu_inputs = inputs.clone();
    let mut replay_inputs = inputs.clone();
    let cpu_spikes = CpuCompute
        .integrate_neurons(&mut cpu, &mut inputs, at)
        .unwrap();
    let gpu_spikes = gpu
        .integrate_neurons(&mut cuda, &mut gpu_inputs, at)
        .unwrap();
    let replay_spikes = gpu
        .integrate_neurons(&mut replay, &mut replay_inputs, at)
        .unwrap();
    assert_eq!(cpu_spikes, gpu_spikes);
    assert_eq!(gpu_spikes, replay_spikes);
    println!(
        "integrate rows=8192 spikes={} max_abs_error={}",
        cpu_spikes.len(),
        close(cpu.states(), cuda.states())
    );
    for (a, b) in cuda.states().iter().zip(replay.states()) {
        assert_eq!(floats(a).map(f32::to_bits), floats(b).map(f32::to_bits));
    }
    cuda.write_back(&mut network).unwrap();
}

fn small_network() -> Network {
    let mut network = Network::new();
    for id in 0..16 {
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
                        refractory_period_us: 10,
                        ..Default::default()
                    },
                    SimTime::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
    }
    for id in 0..16 {
        network
            .add_synapse(
                Synapse::new(
                    SynapseId(id),
                    NeuronId(id),
                    NeuronId((id + 1) % 16),
                    Weight::new(0.3).unwrap(),
                    7,
                    true,
                )
                .unwrap(),
            )
            .unwrap();
    }
    network
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored --nocapture"]
fn cuda_runs_real_simulation_with_stdp_and_utility() {
    let network = small_network();
    let make = |network| {
        Simulation::with_utility_rule(
            network,
            PairStdp::new(&LearningConfig::default()),
            Box::new(ExcitatoryCausalUtility),
            RuntimeConfig::default(),
            1.0,
        )
        .unwrap()
        .with_profiling()
    };
    let mut cpu = make(network.clone());
    let mut gpu = make(network).with_compute_backend(CudaCompute::new(0).unwrap());
    for round in 0..12_u64 {
        for (offset, id, amplitude) in [(0, round % 16, 1.1), (10, (round + 1) % 16, 0.8)] {
            let at = SimTime(round * 1_000 + offset);
            cpu.schedule_external_input(at, NeuronId(id), amplitude)
                .unwrap();
            gpu.schedule_external_input(at, NeuronId(id), amplitude)
                .unwrap();
        }
    }
    let report = cpu.run().unwrap();
    assert_eq!(report, gpu.run().unwrap());
    assert_eq!(cpu.event_log(), gpu.event_log());
    let cpu_cells: Vec<_> = cpu
        .network()
        .neurons()
        .map(Neuron::numerical_state)
        .collect();
    let gpu_cells: Vec<_> = gpu
        .network()
        .neurons()
        .map(Neuron::numerical_state)
        .collect();
    close(&cpu_cells, &gpu_cells);
    assert_eq!(
        cpu.network().synapses().collect::<Vec<_>>(),
        gpu.network().synapses().collect::<Vec<_>>()
    );
    println!(
        "simulation batches={} spikes={} identical event log and synapses",
        report.batches_processed, report.spikes_emitted
    );
    println!(
        "CPU batch total_us={:.3}; CUDA batch total_us={:.3}",
        cpu.profile()
            .unwrap()
            .phase(nerva::runtime::ProfilePhase::Batch)
            .total
            .as_secs_f64()
            * 1e6,
        gpu.profile()
            .unwrap()
            .phase(nerva::runtime::ProfilePhase::Batch)
            .total
            .as_secs_f64()
            * 1e6
    );
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored --nocapture"]
fn cuda_preserves_numerical_error_prefix_and_handles_empty_batches() {
    let mut gpu = CudaCompute::new(0).unwrap();
    let network = small_network();
    let mut empty = DenseNeuronBatch::from_network(&network, &[]).unwrap();
    gpu.advance_neurons(&mut empty, SimTime::ZERO).unwrap();
    assert!(
        gpu.integrate_neurons(
            &mut empty,
            &mut FlatNeuronInputs::from_rows([]).unwrap(),
            SimTime::ZERO
        )
        .unwrap()
        .is_empty()
    );
    let mut cpu = DenseNeuronBatch::from_network(&network, &[NeuronId(0), NeuronId(1)]).unwrap();
    let mut cuda = cpu.clone();
    // GPU detects refractory overflow on both cells; CPU replay preserves the
    // exact first failing cell state/error instead of publishing parallel work.
    let mut inputs =
        FlatNeuronInputs::from_rows([(NeuronId(0), vec![1.1]), (NeuronId(1), vec![1.1])]).unwrap();
    let mut gpu_inputs = inputs.clone();
    assert_eq!(
        CpuCompute.integrate_neurons(&mut cpu, &mut inputs, SimTime(u64::MAX)),
        gpu.integrate_neurons(&mut cuda, &mut gpu_inputs, SimTime(u64::MAX))
    );
    assert_eq!(cpu, cuda);
    assert_eq!(inputs, gpu_inputs);
}
