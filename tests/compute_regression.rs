//! Frozen observations captured from the runtime before extracting CpuCompute.
//! Includes complete cell/synapse state, logs, learning traces and pending clocks.

use nerva::{
    compute::CpuCompute,
    config::{
        HomeostasisConfig, IntrinsicDynamicsConfig, LearningConfig, NeuronConfig, RuntimeConfig,
    },
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    learning::{ExcitatoryCausalUtility, LocalHomeostasis, PairStdp},
    math::Position3D,
    primitives::Weight,
    runtime::{ObservationEvent, Simulation},
};

fn reference_run(frozen: bool, intrinsic: bool, explicit_backend: bool, profiling: bool) -> String {
    let mut network = Network::new();
    for id in [17, 2, 91] {
        let params = NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            membrane_tau_us: 1_000.0,
            refractory_period_us: 3,
            activity_trace_tau_us: 5_000.0,
            intrinsic: if intrinsic {
                IntrinsicDynamicsConfig {
                    intrinsic_drive: 1_200.0,
                    burst_gain: 200.0,
                    adaptation_gain: 100.0,
                    threshold_adaptation_gain: 0.05,
                    rebound_gain: 100.0,
                    ..Default::default()
                }
            } else {
                Default::default()
            },
        };
        network
            .add_neuron(
                Neuron::new(
                    NeuronId(id),
                    Position3D::ORIGIN,
                    if id == 91 {
                        Polarity::Inhibitory
                    } else {
                        Polarity::Excitatory
                    },
                    None,
                    params,
                    SimTime::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
    }
    for (id, pre, post, weight, delay, plastic) in [
        (53, 17, 2, 0.65, 7, true),
        (5, 2, 17, 0.45, 11, true),
        (99, 2, 91, 1.1, 13, false),
        (8, 91, 17, 0.2, 5, true),
    ] {
        network
            .add_synapse(
                Synapse::new(
                    SynapseId(id),
                    NeuronId(pre),
                    NeuronId(post),
                    Weight::new(weight).unwrap(),
                    delay,
                    plastic,
                )
                .unwrap(),
            )
            .unwrap();
    }
    let rule = PairStdp::new(&LearningConfig {
        a_plus: 0.08,
        a_minus: 0.09,
        tau_plus_us: 100.0,
        tau_minus_us: 120.0,
        stdp_window_us: 200,
        ..Default::default()
    });
    let mut simulation = Simulation::with_utility_rule(
        network,
        rule,
        Box::new(ExcitatoryCausalUtility),
        RuntimeConfig::default(),
        1.0,
    )
    .unwrap()
    .with_homeostasis(LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 500,
        ..Default::default()
    }))
    .unwrap();
    if explicit_backend {
        simulation = simulation.with_compute_backend(CpuCompute);
    }
    if profiling {
        simulation.enable_profiling();
    }
    if frozen {
        simulation.freeze_learning();
    }
    // Cancellation, canonical summation, repeated targets, refractory input,
    // excitatory/inhibitory arrivals, and causal/anti-causal learning.
    for (time, target, amplitude) in [
        (20, 2, 0.6),
        (0, 17, 1.1),
        (0, 17, -0.2),
        (0, 17, 0.2),
        (1, 17, 1.5),
        (16, 91, 1.1),
        (7, 2, 0.5),
        (50, 17, 1.1),
        (70, 2, 1.1),
        (70, 2, -0.3),
        (1_000, 17, -0.6),
    ] {
        simulation
            .schedule_external_input(SimTime(time), NeuronId(target), amplitude)
            .unwrap();
    }
    let first = simulation.run_until(SimTime(100)).unwrap();
    // Live topology edits must remain visible without a second state store.
    simulation.remove_synapse(SynapseId(5)).unwrap();
    simulation
        .add_synapse(
            Synapse::new(
                SynapseId(105),
                NeuronId(2),
                NeuronId(17),
                Weight::new(0.55).unwrap(),
                9,
                true,
            )
            .unwrap(),
        )
        .unwrap();
    let second = simulation.run_until(SimTime(2_000)).unwrap();
    assert!(first.spikes_emitted > 0);
    assert!(
        simulation
            .event_log()
            .iter()
            .any(|event| matches!(event, ObservationEvent::HomeostasisChanged { .. }))
    );
    if !frozen {
        assert!(
            simulation
                .event_log()
                .iter()
                .any(|event| matches!(event, ObservationEvent::WeightChanged { .. }))
        );
    }
    let pending = simulation.pending_event_count();
    let clocks: Vec<_> = simulation
        .network()
        .neuron_ids()
        .map(|id| {
            (
                id,
                simulation.next_homeostasis_update(id),
                simulation.next_intrinsic_spike(id),
            )
        })
        .collect();
    let current_time = simulation.current_time();
    let (network, rule, homeostasis, log) = simulation.into_parts();
    let traces: Vec<_> = network
        .neuron_ids()
        .map(|id| (id, rule.post_trace(id).copied()))
        .collect();
    format!(
        "reports={first:?} {second:?}\ntime={current_time:?} pending={pending}\nclocks={clocks:?}\nnetwork={network:#?}\ntraces={traces:?}\nhomeostasis={homeostasis:?}\nlog={log:#?}\n"
    )
}

#[test]
fn cpu_compute_preserves_the_original_runtime_state_and_observations() {
    for profiling in [false, true] {
        for explicit_backend in [false, true] {
            let actual = [(false, false), (true, false), (false, true)]
                .into_iter()
                .map(|(frozen, intrinsic)| {
                    format!(
                        "frozen={frozen} intrinsic={intrinsic}\n{}",
                        reference_run(frozen, intrinsic, explicit_backend, profiling)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            assert_eq!(actual, include_str!("fixtures/cpu_reference.txt"));
        }
    }
}
