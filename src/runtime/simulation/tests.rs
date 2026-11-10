use crate::{
    config::{HomeostasisConfig, NervaConfig, NeuronConfig},
    core::{Neuron, Polarity, Synapse},
    learning::{NoPlasticity, PlasticityRule},
    math::Position3D,
};

use super::*;

fn neuron_params() -> NeuronConfig {
    NeuronConfig {
        resting_potential: 0.0,
        reset_potential: 0.0,
        threshold: 1.0,
        membrane_tau_us: 1_000.0,
        refractory_period_us: 1,
        activity_trace_tau_us: 10_000.0,
        intrinsic: Default::default(),
    }
}

fn neuron(id: u64, polarity: Polarity) -> Neuron {
    Neuron::new(
        NeuronId(id),
        Position3D::ORIGIN,
        polarity,
        None,
        neuron_params(),
        SimTime::ZERO,
    )
    .unwrap()
}

fn one_neuron_network() -> Network {
    let mut network = Network::new();
    network.add_neuron(neuron(1, Polarity::Excitatory)).unwrap();
    network
}

fn connected_network(source_polarity: Polarity) -> Network {
    let mut network = Network::new();
    network.add_neuron(neuron(1, source_polarity)).unwrap();
    network.add_neuron(neuron(2, Polarity::Excitatory)).unwrap();
    network
        .add_synapse(
            Synapse::new(
                SynapseId(1),
                NeuronId(1),
                NeuronId(2),
                Weight::new(1.0).unwrap(),
                5,
                true,
            )
            .unwrap(),
        )
        .unwrap();
    network
}

fn runtime_config(max_events_per_batch: usize) -> RuntimeConfig {
    RuntimeConfig {
        max_events_per_batch,
    }
}

fn simulation<R: PlasticityRule>(network: Network, rule: R) -> Simulation<R> {
    Simulation::new(network, rule, runtime_config(100), 1.0).unwrap()
}

fn network_with_inconsistent_indices() -> Network {
    let mut network = connected_network(Polarity::Excitatory);
    network
        .synapse_mut(SynapseId(1))
        .expect("test synapse exists")
        .pre = NeuronId(2);
    network
}

#[derive(Debug, Default)]
struct AdditiveRule {
    pre_calls: usize,
    post_synapses: usize,
}

impl PlasticityRule for AdditiveRule {
    fn accepts(&self, synapse: &Synapse, polarity: Polarity) -> bool {
        synapse.is_enabled() && synapse.is_plastic() && polarity == Polarity::Excitatory
    }

    fn on_pre_arrival(&mut self, synapse: &mut Synapse, _post: &Neuron, _time: SimTime) {
        self.pre_calls += 1;
        synapse
            .set_weight(Weight::new(synapse.weight().get() + 0.1).unwrap())
            .unwrap();
    }

    fn on_post_spike(&mut self, _neuron: &Neuron, incoming: &mut [Synapse], _time: SimTime) {
        self.post_synapses += incoming.len();
        for synapse in incoming {
            synapse
                .set_weight(Weight::new(synapse.weight().get() + 0.2).unwrap())
                .unwrap();
        }
    }
}

#[test]
fn buffered_propagation_matches_public_plans_and_validated_scheduling() {
    let mut network = connected_network(Polarity::Inhibitory);
    network
        .add_neuron(neuron(91, Polarity::Excitatory))
        .unwrap();
    // Insert out of ID order, include a disabled edge and an empty source.
    for (id, pre, post, delay, enabled) in [
        (90, 2, 91, 9, true),
        (8, 1, 91, 5, false),
        (5, 1, 2, 5, true),
    ] {
        let mut synapse = Synapse::new(
            SynapseId(id),
            NeuronId(pre),
            NeuronId(post),
            Weight::new(0.1).unwrap(),
            delay,
            false,
        )
        .unwrap();
        synapse.set_enabled(enabled);
        network.add_synapse(synapse).unwrap();
    }
    let mut reference = simulation(network.clone(), NoPlasticity);
    let mut buffered = simulation(network, NoPlasticity);
    // Exercise buffer clearing for different degrees and repeated spikes.
    let spikes: Vec<_> = [(1, 0), (91, 0), (2, 0), (1, 20), (2, 20)]
        .into_iter()
        .map(|(id, time)| Spike::new(NeuronId(id), SimTime(time)))
        .collect();
    for &spike in &spikes {
        for transmission in
            super::super::propagation::plan_spike_propagation(reference.network(), spike, 1.0)
                .unwrap()
        {
            reference.schedule_event(transmission.event).unwrap();
        }
    }
    buffered.propagate_batch_spikes(&spikes).unwrap();
    assert_eq!(buffered.run().unwrap(), reference.run().unwrap());
    assert_eq!(buffered.network(), reference.network());
    assert_eq!(buffered.event_log(), reference.event_log());
}

#[test]
fn a_failed_plan_never_enqueues_its_successful_prefix() {
    let mut network = connected_network(Polarity::Excitatory);
    network
        .add_synapse(
            Synapse::new(
                SynapseId(2),
                NeuronId(1),
                NeuronId(2),
                Weight::new(0.1).unwrap(),
                10,
                false,
            )
            .unwrap(),
        )
        .unwrap();
    let mut simulation = simulation(network, NoPlasticity).with_profiling();
    simulation
        .schedule_external_input(SimTime(u64::MAX - 6), NeuronId(1), 1.1)
        .unwrap();
    assert!(simulation.step().is_err());
    assert!(simulation.is_poisoned());
    assert!(matches!(
        simulation.poison_reason(),
        Some(SimulationError::Propagation(
            PropagationError::ArrivalTimeOverflow { .. }
        ))
    ));
    assert_eq!(simulation.pending_event_count(), 0);
    let profile = simulation.profile().unwrap();
    assert_eq!(profile.phase(ProfilePhase::PropagationPlan).failures, 1);
    assert_eq!(profile.phase(ProfilePhase::PropagationEnqueue).calls, 0);
}

#[test]
fn post_learning_validates_all_staged_synapses_before_publishing_any() {
    struct CorruptPost(u8);
    impl PlasticityRule for CorruptPost {
        fn accepts(&self, _: &Synapse, _: Polarity) -> bool {
            true
        }
        fn on_pre_arrival(&mut self, _: &mut Synapse, _: &Neuron, _: SimTime) {}
        fn on_post_spike(&mut self, _: &Neuron, incoming: &mut [Synapse], _: SimTime) {
            incoming[0].set_weight(Weight::new(0.9).unwrap()).unwrap();
            match self.0 {
                0 => incoming[1].id = SynapseId(999),
                1 => incoming[1].pre = NeuronId(2),
                2 => incoming[1].post = NeuronId(1),
                _ => incoming[1].delay_us = 0,
            }
        }
    }
    for corruption in 0..4 {
        let mut network = connected_network(Polarity::Excitatory);
        network
            .add_synapse(
                Synapse::new(
                    SynapseId(2),
                    NeuronId(1),
                    NeuronId(2),
                    Weight::new(0.2).unwrap(),
                    5,
                    true,
                )
                .unwrap(),
            )
            .unwrap();
        let original = network.clone();
        let mut simulation = simulation(network, CorruptPost(corruption));
        assert!(
            simulation
                .apply_post_spike_learning(Spike::new(NeuronId(2), SimTime::ZERO))
                .is_err()
        );
        assert_eq!(simulation.network(), &original);
        assert!(simulation.event_log().is_empty());
    }
}

#[test]
fn constructors_reject_inconsistent_network_indices() {
    let network = network_with_inconsistent_indices();

    assert!(matches!(
        Simulation::new(network.clone(), NoPlasticity, runtime_config(100), 1.0),
        Err(SimulationError::InvalidNetwork(
            NetworkError::InconsistentIndices
        ))
    ));
    assert!(matches!(
        Simulation::from_config(network, NoPlasticity, &NervaConfig::default()),
        Err(SimulationError::InvalidNetwork(
            NetworkError::InconsistentIndices
        ))
    ));
}

#[test]
fn aggregate_constructor_rejects_network_config_mismatch() {
    let network = connected_network(Polarity::Excitatory);
    let error = Simulation::from_config(network, NoPlasticity, &NervaConfig::default())
        .err()
        .expect("default config describes another population");

    assert!(matches!(
        error,
        SimulationError::InvalidNetwork(NetworkError::PopulationMismatch { .. })
    ));
}

#[test]
fn simultaneous_inputs_are_summed_before_one_firing_check() {
    let mut simulation = simulation(one_neuron_network(), NoPlasticity);
    simulation
        .schedule_external_input(SimTime(10), NeuronId(1), 0.6)
        .unwrap();
    simulation
        .schedule_external_input(SimTime(10), NeuronId(1), 0.6)
        .unwrap();

    let report = simulation.step().unwrap().unwrap();

    assert_eq!(
        report,
        BatchReport {
            time: SimTime(10),
            events_processed: 2,
            spikes_emitted: 1,
        }
    );
    assert_eq!(
        simulation
            .network()
            .neuron(NeuronId(1))
            .unwrap()
            .spike_count(),
        1
    );
}

#[test]
fn simultaneous_sum_is_independent_of_event_insertion_order() {
    fn run(amplitudes: [f32; 3]) -> Vec<ObservationEvent> {
        let mut simulation = simulation(one_neuron_network(), NoPlasticity);
        for amplitude in amplitudes {
            simulation
                .schedule_external_input(SimTime(1), NeuronId(1), amplitude)
                .unwrap();
        }
        simulation.run().unwrap();
        simulation.event_log().to_vec()
    }

    let first = run([f32::MAX, -f32::MAX, 1.0]);
    let second = run([1.0, f32::MAX, -f32::MAX]);
    let first_spikes = first
        .iter()
        .filter(|event| matches!(event, ObservationEvent::SpikeEmitted(_)))
        .count();
    let second_spikes = second
        .iter()
        .filter(|event| matches!(event, ObservationEvent::SpikeEmitted(_)))
        .count();

    assert_eq!(first_spikes, 1);
    assert_eq!(first_spikes, second_spikes);
}

#[test]
fn spike_is_delayed_propagated_and_logged() {
    let mut simulation = simulation(connected_network(Polarity::Excitatory), NoPlasticity);
    simulation
        .schedule_external_input(SimTime(10), NeuronId(1), 1.0)
        .unwrap();

    let report = simulation.run().unwrap();

    assert_eq!(report.batches_processed, 2);
    assert_eq!(report.events_processed, 2);
    assert_eq!(report.spikes_emitted, 2);
    assert_eq!(report.last_time, Some(SimTime(15)));
    assert_eq!(
        simulation
            .network()
            .synapse(SynapseId(1))
            .unwrap()
            .transmission_count(),
        1
    );
    assert!(simulation.event_log().iter().any(|event| matches!(
        event,
        ObservationEvent::SynapticArrival {
            time: SimTime(15),
            synapse_id: SynapseId(1),
            ..
        }
    )));
}

#[test]
fn excitatory_learning_hooks_run_and_freeze_prevents_all_weight_mutation() {
    let network = connected_network(Polarity::Excitatory);
    let mut learning = simulation(network.clone(), AdditiveRule::default());
    learning
        .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
        .unwrap();
    learning.run().unwrap();
    let learned_weight = learning
        .network()
        .synapse(SynapseId(1))
        .unwrap()
        .weight()
        .get();
    assert!((learned_weight - 1.3).abs() < 1.0e-6);
    assert_eq!(
        learning
            .event_log()
            .iter()
            .filter(|event| matches!(event, ObservationEvent::WeightChanged { .. }))
            .count(),
        2
    );

    let mut frozen = simulation(network, AdditiveRule::default());
    frozen.freeze_learning();
    frozen
        .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
        .unwrap();
    frozen.run().unwrap();
    let (network, rule, _, _) = frozen.into_parts();
    assert_eq!(
        network.synapse(SynapseId(1)).unwrap().weight(),
        Weight::new(1.0).unwrap()
    );
    assert_eq!(rule.pre_calls, 0);
    assert_eq!(rule.post_synapses, 0);
}

#[test]
fn inhibitory_presynaptic_cell_is_never_passed_to_stdp() {
    let mut simulation = simulation(
        connected_network(Polarity::Inhibitory),
        AdditiveRule::default(),
    );
    simulation
        .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
        .unwrap();
    simulation.run().unwrap();

    let (network, rule, _, _) = simulation.into_parts();
    assert_eq!(
        network.synapse(SynapseId(1)).unwrap().weight(),
        Weight::new(1.0).unwrap()
    );
    assert_eq!(rule.pre_calls, 0);
    assert_eq!(rule.post_synapses, 0);
}

#[test]
fn same_inputs_produce_identical_observation_log() {
    fn run_once() -> EventLog {
        let mut simulation = simulation(connected_network(Polarity::Excitatory), NoPlasticity);
        simulation
            .schedule_external_input(SimTime(4), NeuronId(1), 0.4)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(4), NeuronId(1), 0.6)
            .unwrap();
        simulation.run().unwrap();
        simulation.drain_event_log()
    }

    assert_eq!(run_once(), run_once());
}

#[test]
fn max_events_per_batch_fails_before_batch_mutation() {
    let mut simulation =
        Simulation::new(one_neuron_network(), NoPlasticity, runtime_config(1), 1.0).unwrap();
    simulation
        .schedule_external_input(SimTime(1), NeuronId(1), 0.1)
        .unwrap();
    simulation
        .schedule_external_input(SimTime(1), NeuronId(1), 0.2)
        .unwrap();

    assert!(matches!(
        simulation.step(),
        Err(SimulationError::Scheduler(
            SchedulerError::BatchLimitExceeded { .. }
        ))
    ));
    assert_eq!(simulation.pending_event_count(), 2);
    assert_eq!(
        simulation
            .network()
            .neuron(NeuronId(1))
            .unwrap()
            .last_update(),
        SimTime::ZERO
    );
    // A scheduler safety failure leaves the runtime usable.
    assert!(!simulation.is_poisoned());
}

#[test]
fn controlled_add_neuron_integrates_homeostasis_and_intrinsic_prediction() {
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 1_000,
        ..HomeostasisConfig::default()
    });
    let mut simulation = simulation(one_neuron_network(), NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    let driven = Neuron::new(
        NeuronId(2),
        Position3D::ORIGIN,
        Polarity::Excitatory,
        None,
        NeuronConfig {
            intrinsic: crate::config::IntrinsicDynamicsConfig {
                intrinsic_drive: 2_000_000.0,
                ..crate::config::IntrinsicDynamicsConfig::default()
            },
            ..neuron_params()
        },
        SimTime::ZERO,
    )
    .unwrap();

    simulation.add_neuron(driven).unwrap();

    // The new cell's autonomous crossing was predicted and scheduled.
    assert!(simulation.next_intrinsic_spike(NeuronId(2)).is_some());
    // Its local maintenance clock started with the shared phase rule.
    assert!(simulation.next_homeostasis_update(NeuronId(2)).is_some());
    assert!(simulation.run_until(SimTime(10_000)).is_ok());
}

#[test]
fn remove_neuron_cancels_intrinsic_homeostasis_and_external_bookings() {
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 1_000,
        ..HomeostasisConfig::default()
    });
    let mut simulation = simulation(one_neuron_network(), NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    let driven = Neuron::new(
        NeuronId(9),
        Position3D::ORIGIN,
        Polarity::Excitatory,
        None,
        NeuronConfig {
            intrinsic: crate::config::IntrinsicDynamicsConfig {
                intrinsic_drive: 2_000_000.0,
                ..crate::config::IntrinsicDynamicsConfig::default()
            },
            ..neuron_params()
        },
        SimTime::ZERO,
    )
    .unwrap();
    simulation.add_neuron(driven).unwrap();
    simulation
        .schedule_external_input(SimTime(5_000), NeuronId(9), 0.5)
        .unwrap();
    assert!(simulation.next_intrinsic_spike(NeuronId(9)).is_some());
    assert!(simulation.next_homeostasis_update(NeuronId(9)).is_some());
    assert_eq!(simulation.pending_event_count(), 4);

    let removed = simulation.remove_neuron(NeuronId(9)).unwrap();
    assert_eq!(removed.id(), NeuronId(9));
    assert!(simulation.network().neuron(NeuronId(9)).is_none());
    assert!(simulation.next_intrinsic_spike(NeuronId(9)).is_none());
    assert!(simulation.next_homeostasis_update(NeuronId(9)).is_none());
    // Only neuron 1's recurring local clock remains. No stale event can
    // later fail validation because it refers to the removed cell.
    assert_eq!(simulation.pending_event_count(), 1);
    assert!(simulation.run_until(SimTime(5_000)).is_ok());
}

#[test]
fn failed_connected_removal_preserves_all_runtime_bookings() {
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 1_000,
        ..HomeostasisConfig::default()
    });
    let mut simulation = simulation(connected_network(Polarity::Excitatory), NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    simulation
        .schedule_external_input(SimTime(5_000), NeuronId(1), 0.5)
        .unwrap();
    let before_events = simulation.pending_event_count();
    let before_clock = simulation.next_homeostasis_update(NeuronId(1));

    assert!(matches!(
        simulation.remove_neuron(NeuronId(1)),
        Err(SimulationError::InvalidNetwork(
            NetworkError::NeuronStillConnected { .. }
        ))
    ));
    assert!(simulation.network().neuron(NeuronId(1)).is_some());
    assert_eq!(simulation.pending_event_count(), before_events);
    assert_eq!(
        simulation.next_homeostasis_update(NeuronId(1)),
        before_clock
    );
    assert!(simulation.run_until(SimTime(5_000)).is_ok());
}

#[test]
fn transmission_block_cancels_travelling_inputs_and_preserves_other_inputs() {
    let mut simulation = simulation(connected_network(Polarity::Excitatory), NoPlasticity);
    simulation
        .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
        .unwrap();
    simulation.run_until(SimTime::ZERO).unwrap();
    assert_eq!(simulation.pending_event_count(), 1);
    let before = simulation.network().clone();
    let pending = simulation.pending_event_count();
    assert!(
        simulation
            .disable_synapses_and_cancel_arrivals(&[SynapseId(1), SynapseId(999)])
            .is_err()
    );
    assert_eq!(simulation.network(), &before);
    assert_eq!(simulation.pending_event_count(), pending);
    simulation
        .schedule_external_input(SimTime(20), NeuronId(2), 1.0)
        .unwrap();
    assert_eq!(
        simulation
            .disable_synapses_and_cancel_arrivals(&[SynapseId(1)])
            .unwrap(),
        1
    );
    let mut expected = before;
    expected
        .synapse_mut(SynapseId(1))
        .unwrap()
        .set_enabled(false);
    assert_eq!(simulation.network(), &expected);
    simulation.run_until(SimTime(20)).unwrap();
    assert_eq!(
        simulation
            .network()
            .neuron(NeuronId(2))
            .unwrap()
            .spike_count(),
        1
    );
    simulation
        .schedule_external_input(SimTime(10000), NeuronId(1), 1.0)
        .unwrap();
    simulation.run_until(SimTime(10010)).unwrap();
    assert_eq!(
        simulation
            .network()
            .neuron(NeuronId(2))
            .unwrap()
            .spike_count(),
        1
    );
    assert_eq!(simulation.pending_event_count(), 0);
}

#[test]
fn live_synapse_pruning_cancels_in_flight_arrivals_and_unblocks_neuron_removal() {
    let mut network = Network::new();
    network.add_neuron(neuron(1, Polarity::Excitatory)).unwrap();
    network.add_neuron(neuron(2, Polarity::Excitatory)).unwrap();
    let mut simulation = simulation(network, NoPlasticity);

    simulation
        .add_synapse(
            Synapse::new(
                SynapseId(7),
                NeuronId(1),
                NeuronId(2),
                Weight::new(1.0).unwrap(),
                100,
                true,
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(simulation.network().synapse_count(), 1);

    simulation
        .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
        .unwrap();
    simulation.run_until(SimTime::ZERO).unwrap();
    assert_eq!(simulation.pending_event_count(), 1);

    let removed = simulation.remove_synapse(SynapseId(7)).unwrap();
    assert_eq!(removed.id(), SynapseId(7));
    assert_eq!(simulation.network().synapse_count(), 0);
    assert_eq!(simulation.pending_event_count(), 0);
    // No delayed arrival can reach batch validation after pruning.
    assert!(simulation.run_until(SimTime(100)).is_ok());
    assert_eq!(
        simulation
            .network()
            .neuron(NeuronId(2))
            .unwrap()
            .spike_count(),
        0
    );

    // The public live-topology path now permits explicit dismantling.
    assert_eq!(
        simulation.remove_neuron(NeuronId(1)).unwrap().id(),
        NeuronId(1)
    );
    assert_eq!(
        simulation.remove_neuron(NeuronId(2)).unwrap().id(),
        NeuronId(2)
    );
}

#[test]
fn failed_live_synapse_removal_preserves_queue_and_topology() {
    let mut simulation = simulation(connected_network(Polarity::Excitatory), NoPlasticity);
    simulation
        .schedule_external_input(SimTime(5_000), NeuronId(1), 0.5)
        .unwrap();
    let before_events = simulation.pending_event_count();

    assert!(matches!(
        simulation.remove_synapse(SynapseId(999)),
        Err(SimulationError::InvalidNetwork(
            NetworkError::MissingSynapse(SynapseId(999))
        ))
    ));
    assert!(simulation.network().synapse(SynapseId(1)).is_some());
    assert_eq!(simulation.pending_event_count(), before_events);
}

#[test]
fn failed_live_synapse_addition_preserves_topology() {
    let mut simulation = simulation(one_neuron_network(), NoPlasticity);

    assert!(matches!(
        simulation.add_synapse(
            Synapse::new(
                SynapseId(7),
                NeuronId(1),
                NeuronId(2),
                Weight::new(1.0).unwrap(),
                1,
                true,
            )
            .unwrap(),
        ),
        Err(SimulationError::InvalidNetwork(
            NetworkError::MissingPostsynapticNeuron(NeuronId(2))
        ))
    ));
    assert_eq!(simulation.network().synapse_count(), 0);
    assert_eq!(simulation.pending_event_count(), 0);
}

#[test]
fn update_neuron_refreshes_the_autonomous_prediction() {
    let mut simulation = simulation(one_neuron_network(), NoPlasticity);
    assert!(simulation.next_intrinsic_spike(NeuronId(1)).is_none());

    simulation
        .update_neuron(NeuronId(1), |neuron| {
            neuron.set_intrinsic_current(1_000_000.0)
        })
        .unwrap();

    assert!(simulation.next_intrinsic_spike(NeuronId(1)).is_some());
    let report = simulation.run_until(SimTime(5_000)).unwrap();
    assert!(report.spikes_emitted >= 1);
}

#[test]
fn failed_neuron_mutation_rolls_back_cell_and_booking_state() {
    let mut simulation = simulation(one_neuron_network(), NoPlasticity);
    let before = simulation.network().neuron(NeuronId(1)).unwrap().clone();
    let before_events = simulation.pending_event_count();

    let result = simulation.update_neuron(NeuronId(1), |neuron| {
        neuron.set_intrinsic_current(1_000_000.0)?;
        Err(NeuronError::InvalidHomeostaticValue {
            field: "test rollback",
            value: f32::NAN,
        })
    });

    assert!(matches!(result, Err(SimulationError::Neuron(_))));
    assert_eq!(simulation.network().neuron(NeuronId(1)), Some(&before));
    assert_eq!(simulation.pending_event_count(), before_events);
    assert_eq!(simulation.next_intrinsic_spike(NeuronId(1)), None);
}

#[test]
fn failed_homeostasis_enable_rolls_back_controller_and_cell_anchors() {
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: false,
        update_interval_us: 10,
        ..HomeostasisConfig::default()
    });
    let mut simulation = simulation(one_neuron_network(), NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    simulation
        .schedule_external_input(SimTime(u64::MAX), NeuronId(1), 0.0)
        .unwrap();
    simulation.run_until(SimTime(u64::MAX)).unwrap();
    let before = simulation.network().neuron(NeuronId(1)).unwrap().clone();

    assert!(matches!(
        simulation.set_homeostasis_enabled(true),
        Err(SimulationError::HomeostasisTimeOverflow { .. })
    ));
    assert!(!simulation.homeostasis_enabled());
    assert_eq!(simulation.network().neuron(NeuronId(1)), Some(&before));
    assert_eq!(simulation.next_homeostasis_update(NeuronId(1)), None);
}

#[test]
fn poisoned_simulation_rejects_every_further_operation() {
    // Schedule an event that will fail fatally during batch processing:
    // the batch targets a neuron whose local state rejects the batch's
    // timestamp, while the scheduler itself accepts the booking.
    let mut oversize =
        Simulation::new(one_neuron_network(), NoPlasticity, runtime_config(1), 1.0).unwrap();
    oversize
        .schedule_external_input(SimTime(1), NeuronId(1), 0.1)
        .unwrap();
    oversize
        .schedule_external_input(SimTime(1), NeuronId(1), 0.2)
        .unwrap();

    // A scheduler batch-limit failure is a pre-mutation safety stop and
    // leaves the runtime usable.
    assert!(matches!(
        oversize.step(),
        Err(SimulationError::Scheduler(_))
    ));
    assert!(!oversize.is_poisoned());

    // A neuron-integration failure poisons the runtime instead.
    // Build a network whose neuron rejects time evolution by starting
    // already advanced; supplying an event earlier than the neuron's
    // last-update anchor makes integration fail fatally.
    let mut network = one_neuron_network();
    network
        .neuron_mut(NeuronId(1))
        .unwrap()
        .advance_to(SimTime(100))
        .unwrap();
    let mut poisoned = Simulation::new(network, NoPlasticity, runtime_config(100), 1.0).unwrap();
    // The scheduler accepts the timestamp, but advancing the neuron to a
    // time before its anchor fails inside the batch.
    poisoned
        .scheduler
        .schedule(
            SimTime(50),
            EventKind::Homeostasis {
                neuron_id: NeuronId(1),
            },
        )
        .unwrap();

    let result = poisoned.step();
    assert!(result.is_err());
    assert!(poisoned.is_poisoned());

    // Every further use is rejected with the original reason.
    assert!(matches!(
        poisoned.step(),
        Err(SimulationError::SimulationPoisoned { .. })
    ));
    assert!(matches!(
        poisoned.run_until(SimTime(1_000)),
        Err(SimulationError::SimulationPoisoned { .. })
    ));
    assert!(matches!(
        poisoned.schedule_external_input(SimTime(1_000), NeuronId(1), 1.0),
        Err(SimulationError::SimulationPoisoned { .. })
    ));
    assert!(matches!(
        poisoned.add_neuron(neuron(3, Polarity::Excitatory)),
        Err(SimulationError::SimulationPoisoned { .. })
    ));
    assert!(matches!(
        poisoned.update_neuron(NeuronId(1), |_| Ok(())),
        Err(SimulationError::SimulationPoisoned { .. })
    ));
    assert!(poisoned.poison_reason().is_some());
}

#[test]
fn maintenance_uses_neuron_local_input_and_activity_after_same_time_spike() {
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 1_000,
        target_rate_hz: 0.0,
        target_input_rate: 0.0,
        intrinsic_adjustment_rate: 1.0,
        structural_adjustment_rate: 0.0,
        min_intrinsic_current: -2.0,
        max_intrinsic_current: 2.0,
        min_structural_drive: 0.0,
        max_structural_drive: 2.0,
    });
    let mut simulation = simulation(one_neuron_network(), NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    simulation
        .schedule_external_input(SimTime(1_000), NeuronId(1), 1.0)
        .unwrap();
    simulation
        .schedule_homeostasis(SimTime(1_000), NeuronId(1))
        .unwrap();

    simulation.run_until(SimTime(1_000)).unwrap();

    assert!(
        simulation
            .network()
            .neuron(NeuronId(1))
            .unwrap()
            .intrinsic_current()
            < 0.0
    );
    assert!(simulation.event_log().iter().any(|event| matches!(
        event,
        ObservationEvent::HomeostasisChanged {
            neuron_id: NeuronId(1),
            ..
        }
    )));
}

#[test]
fn local_homeostasis_clock_maintains_a_silent_neuron_without_global_tick() {
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 10,
        target_rate_hz: 1.0,
        target_input_rate: 1.0,
        intrinsic_adjustment_rate: 1.0,
        structural_adjustment_rate: 1.0,
        min_intrinsic_current: -10.0,
        max_intrinsic_current: 10.0,
        min_structural_drive: 0.0,
        max_structural_drive: 10.0,
    });
    let mut simulation = simulation(one_neuron_network(), NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    let first = simulation
        .next_homeostasis_update(NeuronId(1))
        .expect("enabled controller starts a local clock");
    assert!((1..=10).contains(&first.as_micros()));
    assert_eq!(simulation.run(), Err(SimulationError::RunRequiresDeadline));

    let report = simulation
        .run_until(first.checked_add_us(10).unwrap())
        .unwrap();
    let neuron = simulation.network().neuron(NeuronId(1)).unwrap();
    assert_eq!(report.spikes_emitted, 0);
    assert!(neuron.structural_drive() > 0.0);
    assert_eq!(
        simulation.next_homeostasis_update(NeuronId(1)),
        first.checked_add_us(20)
    );
    assert_eq!(
        simulation
            .event_log()
            .iter()
            .filter(|event| matches!(event, ObservationEvent::HomeostasisChanged { .. }))
            .count(),
        2
    );
}

#[test]
fn aggregate_configuration_starts_local_clocks_automatically() {
    let mut config = NervaConfig::default();
    config.network.excitatory_neurons = 1;
    config.network.inhibitory_neurons = 0;
    config.neuron = neuron_params();
    config.learning.homeostasis.enabled = true;
    config.learning.homeostasis.update_interval_us = 10;

    let simulation = Simulation::from_config(one_neuron_network(), NoPlasticity, &config)
        .expect("validated configuration starts runtime");

    assert!(simulation.next_homeostasis_update(NeuronId(1)).is_some());
}

#[test]
fn intrinsic_current_emits_a_predicted_deterministic_spike_without_input() {
    let mut network = one_neuron_network();
    network
        .neuron_mut(NeuronId(1))
        .unwrap()
        .set_intrinsic_current(2_000.0)
        .unwrap();
    let mut simulation = simulation(network, NoPlasticity);

    assert_eq!(
        simulation.run_until(SimTime(693)).unwrap().spikes_emitted,
        0
    );
    assert_eq!(
        simulation.run_until(SimTime(694)).unwrap().spikes_emitted,
        1
    );
    assert!(simulation.event_log().iter().any(|event| matches!(
        event,
        ObservationEvent::SpikeEmitted(spike) if spike.neuron_id == NeuronId(1) && spike.time == SimTime(694)
    )));
}

#[test]
fn superseded_intrinsic_prediction_is_cancelled_eagerly() {
    let mut network = one_neuron_network();
    network
        .neuron_mut(NeuronId(1))
        .unwrap()
        .set_intrinsic_current(2_000.0)
        .unwrap();
    let mut simulation = simulation(network, NoPlasticity);
    assert_eq!(simulation.pending_event_count(), 1);

    simulation
        .schedule_external_input(SimTime(100), NeuronId(1), -0.5)
        .unwrap();
    simulation.run_until(SimTime(100)).unwrap();

    // The original crossing at 694 us was removed, rather than retained
    // as an inert stale event alongside the replacement prediction.
    assert_eq!(simulation.pending_event_count(), 1);
    assert!(
        simulation
            .next_intrinsic_spike(NeuronId(1))
            .is_some_and(|time| time > SimTime(694))
    );
}

#[test]
fn enabling_homeostasis_starts_and_disabling_stops_local_clocks() {
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: false,
        update_interval_us: 100,
        target_rate_hz: 1.0,
        target_input_rate: 1.0,
        intrinsic_adjustment_rate: 1.0,
        structural_adjustment_rate: 1.0,
        min_intrinsic_current: -10.0,
        max_intrinsic_current: 10.0,
        min_structural_drive: 0.0,
        max_structural_drive: 10.0,
    });
    let mut simulation = simulation(one_neuron_network(), NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    assert_eq!(simulation.next_homeostasis_update(NeuronId(1)), None);

    simulation.set_homeostasis_enabled(true).unwrap();
    let first = simulation
        .next_homeostasis_update(NeuronId(1))
        .expect("enabling starts the local clock");
    simulation.set_homeostasis_enabled(false).unwrap();
    simulation.run_until(first).unwrap();

    assert_eq!(simulation.next_homeostasis_update(NeuronId(1)), None);
    assert_eq!(simulation.pending_event_count(), 0);

    simulation.set_homeostasis_enabled(true).unwrap();
    assert!(simulation.next_homeostasis_update(NeuronId(1)).is_some());
}

#[test]
fn reenabled_homeostasis_does_not_integrate_the_disabled_interval() {
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 10,
        target_rate_hz: 1.0,
        target_input_rate: 1.0,
        intrinsic_adjustment_rate: 1.0,
        structural_adjustment_rate: 1.0,
        min_intrinsic_current: -10.0,
        max_intrinsic_current: 10.0,
        min_structural_drive: 0.0,
        max_structural_drive: 10.0,
    });
    let mut simulation = simulation(one_neuron_network(), NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    let first = simulation.next_homeostasis_update(NeuronId(1)).unwrap();
    simulation.run_until(first).unwrap();
    let drive_before_pause = simulation
        .network()
        .neuron(NeuronId(1))
        .unwrap()
        .structural_drive();

    simulation.set_homeostasis_enabled(false).unwrap();
    simulation
        .schedule_external_input(SimTime(100_000), NeuronId(1), 0.0)
        .unwrap();
    simulation.run_until(SimTime(100_000)).unwrap();

    simulation.set_homeostasis_enabled(true).unwrap();
    let first_after_reenable = simulation.next_homeostasis_update(NeuronId(1)).unwrap();
    simulation.run_until(first_after_reenable).unwrap();
    let drive_after_reenable = simulation
        .network()
        .neuron(NeuronId(1))
        .unwrap()
        .structural_drive();

    // Only the freshly enabled local phase (at most 10 us), not the
    // nearly 100 ms disabled interval, contributes to this update.
    assert!(drive_after_reenable - drive_before_pause < 0.000_02);
}

#[test]
fn many_neuron_clocks_have_deterministic_spread_without_a_global_phase() {
    let mut network = Network::new();
    for id in 1..=1_000 {
        network
            .add_neuron(neuron(id, Polarity::Excitatory))
            .unwrap();
    }
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 10_000,
        target_rate_hz: 1.0,
        target_input_rate: 1.0,
        intrinsic_adjustment_rate: 1.0,
        structural_adjustment_rate: 1.0,
        min_intrinsic_current: -10.0,
        max_intrinsic_current: 10.0,
        min_structural_drive: 0.0,
        max_structural_drive: 10.0,
    });
    let simulation = simulation(network, NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    let phases: BTreeSet<_> = simulation
        .network()
        .neuron_ids()
        .map(|neuron_id| simulation.next_homeostasis_update(neuron_id).unwrap())
        .collect();

    assert_eq!(simulation.pending_event_count(), 1_000);
    assert!(phases.len() > 900, "hash phases unexpectedly collided");
    assert!(phases.first().unwrap().as_micros() < 100);
    assert!(phases.last().unwrap().as_micros() > 9_900);
}

#[test]
fn multi_minute_local_homeostasis_stays_bounded_across_input_regimes() {
    let params = NeuronConfig {
        resting_potential: 0.0,
        reset_potential: 0.0,
        threshold: 1.0,
        membrane_tau_us: 1_000.0,
        refractory_period_us: 1,
        // A one-second trace makes the 10 Hz long-run target observable
        // despite the two deliberately different input temporal patterns.
        activity_trace_tau_us: 1_000_000.0,
        intrinsic: Default::default(),
    };
    let mut network = Network::new();
    for id in 1..=3 {
        network
            .add_neuron(
                Neuron::new(
                    NeuronId(id),
                    Position3D::ORIGIN,
                    Polarity::Excitatory,
                    None,
                    params,
                    SimTime::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
    }
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 100_000,
        target_rate_hz: 10.0,
        target_input_rate: 1.0,
        intrinsic_adjustment_rate: 0.01,
        structural_adjustment_rate: 0.001,
        min_intrinsic_current: -1_000.0,
        max_intrinsic_current: 1_000.0,
        min_structural_drive: 0.0,
        max_structural_drive: 10.0,
    });
    let mut simulation = simulation(network, NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    const HORIZON_US: u64 = 180_000_000;

    // Neuron 1 gets regular 10 Hz drive. Neuron 3 receives the same mean
    // rate in short bursts; neuron 2 remains isolated and must request
    // structure without manufacturing output through intrinsic current.
    for time in (0..=HORIZON_US).step_by(100_000) {
        simulation
            .schedule_external_input(SimTime(time), NeuronId(1), 1.1)
            .unwrap();
    }
    for burst_start in (0..=HORIZON_US).step_by(500_000) {
        for offset in [0, 20_000, 40_000, 60_000, 80_000] {
            let time = burst_start + offset;
            if time <= HORIZON_US {
                simulation
                    .schedule_external_input(SimTime(time), NeuronId(3), 1.1)
                    .unwrap();
            }
        }
    }

    simulation.run_until(SimTime(HORIZON_US)).unwrap();
    let regular = simulation.network().neuron(NeuronId(1)).unwrap();
    let isolated = simulation.network().neuron(NeuronId(2)).unwrap();
    let bursty = simulation.network().neuron(NeuronId(3)).unwrap();

    assert!(regular.intrinsic_current().abs() < 100.0);
    assert!(bursty.intrinsic_current().abs() < 100.0);
    assert_eq!(isolated.intrinsic_current(), 0.0);
    assert!(
        (0.1..0.2).contains(&isolated.structural_drive()),
        "isolated structural drive should remain below its clamp: {}",
        isolated.structural_drive()
    );
    assert!(simulation.event_log().iter().any(|event| matches!(
        event,
        ObservationEvent::SpikeEmitted(spike) if spike.neuron_id == NeuronId(1)
    )));
    assert!(simulation.event_log().iter().any(|event| matches!(
        event,
        ObservationEvent::SpikeEmitted(spike) if spike.neuron_id == NeuronId(3)
    )));
}

#[test]
fn sustained_input_raises_current_then_autonomous_spikes_apply_negative_feedback() {
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 1_000,
        target_rate_hz: 90.0,
        target_input_rate: 1.0,
        intrinsic_adjustment_rate: 5_000.0,
        structural_adjustment_rate: 0.0,
        min_intrinsic_current: 0.0,
        max_intrinsic_current: 2_000.0,
        min_structural_drive: 0.0,
        max_structural_drive: 10.0,
    });
    let mut simulation = simulation(one_neuron_network(), NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    for time in (0..=100_000).step_by(1_300) {
        simulation
            .schedule_external_input(SimTime(time), NeuronId(1), 0.1)
            .unwrap();
    }

    simulation.run_until(SimTime(100_000)).unwrap();
    let events = simulation.event_log();
    assert!(events.iter().any(|event| matches!(
        event,
        ObservationEvent::HomeostasisChanged {
            old_intrinsic_current,
            new_intrinsic_current,
            ..
        } if new_intrinsic_current > old_intrinsic_current
    )));
    let spike_times: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            ObservationEvent::SpikeEmitted(spike) => Some(spike.time),
            _ => None,
        })
        .collect();
    assert!(
        spike_times.iter().any(|time| time.as_micros() % 1_300 != 0),
        "{spike_times:?}"
    );
    assert!(events.iter().any(|event| matches!(
        event,
        ObservationEvent::HomeostasisChanged {
            firing_avg,
            old_intrinsic_current,
            new_intrinsic_current,
            ..
        } if (firing_avg - 90.0).abs() <= 5.0
            && new_intrinsic_current < old_intrinsic_current
    )));
    let current = simulation
        .network()
        .neuron(NeuronId(1))
        .unwrap()
        .intrinsic_current();
    assert!(
        (0.0..2_000.0).contains(&current),
        "feedback must settle below its safety clamp, got {current}"
    );
}

#[test]
fn isolated_cell_accumulates_structural_drive_without_artificial_spikes_over_time() {
    let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
        enabled: true,
        update_interval_us: 1_000,
        target_rate_hz: 20.0,
        target_input_rate: 1.0,
        intrinsic_adjustment_rate: 5_000.0,
        structural_adjustment_rate: 1.0,
        min_intrinsic_current: 0.0,
        max_intrinsic_current: 2_000.0,
        min_structural_drive: 0.0,
        max_structural_drive: 10.0,
    });
    let mut simulation = simulation(one_neuron_network(), NoPlasticity)
        .with_homeostasis(homeostasis)
        .unwrap();
    let report = simulation.run_until(SimTime(100_000)).unwrap();
    let neuron = simulation.network().neuron(NeuronId(1)).unwrap();

    assert_eq!(report.spikes_emitted, 0);
    assert_eq!(neuron.intrinsic_current(), 0.0);
    assert!(neuron.structural_drive() > 0.09);
}

#[test]
fn homeostasis_population_excludes_clocks_and_one_shot_maintenance() {
    let mut network = one_neuron_network();
    network.add_neuron(neuron(2, Polarity::Excitatory)).unwrap();
    let mut runtime = simulation(network, NoPlasticity)
        .with_homeostasis(LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 10_000,
            target_input_rate: 50.0,
            structural_adjustment_rate: 1.0,
            max_structural_drive: 100.0,
            ..HomeostasisConfig::default()
        }))
        .unwrap();
    runtime
        .set_homeostasis_population(Some(BTreeSet::from([NeuronId(1)])))
        .unwrap();
    assert!(runtime.next_homeostasis_update(NeuronId(1)).is_some());
    assert!(runtime.next_homeostasis_update(NeuronId(2)).is_none());
    runtime
        .schedule_homeostasis(SimTime(50_000), NeuronId(2))
        .unwrap();
    runtime
        .schedule_external_input(SimTime(100_000), NeuronId(2), 0.0)
        .unwrap();
    runtime.run_until(SimTime(100_000)).unwrap();
    assert_eq!(
        runtime
            .network()
            .neuron(NeuronId(2))
            .unwrap()
            .structural_drive(),
        0.0
    );
    assert!(
        runtime
            .network()
            .neuron(NeuronId(1))
            .unwrap()
            .structural_drive()
            > 0.0
    );
    assert!(runtime.event_log().iter().all(|event| !matches!(
        event,
        ObservationEvent::HomeostasisChanged {
            neuron_id: NeuronId(2),
            ..
        }
    )));
    runtime
        .set_homeostasis_population(Some(BTreeSet::from([NeuronId(2)])))
        .unwrap();
    let first = runtime.next_homeostasis_update(NeuronId(2)).unwrap();
    runtime.run_until(first).unwrap();
    assert!(
        runtime
            .network()
            .neuron(NeuronId(2))
            .unwrap()
            .structural_drive()
            <= 0.5001
    );
    assert!(runtime.next_homeostasis_update(NeuronId(1)).is_none());
    let before = runtime.network().clone();
    let booking = runtime.next_homeostasis_update(NeuronId(2));
    assert!(
        runtime
            .set_homeostasis_population(Some(BTreeSet::from([NeuronId(999)])))
            .is_err()
    );
    assert_eq!(runtime.network(), &before);
    assert_eq!(runtime.next_homeostasis_update(NeuronId(2)), booking);
    runtime.add_neuron(neuron(3, Polarity::Excitatory)).unwrap();
    assert!(runtime.next_homeostasis_update(NeuronId(3)).is_none());
    runtime.set_homeostasis_population(None).unwrap();
    assert!(runtime.next_homeostasis_update(NeuronId(3)).is_some());
    runtime
        .set_homeostasis_population(Some(BTreeSet::new()))
        .unwrap();
    assert!((1..=3).all(|id| runtime.next_homeostasis_update(NeuronId(id)).is_none()));
}

#[test]
fn invalid_external_input_is_rejected_before_scheduling() {
    let mut simulation = simulation(one_neuron_network(), NoPlasticity);

    assert!(matches!(
        simulation.schedule_external_input(SimTime::ZERO, NeuronId(1), f32::NAN),
        Err(SimulationError::NonFiniteInput { .. })
    ));
    assert_eq!(simulation.pending_event_count(), 0);
}
