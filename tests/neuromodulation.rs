//! Public integration checks of shared neuron physics, local chemicals and learning.
use nerva::{
    config::{IntrinsicDynamicsConfig, LearningConfig, NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, ShortTermConfig, SimTime, Synapse, SynapseId},
    learning::{NoPlasticity, PairStdp, ThreeFactorConfig},
    math::Position3D,
    neuromodulation::{
        ChemicalConfig, ChemicalSpecies, ModulationState, ReceptorProfile, ReleaseProfile,
    },
    primitives::Weight,
    runtime::{ObservationEvent, Simulation},
};
fn params() -> NeuronConfig {
    NeuronConfig {
        resting_potential: 0.,
        reset_potential: 0.,
        threshold: 1.,
        refractory_period_us: 1_000,
        ..NeuronConfig::default()
    }
}
fn cell(id: u64, x: f32) -> Neuron {
    Neuron::new(
        NeuronId(id),
        Position3D::new(x, 0., 0.),
        Polarity::Excitatory,
        None,
        params(),
        SimTime::ZERO,
    )
    .unwrap()
}
fn tagged(receptors: ReceptorProfile, reverse: bool) -> Simulation<PairStdp> {
    let mut network = Network::new();
    for (id, x) in [(1, 0.), (2, 0.1), (3, 0.1), (4, 0.15)] {
        let mut n = cell(id, x);
        if id == 2 || id == 3 {
            n.set_receptor_profile(receptors).unwrap();
        }
        network.add_neuron(n).unwrap();
    }
    for (id, post) in [(7, 2), (8, 3)] {
        network
            .add_synapse(
                Synapse::new(
                    SynapseId(id),
                    NeuronId(1),
                    NeuronId(post),
                    Weight::new(0.05).unwrap(),
                    1_000,
                    true,
                )
                .unwrap(),
            )
            .unwrap();
    }
    let rule = PairStdp::new(&LearningConfig::default())
        .with_three_factor(ThreeFactorConfig::default())
        .unwrap();
    let mut sim = Simulation::new(network, rule, RuntimeConfig::default(), 1.)
        .unwrap()
        .with_neuromodulation(ChemicalConfig::default())
        .unwrap();
    sim.schedule_external_input(
        SimTime(if reverse { 10_000 } else { 1_000 }),
        NeuronId(1),
        1.1,
    )
    .unwrap();
    sim.schedule_external_input(
        SimTime(if reverse { 1_000 } else { 10_000 }),
        NeuronId(2),
        1.1,
    )
    .unwrap();
    sim.run_until(SimTime(50_000)).unwrap();
    assert_eq!(
        sim.network().synapse(SynapseId(7)).unwrap().weight().get(),
        0.05
    );
    let e = sim
        .network()
        .synapse(SynapseId(7))
        .unwrap()
        .learning_eligibility();
    assert!(if reverse { e < 0. } else { e > 0. });
    assert_eq!(
        sim.network()
            .synapse(SynapseId(8))
            .unwrap()
            .learning_eligibility(),
        0.
    );
    sim
}
#[test]
fn delayed_chemistry_selects_only_marked_receptive_synapses_and_respects_tag_sign() {
    for reverse in [false, true] {
        let mut sim = tagged(
            ReceptorProfile {
                dopamine_d1_like: 1.,
                ..ReceptorProfile::default()
            },
            reverse,
        );
        sim.schedule_chemical_release(
            SimTime(100_000),
            NeuronId(4),
            ChemicalSpecies::Dopamine,
            0.002,
        )
        .unwrap();
        sim.run_until(SimTime(200_000)).unwrap();
        let learned = sim.network().synapse(SynapseId(7)).unwrap().weight().get();
        assert!(
            if reverse {
                learned < 0.05
            } else {
                learned > 0.05
            },
            "signed tag was not consolidated: {learned}"
        );
        assert_eq!(
            sim.network().synapse(SynapseId(8)).unwrap().weight().get(),
            0.05,
            "unmarked neighbor changed"
        );
    }
}
#[test]
fn receptor_knockout_d2_opposition_and_frozen_weights_are_causal_controls() {
    let d1 = ReceptorProfile {
        dopamine_d1_like: 1.,
        ..ReceptorProfile::default()
    };
    let d2 = ReceptorProfile {
        dopamine_d2_like: 1.,
        ..ReceptorProfile::default()
    };
    for (profile, frozen, sign) in [
        (ReceptorProfile::default(), false, 0),
        (d1, true, 0),
        (d2, false, -1),
    ] {
        let mut sim = tagged(profile, false);
        if frozen {
            sim.freeze_learning();
        }
        sim.schedule_chemical_release(
            SimTime(100_000),
            NeuronId(4),
            ChemicalSpecies::Dopamine,
            0.002,
        )
        .unwrap();
        sim.run_until(SimTime(200_000)).unwrap();
        let weight = sim.network().synapse(SynapseId(7)).unwrap().weight().get();
        if sign == 0 {
            assert_eq!(weight, 0.05);
        } else {
            assert!(weight < 0.05);
        }
    }
}
#[test]
fn source_spikes_release_without_a_classical_synapse_and_empty_fields_create_no_clock() {
    let mut network = Network::new();
    let mut source = cell(1, 0.);
    source
        .set_release_profile(ReleaseProfile {
            dopamine: 0.002,
            ..ReleaseProfile::default()
        })
        .unwrap();
    let mut target = cell(2, 0.1);
    target
        .set_receptor_profile(ReceptorProfile {
            dopamine_d1_like: 1.,
            ..ReceptorProfile::default()
        })
        .unwrap();
    network.add_neuron(source).unwrap();
    network.add_neuron(target).unwrap();
    let mut sim = Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.)
        .unwrap()
        .with_neuromodulation(ChemicalConfig::default())
        .unwrap();
    assert_eq!(sim.pending_event_count(), 0);
    sim.schedule_external_input(SimTime(1_000), NeuronId(1), 1.1)
        .unwrap();
    sim.run_until(SimTime(10_000)).unwrap();
    assert_eq!(sim.network().synapse_count(), 0);
    assert_eq!(sim.chemical_field().unwrap().pulse_count(), 1);
    assert!(
        sim.network()
            .neuron(NeuronId(2))
            .unwrap()
            .modulation()
            .learning_signal
            > 0.
    );
    assert_eq!(
        sim.network().neuron(NeuronId(1)).unwrap().modulation(),
        ModulationState::default()
    );
    sim.run_until(SimTime(9_000_000)).unwrap();
    assert!(sim.chemical_field().unwrap().is_empty());
    assert_eq!(sim.pending_event_count(), 0);
    assert_eq!(
        sim.network().neuron(NeuronId(2)).unwrap().modulation(),
        ModulationState::default()
    );
}
#[test]
fn chemical_threshold_changes_trigger_firing_and_cancel_stale_intrinsic_bookings() {
    // A lowered threshold crosses an already charged cell at the chemical boundary.
    let mut network = Network::new();
    let mut n = cell(1, 0.);
    n.set_receptor_profile(ReceptorProfile {
        dopamine_d1_like: 8.,
        ..ReceptorProfile::default()
    })
    .unwrap();
    network.add_neuron(n).unwrap();
    let mut sim = Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.)
        .unwrap()
        .with_neuromodulation(ChemicalConfig::default())
        .unwrap();
    sim.schedule_external_input(SimTime(1_000), NeuronId(1), 0.9)
        .unwrap();
    sim.schedule_chemical_release(SimTime(2_000), NeuronId(1), ChemicalSpecies::Dopamine, 1.)
        .unwrap();
    sim.run_until(SimTime(2_000)).unwrap();
    assert!(
        sim.event_log()
            .iter()
            .any(|e| matches!(e,ObservationEvent::SpikeEmitted(s) if s.time==SimTime(2_000)))
    );
    assert!(sim.network().neuron(NeuronId(1)).unwrap().threshold() > 0.);
    // A raised threshold must remove the old autonomous crossing.
    let mut network = Network::new();
    let mut n = Neuron::new(
        NeuronId(1),
        Position3D::ORIGIN,
        Polarity::Excitatory,
        None,
        NeuronConfig {
            intrinsic: IntrinsicDynamicsConfig {
                intrinsic_drive: 80.,
                ..Default::default()
            },
            ..params()
        },
        SimTime::ZERO,
    )
    .unwrap();
    n.set_receptor_profile(ReceptorProfile {
        dopamine_d2_like: 8.,
        ..Default::default()
    })
    .unwrap();
    network.add_neuron(n).unwrap();
    let mut sim = Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.)
        .unwrap()
        .with_neuromodulation(ChemicalConfig::default())
        .unwrap();
    let old = sim.next_intrinsic_spike(NeuronId(1)).unwrap();
    assert!(old > SimTime(5_000));
    sim.schedule_chemical_release(SimTime(5_000), NeuronId(1), ChemicalSpecies::Dopamine, 1.)
        .unwrap();
    sim.run_until(old).unwrap();
    assert!(
        !sim.event_log()
            .iter()
            .any(|e| matches!(e, ObservationEvent::SpikeEmitted(_))),
        "obsolete predicted spike fired"
    );
    assert!(
        sim.next_intrinsic_spike(NeuronId(1))
            .is_none_or(|t| t > old)
    );
}
#[test]
fn facilitation_changes_real_inflight_amplitudes_and_preserves_polarity() {
    for polarity in [Polarity::Excitatory, Polarity::Inhibitory] {
        let mut network = Network::new();
        network
            .add_neuron(
                Neuron::new(
                    NeuronId(1),
                    Position3D::ORIGIN,
                    polarity,
                    None,
                    params(),
                    SimTime::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
        let mut target = cell(2, 0.1);
        target.set_threshold_clamped(100., 0.01, 101.).unwrap();
        network.add_neuron(target).unwrap();
        let mut edge = Synapse::new(
            SynapseId(7),
            NeuronId(1),
            NeuronId(2),
            Weight::new(1.).unwrap(),
            1_000,
            true,
        )
        .unwrap();
        edge.set_short_term(Some(ShortTermConfig {
            utilization: 0.05,
            facilitation_increment: 0.15,
            recovery_tau_us: 100_000.,
            facilitation_tau_us: 500_000.,
        }))
        .unwrap();
        network.add_synapse(edge).unwrap();
        let mut sim = Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.).unwrap();
        for at in [1_000, 11_000] {
            sim.schedule_external_input(SimTime(at), NeuronId(1), 1.1)
                .unwrap();
        }
        sim.run_until(SimTime(20_000)).unwrap();
        let arrivals: Vec<_> = sim
            .event_log()
            .iter()
            .filter_map(|e| {
                if let ObservationEvent::SynapticArrival { amplitude, .. } = e {
                    Some(*amplitude)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(arrivals.len(), 2);
        assert!(arrivals[1].abs() > arrivals[0].abs());
        assert!(arrivals.iter().all(|a| a.signum() == polarity.sign()));
        assert_eq!(
            sim.network()
                .synapse(SynapseId(7))
                .unwrap()
                .short_term()
                .unwrap()
                .1
                .updated_at,
            Some(SimTime(11_000))
        );
    }
}
#[test]
fn live_short_term_configuration_rejects_partial_batches_atomically() {
    let mut sim = tagged(ReceptorProfile::default(), false);
    let before = sim.network().clone();
    let config = ShortTermConfig {
        utilization: 0.05,
        facilitation_increment: 0.15,
        recovery_tau_us: 100_000.,
        facilitation_tau_us: 500_000.,
    };
    assert!(
        sim.set_short_term_profiles(&[
            (SynapseId(7), Some(config)),
            (SynapseId(999), Some(config))
        ])
        .is_err()
    );
    assert_eq!(sim.network(), &before);
    sim.set_short_term_profiles(&[(SynapseId(7), Some(config))])
        .unwrap();
    assert!(
        sim.network()
            .synapse(SynapseId(7))
            .unwrap()
            .short_term()
            .is_some()
    );
}

#[test]
fn chemical_capacity_failure_poisoning_is_explicit_and_replay_includes_field_state() {
    let build = || {
        let mut network = Network::new();
        let mut source = cell(1, 0.);
        source
            .set_release_profile(ReleaseProfile {
                dopamine: 0.01,
                ..Default::default()
            })
            .unwrap();
        network.add_neuron(source).unwrap();
        Simulation::new(
            network,
            PairStdp::new(&LearningConfig::default()),
            RuntimeConfig::default(),
            1.,
        )
        .unwrap()
        .with_neuromodulation(ChemicalConfig {
            max_pulses: 1,
            ..Default::default()
        })
        .unwrap()
    };
    let mut a = build();
    let mut b = build();
    for sim in [&mut a, &mut b] {
        sim.schedule_external_input(SimTime(1_000), NeuronId(1), 1.1)
            .unwrap();
        sim.run_until(SimTime(2_000)).unwrap();
    }
    assert!(a.same_replay_state(&b));
    b.schedule_chemical_release(
        SimTime(3_000),
        NeuronId(1),
        ChemicalSpecies::Dopamine,
        0.001,
    )
    .unwrap();
    assert!(!a.same_replay_state(&b));
    a.schedule_external_input(SimTime(11_000), NeuronId(1), 1.1)
        .unwrap();
    assert!(a.run_until(SimTime(12_000)).is_err());
    assert!(
        a.schedule_external_input(SimTime(13_000), NeuronId(1), 1.1)
            .is_err()
    );
}

#[test]
fn cell_profiles_change_actual_electrical_response_without_a_second_neuron_engine() {
    use nerva::cell_profiles::{AreaField, electrical_profile};
    let field = AreaField::new(0., 1., 0.).unwrap();
    let mut observed = Vec::new();
    for x in [1. / 3., 2. / 3.] {
        let config = electrical_profile(
            params(),
            field.at_normalized(x).unwrap(),
            Polarity::Excitatory,
        )
        .unwrap();
        let mut neuron = Neuron::new(
            NeuronId(1),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            None,
            config,
            SimTime::ZERO,
        )
        .unwrap();
        let mut spikes = 0;
        for at in [1_000, 51_000, 101_000] {
            spikes += usize::from(neuron.integrate_input(SimTime(at), 1.1).unwrap().is_some());
        }
        observed.push(spikes);
    }
    assert!(
        observed[1] > observed[0],
        "DG/CA3-like profiles did not change real responses: {observed:?}"
    );
}

#[test]
fn optional_learning_audit_is_neutral_and_reports_overflow() {
    use nerva::runtime::{LearningAuditEvent, LearningPhase};
    let receptors = ReceptorProfile {
        dopamine_d1_like: 1.,
        ..Default::default()
    };
    let mut plain = tagged(receptors, false);
    let mut audited = tagged(receptors, false);
    assert!(audited.enable_learning_audit(0).is_err());
    audited.enable_learning_audit(100_000).unwrap();
    assert!(audited.enable_learning_audit(1).is_err());
    for sim in [&mut plain, &mut audited] {
        sim.schedule_chemical_release(
            SimTime(100_000),
            NeuronId(4),
            ChemicalSpecies::Dopamine,
            0.002,
        )
        .unwrap();
        sim.schedule_external_input(SimTime(210_000), NeuronId(1), 1.1)
            .unwrap();
        sim.schedule_external_input(SimTime(220_000), NeuronId(2), 1.1)
            .unwrap();
        sim.run_until(SimTime(300_000)).unwrap();
    }
    assert!(plain.same_replay_state(&audited));
    assert_eq!(plain.event_log(), audited.event_log());
    let log = audited.drain_learning_audit();
    assert!(log.is_complete());
    assert!(log.events.iter().any(|e| matches!(
        e,
        LearningAuditEvent::ChemicalSample {
            neuron: NeuronId(1),
            ..
        }
    )));
    assert!(log.events.iter().any(|e| matches!(e, LearningAuditEvent::Trace { phase: LearningPhase::Delayed, delayed: Some(u), .. } if u.unrounded_delta > 0.)));
    assert!(log.events.iter().any(|e| matches!(e, LearningAuditEvent::Trace { phase: LearningPhase::PostSpike, tag_before, tag_after, .. } if tag_after > tag_before)));
    assert!(audited.drain_learning_audit().events.is_empty());
    audited.disable_learning_audit();
    audited.enable_learning_audit(1).unwrap();
    for sim in [&mut plain, &mut audited] {
        sim.run_until(SimTime(350_000)).unwrap();
    }
    assert!(plain.same_replay_state(&audited));
    let overflow = audited.disable_learning_audit();
    assert_eq!(overflow.events.len() + overflow.delayed.len(), 1);
    assert!(overflow.dropped_events > 0 && !overflow.is_complete());
}

#[test]
fn delayed_modulated_weight_changes_a_later_spike_under_matched_state() {
    let receptors = ReceptorProfile {
        dopamine_d1_like: 1.,
        ..Default::default()
    };
    let mut learned = tagged(receptors, false);
    let mut restored = tagged(receptors, false);
    for sim in [&mut learned, &mut restored] {
        sim.schedule_chemical_release(
            SimTime(100_000),
            NeuronId(4),
            ChemicalSpecies::Dopamine,
            0.002,
        )
        .unwrap();
        sim.run_until(SimTime(11_000_000)).unwrap();
        sim.freeze_learning();
        sim.drain_event_log();
    }
    assert!(learned.same_replay_state(&restored));
    assert!(learned.chemical_field().unwrap().is_empty());
    let trained = learned
        .network()
        .synapse(SynapseId(7))
        .unwrap()
        .weight()
        .get();
    assert!(trained > 0.05);
    restored
        .update_synapse(SynapseId(7), |s| s.set_weight(Weight::new(0.05).unwrap()))
        .unwrap();
    // A fixed near-threshold bench input, deliberately diagnostic. It is not
    // task learning or an autonomous success. Only the retained weight differs.
    let attenuation = (-0.1f32).exp();
    let background = 1. - attenuation * (trained + 0.05) / 2.;
    for sim in [&mut learned, &mut restored] {
        sim.schedule_external_input(SimTime(11_010_000), NeuronId(1), 1.1)
            .unwrap();
        sim.schedule_external_input(SimTime(11_011_000), NeuronId(2), background)
            .unwrap();
        sim.run_until(SimTime(11_020_000)).unwrap();
    }
    let fired = |sim: &Simulation<PairStdp>| {
        sim.event_log().iter().any(|e| {
            matches!(e,
        ObservationEvent::SpikeEmitted(spike) if spike.neuron_id == NeuronId(2))
        })
    };
    assert!(fired(&learned));
    assert!(!fired(&restored));
}
