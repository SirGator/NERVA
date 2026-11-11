use nerva::{
    config::{LearningConfig, NeuronConfig},
    core::{ModulationState, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    learning::{PairStdp, PlasticityRule, ThreeFactorConfig},
    math::Position3D,
    primitives::Weight,
};
#[test]
fn override_changes_only_the_learning_product_and_rejects_nonfinite_values() {
    let mut post = Neuron::new(
        NeuronId(2),
        Position3D::ORIGIN,
        Polarity::Excitatory,
        None,
        NeuronConfig::default(),
        SimTime::ZERO,
    )
    .unwrap();
    post.set_modulation(
        SimTime::ZERO,
        ModulationState {
            learning_signal: -0.25,
            threshold_shift: 0.1,
            intrinsic_drive: 0.2,
            transmission_gain: 0.8,
        },
    )
    .unwrap();
    let before = post.clone();
    let mut edge = Synapse::new(
        SynapseId(1),
        NeuronId(1),
        NeuronId(2),
        Weight::new(0.5).unwrap(),
        1000,
        true,
    )
    .unwrap();
    let mut original = PairStdp::new(&LearningConfig::default())
        .with_three_factor(ThreeFactorConfig::default())
        .unwrap();
    original.on_pre_arrival(&mut edge, &post, SimTime(1000));
    original.on_post_spike(&post, std::slice::from_mut(&mut edge), SimTime(2000));
    let overridden = PairStdp::new(&LearningConfig::default())
        .with_three_factor(ThreeFactorConfig::default())
        .unwrap()
        .with_learning_signal_override(1.)
        .unwrap();
    let a = original
        .delayed_update_diagnostics(&edge, &post, SimTime(102000), true)
        .unwrap();
    let b = overridden
        .delayed_update_diagnostics(&edge, &post, SimTime(102000), true)
        .unwrap();
    assert_eq!(a.tag_integral_seconds, b.tag_integral_seconds);
    assert_eq!(a.unrounded_delta, -0.25 * b.unrounded_delta);
    assert_eq!(post, before);
    assert_eq!(overridden.learning_signal(&post), 1.);
    for invalid in [f32::NAN, f32::INFINITY, 1_000_001.] {
        assert!(
            PairStdp::new(&LearningConfig::default())
                .with_learning_signal_override(invalid)
                .is_err()
        );
    }
}

#[test]
fn replay_boundaries_are_neutral_and_count_against_the_audit_budget() {
    use nerva::{config::RuntimeConfig, core::Network, runtime::Simulation};
    fn setup(record: bool) -> Simulation<PairStdp> {
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
        let rule = PairStdp::new(&LearningConfig::default())
            .with_three_factor(ThreeFactorConfig::default())
            .unwrap();
        let mut sim = Simulation::new(network, rule, RuntimeConfig::default(), 1.).unwrap();
        if record {
            sim.enable_learning_replay_audit(2).unwrap();
        }
        for time in [1000, 2000, 3000] {
            sim.schedule_external_input(SimTime(time), NeuronId(1), 0.1)
                .unwrap();
        }
        sim.run_until(SimTime(4000)).unwrap();
        sim
    }
    let plain = setup(false);
    let mut recorded = setup(true);
    assert!(plain.same_replay_state(&recorded));
    let log = recorded.disable_learning_audit();
    assert_eq!(log.boundaries, vec![SimTime(1000), SimTime(2000)]);
    assert!(!log.is_complete());
}
