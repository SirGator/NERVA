use nerva::{
    config::{LearningConfig, NeuronConfig, RuntimeConfig},
    core::{
        ModulationState, Network, Neuron, NeuronId, Polarity, ReceptorProfile, SimTime, Synapse,
        SynapseId,
    },
    learning::{PairStdp, PlasticityRule, ThreeFactorConfig, WeightUpdateMode},
    math::Position3D,
    neuromodulation::{ChemicalConfig, ChemicalSpecies},
    primitives::Weight,
    runtime::Simulation,
};
fn cell(id: u64, x: f32) -> Neuron {
    Neuron::new(
        NeuronId(id),
        Position3D::new(x, 0., 0.),
        Polarity::Excitatory,
        None,
        NeuronConfig {
            resting_potential: 0.,
            reset_potential: 0.,
            threshold: 1.,
            refractory_period_us: 1000,
            ..Default::default()
        },
        SimTime::ZERO,
    )
    .unwrap()
}
fn prepared(mode: WeightUpdateMode, weight: f32, signal: f32) -> (PairStdp, Synapse, Neuron) {
    let mut post = cell(2, 0.1);
    post.set_modulation(
        SimTime::ZERO,
        ModulationState {
            learning_signal: signal,
            ..Default::default()
        },
    )
    .unwrap();
    let mut edge = Synapse::new(
        SynapseId(7),
        NeuronId(1),
        NeuronId(2),
        Weight::new(weight).unwrap(),
        1000,
        true,
    )
    .unwrap();
    let mut rule = PairStdp::new(&LearningConfig {
        a_plus: 0.0001,
        a_minus: 0.00012,
        min_weight: 0.,
        max_weight: 2.,
        ..Default::default()
    })
    .with_three_factor(ThreeFactorConfig::default())
    .unwrap()
    .with_weight_update_mode(mode);
    rule.on_pre_arrival(&mut edge, &post, SimTime(1000));
    rule.on_post_spike(&post, std::slice::from_mut(&mut edge), SimTime(2000));
    (rule, edge, post)
}
fn integrated(mode: WeightUpdateMode, parts: u64) -> (f32, f64, f64) {
    let (mut rule, mut edge, post) = prepared(mode, 0.5, 1.);
    let mut requested = 0.;
    for index in 1..=parts {
        let at = SimTime(2000 + 100000 * index / parts);
        requested += rule
            .delayed_update_diagnostics(&edge, &post, at, true)
            .unwrap()
            .unrounded_delta;
        rule.advance_time(&mut edge, &post, at, true);
    }
    (
        edge.weight().get(),
        edge.learning_weight_residual(),
        requested,
    )
}
#[test]
fn accumulated_partitioned_updates_preserve_requested_mass_and_match_single_publication() {
    let direct = integrated(WeightUpdateMode::DirectF32, 1000);
    let single = integrated(WeightUpdateMode::Accumulated, 1);
    let split = integrated(WeightUpdateMode::Accumulated, 1000);
    assert_eq!(direct.0, 0.5);
    assert_eq!(direct.1, 0.);
    assert!(split.0 > 0.5);
    assert_eq!(single.0, split.0);
    for (weight, residual, requested) in [single, split] {
        assert!((f64::from(weight) - 0.5 + residual - requested).abs() < 1e-18);
    }
    // Tag decay is deliberately still f32, so requested masses need not be identical.
    assert!((single.2 - split.2).abs() < 1e-10);
}
#[test]
fn signed_updates_and_bounds_do_not_retain_saturation_debt() {
    for (weight, signal) in [(2., 1.), (0., -1.)] {
        let (mut rule, mut edge, post) = prepared(WeightUpdateMode::Accumulated, weight, signal);
        let d = rule
            .delayed_update_diagnostics(&edge, &post, SimTime(102000), true)
            .unwrap();
        assert_ne!(d.clipped_delta, 0.);
        rule.advance_time(&mut edge, &post, SimTime(102000), true);
        assert_eq!(edge.weight().get(), weight);
        assert_eq!(edge.learning_weight_residual(), 0.);
    }
    let (mut rule, mut edge, post) = prepared(WeightUpdateMode::Accumulated, 0.5, -1.);
    let d = rule
        .delayed_update_diagnostics(&edge, &post, SimTime(102000), true)
        .unwrap();
    rule.advance_time(&mut edge, &post, SimTime(102000), true);
    assert!(edge.weight().get() < 0.5);
    assert!(
        (f64::from(edge.weight().get()) - 0.5 + edge.learning_weight_residual()
            - d.unrounded_delta)
            .abs()
            < 1e-18
    );
}
#[test]
fn freeze_keeps_pending_mass_without_retroactive_learning() {
    let (mut rule, mut edge, post) = prepared(WeightUpdateMode::Accumulated, 0.5, 1.);
    rule.advance_time(&mut edge, &post, SimTime(2001), true);
    let carry = edge.learning_weight_residual();
    assert!(carry > 0.);
    rule.advance_time(&mut edge, &post, SimTime(102001), false);
    assert_eq!(edge.weight().get(), 0.5);
    assert_eq!(edge.learning_weight_residual(), carry);
    let d = rule
        .delayed_update_diagnostics(&edge, &post, SimTime(102002), true)
        .unwrap();
    assert!(d.unrounded_delta < 1e-9);
    rule.advance_time(&mut edge, &post, SimTime(102002), true);
    assert!((edge.learning_weight_residual() - carry - d.unrounded_delta).abs() < 1e-20);
}
#[test]
fn sub_f64_ulp_contributions_survive_and_authoritative_edits_clear_them() {
    let post = cell(2, 0.1);
    let mut edge = Synapse::new(
        SynapseId(7),
        NeuronId(1),
        NeuronId(2),
        Weight::new(0.5).unwrap(),
        1000,
        true,
    )
    .unwrap();
    let mut rule = PairStdp::new(&LearningConfig {
        a_plus: 1e-30,
        ..Default::default()
    })
    .with_weight_update_mode(WeightUpdateMode::Accumulated);
    rule.on_pre_arrival(&mut edge, &post, SimTime(1000));
    rule.on_post_spike(&post, std::slice::from_mut(&mut edge), SimTime(2000));
    assert_eq!(edge.weight().get(), 0.5);
    assert!(edge.learning_weight_residual() > 0.);
    let original = edge.clone();
    edge.set_weight_preserving_learning_residual(Weight::new(0.6).unwrap())
        .unwrap();
    edge.set_weight_preserving_learning_residual(Weight::new(0.5).unwrap())
        .unwrap();
    assert_eq!(edge, original);
    edge.set_weight(Weight::new(0.5).unwrap()).unwrap();
    assert_eq!(edge.learning_weight_residual(), 0.);
    assert_ne!(edge, original);
}
fn simulation(audited: bool) -> Simulation<PairStdp> {
    let mut network = Network::new();
    for (id, x) in [(1, 0.), (2, 0.1), (4, 0.15)] {
        let mut n = cell(id, x);
        if id == 2 {
            n.set_receptor_profile(ReceptorProfile {
                dopamine_d1_like: 1.,
                ..Default::default()
            })
            .unwrap();
        }
        network.add_neuron(n).unwrap();
    }
    network
        .add_synapse(
            Synapse::new(
                SynapseId(7),
                NeuronId(1),
                NeuronId(2),
                Weight::new(0.05).unwrap(),
                1000,
                true,
            )
            .unwrap(),
        )
        .unwrap();
    let rule = PairStdp::new(&LearningConfig::default())
        .with_three_factor(ThreeFactorConfig::default())
        .unwrap()
        .with_weight_update_mode(WeightUpdateMode::Accumulated);
    let mut sim = Simulation::new(network, rule, RuntimeConfig::default(), 1.)
        .unwrap()
        .with_neuromodulation(ChemicalConfig::default())
        .unwrap();
    if audited {
        sim.enable_learning_audit(100000).unwrap();
    }
    sim.schedule_external_input(SimTime(1000), NeuronId(1), 1.1)
        .unwrap();
    sim.schedule_external_input(SimTime(10000), NeuronId(2), 1.1)
        .unwrap();
    sim.schedule_chemical_release(
        SimTime(100000),
        NeuronId(4),
        ChemicalSpecies::Dopamine,
        0.002,
    )
    .unwrap();
    sim.run_until(SimTime(200000)).unwrap();
    sim
}
#[test]
fn precision_audit_is_neutral_and_balances_local_requested_and_published_mass() {
    let plain = simulation(false);
    let mut audited = simulation(true);
    assert!(plain.same_replay_state(&audited));
    let log = audited.disable_learning_audit();
    assert!(log.is_complete());
    let m = &log.delayed[&SynapseId(7)];
    let requested = m.positive_requested + m.negative_requested;
    let actual = m.positive_actual + m.negative_actual;
    let residual = m.last_residual.unwrap() - m.first_residual.unwrap();
    let clipped = m.positive_clipped + m.negative_clipped;
    let error = m.positive_rounding_error + m.negative_rounding_error;
    assert!((requested - actual - residual - clipped - error).abs() < 1e-15);
    assert!(m.positive_rounding_error - m.negative_rounding_error < 1e-15);
    assert!(
        audited
            .network()
            .synapse(SynapseId(7))
            .unwrap()
            .weight()
            .get()
            > 0.05
    );
}
