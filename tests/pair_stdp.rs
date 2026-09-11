use nerva::{
    config::{LearningConfig, NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    learning::{DecayingTrace, PairStdp},
    math::Position3D,
    primitives::Weight,
    runtime::{ObservationEvent, Simulation},
};

const PRE: NeuronId = NeuronId(1);
const POST: NeuronId = NeuronId(2);
const CONNECTION: SynapseId = SynapseId(7);

#[derive(Clone, Copy)]
enum PairOrder {
    Causal,
    AntiCausal,
}

struct PairOutcome {
    weight: Weight,
    pre_trace: f32,
    post_trace: Option<DecayingTrace>,
    transmission_count: u64,
    pre_spikes: u64,
    post_spikes: u64,
    log: Vec<ObservationEvent>,
}

fn neuron(id: NeuronId) -> Neuron {
    Neuron::new(
        id,
        Position3D::ORIGIN,
        Polarity::Excitatory,
        None,
        NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            membrane_tau_us: 1_000_000.0,
            refractory_period_us: 0,
            activity_trace_tau_us: 1_000_000.0,
            intrinsic: Default::default(),
        },
        SimTime::ZERO,
    )
    .expect("valid integration-test neuron")
}

fn learning_config() -> LearningConfig {
    LearningConfig {
        enabled: true,
        a_plus: 0.2,
        a_minus: 0.1,
        tau_plus_us: 100.0,
        tau_minus_us: 200.0,
        stdp_window_us: 500,
        min_weight: 0.2,
        max_weight: 0.8,
        ..LearningConfig::default()
    }
}

fn run_pair(
    config: LearningConfig,
    initial_weight: f32,
    order: PairOrder,
    freeze_before_run: bool,
) -> PairOutcome {
    let mut network = Network::new();
    network.add_neuron(neuron(PRE)).expect("unique pre cell");
    network.add_neuron(neuron(POST)).expect("unique post cell");
    network
        .add_synapse(
            Synapse::new(
                CONNECTION,
                PRE,
                POST,
                Weight::new(initial_weight).unwrap(),
                10,
                true,
            )
            .expect("valid plastic connection"),
        )
        .expect("unique synapse");

    let rule = PairStdp::try_from_config(&config).expect("valid STDP parameters");
    let mut simulation =
        Simulation::new(network, rule, RuntimeConfig::default(), 1.0).expect("valid runtime");
    if freeze_before_run {
        simulation.freeze_learning();
    }

    match order {
        PairOrder::Causal => {
            simulation
                .schedule_external_input(SimTime::ZERO, PRE, 1.0)
                .expect("valid pre stimulus");
            // The external contribution alone is subthreshold. The post cell
            // fires only because the earlier synaptic arrival is still present.
            simulation
                .schedule_external_input(SimTime(20), POST, 0.6)
                .expect("valid post stimulus");
        }
        PairOrder::AntiCausal => {
            simulation
                .schedule_external_input(SimTime::ZERO, POST, 1.0)
                .expect("valid post stimulus");
            simulation
                .schedule_external_input(SimTime(10), PRE, 1.0)
                .expect("valid pre stimulus");
        }
    }

    simulation.run().expect("pair scenario succeeds");
    let (network, rule, _, log) = simulation.into_parts();
    let synapse = network
        .synapse(CONNECTION)
        .expect("synapse remains present");

    PairOutcome {
        weight: synapse.weight(),
        pre_trace: synapse.pre_trace(),
        post_trace: rule.post_trace(POST).copied(),
        transmission_count: synapse.transmission_count(),
        pre_spikes: network.neuron(PRE).expect("pre cell exists").spike_count(),
        post_spikes: network
            .neuron(POST)
            .expect("post cell exists")
            .spike_count(),
        log: log.into_events(),
    }
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1.0e-6,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn runtime_pair_order_selects_potentiation_or_depression() {
    let config = learning_config();
    let causal = run_pair(config, 0.5, PairOrder::Causal, false);
    let anti_causal = run_pair(config, 0.5, PairOrder::AntiCausal, false);

    let expected_ltp = 0.5 + config.a_plus * (-(10.0_f32) / config.tau_plus_us).exp();
    let expected_ltd = 0.5 - config.a_minus * (-(20.0_f32) / config.tau_minus_us).exp();
    assert_close(causal.weight.get(), expected_ltp);
    assert_close(anti_causal.weight.get(), expected_ltd);
    assert!(causal.weight.get() > 0.5);
    assert!(anti_causal.weight.get() < 0.5);
    assert!(
        causal
            .log
            .iter()
            .any(|event| matches!(event, ObservationEvent::WeightChanged { .. }))
    );
    assert!(
        anti_causal
            .log
            .iter()
            .any(|event| matches!(event, ObservationEvent::WeightChanged { .. }))
    );
}

#[test]
fn runtime_freeze_keeps_weights_and_learning_traces_fixed_while_spikes_propagate() {
    let frozen = run_pair(learning_config(), 0.5, PairOrder::Causal, true);

    assert_eq!(frozen.weight, Weight::new(0.5).unwrap());
    assert_eq!(frozen.pre_trace, 0.0);
    assert_eq!(frozen.post_trace, None);
    assert_eq!(frozen.transmission_count, 1);
    assert_eq!(frozen.pre_spikes, 1);
    assert_eq!(frozen.post_spikes, 1);
    assert!(
        !frozen
            .log
            .iter()
            .any(|event| matches!(event, ObservationEvent::WeightChanged { .. }))
    );
}

#[test]
fn runtime_stdp_cannot_escape_configured_weight_bounds() {
    let mut config = learning_config();
    config.a_plus = 10.0;
    config.a_minus = 10.0;

    let potentiated = run_pair(config, 0.5, PairOrder::Causal, false);
    let depressed = run_pair(config, 0.5, PairOrder::AntiCausal, false);

    assert_eq!(potentiated.weight, Weight::new(config.max_weight).unwrap());
    assert_eq!(depressed.weight, Weight::new(config.min_weight).unwrap());
}
