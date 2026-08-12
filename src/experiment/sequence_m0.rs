//! Controlled A→B→C→D training, frozen A-only probe, and G1–G4 comparison.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    config::{DsvlmConfig, NeuronConfig},
    core::{Network, Neuron, NeuronId, NeuronRole, Polarity, SimTime, Synapse, SynapseId},
    environment::{Environment, Observation, Pattern, SequenceEnvironment},
    learning::{NoPlasticity, PairStdp, PlasticityRule},
    math::{Position3D, conduction_delay_us},
    metrics::{FiringMetrics, SequenceMetrics, WeightMetrics},
    nerves::{Fiber, FiberDirection, FiberId, Mapping, Routing},
    roots::{PatternInputRoot, RootChannel, RootId},
    runtime::{ObservationEvent, Simulation},
    transduction::{Encoder, PatternEncoder},
};

use super::{
    Comparison, ExperimentError, M0Group, M0GroupResult, M0Metrics, M0StudyConfig, M0StudyReport,
    M0SuccessReport, rng::DeterministicRng,
};

const PATTERNS: [Pattern; 4] = [Pattern::A, Pattern::B, Pattern::C, Pattern::D];
const TOPOLOGY_STREAM: u64 = 0x746f_706f_6c6f_6779;
const INPUT_STREAM: u64 = 0x696e_7075_742d_6d30;

// The four pattern cells occupy a regular tetrahedron. All twelve directed
// transition candidates therefore begin with identical geometry.
const PATTERN_POSITIONS: [Position3D; 4] = [
    Position3D::new(0.0, 0.0, 0.0),
    Position3D::new(1.0, 0.0, 0.0),
    Position3D::new(0.5, 0.866_025_4, 0.0),
    Position3D::new(0.5, 0.288_675_13, 0.816_496_6),
];

type PatternNeuronMaps = (
    Network,
    BTreeMap<Pattern, NeuronId>,
    BTreeMap<Pattern, NeuronId>,
);

struct M0InputPipeline {
    root: PatternInputRoot,
    encoder: PatternEncoder,
    mapping: Mapping,
}

impl M0InputPipeline {
    fn schedule<R: PlasticityRule>(
        &self,
        simulation: &mut Simulation<R>,
        at: SimTime,
        pattern: Pattern,
    ) -> Result<(), ExperimentError> {
        let observation = Observation::Pattern { at, pattern };
        let spikes = self
            .encoder
            .encode(&observation)
            .map_err(|error| ExperimentError::new("input encoding", error.to_string()))?;
        for spike in spikes {
            let impulse = Routing::sensory(&self.mapping, self.root.root().id, spike)
                .map_err(|error| ExperimentError::new("input routing", error.to_string()))?
                .ok_or_else(|| ExperimentError::new("input routing", "unmapped pattern channel"))?;
            simulation
                .schedule_external_input(impulse.arrives_at, impulse.neuron, impulse.amplitude)
                .map_err(runtime_error)?;
        }
        Ok(())
    }
}

/// Reproducible parameters for all controlled M0 groups.
#[derive(Clone, Debug, PartialEq)]
pub struct M0ExperimentConfig {
    /// Shared validated simulation configuration and seed.
    pub dsvlm: DsvlmConfig,
    /// Number of A→B→C→D repetitions during training.
    pub training_repetitions: usize,
    /// Time between consecutive training symbols.
    pub symbol_interval_us: u64,
    /// Additional time between sequence repetitions.
    pub pause_us: u64,
    /// Delay on feed-forward and recurrent sequence synapses.
    pub sequence_delay_us: u64,
    /// Strong fixed sensory-to-pattern weight.
    pub sensory_weight: f32,
    /// Initial plastic weight between successive pattern populations.
    pub recurrent_weight: f32,
    /// Symmetric seeded half-width around the recurrent starting weight.
    pub recurrent_weight_jitter: f32,
    /// Fixed drive from each pattern neuron to the stabilizing interneuron.
    pub inhibitory_drive_weight: f32,
    /// Fixed feedback magnitude from the interneuron to each pattern neuron.
    pub inhibitory_feedback_weight: f32,
    /// Delay of both legs of the local inhibitory feedback loop.
    pub inhibitory_delay_us: u64,
    /// Probe duration after presenting only A.
    pub probe_duration_us: u64,
    /// Period of local threshold-maintenance events in G4.
    pub homeostasis_interval_us: u64,
    /// Half-width of each expected B/C/D probe window.
    pub probe_tolerance_us: u64,
    /// Final no-spike interval required for a stable probe.
    pub quiet_window_us: u64,
}

impl Default for M0ExperimentConfig {
    fn default() -> Self {
        let mut dsvlm = DsvlmConfig::default();
        dsvlm.network.excitatory_neurons = 8;
        dsvlm.network.inhibitory_neurons = 1;
        dsvlm.network.connection_probability = 1.0;
        dsvlm.network.seed = 7;
        dsvlm.network.distance_decay_length = 1_000_000.0;
        dsvlm.network.conduction_velocity = 0.0001;
        dsvlm.network.max_weight = 2.0;
        dsvlm.learning.max_weight = 2.0;
        dsvlm.learning.a_plus = 0.04;
        dsvlm.learning.a_minus = 0.045;
        dsvlm.learning.tau_plus_us = 20_000.0;
        dsvlm.learning.tau_minus_us = 20_000.0;
        dsvlm.learning.stdp_window_us = 2_000;
        dsvlm.neuron = NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            membrane_tau_us: 20_000.0,
            refractory_period_us: 1_000,
            activity_trace_tau_us: 250_000.0,
        };
        dsvlm.learning.homeostasis.min_threshold = 0.8;
        dsvlm.learning.homeostasis.max_threshold = 1.2;
        dsvlm.learning.homeostasis.target_rate_hz = 10.0;
        dsvlm.learning.homeostasis.adjustment_rate = 0.0001;

        Self {
            dsvlm,
            training_repetitions: 24,
            symbol_interval_us: 10_000,
            pause_us: 30_000,
            sequence_delay_us: 10_000,
            sensory_weight: 1.2,
            recurrent_weight: 0.12,
            recurrent_weight_jitter: 0.005,
            inhibitory_drive_weight: 1.1,
            inhibitory_feedback_weight: 0.05,
            inhibitory_delay_us: 1_000,
            probe_duration_us: 60_000,
            homeostasis_interval_us: 50_000,
            probe_tolerance_us: 2,
            quiet_window_us: 10_000,
        }
    }
}

impl M0ExperimentConfig {
    /// Validates experiment timing, amplitudes and the complete simulation config.
    pub fn validate(&self) -> Result<(), ExperimentError> {
        self.dsvlm
            .validate()
            .map_err(|error| ExperimentError::new("configuration", error.to_string()))?;
        if self.training_repetitions == 0 {
            return Err(ExperimentError::new(
                "configuration",
                "training_repetitions must be greater than zero",
            ));
        }
        for (name, value) in [
            ("symbol_interval_us", self.symbol_interval_us),
            ("sequence_delay_us", self.sequence_delay_us),
            ("inhibitory_delay_us", self.inhibitory_delay_us),
            ("probe_duration_us", self.probe_duration_us),
            ("homeostasis_interval_us", self.homeostasis_interval_us),
            ("quiet_window_us", self.quiet_window_us),
        ] {
            if value == 0 {
                return Err(ExperimentError::new(
                    "configuration",
                    format!("{name} must be greater than zero"),
                ));
            }
        }
        if self.sequence_delay_us != self.symbol_interval_us {
            return Err(ExperimentError::new(
                "configuration",
                "sequence_delay_us must equal symbol_interval_us for the M0 probe windows",
            ));
        }
        if self.probe_tolerance_us >= self.sequence_delay_us / 2 {
            return Err(ExperimentError::new(
                "configuration",
                "probe_tolerance_us must be less than half the transition delay",
            ));
        }
        if self.quiet_window_us > self.probe_duration_us {
            return Err(ExperimentError::new(
                "configuration",
                "quiet_window_us must not exceed probe_duration_us",
            ));
        }
        let required_probe_duration = self
            .sequence_delay_us
            .checked_mul(3)
            .and_then(|duration| duration.checked_add(self.quiet_window_us))
            // One sensory-fiber and one sensory-synapse microsecond precede A.
            .and_then(|duration| duration.checked_add(2))
            .ok_or_else(|| {
                ExperimentError::new("configuration", "required probe duration overflows time")
            })?;
        if self.probe_duration_us < required_probe_duration {
            return Err(ExperimentError::new(
                "configuration",
                format!(
                    "probe_duration_us must be at least {required_probe_duration} to observe D and a final quiet window"
                ),
            ));
        }
        let geometric_delay = conduction_delay_us(
            PATTERN_POSITIONS[0].distance_to(PATTERN_POSITIONS[1]),
            self.dsvlm.network.conduction_velocity,
            1,
        )
        .map_err(|error| ExperimentError::new("configuration", error.to_string()))?;
        if geometric_delay != self.sequence_delay_us {
            return Err(ExperimentError::new(
                "configuration",
                format!(
                    "sequence_delay_us must equal the geometry-derived delay ({geometric_delay} us)"
                ),
            ));
        }
        let repetitions = u64::try_from(self.training_repetitions)
            .map_err(|_| ExperimentError::new("configuration", "training repetitions overflow"))?;
        self.training_repetitions
            .checked_mul(PATTERNS.len())
            .ok_or_else(|| {
                ExperimentError::new("configuration", "training pattern capacity overflows")
            })?;
        let sequence_span = self
            .symbol_interval_us
            .checked_mul(PATTERNS.len() as u64)
            .and_then(|value| value.checked_add(self.pause_us))
            .ok_or_else(|| ExperimentError::new("configuration", "sequence span overflows time"))?;
        let training_span = sequence_span.checked_mul(repetitions).ok_or_else(|| {
            ExperimentError::new("configuration", "training horizon overflows time")
        })?;
        training_span
            .checked_add(self.quiet_window_us)
            .and_then(|value| value.checked_add(self.pause_us))
            .and_then(|value| value.checked_add(self.probe_duration_us))
            .ok_or_else(|| {
                ExperimentError::new("configuration", "experiment horizon overflows time")
            })?;
        for (name, weight) in [
            ("sensory_weight", self.sensory_weight),
            ("recurrent_weight", self.recurrent_weight),
            ("inhibitory_drive_weight", self.inhibitory_drive_weight),
            (
                "inhibitory_feedback_weight",
                self.inhibitory_feedback_weight,
            ),
        ] {
            if !weight.is_finite()
                || weight < self.dsvlm.network.min_weight
                || weight > self.dsvlm.network.max_weight
            {
                return Err(ExperimentError::new(
                    "configuration",
                    format!("{name} must lie inside the network weight bounds"),
                ));
            }
        }
        if !self.recurrent_weight_jitter.is_finite() || self.recurrent_weight_jitter < 0.0 {
            return Err(ExperimentError::new(
                "configuration",
                "recurrent_weight_jitter must be finite and non-negative",
            ));
        }
        let recurrent_min = self.recurrent_weight - self.recurrent_weight_jitter;
        let recurrent_max = self.recurrent_weight + self.recurrent_weight_jitter;
        if recurrent_min < self.dsvlm.learning.min_weight
            || recurrent_max > self.dsvlm.learning.max_weight
        {
            return Err(ExperimentError::new(
                "configuration",
                "the recurrent weight interval must lie inside the learning weight bounds",
            ));
        }
        if self.dsvlm.network.excitatory_neurons != 8 || self.dsvlm.network.inhibitory_neurons != 1
        {
            return Err(ExperimentError::new(
                "configuration",
                "the canonical M0 topology requires 8 excitatory and 1 inhibitory neuron",
            ));
        }
        if self.dsvlm.network.connection_probability != 1.0 {
            return Err(ExperimentError::new(
                "configuration",
                "the canonical M0 assay gives every directed pattern transition an equal candidate synapse; connection_probability must be one",
            ));
        }
        Ok(())
    }
}

/// One executable M0 group assembled from fresh initial state.
pub struct M0Experiment {
    config: M0ExperimentConfig,
    group: M0Group,
}

impl M0Experiment {
    /// Creates and validates one controlled group.
    pub fn new(config: M0ExperimentConfig, group: M0Group) -> Result<Self, ExperimentError> {
        config.validate()?;
        Ok(Self { config, group })
    }

    /// Runs training, freezes weights, presents A, and calculates measurements.
    pub fn run(&self) -> Result<M0GroupResult, ExperimentError> {
        let mut group_config = self.config.dsvlm.clone();
        group_config.learning.enabled = self.group != M0Group::OrderedFixed;
        group_config.learning.homeostasis.enabled = self.group == M0Group::OrderedHomeostasis;

        let (network, sensory_neurons, pattern_neurons) = build_network(&self.config)?;
        let initial_weights: Vec<_> = network.synapses().map(|synapse| synapse.weight()).collect();
        let input = build_input_pipeline(&sensory_neurons, self.config.sensory_weight)?;
        let rule = PairStdp::try_from_config(&group_config.learning)
            .map_err(|error| ExperimentError::new("learning", format!("{error:?}")))?;
        let mut simulation =
            Simulation::from_config(network, rule, &group_config).map_err(runtime_error)?;

        let training_patterns = self.training_patterns();
        let mut scheduled_patterns = Vec::with_capacity(training_patterns.len());
        let mut training_time = SimTime::ZERO;
        for (index, pattern) in training_patterns.iter().enumerate() {
            scheduled_patterns.push((training_time, *pattern));
            training_time = checked_advance(
                training_time,
                self.config.symbol_interval_us,
                "training symbol",
            )?;
            if (index + 1) % PATTERNS.len() == 0 {
                training_time =
                    checked_advance(training_time, self.config.pause_us, "training pause")?;
            }
        }
        let mut environment = SequenceEnvironment::new(scheduled_patterns);
        for observation in environment.observations(training_time) {
            let Observation::Pattern { at, pattern } = observation else {
                unreachable!("SequenceEnvironment emits only pattern observations")
            };
            input.schedule(&mut simulation, at, pattern)?;
        }
        debug_assert!(environment.is_exhausted());

        if self.group == M0Group::OrderedHomeostasis {
            let mut at = SimTime(self.config.homeostasis_interval_us);
            while at < training_time {
                for neuron in pattern_neurons.values().copied() {
                    simulation
                        .schedule_homeostasis(at, neuron)
                        .map_err(runtime_error)?;
                }
                let Some(next) = at.checked_add_us(self.config.homeostasis_interval_us) else {
                    break;
                };
                at = next;
            }
        }

        // Never let a recurrent failure run forever. Residual events after this
        // explicit settling horizon are retained as a failed stability metric.
        let training_deadline = checked_advance(
            training_time,
            self.config.quiet_window_us,
            "training settling horizon",
        )?;
        simulation
            .run_until(training_deadline)
            .map_err(runtime_error)?;
        let training_events = simulation.event_log().to_vec();
        let training_stable = simulation.pending_event_count() == 0
            && !training_events.iter().any(|event| {
                matches!(
                    event,
                    ObservationEvent::SpikeEmitted(spike) if spike.time >= training_time
                )
            });
        let trained_network = simulation.network().clone();
        let trained_weights: Vec<_> = trained_network
            .synapses()
            .map(|synapse| synapse.weight())
            .collect();
        let probe_start =
            checked_advance(training_deadline, self.config.pause_us, "probe separation")?;

        let first_probe = run_frozen_probe(
            trained_network.clone(),
            &input,
            &pattern_neurons,
            &self.config,
            probe_start,
        )?;
        let second_probe = run_frozen_probe(
            trained_network,
            &input,
            &pattern_neurons,
            &self.config,
            probe_start,
        )?;
        let frozen_probe_replay_identical = first_probe.events == second_probe.events
            && first_probe.final_network == second_probe.final_network;
        let final_weights: Vec<_> = first_probe
            .final_network
            .synapses()
            .map(|synapse| synapse.weight())
            .collect();
        let frozen_weights_unchanged = final_weights == trained_weights;

        let mut training_firing = FiringMetrics::default();
        for event in &training_events {
            if let ObservationEvent::SpikeEmitted(spike) = event {
                training_firing.observe(*spike);
            }
        }
        let plastic_weights: Vec<_> = first_probe
            .final_network
            .synapses()
            .filter(|synapse| synapse.is_plastic())
            .map(|synapse| synapse.weight())
            .collect();
        let weight_metrics = WeightMetrics::from_weights(
            plastic_weights.iter().copied(),
            self.config.dsvlm.learning.min_weight,
            self.config.dsvlm.learning.max_weight,
        );
        let all_neurons: Vec<_> = first_probe.final_network.neuron_ids().collect();
        let training_neuron_rates_hz = all_neurons
            .iter()
            .map(|&neuron| {
                (
                    neuron,
                    training_firing.rate_hz(neuron, training_deadline.as_micros()),
                )
            })
            .collect();
        let probe_neuron_rates_hz = all_neurons
            .iter()
            .map(|&neuron| {
                (
                    neuron,
                    first_probe
                        .firing
                        .rate_hz(neuron, self.config.probe_duration_us),
                )
            })
            .collect();
        let expected_indices =
            expected_transition_indices(&pattern_neurons, &first_probe.final_network)?;
        let expected_transition_weight_delta =
            mean_weight_delta(&initial_weights, &trained_weights, &expected_indices);
        let expected_set: BTreeSet<_> = expected_indices.iter().copied().collect();
        let competing_indices: Vec<_> = first_probe
            .final_network
            .synapses()
            .enumerate()
            .filter_map(|(index, synapse)| {
                (synapse.is_plastic() && !expected_set.contains(&index)).then_some(index)
            })
            .collect();
        let competing_transition_weight_delta =
            mean_weight_delta(&initial_weights, &trained_weights, &competing_indices);
        let mut event_log = training_events;
        event_log.extend(first_probe.events.iter().cloned());
        let event_log_digest = digest_event_log(&event_log);
        let stable_after_learning = training_stable && first_probe.stable && second_probe.stable;

        let mut effective_config = self.config.clone();
        effective_config.dsvlm = group_config;
        Ok(M0GroupResult {
            group: self.group,
            seed: self.config.dsvlm.network.seed,
            effective_config,
            training_patterns,
            initial_weights,
            final_weights,
            metrics: M0Metrics {
                transition_hits: first_probe.sequence.transition_hits,
                false_transitions: first_probe.sequence.false_transitions,
                probe_spikes: first_probe.firing.total_spike_count(),
                training_mean_rate_hz: training_firing
                    .mean_rate_hz(all_neurons.iter().copied(), training_deadline.as_micros()),
                training_neuron_rates_hz,
                probe_mean_rate_hz: first_probe
                    .firing
                    .mean_rate_hz(all_neurons.iter().copied(), self.config.probe_duration_us),
                probe_neuron_rates_hz,
                silent_neuron_fraction: first_probe
                    .firing
                    .silent_fraction(all_neurons.iter().copied()),
                persistently_active_neuron_fraction: first_probe.persistent_fraction,
                transition_latency_errors_us: first_probe.sequence.latency_errors_us.clone(),
                weights_at_min_fraction: weight_metrics.at_min_fraction,
                weights_at_max_fraction: weight_metrics.at_max_fraction,
                expected_transition_weight_delta,
                competing_transition_weight_delta,
                stable_after_learning,
                frozen_probe_replay_identical,
                frozen_weights_unchanged,
            },
            predicted_patterns: first_probe.predicted_patterns,
            event_log_digest,
            event_log,
        })
    }

    fn training_patterns(&self) -> Vec<Pattern> {
        let capacity = self
            .config
            .training_repetitions
            .checked_mul(PATTERNS.len())
            .expect("validated training pattern capacity");
        let mut patterns = Vec::with_capacity(capacity);
        for _ in 0..self.config.training_repetitions {
            patterns.extend(PATTERNS);
        }
        if self.group == M0Group::RandomLearning {
            // A separate stream guarantees that randomizing inputs can never alter
            // the identically seeded initial topology.
            let seed = self.config.dsvlm.network.seed ^ INPUT_STREAM;
            DeterministicRng::new(seed).shuffle(&mut patterns);
        }
        patterns
    }
}

/// Executes G1, G2, G3 and G4 from identical initial conditions.
pub fn run_m0_comparison(config: &M0ExperimentConfig) -> Result<Comparison, ExperimentError> {
    config.validate()?;
    let mut groups = Vec::new();
    for group in [
        M0Group::OrderedLearning,
        M0Group::OrderedFixed,
        M0Group::RandomLearning,
        M0Group::OrderedHomeostasis,
    ] {
        groups.push(M0Experiment::new(config.clone(), group)?.run()?);
    }
    Ok(Comparison { groups })
}

/// Executes a paired G1–G4 comparison for every independent seed and derives
/// the predeclared exact sign-test evidence.
pub fn run_m0_study(
    base_config: &M0ExperimentConfig,
    study_config: &M0StudyConfig,
) -> Result<M0StudyReport, ExperimentError> {
    base_config.validate()?;
    if study_config.seeds.is_empty() {
        return Err(ExperimentError::new(
            "study configuration",
            "at least one seed is required",
        ));
    }
    if base_config.recurrent_weight_jitter == 0.0 {
        return Err(ExperimentError::new(
            "study configuration",
            "recurrent_weight_jitter must be positive for independent seeded initial states",
        ));
    }
    let unique: BTreeSet<_> = study_config.seeds.iter().copied().collect();
    if unique.len() != study_config.seeds.len() {
        return Err(ExperimentError::new(
            "study configuration",
            "study seeds must be unique",
        ));
    }

    let mut comparisons = Vec::with_capacity(study_config.seeds.len());
    let mut full_runs_replay_identically = true;
    for &seed in &study_config.seeds {
        let mut config = base_config.clone();
        config.dsvlm.network.seed = seed;
        let comparison = run_m0_comparison(&config)?;
        let replay = run_m0_comparison(&config)?;
        full_runs_replay_identically &= comparison == replay;
        comparisons.push(comparison);
    }

    let mut fixed_differences = Vec::with_capacity(comparisons.len());
    let mut random_differences = Vec::with_capacity(comparisons.len());
    let mut all_runs_stable = true;
    let mut all_replays_identical = full_runs_replay_identically;
    let mut every_g1_has_complete_sequence = true;
    let mut no_runaway_activity = true;
    for comparison in &comparisons {
        let g1 = required_group(comparison, M0Group::OrderedLearning)?;
        let g2 = required_group(comparison, M0Group::OrderedFixed)?;
        let g3 = required_group(comparison, M0Group::RandomLearning)?;
        fixed_differences.push(g1.metrics.sequence_score() - g2.metrics.sequence_score());
        random_differences.push(g1.metrics.sequence_score() - g3.metrics.sequence_score());
        every_g1_has_complete_sequence &= g1.metrics.transition_hits == 3
            && g1.metrics.false_transitions == 0
            && g1.metrics.transition_latency_errors_us.len() == 3;
        no_runaway_activity &= comparison.groups.iter().all(|result| {
            result.metrics.persistently_active_neuron_fraction == 0.0
                && result.metrics.probe_mean_rate_hz.is_finite()
                && result
                    .metrics
                    .probe_neuron_rates_hz
                    .iter()
                    .all(|(_, rate)| rate.is_finite())
                && result.metrics.silent_neuron_fraction < 1.0
        });
        all_runs_stable &= comparison
            .groups
            .iter()
            .all(|result| result.metrics.stable_after_learning);
        all_replays_identical &= comparison.groups.iter().all(|result| {
            result.metrics.frozen_probe_replay_identical && result.metrics.frozen_weights_unchanged
        });
    }

    let wins_over_fixed = fixed_differences.iter().filter(|&&delta| delta > 0).count();
    let wins_over_random = random_differences
        .iter()
        .filter(|&&delta| delta > 0)
        .count();
    let sign_test_p_over_fixed = exact_one_sided_sign_test(&fixed_differences);
    let sign_test_p_over_random = exact_one_sided_sign_test(&random_differences);
    let mean_advantage_over_fixed = mean_score_difference(&fixed_differences);
    let mean_advantage_over_random = mean_score_difference(&random_differences);
    let passed = every_g1_has_complete_sequence
        && no_runaway_activity
        && mean_advantage_over_fixed > 0.0
        && mean_advantage_over_random > 0.0
        && sign_test_p_over_fixed < 0.05
        && sign_test_p_over_random < 0.05
        && all_runs_stable
        && all_replays_identical;

    Ok(M0StudyReport {
        base_config: base_config.clone(),
        study_config: study_config.clone(),
        comparisons,
        success: M0SuccessReport {
            seeds: study_config.seeds.len(),
            wins_over_fixed,
            wins_over_random,
            mean_advantage_over_fixed,
            mean_advantage_over_random,
            sign_test_p_over_fixed,
            sign_test_p_over_random,
            all_runs_stable,
            all_replays_identical,
            passed,
        },
    })
}

fn required_group(
    comparison: &Comparison,
    group: M0Group,
) -> Result<&M0GroupResult, ExperimentError> {
    comparison.group(group).ok_or_else(|| {
        ExperimentError::new(
            "study result",
            format!("comparison omitted {}", group.label()),
        )
    })
}

fn mean_score_difference(differences: &[isize]) -> f32 {
    if differences.is_empty() {
        return 0.0;
    }
    let sum: f64 = differences.iter().map(|&value| value as f64).sum();
    (sum / differences.len() as f64) as f32
}

fn exact_one_sided_sign_test(differences: &[isize]) -> f32 {
    let wins = differences.iter().filter(|&&delta| delta > 0).count();
    let losses = differences.iter().filter(|&&delta| delta < 0).count();
    let trials = wins + losses;
    if trials == 0 {
        return 1.0;
    }

    let denominator = 2_f64.powi(i32::try_from(trials).unwrap_or(i32::MAX));
    let upper_tail: f64 = (wins..=trials)
        .map(|successes| binomial_coefficient(trials, successes))
        .sum();
    (upper_tail / denominator) as f32
}

fn binomial_coefficient(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (1..=k).fold(1.0, |value, index| {
        value * (n - k + index) as f64 / index as f64
    })
}

fn build_network(config: &M0ExperimentConfig) -> Result<PatternNeuronMaps, ExperimentError> {
    // Every group for one seed sees the identical paired initial state. Across
    // seeds, stratified permutations supply genuine independent initial states.
    // One target-blind pool gives all twelve candidates the same marginal.
    // Seed pairing—not knowledge of the trained sequence—controls variation.
    let mut topology_rng = DeterministicRng::new(config.dsvlm.network.seed ^ TOPOLOGY_STREAM);
    let jitter = config.recurrent_weight_jitter;
    let mut recurrent_offsets = vec![
        -jitter, 0.0, jitter, -jitter, 0.0, jitter, -jitter, 0.0, jitter, -jitter, 0.0, jitter,
    ];
    topology_rng.shuffle(&mut recurrent_offsets);
    let mut recurrent_offset_index = 0;
    let mut network = Network::new();
    let mut sensory = BTreeMap::new();
    let mut pattern_cells = BTreeMap::new();

    for (index, pattern) in PATTERNS.into_iter().enumerate() {
        let sensory_id = NeuronId(index as u64);
        let pattern_id = NeuronId(100 + index as u64);
        let position = PATTERN_POSITIONS[index];
        network
            .add_neuron(
                Neuron::new(
                    sensory_id,
                    position,
                    Polarity::Excitatory,
                    Some(NeuronRole::Sensory),
                    config.dsvlm.neuron,
                    SimTime::ZERO,
                )
                .map_err(|error| ExperimentError::new("network", error.to_string()))?,
            )
            .map_err(|error| ExperimentError::new("network", error.to_string()))?;
        network
            .add_neuron(
                Neuron::new(
                    pattern_id,
                    position,
                    Polarity::Excitatory,
                    Some(NeuronRole::Processing),
                    config.dsvlm.neuron,
                    SimTime::ZERO,
                )
                .map_err(|error| ExperimentError::new("network", error.to_string()))?,
            )
            .map_err(|error| ExperimentError::new("network", error.to_string()))?;
        sensory.insert(pattern, sensory_id);
        pattern_cells.insert(pattern, pattern_id);
    }
    let inhibitory_id = NeuronId(200);
    network
        .add_neuron(
            Neuron::new(
                inhibitory_id,
                Position3D::new(0.5, 0.288_675_13, 0.204_124_15),
                Polarity::Inhibitory,
                Some(NeuronRole::Processing),
                config.dsvlm.neuron,
                SimTime::ZERO,
            )
            .map_err(|error| ExperimentError::new("network", error.to_string()))?,
        )
        .map_err(|error| ExperimentError::new("network", error.to_string()))?;

    let mut synapse_id = 0_u64;
    for pattern in PATTERNS {
        network
            .add_synapse(
                Synapse::new(
                    SynapseId(synapse_id),
                    sensory[&pattern],
                    pattern_cells[&pattern],
                    config.sensory_weight,
                    1,
                    false,
                )
                .map_err(|error| ExperimentError::new("network", error.to_string()))?,
            )
            .map_err(|error| ExperimentError::new("network", error.to_string()))?;
        synapse_id += 1;
    }
    // Every possible directed transition begins with exactly the same weight.
    // The expected A→B→C→D edges are deliberately not encoded in topology;
    // only temporal experience may distinguish them through local STDP.
    for pre_pattern in PATTERNS {
        for post_pattern in PATTERNS {
            if pre_pattern == post_pattern {
                continue;
            }
            let pre_position = network
                .neuron(pattern_cells[&pre_pattern])
                .expect("pattern neuron was inserted")
                .position();
            let post_position = network
                .neuron(pattern_cells[&post_pattern])
                .expect("pattern neuron was inserted")
                .position();
            let delay_us = conduction_delay_us(
                pre_position.distance_to(post_position),
                config.dsvlm.network.conduction_velocity,
                1,
            )
            .map_err(|error| ExperimentError::new("network delay", error.to_string()))?;
            let offset = recurrent_offsets[recurrent_offset_index];
            recurrent_offset_index += 1;
            let initial_weight = config.recurrent_weight + offset;
            network
                .add_synapse(
                    Synapse::new(
                        SynapseId(synapse_id),
                        pattern_cells[&pre_pattern],
                        pattern_cells[&post_pattern],
                        initial_weight,
                        delay_us,
                        true,
                    )
                    .map_err(|error| ExperimentError::new("network", error.to_string()))?,
                )
                .map_err(|error| ExperimentError::new("network", error.to_string()))?;
            synapse_id += 1;
        }
    }

    // One fixed interneuron supplies local negative feedback. Its outgoing
    // weights remain non-negative magnitudes; the inhibitory sign is derived
    // exclusively from the cell's polarity by the runtime.
    for pattern in PATTERNS {
        network
            .add_synapse(
                Synapse::new(
                    SynapseId(synapse_id),
                    pattern_cells[&pattern],
                    inhibitory_id,
                    config.inhibitory_drive_weight,
                    config.inhibitory_delay_us,
                    false,
                )
                .map_err(|error| ExperimentError::new("network", error.to_string()))?,
            )
            .map_err(|error| ExperimentError::new("network", error.to_string()))?;
        synapse_id += 1;
        network
            .add_synapse(
                Synapse::new(
                    SynapseId(synapse_id),
                    inhibitory_id,
                    pattern_cells[&pattern],
                    config.inhibitory_feedback_weight,
                    config.inhibitory_delay_us,
                    false,
                )
                .map_err(|error| ExperimentError::new("network", error.to_string()))?,
            )
            .map_err(|error| ExperimentError::new("network", error.to_string()))?;
        synapse_id += 1;
    }

    Ok((network, sensory, pattern_cells))
}

fn build_input_pipeline(
    sensory_neurons: &BTreeMap<Pattern, NeuronId>,
    amplitude: f32,
) -> Result<M0InputPipeline, ExperimentError> {
    let root_id = RootId(1);
    let encoder = PatternEncoder::one_hot(amplitude)
        .map_err(|error| ExperimentError::new("input encoder", format!("{error:?}")))?;
    let mut mapping = Mapping::new();
    let mut channels = Vec::new();

    for (channel, pattern) in PATTERNS.into_iter().enumerate() {
        let channel = channel as u16;
        let fiber_id = FiberId(u64::from(channel));
        let fiber = Fiber::new(
            fiber_id,
            FiberDirection::Sensory,
            sensory_neurons[&pattern],
            1,
            1.0,
        )
        .map_err(|error| ExperimentError::new("input nerve", error))?;
        mapping
            .add_fiber(fiber)
            .map_err(|error| ExperimentError::new("input nerve", format!("{error:?}")))?;
        mapping
            .map_sensory(root_id, channel, fiber_id)
            .map_err(|error| ExperimentError::new("input nerve", format!("{error:?}")))?;
        channels.push(RootChannel {
            channel,
            fiber: fiber_id,
        });
    }

    let root = PatternInputRoot::new(root_id, "m0-pattern-input", channels)
        .map_err(|error| ExperimentError::new("input root", error))?;
    Ok(M0InputPipeline {
        root,
        encoder,
        mapping,
    })
}

struct ProbeOutcome {
    events: Vec<ObservationEvent>,
    predicted_patterns: Vec<(SimTime, Pattern)>,
    sequence: SequenceMetrics,
    firing: FiringMetrics,
    final_network: Network,
    stable: bool,
    persistent_fraction: f32,
}

fn run_frozen_probe(
    network: Network,
    input: &M0InputPipeline,
    pattern_neurons: &BTreeMap<Pattern, NeuronId>,
    config: &M0ExperimentConfig,
    probe_start: SimTime,
) -> Result<ProbeOutcome, ExperimentError> {
    let mut simulation = Simulation::new(
        network,
        NoPlasticity,
        config.dsvlm.runtime,
        config.dsvlm.network.distance_decay_length,
    )
    .map_err(runtime_error)?;
    input.schedule(&mut simulation, probe_start, Pattern::A)?;
    let deadline = checked_advance(probe_start, config.probe_duration_us, "probe horizon")?;
    simulation.run_until(deadline).map_err(runtime_error)?;

    let events = simulation.event_log().to_vec();
    let predicted_patterns = classify_probe(&events, pattern_neurons, probe_start);
    let cue_time = predicted_patterns
        .iter()
        .find_map(|&(time, pattern)| (pattern == Pattern::A).then_some(time))
        .ok_or_else(|| ExperimentError::new("probe", "the A cue produced no pattern-cell spike"))?;
    let sequence = SequenceMetrics::from_timed_probe(
        predicted_patterns.iter().copied(),
        cue_time,
        config.sequence_delay_us,
        config.probe_tolerance_us,
    )
    .map_err(|error| ExperimentError::new("probe metrics", error.to_string()))?;

    let mut firing = FiringMetrics::default();
    for event in &events {
        if let ObservationEvent::SpikeEmitted(spike) = event {
            firing.observe(*spike);
        }
    }
    let quiet_start = SimTime(
        deadline
            .as_micros()
            .checked_sub(config.quiet_window_us)
            .expect("validated quiet window fits inside the probe"),
    );
    let persistent_neurons: BTreeSet<_> = events
        .iter()
        .filter_map(|event| match event {
            ObservationEvent::SpikeEmitted(spike)
                if spike.time >= quiet_start && spike.time <= deadline =>
            {
                Some(spike.neuron_id)
            }
            _ => None,
        })
        .collect();
    let neuron_count = simulation.network().neuron_count();
    let persistent_fraction = if neuron_count == 0 {
        0.0
    } else {
        persistent_neurons.len() as f32 / neuron_count as f32
    };
    let stable = simulation.pending_event_count() == 0 && persistent_neurons.is_empty();
    let final_network = simulation.into_parts().0;

    Ok(ProbeOutcome {
        events,
        predicted_patterns,
        sequence,
        firing,
        final_network,
        stable,
        persistent_fraction,
    })
}

fn expected_transition_indices(
    pattern_neurons: &BTreeMap<Pattern, NeuronId>,
    network: &Network,
) -> Result<Vec<usize>, ExperimentError> {
    let expected = [
        (Pattern::A, Pattern::B),
        (Pattern::B, Pattern::C),
        (Pattern::C, Pattern::D),
    ];
    expected
        .into_iter()
        .map(|(pre, post)| {
            let pre = pattern_neurons[&pre];
            let post = pattern_neurons[&post];
            network
                .synapses()
                .enumerate()
                .find_map(|(index, synapse)| {
                    (synapse.pre() == pre && synapse.post() == post).then_some(index)
                })
                .ok_or_else(|| ExperimentError::new("network", "expected transition is absent"))
        })
        .collect()
}

fn mean_weight_delta(initial: &[f32], final_weights: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let sum: f64 = indices
        .iter()
        .map(|&index| f64::from(final_weights[index]) - f64::from(initial[index]))
        .sum();
    (sum / indices.len() as f64) as f32
}

fn checked_advance(
    time: SimTime,
    duration_us: u64,
    context: &'static str,
) -> Result<SimTime, ExperimentError> {
    time.checked_add_us(duration_us)
        .ok_or_else(|| ExperimentError::new(context, "timestamp overflow"))
}

fn classify_probe(
    events: &[ObservationEvent],
    pattern_neurons: &BTreeMap<Pattern, NeuronId>,
    probe_start: SimTime,
) -> Vec<(SimTime, Pattern)> {
    let by_neuron: BTreeMap<_, _> = pattern_neurons
        .iter()
        .map(|(&pattern, &neuron)| (neuron, pattern))
        .collect();
    events
        .iter()
        .filter_map(|event| match event {
            ObservationEvent::SpikeEmitted(spike) if spike.time >= probe_start => by_neuron
                .get(&spike.neuron_id)
                .copied()
                .map(|pattern| (spike.time, pattern)),
            _ => None,
        })
        .collect()
}

fn digest_event_log(events: &[ObservationEvent]) -> u64 {
    // Stable FNV-1a over Debug's value representation is sufficient for replay
    // identity within the crate version and avoids a serialization dependency.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in format!("{events:?}").bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn runtime_error(error: impl std::fmt::Display) -> ExperimentError {
    ExperimentError::new("runtime", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_seed_replays_identically() {
        let config = M0ExperimentConfig {
            training_repetitions: 3,
            ..M0ExperimentConfig::default()
        };
        let first = M0Experiment::new(config.clone(), M0Group::OrderedLearning)
            .unwrap()
            .run()
            .unwrap();
        let second = M0Experiment::new(config, M0Group::OrderedLearning)
            .unwrap()
            .run()
            .unwrap();

        assert_eq!(first.event_log_digest, second.event_log_digest);
        assert_eq!(first.final_weights, second.final_weights);
        assert_eq!(first.predicted_patterns, second.predicted_patterns);
    }

    #[test]
    fn randomized_group_keeps_same_multiset_but_changes_order() {
        let config = M0ExperimentConfig {
            training_repetitions: 4,
            ..M0ExperimentConfig::default()
        };
        let ordered = M0Experiment::new(config.clone(), M0Group::OrderedLearning)
            .unwrap()
            .training_patterns();
        let random = M0Experiment::new(config, M0Group::RandomLearning)
            .unwrap()
            .training_patterns();
        let mut ordered_counts = BTreeMap::new();
        let mut random_counts = BTreeMap::new();
        for pattern in &ordered {
            *ordered_counts.entry(*pattern).or_insert(0) += 1;
        }
        for pattern in &random {
            *random_counts.entry(*pattern).or_insert(0) += 1;
        }

        assert_eq!(ordered_counts, random_counts);
        assert_ne!(ordered, random);
    }

    #[test]
    fn fixed_group_does_not_change_plastic_weights() {
        let config = M0ExperimentConfig {
            training_repetitions: 3,
            ..M0ExperimentConfig::default()
        };
        let result = M0Experiment::new(config, M0Group::OrderedFixed)
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(result.final_weights[4..16], result.initial_weights[4..16]);
    }

    #[test]
    fn pattern_transition_candidates_are_geometrically_symmetric() {
        let config = M0ExperimentConfig::default();
        let (network, _, pattern_neurons) = build_network(&config).unwrap();
        let pattern_ids: BTreeSet<_> = pattern_neurons.values().copied().collect();
        let candidates: Vec<_> = network
            .synapses()
            .filter(|synapse| synapse.is_plastic())
            .collect();

        assert_eq!(candidates.len(), 12);
        assert!(candidates.iter().all(|synapse| {
            pattern_ids.contains(&synapse.pre())
                && pattern_ids.contains(&synapse.post())
                && synapse.pre() != synapse.post()
                && (synapse.weight() - config.recurrent_weight).abs()
                    <= config.recurrent_weight_jitter + f32::EPSILON
                && synapse.delay_us() == config.sequence_delay_us
        }));
        let distances: Vec<_> = candidates
            .iter()
            .map(|synapse| {
                network
                    .neuron(synapse.pre())
                    .unwrap()
                    .position()
                    .distance_to(network.neuron(synapse.post()).unwrap().position())
            })
            .collect();
        assert!(distances.iter().all(|&distance| distance == 1.0));

        let mut other_seed = config.clone();
        other_seed.dsvlm.network.seed += 1;
        let (other_network, _, _) = build_network(&other_seed).unwrap();
        let weights: Vec<_> = candidates.iter().map(|synapse| synapse.weight()).collect();
        let other_weights: Vec<_> = other_network
            .synapses()
            .filter(|synapse| synapse.is_plastic())
            .map(|synapse| synapse.weight())
            .collect();
        assert_ne!(weights, other_weights);
    }

    #[test]
    fn exact_sign_test_discards_ties() {
        assert_eq!(exact_one_sided_sign_test(&[1, 1, 1, 1, 1, 1]), 0.015625);
        assert_eq!(exact_one_sided_sign_test(&[1, 1, 1, 1, 1, -1]), 0.109375);
        assert_eq!(exact_one_sided_sign_test(&[1, 1, 0, 0]), 0.25);
        assert_eq!(exact_one_sided_sign_test(&[1, -1, -1]), 0.875);
        assert_eq!(exact_one_sided_sign_test(&[0, 0]), 1.0);
    }

    #[test]
    fn study_requires_distinct_seeds() {
        let config = M0ExperimentConfig::default();
        assert!(run_m0_study(&config, &M0StudyConfig { seeds: vec![] }).is_err());
        assert!(run_m0_study(&config, &M0StudyConfig { seeds: vec![7, 7] }).is_err());
        let no_initial_variation = M0ExperimentConfig {
            recurrent_weight_jitter: 0.0,
            ..config
        };
        assert!(run_m0_study(&no_initial_variation, &M0StudyConfig { seeds: vec![7] }).is_err());
    }
}
