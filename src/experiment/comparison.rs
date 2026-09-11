//! M0 comparison groups and neutral result records.

use crate::{
    core::{NeuronId, SimTime},
    environment::Pattern,
    primitives::Weight,
    runtime::ObservationEvent,
};

use super::sequence_m0::M0ExperimentConfig;

/// The four controlled M0 comparison groups from the architecture document.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum M0Group {
    /// Ordered input and Pair-STDP enabled (G1).
    OrderedLearning,
    /// Ordered input with frozen weights (G2).
    OrderedFixed,
    /// Randomized input and Pair-STDP enabled (G3).
    RandomLearning,
    /// Ordered input with Pair-STDP and local homeostasis (G4).
    OrderedHomeostasis,
}

impl M0Group {
    /// Stable short label used in reports.
    pub const fn label(self) -> &'static str {
        match self {
            Self::OrderedLearning => "G1",
            Self::OrderedFixed => "G2",
            Self::RandomLearning => "G3",
            Self::OrderedHomeostasis => "G4",
        }
    }
}

/// Read-only measurements from one training/probe run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct M0Metrics {
    /// Expected B→C→D activations in their configured probe windows.
    pub transition_hits: usize,
    /// Pattern-neuron spikes outside the expected next-state windows.
    pub false_transitions: usize,
    /// Number of emitted spikes in the frozen-weight probe phase.
    pub probe_spikes: usize,
    /// Mean population firing rate during training.
    pub training_mean_rate_hz: f32,
    /// Training firing rate for each neuron in stable ID order.
    pub training_neuron_rates_hz: Vec<(NeuronId, f32)>,
    /// Mean population firing rate during the frozen probe.
    pub probe_mean_rate_hz: f32,
    /// Frozen-probe firing rate for each neuron in stable ID order.
    pub probe_neuron_rates_hz: Vec<(NeuronId, f32)>,
    /// Fraction of neurons that emitted no spike during the frozen probe.
    pub silent_neuron_fraction: f32,
    /// Fraction that still fired in the final configured quiet window.
    pub persistently_active_neuron_fraction: f32,
    /// Signed timing error for each accepted B, C and D hit.
    pub transition_latency_errors_us: Vec<i64>,
    /// Fraction of plastic weights at their lower bound.
    pub weights_at_min_fraction: f32,
    /// Fraction of plastic weights at their upper bound.
    pub weights_at_max_fraction: f32,
    /// Mean delta of the three expected transition weights.
    pub expected_transition_weight_delta: f32,
    /// Mean delta of all competing pattern-transition weights.
    pub competing_transition_weight_delta: f32,
    /// Whether activity ceased before the configured probe deadline.
    pub stable_after_learning: bool,
    /// Whether two independent frozen probes produced identical observations.
    pub frozen_probe_replay_identical: bool,
    /// Whether the frozen probe left every weight unchanged.
    pub frozen_weights_unchanged: bool,
}

impl M0Metrics {
    /// Hits minus false transitions, useful for controlled group comparison.
    pub fn sequence_score(&self) -> isize {
        self.transition_hits as isize - self.false_transitions as isize
    }
}

/// Result of one controlled M0 group.
#[derive(Clone, Debug, PartialEq)]
pub struct M0GroupResult {
    /// Executed group.
    pub group: M0Group,
    /// Identical network seed shared by all groups.
    pub seed: u64,
    /// Complete effective group-specific configuration, including controls.
    pub effective_config: M0ExperimentConfig,
    /// Actual training order after optional G3 shuffling.
    pub training_patterns: Vec<Pattern>,
    /// Pattern-classified spikes observed after the A-only cue.
    pub predicted_patterns: Vec<(SimTime, Pattern)>,
    /// Final internal weights in stable synapse-ID order.
    pub final_weights: Vec<Weight>,
    /// Initial weights in the same stable synapse-ID order.
    pub initial_weights: Vec<Weight>,
    /// Probe and stability measurements.
    pub metrics: M0Metrics,
    /// Stable digest of the immutable observation log.
    pub event_log_digest: u64,
    /// Complete immutable observation stream for exact replay inspection.
    pub event_log: Vec<ObservationEvent>,
}

/// Results of G1 through G4 under controlled initial conditions.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    /// Group results in G1, G2, G3, G4 order.
    pub groups: Vec<M0GroupResult>,
}

impl Comparison {
    /// Returns one group result.
    pub fn group(&self, group: M0Group) -> Option<&M0GroupResult> {
        self.groups.iter().find(|result| result.group == group)
    }

    /// Whether G1 beats both frozen-learning and randomized controls.
    pub fn ordered_learning_outperforms_controls(&self) -> bool {
        let Some(g1) = self.group(M0Group::OrderedLearning) else {
            return false;
        };
        let Some(g2) = self.group(M0Group::OrderedFixed) else {
            return false;
        };
        let Some(g3) = self.group(M0Group::RandomLearning) else {
            return false;
        };
        g1.metrics.transition_hits > 0
            && g1.metrics.sequence_score() > g2.metrics.sequence_score()
            && g1.metrics.sequence_score() > g3.metrics.sequence_score()
            && self.groups.iter().all(|result| {
                result.metrics.stable_after_learning
                    && result.metrics.frozen_probe_replay_identical
                    && result.metrics.frozen_weights_unchanged
            })
    }
}

/// Seeds used for an aggregate paired M0 study.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct M0StudyConfig {
    /// Independent seeds; each seed runs all four controlled groups.
    pub seeds: Vec<u64>,
}

impl Default for M0StudyConfig {
    fn default() -> Self {
        Self {
            seeds: vec![7, 19, 41, 73, 101, 149],
        }
    }
}

/// Transparent aggregate evidence for the M0 success criterion.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct M0SuccessReport {
    /// Number of paired seeds evaluated.
    pub seeds: usize,
    /// Seeds where G1's score exceeded G2.
    pub wins_over_fixed: usize,
    /// Seeds where G1's score exceeded G3.
    pub wins_over_random: usize,
    /// Mean paired score advantage G1−G2.
    pub mean_advantage_over_fixed: f32,
    /// Mean paired score advantage G1−G3.
    pub mean_advantage_over_random: f32,
    /// Exact one-sided paired sign-test probability for G1 > G2.
    pub sign_test_p_over_fixed: f32,
    /// Exact one-sided paired sign-test probability for G1 > G3.
    pub sign_test_p_over_random: f32,
    /// Whether every run was stable after learning.
    pub all_runs_stable: bool,
    /// Whether every full-run and frozen-probe replay plus weight-freeze check passed.
    pub all_replays_identical: bool,
    /// True only when complete sequence effects, non-collapse, stability,
    /// replay, and `p < 0.05` all pass.
    pub passed: bool,
}

/// All paired comparisons and their aggregate success report.
#[derive(Clone, Debug, PartialEq)]
pub struct M0StudyReport {
    /// Complete base experiment configuration before seed pairing.
    pub base_config: M0ExperimentConfig,
    /// Independent seed plan used for the paired comparisons.
    pub study_config: M0StudyConfig,
    /// One G1–G4 comparison per configured seed.
    pub comparisons: Vec<Comparison>,
    /// Aggregate evidence derived from those comparisons.
    pub success: M0SuccessReport,
}
