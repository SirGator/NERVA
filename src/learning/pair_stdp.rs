//! Local pair-based spike-timing-dependent plasticity for M0.

use std::collections::HashMap;

use crate::{
    config::LearningConfig,
    core::{Neuron, NeuronId, SimTime, Synapse, SynapseId},
};

use super::{
    bounds::{WeightBounds, WeightBoundsError},
    rule::PlasticityRule,
    traces::{DecayingTrace, decay_value},
};

/// Invalid parameters supplied directly to [`PairStdp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PairStdpError {
    /// Potentiation and depression amplitudes must be finite and non-negative.
    InvalidLearningAmplitude,
    /// Both trace time constants must be finite and strictly positive.
    InvalidTimeConstant,
    /// The finite pair window must contain at least one microsecond.
    EmptyStdpWindow,
    /// The weight interval is not a valid non-negative magnitude interval.
    InvalidWeightBounds(WeightBoundsError),
}

/// Pair-STDP state shared by local hooks in one network runtime.
///
/// Synapses own their presynaptic traces. This rule owns one postsynaptic trace
/// per neuron plus the latest local spike timestamps required to enforce the
/// finite STDP window exactly.
#[derive(Debug)]
pub struct PairStdp {
    enabled: bool,
    a_plus: f32,
    a_minus: f32,
    tau_plus_us: f32,
    tau_minus_us: f32,
    stdp_window_us: u64,
    bounds: WeightBounds,
    post_traces: HashMap<NeuronId, DecayingTrace>,
    last_pre_spikes: HashMap<SynapseId, SimTime>,
    last_post_spikes: HashMap<NeuronId, SimTime>,
}

impl PairStdp {
    /// Constructs a rule from an already validated learning configuration.
    ///
    /// Use [`Self::try_from_config`] when the configuration has not been
    /// validated at the application boundary.
    pub fn new(config: &LearningConfig) -> Self {
        Self::try_from_config(config).expect("LearningConfig must be validated before use")
    }

    /// Validates the parameters needed by Pair-STDP and constructs the rule.
    pub fn try_from_config(config: &LearningConfig) -> Result<Self, PairStdpError> {
        if !valid_amplitude(config.a_plus) || !valid_amplitude(config.a_minus) {
            return Err(PairStdpError::InvalidLearningAmplitude);
        }
        if !valid_tau(config.tau_plus_us) || !valid_tau(config.tau_minus_us) {
            return Err(PairStdpError::InvalidTimeConstant);
        }
        if config.stdp_window_us == 0 {
            return Err(PairStdpError::EmptyStdpWindow);
        }

        let bounds = WeightBounds::new(config.min_weight, config.max_weight)
            .map_err(PairStdpError::InvalidWeightBounds)?;

        Ok(Self {
            enabled: config.enabled,
            a_plus: config.a_plus,
            a_minus: config.a_minus,
            tau_plus_us: config.tau_plus_us,
            tau_minus_us: config.tau_minus_us,
            stdp_window_us: config.stdp_window_us,
            bounds,
            post_traces: HashMap::new(),
            last_pre_spikes: HashMap::new(),
            last_post_spikes: HashMap::new(),
        })
    }

    /// Enables or freezes this rule without replacing it in the runtime.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether learning and learning-trace updates are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the stored postsynaptic trace at its most recent update.
    pub fn post_trace(&self, neuron_id: NeuronId) -> Option<&DecayingTrace> {
        self.post_traces.get(&neuron_id)
    }

    /// Clears all rule-owned traces and pair timestamps.
    ///
    /// Presynaptic traces remain owned by each synapse and can be reset by the
    /// network when starting an independent run.
    pub fn clear_rule_state(&mut self) {
        self.post_traces.clear();
        self.last_pre_spikes.clear();
        self.last_post_spikes.clear();
    }

    fn eligible(&self, synapse: &Synapse) -> bool {
        self.enabled
            && synapse.enabled
            && synapse.plastic
            && synapse.weight.is_finite()
            && synapse.weight >= 0.0
            && synapse.pre_trace.is_finite()
    }

    fn can_advance_pre_trace(&self, synapse: &Synapse, time: SimTime) -> bool {
        synapse
            .pre_trace_updated_at
            .is_none_or(|updated_at| time.duration_since(updated_at).is_some())
    }

    fn advance_pre_trace(&self, synapse: &mut Synapse, time: SimTime) {
        if let Some(updated_at) = synapse.pre_trace_updated_at {
            let elapsed_us = time
                .duration_since(updated_at)
                .expect("pre-trace time was checked before mutation");
            synapse.pre_trace = decay_value(synapse.pre_trace, elapsed_us, self.tau_plus_us);
        }
        synapse.pre_trace_updated_at = Some(time);
    }

    fn within_window(&self, later: SimTime, earlier: SimTime) -> Option<u64> {
        let elapsed_us = later.duration_since(earlier)?;
        (elapsed_us <= self.stdp_window_us).then_some(elapsed_us)
    }
}

impl PlasticityRule for PairStdp {
    fn accepts(&self, synapse: &Synapse, presynaptic_polarity: crate::core::Polarity) -> bool {
        presynaptic_polarity == crate::core::Polarity::Excitatory && self.eligible(synapse)
    }

    fn on_pre_arrival(&mut self, synapse: &mut Synapse, post: &Neuron, time: SimTime) {
        if !self.eligible(synapse) || !self.can_advance_pre_trace(synapse, time) {
            return;
        }

        let post_id = post.id();
        if self
            .post_traces
            .get(&post_id)
            .is_some_and(|trace| time.duration_since(trace.updated_at()).is_none())
        {
            return;
        }

        self.advance_pre_trace(synapse, time);

        let post_trace = self
            .post_traces
            .entry(post_id)
            .or_insert_with(|| DecayingTrace::new(time));
        post_trace
            .advance_to(time, self.tau_minus_us)
            .expect("post-trace time and tau were validated");

        if let Some(elapsed_us) = self
            .last_post_spikes
            .get(&post_id)
            .copied()
            .and_then(|post_time| self.within_window(time, post_time))
        {
            let delta = -self.a_minus * decay_factor(elapsed_us, self.tau_minus_us);
            synapse.weight = self.bounds.apply_delta(synapse.weight, delta);
        }

        // The arriving spike contributes only after LTD has paired it with a
        // previous postsynaptic spike.
        synapse.pre_trace += 1.0;
        self.last_pre_spikes.insert(synapse.id, time);
    }

    fn on_post_spike(&mut self, neuron: &Neuron, incoming: &mut [Synapse], time: SimTime) {
        if !self.enabled {
            return;
        }

        let neuron_id = neuron.id();
        if self
            .post_traces
            .get(&neuron_id)
            .is_some_and(|trace| time.duration_since(trace.updated_at()).is_none())
        {
            return;
        }

        for synapse in incoming {
            if synapse.post != neuron_id
                || !self.eligible(synapse)
                || !self.can_advance_pre_trace(synapse, time)
            {
                continue;
            }

            self.advance_pre_trace(synapse, time);

            if let Some(elapsed_us) = self
                .last_pre_spikes
                .get(&synapse.id)
                .copied()
                .and_then(|pre_time| self.within_window(time, pre_time))
            {
                let delta = self.a_plus * decay_factor(elapsed_us, self.tau_plus_us);
                synapse.weight = self.bounds.apply_delta(synapse.weight, delta);
            }
        }

        // Recording is idempotent for a given neuron and timestamp, while every
        // locally supplied incoming synapse receives its own LTP update above.
        if self.last_post_spikes.get(&neuron_id) != Some(&time) {
            self.post_traces
                .entry(neuron_id)
                .or_insert_with(|| DecayingTrace::new(time))
                .add_impulse(time, self.tau_minus_us, 1.0)
                .expect("post-trace time, tau, and impulse were validated");
            self.last_post_spikes.insert(neuron_id, time);
        }
    }
}

fn decay_factor(elapsed_us: u64, tau_us: f32) -> f32 {
    (-(elapsed_us as f32) / tau_us).exp()
}

fn valid_amplitude(value: f32) -> bool {
    value.is_finite() && value >= 0.0
}

fn valid_tau(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

#[cfg(test)]
mod tests {
    use crate::{
        config::NeuronConfig,
        core::{NeuronId, Polarity, SynapseId},
        math::Position3D,
    };

    use super::*;

    fn config() -> LearningConfig {
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

    fn neuron(id: u64) -> Neuron {
        Neuron::new(
            NeuronId(id),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            None,
            NeuronConfig {
                resting_potential: 0.0,
                reset_potential: 0.0,
                threshold: 1.0,
                membrane_tau_us: 1_000.0,
                refractory_period_us: 0,
                activity_trace_tau_us: 1_000.0,
            },
            SimTime::ZERO,
        )
        .expect("valid test neuron")
    }

    fn synapse(id: u64, weight: f32) -> Synapse {
        Synapse::new(SynapseId(id), NeuronId(1), NeuronId(2), weight, 1, true)
            .expect("valid test synapse")
    }

    fn close(left: f32, right: f32) {
        assert!((left - right).abs() <= 1.0e-6, "{left} != {right}");
    }

    #[test]
    fn causal_pair_potentiates_by_exact_time_difference() {
        let mut rule = PairStdp::new(&config());
        let post = neuron(2);
        let mut synapse = synapse(10, 0.5);

        rule.on_pre_arrival(&mut synapse, &post, SimTime(100));
        rule.on_post_spike(&post, std::slice::from_mut(&mut synapse), SimTime(150));

        let expected = 0.5 + 0.2 * (-0.5_f32).exp();
        close(synapse.weight(), expected);
        close(synapse.pre_trace(), (-0.5_f32).exp());
    }

    #[test]
    fn anti_causal_pair_depresses_by_exact_time_difference() {
        let mut rule = PairStdp::new(&config());
        let post = neuron(2);
        let mut synapse = synapse(10, 0.5);

        rule.on_post_spike(&post, &mut [], SimTime(100));
        rule.on_pre_arrival(&mut synapse, &post, SimTime(150));

        let expected = 0.5 - 0.1 * (-0.25_f32).exp();
        close(synapse.weight(), expected);
    }

    #[test]
    fn pairs_outside_the_finite_window_do_not_learn() {
        let mut rule = PairStdp::new(&config());
        let post = neuron(2);
        let mut synapse = synapse(10, 0.5);

        rule.on_pre_arrival(&mut synapse, &post, SimTime(0));
        rule.on_post_spike(&post, std::slice::from_mut(&mut synapse), SimTime(501));

        assert_eq!(synapse.weight(), 0.5);
    }

    #[test]
    fn weight_updates_are_clamped_to_magnitude_bounds() {
        let mut strong_config = config();
        strong_config.a_plus = 10.0;
        strong_config.a_minus = 10.0;
        let post = neuron(2);

        let mut potentiated = synapse(10, 0.5);
        let mut ltp = PairStdp::new(&strong_config);
        ltp.on_pre_arrival(&mut potentiated, &post, SimTime(0));
        ltp.on_post_spike(&post, std::slice::from_mut(&mut potentiated), SimTime(1));
        assert_eq!(potentiated.weight(), strong_config.max_weight);

        let mut depressed = synapse(11, 0.5);
        let mut ltd = PairStdp::new(&strong_config);
        ltd.on_post_spike(&post, &mut [], SimTime(0));
        ltd.on_pre_arrival(&mut depressed, &post, SimTime(1));
        assert_eq!(depressed.weight(), strong_config.min_weight);
    }

    #[test]
    fn fixed_and_disabled_synapses_remain_completely_unchanged() {
        let post = neuron(2);
        for (plastic, enabled) in [(false, true), (true, false)] {
            let mut rule = PairStdp::new(&config());
            let mut synapse = synapse(10, 0.5);
            synapse.set_plastic(plastic);
            synapse.set_enabled(enabled);

            rule.on_pre_arrival(&mut synapse, &post, SimTime(0));
            rule.on_post_spike(&post, std::slice::from_mut(&mut synapse), SimTime(1));

            assert_eq!(synapse.weight(), 0.5);
            assert_eq!(synapse.pre_trace(), 0.0);
            assert_eq!(synapse.pre_trace_updated_at(), None);
        }
    }

    #[test]
    fn sparse_per_synapse_post_hooks_record_only_one_post_event() {
        let mut rule = PairStdp::new(&config());
        let post = neuron(2);
        let mut first = synapse(10, 0.5);
        let mut second = synapse(11, 0.5);
        rule.on_pre_arrival(&mut first, &post, SimTime(0));
        rule.on_pre_arrival(&mut second, &post, SimTime(0));

        rule.on_post_spike(&post, std::slice::from_mut(&mut first), SimTime(10));
        rule.on_post_spike(&post, std::slice::from_mut(&mut second), SimTime(10));

        close(first.weight(), second.weight());
        assert_eq!(
            rule.post_trace(post.id()).map(|trace| trace.value()),
            Some(1.0)
        );
    }

    #[test]
    fn disabled_rule_freezes_weights_and_traces() {
        let mut disabled = config();
        disabled.enabled = false;
        let mut rule = PairStdp::new(&disabled);
        let post = neuron(2);
        let mut synapse = synapse(10, 0.5);

        rule.on_pre_arrival(&mut synapse, &post, SimTime(0));
        rule.on_post_spike(&post, std::slice::from_mut(&mut synapse), SimTime(1));

        assert_eq!(synapse.weight(), 0.5);
        assert_eq!(synapse.pre_trace(), 0.0);
        assert_eq!(rule.post_trace(post.id()), None);
    }
}
