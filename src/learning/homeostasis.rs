//! Per-neuron firing-rate homeostasis with no population-wide signal.

use crate::{
    config::HomeostasisConfig,
    core::{Neuron, NeuronError, NeuronId, SimTime},
};

/// Why a local maintenance update could not be applied.
#[derive(Clone, Debug, PartialEq)]
pub enum HomeostasisError {
    /// The neuron's local state could not be advanced or adjusted.
    Neuron(NeuronError),
}

impl From<NeuronError> for HomeostasisError {
    fn from(error: NeuronError) -> Self {
        Self::Neuron(error)
    }
}

/// Description of one locally applied threshold adjustment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThresholdChange {
    /// Neuron whose threshold was changed.
    pub neuron_id: NeuronId,
    /// Threshold before local maintenance.
    pub old_threshold: f32,
    /// Threshold after local maintenance and clamping.
    pub new_threshold: f32,
    /// Local exponentially weighted firing-rate estimate used by the update.
    pub estimated_rate_hz: f32,
}

/// Local, independently switchable firing-rate homeostasis.
///
/// The activity estimate lives in the neuron itself and no global mean or error
/// exists. Core spike emission increments that trace; maintenance only reads it
/// after analytical time advancement.
#[derive(Debug)]
pub struct LocalHomeostasis {
    enabled: bool,
    target_rate_hz: f32,
    adjustment_rate: f32,
    min_threshold: f32,
    max_threshold: f32,
}

impl LocalHomeostasis {
    /// Constructs homeostasis from validated immutable configuration.
    pub fn new(config: &HomeostasisConfig) -> Self {
        debug_assert!(config.validate().is_ok());
        Self {
            enabled: config.enabled,
            target_rate_hz: config.target_rate_hz,
            adjustment_rate: config.adjustment_rate,
            min_threshold: config.min_threshold,
            max_threshold: config.max_threshold,
        }
    }

    /// Constructs homeostasis from the nested section of a learning config.
    pub fn from_learning_config(config: &crate::config::LearningConfig) -> Self {
        Self::new(&config.homeostasis)
    }

    /// Enables or freezes local threshold updates.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether local maintenance is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Acknowledges a local spike notification from the runtime.
    ///
    /// The core already increments the neuron's activity trace atomically when
    /// it emits the spike, so this compatibility hook deliberately performs no
    /// second increment.
    pub fn record_spike(&mut self, neuron: &Neuron, time: SimTime) -> Result<(), HomeostasisError> {
        let _ = (neuron, time);
        Ok(())
    }

    /// Returns this neuron's local firing-rate estimate at `time`.
    pub fn estimated_rate_hz(
        &self,
        neuron: &mut Neuron,
        time: SimTime,
    ) -> Result<f32, HomeostasisError> {
        neuron.advance_to(time)?;
        Ok(neuron.estimated_firing_rate_hz())
    }

    /// Applies one local maintenance event and reports an actual change.
    ///
    /// The update is proportional to this neuron's own rate error. Positive
    /// error raises the threshold; negative error lowers it. The core clamps the
    /// effective threshold to the configured interval.
    pub fn maintain(
        &mut self,
        neuron: &mut Neuron,
        time: SimTime,
    ) -> Result<Option<ThresholdChange>, HomeostasisError> {
        if !self.enabled {
            return Ok(None);
        }

        let neuron_id = neuron.id();
        let estimated_rate_hz = self.estimated_rate_hz(neuron, time)?;
        let old_threshold = neuron.threshold();
        let rate_error = estimated_rate_hz - self.target_rate_hz;
        let requested = old_threshold + self.adjustment_rate * rate_error;
        let new_threshold = neuron
            .set_threshold_clamped(requested, self.min_threshold, self.max_threshold)
            .map_err(HomeostasisError::Neuron)?;

        if new_threshold == old_threshold {
            return Ok(None);
        }

        Ok(Some(ThresholdChange {
            neuron_id,
            old_threshold,
            new_threshold,
            estimated_rate_hz,
        }))
    }

    /// Leaves neuron-owned activity history intact.
    ///
    /// This method exists for orchestration symmetry with [`crate::learning::PairStdp`]; a new
    /// experiment should construct fresh core neurons when it needs fresh local
    /// firing-rate traces.
    pub fn clear(&mut self) {}
}

#[cfg(test)]
mod tests {
    use crate::{
        config::NeuronConfig,
        core::{Polarity, SimTime},
        math::Position3D,
    };

    use super::*;

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
                membrane_tau_us: 20_000.0,
                refractory_period_us: 0,
                activity_trace_tau_us: 1_000_000.0,
            },
            SimTime::ZERO,
        )
        .expect("valid test neuron")
    }

    fn config() -> HomeostasisConfig {
        HomeostasisConfig {
            enabled: true,
            target_rate_hz: 0.5,
            adjustment_rate: 0.5,
            min_threshold: 0.1,
            max_threshold: 2.0,
        }
    }

    fn close(left: f32, right: f32) {
        assert!((left - right).abs() <= 1.0e-6, "{left} != {right}");
    }

    #[test]
    fn active_neuron_raises_only_its_own_threshold() {
        let mut homeostasis = LocalHomeostasis::new(&config());
        let mut active = neuron(1);
        let silent = neuron(2);
        active
            .integrate_input(SimTime::ZERO, 1.0)
            .expect("valid input")
            .expect("spike");

        let change = homeostasis
            .maintain(&mut active, SimTime::ZERO)
            .expect("local maintenance")
            .expect("threshold change");

        close(change.estimated_rate_hz, 1.0);
        close(active.threshold(), 1.25);
        assert_eq!(silent.threshold(), 1.0);
    }

    #[test]
    fn underactive_neuron_lowers_threshold() {
        let mut homeostasis = LocalHomeostasis::new(&config());
        let mut silent = neuron(1);

        homeostasis
            .maintain(&mut silent, SimTime::ZERO)
            .expect("local maintenance")
            .expect("threshold change");

        close(silent.threshold(), 0.75);
    }

    #[test]
    fn threshold_adjustment_respects_local_bounds() {
        let mut high_gain = config();
        high_gain.target_rate_hz = 0.0;
        high_gain.adjustment_rate = 10.0;
        high_gain.max_threshold = 1.5;
        let mut homeostasis = LocalHomeostasis::new(&high_gain);
        let mut active = neuron(1);
        active
            .integrate_input(SimTime::ZERO, 1.0)
            .expect("valid input");

        homeostasis
            .maintain(&mut active, SimTime::ZERO)
            .expect("local maintenance");

        assert_eq!(active.threshold(), 1.5);
    }

    #[test]
    fn firing_rate_estimate_decays_analytically_with_neuron_state() {
        let homeostasis = LocalHomeostasis::new(&config());
        let mut active = neuron(1);
        active
            .integrate_input(SimTime::ZERO, 1.0)
            .expect("valid input");

        let rate = homeostasis
            .estimated_rate_hz(&mut active, SimTime(1_000_000))
            .expect("forward time");

        close(rate, (-1.0_f32).exp());
    }

    #[test]
    fn disabled_homeostasis_does_not_advance_or_change_neuron() {
        let mut disabled = config();
        disabled.enabled = false;
        let mut homeostasis = LocalHomeostasis::new(&disabled);
        let mut neuron = neuron(1);

        assert_eq!(
            homeostasis
                .maintain(&mut neuron, SimTime(100))
                .expect("disabled is a no-op"),
            None
        );
        assert_eq!(neuron.last_update(), SimTime::ZERO);
        assert_eq!(neuron.threshold(), 1.0);
    }
}
