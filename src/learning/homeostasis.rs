//! Per-neuron cellular and structural homeostasis with no population signal.

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

/// Description of one locally applied cellular or structural adjustment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HomeostaticChange {
    /// Neuron whose local state changed.
    pub neuron_id: NeuronId,
    /// Local exponentially weighted firing-rate estimate used by the update.
    pub firing_avg: f32,
    /// Local exponentially weighted absolute-input estimate used by the update.
    pub input_avg: f32,
    /// Intrinsic current before the update.
    pub old_intrinsic_current: f32,
    /// Intrinsic current after clamping.
    pub new_intrinsic_current: f32,
    /// Structural drive before the update.
    pub old_structural_drive: f32,
    /// Structural drive after clamping.
    pub new_structural_drive: f32,
}

/// Local, independently switchable slow regulation.
///
/// A neuron with adequate input but too little output becomes intrinsically
/// more excitable. A neuron with too little input raises structural drive
/// instead, allowing a later growth/pruning slice to search locally without
/// mistaking missing connectivity for missing excitability.
#[derive(Debug)]
pub struct LocalHomeostasis {
    enabled: bool,
    update_interval_us: u64,
    target_rate_hz: f32,
    target_input_rate: f32,
    intrinsic_adjustment_rate: f32,
    structural_adjustment_rate: f32,
    min_intrinsic_current: f32,
    max_intrinsic_current: f32,
    min_structural_drive: f32,
    max_structural_drive: f32,
}

impl LocalHomeostasis {
    /// Constructs homeostasis from validated immutable configuration.
    pub fn new(config: &HomeostasisConfig) -> Self {
        debug_assert!(config.validate().is_ok());
        Self {
            enabled: config.enabled,
            update_interval_us: config.update_interval_us,
            target_rate_hz: config.target_rate_hz,
            target_input_rate: config.target_input_rate,
            intrinsic_adjustment_rate: config.intrinsic_adjustment_rate,
            structural_adjustment_rate: config.structural_adjustment_rate,
            min_intrinsic_current: config.min_intrinsic_current,
            max_intrinsic_current: config.max_intrinsic_current,
            min_structural_drive: config.min_structural_drive,
            max_structural_drive: config.max_structural_drive,
        }
    }

    /// Constructs homeostasis from the nested section of a learning config.
    pub fn from_learning_config(config: &crate::config::LearningConfig) -> Self {
        Self::new(&config.homeostasis)
    }

    /// Enables or freezes local maintenance updates.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether local maintenance is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Per-neuron period of the local maintenance clock.
    pub const fn update_interval_us(&self) -> u64 {
        self.update_interval_us
    }

    /// Acknowledges a local spike notification from the runtime.
    ///
    /// The core increments the neuron's firing trace atomically when it emits
    /// the spike, so this compatibility hook deliberately performs no second
    /// increment.
    pub fn record_spike(&mut self, neuron: &Neuron, time: SimTime) -> Result<(), HomeostasisError> {
        let _ = (neuron, time);
        Ok(())
    }

    /// Applies one local slow-maintenance event and reports an actual change.
    ///
    /// Every change is multiplied by elapsed simulation time since this
    /// neuron's previous maintenance event. Therefore neither the intrinsic
    /// state nor its regulation depends on how many unrelated runtime updates
    /// happen between two timestamps.
    pub fn maintain(
        &mut self,
        neuron: &mut Neuron,
        time: SimTime,
    ) -> Result<Option<HomeostaticChange>, HomeostasisError> {
        if !self.enabled {
            return Ok(None);
        }

        neuron.advance_to(time)?;
        let elapsed_seconds = neuron.homeostasis_elapsed_us(time)? as f32 / 1_000_000.0;
        let state = neuron.homeostatic_state();
        let neuron_id = neuron.id();

        let mut requested_intrinsic_current = state.intrinsic_current;
        let mut requested_structural_drive = state.structural_drive;
        if state.input_avg >= self.target_input_rate {
            // Input is available, so a firing-rate deficit diagnoses local
            // excitability rather than missing connectivity.
            let firing_error = state.firing_avg - self.target_rate_hz;
            requested_intrinsic_current -=
                self.intrinsic_adjustment_rate * firing_error * elapsed_seconds;

            // Sufficient input makes an outstanding local search request less
            // urgent. This is not topology mutation; it only relaxes the cue.
            requested_structural_drive -= self.structural_adjustment_rate
                * (state.input_avg - self.target_input_rate)
                * elapsed_seconds;
        } else {
            let missing_input = self.target_input_rate - state.input_avg;
            requested_structural_drive +=
                self.structural_adjustment_rate * missing_input * elapsed_seconds;

            // A neuron firing despite missing input is overexcitable. Reduce
            // its intrinsic current more strongly than for an ordinary
            // high-firing, adequately driven cell.
            if state.firing_avg > self.target_rate_hz {
                let excess_firing = state.firing_avg - self.target_rate_hz;
                requested_intrinsic_current -= self.intrinsic_adjustment_rate
                    * (excess_firing + missing_input)
                    * elapsed_seconds;
            }
        }

        let new_intrinsic_current = neuron.set_intrinsic_current_clamped(
            requested_intrinsic_current,
            self.min_intrinsic_current,
            self.max_intrinsic_current,
        )?;
        let new_structural_drive = neuron.set_structural_drive_clamped(
            requested_structural_drive,
            self.min_structural_drive,
            self.max_structural_drive,
        )?;
        neuron.record_homeostasis_update(time)?;

        if new_intrinsic_current == state.intrinsic_current
            && new_structural_drive == state.structural_drive
        {
            return Ok(None);
        }

        Ok(Some(HomeostaticChange {
            neuron_id,
            firing_avg: state.firing_avg,
            input_avg: state.input_avg,
            old_intrinsic_current: state.intrinsic_current,
            new_intrinsic_current,
            old_structural_drive: state.structural_drive,
            new_structural_drive,
        }))
    }

    /// Leaves neuron-owned activity and input history intact.
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
                threshold: 10.0,
                membrane_tau_us: 20_000.0,
                refractory_period_us: 0,
                activity_trace_tau_us: 1_000_000.0,
                intrinsic: Default::default(),
            },
            SimTime::ZERO,
        )
        .expect("valid test neuron")
    }

    fn config() -> HomeostasisConfig {
        HomeostasisConfig {
            enabled: true,
            update_interval_us: 1_000_000,
            target_rate_hz: 1.0,
            target_input_rate: 0.5,
            intrinsic_adjustment_rate: 2.0,
            structural_adjustment_rate: 3.0,
            min_intrinsic_current: -10.0,
            max_intrinsic_current: 10.0,
            min_structural_drive: 0.0,
            max_structural_drive: 10.0,
        }
    }

    fn close(left: f32, right: f32) {
        assert!((left - right).abs() <= 1.0e-6, "{left} != {right}");
    }

    #[test]
    fn adequate_input_with_low_output_raises_intrinsic_current() {
        let mut homeostasis = LocalHomeostasis::new(&config());
        let mut cell = neuron(1);
        cell.integrate_input(SimTime::ZERO, 1.0)
            .expect("subthreshold input");

        let change = homeostasis
            .maintain(&mut cell, SimTime(100_000))
            .expect("local maintenance")
            .expect("intrinsic adjustment");

        assert!(change.input_avg >= 0.5);
        assert!(change.firing_avg < 1.0);
        close(cell.intrinsic_current(), 0.2);
        close(cell.structural_drive(), 0.0);
    }

    #[test]
    fn missing_input_with_low_output_raises_structural_drive_not_current() {
        let mut homeostasis = LocalHomeostasis::new(&config());
        let mut cell = neuron(1);

        let change = homeostasis
            .maintain(&mut cell, SimTime(1_000_000))
            .expect("local maintenance")
            .expect("structural adjustment");

        close(change.input_avg, 0.0);
        close(cell.intrinsic_current(), 0.0);
        close(cell.structural_drive(), 1.5);
    }

    #[test]
    fn high_output_with_missing_input_strongly_lowers_intrinsic_current() {
        let mut high_output_config = config();
        high_output_config.target_rate_hz = 0.1;
        high_output_config.target_input_rate = 5.0;
        let mut homeostasis = LocalHomeostasis::new(&high_output_config);
        let mut cell = neuron(1);
        cell.set_intrinsic_current(5.0).unwrap();
        cell.integrate_input(SimTime::ZERO, 10.0)
            .expect("spike input");

        homeostasis
            .maintain(&mut cell, SimTime(1_000_000))
            .expect("local maintenance");

        assert!(cell.intrinsic_current() < 5.0);
        assert!(cell.structural_drive() > 0.0);
    }

    #[test]
    fn disabled_homeostasis_does_not_advance_or_change_neuron() {
        let mut disabled = config();
        disabled.enabled = false;
        let mut homeostasis = LocalHomeostasis::new(&disabled);
        let mut cell = neuron(1);

        assert_eq!(
            homeostasis
                .maintain(&mut cell, SimTime(100))
                .expect("disabled is a no-op"),
            None
        );
        assert_eq!(cell.last_update(), SimTime::ZERO);
        assert_eq!(cell.intrinsic_current(), 0.0);
        assert_eq!(cell.structural_drive(), 0.0);
    }
}
