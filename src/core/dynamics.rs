//! Pointer-free local numerical state and the shared analytical cell kernels.
//!
//! Core and compute use the same arithmetic. Identity, geometry, graph state,
//! scheduler bookings and concrete backends are deliberately absent here.

use super::{
    NeuronError, SimTime,
    intrinsic::{IntrinsicGapModel, exponential_current_response},
};
use crate::{
    config::NeuronConfig,
    math::{decay_to_zero, decay_towards},
};

/// Complete mutable numerical state of one cell, without identity or topology.
///
/// `last_spike_valid` is 0 or 1, so no Rust enum or pointer is needed to encode
/// an optional timestamp. Zero and u64::MAX remain valid spike timestamps.
/// C layout makes the record explicit; a device transfer still needs a matching
/// ABI and must account for padding rather than treating it as serialized bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuronDynamicsState {
    /// Current membrane potential.
    pub membrane_potential: f32,
    /// Last analytical update, in exact microseconds.
    pub last_update: SimTime,
    /// Earliest timestamp at which another spike may be emitted.
    pub refractory_until: SimTime,
    /// Exponentially decaying local spike trace.
    pub activity_trace: f32,
    /// Exponentially decaying absolute-input trace.
    pub input_trace: f32,
    /// Most recent spike time in microseconds, meaningful when the flag is one.
    pub last_spike_time: u64,
    /// One when a spike timestamp exists, zero otherwise.
    pub last_spike_valid: u32,
    /// Saturating count of emitted spikes.
    pub spike_count: u64,
    /// Mutable base threshold before transient adaptation.
    pub threshold: f32,
    /// Spike-triggered positive current.
    pub burst_drive: f32,
    /// Spike-triggered negative current magnitude.
    pub adaptation_drive: f32,
    /// Positive after-current caused by inhibitory impulses.
    pub rebound_drive: f32,
    /// Transient increase above the mutable base threshold.
    pub threshold_adaptation: f32,
    /// Homeostatically controlled constant current.
    pub intrinsic_current: f32,
    /// Local incoming-growth demand, preserved across compute phases.
    pub structural_drive: f32,
    /// Elapsed-time anchor of the last local maintenance update.
    pub last_homeostasis_update: SimTime,
}

/// Immutable, flattened numerical parameters, including intrinsic capabilities.
/// The mutable base threshold is in NeuronDynamicsState; `threshold` here is
/// the original configured threshold and is never overwritten by compute.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuronDynamicsParams {
    /// Resting membrane potential.
    pub resting_potential: f32,
    /// Membrane potential immediately after a spike.
    pub reset_potential: f32,
    /// Original configured threshold, unchanged by local adaptation.
    pub threshold: f32,
    /// Membrane decay time constant in microseconds.
    pub membrane_tau_us: f32,
    /// Refractory duration in exact microseconds.
    pub refractory_period_us: u64,
    /// Time constant of spike and absolute-input traces in microseconds.
    pub activity_trace_tau_us: f32,
    /// Configured constant autonomous current.
    pub intrinsic_drive: f32,
    /// Positive current added by each spike.
    pub burst_gain: f32,
    /// Burst-current decay time constant in microseconds.
    pub burst_tau_us: f32,
    /// Negative current magnitude added by each spike.
    pub adaptation_gain: f32,
    /// Adaptation-current decay time constant in microseconds.
    pub adaptation_tau_us: f32,
    /// Threshold increment added by each spike.
    pub threshold_adaptation_gain: f32,
    /// Threshold-adaptation decay time constant in microseconds.
    pub threshold_adaptation_tau_us: f32,
    /// Positive after-current per unit of inhibitory input.
    pub rebound_gain: f32,
    /// Rebound-current decay time constant in microseconds.
    pub rebound_tau_us: f32,
}

impl From<&NeuronConfig> for NeuronDynamicsParams {
    fn from(params: &NeuronConfig) -> Self {
        Self {
            resting_potential: params.resting_potential,
            reset_potential: params.reset_potential,
            threshold: params.threshold,
            membrane_tau_us: params.membrane_tau_us,
            refractory_period_us: params.refractory_period_us,
            activity_trace_tau_us: params.activity_trace_tau_us,
            intrinsic_drive: params.intrinsic.intrinsic_drive,
            burst_gain: params.intrinsic.burst_gain,
            burst_tau_us: params.intrinsic.burst_tau_us,
            adaptation_gain: params.intrinsic.adaptation_gain,
            adaptation_tau_us: params.intrinsic.adaptation_tau_us,
            threshold_adaptation_gain: params.intrinsic.threshold_adaptation_gain,
            threshold_adaptation_tau_us: params.intrinsic.threshold_adaptation_tau_us,
            rebound_gain: params.intrinsic.rebound_gain,
            rebound_tau_us: params.intrinsic.rebound_tau_us,
        }
    }
}

impl NeuronDynamicsParams {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.resting_potential.to_bits() == other.resting_potential.to_bits()
            && self.reset_potential.to_bits() == other.reset_potential.to_bits()
            && self.threshold.to_bits() == other.threshold.to_bits()
            && self.membrane_tau_us.to_bits() == other.membrane_tau_us.to_bits()
            && self.refractory_period_us == other.refractory_period_us
            && self.activity_trace_tau_us.to_bits() == other.activity_trace_tau_us.to_bits()
            && self.intrinsic_drive.to_bits() == other.intrinsic_drive.to_bits()
            && self.burst_gain.to_bits() == other.burst_gain.to_bits()
            && self.burst_tau_us.to_bits() == other.burst_tau_us.to_bits()
            && self.adaptation_gain.to_bits() == other.adaptation_gain.to_bits()
            && self.adaptation_tau_us.to_bits() == other.adaptation_tau_us.to_bits()
            && self.threshold_adaptation_gain.to_bits() == other.threshold_adaptation_gain.to_bits()
            && self.threshold_adaptation_tau_us.to_bits()
                == other.threshold_adaptation_tau_us.to_bits()
            && self.rebound_gain.to_bits() == other.rebound_gain.to_bits()
            && self.rebound_tau_us.to_bits() == other.rebound_tau_us.to_bits()
    }

    fn intrinsic_is_active(&self) -> bool {
        self.intrinsic_drive != 0.0
            || self.burst_gain != 0.0
            || self.adaptation_gain != 0.0
            || self.threshold_adaptation_gain != 0.0
            || self.rebound_gain != 0.0
    }
}

impl NeuronDynamicsState {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.membrane_potential.to_bits() == other.membrane_potential.to_bits()
            && self.last_update == other.last_update
            && self.refractory_until == other.refractory_until
            && self.activity_trace.to_bits() == other.activity_trace.to_bits()
            && self.input_trace.to_bits() == other.input_trace.to_bits()
            && self.last_spike_time == other.last_spike_time
            && self.last_spike_valid == other.last_spike_valid
            && self.spike_count == other.spike_count
            && self.threshold.to_bits() == other.threshold.to_bits()
            && self.burst_drive.to_bits() == other.burst_drive.to_bits()
            && self.adaptation_drive.to_bits() == other.adaptation_drive.to_bits()
            && self.rebound_drive.to_bits() == other.rebound_drive.to_bits()
            && self.threshold_adaptation.to_bits() == other.threshold_adaptation.to_bits()
            && self.intrinsic_current.to_bits() == other.intrinsic_current.to_bits()
            && self.structural_drive.to_bits() == other.structural_drive.to_bits()
            && self.last_homeostasis_update == other.last_homeostasis_update
    }

    pub(crate) fn validate(&self, params: &NeuronDynamicsParams) -> Result<(), NeuronError> {
        for (field, value) in [
            ("membrane_potential", self.membrane_potential),
            ("activity_trace", self.activity_trace),
            ("input_trace", self.input_trace),
            ("threshold", self.threshold),
            ("burst_drive", self.burst_drive),
            ("adaptation_drive", self.adaptation_drive),
            ("rebound_drive", self.rebound_drive),
            ("threshold_adaptation", self.threshold_adaptation),
            ("intrinsic_current", self.intrinsic_current),
            ("structural_drive", self.structural_drive),
        ] {
            if !value.is_finite() {
                return Err(NeuronError::InvalidNumericalState { field });
            }
        }
        for (field, value) in [
            ("activity_trace", self.activity_trace),
            ("input_trace", self.input_trace),
            ("burst_drive", self.burst_drive),
            ("adaptation_drive", self.adaptation_drive),
            ("rebound_drive", self.rebound_drive),
            ("threshold_adaptation", self.threshold_adaptation),
        ] {
            if value < 0.0 {
                return Err(NeuronError::InvalidNumericalState { field });
            }
        }
        if self.threshold <= params.resting_potential || self.threshold <= params.reset_potential {
            return Err(NeuronError::InvalidThreshold(self.threshold));
        }
        checked_effective_sum(
            "effective_threshold",
            self.threshold,
            self.threshold_adaptation,
        )?;
        checked_effective_sum(
            "effective_intrinsic_drive",
            params.intrinsic_drive,
            self.intrinsic_current,
        )?;
        if self.last_spike_valid > 1
            || (self.last_spike_valid == 0 && (self.last_spike_time != 0 || self.spike_count != 0))
            || (self.last_spike_valid == 1
                && (self.last_spike_time > self.last_update.as_micros() || self.spike_count == 0))
        {
            return Err(NeuronError::InvalidNumericalState {
                field: "last_spike",
            });
        }
        Ok(())
    }

    fn threshold(&self) -> f32 {
        self.threshold + self.threshold_adaptation
    }
    fn is_refractory_at(&self, time: SimTime) -> bool {
        time < self.refractory_until
    }
    pub(crate) fn advance_to(
        &mut self,
        params: &NeuronDynamicsParams,
        time: SimTime,
    ) -> Result<(), NeuronError> {
        let elapsed_us =
            time.duration_since(self.last_update)
                .ok_or(NeuronError::TimeWentBackwards {
                    current: self.last_update,
                    requested: time,
                })?;

        if elapsed_us == 0 {
            return Ok(());
        }

        if !params.intrinsic_is_active() {
            // Preserve the original f32 LIF path bit-for-bit when all new
            // capabilities use their backward-compatible defaults.
            let intrinsic_equilibrium = params.resting_potential
                + self.intrinsic_current * (params.membrane_tau_us / 1_000_000.0);
            if !intrinsic_equilibrium.is_finite() {
                return Err(NeuronError::NonFiniteIntrinsicEquilibrium(
                    intrinsic_equilibrium,
                ));
            }
            self.membrane_potential = decay_towards(
                self.membrane_potential,
                intrinsic_equilibrium,
                elapsed_us,
                params.membrane_tau_us,
            );
        } else {
            let intrinsic_equilibrium = self.constant_intrinsic_equilibrium(params);
            if !intrinsic_equilibrium.is_finite()
                || intrinsic_equilibrium.abs() > f64::from(f32::MAX)
            {
                return Err(NeuronError::NonFiniteIntrinsicEquilibrium(
                    intrinsic_equilibrium as f32,
                ));
            }
            let next_potential = self.projected_membrane_potential(params, elapsed_us);
            if !next_potential.is_finite() || next_potential.abs() > f64::from(f32::MAX) {
                return Err(NeuronError::NonFiniteMembranePotential(
                    next_potential as f32,
                ));
            }
            self.membrane_potential = next_potential as f32;
        }
        if !self.membrane_potential.is_finite() {
            return Err(NeuronError::NonFiniteMembranePotential(
                self.membrane_potential,
            ));
        }
        self.burst_drive = decay_to_zero(self.burst_drive, elapsed_us, params.burst_tau_us);
        self.adaptation_drive =
            decay_to_zero(self.adaptation_drive, elapsed_us, params.adaptation_tau_us);
        self.rebound_drive = decay_to_zero(self.rebound_drive, elapsed_us, params.rebound_tau_us);
        self.threshold_adaptation = decay_to_zero(
            self.threshold_adaptation,
            elapsed_us,
            params.threshold_adaptation_tau_us,
        );
        self.activity_trace = decay_to_zero(
            self.activity_trace,
            elapsed_us,
            params.activity_trace_tau_us,
        );
        self.input_trace =
            decay_to_zero(self.input_trace, elapsed_us, params.activity_trace_tau_us);
        self.last_update = time;

        Ok(())
    }

    pub(crate) fn integrate_input_with_components(
        &mut self,
        params: &NeuronDynamicsParams,
        time: SimTime,
        summed_input: f32,
        input_magnitude: f32,
        inhibitory_input_magnitude: f32,
    ) -> Result<bool, NeuronError> {
        validate_input_summary(summed_input, input_magnitude, inhibitory_input_magnitude)?;

        self.advance_to(params, time)?;
        self.observe_input(input_magnitude);
        if self.is_refractory_at(time) {
            return Ok(false);
        }

        let next_potential = self.membrane_potential + summed_input;
        if !next_potential.is_finite() {
            return Err(NeuronError::NonFiniteMembranePotential(next_potential));
        }
        if next_potential < self.threshold() {
            let rebound_drive = checked_intrinsic_sum(
                "rebound_drive",
                self.rebound_drive,
                params.rebound_gain * inhibitory_input_magnitude,
            )?;
            self.membrane_potential = next_potential;
            self.rebound_drive = rebound_drive;
            return Ok(false);
        }

        let refractory_until = time.checked_add_us(params.refractory_period_us).ok_or(
            NeuronError::RefractoryTimeOverflow {
                spike_time: time,
                refractory_period_us: params.refractory_period_us,
            },
        )?;
        let burst_drive =
            checked_intrinsic_sum("burst_drive", self.burst_drive, params.burst_gain)?;
        let adaptation_drive = checked_intrinsic_sum(
            "adaptation_drive",
            self.adaptation_drive,
            params.adaptation_gain,
        )?;
        let threshold_adaptation = checked_intrinsic_sum(
            "threshold_adaptation",
            self.threshold_adaptation,
            params.threshold_adaptation_gain,
        )?;
        checked_effective_sum("effective_threshold", self.threshold, threshold_adaptation)?;
        let rebound_drive = checked_intrinsic_sum(
            "rebound_drive",
            self.rebound_drive,
            params.rebound_gain * inhibitory_input_magnitude,
        )?;

        self.membrane_potential = params.reset_potential;
        self.refractory_until = refractory_until;
        self.burst_drive = burst_drive;
        self.adaptation_drive = adaptation_drive;
        self.threshold_adaptation = threshold_adaptation;
        self.rebound_drive = rebound_drive;
        self.activity_trace += 1.0;
        self.last_spike_time = time.as_micros();
        self.last_spike_valid = 1;
        self.spike_count = self.spike_count.saturating_add(1);

        Ok(true)
    }

    pub(crate) fn predicted_intrinsic_spike_at(
        &self,
        params: &NeuronDynamicsParams,
        now: SimTime,
    ) -> Option<SimTime> {
        if now != self.last_update
            || (!self.is_refractory_at(now) && self.membrane_potential >= self.threshold())
        {
            return None;
        }

        if !params.intrinsic_is_active() {
            return self.predict_legacy_constant_drive_crossing(params, now);
        }

        let max_elapsed = (u64::MAX - now.as_micros()).checked_sub(params.refractory_period_us)?;
        if max_elapsed == 0 {
            return None;
        }
        let refractory_elapsed = self
            .refractory_until
            .duration_since(now)
            .unwrap_or(0)
            .min(max_elapsed);
        let first_eligible = refractory_elapsed.max(1);

        if first_eligible > max_elapsed {
            return None;
        }
        if self.projected_will_fire(params, first_eligible) {
            return now.checked_add_us(first_eligible);
        }

        if !self.has_transient_intrinsic_state() {
            return self.predict_constant_drive_crossing(params, now, first_eligible);
        }

        let model = self.intrinsic_gap_model(params);
        let mut interval_start = first_eligible.saturating_add(1);
        let mut interval_width = 1_u64;
        loop {
            if interval_start > max_elapsed {
                return None;
            }
            let interval_end = interval_start
                .saturating_add(interval_width.saturating_sub(1))
                .min(max_elapsed);
            if let Some(crossing) =
                self.first_crossing_in_interval(params, &model, interval_start, interval_end)
            {
                return now.checked_add_us(crossing);
            }
            if interval_end == max_elapsed {
                return None;
            }
            let next_start = interval_end + 1;
            if model.tail_is_strictly_negative(next_start) {
                return None;
            }
            interval_start = next_start;
            interval_width = interval_width.saturating_mul(2);
        }
    }

    fn observe_input(&mut self, input_magnitude: f32) {
        // Input history is a diagnostic/homeostatic estimate. Saturating it
        // preserves valid membrane dynamics even for an extreme but finite
        // simultaneous batch such as `MAX + -MAX`.
        self.input_trace = ((f64::from(self.input_trace) + f64::from(input_magnitude))
            .min(f64::from(f32::MAX))) as f32;
    }

    fn projected_membrane_potential(&self, params: &NeuronDynamicsParams, elapsed_us: u64) -> f64 {
        let elapsed_seconds = elapsed_us as f64 / 1_000_000.0;
        let membrane_tau_seconds = f64::from(params.membrane_tau_us) / 1_000_000.0;
        let resting = f64::from(params.resting_potential);
        let constant_current =
            f64::from(params.intrinsic_drive) + f64::from(self.intrinsic_current);
        let equilibrium = resting + constant_current * membrane_tau_seconds;
        let membrane_decay = (-elapsed_seconds / membrane_tau_seconds).exp();
        let mut potential =
            equilibrium + (f64::from(self.membrane_potential) - equilibrium) * membrane_decay;

        potential += exponential_current_response(
            self.burst_drive,
            params.burst_tau_us,
            params.membrane_tau_us,
            elapsed_us,
        );
        potential -= exponential_current_response(
            self.adaptation_drive,
            params.adaptation_tau_us,
            params.membrane_tau_us,
            elapsed_us,
        );
        potential += exponential_current_response(
            self.rebound_drive,
            params.rebound_tau_us,
            params.membrane_tau_us,
            elapsed_us,
        );
        potential
    }

    pub(crate) fn projected_will_fire(
        &self,
        params: &NeuronDynamicsParams,
        elapsed_us: u64,
    ) -> bool {
        let potential = self.projected_membrane_potential(params, elapsed_us) as f32;
        let threshold = self.threshold
            + decay_to_zero(
                self.threshold_adaptation,
                elapsed_us,
                params.threshold_adaptation_tau_us,
            );
        potential.is_finite() && threshold.is_finite() && potential >= threshold
    }

    fn has_transient_intrinsic_state(&self) -> bool {
        self.burst_drive != 0.0
            || self.adaptation_drive != 0.0
            || self.rebound_drive != 0.0
            || self.threshold_adaptation != 0.0
    }

    fn constant_intrinsic_equilibrium(&self, params: &NeuronDynamicsParams) -> f64 {
        f64::from(params.resting_potential)
            + (f64::from(params.intrinsic_drive) + f64::from(self.intrinsic_current))
                * f64::from(params.membrane_tau_us)
                / 1_000_000.0
    }

    fn predict_constant_drive_crossing(
        &self,
        params: &NeuronDynamicsParams,
        now: SimTime,
        first_eligible: u64,
    ) -> Option<SimTime> {
        let equilibrium = self.constant_intrinsic_equilibrium(params);
        if !equilibrium.is_finite() || equilibrium <= f64::from(self.threshold) {
            return None;
        }

        let numerator = f64::from(self.threshold) - equilibrium;
        let denominator = f64::from(self.membrane_potential) - equilibrium;
        let ratio = numerator / denominator;
        if !(0.0..1.0).contains(&ratio) {
            return None;
        }
        let elapsed_us = (-f64::from(params.membrane_tau_us) * ratio.ln()).ceil();
        if !elapsed_us.is_finite() || elapsed_us > (u64::MAX - now.as_micros()) as f64 {
            return None;
        }
        now.checked_add_us((elapsed_us as u64).max(first_eligible))
    }

    fn predict_legacy_constant_drive_crossing(
        &self,
        params: &NeuronDynamicsParams,
        now: SimTime,
    ) -> Option<SimTime> {
        let tau_seconds = params.membrane_tau_us / 1_000_000.0;
        let equilibrium = params.resting_potential + self.intrinsic_current * tau_seconds;
        if !equilibrium.is_finite() || equilibrium <= self.threshold {
            return None;
        }

        let ratio = (self.threshold - equilibrium) / (self.membrane_potential - equilibrium);
        if !(0.0..1.0).contains(&ratio) {
            return None;
        }
        let elapsed_us = (-params.membrane_tau_us * ratio.ln()).ceil();
        if !elapsed_us.is_finite() || elapsed_us > u64::MAX as f32 {
            return None;
        }
        let crossing = now.checked_add_us((elapsed_us as u64).max(1))?;
        Some(crossing.max(self.refractory_until))
    }

    fn intrinsic_gap_model(&self, params: &NeuronDynamicsParams) -> IntrinsicGapModel {
        let equilibrium = self.constant_intrinsic_equilibrium(params);
        let mut model = IntrinsicGapModel::new(equilibrium - f64::from(self.threshold));
        model.add_term(
            params.membrane_tau_us,
            f64::from(self.membrane_potential) - equilibrium,
            0.0,
        );
        model.add_current_response(
            self.burst_drive,
            params.burst_tau_us,
            params.membrane_tau_us,
            1.0,
        );
        model.add_current_response(
            self.adaptation_drive,
            params.adaptation_tau_us,
            params.membrane_tau_us,
            -1.0,
        );
        model.add_current_response(
            self.rebound_drive,
            params.rebound_tau_us,
            params.membrane_tau_us,
            1.0,
        );
        model.add_term(
            params.threshold_adaptation_tau_us,
            -f64::from(self.threshold_adaptation),
            0.0,
        );
        model.remove_zero_terms();
        model
    }

    fn first_crossing_in_interval(
        &self,
        params: &NeuronDynamicsParams,
        model: &IntrinsicGapModel,
        start: u64,
        end: u64,
    ) -> Option<u64> {
        if self.projected_will_fire(params, start) {
            return Some(start);
        }
        if start == end {
            return None;
        }

        let gap_range = model.range(start, end);
        if gap_range.upper < -gap_range.rounding_margin() {
            return None;
        }

        let derivative_range = model.derivative_range(start, end);
        if derivative_range.upper < -derivative_range.rounding_margin() {
            return None;
        }
        if derivative_range.lower > derivative_range.rounding_margin()
            && !self.projected_will_fire(params, end)
        {
            return None;
        }

        let midpoint = start + (end - start) / 2;
        self.first_crossing_in_interval(params, model, start, midpoint)
            .or_else(|| self.first_crossing_in_interval(params, model, midpoint + 1, end))
    }
}

fn validate_input_summary(
    summed_input: f32,
    input_magnitude: f32,
    inhibitory_input_magnitude: f32,
) -> Result<(), NeuronError> {
    if !summed_input.is_finite() {
        return Err(NeuronError::NonFiniteInput(summed_input));
    }
    if !input_magnitude.is_finite() || input_magnitude < 0.0 {
        return Err(NeuronError::InvalidInputMagnitude(input_magnitude));
    }
    if !inhibitory_input_magnitude.is_finite()
        || inhibitory_input_magnitude < 0.0
        || inhibitory_input_magnitude > input_magnitude
    {
        return Err(NeuronError::InvalidInhibitoryInputMagnitude {
            inhibitory: inhibitory_input_magnitude,
            total: input_magnitude,
        });
    }
    Ok(())
}

fn checked_intrinsic_sum(
    field: &'static str,
    current: f32,
    increment: f32,
) -> Result<f32, NeuronError> {
    let next = f64::from(current) + f64::from(increment);
    if !next.is_finite() || next > f64::from(f32::MAX) || next < f64::from(f32::MIN) {
        Err(NeuronError::NonFiniteIntrinsicState {
            field,
            value: next as f32,
        })
    } else {
        Ok(next as f32)
    }
}

pub(super) fn checked_effective_sum(
    field: &'static str,
    left: f32,
    right: f32,
) -> Result<f32, NeuronError> {
    let value = f64::from(left) + f64::from(right);
    if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
        Err(NeuronError::NonFiniteIntrinsicState {
            field,
            value: value as f32,
        })
    } else {
        Ok(value as f32)
    }
}
