//! Analytical leaky integrate-and-fire neuron state.

use std::{error::Error, fmt};

use crate::{
    config::{ConfigError, NeuronConfig},
    math::{Position3D, PositionError, decay_to_zero, decay_towards},
};

use super::{
    IntrinsicState, NeuronId, SimTime, Spike,
    intrinsic::{IntrinsicGapModel, exponential_current_response},
};

/// Sign contributed by every outgoing synapse of a neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Polarity {
    /// Raises the postsynaptic membrane potential.
    Excitatory,
    /// Lowers the postsynaptic membrane potential.
    Inhibitory,
}

impl Polarity {
    /// Numeric sign applied to a non-negative synaptic weight magnitude.
    pub const fn sign(self) -> f32 {
        match self {
            Self::Excitatory => 1.0,
            Self::Inhibitory => -1.0,
        }
    }

    /// Applies this polarity to a non-negative magnitude.
    pub fn apply(self, magnitude: f32) -> f32 {
        self.sign() * magnitude
    }
}

/// Optional metadata used by adapters; it never changes spike dynamics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NeuronRole {
    /// Receives spikes from a sensory nerve mapping.
    Sensory,
    /// Participates in internal recurrent processing.
    Processing,
    /// Is observed by a motor nerve mapping.
    Motor,
}

/// Locally observed state used by cellular and future structural plasticity.
///
/// Both averages are exponentially decayed, neuron-owned quantities. No
/// population mean or global error signal is involved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HomeostaticState {
    /// Estimated firing rate in hertz.
    pub firing_avg: f32,
    /// Estimated absolute input magnitude per second.
    pub input_avg: f32,
    /// Constant intrinsic current in potential units per second.
    pub intrinsic_current: f32,
    /// Local request signal for future connection growth or pruning.
    pub structural_drive: f32,
}

/// One LIF cell with immutable identity/geometry/parameters and local state.
#[derive(Clone, Debug, PartialEq)]
pub struct Neuron {
    id: NeuronId,
    position: Position3D,
    polarity: Polarity,
    role: Option<NeuronRole>,
    params: NeuronConfig,
    membrane_potential: f32,
    last_update: SimTime,
    refractory_until: SimTime,
    activity_trace: f32,
    input_trace: f32,
    last_spike: Option<SimTime>,
    spike_count: u64,
    /// Mutable base threshold controlled independently of spike adaptation.
    threshold: f32,
    burst_drive: f32,
    adaptation_drive: f32,
    rebound_drive: f32,
    threshold_adaptation: f32,
    /// Constant local current, integrated analytically over elapsed time.
    intrinsic_current: f32,
    /// Local future-facing signal; it never directly mutates graph topology.
    structural_drive: f32,
    last_homeostasis_update: SimTime,
    next_homeostasis_update: Option<SimTime>,
    next_intrinsic_spike: Option<SimTime>,
    /// Scheduler sequence for the prediction above, allowing eager removal
    /// when a local input or current adjustment supersedes it.
    next_intrinsic_spike_sequence: Option<u64>,
}

impl Neuron {
    /// Constructs a resting neuron after validating cell parameters and geometry.
    pub fn new(
        id: NeuronId,
        position: Position3D,
        polarity: Polarity,
        role: Option<NeuronRole>,
        params: NeuronConfig,
        start_time: SimTime,
    ) -> Result<Self, NeuronError> {
        params.validate().map_err(NeuronError::InvalidConfig)?;
        position.validate().map_err(NeuronError::InvalidPosition)?;

        Ok(Self {
            id,
            position,
            polarity,
            role,
            params,
            membrane_potential: params.resting_potential,
            last_update: start_time,
            refractory_until: start_time,
            activity_trace: 0.0,
            input_trace: 0.0,
            last_spike: None,
            spike_count: 0,
            threshold: params.threshold,
            burst_drive: 0.0,
            adaptation_drive: 0.0,
            rebound_drive: 0.0,
            threshold_adaptation: 0.0,
            intrinsic_current: 0.0,
            structural_drive: 0.0,
            last_homeostasis_update: start_time,
            next_homeostasis_update: None,
            next_intrinsic_spike: None,
            next_intrinsic_spike_sequence: None,
        })
    }

    /// Stable identity of this neuron.
    pub const fn id(&self) -> NeuronId {
        self.id
    }

    /// Spatial location used only for geometry and propagation.
    pub const fn position(&self) -> Position3D {
        self.position
    }

    /// Sign shared by every outgoing synapse.
    pub const fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// Optional adapter metadata. This value is not read by LIF dynamics.
    pub const fn role(&self) -> Option<NeuronRole> {
        self.role
    }

    /// Immutable electrical parameters.
    pub const fn params(&self) -> &NeuronConfig {
        &self.params
    }

    /// Current membrane potential.
    pub const fn membrane_potential(&self) -> f32 {
        self.membrane_potential
    }

    /// Last timestamp to which continuous local state was advanced.
    pub const fn last_update(&self) -> SimTime {
        self.last_update
    }

    /// Earliest timestamp at which this neuron may emit another spike.
    pub const fn refractory_until(&self) -> SimTime {
        self.refractory_until
    }

    /// Local exponentially decaying spike trace.
    pub const fn activity_trace(&self) -> f32 {
        self.activity_trace
    }

    /// Local exponentially decaying trace of received input magnitudes.
    pub const fn input_trace(&self) -> f32 {
        self.input_trace
    }

    /// Most recent spike timestamp, if any.
    pub const fn last_spike(&self) -> Option<SimTime> {
        self.last_spike
    }

    /// Number of spikes emitted since construction.
    pub const fn spike_count(&self) -> u64 {
        self.spike_count
    }

    /// Current local firing threshold, including spike-triggered adaptation.
    pub const fn threshold(&self) -> f32 {
        self.threshold + self.threshold_adaptation
    }

    /// Mutable base threshold before transient spike-triggered adaptation.
    pub const fn base_threshold(&self) -> f32 {
        self.threshold
    }

    /// Constant intrinsic current in potential units per second.
    pub const fn intrinsic_current(&self) -> f32 {
        self.intrinsic_current
    }

    /// Snapshot of all continuously evolving intrinsic quantities.
    pub fn intrinsic_state(&self) -> IntrinsicState {
        IntrinsicState {
            intrinsic_drive: self.params.intrinsic.intrinsic_drive + self.intrinsic_current,
            burst_drive: self.burst_drive,
            adaptation_drive: self.adaptation_drive,
            rebound_drive: self.rebound_drive,
            threshold_adaptation: self.threshold_adaptation,
            effective_threshold: self.threshold(),
        }
    }

    /// Local structural-plasticity request signal.
    pub const fn structural_drive(&self) -> f32 {
        self.structural_drive
    }

    /// Snapshot of the quantities used by local homeostasis.
    pub fn homeostatic_state(&self) -> HomeostaticState {
        HomeostaticState {
            firing_avg: self.estimated_firing_rate_hz(),
            input_avg: self.estimated_input_rate(),
            intrinsic_current: self.intrinsic_current,
            structural_drive: self.structural_drive,
        }
    }

    /// Timestamp of the next local maintenance event, if its clock is active.
    pub const fn next_homeostasis_update(&self) -> Option<SimTime> {
        self.next_homeostasis_update
    }

    /// Timestamp of the currently predicted autonomous threshold crossing.
    pub const fn next_intrinsic_spike(&self) -> Option<SimTime> {
        self.next_intrinsic_spike
    }

    /// Scheduler sequence for the currently predicted autonomous crossing.
    ///
    /// This is runtime bookkeeping rather than neural state. It is paired
    /// with [`Self::next_intrinsic_spike`] whenever a prediction is queued.
    pub const fn next_intrinsic_spike_sequence(&self) -> Option<u64> {
        self.next_intrinsic_spike_sequence
    }

    /// Whether a spike is prohibited at `time` by the local refractory state.
    pub fn is_refractory_at(&self, time: SimTime) -> bool {
        time < self.refractory_until
    }

    /// Analytically advances continuous local state to `time`.
    ///
    /// The intrinsic current is a current per simulation second, not an amount
    /// added per update. It shifts the LIF equilibrium by `I * tau`, so calling
    /// this once or through many unrelated event timestamps gives the same
    /// membrane potential at the same final time.
    pub fn advance_to(&mut self, time: SimTime) -> Result<(), NeuronError> {
        let elapsed_us =
            time.duration_since(self.last_update)
                .ok_or(NeuronError::TimeWentBackwards {
                    current: self.last_update,
                    requested: time,
                })?;

        if elapsed_us == 0 {
            return Ok(());
        }

        if !self.params.intrinsic.is_active() {
            // Preserve the original f32 LIF path bit-for-bit when all new
            // capabilities use their backward-compatible defaults.
            let intrinsic_equilibrium = self.params.resting_potential
                + self.intrinsic_current * (self.params.membrane_tau_us / 1_000_000.0);
            if !intrinsic_equilibrium.is_finite() {
                return Err(NeuronError::NonFiniteIntrinsicEquilibrium(
                    intrinsic_equilibrium,
                ));
            }
            self.membrane_potential = decay_towards(
                self.membrane_potential,
                intrinsic_equilibrium,
                elapsed_us,
                self.params.membrane_tau_us,
            );
        } else {
            let intrinsic_equilibrium = self.constant_intrinsic_equilibrium();
            if !intrinsic_equilibrium.is_finite()
                || intrinsic_equilibrium.abs() > f64::from(f32::MAX)
            {
                return Err(NeuronError::NonFiniteIntrinsicEquilibrium(
                    intrinsic_equilibrium as f32,
                ));
            }
            let next_potential = self.projected_membrane_potential(elapsed_us);
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
        self.burst_drive = decay_to_zero(
            self.burst_drive,
            elapsed_us,
            self.params.intrinsic.burst_tau_us,
        );
        self.adaptation_drive = decay_to_zero(
            self.adaptation_drive,
            elapsed_us,
            self.params.intrinsic.adaptation_tau_us,
        );
        self.rebound_drive = decay_to_zero(
            self.rebound_drive,
            elapsed_us,
            self.params.intrinsic.rebound_tau_us,
        );
        self.threshold_adaptation = decay_to_zero(
            self.threshold_adaptation,
            elapsed_us,
            self.params.intrinsic.threshold_adaptation_tau_us,
        );
        self.activity_trace = decay_to_zero(
            self.activity_trace,
            elapsed_us,
            self.params.activity_trace_tau_us,
        );
        self.input_trace = decay_to_zero(
            self.input_trace,
            elapsed_us,
            self.params.activity_trace_tau_us,
        );
        self.last_update = time;

        Ok(())
    }

    /// Applies one already-aggregated exact-timestamp input and checks firing
    /// exactly once.
    ///
    /// Input during the refractory interval is ignored after the analytical
    /// state advance. On firing, reset and refractory state are updated and an
    /// immutable spike record is returned.
    pub fn integrate_input(
        &mut self,
        time: SimTime,
        summed_input: f32,
    ) -> Result<Option<Spike>, NeuronError> {
        if !summed_input.is_finite() {
            return Err(NeuronError::NonFiniteInput(summed_input));
        }

        self.integrate_input_with_components(
            time,
            summed_input,
            summed_input.abs(),
            (-summed_input).max(0.0),
        )
    }

    /// Applies an already-aggregated input together with its total absolute
    /// arrival magnitude.
    ///
    /// Runtime timestamp batching can cancel an excitatory and inhibitory
    /// contribution in the membrane sum. The separate magnitude still records
    /// that the neuron received substantial local input, allowing homeostasis
    /// to distinguish inhibition from disconnection.
    pub fn integrate_input_with_magnitude(
        &mut self,
        time: SimTime,
        summed_input: f32,
        input_magnitude: f32,
    ) -> Result<Option<Spike>, NeuronError> {
        self.integrate_input_with_components(
            time,
            summed_input,
            input_magnitude,
            (-summed_input).max(0.0),
        )
    }

    /// Applies one exact-timestamp input summary while preserving its true
    /// inhibitory magnitude for rebound dynamics.
    ///
    /// An inhibitory impulse ends at the batch boundary, so its configured
    /// rebound after-current starts immediately after that batch. Passing the
    /// separate magnitude prevents simultaneous excitation from hiding it.
    pub fn integrate_input_with_components(
        &mut self,
        time: SimTime,
        summed_input: f32,
        input_magnitude: f32,
        inhibitory_input_magnitude: f32,
    ) -> Result<Option<Spike>, NeuronError> {
        validate_input_summary(summed_input, input_magnitude, inhibitory_input_magnitude)?;

        self.advance_to(time)?;
        self.observe_input(input_magnitude);
        if self.is_refractory_at(time) {
            return Ok(None);
        }

        let next_potential = self.membrane_potential + summed_input;
        if !next_potential.is_finite() {
            return Err(NeuronError::NonFiniteMembranePotential(next_potential));
        }
        if next_potential < self.threshold() {
            let rebound_drive = checked_intrinsic_sum(
                "rebound_drive",
                self.rebound_drive,
                self.params.intrinsic.rebound_gain * inhibitory_input_magnitude,
            )?;
            self.membrane_potential = next_potential;
            self.rebound_drive = rebound_drive;
            return Ok(None);
        }

        let refractory_until = time
            .checked_add_us(self.params.refractory_period_us)
            .ok_or(NeuronError::RefractoryTimeOverflow {
                spike_time: time,
                refractory_period_us: self.params.refractory_period_us,
            })?;
        let burst_drive = checked_intrinsic_sum(
            "burst_drive",
            self.burst_drive,
            self.params.intrinsic.burst_gain,
        )?;
        let adaptation_drive = checked_intrinsic_sum(
            "adaptation_drive",
            self.adaptation_drive,
            self.params.intrinsic.adaptation_gain,
        )?;
        let threshold_adaptation = checked_intrinsic_sum(
            "threshold_adaptation",
            self.threshold_adaptation,
            self.params.intrinsic.threshold_adaptation_gain,
        )?;
        checked_effective_sum("effective_threshold", self.threshold, threshold_adaptation)?;
        let rebound_drive = checked_intrinsic_sum(
            "rebound_drive",
            self.rebound_drive,
            self.params.intrinsic.rebound_gain * inhibitory_input_magnitude,
        )?;

        self.membrane_potential = self.params.reset_potential;
        self.refractory_until = refractory_until;
        self.burst_drive = burst_drive;
        self.adaptation_drive = adaptation_drive;
        self.threshold_adaptation = threshold_adaptation;
        self.rebound_drive = rebound_drive;
        self.activity_trace += 1.0;
        self.last_spike = Some(time);
        self.spike_count = self.spike_count.saturating_add(1);

        Ok(Some(Spike::new(self.id, time)))
    }

    /// Alias matching batch-oriented runtime terminology.
    pub fn integrate_input_batch(
        &mut self,
        time: SimTime,
        summed_input: f32,
    ) -> Result<Option<Spike>, NeuronError> {
        self.integrate_input(time, summed_input)
    }

    /// Sets the local threshold while preserving the LIF potential ordering.
    pub fn set_threshold(&mut self, threshold: f32) -> Result<(), NeuronError> {
        if !threshold.is_finite() {
            return Err(NeuronError::InvalidThreshold(threshold));
        }
        if threshold <= self.params.resting_potential || threshold <= self.params.reset_potential {
            return Err(NeuronError::InvalidThreshold(threshold));
        }
        checked_effective_sum("effective_threshold", threshold, self.threshold_adaptation)?;
        self.threshold = threshold;
        Ok(())
    }

    /// Clamps a requested threshold to inclusive local bounds, sets it, and
    /// returns the value actually used.
    pub fn set_threshold_clamped(
        &mut self,
        requested: f32,
        min_threshold: f32,
        max_threshold: f32,
    ) -> Result<f32, NeuronError> {
        if !requested.is_finite()
            || !min_threshold.is_finite()
            || !max_threshold.is_finite()
            || min_threshold > max_threshold
        {
            return Err(NeuronError::InvalidThresholdBounds {
                requested,
                min: min_threshold,
                max: max_threshold,
            });
        }

        let threshold = requested.clamp(min_threshold, max_threshold);
        self.set_threshold(threshold)?;
        Ok(threshold)
    }

    /// Converts the local exponential trace to a per-second rate estimate.
    pub fn estimated_firing_rate_hz(&self) -> f32 {
        (f64::from(self.activity_trace) * 1_000_000.0
            / f64::from(self.params.activity_trace_tau_us))
        .min(f64::from(f32::MAX)) as f32
    }

    /// Converts the local absolute-input trace to a per-second estimate.
    pub fn estimated_input_rate(&self) -> f32 {
        (f64::from(self.input_trace) * 1_000_000.0 / f64::from(self.params.activity_trace_tau_us))
            .min(f64::from(f32::MAX)) as f32
    }

    /// Sets the local intrinsic current after validating finiteness.
    pub fn set_intrinsic_current(&mut self, intrinsic_current: f32) -> Result<(), NeuronError> {
        if !intrinsic_current.is_finite() {
            return Err(NeuronError::InvalidHomeostaticValue {
                field: "intrinsic_current",
                value: intrinsic_current,
            });
        }
        validate_effective_intrinsic_drive(
            self.params.intrinsic.intrinsic_drive,
            intrinsic_current,
        )?;
        self.intrinsic_current = intrinsic_current;
        Ok(())
    }

    /// Clamps and applies a requested local intrinsic current.
    pub fn set_intrinsic_current_clamped(
        &mut self,
        requested: f32,
        min: f32,
        max: f32,
    ) -> Result<f32, NeuronError> {
        let intrinsic_current =
            validate_clamped_homeostatic_value("intrinsic_current", requested, min, max)?;
        validate_effective_intrinsic_drive(
            self.params.intrinsic.intrinsic_drive,
            intrinsic_current,
        )?;
        self.intrinsic_current = intrinsic_current;
        Ok(intrinsic_current)
    }

    /// Clamps and applies a requested local structural-drive value.
    pub fn set_structural_drive_clamped(
        &mut self,
        requested: f32,
        min: f32,
        max: f32,
    ) -> Result<f32, NeuronError> {
        let structural_drive =
            validate_clamped_homeostatic_value("structural_drive", requested, min, max)?;
        self.structural_drive = structural_drive;
        Ok(structural_drive)
    }

    /// Elapsed time since this neuron's previous local maintenance update.
    pub fn homeostasis_elapsed_us(&self, time: SimTime) -> Result<u64, NeuronError> {
        time.duration_since(self.last_homeostasis_update)
            .ok_or(NeuronError::TimeWentBackwards {
                current: self.last_homeostasis_update,
                requested: time,
            })
    }

    /// Marks a completed local maintenance update.
    pub fn record_homeostasis_update(&mut self, time: SimTime) -> Result<(), NeuronError> {
        self.homeostasis_elapsed_us(time)?;
        self.last_homeostasis_update = time;
        Ok(())
    }

    /// Resets the elapsed-time anchor used by the next local maintenance
    /// update without changing any cellular or structural state.
    ///
    /// The runtime calls this when local homeostasis is re-enabled so a
    /// disabled interval is not retrospectively integrated as regulation
    /// time on the first newly enabled maintenance event.
    pub fn reset_homeostasis_time_anchor(&mut self, time: SimTime) -> Result<(), NeuronError> {
        self.record_homeostasis_update(time)
    }

    /// Records the next local maintenance deadline owned by this neuron.
    pub fn set_next_homeostasis_update(&mut self, time: Option<SimTime>) {
        self.next_homeostasis_update = time;
    }

    /// Records the currently scheduled autonomous threshold-crossing event
    /// and its scheduler sequence.
    pub fn set_next_intrinsic_spike(&mut self, time: Option<SimTime>, sequence: Option<u64>) {
        debug_assert_eq!(time.is_some(), sequence.is_some());
        self.next_intrinsic_spike = time;
        self.next_intrinsic_spike_sequence = sequence;
    }

    /// Predicts the next autonomous threshold crossing from local continuous
    /// dynamics, starting from the already current state.
    ///
    /// Constant-drive-only cells retain the closed-form LIF solution. For
    /// transient state, a chronological interval search discards a range only
    /// when conservative exponential bounds prove that it cannot fire. Leaf
    /// decisions use the same integer-microsecond and `f32` comparison as
    /// actual integration. This creates local future events without a global
    /// tick or a fixed prediction horizon.
    pub fn predicted_intrinsic_spike_at(&self, now: SimTime) -> Option<SimTime> {
        if now != self.last_update
            || (!self.is_refractory_at(now) && self.membrane_potential >= self.threshold())
        {
            return None;
        }

        if !self.params.intrinsic.is_active() {
            return self.predict_legacy_constant_drive_crossing(now);
        }

        let max_elapsed =
            (u64::MAX - now.as_micros()).checked_sub(self.params.refractory_period_us)?;
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
        if self.projected_will_fire(first_eligible) {
            return now.checked_add_us(first_eligible);
        }

        if !self.has_transient_intrinsic_state() {
            return self.predict_constant_drive_crossing(now, first_eligible);
        }

        let model = self.intrinsic_gap_model();
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
                self.first_crossing_in_interval(&model, interval_start, interval_end)
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

    fn projected_membrane_potential(&self, elapsed_us: u64) -> f64 {
        let elapsed_seconds = elapsed_us as f64 / 1_000_000.0;
        let membrane_tau_seconds = f64::from(self.params.membrane_tau_us) / 1_000_000.0;
        let resting = f64::from(self.params.resting_potential);
        let constant_current =
            f64::from(self.params.intrinsic.intrinsic_drive) + f64::from(self.intrinsic_current);
        let equilibrium = resting + constant_current * membrane_tau_seconds;
        let membrane_decay = (-elapsed_seconds / membrane_tau_seconds).exp();
        let mut potential =
            equilibrium + (f64::from(self.membrane_potential) - equilibrium) * membrane_decay;

        potential += exponential_current_response(
            self.burst_drive,
            self.params.intrinsic.burst_tau_us,
            self.params.membrane_tau_us,
            elapsed_us,
        );
        potential -= exponential_current_response(
            self.adaptation_drive,
            self.params.intrinsic.adaptation_tau_us,
            self.params.membrane_tau_us,
            elapsed_us,
        );
        potential += exponential_current_response(
            self.rebound_drive,
            self.params.intrinsic.rebound_tau_us,
            self.params.membrane_tau_us,
            elapsed_us,
        );
        potential
    }

    fn projected_will_fire(&self, elapsed_us: u64) -> bool {
        let potential = self.projected_membrane_potential(elapsed_us) as f32;
        let threshold = self.threshold
            + decay_to_zero(
                self.threshold_adaptation,
                elapsed_us,
                self.params.intrinsic.threshold_adaptation_tau_us,
            );
        potential.is_finite() && threshold.is_finite() && potential >= threshold
    }

    fn has_transient_intrinsic_state(&self) -> bool {
        self.burst_drive != 0.0
            || self.adaptation_drive != 0.0
            || self.rebound_drive != 0.0
            || self.threshold_adaptation != 0.0
    }

    fn constant_intrinsic_equilibrium(&self) -> f64 {
        f64::from(self.params.resting_potential)
            + (f64::from(self.params.intrinsic.intrinsic_drive) + f64::from(self.intrinsic_current))
                * f64::from(self.params.membrane_tau_us)
                / 1_000_000.0
    }

    fn predict_constant_drive_crossing(
        &self,
        now: SimTime,
        first_eligible: u64,
    ) -> Option<SimTime> {
        let equilibrium = self.constant_intrinsic_equilibrium();
        if !equilibrium.is_finite() || equilibrium <= f64::from(self.threshold) {
            return None;
        }

        let numerator = f64::from(self.threshold) - equilibrium;
        let denominator = f64::from(self.membrane_potential) - equilibrium;
        let ratio = numerator / denominator;
        if !(0.0..1.0).contains(&ratio) {
            return None;
        }
        let elapsed_us = (-f64::from(self.params.membrane_tau_us) * ratio.ln()).ceil();
        if !elapsed_us.is_finite() || elapsed_us > (u64::MAX - now.as_micros()) as f64 {
            return None;
        }
        now.checked_add_us((elapsed_us as u64).max(first_eligible))
    }

    fn predict_legacy_constant_drive_crossing(&self, now: SimTime) -> Option<SimTime> {
        let tau_seconds = self.params.membrane_tau_us / 1_000_000.0;
        let equilibrium = self.params.resting_potential + self.intrinsic_current * tau_seconds;
        if !equilibrium.is_finite() || equilibrium <= self.threshold {
            return None;
        }

        let ratio = (self.threshold - equilibrium) / (self.membrane_potential - equilibrium);
        if !(0.0..1.0).contains(&ratio) {
            return None;
        }
        let elapsed_us = (-self.params.membrane_tau_us * ratio.ln()).ceil();
        if !elapsed_us.is_finite() || elapsed_us > u64::MAX as f32 {
            return None;
        }
        let crossing = now.checked_add_us((elapsed_us as u64).max(1))?;
        Some(crossing.max(self.refractory_until))
    }

    fn intrinsic_gap_model(&self) -> IntrinsicGapModel {
        let equilibrium = self.constant_intrinsic_equilibrium();
        let mut model = IntrinsicGapModel::new(equilibrium - f64::from(self.threshold));
        model.add_term(
            self.params.membrane_tau_us,
            f64::from(self.membrane_potential) - equilibrium,
            0.0,
        );
        model.add_current_response(
            self.burst_drive,
            self.params.intrinsic.burst_tau_us,
            self.params.membrane_tau_us,
            1.0,
        );
        model.add_current_response(
            self.adaptation_drive,
            self.params.intrinsic.adaptation_tau_us,
            self.params.membrane_tau_us,
            -1.0,
        );
        model.add_current_response(
            self.rebound_drive,
            self.params.intrinsic.rebound_tau_us,
            self.params.membrane_tau_us,
            1.0,
        );
        model.add_term(
            self.params.intrinsic.threshold_adaptation_tau_us,
            -f64::from(self.threshold_adaptation),
            0.0,
        );
        model.remove_zero_terms();
        model
    }

    fn first_crossing_in_interval(
        &self,
        model: &IntrinsicGapModel,
        start: u64,
        end: u64,
    ) -> Option<u64> {
        if self.projected_will_fire(start) {
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
            && !self.projected_will_fire(end)
        {
            return None;
        }

        let midpoint = start + (end - start) / 2;
        self.first_crossing_in_interval(model, start, midpoint)
            .or_else(|| self.first_crossing_in_interval(model, midpoint + 1, end))
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

fn checked_effective_sum(field: &'static str, left: f32, right: f32) -> Result<f32, NeuronError> {
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

fn validate_effective_intrinsic_drive(
    configured_drive: f32,
    homeostatic_current: f32,
) -> Result<(), NeuronError> {
    checked_effective_sum(
        "effective_intrinsic_drive",
        configured_drive,
        homeostatic_current,
    )
    .map(|_| ())
}

fn validate_clamped_homeostatic_value(
    field: &'static str,
    requested: f32,
    min: f32,
    max: f32,
) -> Result<f32, NeuronError> {
    if !requested.is_finite() || !min.is_finite() || !max.is_finite() || min > max {
        return Err(NeuronError::InvalidHomeostaticBounds {
            field,
            requested,
            min,
            max,
        });
    }
    Ok(requested.clamp(min, max))
}

/// Invalid construction or temporal evolution of a neuron.
#[derive(Clone, Debug, PartialEq)]
pub enum NeuronError {
    /// Invalid immutable LIF parameters.
    InvalidConfig(ConfigError),
    /// Invalid geometric coordinate.
    InvalidPosition(PositionError),
    /// State cannot be evolved into the past.
    TimeWentBackwards {
        /// Current last-update timestamp.
        current: SimTime,
        /// Earlier requested timestamp.
        requested: SimTime,
    },
    /// An input contribution is NaN or infinite.
    NonFiniteInput(f32),
    /// An absolute input magnitude is negative, NaN, or infinite.
    InvalidInputMagnitude(f32),
    /// The inhibitory component is invalid or exceeds total input magnitude.
    InvalidInhibitoryInputMagnitude {
        /// Rejected inhibitory component.
        inhibitory: f32,
        /// Total absolute input magnitude supplied with the same batch.
        total: f32,
    },
    /// Finite operands overflowed the membrane representation.
    NonFiniteMembranePotential(f32),
    /// A finite intrinsic impulse overflowed one local state variable.
    NonFiniteIntrinsicState {
        /// Name of the affected state variable.
        field: &'static str,
        /// Non-finite or unrepresentable result.
        value: f32,
    },
    /// A current and membrane time constant produced a non-finite equilibrium.
    NonFiniteIntrinsicEquilibrium(f32),
    /// Adding the refractory duration exceeded simulation time.
    RefractoryTimeOverflow {
        /// Timestamp at which the spike would have occurred.
        spike_time: SimTime,
        /// Configured local refractory duration.
        refractory_period_us: u64,
    },
    /// A threshold is non-finite or not above resting/reset potential.
    InvalidThreshold(f32),
    /// Threshold clamping inputs are non-finite or reversed.
    InvalidThresholdBounds {
        /// Requested value.
        requested: f32,
        /// Inclusive lower bound.
        min: f32,
        /// Inclusive upper bound.
        max: f32,
    },
    /// A local current or structural value was non-finite.
    InvalidHomeostaticValue {
        /// Name of the local quantity.
        field: &'static str,
        /// Rejected value.
        value: f32,
    },
    /// Requested homeostatic value or inclusive bounds were invalid.
    InvalidHomeostaticBounds {
        /// Name of the local quantity.
        field: &'static str,
        /// Requested value.
        requested: f32,
        /// Inclusive lower bound.
        min: f32,
        /// Inclusive upper bound.
        max: f32,
    },
}

impl fmt::Display for NeuronError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => {
                write!(formatter, "invalid neuron configuration: {error}")
            }
            Self::InvalidPosition(error) => write!(formatter, "invalid neuron position: {error}"),
            Self::TimeWentBackwards { current, requested } => write!(
                formatter,
                "cannot advance neuron from {current} backwards to {requested}"
            ),
            Self::NonFiniteInput(value) => {
                write!(formatter, "neuron input must be finite, got {value}")
            }
            Self::InvalidInputMagnitude(value) => {
                write!(
                    formatter,
                    "input magnitude must be finite and non-negative, got {value}"
                )
            }
            Self::InvalidInhibitoryInputMagnitude { inhibitory, total } => write!(
                formatter,
                "inhibitory input magnitude must be finite and within 0..={total}, got {inhibitory}"
            ),
            Self::NonFiniteMembranePotential(value) => {
                write!(formatter, "membrane potential became non-finite: {value}")
            }
            Self::NonFiniteIntrinsicState { field, value } => {
                write!(
                    formatter,
                    "intrinsic state {field} became non-finite: {value}"
                )
            }
            Self::NonFiniteIntrinsicEquilibrium(value) => {
                write!(
                    formatter,
                    "intrinsic-current equilibrium became non-finite: {value}"
                )
            }
            Self::RefractoryTimeOverflow {
                spike_time,
                refractory_period_us,
            } => write!(
                formatter,
                "adding refractory period {refractory_period_us} us to spike time {spike_time} overflows simulation time"
            ),
            Self::InvalidThreshold(value) => write!(
                formatter,
                "threshold must be finite and above resting/reset potential, got {value}"
            ),
            Self::InvalidThresholdBounds {
                requested,
                min,
                max,
            } => write!(
                formatter,
                "invalid threshold clamp: requested={requested}, min={min}, max={max}"
            ),
            Self::InvalidHomeostaticValue { field, value } => {
                write!(formatter, "{field} must be finite, got {value}")
            }
            Self::InvalidHomeostaticBounds {
                field,
                requested,
                min,
                max,
            } => write!(
                formatter,
                "invalid {field} clamp: requested={requested}, min={min}, max={max}"
            ),
        }
    }
}

impl Error for NeuronError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidConfig(error) => Some(error),
            Self::InvalidPosition(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::IntrinsicDynamicsConfig;

    use super::*;

    fn neuron(polarity: Polarity) -> Neuron {
        Neuron::new(
            NeuronId(1),
            Position3D::ORIGIN,
            polarity,
            None,
            NeuronConfig {
                resting_potential: 0.0,
                reset_potential: 0.0,
                threshold: 1.0,
                membrane_tau_us: 10.0,
                refractory_period_us: 5,
                activity_trace_tau_us: 100.0,
                intrinsic: Default::default(),
            },
            SimTime::ZERO,
        )
        .expect("valid test neuron")
    }

    fn close(left: f32, right: f32) {
        assert!((left - right).abs() <= 1.0e-6, "{left} != {right}");
    }

    fn dynamic_neuron(intrinsic: IntrinsicDynamicsConfig, refractory_period_us: u64) -> Neuron {
        Neuron::new(
            NeuronId(9),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            None,
            NeuronConfig {
                resting_potential: 0.0,
                reset_potential: 0.0,
                threshold: 1.0,
                membrane_tau_us: 10_000.0,
                refractory_period_us,
                activity_trace_tau_us: 100_000.0,
                intrinsic,
            },
            SimTime::ZERO,
        )
        .unwrap()
    }

    #[test]
    fn lif_potential_decays_analytically_between_events() {
        let mut neuron = neuron(Polarity::Excitatory);
        assert_eq!(neuron.integrate_input(SimTime(0), 0.5), Ok(None));

        neuron.advance_to(SimTime(10)).expect("forward time");

        close(neuron.membrane_potential(), 0.5 / std::f32::consts::E);
    }

    #[test]
    fn default_intrinsics_preserve_the_legacy_f32_lif_path_exactly() {
        let mut neuron = neuron(Polarity::Excitatory);
        neuron.integrate_input(SimTime::ZERO, 0.625).unwrap();
        neuron.set_intrinsic_current(12_500.0).unwrap();
        let equilibrium = neuron.params().resting_potential
            + neuron.intrinsic_current() * (neuron.params().membrane_tau_us / 1_000_000.0);
        let expected = decay_towards(
            neuron.membrane_potential(),
            equilibrium,
            7,
            neuron.params().membrane_tau_us,
        );

        neuron.advance_to(SimTime(7)).unwrap();

        assert_eq!(neuron.membrane_potential().to_bits(), expected.to_bits());
    }

    #[test]
    fn intrinsic_current_depends_on_elapsed_time_not_update_count() {
        let mut one_update = neuron(Polarity::Excitatory);
        let mut many_updates = one_update.clone();
        one_update.set_intrinsic_current(100_000.0).unwrap();
        many_updates.set_intrinsic_current(100_000.0).unwrap();

        one_update.advance_to(SimTime(10)).unwrap();
        for time in [SimTime(1), SimTime(4), SimTime(10)] {
            many_updates.advance_to(time).unwrap();
        }

        let expected = 1.0 - (-1.0_f32).exp();
        close(one_update.membrane_potential(), expected);
        close(many_updates.membrane_potential(), expected);
        close(
            one_update.membrane_potential(),
            many_updates.membrane_potential(),
        );
    }

    #[test]
    fn threshold_crossing_emits_spike_and_resets() {
        let mut neuron = neuron(Polarity::Excitatory);

        let spike = neuron
            .integrate_input(SimTime(7), 1.0)
            .expect("valid input")
            .expect("threshold crossing");

        assert_eq!(spike, Spike::new(NeuronId(1), SimTime(7)));
        assert_eq!(neuron.membrane_potential(), 0.0);
        assert_eq!(neuron.refractory_until(), SimTime(12));
        assert_eq!(neuron.activity_trace(), 1.0);
        assert_eq!(neuron.spike_count(), 1);
    }

    #[test]
    fn input_is_ignored_during_refractory_interval() {
        let mut neuron = neuron(Polarity::Excitatory);
        neuron
            .integrate_input(SimTime(1), 1.0)
            .expect("valid first input");

        assert_eq!(neuron.integrate_input(SimTime(2), 100.0), Ok(None));
        assert_eq!(neuron.spike_count(), 1);
    }

    #[test]
    fn rejects_backwards_time_without_mutating_timestamp() {
        let mut neuron = neuron(Polarity::Excitatory);
        neuron.advance_to(SimTime(10)).expect("forward time");

        assert!(matches!(
            neuron.advance_to(SimTime(9)),
            Err(NeuronError::TimeWentBackwards { .. })
        ));
        assert_eq!(neuron.last_update(), SimTime(10));
    }

    #[test]
    fn refractory_timestamp_overflow_is_rejected_before_spike_state_mutates() {
        let mut neuron = neuron(Polarity::Excitatory);
        let error = neuron.integrate_input(SimTime(u64::MAX), 1.0).unwrap_err();

        assert!(matches!(error, NeuronError::RefractoryTimeOverflow { .. }));
        assert_eq!(neuron.spike_count(), 0);
        assert_eq!(neuron.last_spike(), None);
    }

    #[test]
    fn polarity_is_the_only_source_of_synaptic_sign() {
        assert_eq!(Polarity::Excitatory.apply(0.5), 0.5);
        assert_eq!(Polarity::Inhibitory.apply(0.5), -0.5);
    }

    #[test]
    fn threshold_homeostasis_is_clamped_locally() {
        let mut neuron = neuron(Polarity::Excitatory);

        assert_eq!(neuron.set_threshold_clamped(2.0, 0.5, 1.5), Ok(1.5));
        assert_eq!(neuron.threshold(), 1.5);
    }

    #[test]
    fn metadata_role_does_not_change_dynamics() {
        let params = NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            membrane_tau_us: 10.0,
            refractory_period_us: 5,
            activity_trace_tau_us: 100.0,
            intrinsic: Default::default(),
        };
        let mut sensory = Neuron::new(
            NeuronId(1),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            Some(NeuronRole::Sensory),
            params,
            SimTime::ZERO,
        )
        .unwrap();
        let mut motor = Neuron::new(
            NeuronId(2),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            Some(NeuronRole::Motor),
            params,
            SimTime::ZERO,
        )
        .unwrap();

        assert!(sensory.integrate_input(SimTime(1), 1.0).unwrap().is_some());
        assert!(motor.integrate_input(SimTime(1), 1.0).unwrap().is_some());
        assert_eq!(sensory.membrane_potential(), motor.membrane_potential());
    }

    #[test]
    fn one_spike_updates_all_configured_intrinsic_capabilities() {
        let mut neuron = dynamic_neuron(
            IntrinsicDynamicsConfig {
                intrinsic_drive: 3.0,
                burst_gain: 40.0,
                burst_tau_us: 20_000.0,
                adaptation_gain: 10.0,
                adaptation_tau_us: 30_000.0,
                threshold_adaptation_gain: 0.25,
                threshold_adaptation_tau_us: 40_000.0,
                rebound_gain: 5.0,
                rebound_tau_us: 50_000.0,
            },
            0,
        );

        assert!(
            neuron
                .integrate_input(SimTime::ZERO, 1.0)
                .unwrap()
                .is_some()
        );
        let state = neuron.intrinsic_state();
        assert_eq!(state.intrinsic_drive, 3.0);
        assert_eq!(state.burst_drive, 40.0);
        assert_eq!(state.adaptation_drive, 10.0);
        assert_eq!(state.threshold_adaptation, 0.25);
        assert_eq!(state.effective_threshold, 1.25);
        assert_eq!(state.rebound_drive, 0.0);
    }

    #[test]
    fn intrinsic_trajectory_is_independent_of_unrelated_update_count() {
        let intrinsic = IntrinsicDynamicsConfig {
            burst_gain: 400.0,
            burst_tau_us: 20_000.0,
            adaptation_gain: 70.0,
            adaptation_tau_us: 30_000.0,
            threshold_adaptation_gain: 0.4,
            threshold_adaptation_tau_us: 40_000.0,
            ..IntrinsicDynamicsConfig::default()
        };
        let mut one_update = dynamic_neuron(intrinsic, 0);
        one_update.integrate_input(SimTime::ZERO, 1.0).unwrap();
        let mut many_updates = one_update.clone();

        one_update.advance_to(SimTime(5_000)).unwrap();
        for time in [SimTime(500), SimTime(2_000), SimTime(5_000)] {
            many_updates.advance_to(time).unwrap();
        }

        assert!(
            (one_update.membrane_potential() - many_updates.membrane_potential()).abs() < 1.0e-5
        );
        let one = one_update.intrinsic_state();
        let many = many_updates.intrinsic_state();
        assert!((one.burst_drive - many.burst_drive).abs() < 1.0e-5);
        assert!((one.adaptation_drive - many.adaptation_drive).abs() < 1.0e-5);
        assert!((one.threshold_adaptation - many.threshold_adaptation).abs() < 1.0e-5);
    }

    #[test]
    fn inhibitory_component_creates_rebound_even_when_net_input_cancels() {
        let mut neuron = dynamic_neuron(
            IntrinsicDynamicsConfig {
                rebound_gain: 400.0,
                rebound_tau_us: 20_000.0,
                ..IntrinsicDynamicsConfig::default()
            },
            0,
        );

        assert_eq!(
            neuron.integrate_input_with_components(SimTime::ZERO, 0.0, 2.0, 1.0),
            Ok(None)
        );
        assert_eq!(neuron.intrinsic_state().rebound_drive, 400.0);
        assert!(neuron.predicted_intrinsic_spike_at(SimTime::ZERO).is_some());
    }

    #[test]
    fn adaptation_can_cancel_a_burst_capability_continuously() {
        let mut burst_only = dynamic_neuron(
            IntrinsicDynamicsConfig {
                burst_gain: 400.0,
                burst_tau_us: 20_000.0,
                ..IntrinsicDynamicsConfig::default()
            },
            0,
        );
        let mut balanced = dynamic_neuron(
            IntrinsicDynamicsConfig {
                burst_gain: 400.0,
                burst_tau_us: 20_000.0,
                adaptation_gain: 400.0,
                adaptation_tau_us: 20_000.0,
                ..IntrinsicDynamicsConfig::default()
            },
            0,
        );
        burst_only.integrate_input(SimTime::ZERO, 1.0).unwrap();
        balanced.integrate_input(SimTime::ZERO, 1.0).unwrap();

        assert!(
            burst_only
                .predicted_intrinsic_spike_at(SimTime::ZERO)
                .is_some()
        );
        assert_eq!(balanced.predicted_intrinsic_spike_at(SimTime::ZERO), None);
    }

    #[test]
    fn transient_crossing_during_refractory_time_is_not_shifted_to_release() {
        let mut neuron = dynamic_neuron(
            IntrinsicDynamicsConfig {
                burst_gain: 150_000.0,
                burst_tau_us: 10.0,
                ..IntrinsicDynamicsConfig::default()
            },
            10_000,
        );
        neuron.integrate_input(SimTime::ZERO, 1.0).unwrap();

        assert_eq!(neuron.predicted_intrinsic_spike_at(SimTime::ZERO), None);
    }

    #[test]
    fn threshold_adaptation_decays_back_towards_the_base_threshold() {
        let mut neuron = dynamic_neuron(
            IntrinsicDynamicsConfig {
                threshold_adaptation_gain: 0.5,
                threshold_adaptation_tau_us: 10_000.0,
                ..IntrinsicDynamicsConfig::default()
            },
            0,
        );
        neuron.integrate_input(SimTime::ZERO, 1.0).unwrap();
        assert_eq!(neuron.base_threshold(), 1.0);
        assert_eq!(neuron.threshold(), 1.5);

        neuron.advance_to(SimTime(10_000)).unwrap();

        let expected = 1.0 + 0.5 / std::f32::consts::E;
        assert!((neuron.threshold() - expected).abs() < 1.0e-6);
        assert_eq!(neuron.base_threshold(), 1.0);
    }

    #[test]
    fn adaptation_suppresses_a_repeated_identical_input() {
        let mut control = dynamic_neuron(IntrinsicDynamicsConfig::default(), 0);
        let mut adaptive = dynamic_neuron(
            IntrinsicDynamicsConfig {
                adaptation_gain: 2_000.0,
                adaptation_tau_us: 20_000.0,
                ..IntrinsicDynamicsConfig::default()
            },
            0,
        );
        control.integrate_input(SimTime::ZERO, 1.0).unwrap();
        adaptive.integrate_input(SimTime::ZERO, 1.0).unwrap();

        assert!(
            control
                .integrate_input(SimTime(1_000), 1.0)
                .unwrap()
                .is_some()
        );
        assert_eq!(adaptive.integrate_input(SimTime(1_000), 1.0), Ok(None));
    }

    #[test]
    fn bounded_predictor_matches_brute_force_integer_microseconds() {
        let cases = [
            IntrinsicDynamicsConfig {
                burst_gain: 4_000.0,
                burst_tau_us: 500.0,
                adaptation_gain: 3_200.0,
                adaptation_tau_us: 900.0,
                threshold_adaptation_gain: 0.15,
                threshold_adaptation_tau_us: 700.0,
                ..IntrinsicDynamicsConfig::default()
            },
            IntrinsicDynamicsConfig {
                burst_gain: 700.0,
                burst_tau_us: 20_000.0,
                adaptation_gain: 1_200.0,
                adaptation_tau_us: 4_000.0,
                threshold_adaptation_gain: 0.4,
                threshold_adaptation_tau_us: 8_000.0,
                ..IntrinsicDynamicsConfig::default()
            },
            IntrinsicDynamicsConfig {
                burst_gain: 400.0,
                burst_tau_us: 10_000.0,
                adaptation_gain: 390.0,
                adaptation_tau_us: 10_100.0,
                rebound_gain: 600.0,
                rebound_tau_us: 2_000.0,
                ..IntrinsicDynamicsConfig::default()
            },
        ];

        for intrinsic in cases {
            let mut neuron = dynamic_neuron(intrinsic, 25);
            neuron
                .integrate_input_with_components(SimTime::ZERO, 1.0, 3.0, 1.0)
                .unwrap();
            let brute_force = (25..=200_000_u64)
                .find(|elapsed| neuron.projected_will_fire(*elapsed))
                .map(SimTime);
            let predicted = neuron.predicted_intrinsic_spike_at(SimTime::ZERO);

            assert_eq!(predicted.filter(|time| time.0 <= 200_000), brute_force);
        }
    }
}
