//! Analytical leaky integrate-and-fire neuron state.

use std::{error::Error, fmt};

use crate::{
    config::{ConfigError, NeuronConfig},
    math::{Position3D, PositionError, decay_to_zero, decay_towards},
};

use super::{NeuronId, SimTime, Spike};

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
    threshold: f32,
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

    /// Current local firing threshold, including controlled adaptations.
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Constant intrinsic current in potential units per second.
    pub const fn intrinsic_current(&self) -> f32 {
        self.intrinsic_current
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
        if !self.membrane_potential.is_finite() {
            return Err(NeuronError::NonFiniteMembranePotential(
                self.membrane_potential,
            ));
        }
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

        self.integrate_input_with_magnitude(time, summed_input, summed_input.abs())
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
        if !summed_input.is_finite() {
            return Err(NeuronError::NonFiniteInput(summed_input));
        }
        if !input_magnitude.is_finite() || input_magnitude < 0.0 {
            return Err(NeuronError::InvalidInputMagnitude(input_magnitude));
        }

        self.advance_to(time)?;
        self.observe_input(input_magnitude);
        if self.is_refractory_at(time) {
            return Ok(None);
        }

        let next_potential = self.membrane_potential + summed_input;
        if !next_potential.is_finite() {
            return Err(NeuronError::NonFiniteMembranePotential(next_potential));
        }
        if next_potential < self.threshold {
            self.membrane_potential = next_potential;
            return Ok(None);
        }

        let refractory_until = time
            .checked_add_us(self.params.refractory_period_us)
            .ok_or(NeuronError::RefractoryTimeOverflow {
                spike_time: time,
                refractory_period_us: self.params.refractory_period_us,
            })?;
        self.membrane_potential = self.params.reset_potential;
        self.refractory_until = refractory_until;
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

    /// Predicts the next threshold crossing caused solely by the constant
    /// intrinsic current, starting from the already current local state.
    pub fn predicted_intrinsic_spike_at(&self, now: SimTime) -> Option<SimTime> {
        if now != self.last_update || self.membrane_potential >= self.threshold {
            return None;
        }

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

    fn observe_input(&mut self, input_magnitude: f32) {
        // Input history is a diagnostic/homeostatic estimate. Saturating it
        // preserves valid membrane dynamics even for an extreme but finite
        // simultaneous batch such as `MAX + -MAX`.
        self.input_trace = ((f64::from(self.input_trace) + f64::from(input_magnitude))
            .min(f64::from(f32::MAX))) as f32;
    }
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
    /// Finite operands overflowed the membrane representation.
    NonFiniteMembranePotential(f32),
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
            Self::NonFiniteMembranePotential(value) => {
                write!(formatter, "membrane potential became non-finite: {value}")
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
            },
            SimTime::ZERO,
        )
        .expect("valid test neuron")
    }

    fn close(left: f32, right: f32) {
        assert!((left - right).abs() <= 1.0e-6, "{left} != {right}");
    }

    #[test]
    fn lif_potential_decays_analytically_between_events() {
        let mut neuron = neuron(Polarity::Excitatory);
        assert_eq!(neuron.integrate_input(SimTime(0), 0.5), Ok(None));

        neuron.advance_to(SimTime(10)).expect("forward time");

        close(neuron.membrane_potential(), 0.5 / std::f32::consts::E);
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
}
