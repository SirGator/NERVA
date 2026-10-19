//! Analytical leaky integrate-and-fire neuron state.

use std::{error::Error, fmt};

use crate::{
    config::{ConfigError, NeuronConfig},
    math::{Position3D, PositionError, decay_to_zero},
    primitives::{SignalStrength, Weight},
};

use super::{
    IntrinsicState, NeuronId, SimTime, Spike,
    dynamics::{NeuronDynamicsParams, NeuronDynamicsState, checked_effective_sum},
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

    /// Derives the signed signal amplitude of a typed weight magnitude.
    ///
    /// A `Weight` is never signed; the contribution sign is exclusively the
    /// polarity of the emitting cell.
    pub fn apply_to_weight(self, weight: Weight) -> SignalStrength {
        SignalStrength::new(self.sign() * weight.get())
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
        })
    }

    /// Exports a complete numerical copy without identity or graph metadata.
    pub fn numerical_state(&self) -> NeuronDynamicsState {
        NeuronDynamicsState {
            membrane_potential: self.membrane_potential,
            last_update: self.last_update,
            refractory_until: self.refractory_until,
            activity_trace: self.activity_trace,
            input_trace: self.input_trace,
            last_spike_time: self.last_spike.map_or(0, SimTime::as_micros),
            last_spike_valid: u32::from(self.last_spike.is_some()),
            spike_count: self.spike_count,
            threshold: self.threshold,
            burst_drive: self.burst_drive,
            adaptation_drive: self.adaptation_drive,
            rebound_drive: self.rebound_drive,
            threshold_adaptation: self.threshold_adaptation,
            intrinsic_current: self.intrinsic_current,
            structural_drive: self.structural_drive,
            last_homeostasis_update: self.last_homeostasis_update,
        }
    }

    /// Exports immutable flattened numerical parameters.
    pub fn numerical_params(&self) -> NeuronDynamicsParams {
        (&self.params).into()
    }

    /// Validates and publishes numerical state without changing identity,
    /// position, polarity, role or immutable configuration.
    pub fn apply_numerical_state(&mut self, state: NeuronDynamicsState) -> Result<(), NeuronError> {
        self.validate_numerical_state(&state)?;
        self.set_numerical_state_unchecked(state);
        Ok(())
    }

    pub(crate) fn validate_numerical_state(
        &self,
        state: &NeuronDynamicsState,
    ) -> Result<(), NeuronError> {
        state.validate(&self.numerical_params())?;
        if state.last_update < self.last_update {
            return Err(NeuronError::TimeWentBackwards {
                current: self.last_update,
                requested: state.last_update,
            });
        }
        Ok(())
    }

    // Batch failures preserve the original partial numerical state before poisoning.
    pub(crate) fn set_numerical_state_unchecked(&mut self, state: NeuronDynamicsState) {
        self.membrane_potential = state.membrane_potential;
        self.last_update = state.last_update;
        self.refractory_until = state.refractory_until;
        self.activity_trace = state.activity_trace;
        self.input_trace = state.input_trace;
        self.spike_count = state.spike_count;
        self.threshold = state.threshold;
        self.burst_drive = state.burst_drive;
        self.adaptation_drive = state.adaptation_drive;
        self.rebound_drive = state.rebound_drive;
        self.threshold_adaptation = state.threshold_adaptation;
        self.intrinsic_current = state.intrinsic_current;
        self.structural_drive = state.structural_drive;
        self.last_homeostasis_update = state.last_homeostasis_update;
        self.last_spike = (state.last_spike_valid == 1).then_some(SimTime(state.last_spike_time));
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

    /// Projected local spike trace at `now` without mutating neuron state.
    ///
    /// Computes `A_i(now) = A_i(t_last) * exp(-(now - t_last) / tau_A)`,
    /// returning the stored value when `now == t_last` and zero when `now`
    /// precedes `t_last`. This lets read-only consumers such as the
    /// development candidate search see a time-correct trace even when the
    /// neuron has not received an event since its last update.
    pub fn activity_trace_at(&self, now: SimTime) -> f32 {
        let Some(elapsed_us) = now.duration_since(self.last_update) else {
            return 0.0;
        };
        if elapsed_us == 0 {
            return self.activity_trace;
        }
        decay_to_zero(
            self.activity_trace,
            elapsed_us,
            self.params.activity_trace_tau_us,
        )
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
        let mut state = self.numerical_state();
        let result = state.advance_to(&self.numerical_params(), time);
        self.set_numerical_state_unchecked(state);
        result
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
        let mut state = self.numerical_state();
        let result = state.integrate_input_with_components(
            &self.numerical_params(),
            time,
            summed_input,
            input_magnitude,
            inhibitory_input_magnitude,
        );
        self.set_numerical_state_unchecked(state);
        result.map(|fired| fired.then(|| Spike::new(self.id, time)))
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
        self.numerical_state()
            .predicted_intrinsic_spike_at(&self.numerical_params(), now)
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
    /// Imported numerical state violated a local invariant.
    InvalidNumericalState { field: &'static str },
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
            Self::InvalidNumericalState { field } => {
                write!(formatter, "invalid numerical neuron state: {field}")
            }
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
    use crate::{config::IntrinsicDynamicsConfig, math::decay_towards};

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
        assert_eq!(
            Polarity::Excitatory.apply_to_weight(Weight::new(0.5).unwrap()),
            SignalStrength::new(0.5)
        );
        assert_eq!(
            Polarity::Inhibitory.apply_to_weight(Weight::new(0.5).unwrap()),
            SignalStrength::new(-0.5)
        );
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
                .find(|elapsed| {
                    neuron
                        .numerical_state()
                        .projected_will_fire(&neuron.numerical_params(), *elapsed)
                })
                .map(SimTime);
            let predicted = neuron.predicted_intrinsic_spike_at(SimTime::ZERO);

            assert_eq!(predicted.filter(|time| time.0 <= 200_000), brute_force);
        }
    }
}
