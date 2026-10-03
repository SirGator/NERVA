//! Directed synaptic connection state.

use std::{error::Error, fmt};

use crate::{
    math::decay_to_zero,
    primitives::{SignalStrength, Weight},
};

use super::{NeuronId, Polarity, SimTime, SynapseId};

/// A directed connection whose weight is always a non-negative magnitude.
///
/// Excitatory versus inhibitory effect is intentionally absent: callers derive
/// the sign exclusively from the presynaptic neuron's [`Polarity`] when a
/// signed [`SignalStrength`] amplitude is formed. This makes it impossible for
/// an STDP update to flip cell polarity.
#[derive(Clone, Debug, PartialEq)]
pub struct Synapse {
    /// Stable identity of this connection.
    pub(crate) id: SynapseId,
    /// Presynaptic neuron identity.
    pub(crate) pre: NeuronId,
    /// Postsynaptic neuron identity.
    pub(crate) post: NeuronId,
    /// Current non-negative connection magnitude.
    pub(crate) weight: Weight,
    /// Strictly positive propagation delay in microseconds.
    pub(crate) delay_us: u64,
    /// Whether a local learning rule may change this connection.
    pub(crate) plastic: bool,
    /// Whether spikes currently propagate through this connection.
    pub(crate) enabled: bool,
    /// Local exponentially decaying trace of presynaptic arrivals.
    pub(crate) pre_trace: f32,
    /// Timestamp to which [`Self::pre_trace`] has been decayed.
    pub(crate) pre_trace_updated_at: Option<SimTime>,
    /// Number of arrivals recorded for local diagnostics/statistics.
    pub(crate) transmission_count: u64,
    /// Synapse-local **utility eligibility trace** `e_ij`.
    ///
    /// This is structurally parallel to [`Self::pre_trace`] but is owned by
    /// the structural-utility pathway, not by STDP. It is incremented on every
    /// presynaptic arrival through [`Self::record_utility_arrival`] and decays
    /// exponentially toward zero with time constant `τ_e`:
    ///
    /// `e_ij(t) = e_ij(t_0) · exp(-(t − t_0) / τ_e)`
    ///
    /// The decay is applied lazily inside [`Self::record_utility_arrival`] and
    /// [`Self::advance_utility_eligibility_to`], and can be observed without
    /// mutation via [`Self::utility_eligibility_at`]. Keeping this trace
    /// separate from [`Self::pre_trace`] ensures structural utility does not
    /// depend on whether a concrete STDP rule happens to update `pre_trace`.
    pub(crate) utility_eligibility: f32,
    /// Timestamp to which [`Self::utility_eligibility`] has been decayed.
    pub(crate) utility_eligibility_updated_at: Option<SimTime>,
    /// Synapse-local smoothed utility estimate used by structural pruning.
    ///
    /// This is an exponential moving average of recent per-arrival utility
    /// samples that additionally **decays toward zero** between updates with
    /// time constant `τ_U`:
    ///
    /// `U(t)  = U(t₀) · exp(-(t − t₀) / τ_U)`           (memory decay)
    /// `U'(t) = (1 − η) · U(t) + η · u_ij`               (EMA on decayed value)
    ///
    /// The decay is applied lazily inside [`Self::update_utility`] and can be
    /// observed without mutation via [`Self::utility_at`]. When `U` stays
    /// below `pruning_threshold` for long enough, the pruning slice may remove
    /// this connection. Defaults to zero so a new or never-useful synapse is
    /// immediately eligible for pruning if no positive utility is ever
    /// recorded.
    pub(crate) utility: f32,
    /// Timestamp of the last [`Self::update_utility`] call.
    pub(crate) utility_updated_at: Option<SimTime>,
}

impl Synapse {
    /// Constructs an enabled synapse with an empty local trace.
    pub fn new(
        id: SynapseId,
        pre: NeuronId,
        post: NeuronId,
        weight: Weight,
        delay_us: u64,
        plastic: bool,
    ) -> Result<Self, SynapseError> {
        validate_weight(weight)?;
        validate_delay(delay_us)?;

        Ok(Self {
            id,
            pre,
            post,
            weight,
            delay_us,
            plastic,
            enabled: true,
            pre_trace: 0.0,
            pre_trace_updated_at: None,
            transmission_count: 0,
            utility_eligibility: 0.0,
            utility_eligibility_updated_at: None,
            utility: 0.0,
            utility_updated_at: None,
        })
    }

    /// Stable identity of this connection.
    pub const fn id(&self) -> SynapseId {
        self.id
    }

    /// Presynaptic neuron identity.
    pub const fn pre(&self) -> NeuronId {
        self.pre
    }

    /// Postsynaptic neuron identity.
    pub const fn post(&self) -> NeuronId {
        self.post
    }

    /// Current non-negative connection magnitude.
    pub const fn weight(&self) -> Weight {
        self.weight
    }

    /// Strictly positive propagation delay in microseconds.
    pub const fn delay_us(&self) -> u64 {
        self.delay_us
    }

    /// Whether a local learning rule may change this connection.
    pub const fn is_plastic(&self) -> bool {
        self.plastic
    }

    /// Whether spikes currently propagate through this connection.
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Local exponentially decaying trace of presynaptic arrivals.
    pub const fn pre_trace(&self) -> f32 {
        self.pre_trace
    }

    /// Timestamp to which [`Self::pre_trace`] has been decayed.
    pub const fn pre_trace_updated_at(&self) -> Option<SimTime> {
        self.pre_trace_updated_at
    }

    /// Number of arrivals recorded for local diagnostics/statistics.
    pub const fn transmission_count(&self) -> u64 {
        self.transmission_count
    }

    /// Revalidates invariants after trusted crate-internal state restoration.
    pub fn validate(&self) -> Result<(), SynapseError> {
        validate_weight(self.weight)?;
        validate_delay(self.delay_us)?;
        if !self.pre_trace.is_finite() || self.pre_trace < 0.0 {
            return Err(SynapseError::InvalidPreTrace(self.pre_trace));
        }
        if !self.utility_eligibility.is_finite() || self.utility_eligibility < 0.0 {
            return Err(SynapseError::InvalidUtilityEligibility(
                self.utility_eligibility,
            ));
        }
        Ok(())
    }

    /// Returns the signed signal amplitude implied by the emitting neuron's
    /// polarity.
    ///
    /// A `Weight` is a non-negative magnitude; the signed contribution of a
    /// connection is a distinct domain, expressed as a `SignalStrength`.
    pub fn signed_amplitude(&self, presynaptic_polarity: Polarity) -> SignalStrength {
        presynaptic_polarity.apply_to_weight(self.weight)
    }

    /// Applies a validated spatial attenuation factor to the signed amplitude.
    pub fn effective_amplitude(
        &self,
        presynaptic_polarity: Polarity,
        attenuation: f32,
    ) -> Result<SignalStrength, SynapseError> {
        if !attenuation.is_finite() || !(0.0..=1.0).contains(&attenuation) {
            return Err(SynapseError::InvalidAttenuation(attenuation));
        }
        Ok(SignalStrength::new(
            presynaptic_polarity.sign() * self.weight.get() * attenuation,
        ))
    }

    /// Changes the weight without permitting a sign-bearing negative value.
    pub fn set_weight(&mut self, weight: Weight) -> Result<(), SynapseError> {
        validate_weight(weight)?;
        self.weight = weight;
        Ok(())
    }

    /// Changes the propagation delay while preserving strict causal ordering.
    pub fn set_delay_us(&mut self, delay_us: u64) -> Result<(), SynapseError> {
        validate_delay(delay_us)?;
        self.delay_us = delay_us;
        Ok(())
    }

    /// Enables or disables local plasticity for this connection.
    pub fn set_plastic(&mut self, plastic: bool) {
        self.plastic = plastic;
    }

    /// Enables or disables spike propagation through this connection.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Clamps a proposed weight to inclusive non-negative bounds and returns the
    /// value actually stored.
    pub fn set_weight_clamped(
        &mut self,
        proposed: Weight,
        min_weight: Weight,
        max_weight: Weight,
    ) -> Result<Weight, SynapseError> {
        validate_weight(proposed)?;
        validate_weight(min_weight)?;
        validate_weight(max_weight)?;
        if min_weight > max_weight {
            return Err(SynapseError::InvalidWeightBounds {
                min: min_weight.get(),
                max: max_weight.get(),
            });
        }

        let clamped = proposed.clamp_to(min_weight, max_weight);
        self.weight = clamped;
        Ok(clamped)
    }

    /// Decays the local presynaptic trace to an exact time.
    pub fn advance_pre_trace_to(&mut self, time: SimTime, tau_us: f32) -> Result<(), SynapseError> {
        if !tau_us.is_finite() || tau_us <= 0.0 {
            return Err(SynapseError::InvalidTraceTimeConstant(tau_us));
        }

        if let Some(last_update) = self.pre_trace_updated_at {
            let elapsed_us =
                time.duration_since(last_update)
                    .ok_or(SynapseError::TraceTimeWentBackwards {
                        current: last_update,
                        requested: time,
                    })?;
            self.pre_trace = decay_to_zero(self.pre_trace, elapsed_us, tau_us);
        }
        self.pre_trace_updated_at = Some(time);
        Ok(())
    }

    /// Decays then increments the local presynaptic trace for one arrival.
    pub fn record_pre_arrival(&mut self, time: SimTime, tau_us: f32) -> Result<(), SynapseError> {
        self.advance_pre_trace_to(time, tau_us)?;
        self.pre_trace += 1.0;
        self.transmission_count = self.transmission_count.saturating_add(1);
        Ok(())
    }

    /// Records propagation without touching the learning trace.
    pub fn record_transmission(&mut self) {
        self.transmission_count = self.transmission_count.saturating_add(1);
    }

    /// Current synapse-local utility eligibility trace `e_ij`.
    pub const fn utility_eligibility(&self) -> f32 {
        self.utility_eligibility
    }

    /// Timestamp to which the utility eligibility trace has been decayed.
    pub const fn utility_eligibility_updated_at(&self) -> Option<SimTime> {
        self.utility_eligibility_updated_at
    }

    /// Decays the utility eligibility trace to an exact time without
    /// incrementing it.
    ///
    /// `e_ij(t) = e_ij(t_0) · exp(-(t − t_0) / τ_e)`
    ///
    /// `tau_e` is the eligibility time constant in microseconds and must be
    /// strictly positive and finite.
    pub fn advance_utility_eligibility_to(
        &mut self,
        time: SimTime,
        tau_e: f32,
    ) -> Result<(), SynapseError> {
        if !tau_e.is_finite() || tau_e <= 0.0 {
            return Err(SynapseError::InvalidEligibilityTimeConstant(tau_e));
        }
        if let Some(last_update) = self.utility_eligibility_updated_at {
            let elapsed_us = time.duration_since(last_update).ok_or(
                SynapseError::EligibilityTimeWentBackwards {
                    current: last_update,
                    requested: time,
                },
            )?;
            self.utility_eligibility = decay_to_zero(self.utility_eligibility, elapsed_us, tau_e);
        }
        self.utility_eligibility_updated_at = Some(time);
        Ok(())
    }

    /// Decays then increments the utility eligibility trace for one arrival.
    ///
    /// This is the structural-utility analogue of [`Self::record_pre_arrival`]
    /// and is intentionally independent of STDP: it is called by the runtime
    /// on every presynaptic arrival through a plastic connection, regardless
    /// of whether a concrete `PlasticityRule` updates `pre_trace`.
    pub fn record_utility_arrival(
        &mut self,
        time: SimTime,
        tau_e: f32,
    ) -> Result<(), SynapseError> {
        self.advance_utility_eligibility_to(time, tau_e)?;
        self.utility_eligibility += 1.0;
        Ok(())
    }

    /// Projected utility eligibility at `now` under exponential decay, without
    /// mutation.
    ///
    /// Analogous to [`Self::utility_at`]. If the trace has never been updated,
    /// the projected value is zero.
    pub fn utility_eligibility_at(&self, now: SimTime, tau_e: f32) -> Result<f32, SynapseError> {
        if !tau_e.is_finite() || tau_e <= 0.0 {
            return Err(SynapseError::InvalidEligibilityTimeConstant(tau_e));
        }
        let Some(last_update) = self.utility_eligibility_updated_at else {
            return Ok(0.0);
        };
        let elapsed_us =
            now.duration_since(last_update)
                .ok_or(SynapseError::EligibilityTimeWentBackwards {
                    current: last_update,
                    requested: now,
                })?;
        Ok(decay_to_zero(self.utility_eligibility, elapsed_us, tau_e))
    }

    /// Current synapse-local smoothed utility estimate.
    pub const fn utility(&self) -> f32 {
        self.utility
    }

    /// Timestamp of the last utility update, if any.
    pub const fn utility_updated_at(&self) -> Option<SimTime> {
        self.utility_updated_at
    }

    /// Projected utility at `now` under exponential decay, without mutation.
    ///
    /// `U(t) = U(t₀) · exp(-(t − t₀) / τ_U)`
    ///
    /// If the synapse has never been updated, the projected utility is zero.
    /// `tau_u` is the utility memory time constant in microseconds; it must be
    /// strictly positive and finite. This is the read side of the
    /// event-driven utility memory: callers (in particular the
    /// `PruningController`) observe the decayed value without forcing an
    /// update, preserving NERVA's no-global-tick design.
    pub fn utility_at(&self, now: SimTime, tau_u: f32) -> Result<f32, SynapseError> {
        if !tau_u.is_finite() || tau_u <= 0.0 {
            return Err(SynapseError::InvalidUtilityTimeConstant(tau_u));
        }
        let Some(last_update) = self.utility_updated_at else {
            return Ok(0.0);
        };
        let elapsed_us =
            now.duration_since(last_update)
                .ok_or(SynapseError::UtilityTimeWentBackwards {
                    current: last_update,
                    requested: now,
                })?;
        Ok(decay_to_zero(self.utility, elapsed_us, tau_u))
    }

    /// Decays then exponentially smooths the synapse-local utility estimate.
    ///
    /// `U(t)  = U(t₀) · exp(-(t − t₀) / τ_U)`           (memory decay)
    /// `U'(t) = (1 − η) · U(t) + η · sample`             (EMA on decayed value)
    ///
    /// The smoothing factor `eta` is clamped to `[0, 1]`. A synapse with no
    /// prior update initializes `U` to the sample directly, bypassing the
    /// decay step. `tau_u` is the utility memory time constant in microseconds
    /// and must be strictly positive and finite. This is a purely local
    /// quantity: no global population statistic is involved.
    pub fn update_utility(
        &mut self,
        sample: f32,
        eta: f32,
        time: SimTime,
        tau_u: f32,
    ) -> Result<f32, SynapseError> {
        if !sample.is_finite() {
            return Err(SynapseError::NonFiniteUtility(sample));
        }
        if !eta.is_finite() || !(0.0..=1.0).contains(&eta) {
            return Err(SynapseError::InvalidEta(eta));
        }
        if !tau_u.is_finite() || tau_u <= 0.0 {
            return Err(SynapseError::InvalidUtilityTimeConstant(tau_u));
        }
        if let Some(prev_time) = self.utility_updated_at
            && time < prev_time
        {
            return Err(SynapseError::UtilityTimeWentBackwards {
                current: prev_time,
                requested: time,
            });
        }
        let new_utility = if self.utility_updated_at.is_none() {
            sample
        } else {
            let decayed = self.utility_at(time, tau_u)?;
            (1.0 - eta) * decayed + eta * sample
        };
        self.utility = new_utility;
        self.utility_updated_at = Some(time);
        Ok(new_utility)
    }
}

fn validate_weight(weight: Weight) -> Result<(), SynapseError> {
    if !weight.is_finite() {
        return Err(SynapseError::NonFiniteWeight(weight.get()));
    }
    if weight.get() < 0.0 {
        return Err(SynapseError::NegativeWeight(weight.get()));
    }
    Ok(())
}

fn validate_delay(delay_us: u64) -> Result<(), SynapseError> {
    if delay_us == 0 {
        Err(SynapseError::ZeroDelay)
    } else {
        Ok(())
    }
}

/// Invalid synapse construction or local state transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SynapseError {
    /// A weight magnitude is NaN or infinite.
    NonFiniteWeight(f32),
    /// A negative weight would duplicate/violate presynaptic polarity.
    NegativeWeight(f32),
    /// Zero propagation delay has undefined same-batch causal semantics.
    ZeroDelay,
    /// Spatial attenuation must be finite and in `0..=1`.
    InvalidAttenuation(f32),
    /// Inclusive weight bounds are reversed.
    InvalidWeightBounds {
        /// Lower magnitude bound.
        min: f32,
        /// Upper magnitude bound.
        max: f32,
    },
    /// The trace decay time constant is non-finite or non-positive.
    InvalidTraceTimeConstant(f32),
    /// A directly supplied local trace is non-finite or negative.
    InvalidPreTrace(f32),
    /// A trace decay was requested before its last update.
    TraceTimeWentBackwards {
        /// Current trace timestamp.
        current: SimTime,
        /// Earlier requested timestamp.
        requested: SimTime,
    },
    /// A utility sample was NaN or infinite.
    NonFiniteUtility(f32),
    /// The utility smoothing factor was outside `[0, 1]`.
    InvalidEta(f32),
    /// The utility memory time constant was non-finite or non-positive.
    InvalidUtilityTimeConstant(f32),
    /// The utility eligibility time constant was non-finite or non-positive.
    InvalidEligibilityTimeConstant(f32),
    /// The utility eligibility trace was non-finite or negative.
    InvalidUtilityEligibility(f32),
    /// A utility eligibility decay was requested before its last update.
    EligibilityTimeWentBackwards {
        /// Current eligibility timestamp.
        current: SimTime,
        /// Earlier requested timestamp.
        requested: SimTime,
    },
    /// A utility update was requested before its last update.
    UtilityTimeWentBackwards {
        /// Current utility timestamp.
        current: SimTime,
        /// Earlier requested timestamp.
        requested: SimTime,
    },
}

impl fmt::Display for SynapseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteWeight(value) => {
                write!(formatter, "synaptic weight must be finite, got {value}")
            }
            Self::NegativeWeight(value) => write!(
                formatter,
                "synaptic weight is a non-negative magnitude, got {value}"
            ),
            Self::ZeroDelay => {
                formatter.write_str("synaptic delay must be at least one microsecond")
            }
            Self::InvalidAttenuation(value) => write!(
                formatter,
                "spatial attenuation must be finite and between zero and one, got {value}"
            ),
            Self::InvalidWeightBounds { min, max } => {
                write!(
                    formatter,
                    "weight bounds are reversed: min={min}, max={max}"
                )
            }
            Self::InvalidTraceTimeConstant(value) => {
                write!(
                    formatter,
                    "trace time constant must be positive, got {value}"
                )
            }
            Self::InvalidPreTrace(value) => {
                write!(
                    formatter,
                    "presynaptic trace must be finite and non-negative, got {value}"
                )
            }
            Self::TraceTimeWentBackwards { current, requested } => write!(
                formatter,
                "cannot decay synapse trace from {current} backwards to {requested}"
            ),
            Self::NonFiniteUtility(value) => {
                write!(formatter, "utility sample must be finite, got {value}")
            }
            Self::InvalidEta(value) => {
                write!(
                    formatter,
                    "utility smoothing factor must be in [0, 1], got {value}"
                )
            }
            Self::InvalidUtilityTimeConstant(value) => {
                write!(
                    formatter,
                    "utility memory time constant must be positive, got {value}"
                )
            }
            Self::InvalidEligibilityTimeConstant(value) => {
                write!(
                    formatter,
                    "utility eligibility time constant must be positive, got {value}"
                )
            }
            Self::InvalidUtilityEligibility(value) => {
                write!(
                    formatter,
                    "utility eligibility trace must be finite and non-negative, got {value}"
                )
            }
            Self::EligibilityTimeWentBackwards { current, requested } => write!(
                formatter,
                "cannot decay utility eligibility from {current} backwards to {requested}"
            ),
            Self::UtilityTimeWentBackwards { current, requested } => write!(
                formatter,
                "cannot update utility from {current} backwards to {requested}"
            ),
        }
    }
}

impl Error for SynapseError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::WeightError;

    fn synapse(weight: f32) -> Result<Synapse, SynapseError> {
        Synapse::new(
            SynapseId(10),
            NeuronId(1),
            NeuronId(2),
            Weight::new(weight).map_err(synapse_weight_error)?,
            5,
            true,
        )
    }

    fn synapse_weight_error(error: WeightError) -> SynapseError {
        match error {
            WeightError::NonFinite(value) => SynapseError::NonFiniteWeight(value),
            WeightError::Negative(value) => SynapseError::NegativeWeight(value),
        }
    }

    #[test]
    fn rejects_negative_weight_to_protect_sender_polarity() {
        assert!(matches!(
            Weight::new(-0.1),
            Err(WeightError::Negative(-0.1))
        ));
        assert!(matches!(
            synapse(-0.1),
            Err(SynapseError::NegativeWeight(-0.1))
        ));
        assert!(matches!(
            Weight::new(f32::NAN),
            Err(WeightError::NonFinite(_))
        ));
        assert!(matches!(
            Weight::new(f32::INFINITY),
            Err(WeightError::NonFinite(_))
        ));
    }

    #[test]
    fn sender_polarity_determines_effect_sign() {
        let synapse = synapse(0.5).expect("valid synapse");

        assert_eq!(
            synapse.signed_amplitude(Polarity::Excitatory),
            SignalStrength::new(0.5)
        );
        assert_eq!(
            synapse.signed_amplitude(Polarity::Inhibitory),
            SignalStrength::new(-0.5)
        );
    }

    #[test]
    fn learning_cannot_cross_zero() {
        let mut synapse = synapse(0.1).expect("valid synapse");

        assert_eq!(
            synapse.set_weight_clamped(
                Weight::new(0.0).unwrap(),
                Weight::new(0.05).unwrap(),
                Weight::new(1.0).unwrap()
            ),
            Ok(Weight::new(0.05).unwrap())
        );
        assert_eq!(synapse.weight(), Weight::new(0.05).unwrap());
        // The type itself rejects a negative magnitude, so no rule can even
        // propose a sign-bearing weight.
        assert!(matches!(
            Weight::new(-0.1),
            Err(WeightError::Negative(-0.1))
        ));
    }

    #[test]
    fn local_pre_trace_decays_before_increment() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        synapse
            .record_pre_arrival(SimTime(0), 10.0)
            .expect("first arrival");
        synapse
            .record_pre_arrival(SimTime(10), 10.0)
            .expect("second arrival");

        let expected = 1.0 + 1.0 / std::f32::consts::E;
        assert!((synapse.pre_trace() - expected).abs() <= 1.0e-6);
        assert_eq!(synapse.transmission_count(), 2);
    }

    #[test]
    fn public_accessors_cover_state_and_mutations_preserve_invariants() {
        let mut synapse = synapse(0.5).expect("valid synapse");

        assert_eq!(synapse.id(), SynapseId(10));
        assert_eq!(synapse.pre(), NeuronId(1));
        assert_eq!(synapse.post(), NeuronId(2));
        assert_eq!(synapse.weight(), Weight::new(0.5).unwrap());
        assert_eq!(synapse.delay_us(), 5);
        assert!(synapse.is_plastic());
        assert!(synapse.is_enabled());
        assert_eq!(synapse.pre_trace(), 0.0);
        assert_eq!(synapse.pre_trace_updated_at(), None);
        assert_eq!(synapse.transmission_count(), 0);

        synapse.set_delay_us(8).expect("positive delay");
        synapse.set_plastic(false);
        synapse.set_enabled(false);

        assert_eq!(synapse.delay_us(), 8);
        assert!(!synapse.is_plastic());
        assert!(!synapse.is_enabled());
        // A negative magnitude cannot be constructed, preserving the weight.
        assert!(matches!(
            Weight::new(-0.1),
            Err(WeightError::Negative(-0.1))
        ));
        assert_eq!(synapse.weight(), Weight::new(0.5).unwrap());
        assert_eq!(synapse.set_delay_us(0), Err(SynapseError::ZeroDelay));
        assert_eq!(synapse.delay_us(), 8);
    }

    #[test]
    fn zero_delay_is_rejected_to_keep_batches_causal() {
        assert_eq!(
            Synapse::new(
                SynapseId(1),
                NeuronId(1),
                NeuronId(2),
                Weight::new(0.5).unwrap(),
                0,
                false,
            ),
            Err(SynapseError::ZeroDelay)
        );
    }

    #[test]
    fn utility_at_decays_exponentially_without_mutation() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        // Seed the utility memory at t0 = 0 with U = 0.8.
        synapse
            .update_utility(0.8, 1.0, SimTime::ZERO, 1_000_000.0)
            .expect("seed utility");
        let stored = synapse.utility();
        assert!((stored - 0.8).abs() <= 1.0e-6);

        // Read the decayed value at one time constant later.
        let projected = synapse
            .utility_at(SimTime(1_000_000), 1_000_000.0)
            .expect("decayed read");
        let expected = 0.8 / std::f32::consts::E;
        assert!(
            (projected - expected).abs() <= 1.0e-5,
            "projected {projected} should be ~{expected}"
        );

        // The stored value must be unchanged: utility_at does not mutate.
        assert!(
            (synapse.utility() - stored).abs() <= 1.0e-7,
            "utility_at must not mutate stored utility"
        );
        assert_eq!(synapse.utility_updated_at(), Some(SimTime::ZERO));
    }

    #[test]
    fn utility_at_returns_zero_for_never_updated_synapse() {
        let synapse = synapse(0.5).expect("valid synapse");
        assert_eq!(
            synapse.utility_at(SimTime(123), 1_000_000.0).unwrap(),
            0.0,
            "never-updated synapse projects zero utility"
        );
        assert_eq!(synapse.utility_updated_at(), None);
    }

    #[test]
    fn utility_at_rejects_non_positive_tau() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        synapse
            .update_utility(0.5, 1.0, SimTime::ZERO, 1_000_000.0)
            .unwrap();
        assert!(matches!(
            synapse.utility_at(SimTime(1), 0.0),
            Err(SynapseError::InvalidUtilityTimeConstant(0.0))
        ));
        assert!(matches!(
            synapse.utility_at(SimTime(1), f32::NAN),
            Err(SynapseError::InvalidUtilityTimeConstant(_))
        ));
    }

    #[test]
    fn utility_at_rejects_time_going_backwards() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        synapse
            .update_utility(0.5, 1.0, SimTime(100), 1_000_000.0)
            .unwrap();
        assert!(matches!(
            synapse.utility_at(SimTime(50), 1_000_000.0),
            Err(SynapseError::UtilityTimeWentBackwards { .. })
        ));
    }

    #[test]
    fn update_utility_decays_then_smooths_on_existing_memory() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        // Seed U = 1.0 at t = 0.
        synapse
            .update_utility(1.0, 1.0, SimTime::ZERO, 1_000_000.0)
            .unwrap();
        // After one full time constant, decayed U = 1/e ≈ 0.3679.
        // With eta = 0.5 and sample = 0.0:
        // U' = 0.5 * (1/e) + 0.5 * 0 = 0.5/e ≈ 0.1839.
        let new_u = synapse
            .update_utility(0.0, 0.5, SimTime(1_000_000), 1_000_000.0)
            .expect("decay-then-EMA");
        let expected = 0.5 / std::f32::consts::E;
        assert!(
            (new_u - expected).abs() <= 1.0e-5,
            "new utility {new_u} should be ~{expected}"
        );
        assert_eq!(synapse.utility_updated_at(), Some(SimTime(1_000_000)));
    }

    #[test]
    fn update_utility_first_call_seeds_utility_directly_without_decay() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        let new_u = synapse
            .update_utility(0.7, 0.3, SimTime(42), 1_000_000.0)
            .expect("seed");
        assert!(
            (new_u - 0.7).abs() <= 1.0e-7,
            "first update should seed directly, got {new_u}"
        );
        assert_eq!(synapse.utility(), 0.7);
        assert_eq!(synapse.utility_updated_at(), Some(SimTime(42)));
    }

    #[test]
    fn update_utility_rejects_invalid_tau_and_sample_and_eta() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        synapse
            .update_utility(0.5, 1.0, SimTime(10), 1_000_000.0)
            .unwrap();
        assert!(matches!(
            synapse.update_utility(0.5, 0.5, SimTime(11), 0.0),
            Err(SynapseError::InvalidUtilityTimeConstant(0.0))
        ));
        assert!(matches!(
            synapse.update_utility(f32::NAN, 0.5, SimTime(11), 1_000_000.0),
            Err(SynapseError::NonFiniteUtility(_))
        ));
        assert!(matches!(
            synapse.update_utility(0.5, 1.5, SimTime(11), 1_000_000.0),
            Err(SynapseError::InvalidEta(1.5))
        ));
        assert!(matches!(
            synapse.update_utility(0.5, 0.5, SimTime(5), 1_000_000.0),
            Err(SynapseError::UtilityTimeWentBackwards { .. })
        ));
    }

    #[test]
    fn utility_memory_decays_to_near_zero_over_many_time_constants() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        synapse
            .update_utility(1.0, 1.0, SimTime::ZERO, 1_000_000.0)
            .unwrap();
        // After 10 time constants, decay factor is e^-10 ≈ 4.5e-5.
        let projected = synapse
            .utility_at(SimTime(10_000_000), 1_000_000.0)
            .unwrap();
        assert!(
            projected < 1.0e-3,
            "decayed utility should be near zero, got {projected}"
        );
        assert!(projected > 0.0, "decayed utility should still be positive");
    }

    #[test]
    fn utility_eligibility_starts_at_zero_and_increments_on_arrival() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        assert_eq!(synapse.utility_eligibility(), 0.0);
        assert_eq!(synapse.utility_eligibility_updated_at(), None);

        synapse
            .record_utility_arrival(SimTime::ZERO, 100_000.0)
            .unwrap();
        assert!((synapse.utility_eligibility() - 1.0).abs() <= 1.0e-6);
        assert_eq!(
            synapse.utility_eligibility_updated_at(),
            Some(SimTime::ZERO)
        );
    }

    #[test]
    fn utility_eligibility_decays_between_arrivals() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        synapse
            .record_utility_arrival(SimTime(0), 100_000.0)
            .unwrap();
        // One time constant later: e ≈ 1/e.
        let projected = synapse
            .utility_eligibility_at(SimTime(100_000), 100_000.0)
            .unwrap();
        let expected = 1.0 / std::f32::consts::E;
        assert!(
            (projected - expected).abs() <= 1.0e-5,
            "projected {projected} should be ~{expected}"
        );
        // Second arrival at t=100_000: 1/e + 1.
        synapse
            .record_utility_arrival(SimTime(100_000), 100_000.0)
            .unwrap();
        let expected_after = 1.0 + 1.0 / std::f32::consts::E;
        assert!(
            (synapse.utility_eligibility() - expected_after).abs() <= 1.0e-5,
            "after second arrival {projected} should be ~{expected_after}"
        );
    }

    #[test]
    fn utility_eligibility_is_independent_of_pre_trace() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        // Record a utility arrival without touching pre_trace.
        synapse
            .record_utility_arrival(SimTime::ZERO, 100_000.0)
            .unwrap();
        assert_eq!(
            synapse.pre_trace(),
            0.0,
            "utility arrival must not touch pre_trace"
        );
        assert_eq!(synapse.pre_trace_updated_at(), None);
        assert!((synapse.utility_eligibility() - 1.0).abs() <= 1.0e-6);

        // Now record a pre arrival: utility eligibility must not jump.
        synapse.record_pre_arrival(SimTime(10), 100_000.0).unwrap();
        assert!((synapse.pre_trace() - 1.0).abs() <= 1.0e-6);
        // The stored utility_eligibility is unchanged because record_pre_arrival
        // does not touch it. The projected value at t=10 reflects only decay.
        let projected = synapse
            .utility_eligibility_at(SimTime(10), 100_000.0)
            .unwrap();
        let expected = (-(10.0_f32 / 100_000.0_f32)).exp();
        assert!(
            (projected - expected).abs() <= 1.0e-5,
            "pre arrival must not increment utility eligibility, projected {projected} expected ~{expected}"
        );
        // And pre_trace must not have been touched by utility arrival.
        assert!((synapse.pre_trace() - 1.0).abs() <= 1.0e-6);
    }

    #[test]
    fn utility_eligibility_at_returns_zero_for_never_updated_synapse() {
        let synapse = synapse(0.5).expect("valid synapse");
        assert_eq!(
            synapse
                .utility_eligibility_at(SimTime(123), 100_000.0)
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn utility_eligibility_rejects_invalid_tau_and_backwards_time() {
        let mut synapse = synapse(0.5).expect("valid synapse");
        synapse
            .record_utility_arrival(SimTime(100), 100_000.0)
            .unwrap();
        assert!(matches!(
            synapse.advance_utility_eligibility_to(SimTime(101), 0.0),
            Err(SynapseError::InvalidEligibilityTimeConstant(0.0))
        ));
        assert!(matches!(
            synapse.utility_eligibility_at(SimTime(101), f32::NAN),
            Err(SynapseError::InvalidEligibilityTimeConstant(_))
        ));
        assert!(matches!(
            synapse.advance_utility_eligibility_to(SimTime(50), 100_000.0),
            Err(SynapseError::EligibilityTimeWentBackwards { .. })
        ));
    }
}
