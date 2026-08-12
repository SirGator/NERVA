//! Directed synaptic connection state.

use std::{error::Error, fmt};

use crate::math::decay_to_zero;

use super::{NeuronId, Polarity, SimTime, SynapseId};

/// A directed connection whose weight is always a non-negative magnitude.
///
/// Excitatory versus inhibitory effect is intentionally absent: callers derive
/// the sign exclusively from the presynaptic neuron's [`Polarity`]. This makes
/// it impossible for an STDP update to flip cell polarity.
#[derive(Clone, Debug, PartialEq)]
pub struct Synapse {
    /// Stable identity of this connection.
    pub(crate) id: SynapseId,
    /// Presynaptic neuron identity.
    pub(crate) pre: NeuronId,
    /// Postsynaptic neuron identity.
    pub(crate) post: NeuronId,
    /// Current non-negative connection magnitude.
    pub(crate) weight: f32,
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
}

impl Synapse {
    /// Constructs an enabled synapse with an empty local trace.
    pub fn new(
        id: SynapseId,
        pre: NeuronId,
        post: NeuronId,
        weight: f32,
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
    pub const fn weight(&self) -> f32 {
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
        Ok(())
    }

    /// Returns the signed weight implied by the emitting neuron's polarity.
    pub fn signed_weight(&self, presynaptic_polarity: Polarity) -> f32 {
        presynaptic_polarity.apply(self.weight)
    }

    /// Applies a validated spatial attenuation factor to the signed weight.
    pub fn effective_weight(
        &self,
        presynaptic_polarity: Polarity,
        attenuation: f32,
    ) -> Result<f32, SynapseError> {
        if !attenuation.is_finite() || !(0.0..=1.0).contains(&attenuation) {
            return Err(SynapseError::InvalidAttenuation(attenuation));
        }
        Ok(self.signed_weight(presynaptic_polarity) * attenuation)
    }

    /// Changes the weight without permitting a sign-bearing negative value.
    pub fn set_weight(&mut self, weight: f32) -> Result<(), SynapseError> {
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
        proposed: f32,
        min_weight: f32,
        max_weight: f32,
    ) -> Result<f32, SynapseError> {
        validate_weight(proposed)?;
        validate_weight(min_weight)?;
        validate_weight(max_weight)?;
        if min_weight > max_weight {
            return Err(SynapseError::InvalidWeightBounds {
                min: min_weight,
                max: max_weight,
            });
        }

        let clamped = proposed.clamp(min_weight, max_weight);
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
}

fn validate_weight(weight: f32) -> Result<(), SynapseError> {
    if !weight.is_finite() {
        return Err(SynapseError::NonFiniteWeight(weight));
    }
    if weight < 0.0 {
        return Err(SynapseError::NegativeWeight(weight));
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
        }
    }
}

impl Error for SynapseError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn synapse(weight: f32) -> Result<Synapse, SynapseError> {
        Synapse::new(SynapseId(10), NeuronId(1), NeuronId(2), weight, 5, true)
    }

    #[test]
    fn rejects_negative_weight_to_protect_sender_polarity() {
        assert_eq!(synapse(-0.1), Err(SynapseError::NegativeWeight(-0.1)));
    }

    #[test]
    fn sender_polarity_determines_effect_sign() {
        let synapse = synapse(0.5).expect("valid synapse");

        assert_eq!(synapse.signed_weight(Polarity::Excitatory), 0.5);
        assert_eq!(synapse.signed_weight(Polarity::Inhibitory), -0.5);
    }

    #[test]
    fn learning_cannot_cross_zero() {
        let mut synapse = synapse(0.1).expect("valid synapse");

        assert_eq!(synapse.set_weight_clamped(0.0, 0.05, 1.0), Ok(0.05));
        assert_eq!(synapse.weight(), 0.05);
        assert!(matches!(
            synapse.set_weight_clamped(-0.1, 0.0, 1.0),
            Err(SynapseError::NegativeWeight(_))
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
        assert_eq!(synapse.weight(), 0.5);
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
        assert_eq!(
            synapse.set_weight(-0.1),
            Err(SynapseError::NegativeWeight(-0.1))
        );
        assert_eq!(synapse.weight(), 0.5);
        assert_eq!(synapse.set_delay_us(0), Err(SynapseError::ZeroDelay));
        assert_eq!(synapse.delay_us(), 8);
    }

    #[test]
    fn zero_delay_is_rejected_to_keep_batches_causal() {
        assert_eq!(
            Synapse::new(SynapseId(1), NeuronId(1), NeuronId(2), 0.5, 0, false,),
            Err(SynapseError::ZeroDelay)
        );
    }
}
