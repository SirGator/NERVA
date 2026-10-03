//! Synapse-local structural utility evidence and memory.
//!
//! This module separates **utility evidence** (what just happened that makes a
//! connection locally useful?) from **utility memory** (the synapse's smoothed,
//! decaying estimate `U_ij`, owned by [`Synapse`]). The memory side lives in
//! the core and is updated through [`Synapse::update_utility`]; the evidence
//! side is a runtime-facing boundary implemented by [`UtilityRule`].
//!
//! The split keeps NERVA's event-driven shape: there is no global tick that
//! decays every synapse. Instead, the decayed value is computed lazily inside
//! `update_utility` and read without mutation through
//! [`Synapse::utility_at`], so the pruning controller can observe the current
//! memory state at any simulation time without forcing an evidence event.
//!
//! ## EMA on the decayed value
//!
//! At an evidence event at time `t`:
//!
//! ```text
//! U(t)  = U(t_0) · exp(-(t - t_0) / tau_U)        (memory decay)
//! U'(t) = (1 - eta) · U(t) + eta · u_ij(t)         (EMA on decayed value)
//! ```
//!
//! ## Utility vs. STDP
//!
//! STDP answers *"how strong should the existing connection be?"*.
//! Utility answers *"should this connection exist at all?"*. The two are
//! deliberately separate: a synapse may carry a large STDP-potentiated weight
//! and still be pruned when its structural utility estimate decays below the
//! pruning threshold for long enough.
//!
//! ## Evidence vs. no-evidence
//!
//! [`UtilityRule::evidence`] returns [`Option<f32>`] rather than `f32`:
//!
//! - `None` → **no statement** about utility at this event. The synapse's
//!   utility memory is left untouched and simply continues to decay. This is
//!   what [`NoUtility`] returns for every event, so attaching it to the runtime
//!   is a true no-op rather than a slow drain to zero.
//! - `Some(0.0)` → **explicit evidence** that the connection was not useful at
//!   this event. The memory is updated toward zero.
//! - `Some(v > 0.0)` → positive evidence that the connection was useful.
//!
//! This distinction matters: a synapse whose rule has nothing to say should
//! not be punished by a synthetic zero sample, otherwise every connection not
//! yet covered by a concrete rule would drift toward pruning.
//!
//! ## Inhibitory caveat
//!
//! A postsynaptic spike is **not** universal evidence of utility. An
//! inhibitory synapse is most useful precisely when the postsynaptic cell does
//! *not* fire. Concrete evidence rules therefore live in implementations of
//! [`UtilityRule`]; until one is supplied, [`NoUtility`] keeps the utility
//! pathway inert.

use crate::{
    config::{ConfigError, positive},
    core::{Neuron, Polarity, SimTime, Synapse},
};

/// Canonical, validated parameters for the utility memory and eligibility
/// dynamics.
///
/// This is the **single source of truth** for the three time constants and
/// the smoothing factor that govern structural utility. The runtime owns one
/// instance, the pruning controller reads from it, and utility rules receive
/// the already-decayed eligibility value through [`UtilityContext`], so no
/// second copy of `τ_e` can disagree with the runtime's copy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UtilityDynamicsConfig {
    /// Eligibility trace time constant `τ_e` in microseconds.
    pub eligibility_tau_us: f32,
    /// EMA smoothing factor `η` for utility memory updates, in `[0, 1]`.
    pub eta: f32,
    /// Utility memory time constant `τ_U` in microseconds.
    pub utility_tau_us: f32,
}

impl UtilityDynamicsConfig {
    /// Validates all three parameters.
    pub fn validate(&self) -> Result<(), ConfigError> {
        positive(self.eligibility_tau_us, "utility.eligibility_tau_us")?;
        if !self.eta.is_finite() || !(0.0..=1.0).contains(&self.eta) {
            return Err(ConfigError::ProbabilityOutOfRange {
                field: "utility.eta",
                value: self.eta,
            });
        }
        positive(self.utility_tau_us, "utility.utility_tau_us")?;
        Ok(())
    }
}

impl Default for UtilityDynamicsConfig {
    fn default() -> Self {
        Self {
            eligibility_tau_us: 20_000.0,
            eta: 0.1,
            utility_tau_us: 1_000_000.0,
        }
    }
}

/// What kind of runtime event triggered a utility evidence evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UtilityTrigger {
    /// A presynaptic spike arrived through the synapse under evaluation.
    PreArrival,
    /// The postsynaptic neuron just emitted a spike.
    PostSpike,
    /// A local maintenance event touched the postsynaptic neuron.
    Maintenance,
}

/// Read-only context handed to a [`UtilityRule`] at one evidence opportunity.
///
/// Everything a local evidence rule needs is available here: the synapse being
/// evaluated, its postsynaptic neuron, the current simulation time, the
/// presynaptic polarity, the trigger kind, and the **already-decayed**
/// eligibility `e_ij` computed by the runtime with its canonical time
/// constant `τ_e`. Rules do not need their own copy of `τ_e`; they only
/// interpret the value the runtime hands them.
#[derive(Clone, Copy, Debug)]
pub struct UtilityContext<'a> {
    /// The synapse whose utility is being evaluated.
    pub synapse: &'a Synapse,
    /// The postsynaptic neuron of `synapse`.
    pub post_neuron: &'a Neuron,
    /// Presynaptic polarity of the neuron feeding `synapse`.
    pub presynaptic_polarity: Polarity,
    /// Current simulation time of the evidence event.
    pub now: SimTime,
    /// Which runtime hook triggered this evaluation.
    pub trigger: UtilityTrigger,
    /// Synapse-local eligibility `e_ij(t)`, already decayed to `now` by the
    /// runtime using the canonical `τ_e` from [`UtilityDynamicsConfig`].
    /// Rules read this value instead of calling
    /// [`Synapse::utility_eligibility_at`] themselves, so there is no risk of
    /// a second, conflicting `τ_e`.
    pub eligibility: f32,
}

/// Local evidence rule that maps one runtime event to an optional utility
/// sample.
///
/// The returned [`Option<f32>`] is the instantaneous evidence `u_ij` used by
/// [`Synapse::update_utility`](crate::core::Synapse::update_utility) to update
/// the synapse's smoothed, decaying utility memory:
///
/// - `None` → no statement, the utility memory is left untouched.
/// - `Some(v)` → `v` is used as the new sample. It must be finite;
///   non-finite values are rejected by `update_utility`. Negative values are
///   permitted in principle (a rule may actively penalise a connection) but
///   the memory itself is not clamped, so rule authors are responsible for
///   keeping the resulting `U_ij` well-defined.
///
/// ## Capability declaration
///
/// [`UtilityRule::supports`] declares which presynaptic polarities the rule
/// can actually judge. It separates **"utility unknown"** from **"utility
/// bad"**: a synapse whose presynaptic polarity is not supported never
/// receives utility evidence, so its memory stays at zero — but zero here
/// means *no statement*, not *bad*. The pruning controller therefore only
/// tracks synapses whose polarity the active rule supports; unsupported
/// synapses are exempt from utility-based pruning until a rule that can
/// judge them is installed.
///
/// Implementations are expected to be local: they may inspect only the synapse
/// under evaluation, its postsynaptic neuron, and the supplied trigger
/// context. There is deliberately no access to the global network, so a
/// utility rule cannot base its judgement on population statistics.
pub trait UtilityRule {
    /// Whether this rule can produce utility evidence for a synapse whose
    /// presynaptic neuron has the given `polarity`.
    ///
    /// Returning `false` exempts such synapses from utility-based pruning:
    /// their utility memory is never written, so a zero memory must not be
    /// interpreted as "useless".
    fn supports(&self, polarity: Polarity) -> bool;

    /// Computes the optional instantaneous utility sample `u_ij` for one event.
    fn evidence(&self, context: &UtilityContext<'_>) -> Option<f32>;
}

/// A utility rule that makes no statement about any event.
///
/// `supports` returns `false` for every polarity and `evidence` always
/// returns `None`, so attaching [`NoUtility`] to the runtime leaves every
/// synapse's utility memory untouched: no synthetic zero samples are
/// written, and `U_ij` simply decays according to its own time constant.
/// Because no polarity is supported, the pruning controller exempts every
/// synapse from utility-based pruning while this rule is active. This is the
/// safe default until a real evidence rule is implemented, and it is the
/// rule used by tests that do not exercise the evidence pathway.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoUtility;

impl UtilityRule for NoUtility {
    fn supports(&self, _polarity: Polarity) -> bool {
        false
    }

    fn evidence(&self, _context: &UtilityContext<'_>) -> Option<f32> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::NeuronConfig,
        core::{NeuronId, SynapseId},
        math::Position3D,
        primitives::Weight,
    };

    fn make_synapse_and_post() -> (Synapse, Neuron) {
        let synapse = Synapse::new(
            SynapseId(10),
            NeuronId(1),
            NeuronId(2),
            Weight::new(0.5).unwrap(),
            1,
            true,
        )
        .unwrap();
        let post = Neuron::new(
            NeuronId(2),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            None,
            NeuronConfig::default(),
            SimTime::ZERO,
        )
        .unwrap();
        (synapse, post)
    }

    #[test]
    fn no_utility_reports_none_for_every_trigger() {
        let (synapse, post) = make_synapse_and_post();
        let rule = NoUtility;
        for trigger in [
            UtilityTrigger::PreArrival,
            UtilityTrigger::PostSpike,
            UtilityTrigger::Maintenance,
        ] {
            let context = UtilityContext {
                synapse: &synapse,
                post_neuron: &post,
                presynaptic_polarity: Polarity::Excitatory,
                now: SimTime::ZERO,
                trigger,
                eligibility: 0.0,
            };
            assert_eq!(rule.evidence(&context), None);
        }
    }

    #[test]
    fn utility_context_carries_trigger_and_polarity() {
        let (synapse, post) = make_synapse_and_post();
        let context = UtilityContext {
            synapse: &synapse,
            post_neuron: &post,
            presynaptic_polarity: Polarity::Inhibitory,
            now: SimTime(42),
            trigger: UtilityTrigger::PreArrival,
            eligibility: 0.0,
        };
        assert_eq!(context.trigger, UtilityTrigger::PreArrival);
        assert_eq!(context.presynaptic_polarity, Polarity::Inhibitory);
        assert_eq!(context.now, SimTime(42));
        assert_eq!(context.synapse.id(), SynapseId(10));
        assert_eq!(context.post_neuron.id(), NeuronId(2));
    }

    #[test]
    fn custom_rule_can_return_positive_evidence() {
        struct AlwaysFull;
        impl UtilityRule for AlwaysFull {
            fn supports(&self, _polarity: Polarity) -> bool {
                true
            }

            fn evidence(&self, _context: &UtilityContext<'_>) -> Option<f32> {
                Some(1.0)
            }
        }
        let (synapse, post) = make_synapse_and_post();
        let context = UtilityContext {
            synapse: &synapse,
            post_neuron: &post,
            presynaptic_polarity: Polarity::Excitatory,
            now: SimTime::ZERO,
            trigger: UtilityTrigger::PreArrival,
            eligibility: 0.0,
        };
        assert_eq!(AlwaysFull.evidence(&context), Some(1.0));
    }

    #[test]
    fn custom_rule_can_distinguish_zero_evidence_from_no_statement() {
        // A rule that explicitly says "not useful" at PreArrival but has no
        // opinion at PostSpike.
        struct DrainOnArrival;
        impl UtilityRule for DrainOnArrival {
            fn supports(&self, _polarity: Polarity) -> bool {
                true
            }

            fn evidence(&self, context: &UtilityContext<'_>) -> Option<f32> {
                match context.trigger {
                    UtilityTrigger::PreArrival => Some(0.0),
                    _ => None,
                }
            }
        }
        let (synapse, post) = make_synapse_and_post();
        let arrival_ctx = UtilityContext {
            synapse: &synapse,
            post_neuron: &post,
            presynaptic_polarity: Polarity::Excitatory,
            now: SimTime::ZERO,
            trigger: UtilityTrigger::PreArrival,
            eligibility: 0.0,
        };
        let spike_ctx = UtilityContext {
            trigger: UtilityTrigger::PostSpike,
            ..arrival_ctx
        };
        assert_eq!(DrainOnArrival.evidence(&arrival_ctx), Some(0.0));
        assert_eq!(DrainOnArrival.evidence(&spike_ctx), None);
    }
}
