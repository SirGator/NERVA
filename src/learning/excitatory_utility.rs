//! Excitatory causal utility rule.
//!
//! This is the first concrete [`UtilityRule`]. It rewards an excitatory
//! synapse `i → j` when the postsynaptic neuron `j` fires shortly after a
//! presynaptic spike arrived through the connection. The reward is
//! proportional to the synapse's utility eligibility trace `e_ij` at the
//! moment of the postsynaptic spike:
//!
//! ```text
//! u_ij = 1 - exp(-e_ij)
//! ```
//!
//! This shape saturates at 1.0 for strong eligibility and stays near zero for
//! weak eligibility, so a single coincident arrival is a mild positive signal
//! while a burst of closely spaced arrivals is a strong one.
//!
//! ## Scope
//!
//! Only **excitatory** synapses are evaluated. Inhibitory synapses return
//! `None` from [`ExcitatoryCausalUtility::evidence`]: their utility is not
//! captured by a postsynaptic-spike trigger, because an inhibitory connection
//! is most useful precisely when the postsynaptic cell does *not* fire. A
//! dedicated inhibitory utility rule is deferred to a later slice.
//!
//! ## No owned time constant
//!
//! The rule is **stateless** and owns no `τ_e`. The runtime decays `e_ij`
//! with its canonical [`UtilityDynamicsConfig::eligibility_tau_us`] and hands
//! the already-decayed value to the rule via [`UtilityContext::eligibility`].
//! This guarantees a single source of truth for the eligibility dynamics.

use crate::{
    core::Polarity,
    learning::{UtilityContext, UtilityRule, UtilityTrigger},
};

/// Excitatory causal utility rule.
///
/// At a postsynaptic spike, for each incoming **excitatory** synapse, the rule
/// reads the decayed utility eligibility `e_ij` (supplied by the runtime via
/// [`UtilityContext::eligibility`]) and returns
///
/// ```text
/// u_ij = 1 - exp(-e_ij)
/// ```
///
/// For inhibitory synapses, or for triggers other than [`UtilityTrigger::PostSpike`],
/// it returns `None` (no statement). Because inhibitory utility is not yet
/// defined, [`UtilityRule::supports`] only accepts
/// [`Polarity::Excitatory`]: inhibitory synapses are exempt from
/// utility-based pruning until a dedicated rule exists.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExcitatoryCausalUtility;

impl ExcitatoryCausalUtility {
    /// Creates a new excitatory causal utility rule.
    pub const fn new() -> Self {
        Self
    }
}

impl UtilityRule for ExcitatoryCausalUtility {
    fn supports(&self, polarity: Polarity) -> bool {
        polarity == Polarity::Excitatory
    }

    fn evidence(&self, context: &UtilityContext<'_>) -> Option<f32> {
        if context.trigger != UtilityTrigger::PostSpike {
            return None;
        }
        if context.presynaptic_polarity != Polarity::Excitatory {
            return None;
        }
        let e_ij = context.eligibility;
        if !e_ij.is_finite() || e_ij <= 0.0 {
            return None;
        }
        Some(1.0 - (-e_ij).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::NeuronConfig,
        core::{Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
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

    fn context_with<'a>(
        synapse: &'a Synapse,
        post: &'a Neuron,
        polarity: Polarity,
        now: SimTime,
        trigger: UtilityTrigger,
        eligibility: f32,
    ) -> UtilityContext<'a> {
        UtilityContext {
            synapse,
            post_neuron: post,
            presynaptic_polarity: polarity,
            now,
            trigger,
            eligibility,
        }
    }

    #[test]
    fn returns_none_for_non_post_spike_triggers() {
        let (synapse, post) = make_synapse_and_post();
        let rule = ExcitatoryCausalUtility::new();
        for trigger in [UtilityTrigger::PreArrival, UtilityTrigger::Maintenance] {
            let ctx = context_with(
                &synapse,
                &post,
                Polarity::Excitatory,
                SimTime::ZERO,
                trigger,
                1.0,
            );
            assert_eq!(rule.evidence(&ctx), None);
        }
    }

    #[test]
    fn returns_none_for_inhibitory_synapse() {
        let (synapse, post) = make_synapse_and_post();
        let rule = ExcitatoryCausalUtility::new();
        let ctx = context_with(
            &synapse,
            &post,
            Polarity::Inhibitory,
            SimTime::ZERO,
            UtilityTrigger::PostSpike,
            1.0,
        );
        assert_eq!(rule.evidence(&ctx), None);
    }

    #[test]
    fn returns_none_when_eligibility_is_zero() {
        let (synapse, post) = make_synapse_and_post();
        let rule = ExcitatoryCausalUtility::new();
        let ctx = context_with(
            &synapse,
            &post,
            Polarity::Excitatory,
            SimTime::ZERO,
            UtilityTrigger::PostSpike,
            0.0,
        );
        assert_eq!(rule.evidence(&ctx), None, "zero eligibility → no statement");
    }

    #[test]
    fn returns_positive_evidence_for_positive_eligibility() {
        let (synapse, post) = make_synapse_and_post();
        let rule = ExcitatoryCausalUtility::new();
        // e_ij = 1.0 → u_ij = 1 - exp(-1) ≈ 0.6321.
        let ctx = context_with(
            &synapse,
            &post,
            Polarity::Excitatory,
            SimTime::ZERO,
            UtilityTrigger::PostSpike,
            1.0,
        );
        let evidence = rule.evidence(&ctx).expect("positive evidence");
        let expected = 1.0 - (-1.0_f32).exp();
        assert!(
            (evidence - expected).abs() <= 1.0e-5,
            "evidence {evidence} should be ~{expected}"
        );
        assert!(evidence > 0.0 && evidence < 1.0);
    }

    #[test]
    fn evidence_saturates_for_strong_eligibility() {
        let (synapse, post) = make_synapse_and_post();
        let rule = ExcitatoryCausalUtility::new();
        let ctx = context_with(
            &synapse,
            &post,
            Polarity::Excitatory,
            SimTime::ZERO,
            UtilityTrigger::PostSpike,
            10.0,
        );
        let evidence = rule.evidence(&ctx).expect("strong positive evidence");
        assert!(
            evidence > 0.99,
            "strong eligibility should saturate near 1.0, got {evidence}"
        );
    }
}
