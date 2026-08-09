//! Eligibility-trace parameters and state.

use crate::common::SimTimeUs;

/// Fixed parameters of an eligibility trace.
#[derive(Debug)]
pub struct EligibilityParams {
    /// Exponential trace-decay time constant in microseconds.
    pub trace_tau_us: f32,
}

/// Mutable eligibility trace attached to a synapse.
#[derive(Debug)]
pub struct EligibilityState {
    /// Current eligibility value.
    pub value: f32,
    /// Timestamp at which the trace was last decayed or updated.
    pub last_update_at: SimTimeUs,
}
