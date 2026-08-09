//! Spike-timing-dependent plasticity parameters and state.

use crate::common::SimTimeUs;

/// Fixed learning rates and decay constants for pair-based STDP.
#[derive(Debug)]
pub struct StdpParams {
    /// Weight-increase rate for potentiating spike pairs.
    pub potentiation_rate: f32,
    /// Weight-decrease rate for depressing spike pairs.
    pub depression_rate: f32,
    /// Decay constant of the presynaptic trace in microseconds.
    pub pre_trace_tau_us: f32,
    /// Decay constant of the postsynaptic trace in microseconds.
    pub post_trace_tau_us: f32,
}

/// Mutable traces and timestamps for pair-based STDP.
#[derive(Debug)]
pub struct StdpState {
    /// Current decaying presynaptic trace.
    pub pre_trace: f32,
    /// Current decaying postsynaptic trace.
    pub post_trace: f32,
    /// Timestamp at which both traces were last updated.
    pub last_update_at: SimTimeUs,
    /// Timestamp of the latest presynaptic spike, if any.
    pub last_pre_spike_at: Option<SimTimeUs>,
    /// Timestamp of the latest postsynaptic spike, if any.
    pub last_post_spike_at: Option<SimTimeUs>,
}
