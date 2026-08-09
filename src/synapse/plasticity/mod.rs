//! Synaptic plasticity mechanisms.

/// Eligibility-trace parameters and state.
pub mod eligibility;
/// Spike-timing-dependent plasticity parameters and state.
pub mod stdp;

pub use eligibility::{EligibilityParams, EligibilityState};
pub use stdp::{StdpParams, StdpState};

#[derive(Debug)]
/// Plasticity rule and associated mutable learning state of a synapse.
pub enum Plasticity {
    /// Keeps the synaptic weight fixed.
    Fixed,

    /// Learns with pair-based spike-timing-dependent plasticity.
    PairStdp {
        /// Fixed STDP learning parameters.
        params: StdpParams,
        /// Mutable STDP traces and timestamps.
        state: StdpState,
    },

    /// Learns with STDP gated by a decaying eligibility trace.
    RewardModulatedStdp {
        /// Fixed STDP learning parameters.
        stdp_params: StdpParams,
        /// Mutable STDP traces and timestamps.
        stdp_state: StdpState,

        /// Fixed eligibility-trace parameters.
        eligibility_params: EligibilityParams,
        /// Mutable eligibility-trace state.
        eligibility_state: EligibilityState,
    },
}
