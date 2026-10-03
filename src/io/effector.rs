//! Values applied to an external system.

use crate::primitives::SimTime;

use super::ChannelId;

/// One timestamped value addressed to an effector channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectorSignal {
    /// Channel whose physical meaning is defined by the adapter.
    pub channel: ChannelId,
    /// Simulation time at which the value reaches the effector.
    pub at: SimTime,
    /// Value to apply to the external system.
    pub value: f32,
}

/// Sink for values leaving NERVA.
///
/// An effector knows channels, values, and simulation time, but never neuron
/// IDs. [`crate::nerves::Mapping`] owns the physical neural association.
pub trait Effector {
    /// Applies one signal to the external adapter.
    fn apply(&mut self, signal: EffectorSignal);
}
