//! Values sampled from an external system.

use crate::primitives::SimTime;

use super::ChannelId;

/// One timestamped value published on a device-neutral receptor channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReceptorSignal {
    /// Channel whose physical meaning is defined by the adapter.
    pub channel: ChannelId,
    /// Simulation time at which the value became observable.
    pub at: SimTime,
    /// Current value supplied by the adapter.
    pub value: f32,
}

/// Event-time source of external values.
///
/// A receptor knows channels, values, and simulation time, but never neuron
/// IDs. [`crate::nerves::Mapping`] owns the physical neural association.
pub trait Receptor {
    /// Removes and returns all signals observable at or before `until`.
    fn observations_until(&mut self, until: SimTime) -> Vec<ReceptorSignal>;
}
