//! One fixed sensory or motor connection.

use crate::core::NeuronId;

/// Stable identity of a nerve fiber.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FiberId(pub u64);

/// Direction in which a fiber transports spikes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FiberDirection {
    /// Root channel to a core neuron.
    Sensory,
    /// Core neuron to a root channel.
    Motor,
}

/// A fixed transport path. It never interprets the carried signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fiber {
    /// Stable fiber identity.
    pub id: FiberId,
    /// Transport direction.
    pub direction: FiberDirection,
    /// Core endpoint.
    pub neuron: NeuronId,
    /// Constant conduction delay in microseconds.
    pub delay_us: u64,
    /// Constant amplitude multiplier.
    pub gain: f32,
}

impl Fiber {
    /// Constructs a validated fiber.
    pub fn new(
        id: FiberId,
        direction: FiberDirection,
        neuron: NeuronId,
        delay_us: u64,
        gain: f32,
    ) -> Result<Self, &'static str> {
        if delay_us == 0 {
            return Err("fiber delay must be greater than zero");
        }
        if !gain.is_finite() || gain <= 0.0 {
            return Err("fiber gain must be finite and greater than zero");
        }
        Ok(Self {
            id,
            direction,
            neuron,
            delay_us,
            gain,
        })
    }
}
