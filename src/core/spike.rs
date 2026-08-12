//! Immutable records of emitted spikes.

use super::{NeuronId, SimTime};

/// A neuron emission at an exact simulation time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Spike {
    /// Identity of the emitting neuron.
    pub neuron_id: NeuronId,
    /// Exact emission time.
    pub time: SimTime,
}

impl Spike {
    /// Records a new emission.
    pub const fn new(neuron_id: NeuronId, time: SimTime) -> Self {
        Self { neuron_id, time }
    }

    /// Alias exposing the emitter as the spike source.
    pub const fn source(self) -> NeuronId {
        self.neuron_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spike_preserves_identity_and_exact_time() {
        let spike = Spike::new(NeuronId(7), SimTime(123));

        assert_eq!(spike.source(), NeuronId(7));
        assert_eq!(spike.time, SimTime(123));
    }
}
