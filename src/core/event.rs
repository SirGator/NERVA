//! Precisely timestamped core inputs. Queue ordering belongs to `runtime`.

use super::{NeuronId, SynapseId};

pub use crate::primitives::{SimTime, SimTimeError};

/// Payload of an event consumed by the runtime.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventKind {
    /// A root or nerve contributes directly to an input neuron.
    ExternalInput {
        /// Receiving neuron.
        target: NeuronId,
        /// Signed contribution to the target potential.
        amplitude: f32,
    },
    /// A previously emitted spike arrives through a core synapse.
    SynapticArrival {
        /// Connection through which the spike travelled.
        synapse_id: SynapseId,
        /// Receiving neuron, captured for efficient timestamp batching.
        target: NeuronId,
        /// Signed, distance-attenuated contribution captured for this arrival.
        amplitude: f32,
    },
    /// Local maintenance of one neuron's own input/output estimates and slow
    /// cellular or structural state.
    Homeostasis {
        /// Neuron whose local state is maintained.
        neuron_id: NeuronId,
    },
    /// Predicted local threshold crossing caused by continuous intrinsic
    /// drive, burst, adaptation, rebound, or threshold recovery. This is
    /// scheduled only for that one neuron.
    IntrinsicSpike {
        /// Neuron whose intrinsic state predicted the crossing.
        neuron_id: NeuronId,
    },
}

/// One timestamped input to the core state model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event {
    /// Exact simulation timestamp.
    pub time: SimTime,
    /// Local operation to execute at that timestamp.
    pub kind: EventKind,
}

impl Event {
    /// Creates a timestamped event without assigning queue insertion order.
    pub const fn new(time: SimTime, kind: EventKind) -> Self {
        Self { time, kind }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn time_arithmetic_is_exact_and_checked() {
        let start = SimTime::from_micros(10);
        let end = start.saturating_add_us(15);

        assert_eq!(end.as_micros(), 25);
        assert_eq!(end.duration_since(start), Some(15));
        assert_eq!(start.duration_since(end), None);
    }

    #[test]
    fn saturating_addition_does_not_wrap_replay_time() {
        assert_eq!(SimTime(u64::MAX).saturating_add_us(1), SimTime(u64::MAX));
    }

    #[test]
    fn duration_conversion_rejects_unrepresentable_microseconds() {
        let duration = Duration::new(u64::MAX, 999_999_999);
        assert!(matches!(
            SimTime::try_from(duration),
            Err(SimTimeError::DurationOutOfRange(_))
        ));
    }
}
