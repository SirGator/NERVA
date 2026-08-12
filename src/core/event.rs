//! Precisely timestamped core inputs. Queue ordering belongs to `runtime`.

use std::{error::Error, fmt, time::Duration};

use super::{NeuronId, SynapseId};

/// Monotonic simulation time measured in integer microseconds.
///
/// Integer timestamps allow exact equality batching and deterministic replay;
/// there is no implication of a global simulation tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SimTime(pub u64);

impl SimTime {
    /// Beginning of simulation time.
    pub const ZERO: Self = Self(0);

    /// Creates a timestamp measured in microseconds.
    pub const fn from_micros(microseconds: u64) -> Self {
        Self(microseconds)
    }

    /// Returns this timestamp in microseconds.
    pub const fn as_micros(self) -> u64 {
        self.0
    }

    /// Advances by an integer number of microseconds, saturating at the largest
    /// representable timestamp.
    pub const fn saturating_add_us(self, microseconds: u64) -> Self {
        Self(self.0.saturating_add(microseconds))
    }

    /// Alias for [`Self::saturating_add_us`].
    pub const fn saturating_add(self, microseconds: u64) -> Self {
        self.saturating_add_us(microseconds)
    }

    /// Advances by an integer number of microseconds or returns `None` on
    /// overflow.
    pub const fn checked_add_us(self, microseconds: u64) -> Option<Self> {
        match self.0.checked_add(microseconds) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Elapsed microseconds since `earlier`, or `None` if time would go
    /// backwards.
    pub const fn duration_since(self, earlier: Self) -> Option<u64> {
        self.0.checked_sub(earlier.0)
    }
}

impl From<u64> for SimTime {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<SimTime> for u64 {
    fn from(value: SimTime) -> Self {
        value.0
    }
}

impl TryFrom<Duration> for SimTime {
    type Error = SimTimeError;

    fn try_from(value: Duration) -> Result<Self, Self::Error> {
        u64::try_from(value.as_micros())
            .map(Self)
            .map_err(|_| SimTimeError::DurationOutOfRange(value))
    }
}

/// A wall-clock duration cannot be represented as exact simulation time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimTimeError {
    /// Whole microseconds exceed the `u64` simulation-time domain.
    DurationOutOfRange(Duration),
}

impl fmt::Display for SimTimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DurationOutOfRange(duration) => write!(
                formatter,
                "duration of {} microseconds exceeds simulation time",
                duration.as_micros()
            ),
        }
    }
}

impl Error for SimTimeError {}

impl fmt::Display for SimTime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} us", self.0)
    }
}

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
    /// Predicted local threshold crossing caused by a neuron's continuous
    /// intrinsic current. This is scheduled only for that one neuron.
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
