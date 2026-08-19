//! Exact simulation-time primitives for the event-driven runtime.

use std::{error::Error, fmt, time::Duration};

/// Monotonic simulation time measured in integer microseconds.
///
/// Integer timestamps allow exact equality batching and deterministic replay;
/// there is no implication of a global simulation tick.
#[repr(transparent)]
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

    /// Advances by microseconds, saturating at the end of the time domain.
    pub const fn saturating_add_us(self, microseconds: u64) -> Self {
        Self(self.0.saturating_add(microseconds))
    }

    /// Alias for [`Self::saturating_add_us`].
    pub const fn saturating_add(self, microseconds: u64) -> Self {
        self.saturating_add_us(microseconds)
    }

    /// Advances by microseconds, returning `None` instead of wrapping.
    pub const fn checked_add_us(self, microseconds: u64) -> Option<Self> {
        match self.0.checked_add(microseconds) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Elapsed microseconds, or `None` when time would move backwards.
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

impl fmt::Display for SimTime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} us", self.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn microsecond_arithmetic_is_checked_and_monotonic() {
        let start = SimTime::from_micros(10);
        let end = start.checked_add_us(7).unwrap();

        assert_eq!(end, SimTime(17));
        assert_eq!(end.duration_since(start), Some(7));
        assert_eq!(start.duration_since(end), None);
    }

    #[test]
    fn saturating_addition_never_wraps_time() {
        assert_eq!(SimTime(u64::MAX).saturating_add_us(1), SimTime(u64::MAX));
        assert_eq!(SimTime(u64::MAX).checked_add_us(1), None);
    }
}
