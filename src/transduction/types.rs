//! Neutral values crossing the environment/transduction boundary.

use crate::primitives::SimTime;

/// The four sparse input patterns used by the M0 experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Pattern {
    /// First item of the learned sequence.
    A,
    /// Second item of the learned sequence.
    B,
    /// Third item of the learned sequence.
    C,
    /// Fourth item of the learned sequence.
    D,
}

/// One timestamped value offered by an environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observation {
    /// A symbolic M0 pattern.
    Pattern {
        /// Time at which the pattern becomes observable.
        at: SimTime,
        /// Symbolic pattern value.
        pattern: Pattern,
    },
    /// A binary sensor value.
    Bit {
        /// Time at which the bit becomes observable.
        at: SimTime,
        /// Binary sensor value.
        value: bool,
    },
}

impl Observation {
    /// Timestamp at which this observation becomes available.
    pub fn time(&self) -> SimTime {
        match *self {
            Self::Pattern { at, .. } | Self::Bit { at, .. } => at,
        }
    }
}

/// An action decoded from motor spikes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Set the binary actuator to a value.
    SetBit(bool),
    /// No externally visible change.
    NoOp,
}
