//! Errors at the event-time transduction boundary.

use std::{error::Error, fmt};

use crate::{io::ChannelId, primitives::SimTime};

/// A transducer rejected an input or simulation horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransductionError {
    /// An input belongs to an interval that has already been completed.
    InputAlreadyProcessed {
        /// Last inclusive horizon.
        completed_until: SimTime,
        /// Timestamp of the rejected input.
        at: SimTime,
    },
    /// A requested horizon is earlier than the last completed horizon.
    TimeWentBackwards {
        /// Last inclusive horizon.
        current: SimTime,
        /// Rejected horizon.
        requested: SimTime,
    },
    /// Direct spike or pulse amplitudes must be finite and strictly positive.
    InvalidAmplitude {
        /// Channel of the rejected input.
        channel: ChannelId,
        /// Rejected amplitude.
        amplitude: f32,
    },
}

impl fmt::Display for TransductionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputAlreadyProcessed {
                completed_until,
                at,
            } => write!(
                formatter,
                "input at {at} belongs to an interval already completed through {completed_until}"
            ),
            Self::TimeWentBackwards { current, requested } => write!(
                formatter,
                "transduction horizon cannot move backwards from {current} to {requested}"
            ),
            Self::InvalidAmplitude { channel, amplitude } => write!(
                formatter,
                "direct transduction amplitude on channel {channel} must be finite and greater than zero, got {amplitude}"
            ),
        }
    }
}

impl Error for TransductionError {}
