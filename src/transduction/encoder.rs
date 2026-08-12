//! Generic observation encoder.

use std::{error::Error, fmt};

use crate::{core::SimTime, environment::Observation};

/// A spike on a root-local channel, before nerve routing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelSpike {
    /// Root-local channel number.
    pub channel: u16,
    /// Emission timestamp at the root.
    pub at: SimTime,
    /// Positive stimulus amplitude.
    pub amplitude: f32,
}

/// A timestamped observation could not be represented as a spike train.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodingError {
    /// Adding a configured burst offset would exceed [`SimTime`].
    TimeOverflow {
        /// Observation timestamp before applying the burst offset.
        observation_time: SimTime,
        /// Configured offset that did not fit.
        offset_us: u64,
    },
}

impl fmt::Display for EncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimeOverflow {
                observation_time,
                offset_us,
            } => write!(
                formatter,
                "adding encoder offset {offset_us} us to observation time {observation_time} overflows simulation time"
            ),
        }
    }
}

impl Error for EncodingError {}

/// Converts neutral observations into root-channel spike trains.
pub trait Encoder {
    /// Encodes one observation without accessing core state.
    fn encode(&self, observation: &Observation) -> Result<Vec<ChannelSpike>, EncodingError>;
}
