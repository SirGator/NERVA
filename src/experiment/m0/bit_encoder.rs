//! Deterministic two-channel encoder for binary M0 observations.

use std::{error::Error, fmt};

use crate::io::ChannelId;
use crate::transduction::ChannelSpike;

use super::{Encoder, EncodingError, Observation};

/// Invalid fixed bit-channel encoding parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BitEncoderError {
    /// False and true must remain distinguishable at the root boundary.
    DuplicateChannels(ChannelId),
    /// Encoded spike amplitude must be finite and strictly positive.
    InvalidAmplitude(f32),
}

impl fmt::Display for BitEncoderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateChannels(channel) => {
                write!(
                    formatter,
                    "bit channels must be distinct, both were {channel}"
                )
            }
            Self::InvalidAmplitude(amplitude) => write!(
                formatter,
                "bit spike amplitude must be finite and greater than zero, got {amplitude}"
            ),
        }
    }
}

impl Error for BitEncoderError {}

/// Encodes `false` and `true` as one spike on separate fixed channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BitEncoder {
    false_channel: ChannelId,
    true_channel: ChannelId,
    amplitude: f32,
}

impl BitEncoder {
    /// Creates a validated two-channel encoder.
    pub fn new(
        false_channel: ChannelId,
        true_channel: ChannelId,
        amplitude: f32,
    ) -> Result<Self, BitEncoderError> {
        if false_channel == true_channel {
            return Err(BitEncoderError::DuplicateChannels(false_channel));
        }
        if !amplitude.is_finite() || amplitude <= 0.0 {
            return Err(BitEncoderError::InvalidAmplitude(amplitude));
        }
        Ok(Self {
            false_channel,
            true_channel,
            amplitude,
        })
    }

    /// Conventional channel `0 = false`, channel `1 = true` mapping.
    pub fn binary(amplitude: f32) -> Result<Self, BitEncoderError> {
        Self::new(ChannelId(0), ChannelId(1), amplitude)
    }

    /// Channel used for a bit value.
    pub const fn channel_for(self, value: bool) -> ChannelId {
        if value {
            self.true_channel
        } else {
            self.false_channel
        }
    }
}

impl Default for BitEncoder {
    fn default() -> Self {
        Self {
            false_channel: ChannelId(0),
            true_channel: ChannelId(1),
            amplitude: 1.0,
        }
    }
}

impl Encoder for BitEncoder {
    fn encode(&self, observation: &Observation) -> Result<Vec<ChannelSpike>, EncodingError> {
        let Observation::Bit { at, value } = *observation else {
            return Ok(Vec::new());
        };
        Ok(vec![ChannelSpike {
            channel: self.channel_for(value),
            at,
            amplitude: self.amplitude,
        }])
    }
}

#[cfg(test)]
mod tests {
    use crate::core::SimTime;
    use crate::experiment::m0::Pattern;

    use super::*;

    #[test]
    fn bit_values_use_distinct_fixed_channels() {
        let encoder = BitEncoder::binary(1.25).unwrap();

        assert_eq!(
            encoder
                .encode(&Observation::Bit {
                    at: SimTime(7),
                    value: false,
                })
                .unwrap(),
            vec![ChannelSpike {
                channel: ChannelId(0),
                at: SimTime(7),
                amplitude: 1.25,
            }]
        );
        assert_eq!(encoder.channel_for(true), ChannelId(1));
    }

    #[test]
    fn ignores_observations_from_another_transduction_domain() {
        assert_eq!(
            BitEncoder::default()
                .encode(&Observation::Pattern {
                    at: SimTime::ZERO,
                    pattern: Pattern::A,
                })
                .unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn validates_channels_and_amplitude() {
        assert_eq!(
            BitEncoder::new(ChannelId(3), ChannelId(3), 1.0),
            Err(BitEncoderError::DuplicateChannels(ChannelId(3)))
        );
        assert!(matches!(
            BitEncoder::binary(f32::NAN),
            Err(BitEncoderError::InvalidAmplitude(_))
        ));
    }
}