//! Reproducible sparse encoder for the M0 patterns A, B, C and D.

use std::collections::BTreeMap;

use crate::environment::{Observation, Pattern};

use super::{ChannelSpike, Encoder, EncodingError};

/// Invalid pattern encoder setup.
#[derive(Clone, Debug, PartialEq)]
pub enum PatternEncoderError {
    /// A pattern was assigned no channels.
    EmptyPattern(Pattern),
    /// A channel occurs more than once for the same pattern.
    DuplicateChannel { pattern: Pattern, channel: u16 },
    /// A burst offset list was empty.
    EmptyBurst,
    /// Burst offsets must be ordered to make output deterministic.
    UnorderedBurst,
    /// Amplitude must be finite and positive.
    InvalidAmplitude,
}

/// Fixed sparse channel groups and a fixed temporal burst shape.
#[derive(Clone, Debug)]
pub struct PatternEncoder {
    channels: BTreeMap<Pattern, Vec<u16>>,
    burst_offsets_us: Vec<u64>,
    amplitude: f32,
}

impl PatternEncoder {
    /// Creates and validates a deterministic encoder.
    pub fn new(
        channels: BTreeMap<Pattern, Vec<u16>>,
        burst_offsets_us: Vec<u64>,
        amplitude: f32,
    ) -> Result<Self, PatternEncoderError> {
        for pattern in [Pattern::A, Pattern::B, Pattern::C, Pattern::D] {
            let assigned = channels
                .get(&pattern)
                .ok_or(PatternEncoderError::EmptyPattern(pattern))?;
            if assigned.is_empty() {
                return Err(PatternEncoderError::EmptyPattern(pattern));
            }
            let mut sorted = assigned.clone();
            sorted.sort_unstable();
            if let Some(pair) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
                return Err(PatternEncoderError::DuplicateChannel {
                    pattern,
                    channel: pair[0],
                });
            }
        }
        if burst_offsets_us.is_empty() {
            return Err(PatternEncoderError::EmptyBurst);
        }
        if burst_offsets_us.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(PatternEncoderError::UnorderedBurst);
        }
        if !amplitude.is_finite() || amplitude <= 0.0 {
            return Err(PatternEncoderError::InvalidAmplitude);
        }

        Ok(Self {
            channels,
            burst_offsets_us,
            amplitude,
        })
    }

    /// One-channel-per-pattern encoder suitable for compact tests and examples.
    pub fn one_hot(amplitude: f32) -> Result<Self, PatternEncoderError> {
        Self::new(
            BTreeMap::from([
                (Pattern::A, vec![0]),
                (Pattern::B, vec![1]),
                (Pattern::C, vec![2]),
                (Pattern::D, vec![3]),
            ]),
            vec![0],
            amplitude,
        )
    }

    /// Assigned channels for a pattern.
    pub fn channels(&self, pattern: Pattern) -> &[u16] {
        self.channels
            .get(&pattern)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

impl Encoder for PatternEncoder {
    fn encode(&self, observation: &Observation) -> Result<Vec<ChannelSpike>, EncodingError> {
        let Observation::Pattern { at, pattern } = *observation else {
            return Ok(Vec::new());
        };

        let mut spikes = Vec::new();
        for &offset in &self.burst_offsets_us {
            let spike_time = at
                .checked_add_us(offset)
                .ok_or(EncodingError::TimeOverflow {
                    observation_time: at,
                    offset_us: offset,
                })?;
            for &channel in self.channels(pattern) {
                spikes.push(ChannelSpike {
                    channel,
                    at: spike_time,
                    amplitude: self.amplitude,
                });
            }
        }
        Ok(spikes)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::SimTime;

    use super::*;

    #[test]
    fn one_hot_encoder_never_touches_a_neuron_directly() {
        let encoder = PatternEncoder::one_hot(1.25).unwrap();
        let spikes = encoder
            .encode(&Observation::Pattern {
                at: SimTime(7),
                pattern: Pattern::C,
            })
            .unwrap();

        assert_eq!(
            spikes,
            vec![ChannelSpike {
                channel: 2,
                at: SimTime(7),
                amplitude: 1.25,
            }]
        );
    }

    #[test]
    fn reports_burst_timestamp_overflow() {
        let encoder = PatternEncoder::new(
            BTreeMap::from([
                (Pattern::A, vec![0]),
                (Pattern::B, vec![1]),
                (Pattern::C, vec![2]),
                (Pattern::D, vec![3]),
            ]),
            vec![2],
            1.0,
        )
        .unwrap();

        assert_eq!(
            encoder.encode(&Observation::Pattern {
                at: SimTime(u64::MAX - 1),
                pattern: Pattern::A,
            }),
            Err(EncodingError::TimeOverflow {
                observation_time: SimTime(u64::MAX - 1),
                offset_us: 2,
            })
        );
    }
}
