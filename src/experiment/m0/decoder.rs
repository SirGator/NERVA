//! M0 motor-root output decoder trait and bit decoder.

use crate::{io::ChannelId, roots::MotorOutput};

use super::Action;

/// Converts observed motor-channel spikes into M0 environment actions.
pub trait Decoder {
    /// Decodes a read-only output batch.
    fn decode(&self, outputs: &[MotorOutput]) -> Vec<Action>;
}

/// Two-channel decoder: channel 0 clears and channel 1 sets a bit.
#[derive(Clone, Copy, Debug, Default)]
pub struct BitDecoder;

impl Decoder for BitDecoder {
    fn decode(&self, outputs: &[MotorOutput]) -> Vec<Action> {
        outputs
            .iter()
            .filter_map(|output| match output.channel {
                ChannelId(0) => Some(Action::SetBit(false)),
                ChannelId(1) => Some(Action::SetBit(true)),
                _ => None,
            })
            .collect()
    }
}