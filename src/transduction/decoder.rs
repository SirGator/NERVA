//! Motor-root output decoders.

use crate::roots::MotorOutput;

use super::Action;

/// Converts observed motor-channel spikes into environment actions.
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
                0 => Some(Action::SetBit(false)),
                1 => Some(Action::SetBit(true)),
                _ => None,
            })
            .collect()
    }
}
