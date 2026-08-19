//! Translation between external values and temporal root-channel activity.

mod bit_encoder;
mod decoder;
mod encoder;
mod pattern_encoder;
mod types;

pub use bit_encoder::{BitEncoder, BitEncoderError};
pub use decoder::{BitDecoder, Decoder};
pub use encoder::{ChannelSpike, Encoder, EncodingError};
pub use pattern_encoder::{PatternEncoder, PatternEncoderError};
pub use types::{Action, Observation, Pattern};
