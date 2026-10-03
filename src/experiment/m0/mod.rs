//! M0 reference transduction: patterns, observations, actions and encoders.
//!
//! These types belong to the M0 sequence experiment. The general NERVA
//! transduction boundary lives in [`crate::transduction`] and is coding-agnostic.

mod bit_encoder;
mod bit_world;
mod decoder;
mod encoder;
mod environment;
mod pattern_encoder;
mod types;

pub use bit_encoder::{BitEncoder, BitEncoderError};
pub use bit_world::BitWorld;
pub use decoder::{BitDecoder, Decoder};
pub use encoder::{EncodingError, Encoder};
pub use environment::{Environment, SequenceEnvironment};
pub use pattern_encoder::{PatternEncoder, PatternEncoderError};
pub use types::{Action, Observation, Pattern};