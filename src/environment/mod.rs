//! Neutral interfaces between experiments and their environments.

mod bit_world;
#[allow(clippy::module_inception)]
mod environment;

pub use bit_world::BitWorld;
pub use environment::{Action, Environment, Observation, Pattern, SequenceEnvironment};
