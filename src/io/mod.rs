//! Device-neutral input and output primitives.
//!
//! Receptors and effectors exchange timestamped values through stable channels.
//! Physical channel-to-neuron connections belong exclusively to
//! [`crate::nerves::Mapping`], so adapters remain unaware of the neural model.

mod channel;
mod effector;
mod receptor;

pub use channel::ChannelId;
pub use effector::{Effector, EffectorSignal};
pub use receptor::{Receptor, ReceptorSignal};
