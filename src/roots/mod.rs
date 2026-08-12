//! Stable sensor and motor connection points at the system boundary.

mod motor;
mod registry;
mod root;
mod sensory;

pub use motor::{MotorOutput, MotorRoot};
pub use registry::{RootRegistry, RootRegistryError};
pub use root::{Root, RootChannel, RootDirection, RootId};
pub use sensory::PatternInputRoot;
