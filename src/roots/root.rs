//! Root identity and channel metadata.

use crate::nerves::FiberId;

/// Stable identity of an external connection point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RootId(pub u64);

/// Permitted signal direction of a root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RootDirection {
    /// Environment to the core.
    Sensory,
    /// Core to the environment.
    Motor,
}

/// One numbered root channel and its fixed fiber.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootChannel {
    /// Channel number interpreted only by transduction.
    pub channel: u16,
    /// Fixed nerve fiber used for transport.
    pub fiber: FiberId,
}

/// Common immutable root metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Root {
    /// Stable identity.
    pub id: RootId,
    /// Human-readable diagnostics label.
    pub name: String,
    /// Signal direction.
    pub direction: RootDirection,
    /// Fixed channel layout.
    pub channels: Vec<RootChannel>,
}

impl Root {
    /// Builds a root and validates that channel numbers and fibers are unique.
    pub fn new(
        id: RootId,
        name: impl Into<String>,
        direction: RootDirection,
        mut channels: Vec<RootChannel>,
    ) -> Result<Self, &'static str> {
        channels.sort_by_key(|entry| entry.channel);
        if channels
            .windows(2)
            .any(|pair| pair[0].channel == pair[1].channel)
        {
            return Err("root channel numbers must be unique");
        }
        let mut fibers: Vec<_> = channels.iter().map(|entry| entry.fiber).collect();
        fibers.sort_unstable();
        if fibers.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("root fibers must be unique");
        }
        Ok(Self {
            id,
            name: name.into(),
            direction,
            channels,
        })
    }
}
