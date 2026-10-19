//! Root identity and channel metadata.

use crate::io::ChannelId;

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

/// One declared root channel. Physical transport belongs to nerve mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootChannel {
    /// Channel identity interpreted only by transduction.
    pub channel: ChannelId,
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
    /// Builds a root and validates that channel identities are unique.
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
        Ok(Self {
            id,
            name: name.into(),
            direction,
            channels,
        })
    }
}
