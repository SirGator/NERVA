//! Stable identities for device-neutral I/O channels.

use std::fmt;

/// Stable identity of one receptor or effector channel.
///
/// The meaning of a channel belongs to the external adapter and its
/// configuration. NERVA does not interpret the numeric value.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelId(pub u64);

impl ChannelId {
    /// Creates a channel identity from its stable integer representation.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the stable integer representation.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for ChannelId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<ChannelId> for u64 {
    fn from(value: ChannelId) -> Self {
        value.0
    }
}

impl fmt::Display for ChannelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}
