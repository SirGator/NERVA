//! Device-neutral sensory root.

use super::{Root, RootChannel, RootDirection, RootId};

/// Stable sensory attachment for transduced root-channel spikes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SensoryRoot {
    root: Root,
}

impl SensoryRoot {
    /// Builds a sensory root from fixed channels.
    pub fn new(
        id: RootId,
        name: impl Into<String>,
        channels: Vec<RootChannel>,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            root: Root::new(id, name, RootDirection::Sensory, channels)?,
        })
    }

    /// Shared root metadata.
    pub fn root(&self) -> &Root {
        &self.root
    }
}
