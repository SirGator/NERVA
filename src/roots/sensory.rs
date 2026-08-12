//! M0 pattern input root.

use super::{Root, RootChannel, RootDirection, RootId};

/// Stable input attachment used by the four-symbol encoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatternInputRoot {
    root: Root,
}

impl PatternInputRoot {
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
