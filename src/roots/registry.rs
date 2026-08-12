//! Deterministic root registry.

use std::collections::BTreeMap;

use super::{Root, RootId};

/// Root registry mutation error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RootRegistryError {
    /// The stable ID already exists.
    Duplicate(RootId),
    /// No root has the requested ID.
    Unknown(RootId),
}

/// Root metadata indexed by stable identity.
#[derive(Clone, Debug, Default)]
pub struct RootRegistry {
    roots: BTreeMap<RootId, Root>,
}

impl RootRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one root.
    pub fn insert(&mut self, root: Root) -> Result<(), RootRegistryError> {
        if self.roots.contains_key(&root.id) {
            return Err(RootRegistryError::Duplicate(root.id));
        }
        self.roots.insert(root.id, root);
        Ok(())
    }

    /// Returns immutable metadata.
    pub fn get(&self, id: RootId) -> Option<&Root> {
        self.roots.get(&id)
    }

    /// Removes metadata for one root.
    pub fn remove(&mut self, id: RootId) -> Result<Root, RootRegistryError> {
        self.roots.remove(&id).ok_or(RootRegistryError::Unknown(id))
    }

    /// Iterates in ID order.
    pub fn iter(&self) -> impl Iterator<Item = &Root> {
        self.roots.values()
    }
}
