//! Named grouping of related fixed fibers.

use super::FiberId;

/// A non-owning collection of fibers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bundle {
    /// Human-readable diagnostics label.
    pub name: String,
    /// Members in deterministic routing order.
    pub fibers: Vec<FiberId>,
}

impl Bundle {
    /// Creates a bundle and removes duplicate IDs while retaining first occurrence.
    pub fn new(name: impl Into<String>, fibers: impl IntoIterator<Item = FiberId>) -> Self {
        let mut unique = Vec::new();
        for fiber in fibers {
            if !unique.contains(&fiber) {
                unique.push(fiber);
            }
        }
        Self {
            name: name.into(),
            fibers: unique,
        }
    }
}
