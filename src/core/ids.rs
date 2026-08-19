//! Compatibility exports for stable identities used by the M0 core.
//!
//! The canonical definitions live in [`crate::primitives`], allowing later
//! layers to share identities without depending on core neuron logic.

pub use crate::primitives::{NeuronId, SynapseId};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_are_orderable_hashable_and_round_trip() {
        let first = NeuronId::new(4);
        let second = NeuronId::from(5);
        let set = HashSet::from([first, second]);

        assert!(first < second);
        assert!(set.contains(&NeuronId(4)));
        assert_eq!(u64::from(second), 5);
    }

    #[test]
    fn different_id_domains_do_not_compare_accidentally() {
        let neuron = NeuronId(1);
        let synapse = SynapseId(1);

        assert_eq!(neuron.get(), synapse.get());
        // A direct equality comparison intentionally does not compile: the
        // wrapper types keep the identity domains distinct.
    }
}
