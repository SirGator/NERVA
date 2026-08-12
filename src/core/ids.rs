//! Stable identifiers. Geometry is deliberately absent from identity.

use std::fmt;

macro_rules! id_type {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u64);

        impl $name {
            /// Wraps the stable integer representation.
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            /// Returns the stable integer representation.
            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl From<u64> for $name {
            fn from(value: u64) -> Self {
                Self(value)
            }
        }

        impl From<$name> for u64 {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

id_type!(NeuronId, "Stable identity of one neuron within a network.");
id_type!(
    SynapseId,
    "Stable identity of one directed synapse within a network."
);

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
