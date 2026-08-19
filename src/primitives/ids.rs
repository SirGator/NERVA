//! Stable identities that do not depend on storage location.

use std::fmt;

macro_rules! define_id {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u64);

        impl $name {
            /// Creates an identity from its stable integer representation.
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

define_id!(NeuronId, "Stable identity of one neuron.");
define_id!(SynapseId, "Stable identity of one synapse.");
define_id!(RegionId, "Stable identity of one neural region.");
define_id!(SystemId, "Stable identity of one neural system.");
define_id!(SensorId, "Stable identity of one sensor.");
define_id!(ActuatorId, "Stable identity of one actuator.");
define_id!(ModulatorId, "Stable identity of one modulator source.");

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn identities_round_trip_and_remain_orderable() {
        let first = NeuronId::new(3);
        let second = NeuronId::from(8);
        let ids = BTreeSet::from([second, first]);

        assert_eq!(ids.into_iter().collect::<Vec<_>>(), vec![first, second]);
        assert_eq!(u64::from(second), 8);
        assert_eq!(first.to_string(), "3");
    }

    #[test]
    fn every_identity_domain_preserves_its_value() {
        assert_eq!(SynapseId::new(1).get(), 1);
        assert_eq!(RegionId::new(2).get(), 2);
        assert_eq!(SystemId::new(3).get(), 3);
        assert_eq!(SensorId::new(4).get(), 4);
        assert_eq!(ActuatorId::new(5).get(), 5);
        assert_eq!(ModulatorId::new(6).get(), 6);
    }
}
