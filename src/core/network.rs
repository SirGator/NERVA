//! Sparse directed graph with deterministic, synchronized adjacency indices.

use std::{collections::BTreeMap, error::Error, fmt};

use crate::{
    config::{NetworkConfig, NeuronConfig},
    math::{DecayError, try_distance_attenuation},
    primitives::SignalStrength,
};

use super::{Neuron, NeuronId, Synapse, SynapseError, SynapseId};

const NO_SYNAPSES: &[SynapseId] = &[];

/// Core network state. It owns no scheduler and executes no simulation loop.
///
/// Ordered maps and sorted adjacency vectors make traversal independent of hash
/// randomization and insertion order, which is essential for deterministic
/// replay.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Network {
    neurons: BTreeMap<NeuronId, Neuron>,
    synapses: BTreeMap<SynapseId, Synapse>,
    incoming: BTreeMap<NeuronId, Vec<SynapseId>>,
    outgoing: BTreeMap<NeuronId, Vec<SynapseId>>,
}

impl Network {
    /// Creates an empty graph.
    pub const fn new() -> Self {
        Self {
            neurons: BTreeMap::new(),
            synapses: BTreeMap::new(),
            incoming: BTreeMap::new(),
            outgoing: BTreeMap::new(),
        }
    }

    /// Number of neurons.
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Number of directed synapses.
    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    /// Whether the graph has no neurons.
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Adds a neuron and empty incoming/outgoing adjacency entries.
    pub fn add_neuron(&mut self, neuron: Neuron) -> Result<(), NetworkError> {
        let id = neuron.id();
        if self.neurons.contains_key(&id) {
            return Err(NetworkError::DuplicateNeuron(id));
        }

        self.neurons.insert(id, neuron);
        self.incoming.insert(id, Vec::new());
        self.outgoing.insert(id, Vec::new());
        Ok(())
    }

    /// Adds a validated synapse whose two endpoints already exist.
    pub fn add_synapse(&mut self, synapse: Synapse) -> Result<(), NetworkError> {
        synapse.validate().map_err(NetworkError::InvalidSynapse)?;
        if self.synapses.contains_key(&synapse.id()) {
            return Err(NetworkError::DuplicateSynapse(synapse.id()));
        }
        if !self.neurons.contains_key(&synapse.pre()) {
            return Err(NetworkError::MissingPresynapticNeuron(synapse.pre()));
        }
        if !self.neurons.contains_key(&synapse.post()) {
            return Err(NetworkError::MissingPostsynapticNeuron(synapse.post()));
        }

        let id = synapse.id();
        let pre = synapse.pre();
        let post = synapse.post();
        self.synapses.insert(id, synapse);
        insert_sorted(
            self.outgoing
                .get_mut(&pre)
                .expect("index exists for every stored neuron"),
            id,
        );
        insert_sorted(
            self.incoming
                .get_mut(&post)
                .expect("index exists for every stored neuron"),
            id,
        );
        Ok(())
    }

    /// Removes a synapse and both adjacency references.
    pub fn remove_synapse(&mut self, id: SynapseId) -> Result<Synapse, NetworkError> {
        let synapse = self
            .synapses
            .remove(&id)
            .ok_or(NetworkError::MissingSynapse(id))?;

        remove_sorted(
            self.outgoing
                .get_mut(&synapse.pre())
                .expect("index exists for every stored neuron"),
            id,
        );
        remove_sorted(
            self.incoming
                .get_mut(&synapse.post())
                .expect("index exists for every stored neuron"),
            id,
        );
        Ok(synapse)
    }

    /// Removes an isolated neuron. Callers must remove its connections first so
    /// topology deletion is always explicit.
    pub fn remove_neuron(&mut self, id: NeuronId) -> Result<Neuron, NetworkError> {
        if !self.neurons.contains_key(&id) {
            return Err(NetworkError::MissingNeuron(id));
        }

        let incoming = self.incoming_synapse_ids(id);
        let outgoing = self.outgoing_synapse_ids(id);
        if !incoming.is_empty() || !outgoing.is_empty() {
            return Err(NetworkError::NeuronStillConnected {
                neuron_id: id,
                incoming: incoming.len(),
                outgoing: outgoing.len(),
            });
        }

        self.incoming.remove(&id);
        self.outgoing.remove(&id);
        Ok(self
            .neurons
            .remove(&id)
            .expect("presence was checked above"))
    }

    /// Looks up one neuron by stable identity.
    pub fn neuron(&self, id: NeuronId) -> Option<&Neuron> {
        self.neurons.get(&id)
    }

    /// Mutably looks up one neuron's local state.
    pub fn neuron_mut(&mut self, id: NeuronId) -> Option<&mut Neuron> {
        self.neurons.get_mut(&id)
    }

    /// Looks up one synapse by stable identity.
    pub fn synapse(&self, id: SynapseId) -> Option<&Synapse> {
        self.synapses.get(&id)
    }

    /// Mutably looks up one synapse's local state.
    ///
    /// Public synapse mutation cannot alter topology; topology changes go
    /// through [`Self::remove_synapse`] and [`Self::add_synapse`] so indices
    /// stay in sync.
    pub fn synapse_mut(&mut self, id: SynapseId) -> Option<&mut Synapse> {
        self.synapses.get_mut(&id)
    }

    /// All neurons in ascending stable-ID order.
    pub fn neurons(&self) -> impl ExactSizeIterator<Item = &Neuron> {
        self.neurons.values()
    }

    /// All mutable neurons in ascending stable-ID order.
    pub fn neurons_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Neuron> {
        self.neurons.values_mut()
    }

    /// All neuron IDs in ascending order.
    pub fn neuron_ids(&self) -> impl ExactSizeIterator<Item = NeuronId> + '_ {
        self.neurons.keys().copied()
    }

    /// All synapses in ascending stable-ID order.
    pub fn synapses(&self) -> impl ExactSizeIterator<Item = &Synapse> {
        self.synapses.values()
    }

    /// All mutable synapses in ascending stable-ID order.
    pub fn synapses_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Synapse> {
        self.synapses.values_mut()
    }

    /// All synapse IDs in ascending order.
    pub fn synapse_ids(&self) -> impl ExactSizeIterator<Item = SynapseId> + '_ {
        self.synapses.keys().copied()
    }

    /// Sorted IDs of synapses arriving at `neuron_id`. Unknown IDs have no
    /// adjacency and therefore return an empty slice.
    pub fn incoming_synapse_ids(&self, neuron_id: NeuronId) -> &[SynapseId] {
        self.incoming
            .get(&neuron_id)
            .map(Vec::as_slice)
            .unwrap_or(NO_SYNAPSES)
    }

    /// Sorted IDs of synapses emitted by `neuron_id`. Unknown IDs have no
    /// adjacency and therefore return an empty slice.
    pub fn outgoing_synapse_ids(&self, neuron_id: NeuronId) -> &[SynapseId] {
        self.outgoing
            .get(&neuron_id)
            .map(Vec::as_slice)
            .unwrap_or(NO_SYNAPSES)
    }

    /// Immutable incoming synapses in ascending stable-ID order.
    pub fn incoming_synapses(
        &self,
        neuron_id: NeuronId,
    ) -> Result<impl Iterator<Item = &Synapse>, NetworkError> {
        let ids = self
            .incoming
            .get(&neuron_id)
            .ok_or(NetworkError::MissingNeuron(neuron_id))?;
        Ok(ids.iter().map(|id| {
            self.synapses
                .get(id)
                .expect("adjacency ID always references a stored synapse")
        }))
    }

    /// Immutable outgoing synapses in ascending stable-ID order.
    pub fn outgoing_synapses(
        &self,
        neuron_id: NeuronId,
    ) -> Result<impl Iterator<Item = &Synapse>, NetworkError> {
        let ids = self
            .outgoing
            .get(&neuron_id)
            .ok_or(NetworkError::MissingNeuron(neuron_id))?;
        Ok(ids.iter().map(|id| {
            self.synapses
                .get(id)
                .expect("adjacency ID always references a stored synapse")
        }))
    }

    /// Visits mutable incoming synapses in ascending stable-ID order.
    pub fn for_each_incoming_synapse_mut(
        &mut self,
        neuron_id: NeuronId,
        mut operation: impl FnMut(&mut Synapse),
    ) -> Result<(), NetworkError> {
        let ids = self
            .incoming
            .get(&neuron_id)
            .ok_or(NetworkError::MissingNeuron(neuron_id))?
            .clone();
        for id in ids {
            operation(
                self.synapses
                    .get_mut(&id)
                    .expect("adjacency ID always references a stored synapse"),
            );
        }
        Ok(())
    }

    /// Visits mutable outgoing synapses in ascending stable-ID order.
    pub fn for_each_outgoing_synapse_mut(
        &mut self,
        neuron_id: NeuronId,
        mut operation: impl FnMut(&mut Synapse),
    ) -> Result<(), NetworkError> {
        let ids = self
            .outgoing
            .get(&neuron_id)
            .ok_or(NetworkError::MissingNeuron(neuron_id))?
            .clone();
        for id in ids {
            operation(
                self.synapses
                    .get_mut(&id)
                    .expect("adjacency ID always references a stored synapse"),
            );
        }
        Ok(())
    }

    /// Distance attenuation for one synapse's current endpoints.
    pub fn synaptic_attenuation(
        &self,
        synapse_id: SynapseId,
        decay_length: f32,
    ) -> Result<f32, NetworkError> {
        let synapse = self
            .synapse(synapse_id)
            .ok_or(NetworkError::MissingSynapse(synapse_id))?;
        let pre = self
            .neuron(synapse.pre())
            .ok_or(NetworkError::MissingPresynapticNeuron(synapse.pre()))?;
        let post = self
            .neuron(synapse.post())
            .ok_or(NetworkError::MissingPostsynapticNeuron(synapse.post()))?;
        try_distance_attenuation(pre.position().distance_to(post.position()), decay_length)
            .map_err(NetworkError::InvalidAttenuation)
    }

    /// Signed, distance-attenuated signal amplitude of a synapse, deriving the
    /// sign exclusively from the stored presynaptic neuron.
    pub fn synaptic_amplitude(
        &self,
        synapse_id: SynapseId,
        decay_length: f32,
    ) -> Result<SignalStrength, NetworkError> {
        let synapse = self
            .synapse(synapse_id)
            .ok_or(NetworkError::MissingSynapse(synapse_id))?;
        let pre = self
            .neuron(synapse.pre())
            .ok_or(NetworkError::MissingPresynapticNeuron(synapse.pre()))?;
        let attenuation = self.synaptic_attenuation(synapse_id, decay_length)?;
        synapse
            .effective_amplitude(pre.polarity(), attenuation)
            .map_err(NetworkError::InvalidSynapse)
    }

    /// Checks that all four stores agree. Useful after loading a snapshot or in
    /// debug assertions; ordinary mutation methods preserve this invariant.
    pub fn validate_indices(&self) -> Result<(), NetworkError> {
        for neuron_id in self.neurons.keys() {
            if !self.incoming.contains_key(neuron_id) || !self.outgoing.contains_key(neuron_id) {
                return Err(NetworkError::InconsistentIndices);
            }
        }
        if self.incoming.len() != self.neurons.len() || self.outgoing.len() != self.neurons.len() {
            return Err(NetworkError::InconsistentIndices);
        }

        let mut expected_incoming: BTreeMap<NeuronId, Vec<SynapseId>> = self
            .neurons
            .keys()
            .copied()
            .map(|id| (id, Vec::new()))
            .collect();
        let mut expected_outgoing = expected_incoming.clone();
        for synapse in self.synapses.values() {
            synapse.validate().map_err(NetworkError::InvalidSynapse)?;
            let outgoing = expected_outgoing
                .get_mut(&synapse.pre())
                .ok_or(NetworkError::MissingPresynapticNeuron(synapse.pre()))?;
            insert_sorted(outgoing, synapse.id());
            let incoming = expected_incoming
                .get_mut(&synapse.post())
                .ok_or(NetworkError::MissingPostsynapticNeuron(synapse.post()))?;
            insert_sorted(incoming, synapse.id());
        }

        if self.incoming != expected_incoming || self.outgoing != expected_outgoing {
            return Err(NetworkError::InconsistentIndices);
        }
        Ok(())
    }

    /// Checks the observable graph state against its authoritative start
    /// configuration. Stochastic construction details such as seed and
    /// connection probability remain the builder's responsibility.
    pub fn validate_against_config(
        &self,
        network_config: &NetworkConfig,
        neuron_config: &NeuronConfig,
    ) -> Result<(), NetworkError> {
        let actual_excitatory = self
            .neurons()
            .filter(|neuron| neuron.polarity() == super::Polarity::Excitatory)
            .count();
        let actual_inhibitory = self.neuron_count() - actual_excitatory;
        if actual_excitatory != network_config.excitatory_neurons
            || actual_inhibitory != network_config.inhibitory_neurons
        {
            return Err(NetworkError::PopulationMismatch {
                expected_excitatory: network_config.excitatory_neurons,
                expected_inhibitory: network_config.inhibitory_neurons,
                actual_excitatory,
                actual_inhibitory,
            });
        }
        if let Some(neuron) = self
            .neurons()
            .find(|neuron| neuron.params() != neuron_config)
        {
            return Err(NetworkError::NeuronConfigMismatch(neuron.id()));
        }
        if let Some(synapse) = self.synapses().find(|synapse| {
            synapse.weight().get() < network_config.min_weight
                || synapse.weight().get() > network_config.max_weight
        }) {
            return Err(NetworkError::WeightOutOfConfiguredBounds {
                synapse_id: synapse.id(),
                weight: synapse.weight().get(),
                min: network_config.min_weight,
                max: network_config.max_weight,
            });
        }
        Ok(())
    }
}

fn insert_sorted(ids: &mut Vec<SynapseId>, id: SynapseId) {
    match ids.binary_search(&id) {
        Ok(_) => {}
        Err(index) => ids.insert(index, id),
    }
}

fn remove_sorted(ids: &mut Vec<SynapseId>, id: SynapseId) {
    if let Ok(index) = ids.binary_search(&id) {
        ids.remove(index);
    }
}

/// Invalid graph mutation or lookup required by a graph operation.
#[derive(Clone, Debug, PartialEq)]
pub enum NetworkError {
    /// A neuron ID is already present.
    DuplicateNeuron(NeuronId),
    /// A synapse ID is already present.
    DuplicateSynapse(SynapseId),
    /// A requested neuron does not exist.
    MissingNeuron(NeuronId),
    /// A requested synapse does not exist.
    MissingSynapse(SynapseId),
    /// A new synapse references an absent source.
    MissingPresynapticNeuron(NeuronId),
    /// A new synapse references an absent target.
    MissingPostsynapticNeuron(NeuronId),
    /// A synapse violates its local invariants.
    InvalidSynapse(SynapseError),
    /// A geometric distance or attenuation length is invalid.
    InvalidAttenuation(DecayError),
    /// An explicit neuron removal was attempted before disconnecting it.
    NeuronStillConnected {
        /// Neuron requested for removal.
        neuron_id: NeuronId,
        /// Number of incoming connections.
        incoming: usize,
        /// Number of outgoing connections.
        outgoing: usize,
    },
    /// Stored graph objects and adjacency lists do not agree.
    InconsistentIndices,
    /// Configured and actual excitatory/inhibitory population sizes differ.
    PopulationMismatch {
        /// Requested excitatory population.
        expected_excitatory: usize,
        /// Requested inhibitory population.
        expected_inhibitory: usize,
        /// Stored excitatory population.
        actual_excitatory: usize,
        /// Stored inhibitory population.
        actual_inhibitory: usize,
    },
    /// A cell was constructed with parameters different from the shared config.
    NeuronConfigMismatch(NeuronId),
    /// A stored magnitude lies outside configured hard network bounds.
    WeightOutOfConfiguredBounds {
        /// Connection with the invalid magnitude.
        synapse_id: SynapseId,
        /// Stored magnitude.
        weight: f32,
        /// Configured lower bound.
        min: f32,
        /// Configured upper bound.
        max: f32,
    },
}

impl fmt::Display for NetworkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNeuron(id) => write!(formatter, "duplicate neuron ID {id}"),
            Self::DuplicateSynapse(id) => write!(formatter, "duplicate synapse ID {id}"),
            Self::MissingNeuron(id) => write!(formatter, "neuron {id} does not exist"),
            Self::MissingSynapse(id) => write!(formatter, "synapse {id} does not exist"),
            Self::MissingPresynapticNeuron(id) => {
                write!(formatter, "presynaptic neuron {id} does not exist")
            }
            Self::MissingPostsynapticNeuron(id) => {
                write!(formatter, "postsynaptic neuron {id} does not exist")
            }
            Self::InvalidSynapse(error) => write!(formatter, "invalid synapse: {error}"),
            Self::InvalidAttenuation(error) => write!(formatter, "invalid attenuation: {error}"),
            Self::NeuronStillConnected {
                neuron_id,
                incoming,
                outgoing,
            } => write!(
                formatter,
                "neuron {neuron_id} still has {incoming} incoming and {outgoing} outgoing synapses"
            ),
            Self::InconsistentIndices => {
                formatter.write_str("network adjacency indices are inconsistent")
            }
            Self::PopulationMismatch {
                expected_excitatory,
                expected_inhibitory,
                actual_excitatory,
                actual_inhibitory,
            } => write!(
                formatter,
                "configured population E={expected_excitatory}, I={expected_inhibitory} differs from stored E={actual_excitatory}, I={actual_inhibitory}"
            ),
            Self::NeuronConfigMismatch(id) => {
                write!(
                    formatter,
                    "neuron {id} does not use the configured cell parameters"
                )
            }
            Self::WeightOutOfConfiguredBounds {
                synapse_id,
                weight,
                min,
                max,
            } => write!(
                formatter,
                "synapse {synapse_id} weight {weight} lies outside configured bounds [{min}, {max}]"
            ),
        }
    }
}

impl Error for NetworkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidSynapse(error) => Some(error),
            Self::InvalidAttenuation(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{config::NeuronConfig, math::Position3D, primitives::Weight};

    use super::*;
    use crate::core::{NeuronRole, Polarity, SimTime};

    fn neuron(id: u64, position: Position3D, polarity: Polarity) -> Neuron {
        Neuron::new(
            NeuronId(id),
            position,
            polarity,
            Some(NeuronRole::Processing),
            NeuronConfig::default(),
            SimTime::ZERO,
        )
        .expect("valid test neuron")
    }

    fn synapse(id: u64, pre: u64, post: u64) -> Synapse {
        Synapse::new(
            SynapseId(id),
            NeuronId(pre),
            NeuronId(post),
            Weight::new(0.5).unwrap(),
            10,
            true,
        )
        .expect("valid test synapse")
    }

    #[test]
    fn rejects_missing_endpoints_without_partial_index_mutation() {
        let mut network = Network::new();
        network
            .add_neuron(neuron(1, Position3D::ORIGIN, Polarity::Excitatory))
            .unwrap();

        assert_eq!(
            network.add_synapse(synapse(1, 1, 2)),
            Err(NetworkError::MissingPostsynapticNeuron(NeuronId(2)))
        );
        assert_eq!(network.synapse_count(), 0);
        assert!(network.outgoing_synapse_ids(NeuronId(1)).is_empty());
        assert_eq!(network.validate_indices(), Ok(()));
    }

    #[test]
    fn maintains_sorted_incoming_and_outgoing_indices() {
        let mut network = Network::new();
        for id in 1..=3 {
            network
                .add_neuron(neuron(id, Position3D::ORIGIN, Polarity::Excitatory))
                .unwrap();
        }

        network.add_synapse(synapse(20, 1, 3)).unwrap();
        network.add_synapse(synapse(10, 1, 2)).unwrap();
        network.add_synapse(synapse(15, 3, 2)).unwrap();

        assert_eq!(
            network.outgoing_synapse_ids(NeuronId(1)),
            &[SynapseId(10), SynapseId(20)]
        );
        assert_eq!(
            network.incoming_synapse_ids(NeuronId(2)),
            &[SynapseId(10), SynapseId(15)]
        );
        assert_eq!(network.validate_indices(), Ok(()));
    }

    #[test]
    fn removing_synapse_updates_both_indices() {
        let mut network = Network::new();
        network
            .add_neuron(neuron(1, Position3D::ORIGIN, Polarity::Excitatory))
            .unwrap();
        network
            .add_neuron(neuron(2, Position3D::ORIGIN, Polarity::Excitatory))
            .unwrap();
        network.add_synapse(synapse(7, 1, 2)).unwrap();

        network.remove_synapse(SynapseId(7)).unwrap();

        assert!(network.outgoing_synapse_ids(NeuronId(1)).is_empty());
        assert!(network.incoming_synapse_ids(NeuronId(2)).is_empty());
        assert_eq!(network.validate_indices(), Ok(()));
    }

    #[test]
    fn equal_positions_do_not_merge_distinct_neuron_identities() {
        let mut network = Network::new();
        network
            .add_neuron(neuron(1, Position3D::ORIGIN, Polarity::Excitatory))
            .unwrap();
        network
            .add_neuron(neuron(2, Position3D::ORIGIN, Polarity::Excitatory))
            .unwrap();

        assert_eq!(network.neuron_count(), 2);
        assert_eq!(
            network.neuron(NeuronId(1)).unwrap().position(),
            Position3D::ORIGIN
        );
        assert_eq!(
            network.neuron(NeuronId(2)).unwrap().position(),
            Position3D::ORIGIN
        );
    }

    #[test]
    fn signed_amplitude_comes_from_presynaptic_polarity_and_distance() {
        let mut network = Network::new();
        network
            .add_neuron(neuron(
                1,
                Position3D::new(0.0, 0.0, 0.0),
                Polarity::Inhibitory,
            ))
            .unwrap();
        network
            .add_neuron(neuron(
                2,
                Position3D::new(1.0, 0.0, 0.0),
                Polarity::Excitatory,
            ))
            .unwrap();
        network.add_synapse(synapse(1, 1, 2)).unwrap();

        let amplitude = network.synaptic_amplitude(SynapseId(1), 1.0).unwrap().get();
        let expected = -0.5 / std::f32::consts::E;
        assert!((amplitude - expected).abs() <= 1.0e-6);
        assert!(matches!(
            network.synaptic_attenuation(SynapseId(1), 0.0),
            Err(NetworkError::InvalidAttenuation(_))
        ));
    }

    #[test]
    fn mutable_incoming_visit_is_deterministic() {
        let mut network = Network::new();
        for id in 1..=3 {
            network
                .add_neuron(neuron(id, Position3D::ORIGIN, Polarity::Excitatory))
                .unwrap();
        }
        network.add_synapse(synapse(20, 1, 3)).unwrap();
        network.add_synapse(synapse(10, 2, 3)).unwrap();

        let mut visited = Vec::new();
        network
            .for_each_incoming_synapse_mut(NeuronId(3), |synapse| {
                visited.push(synapse.id());
                synapse
                    .set_weight(Weight::new(synapse.weight().get() + 0.1).unwrap())
                    .expect("positive finite weight");
            })
            .unwrap();

        assert_eq!(visited, vec![SynapseId(10), SynapseId(20)]);
    }
}
