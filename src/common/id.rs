/// Unique identifier of a neuron within a network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NeuronId(pub u64);

/// Unique identifier of a synapse within a network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SynapseId(pub u64);

/// Unique identifier of a brain area.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AreaId(pub u32);

/// Identifier of a layer inside a brain area.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(pub u16);

/// Identifier of a neuron population inside an area or layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PopulationId(pub u32);
