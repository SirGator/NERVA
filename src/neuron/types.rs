use crate::common::{AreaId, LayerId, PopulationId};

/// Hierarchical location of a neuron inside the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NeuronAddress {
    /// Containing brain area.
    pub area: AreaId,
    /// Optional layer within the area.
    pub layer: Option<LayerId>,
    /// Containing population.
    pub population: PopulationId,
    /// Index local to the population.
    pub local_index: u32,
}

/// Functional role assigned to a neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NeuronRole {
    /// Encodes external sensory input.
    SensoryInput,
    /// Performs local recurrent computation.
    LocalProcessing,
    /// Stores or supplies contextual information.
    Context,
    /// Represents an expected future state.
    Prediction,
    /// Represents the difference between an observation and a prediction.
    PredictionError,
    /// Contributes to action selection or motor output.
    Action,
    /// Produces a neuromodulatory signal.
    Modulatory,
}

/// Biological or computational class of a neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CellType {
    /// Unconstrained general-purpose neuron.
    Generic,
    /// Excitatory pyramidal projection neuron.
    Pyramidal,
    /// Fast inhibitory interneuron.
    FastSpikingInterneuron,
    /// Inhibitory interneuron targeting dendrites.
    DendriteTargetingInterneuron,
    /// Neuron projecting a neuromodulatory signal.
    ModulatoryProjection,
}

/// Primary neurotransmitter or signaling substance of a neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrimaryTransmitter {
    /// Primary excitatory transmitter.
    Glutamate,
    /// Primary inhibitory transmitter.
    Gaba,
    /// Dopaminergic modulator.
    Dopamine,
    /// Cholinergic modulator.
    Acetylcholine,
    /// Noradrenergic modulator.
    Noradrenaline,
    /// Serotonergic modulator.
    Serotonin,
    /// No primary transmitter is assigned.
    None,
}

impl PrimaryTransmitter {
    /// Returns whether this transmitter is a neuromodulator.
    pub fn is_modulatory(self) -> bool {
        matches!(
            self,
            Self::Dopamine | Self::Acetylcholine | Self::Noradrenaline | Self::Serotonin
        )
    }
}
