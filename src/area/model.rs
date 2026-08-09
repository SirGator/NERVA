use crate::common::{AreaId, NeuronId};

use super::ModulatorLevels;

/// A named brain region containing neurons and local modulator levels.
pub struct BrainArea {
    /// Stable identifier of this area.
    pub id: AreaId,
    /// Human-readable label for diagnostics and configuration.
    pub name: String,

    /// Current local neuromodulator concentrations.
    pub modulators: ModulatorLevels,
    /// Neurons assigned to this area.
    pub neuron_ids: Vec<NeuronId>,
}
