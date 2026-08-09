use crate::common::{NeuronId, SimTimeUs};

use super::error::NeuronConfigError;
use super::{
    CellType, NeuronAddress, NeuronModulatorSensitivity, NeuronParams, NeuronRole, NeuronState,
    PrimaryTransmitter,
};

/// Validated input used to construct a [`Neuron`].
#[derive(Clone, Debug)]
pub struct NeuronSpec {
    /// Stable identifier of the new neuron.
    pub id: NeuronId,
    /// Location of the neuron in the network hierarchy.
    pub address: NeuronAddress,

    /// Functional role of the neuron.
    pub role: NeuronRole,
    /// Cellular class constraining transmitter compatibility.
    pub cell_type: CellType,
    /// Primary chemical or functional signal emitted by the neuron.
    pub transmitter: PrimaryTransmitter,

    /// Static electrical parameters.
    pub params: NeuronParams,
    /// Sensitivity to local neuromodulators.
    pub modulator_sensitivity: NeuronModulatorSensitivity,
}

/// A single neuron with immutable specification and mutable simulation state.
#[derive(Clone, Debug)]
pub struct Neuron {
    /// Stable identifier copied from the validated specification.
    pub(crate) id: NeuronId,
    /// Hierarchical location copied from the validated specification.
    pub(crate) address: NeuronAddress,

    /// Functional role copied from the validated specification.
    pub(crate) role: NeuronRole,
    /// Cell class copied from the validated specification.
    pub(crate) cell_type: CellType,
    /// Primary transmitter copied from the validated specification.
    pub(crate) transmitter: PrimaryTransmitter,

    /// Fixed electrical parameters copied from the validated specification.
    pub(crate) params: NeuronParams,
    /// State mutated by time evolution and input integration.
    pub(crate) state: NeuronState,

    /// Fixed neuromodulator sensitivity copied from the validated specification.
    pub(crate) modulator_sensitivity: NeuronModulatorSensitivity,
}

impl NeuronSpec {
    /// Checks parameter, cell-type, transmitter, and role invariants.
    pub fn validate(&self) -> Result<(), NeuronConfigError> {
        self.params.validate()?;

        if !is_valid_cell_transmitter(self.cell_type, self.transmitter) {
            return Err(NeuronConfigError::InvalidCellTransmitter {
                cell_type: self.cell_type,
                transmitter: self.transmitter,
            });
        }

        if self.role == NeuronRole::Modulatory && !self.transmitter.is_modulatory() {
            return Err(NeuronConfigError::ModulatoryRoleRequiresModulator);
        }

        Ok(())
    }
}

impl Neuron {
    /// Validates `spec` and constructs a neuron in its resting state at `start_time`.
    pub fn new(spec: NeuronSpec, start_time: SimTimeUs) -> Result<Self, NeuronConfigError> {
        spec.validate()?;

        let state = NeuronState::new(start_time, spec.params.resting_potential);

        Ok(Self {
            id: spec.id,
            address: spec.address,
            role: spec.role,
            cell_type: spec.cell_type,
            transmitter: spec.transmitter,
            params: spec.params,
            state,
            modulator_sensitivity: spec.modulator_sensitivity,
        })
    }

    /// Returns the neuron's stable identifier.
    pub fn id(&self) -> NeuronId {
        self.id
    }

    /// Returns the current mutable simulation state.
    pub fn state(&self) -> &NeuronState {
        &self.state
    }

    /// Returns the neuron's hierarchical network address.
    pub fn address(&self) -> NeuronAddress {
        self.address
    }

    /// Returns the neuron's functional role.
    pub fn role(&self) -> NeuronRole {
        self.role
    }

    /// Returns the neuron's cell class.
    pub fn cell_type(&self) -> CellType {
        self.cell_type
    }

    /// Returns the neuron's primary transmitter.
    pub fn transmitter(&self) -> PrimaryTransmitter {
        self.transmitter
    }

    /// Returns the neuron's fixed electrical parameters.
    pub fn params(&self) -> &NeuronParams {
        &self.params
    }

    /// Returns the neuron's configured neuromodulator sensitivity.
    pub fn modulator_sensitivity(&self) -> NeuronModulatorSensitivity {
        self.modulator_sensitivity
    }
}

/// Returns whether `transmitter` is valid for `cell_type`.
fn is_valid_cell_transmitter(cell_type: CellType, transmitter: PrimaryTransmitter) -> bool {
    match cell_type {
        CellType::Generic => true,
        CellType::Pyramidal => transmitter == PrimaryTransmitter::Glutamate,
        CellType::FastSpikingInterneuron | CellType::DendriteTargetingInterneuron => {
            transmitter == PrimaryTransmitter::Gaba
        }
        CellType::ModulatoryProjection => transmitter.is_modulatory(),
    }
}
