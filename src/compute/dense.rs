//! Dense numerical cell batches exported from the authoritative sparse graph.

use crate::core::{Network, NeuronId};

use super::ComputeError;

pub use crate::core::{
    NeuronDynamicsParams as ComputeNeuronParams, NeuronDynamicsState as ComputeNeuronState,
};

/// Contiguous records for a unique, ascending set of stable neuron identities.
///
/// Row `i` in all three arrays describes the same neuron. Backends can mutate
/// only state records; identities and immutable parameters are read-only.
/// Each batch is freshly exported, never a second authoritative network.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseNeuronBatch {
    neuron_ids: Vec<NeuronId>,
    params: Vec<ComputeNeuronParams>,
    states: Vec<ComputeNeuronState>,
    // Host-only source snapshot detects stale exports before publication.
    source_states: Vec<ComputeNeuronState>,
}

impl DenseNeuronBatch {
    /// Exports requested cells in dense order without modifying the network.
    pub fn from_network(network: &Network, neuron_ids: &[NeuronId]) -> Result<Self, ComputeError> {
        validate_ids(neuron_ids)?;
        let mut params = Vec::with_capacity(neuron_ids.len());
        let mut states = Vec::with_capacity(neuron_ids.len());
        for &id in neuron_ids {
            let neuron = network.neuron(id).ok_or(ComputeError::UnknownNeuron(id))?;
            params.push(neuron.numerical_params());
            states.push(neuron.numerical_state());
        }
        let source_states = states.clone();
        Ok(Self {
            neuron_ids: neuron_ids.to_vec(),
            params,
            states,
            source_states,
        })
    }

    /// Stable IDs corresponding to each dense row.
    pub fn neuron_ids(&self) -> &[NeuronId] {
        &self.neuron_ids
    }
    /// Immutable flattened electrical parameters.
    pub fn params(&self) -> &[ComputeNeuronParams] {
        &self.params
    }
    /// Current numerical state records.
    pub fn states(&self) -> &[ComputeNeuronState] {
        &self.states
    }
    /// Mutable numerical records; identities and parameters remain fixed.
    pub fn states_mut(&mut self) -> &mut [ComputeNeuronState] {
        &mut self.states
    }
    /// Disjoint dense arrays for backend kernels without graph access.
    pub fn parts_mut(
        &mut self,
    ) -> (
        &[NeuronId],
        &[ComputeNeuronParams],
        &mut [ComputeNeuronState],
    ) {
        (&self.neuron_ids, &self.params, &mut self.states)
    }

    /// Validates the entire export before publishing any cell state.
    ///
    /// The graph must still have the same cells, parameters and source state.
    /// Use a fresh batch after any intervening state or topology mutation.
    pub fn write_back(&self, network: &mut Network) -> Result<(), ComputeError> {
        self.validate_destination(network)?;
        for (&id, state) in self.neuron_ids.iter().zip(&self.states) {
            network
                .neuron(id)
                .ok_or(ComputeError::UnknownNeuron(id))?
                .validate_numerical_state(state)?;
        }
        self.publish(network);
        Ok(())
    }

    fn validate_destination(&self, network: &Network) -> Result<(), ComputeError> {
        for ((&id, params), source) in self
            .neuron_ids
            .iter()
            .zip(&self.params)
            .zip(&self.source_states)
        {
            let neuron = network.neuron(id).ok_or(ComputeError::UnknownNeuron(id))?;
            if !neuron.numerical_params().same_bits(params) {
                return Err(ComputeError::InvalidBatch {
                    reason: "cell parameters changed after export",
                });
            }
            if !neuron.numerical_state().same_bits(source) {
                return Err(ComputeError::InvalidBatch {
                    reason: "cell state changed after export",
                });
            }
        }
        Ok(())
    }

    fn publish(&self, network: &mut Network) {
        for (&id, &state) in self.neuron_ids.iter().zip(&self.states) {
            network
                .neuron_mut(id)
                .expect("destination was validated")
                .set_numerical_state_unchecked(state);
        }
    }

    /// Failed numerical transitions may already have updated a prefix and the
    /// failing row. Publish that partial state before runtime poisoning, as in
    /// the original in-place CPU path; no successful continuation is allowed.
    pub(crate) fn complete<T>(
        self,
        network: &mut Network,
        result: Result<T, ComputeError>,
    ) -> Result<T, ComputeError> {
        match result {
            Ok(value) => {
                self.write_back(network)?;
                Ok(value)
            }
            Err(error) => {
                self.validate_destination(network)?;
                self.publish(network);
                Err(error)
            }
        }
    }
}

pub(super) fn validate_ids(ids: &[NeuronId]) -> Result<(), ComputeError> {
    if ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(ComputeError::InvalidBatch {
            reason: "neuron IDs must be unique and ascending",
        });
    }
    Ok(())
}
