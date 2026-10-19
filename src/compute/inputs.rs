//! Flat simultaneous amplitudes with explicit per-cell row offsets.

use crate::core::NeuronId;

use super::{ComputeError, dense::validate_ids};

/// GPU-friendly input storage without nested vectors or pointer-bearing rows.
///
/// Row `i` is `amplitudes[offsets[i]..offsets[i + 1]]`. An empty row requests
/// an autonomous firing check. `u64` offsets are explicit, not host-sized;
/// a future device implementation may narrow them with checked conversion.
/// Canonical summation remains a per-row backend operation so numeric errors
/// still occur in the original cell order after earlier rows were integrated.
#[derive(Clone, Debug, PartialEq)]
pub struct FlatNeuronInputs {
    neuron_ids: Vec<NeuronId>,
    offsets: Vec<u64>,
    amplitudes: Vec<f32>,
}

impl FlatNeuronInputs {
    /// Validates IDs and row layout. Amplitude validation and aggregation occur
    /// in cell order during computation, preserving partial-failure semantics.
    pub fn new(
        neuron_ids: Vec<NeuronId>,
        offsets: Vec<u64>,
        amplitudes: Vec<f32>,
    ) -> Result<Self, ComputeError> {
        validate_ids(&neuron_ids)?;
        let length = u64::try_from(amplitudes.len()).map_err(|_| ComputeError::InvalidBatch {
            reason: "amplitude count exceeds u64",
        })?;
        if offsets.len() != neuron_ids.len().saturating_add(1)
            || offsets.first() != Some(&0)
            || offsets.last() != Some(&length)
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(ComputeError::InvalidBatch {
                reason: "invalid input row offsets",
            });
        }
        Ok(Self {
            neuron_ids,
            offsets,
            amplitudes,
        })
    }

    /// Flattens host input rows; no nested vectors cross the backend boundary.
    pub fn from_rows(
        rows: impl IntoIterator<Item = (NeuronId, Vec<f32>)>,
    ) -> Result<Self, ComputeError> {
        let mut ids = Vec::new();
        let mut offsets = vec![0];
        let mut amplitudes = Vec::new();
        for (id, row) in rows {
            ids.push(id);
            amplitudes.extend(row);
            offsets.push(u64::try_from(amplitudes.len()).map_err(|_| {
                ComputeError::InvalidBatch {
                    reason: "amplitude count exceeds u64",
                }
            })?);
        }
        Self::new(ids, offsets, amplitudes)
    }

    /// Stable IDs, in the same order as the integration state batch.
    pub fn neuron_ids(&self) -> &[NeuronId] {
        &self.neuron_ids
    }
    /// Validated row boundaries including the final sentinel.
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }
    /// All amplitudes in one contiguous array.
    pub fn amplitudes(&self) -> &[f32] {
        &self.amplitudes
    }
    /// One row, or None when the row index is outside this batch.
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        let end = *self.offsets.get(index.checked_add(1)?)? as usize;
        let start = *self.offsets.get(index)? as usize;
        self.amplitudes.get(start..end)
    }
    /// Mutable amplitudes of one row, for canonical ordering and aggregation.
    pub fn row_mut(&mut self, index: usize) -> Option<&mut [f32]> {
        let end = *self.offsets.get(index.checked_add(1)?)? as usize;
        let start = *self.offsets.get(index)? as usize;
        self.amplitudes.get_mut(start..end)
    }
}
