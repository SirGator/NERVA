//! Original deterministic CPU arithmetic behind the compute contract.

use crate::core::{NeuronId, SimTime, Spike};

use super::{ComputeBackend, ComputeError, DenseNeuronBatch, FlatNeuronInputs};

/// Stateless reference backend; all authoritative state stays in the network.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuCompute;

impl ComputeBackend for CpuCompute {
    fn advance_neurons(
        &mut self,
        neurons: &mut DenseNeuronBatch,
        time: SimTime,
    ) -> Result<(), ComputeError> {
        let (_, params, states) = neurons.parts_mut();
        for (params, state) in params.iter().zip(states) {
            state.advance_to(params, time)?;
        }
        Ok(())
    }

    fn integrate_neurons(
        &mut self,
        neurons: &mut DenseNeuronBatch,
        inputs: &mut FlatNeuronInputs,
        time: SimTime,
    ) -> Result<Vec<Spike>, ComputeError> {
        if neurons.neuron_ids() != inputs.neuron_ids() {
            return Err(ComputeError::InvalidBatch {
                reason: "input IDs differ from dense state IDs",
            });
        }
        let mut spikes = Vec::new();
        let (ids, params, states) = neurons.parts_mut();
        for (row, ((&id, params), state)) in ids.iter().zip(params).zip(states).enumerate() {
            let summary = canonical_input_summary(
                id,
                inputs.row_mut(row).expect("input layout was validated"),
            )?;
            if state.integrate_input_with_components(
                params,
                time,
                summary.summed_input,
                summary.input_magnitude,
                summary.inhibitory_input_magnitude,
            )? {
                spikes.push(Spike::new(id, time));
            }
        }
        Ok(spikes)
    }
}

pub(super) struct InputSummary {
    pub(super) summed_input: f32,
    pub(super) input_magnitude: f32,
    pub(super) inhibitory_input_magnitude: f32,
}

pub(super) fn canonical_input_summary(
    target: NeuronId,
    amplitudes: &mut [f32],
) -> Result<InputSummary, ComputeError> {
    for &amplitude in amplitudes.iter() {
        if !amplitude.is_finite() {
            return Err(ComputeError::NonFiniteInput { target, amplitude });
        }
    }
    amplitudes.sort_by(|left, right| {
        right
            .abs()
            .total_cmp(&left.abs())
            .then_with(|| left.total_cmp(right))
    });
    let input_magnitude = amplitudes
        .iter()
        .fold(0.0_f64, |sum, amplitude| sum + f64::from(amplitude.abs()));
    if !input_magnitude.is_finite() {
        return Err(ComputeError::NonFiniteInputMagnitude { target });
    }
    let input_magnitude = input_magnitude.min(f64::from(f32::MAX)) as f32;
    let inhibitory_input_magnitude = amplitudes
        .iter()
        .filter(|amplitude| **amplitude < 0.0)
        .fold(0.0_f64, |sum, amplitude| sum + f64::from(-*amplitude));
    if !inhibitory_input_magnitude.is_finite() {
        return Err(ComputeError::NonFiniteInputMagnitude { target });
    }
    let inhibitory_input_magnitude = inhibitory_input_magnitude.min(f64::from(f32::MAX)) as f32;
    let sum = amplitudes
        .iter()
        .fold(0.0_f64, |sum, amplitude| sum + f64::from(*amplitude)) as f32;
    if sum.is_finite() {
        Ok(InputSummary {
            summed_input: sum,
            input_magnitude,
            inhibitory_input_magnitude,
        })
    } else {
        Err(ComputeError::NonFiniteSummedInput { target })
    }
}
