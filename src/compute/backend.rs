//! Narrow numerical operations used between runtime phases.

use std::{error::Error, fmt};

use crate::core::{NeuronError, NeuronId, SimTime, Spike};

use super::{DenseNeuronBatch, FlatNeuronInputs};

/// Synchronous numerical phases over contiguous exported records.
///
/// Batches contain unique, ascending stable IDs and complete numerical state,
/// with immutable flattened parameters. Backends never receive Network,
/// Neuron, event queues, learning rules, I/O or topology mutation methods.
/// Results and errors must preserve row order. The runtime publishes state
/// before pre-arrival learning (advance) and post-spike learning (integration).
/// A fresh export for each call observes live state changes without an
/// authoritative backend cache. On failure partial records are published
/// before runtime poisoning, preserving the original CPU error semantics.
///
/// CpuCompute preserves the original CPU arithmetic exactly. Future backends
/// must declare numerical tolerances and replay guarantees explicitly.
pub trait ComputeBackend {
    /// Analytically advances all exported cells to the shared timestamp.
    fn advance_neurons(
        &mut self,
        neurons: &mut DenseNeuronBatch,
        time: SimTime,
    ) -> Result<(), ComputeError>;

    /// Canonically aggregates each flat input row and checks that cell once.
    /// Input IDs must match state IDs exactly, including empty autonomous rows.
    /// Maintenance-only cells are excluded from this phase.
    fn integrate_neurons(
        &mut self,
        neurons: &mut DenseNeuronBatch,
        inputs: &mut FlatNeuronInputs,
        time: SimTime,
    ) -> Result<Vec<Spike>, ComputeError>;
}

/// Numerical failures without knowledge of runtime or scheduler types.
#[derive(Clone, Debug, PartialEq)]
pub enum ComputeError {
    /// Optional accelerator initialization, transfer or kernel failure.
    Backend {
        operation: &'static str,
        detail: String,
    },
    /// Dense state or input layout violated its contract.
    InvalidBatch { reason: &'static str },
    /// A requested cell is absent.
    UnknownNeuron(NeuronId),
    /// Cell dynamics rejected the operation.
    Neuron(NeuronError),
    /// One amplitude was not finite.
    NonFiniteInput { target: NeuronId, amplitude: f32 },
    /// Signed accumulation exceeded the scalar domain.
    NonFiniteSummedInput { target: NeuronId },
    /// Absolute accumulation exceeded the numerical domain.
    NonFiniteInputMagnitude { target: NeuronId },
}

impl From<NeuronError> for ComputeError {
    fn from(error: NeuronError) -> Self {
        Self::Neuron(error)
    }
}

impl fmt::Display for ComputeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend { operation, detail } => {
                write!(formatter, "compute backend {operation}: {detail}")
            }
            Self::InvalidBatch { reason } => write!(formatter, "invalid compute batch: {reason}"),
            Self::UnknownNeuron(id) => write!(formatter, "compute references unknown neuron {id}"),
            Self::Neuron(error) => write!(formatter, "neuron compute error: {error}"),
            Self::NonFiniteInput { target, amplitude } => write!(
                formatter,
                "input for neuron {target} must be finite, got {amplitude}"
            ),
            Self::NonFiniteSummedInput { target } => write!(
                formatter,
                "simultaneous input sum for neuron {target} became non-finite"
            ),
            Self::NonFiniteInputMagnitude { target } => write!(
                formatter,
                "simultaneous input magnitude for neuron {target} became non-finite"
            ),
        }
    }
}

impl Error for ComputeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Neuron(error) => Some(error),
            _ => None,
        }
    }
}
