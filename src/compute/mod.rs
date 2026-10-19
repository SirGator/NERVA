//! Numerical neuron work over dense records, independent of event orchestration.
//!
//! Network and Neuron remain authoritative. The runtime exports only the
//! affected cells into contiguous state/parameter arrays, invokes compute,
//! then publishes results before the next learning phase. Numerical records
//! and shared analytical kernels live in core, which has no backend dependency.
//! Inputs use a flat amplitude array with explicit per-cell offsets.
//! The optional CUDA prototype translates these records into its own device
//! arrays; no dense synapse graph or persistent device state is provided.

/// Backend contract and numerical errors.
pub mod backend;
/// Deterministic reference CPU implementation.
pub mod cpu;
/// Optional synchronous CUDA prototype (fresh device buffers per call).
#[cfg(feature = "cuda")]
pub mod cuda;
/// Dense numerical cell exports and checked state publication.
pub mod dense;
/// Flat input batches with explicit row offsets.
pub mod inputs;

pub use backend::{ComputeBackend, ComputeError};
pub use cpu::CpuCompute;
#[cfg(feature = "cuda")]
pub use cuda::CudaCompute;
pub use dense::{ComputeNeuronParams, ComputeNeuronState, DenseNeuronBatch};
pub use inputs::FlatNeuronInputs;
