//! Synchronous CUDA proof of concept, with fresh SoA device buffers per call.
//!
//! Canonical input reduction stays on the host. CUDA computes analytical cell
//! transitions; CPU scheduling, learning, propagation and prediction stay intact.
//! Numerical failures replay the original CPU error path to preserve prefix
//! mutation and poisoning semantics. Driver/transfer failures never silently
//! fall back. This prototype makes no speed or bit-identical replay promise.

use std::sync::Arc;

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use super::{
    ComputeBackend, ComputeError, ComputeNeuronState, CpuCompute, DenseNeuronBatch,
    FlatNeuronInputs, cpu::canonical_input_summary,
};
use crate::core::{SimTime, Spike};

/// Tested comparison envelope; discrete state and spike decisions are checked exactly.
pub const ABS_TOLERANCE: f32 = 2.0e-5;
pub const REL_TOLERANCE: f32 = 2.0e-6;

/// Optional accelerator holding only context, compiled kernel and stream.
/// There is no persistent device copy of neural state.
pub struct CudaCompute {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    device_name: String,
}

fn failure(operation: &'static str, error: impl std::fmt::Display) -> ComputeError {
    ComputeError::Backend {
        operation,
        detail: error.to_string(),
    }
}

impl CudaCompute {
    /// Loads the NVIDIA driver and NVRTC dynamically and compiles for the device.
    /// Missing libraries or inaccessible hardware return an error.
    pub fn new(ordinal: usize) -> Result<Self, ComputeError> {
        // SAFETY: cudarc probes trusted native CUDA library names; no pointers
        // or borrowed memory are passed to these library-presence checks.
        if !unsafe {
            cudarc::driver::sys::is_culib_present() && cudarc::nvrtc::sys::is_culib_present()
        } {
            return Err(failure(
                "initialization",
                "NVIDIA driver and NVRTC shared libraries are required",
            ));
        }
        let context = CudaContext::new(ordinal).map_err(|e| failure("context", e))?;
        let device_name = context.name().map_err(|e| failure("device name", e))?;
        let (major, minor) = context
            .compute_capability()
            .map_err(|e| failure("device architecture", e))?;
        let ptx = compile_ptx_with_opts(
            include_str!("cuda_kernel.cu"),
            CompileOptions {
                fmad: Some(false),
                ftz: Some(false),
                use_fast_math: Some(false),
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            },
        )
        .map_err(|e| failure("NVRTC compilation", e))?;
        let module = context
            .load_module(ptx)
            .map_err(|e| failure("module load", e))?;
        let kernel = module
            .load_function("nerva_compute")
            .map_err(|e| failure("kernel load", e))?;
        Ok(Self {
            stream: context.default_stream(),
            kernel,
            device_name,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    fn execute(
        &self,
        neurons: &DenseNeuronBatch,
        time: SimTime,
        mode: u32,
        inputs: &[f32],
    ) -> Result<(Vec<ComputeNeuronState>, Vec<u32>), ComputeError> {
        let n = neurons.states().len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let count = u32::try_from(n)
            .ok()
            .filter(|n| n.checked_mul(14).is_some())
            .ok_or(ComputeError::InvalidBatch {
                reason: "CUDA batch exceeds device indexing range",
            })?;
        let mut sf = vec![0.0_f32; n * 10];
        let mut st = vec![0_u64; n * 6];
        let mut pf = vec![0.0_f32; n * 14];
        let mut pt = Vec::with_capacity(n);
        for (row, (s, p)) in neurons.states().iter().zip(neurons.params()).enumerate() {
            for (column, value) in [
                s.membrane_potential,
                s.activity_trace,
                s.input_trace,
                s.threshold,
                s.burst_drive,
                s.adaptation_drive,
                s.rebound_drive,
                s.threshold_adaptation,
                s.intrinsic_current,
                s.structural_drive,
            ]
            .into_iter()
            .enumerate()
            {
                sf[column * n + row] = value;
            }
            for (column, value) in [
                s.last_update.as_micros(),
                s.refractory_until.as_micros(),
                s.last_spike_time,
                u64::from(s.last_spike_valid),
                s.spike_count,
                s.last_homeostasis_update.as_micros(),
            ]
            .into_iter()
            .enumerate()
            {
                st[column * n + row] = value;
            }
            for (column, value) in [
                p.resting_potential,
                p.reset_potential,
                p.threshold,
                p.membrane_tau_us,
                p.activity_trace_tau_us,
                p.intrinsic_drive,
                p.burst_gain,
                p.burst_tau_us,
                p.adaptation_gain,
                p.adaptation_tau_us,
                p.threshold_adaptation_gain,
                p.threshold_adaptation_tau_us,
                p.rebound_gain,
                p.rebound_tau_us,
            ]
            .into_iter()
            .enumerate()
            {
                pf[column * n + row] = value;
            }
            pt.push(p.refractory_period_us);
        }
        let mut dsf = self
            .stream
            .clone_htod(&sf)
            .map_err(|e| failure("state upload", e))?;
        let mut dst = self
            .stream
            .clone_htod(&st)
            .map_err(|e| failure("time upload", e))?;
        let dpf = self
            .stream
            .clone_htod(&pf)
            .map_err(|e| failure("parameter upload", e))?;
        let dpt = self
            .stream
            .clone_htod(&pt)
            .map_err(|e| failure("duration upload", e))?;
        let di = self
            .stream
            .clone_htod(inputs)
            .map_err(|e| failure("input upload", e))?;
        let mut flags = self
            .stream
            .alloc_zeros::<u32>(n)
            .map_err(|e| failure("flag allocation", e))?;
        let at = time.as_micros();
        let mut launch = self.stream.launch_builder(&self.kernel);
        launch
            .arg(&mut dsf)
            .arg(&mut dst)
            .arg(&dpf)
            .arg(&dpt)
            .arg(&di)
            .arg(&mut flags)
            .arg(&count)
            .arg(&at)
            .arg(&mode);
        // SAFETY: kernel signature matches these primitive pointers/scalars.
        // Each SoA buffer has its documented column count times n elements;
        // inputs has 3*n entries. The kernel bounds-checks every row. cudarc
        // keeps allocations alive and tracks stream read/write dependencies.
        // Analytical intrinsic dynamics use many registers; a small block also
        // works for sparse batches and avoids oversized default launch blocks.
        let config = LaunchConfig {
            grid_dim: (count.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }.map_err(|e| failure("kernel launch", e))?;
        let sf = self
            .stream
            .clone_dtoh(&dsf)
            .map_err(|e| failure("state download", e))?;
        let st = self
            .stream
            .clone_dtoh(&dst)
            .map_err(|e| failure("time download", e))?;
        let flags = self
            .stream
            .clone_dtoh(&flags)
            .map_err(|e| failure("flag download", e))?;
        self.stream
            .synchronize()
            .map_err(|e| failure("synchronize", e))?;
        let mut states = neurons.states().to_vec();
        for (row, s) in states.iter_mut().enumerate() {
            s.membrane_potential = sf[row];
            s.activity_trace = sf[n + row];
            s.input_trace = sf[2 * n + row];
            s.threshold = sf[3 * n + row];
            s.burst_drive = sf[4 * n + row];
            s.adaptation_drive = sf[5 * n + row];
            s.rebound_drive = sf[6 * n + row];
            s.threshold_adaptation = sf[7 * n + row];
            s.intrinsic_current = sf[8 * n + row];
            s.structural_drive = sf[9 * n + row];
            s.last_update = SimTime(st[row]);
            s.refractory_until = SimTime(st[n + row]);
            s.last_spike_time = st[2 * n + row];
            s.last_spike_valid = st[3 * n + row] as u32;
            s.spike_count = st[4 * n + row];
            s.last_homeostasis_update = SimTime(st[5 * n + row]);
        }
        Ok((states, flags))
    }
}

impl ComputeBackend for CudaCompute {
    fn advance_neurons(
        &mut self,
        neurons: &mut DenseNeuronBatch,
        time: SimTime,
    ) -> Result<(), ComputeError> {
        let inputs = vec![0.0; neurons.states().len() * 3];
        let (states, flags) = self.execute(neurons, time, 0, &inputs)?;
        if flags.contains(&2) {
            return match CpuCompute.advance_neurons(neurons, time) {
                Err(error) => Err(error),
                Ok(()) => Err(failure(
                    "advance",
                    "GPU numerical failure not reproduced by CPU",
                )),
            };
        }
        neurons.states_mut().copy_from_slice(&states);
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
        let original_inputs = inputs.clone();
        let n = neurons.states().len();
        let mut summaries = vec![0.0; n * 3];
        for row in 0..n {
            let summary = canonical_input_summary(
                neurons.neuron_ids()[row],
                inputs.row_mut(row).expect("validated layout"),
            );
            let Ok(summary) = summary else {
                *inputs = original_inputs;
                return CpuCompute.integrate_neurons(neurons, inputs, time);
            };
            summaries[row] = summary.summed_input;
            summaries[n + row] = summary.input_magnitude;
            summaries[2 * n + row] = summary.inhibitory_input_magnitude;
        }
        let (states, flags) = self.execute(neurons, time, 1, &summaries)?;
        if flags.contains(&2) {
            *inputs = original_inputs;
            return match CpuCompute.integrate_neurons(neurons, inputs, time) {
                Err(error) => Err(error),
                Ok(_) => Err(failure(
                    "integrate",
                    "GPU numerical failure not reproduced by CPU",
                )),
            };
        }
        neurons.states_mut().copy_from_slice(&states);
        Ok(neurons
            .neuron_ids()
            .iter()
            .zip(flags)
            .filter_map(|(&id, flag)| (flag == 1).then_some(Spike::new(id, time)))
            .collect())
    }
}
