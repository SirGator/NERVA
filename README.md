# NERVA

**NERVA — Neural Emergent Reactive & Versatile Architectur**

NERVA is an embeddable Rust library for deterministic, event-driven spatial
spiking networks with local Pair-STDP. The host application constructs the
network, supplies timestamped inputs, controls execution, and reads the results.
The library models local neuronal and synaptic dynamics without a global
simulation tick, backpropagation, global loss, or reward signal.

## Use as a library

Add a local dependency to your application's `Cargo.toml`, adjusting `path` to
the directory containing NERVA's `Cargo.toml`:

```toml
[dependencies]
nerva = { path = "../nerva" }
```

Build a `core::Network` from neurons and synapses, then give it to
`runtime::Simulation` with a learning rule such as `learning::NoPlasticity` or
`learning::PairStdp`. Supply input with `schedule_external_input` and advance
the simulation with `run_until`. Use `run` for finite event streams; autonomous
firing and recurring homeostasis require an explicit time horizon.

The [minimal network example](examples/minimal_network.rs) demonstrates two
connected neurons, delayed spike propagation, and reading the observation log:

```bash
cargo run --example minimal_network
cargo doc --no-deps
```

Optional Cargo features expose additional library modules:

- `diagnostics`: read-only event inspection, snapshots, and metrics
- `visualization`: CSV and JSON exports
- `development`: local structural-drive, candidate search, and M1.3 synapse formation
- `cuda`: experimental CUDA neuron backend using the optional `cudarc` dependency

The default build uses no external crate dependencies. The public API is in
early development and may change between releases.

## Public slices

- `primitives`, `math`, and `config`: policy-free foundations and validation
- `core`: neurons, synapses, spikes, and deterministic graph state
- `learning`: local Pair-STDP and per-neuron homeostasis
- `runtime`: deterministic timestamp batches and spike propagation
- `compute`: synchronous numerical backend boundary with deterministic `CpuCompute`
- `io`, `roots`, `transduction`, and `nerves`: device-neutral external connections
- `experiment`, including `experiment::m0`: reproducible reference orchestration
- feature-gated `metrics`, `debug`, `visualization`, and post-M0 `development`

Every neuron can additionally carry a continuous intrinsic capability vector:
constant drive, burst, adaptation, threshold adaptation, and rebound. The
neutral defaults preserve the original LIF behavior. Non-zero capabilities are
integrated analytically between events and remain local neuron state; they do
not create discrete neuron classes or a global simulation tick.

Sensory and motor neuron populations live inside the same `core::Network`, so
their synapses can use the same local learning rules as the internal network.
Only the physical root/transduction/nerve attachment remains a fixed boundary;
future structural formation and pruning belongs to `development`.

External adapters implement `io::Receptor` or `io::Effector` and exchange
timestamped values through `ChannelId`. Sensory input follows
`Receptor → SensoryTransducer → SensoryRoot → Fiber → sensory neuron`;
motor spikes follow `motor neuron → Fiber → MotorRoot → MotorTransducer → Effector`.
Only `nerves::Mapping` associates physical channels with neuron IDs.
Roots only declare which channels exist. `RootChannel` contains `ChannelId`;
`Mapping` owns `(RootId, ChannelId) → FiberId`, and each `Fiber` owns its neuron
endpoint, delay, and gain. A second association for the same root channel is
rejected, including motor associations from different neurons.
Bindings take `&Root`: `map_sensory` and `map_motor` validate channel existence
and root direction before changing the mapping. Undeclared channels return
`MappingError::UnknownChannel`; wrong root directions return
`MappingError::RootDirectionMismatch`. Root identities and channel layouts
remain fixed after binding. `RootRegistry` is optional.

Both transducer traits queue input with `push` and append output with
`advance_until(until, &mut output)`. A horizon includes its timestamp and closes
that interval: queue every input through `until` before advancing, and keep
later inputs for later horizons. Repeated horizons produce no duplicate output;
backwards horizons and inputs in completed intervals return `TransductionError`.
Advancing without new input also gives continuous encodings a time span for
rate generation or decay, without introducing a global tick.

`DirectSensoryTransducer` maps each finite, positive receptor value to one
equal-amplitude spike; `DirectMotorTransducer` maps each finite, positive motor
amplitude to one equal-value pulse. Both preserve channels and timestamps.

`RateSensoryTransducer` interprets each receptor value as a firing rate in Hertz
and emits unit-amplitude spikes while that value is held. Channels integrate
independently; rate changes preserve fractional phase and zero pauses it.
Accepted rates are finite values in `0..=1_000_000` Hz, at most one spike per
channel per microsecond. Predicted crossings are rounded up to whole
microseconds without accumulating rounded-period drift. Spike trains replay
exactly across different advance horizons. The adapter converts physical
values to rates before publishing them.

`PoissonSensoryTransducer::new(seed)` offers stochastic coding of the same
held Hertz values. Each channel has an independent seeded SplitMix64 stream
and exponentially distributed waiting intervals. Rate changes preserve the
remaining waiting hazard, zero pauses it, and identical inputs and seeds
reproduce identical spike times across advance horizons. Arrivals are rounded
up to microseconds; multiple spikes may share a timestamp. Accepted rates
are finite values in `0..=1_000_000` Hz and amplitudes are `1.0`.

M0-specific patterns, bits, encoders, and decoders live in `experiment::m0`.
The [I/O integration test](tests/io_contract.rs) exercises the entire path
through the public transducer traits and actual neural dynamics.
It covers direct values, regular rates, and seeded Poisson rates; rate changes, pauses, replay, and
time-domain boundaries are checked in [rate_transduction.rs](tests/rate_transduction.rs).
Poisson stream independence and waiting-time statistics are checked in
[poisson_transduction.rs](tests/poisson_transduction.rs).

The typed scalar primitives (`Potential`, `Threshold`, `Weight`,
`SignalStrength`, and related domains) support a gradual migration. The
synaptic weight path (`core::Synapse`, `learning::WeightBounds` and
`PairStdp`, `runtime::ObservationEvent::WeightChanged`, and the M0 experiment
weight records) is migrated end to end to `Weight`; membrane potential,
threshold, and input amplitude still use raw `f32` and will each be converted
as one complete path instead of mixing representations within one invariant.

Weights are non-negative magnitudes whose type validates finiteness and sign
at construction. Excitatory or inhibitory effect is always derived from the
presynaptic neuron's polarity when a signed signal amplitude is formed, so
learning cannot flip a connection's sign. Positions affect geometry,
attenuation, and delay; only `NeuronId` defines identity.

A live `Simulation` is only mutated through controlled methods
(`add_neuron`, `remove_neuron`, `add_synapse`, `remove_synapse`,
`update_neuron`, `update_synapse`) that keep
the scheduler, the per-neuron homeostasis clocks, and the intrinsic spike
predictions consistent. A fatal error inside a batch poisons the runtime:
partially applied state may remain, and every further use is rejected.

`Simulation` uses `compute::CpuCompute` by default; existing constructors keep
their behavior. Select a numerical backend with
`simulation.with_compute_backend(CpuCompute)` or an implementation of
`compute::ComputeBackend`. The backend analytically advances affected cells,
then aggregates and integrates each firing candidate once. Pre-arrival learning
hooks run between these two phases, so their ordering is unchanged. Scheduling,
topology, learning, utility, maintenance and propagation remain in `runtime`.

The sparse `Network` remains the authoritative structure and state. Backends
receive a `DenseNeuronBatch`: stable IDs plus contiguous `ComputeNeuronState`
and immutable `ComputeNeuronParams` records for the requested cells. They
receive no `Network` or `Neuron` references. Core numerical transitions and
`CpuCompute` share the analytical kernels in `core::dynamics`; there is one
implementation of the mathematics. `FlatNeuronInputs` stores IDs, `u64` row
offsets and one amplitude array. Canonical sorting and summation remain per row.

The runtime exports fresh state for each compute phase and writes results back
before the next learning hook. Successful publication validates every result
and rejects stale exports. Failed computation preserves the original partial
state and poisoning behavior. Both default and explicitly selected CpuCompute
are checked against unchanged pre-refactor records in
`tests/compute_regression.rs`; round trips, flat layouts and synchronization
are covered in `tests/dense_compute.rs`.

The two backend phases are unchanged, but backend implementations now accept
dense records and flat input batches instead of the earlier `NeuronStates`
view and per-cell `NeuronInputs` vectors. The public Simulation constructors,
I/O API and sparse graph representation remain unchanged. These batches cover
neurons only: there is no dense synapse graph or persistent device state.

Opt-in host profiling records batch/input/fanout histograms and time in compute,
export/publication, STDP hooks, propagation, utility, maintenance and prediction:

```rust
let simulation = simulation.with_profiling();
// After running, inspect simulation.profile() or take_profile().
```

Profiling is disabled by default and makes no wall-clock reads while disabled.
Its bounded aggregates remain separate from replay logs and experiment results.
`M0Experiment::run_profiled` returns diagnostics alongside the unchanged result,
including training and both frozen probes. Run the release profiling example:

```bash
cargo run --release --example compute_profile
```

An [isolated scheduler benchmark](docs/scheduler-profiling.md) compares the
current queue with timestamp buckets, including eager cancellation, topology
removal and interleaved prediction refreshes:

```bash
cargo run --release --example scheduler_profile
```

The [asynchronous development workload](docs/self-organisation-profiling.md)
measures larger spatial networks with Pair-STDP, utility, local synapse
formation/pruning and a paired control with fixed topology. Profiles include
complete scheduler batch sizes, exact singleton counts and dequeue time:

```bash
cargo run --release --features development --example self_organisation_profile
```

The [pruning optimization and repeated comparison](docs/pruning-optimization.md)
replace linear membership scans with lookups in the existing synapse index.

With `--features cuda`, `CudaCompute::new(0)` selects the experimental GPU
backend through the existing `with_compute_backend` method. It dynamically
requires the NVIDIA driver and NVRTC (CUDA 12.8 bindings), but needs no `nvcc`
at build time. Host records are packed field by field into device SoA buffers;
Rust structs and padding are never copied as a device ABI. Each call allocates,
uploads, computes, downloads and synchronizes. Canonical input reduction stays
on the CPU, while CUDA computes local neuron dynamics. Numerical errors replay
the CPU error path to preserve partial state; device failures are reported.

`CpuCompute` remains the exact reference. CUDA is compared with
`abs_error <= 2e-5 + 2e-6 * max(abs(cpu), abs(gpu))` for scalar state and exact
checks for discrete state/spikes in the validation cases. Floating-point
differences near thresholds can change later spike times; this is not a promise
of identical CPU/GPU trajectories for arbitrary inputs. Explicit GPU tests
require accessible hardware and NVRTC on the shared-library search path:

```bash
cargo test --release --features cuda --test cuda_compute -- --ignored --nocapture --test-threads=1
```

Ordinary test runs leave these three hardware tests ignored instead of silently
reporting a GPU pass. See [measured results and commands](docs/profiling.md).

## M0 reference experiment

M0 is the library's first controlled sequence-learning experiment. It trains a
small recurrent network on the repeated temporal sequence `A → B → C → D`,
freezes its weights, then presents only `A`. Four controlled groups compare
ordered learning, frozen weights, randomized input, and ordered learning with
local cellular/structural homeostasis. The executable study uses six paired,
independently seeded initial weight states, repeats every frozen probe and
complete run exactly, and reports one-sided exact sign tests.

The assay gives all twelve directed non-self pattern transitions identical
tetrahedral geometry. A target-blind, balanced pool of seeded initial-weight
offsets is shared by G1–G4 within each paired seed. The target sequence is
supplied by temporal experience rather than hard-coded connectivity.

M0 is an engineering criterion, not a scientific proof: it demonstrates that
this concrete setup reproduces the declared behavior deterministically under
the six fixed seeds. It does not establish robustness across wider parameter
ranges, other sequences or network sizes, or seeds outside the fixed set.

```bash
cargo run --example m0_sequence
cargo test --all-features --all-targets
cargo test --all-features --doc
```

The complete architectural rules, slice responsibilities, experiment design,
and success criteria are documented in
[`NERVA_SLICE_ARCHITECTURE.md`](NERVA_SLICE_ARCHITECTURE.md).
The currently compiled dependency layers and consolidation decisions are kept
in [`docs/architecture.md`](docs/architecture.md).
