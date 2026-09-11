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
- `development`: the reserved post-M0 boundary; growth is not implemented yet

The default build uses no external crate dependencies. The public API is in
early development and may change between releases.

## Public slices

- `primitives`, `math`, and `config`: policy-free foundations and validation
- `core`: neurons, synapses, spikes, and deterministic graph state
- `learning`: local Pair-STDP and per-neuron homeostasis
- `runtime`: deterministic timestamp batches and spike propagation
- `roots`, `transduction`, and `nerves`: fixed external connections
- `environment` and `experiment`: reproducible M0 orchestration
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
