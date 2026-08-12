# DSVLM-M0

DSVLM-M0 is a Rust library for a deterministic, event-driven spatial spiking
network with local Pair-STDP. It deliberately has no global simulation tick,
backpropagation, global loss, reward signal, morphogen fields, or hidden
task-specific neuron behavior.

The first experiment trains a small recurrent network on the repeated temporal
sequence `A → B → C → D`, freezes its weights, then presents only `A`. Four
controlled groups compare ordered learning, frozen weights, randomized input,
and ordered learning with local cellular/structural homeostasis. The default executable
study uses six paired, independently seeded initial weight states, repeats every
frozen probe and complete run exactly, and reports one-sided exact sign tests.

Run all checks and the M0 example with:

```bash
cargo test --all-features --all-targets
cargo run --example m0_sequence
```

The complete architectural rules, slice responsibilities, experiment design,
and success criteria are documented in
[`DSVLM_SLICE_ARCHITECTURE.md`](DSVLM_SLICE_ARCHITECTURE.md).

## Public slices

- `config`, `math`, and `core`: validated parameters and state only
- `runtime`: deterministic timestamp batches and spike propagation
- `learning`: local Pair-STDP and per-neuron homeostasis
- `roots`, `transduction`, and `nerves`: fixed external connections
- `environment` and `experiment`: reproducible M0 orchestration
- feature-gated `metrics`, `debug`, `visualization`, and post-M0 `development`

Weights are non-negative magnitudes. Excitatory or inhibitory effect is always
derived from the presynaptic neuron's polarity, so learning cannot flip a
connection's sign. Positions affect geometry, attenuation, and delay; only
`NeuronId` defines identity.

The M0 assay gives all twelve directed non-self pattern transitions identical
tetrahedral geometry. A target-blind, balanced pool of seeded initial-weight
offsets is shared by G1–G4 within each paired seed. The target sequence is
supplied by temporal experience rather than hard-coded connectivity.
