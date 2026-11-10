# Local neuromodulation and continuous tissue profiles

The library provides the mechanisms independently of `NERVA_Modell`. One shared
LIF/intrinsic neuron model supports different electrical construction profiles,
receptors, local chemical release, signed learning eligibility and optional
synapse-local resource/facilitation dynamics. No memory routine, external reward,
object label or fixed cognitive region is built into these mechanisms.

```bash
cargo run --offline --release --example neuromodulated_tissue
cargo test --offline --release --test neuromodulation
```

`cell_profiles::AreaField` returns convex cortex/DG/CA3/CA1-like construction
influences along an explicitly supplied spatial axis. A positive width gives
smooth overlap; width zero is the sharp control. `electrical_profile` blends
threshold distance above rest, membrane constants and intrinsic capabilities.
Polarity remains discrete. `connection_affinity` keeps every area pair possible;
the caller supplies spatial candidate limits and seeded sampling. An explicit
DG→CA3-like synapse-class tendency is available through `short_term_profile`.
These synthetic values are not physiological calibration or demonstrated memory.

The passive types live in `core::modulation` and are re-exported from
`neuromodulation`. Core and compute still compile without the field solver,
runtime, learning or model. Each neuron can receive a validated `ReceptorProfile`
and `ReleaseProfile` before insertion in a `Network`. The host supplies source
identities; `ChemicalSpecies` distinguishes dopamine-, acetylcholine- and
noradrenaline-like material from `ModulatorId`, which identifies its source.

Opt in using `Simulation::with_neuromodulation(ChemicalConfig)` or
`enable_neuromodulation`. Source spikes release their configured mass from the
cell position. An explicit `schedule_chemical_release` is available for isolated
mechanism tests. Material reaches receptive cells without requiring a classical
synapse. There is no external novelty detector or hardcoded consequence evaluator.
Source cells use the same dynamics and may receive plastic incoming connections.

Each release is a finite-width Gaussian mass in unbounded three-dimensional
space. Querying uses the analytic diffusion/linear-clearance solution with
variance σ₀²+2Dt and mass M·exp(−t/τ). Default clearance time constants are 400/200/500 ms;
Gaussian width starts at 0.08 geometry units. Pulses are retained for 20 clearance
time constants (at most ten seconds by default), explicitly truncating their
remaining relative mass at exp(−20). The bounded field coalesces simultaneous
same-source releases and rejects capacity overflow instead of silently dropping
new material. Release, coordinate, capacity and time checks are explicit.

Receptor occupancy is a synthetic saturating C/(1+C) response. D1-like and
D2-like sensitivities have opposite configured signs. Bounded effects change
threshold offset, current, transmission gain and the signed third learning
factor. Homeostatic base current/threshold stay separate. CPU dense records and
the optional CUDA transfer/kernel include the held current/threshold offsets.
Four explicit GPU tests now pass on the local RTX 3060, including source-driven
chemistry, delayed eligibility learning and facilitated transmission. That
integrated case has identical CPU/GPU events, synapses and chemical state; neuron
floats remain within the declared absolute-plus-relative tolerance. CPU remains
the default. The small reference workload is slower on this CUDA backend because
of transfer/launch overhead; no general acceleration is claimed.
[Validation and runtime provenance](benchmarks/neuromodulation-gpu-20261109.json).

Receptor effects are held between optional 5-ms chemical events. Explicit test
releases refresh before same-time input integration; endogenous spike release
first affects receptors on the next chemical event. Empty fields create no
maintenance clock. Receptor-free cells skip concentration queries. Changed cells
receive a threshold check, and their intrinsic predictions are refreshed so an
old spontaneous spike cannot survive an excitability change. The unchanged
baseline schedules no chemical events and has identity transmission gains.

`PairStdp::with_three_factor(ThreeFactorConfig)` retains nearest-pair timing but
stores its signed delta in a separate synapse learning tag. Between events the
tag decays exponentially; at native events its analytic integral is multiplied
by the previously held postsynaptic receptor factor and bounded into the weight.
This tag is independent of the existing nonnegative structural utility eligibility.
Frozen weights still permit time to decay tags, preventing frozen intervals from
being consolidated retrospectively. Ordinary `PairStdp::new` keeps immediate STDP.

Short-term release captures both resource/facilitation state and receptor-dependent
transmission gain at emission. Later changes cannot modify a travelling impulse.
Absent short-term configuration means exactly the old gain of one. Live batch
configuration through `set_short_term_profiles` validates all selected edges
before committing any change. Excitatory/inhibitory sign remains presynaptic.

Public integration tests isolate delayed learning against receptor knockout,
D2-like opposition, inactive neighboring synapses and frozen weights. They also
exercise source-driven volume transmission, threshold-triggered firing,
cancellation of stale intrinsic predictions, actual facilitated arrivals,
capacity failure and replay state. The CPU regression fixture retains every
historical numerical value; only newly introduced neutral fields were added.
These are mechanism guarantees, not a functional hippocampal memory result.

Biological motivation: mossy-fiber frequency facilitation
([Salin et al., 1996](https://doi.org/10.1073/pnas.93.23.13304)), delayed local
eligibility ([Izhikevich, 2007](https://www.izhikevich.org/publications/dastdp.pdf))
and synapse-specific dopaminergic modulation
([Otmakhova & Lisman, 1996](https://pmc.ncbi.nlm.nih.gov/articles/PMC6579102/)).
Their quantitative parameters and global-reward assumptions are not adopted.
Dendritic compartments, physiological calibration, learned novelty detection,
pattern separation/completion benefit and integrated memory remain to be tested.

Verification on 2026-11-09: 299 default-build and 348 feature-enabled tests passed,
including documentation and the unchanged historical numeric reference; Clippy
with warnings as errors and formatting passed. The integrated example reports
16 cells, 84 edges, 70 spikes, 411 weight updates and 30 local chemical pulses.
[Historical CPU-stage verification](neuromodulation-verification.json).

## Repeating the optional GPU checks

The matching [NVIDIA NVRTC wheel 12.8.93](https://pypi.org/project/nvidia-cuda-nvrtc-cu12/12.8.93/)
was downloaded, checksum-verified and extracted only under `/tmp/nerva-nvrtc-12.8.93`.
The system CUDA installation and global Python packages were not changed. If the
temporary runtime has been removed, it can also be prepared with a target-only
installation:

```bash
python3 -m pip install --no-deps --target /tmp/nerva-nvrtc-12.8.93 nvidia-cuda-nvrtc-cu12==12.8.93
```

On this WSL host, the test command is:

```bash
LD_LIBRARY_PATH=/tmp/nerva-nvrtc-12.8.93/nvidia/cuda_nvrtc/lib:/usr/lib/wsl/lib \
  cargo test --offline --release --features cuda --test cuda_compute -- --ignored --nocapture --test-threads=1
```

`/tmp` is temporary; repeat the isolated runtime preparation if that directory is
removed. The package name, wheel SHA-256 and exact test output are retained in the
validation artifact. Earlier missing-runtime reports are historical snapshots.


## Optional causal learning observations

`Simulation::enable_learning_audit(max_records)` enables a separate bounded
read-only stream. It is excluded from neural replay comparison, like profiling.
`drain_learning_audit()` preserves recording and returns the current buffer;
`disable_learning_audit()` returns pending records and stops recording.
`LearningAuditLog::is_complete()` must be checked. Overflow is counted explicitly;
the model adapter rejects incomplete diagnostic trials.

Pair-tag hooks and all actual weight changes are retained individually. Unchanged
delayed integrations are aggregated per synapse between drains: exact decaying-tag
integrals, unrounded requested deltas, actual positive/negative changes, requests
without a weight change, signal timing/ranges and weight bounds. Chemical events
record concentrations and applied receptor effects, including receptor-free cells.
The ordinary execution path schedules no additional events for this observation.

`PlasticityRule::delayed_update_diagnostics` defaults to no arithmetic description.
PairStdp uses the same pure local calculation for actual integration and optional
observations; learning parameters and f32 update behavior were not changed.
Core neuron state and the historical numerical reference remain unchanged.

A public integration test now follows the complete controlled chain from pairing
and delayed local DA to a retained weight and a later spike difference. The shared
near-threshold bench probe is chosen diagnostically between the two weight effects.
A second test checks exact runtime neutrality and explicit overflow. Model tests
also compare complete replay state with and without observation.

The [256-cell matched-weight diagnosis](../../NERVA_Modell/research/results/hippocampal_causality_256_20261109/RESULTS.md)
records the model-level limits: native hippocampal weights change delivered
amplitudes and stored potentials but not spikes in the tested probes; a fixed
100× artificial end-delta contrast changes spikes/decoded keys for one seed.
This is a diagnostic effect, not task learning or validated memory.
