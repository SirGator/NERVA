# Synapse-local preservation of small learning contributions

`PairStdp` supports an explicitly selected numerical publication policy.
`WeightUpdateMode::DirectF32` is the default and retains the historical
f32 delta conversion, addition and clipping. `Accumulated` changes only
publication of requested learning mass:

```rust
use nerva::config::LearningConfig;
use nerva::learning::{PairStdp, ThreeFactorConfig, WeightUpdateMode};

let rule = PairStdp::new(&LearningConfig::default())
    .with_three_factor(ThreeFactorConfig::default())
    .expect("valid local learning parameters")
    .with_weight_update_mode(WeightUpdateMode::Accumulated);
```

Pair timing, nearest-neighbor selection, learning rates, eligibility lifetime,
local receptor factors and scheduling are unchanged. Immediate pair updates
also use the selected publication policy. The controlled host comparison
uses the same three-factor rule in both arms.

## State and arithmetic

Each synapse owns one signed `f64` carry, initialized to zero. The transmitted
weight remains a bounded nonnegative `f32`; propagation never reads the carry.
For a requested signed f64 contribution `delta`:

```text
pending = carry + delta
accepted = clamp(pending, min_weight - weight, max_weight - weight)
next_weight = f32(f64(weight) + accepted)
next_carry = accepted - (f64(next_weight) - f64(weight))
```

The small contributions are summed near zero before addition to the larger
visible weight. This also retains contributions below an f64 ULP of the visible
weight. It is still finite-precision arithmetic, not exact real arithmetic.

Nearest rounding can publish slightly more than the requested amount. The
carry then becomes negative and corrects subsequent updates. Opposite
contributions can cancel locally. Saturation discards the out-of-bounds part;
it never stores hidden debt that would delay a later inward update.

The state costs one f64 per synapse and constant arithmetic per publication,
without a per-synapse allocation, global accumulator, new tick or dependency.

## Freeze, edits and replay

Freezing learning neither flushes nor deletes already accumulated mass.
Eligibility time still advances under the existing freeze contract, preventing
retroactive integration of frozen time. Propagation continues using only the
unchanged visible weight until learning is explicitly resumed.

`Synapse::set_weight` and `set_weight_clamped` are authoritative replacements
and clear the carry. Budget normalization uses this same replacement semantics.
Changing topology removes the carry with the synapse; newly formed synapses
start with zero carry.

`set_weight_preserving_learning_residual` supports frozen counterfactuals.
The model exposes this only behind its existing frozen diagnostic guard:
the experiment changes the transmitted weight, retains the unused learning
state and verifies full replay equality when the weights are restored.

Carry participates in clone/equality and therefore in the existing network
and simulation replay checks. Zero carry keeps the historical Synapse Debug
representation, so the original CPU reference fixture is checked byte-for-byte
without regenerating it. Nonzero carry remains visible in diagnostic snapshots.
Debug output is not a persistence format.

## Audits and conservation

`DelayedLearningUpdate` previews the exact next weight, residual before/after,
signed clipped mass and signed numerical publication error. Optional learning
audits aggregate these values per synapse and drain window. The conservation
identity is:

```text
requested = visible_weight_change + carry_change + clipped_mass + rounding_error
```

Counts of unchanged requests are not counts of lost contributions under
accumulation: a request may remain in the carry. Requested and published
gross mass also differ from the absolute final net weight change because
opposite updates can cancel. Interpret each quantity separately.

The research harness records these balances only for the explicit precision
comparison, retaining compact per-synapse totals rather than full event traces.
The opt-in audit does not change the model replay state.

## Verification

```bash
cargo test --offline --release --test weight_accumulation
cargo test --offline --release --test compute_regression
```

Tests cover the original 1-vs-1,000-interval failure, signed carry, cancellation,
tiny contributions, both bounds without saturation debt, freeze/resume,
authoritative edits, diagnostic restoration, audit neutrality and conservation.

The remaining f32 eligibility decay can still depend on event partitioning.
The correction does not promise equal arbitrary trajectories, biological
calibration or functional memory.
[Controlled model comparison](../../NERVA_Modell/research/PRECISION.md).
