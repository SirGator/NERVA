# Read-only electrical boundary observations

`Simulation::set_dynamics_auditing(true)` enables two optional event variants:

- `ObservationEvent::NeuronAdvanced`: time, neuron, original local timestamp
  and membrane potential before/after continuous evolution.
- `ObservationEvent::NeuronIntegrated`: time, neuron, potential before/after,
  effective threshold before integration, refractory deadline, canonical f32
  net input, positive/negative f64 input sums and actual spike decision.

The default is disabled. The ordinary CPU observation stream and numerical
trajectory retain their previous behavior. Observation settings/logs are
excluded from `same_replay_state`; no neural state or learning parameter changes.

Continuous evolution triggered by `set_modulation` is recorded too. Otherwise
a potential ledger could incorrectly treat that evolution as missing.
Continuous change includes leak, intrinsic current and modulation; the event
does not attribute all of it to passive leak.

The compute backend integrates the original input rows. The audit reads the
canonical rows after computation and observes published native cell state.
It does not perform a second integration or mutate input ordering. Before
integration, the effective threshold includes transient adaptation/modulation.
A spike's stored end potential is the reset potential, not its peak.

Drain the event log regularly. These optional events share the existing
observation log; the library does not silently drop them or impose a new limit.
The model exposes the same switch through
`Model::set_dynamics_auditing`; collect `AuditOptions { raw_events: true, ... }`
and drain each frame. The
[research protocol](../../NERVA_Modell/research/DIRECTION_PATH.md) imposes
explicit frame/branch limits and rejects incomplete or unbalanced observations.

Tests compare audited and unaudited native trajectories, cover simultaneous
excitation/inhibition, reset, refractory rejection and continuous decay, and
compare the full replay state of a chemically active model.

The [direction/strength factorial study](../../NERVA_Modell/research/DIRECTION_STRENGTH.md)
reuses these observations while independently varying weight-delta direction
and L2 strength. It checks bounds and published f32 norm/direction fidelity
before interpreting effects. Recurrent relay searches change existing core
weights only; no additional library dynamics or learning mechanisms are needed.
