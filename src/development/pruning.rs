//! Synapse-local pruning based on a smoothed, decaying utility criterion.
//!
//! Unlike growth, which is derived from the postsynaptic neuron's structural
//! drive, pruning evaluates each synapse individually. A synapse whose local
//! smoothed utility stays below a threshold for a sustained period is a
//! candidate for removal.
//!
//! The utility memory is event-driven: between evidence samples the stored
//! estimate decays toward zero with time constant `τ_U`,
//!
//! $$ U_{ij}(t) = U_{ij}(t_0)\,e^{-(t-t_0)/\tau_U}, $$
//!
//! and at each evidence event an exponential moving average is applied to the
//! decayed value,
//!
//! $$ U'_{ij}(t) = (1-\eta)\,U_{ij}(t) + \eta\,u_{ij}(t). $$
//!
//! When $U_{ij} < \theta_p$ over a sufficient duration, the synapse is pruned.
//! This is a synapse-local decision, not a neuron-level "I feel over-supplied,
//! delete something" signal.
//!
//! ## Below-threshold tracking
//!
//! The duration condition is not "time since last utility update" but
//! "continuous time below threshold". A `PruningController` tracks
//! `below_threshold_since` per synapse:
//!
//! - `U < threshold` and no timer → start timer at the **exact analytic
//!   crossing time**, not at the last update timestamp.
//! - `U < threshold` and timer already running → keep it.
//! - `U >= threshold` → reset timer to `None`.
//!
//! A synapse is eligible when `now - below_threshold_since >= T_min`.
//!
//! ### Analytic crossing time
//!
//! With pure exponential decay, a utility that was above the threshold at the
//! last evidence event crosses the threshold at an exactly computable time:
//!
//! $$ t_{\text{cross}} = t_0 + \tau_U \ln\!\left(\frac{U(t_0)}{\theta_p}\right). $$
//!
//! If the stored utility was already at or below the threshold at $t_0$, the
//! crossing happened at $t_0$ itself. Using this analytic value (rather than
//! `utility_updated_at`) prevents the controller from counting time during
//! which the synapse was still above the threshold, which would prune a
//! decaying-but-still-useful connection far too early.
//!
//! The decayed utility is evaluated through [`Synapse::utility_at`] so that
//! even a synapse that has not received a new evidence sample for a while is
//! judged by its current decayed value, not the stale stored one. This keeps
//! pruning consistent with NERVA's no-global-tick, event-driven design.

use std::{collections::BTreeMap, error::Error, fmt};

use crate::{
    core::{NeuronId, SynapseError, SynapseId},
    learning::PlasticityRule,
    primitives::SimTime,
    runtime::Simulation,
};

use super::plan::{DevelopmentPlan, PlannedPruning};

/// Immutable parameters for synapse-local pruning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PruningConfig {
    /// Utility threshold below which a synapse is eligible for pruning.
    pub pruning_threshold: f32,
    /// Minimum simulation duration in microseconds that utility must stay
    /// continuously below the threshold before the synapse is pruned.
    pub min_below_duration_us: u64,
    /// Whether non-plastic synapses are also eligible for pruning.
    pub prune_non_plastic: bool,
}

impl PruningConfig {
    /// Validates pruning parameters.
    pub fn validate(&self) -> Result<(), PruningError> {
        if !self.pruning_threshold.is_finite() || self.pruning_threshold < 0.0 {
            return Err(PruningError::InvalidThreshold(self.pruning_threshold));
        }
        Ok(())
    }
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        }
    }
}

/// Per-synapse pruning tracking state.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PruningState {
    /// When utility first dropped below the threshold, or `None` if utility
    /// is currently at or above threshold (or the synapse has not been
    /// observed yet).
    below_threshold_since: Option<SimTime>,
    /// Snapshot of `(utility, utility_updated_at)` at the last `observe()`.
    /// Used to detect that new utility evidence has been written between two
    /// observations, which invalidates the cached `below_threshold_since`
    /// timer and forces a re-computation from the new memory state.
    last_observed_utility: Option<(f32, Option<SimTime>)>,
}

/// Stateful controller that tracks continuous below-threshold utility per
/// synapse and identifies pruning candidates.
///
/// The controller owns `below_threshold_since` timers keyed by `SynapseId`.
/// The threshold itself lives in [`PruningConfig`]; the smoothed utility
/// estimate lives on each `Synapse` and is updated by the runtime via
/// `Synapse::update_utility`. This separation keeps the pruning policy
/// out of the synapse data model and avoids re-deriving "below since" from
/// `utility_updated_at`, which would reset on every update even when utility
/// stays low.
#[derive(Clone, Debug, Default)]
pub struct PruningController {
    config: PruningConfig,
    states: BTreeMap<SynapseId, PruningState>,
}

impl PruningController {
    /// Creates a controller with the given configuration.
    pub fn new(config: PruningConfig) -> Result<Self, PruningError> {
        config.validate()?;
        Ok(Self {
            config,
            states: BTreeMap::new(),
        })
    }

    /// Immutable configuration.
    pub const fn config(&self) -> &PruningConfig {
        &self.config
    }

    /// Updates the below-threshold timer for every synapse in the network.
    ///
    /// Call this after utility samples have been updated by the runtime.
    /// Synapses that no longer exist are pruned from the tracking map. The
    /// decayed utility is evaluated at the current simulation time using
    /// `utility_tau_us`, so a synapse whose last evidence event was long ago
    /// is judged by its decayed value, not the stale stored one.
    ///
    /// Only synapses whose presynaptic polarity is **supported** by the
    /// simulation's utility rule are tracked. A rule that cannot judge a
    /// polarity never writes utility evidence for it, so a zero memory there
    /// means *utility unknown*, not *utility bad* — such synapses are exempt
    /// from utility-based pruning and their tracking state is removed.
    ///
    /// When a synapse is observed below the threshold for the first time, the
    /// timer is started at the **analytic crossing time** derived from the
    /// last stored utility sample and `τ_U`, not at `utility_updated_at`.
    /// This avoids counting the interval during which the decaying utility was
    /// still above the threshold, which would otherwise prune a still-useful
    /// connection far too early.
    ///
    /// If new utility evidence has been written between two observations
    /// (detected via a change in `utility_updated_at`), the cached timer is
    /// invalidated and re-computed from the new memory state. This prevents a
    /// silent recovery between observations from being missed: if a synapse
    /// recovered above threshold and then decayed back below, the timer starts
    /// at the **second** crossing, not at the original one.
    pub fn observe<R: PlasticityRule>(
        &mut self,
        simulation: &Simulation<R>,
    ) -> Result<(), PruningError> {
        self.config.validate()?;
        let now = simulation.current_time();
        let network = simulation.network();
        let tau_u = simulation.utility_dynamics().utility_tau_us;
        let threshold = self.config.pruning_threshold;
        let utility_rule = simulation.utility_rule();

        let live_ids: Vec<SynapseId> = network.synapse_ids().collect();

        self.states.retain(|id, _| live_ids.contains(id));

        for synapse in network.synapses() {
            if !self.config.prune_non_plastic && !synapse.is_plastic() {
                self.states.remove(&synapse.id());
                continue;
            }

            let presynaptic_polarity = network
                .neuron(synapse.pre())
                .map(|neuron| neuron.polarity())
                .ok_or(PruningError::UnknownPresynapticNeuron(synapse.pre()))?;
            if !utility_rule.supports(presynaptic_polarity) {
                // The active rule cannot judge this synapse: its utility
                // memory is (and stays) zero because no evidence is ever
                // written. Zero here means "unknown", not "bad", so the
                // synapse must not become a pruning candidate.
                self.states.remove(&synapse.id());
                continue;
            }

            let state = self.states.entry(synapse.id()).or_insert(PruningState {
                below_threshold_since: None,
                last_observed_utility: None,
            });

            let current_memory = (synapse.utility(), synapse.utility_updated_at());
            let memory_changed = state.last_observed_utility != Some(current_memory);
            state.last_observed_utility = Some(current_memory);

            let decayed_utility = synapse
                .utility_at(now, tau_u)
                .map_err(PruningError::Synapse)?;
            if decayed_utility < threshold {
                let needs_recompute = state.below_threshold_since.is_none() || memory_changed;
                if needs_recompute {
                    state.below_threshold_since = Some(below_threshold_crossing(
                        synapse.utility(),
                        synapse.utility_updated_at(),
                        threshold,
                        tau_u,
                        now,
                    ));
                }
            } else {
                state.below_threshold_since = None;
            }
        }

        Ok(())
    }

    /// Finds all synapses currently eligible for pruning.
    ///
    /// A synapse is eligible when its `below_threshold_since` timer has been
    /// running for at least `min_below_duration_us` and its decayed utility is
    /// still below the threshold at the current simulation time. This function
    /// does not mutate the network.
    pub fn candidates<R: PlasticityRule>(
        &self,
        simulation: &Simulation<R>,
    ) -> Result<Vec<PlannedPruning>, PruningError> {
        self.config.validate()?;
        let now = simulation.current_time();
        let network = simulation.network();
        let tau_u = simulation.utility_dynamics().utility_tau_us;
        let mut candidates = Vec::new();

        for (synapse_id, state) in &self.states {
            let Some(since) = state.below_threshold_since else {
                continue;
            };
            let Some(elapsed) = now.duration_since(since) else {
                continue;
            };
            if elapsed < self.config.min_below_duration_us {
                continue;
            }
            let Some(synapse) = network.synapse(*synapse_id) else {
                continue;
            };
            let decayed_utility = synapse
                .utility_at(now, tau_u)
                .map_err(PruningError::Synapse)?;
            if decayed_utility >= self.config.pruning_threshold {
                continue;
            }
            candidates.push(PlannedPruning {
                synapse_id: *synapse_id,
                target: synapse.post(),
            });
        }

        candidates.sort_by_key(|p| p.synapse_id);
        Ok(candidates)
    }

    /// Removes tracking state for a pruned synapse.
    ///
    /// Call this after a `DevelopmentPlan` containing the pruning has been
    /// committed, so stale timers do not linger.
    pub fn forget(&mut self, synapse_id: SynapseId) {
        self.states.remove(&synapse_id);
    }

    /// Removes tracking state for all synapses in the given plan.
    pub fn forget_plan(&mut self, plan: &DevelopmentPlan) {
        for pruning in &plan.prunings {
            self.forget(pruning.synapse_id);
        }
    }

    /// Builds a development plan with pruning candidates from this controller.
    ///
    /// Formations in the plan are left untouched; only `prunings` is filled.
    pub fn populate_plan<R: PlasticityRule>(
        &self,
        plan: &mut DevelopmentPlan,
        simulation: &Simulation<R>,
    ) -> Result<(), PruningError> {
        plan.prunings = self.candidates(simulation)?;
        Ok(())
    }

    /// Returns the `below_threshold_since` timestamp for a synapse, if any.
    ///
    /// Useful for tests and diagnostics.
    pub fn below_threshold_since(&self, synapse_id: SynapseId) -> Option<SimTime> {
        self.states
            .get(&synapse_id)
            .and_then(|s| s.below_threshold_since)
    }
}

/// Pruning configuration or evaluation failed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PruningError {
    /// The pruning threshold was negative, NaN, or infinite.
    InvalidThreshold(f32),
    /// A synapse rejected a utility read or update.
    Synapse(SynapseError),
    /// The presynaptic neuron of a tracked synapse is missing from the
    /// network, so its polarity (and therefore rule support) cannot be
    /// determined.
    UnknownPresynapticNeuron(NeuronId),
}

impl fmt::Display for PruningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidThreshold(value) => {
                write!(
                    formatter,
                    "pruning threshold must be finite and non-negative, got {value}"
                )
            }
            Self::Synapse(error) => {
                write!(formatter, "synapse utility read failed: {error}")
            }
            Self::UnknownPresynapticNeuron(neuron_id) => {
                write!(
                    formatter,
                    "presynaptic neuron {neuron_id} of a tracked synapse is missing"
                )
            }
        }
    }
}

impl Error for PruningError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Synapse(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SynapseError> for PruningError {
    fn from(error: SynapseError) -> Self {
        Self::Synapse(error)
    }
}

/// Computes the exact simulation time at which a decaying utility estimate
/// first crosses below `threshold`.
///
/// Given a stored utility `u0` sampled at time `t0` and a decay time constant
/// `tau_u`, the decayed utility is `u0 · exp(-(t − t0) / tau_u)`. Solving for
/// the crossing time gives
///
/// ```text
/// t_cross = t0 + tau_u · ln(u0 / threshold)
/// ```
///
/// when `u0 > threshold`. When `u0` is already at or below the threshold, the
/// crossing happened at or before `t0`, so `t0` is returned. When the synapse
/// has never been updated (`t0 == None`), the caller-supplied `now` is
/// returned as a conservative fallback: the synapse has never had positive
/// evidence, so it is considered below threshold since the current
/// observation time.
///
/// The result is clamped to `t0` from below and to `now` from above. The
/// upper clamp handles floating-point noise at the crossing boundary, so a
/// timer never starts in the future relative to the observation that detected
/// the below-threshold state.
fn below_threshold_crossing(
    u0: f32,
    t0: Option<SimTime>,
    threshold: f32,
    tau_u: f32,
    now: SimTime,
) -> SimTime {
    let Some(t0) = t0 else {
        return now;
    };
    if !u0.is_finite() || u0 <= 0.0 || !threshold.is_finite() || threshold <= 0.0 {
        return t0;
    }
    if u0 <= threshold {
        return t0;
    }
    let ratio = f64::from(u0) / f64::from(threshold);
    let ln_ratio = ratio.ln();
    let delta_us = f64::from(tau_u) * ln_ratio;
    if !delta_us.is_finite() || delta_us <= 0.0 {
        return t0;
    }
    let delta_us_int = delta_us.round();
    if !(0.0..=u64::MAX as f64).contains(&delta_us_int) {
        return t0;
    }
    let crossing = t0.saturating_add_us(delta_us_int as u64);
    if crossing > now { now } else { crossing }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{NeuronConfig, RuntimeConfig},
        core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
        learning::{ExcitatoryCausalUtility, NoPlasticity},
        math::Position3D,
        primitives::Weight,
    };

    fn make_network_with_synapse(utility: f32) -> Network {
        let mut network = Network::new();
        network
            .add_neuron(
                Neuron::new(
                    NeuronId(1),
                    Position3D::ORIGIN,
                    Polarity::Excitatory,
                    None,
                    NeuronConfig::default(),
                    SimTime::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
        network
            .add_neuron(
                Neuron::new(
                    NeuronId(2),
                    Position3D::new(1.0, 0.0, 0.0),
                    Polarity::Excitatory,
                    None,
                    NeuronConfig::default(),
                    SimTime::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
        let mut synapse = Synapse::new(
            SynapseId(10),
            NeuronId(1),
            NeuronId(2),
            Weight::new(0.5).unwrap(),
            1,
            true,
        )
        .unwrap();
        synapse
            .update_utility(utility, 0.5, SimTime::ZERO, 1_000_000.0)
            .unwrap();
        network.add_synapse(synapse).unwrap();
        network
    }

    /// Builds a simulation whose utility rule supports the excitatory test
    /// synapses, so the controller actually tracks them. (`NoUtility`
    /// supports no polarity and would exempt every synapse.)
    fn make_simulation(network: Network) -> Simulation<NoPlasticity> {
        Simulation::with_utility_rule(
            network,
            NoPlasticity,
            Box::new(ExcitatoryCausalUtility::new()),
            RuntimeConfig::default(),
            1.0,
        )
        .unwrap()
    }

    fn run_to(network: Network, target_us: u64) -> Simulation<NoPlasticity> {
        let mut simulation = make_simulation(network);
        simulation
            .schedule_external_input(SimTime(target_us), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(target_us)).unwrap();
        simulation
    }

    #[test]
    fn controller_marks_continuously_low_utility_as_candidate() {
        let network = make_network_with_synapse(0.0);
        let simulation = run_to(network, 2_000_000);

        let mut controller = PruningController::new(PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        })
        .unwrap();
        controller.observe(&simulation).unwrap();

        let since = controller
            .below_threshold_since(SynapseId(10))
            .expect("timer should be started");
        let now = simulation.current_time();
        let elapsed = now.duration_since(since).unwrap_or(0);
        assert!(
            elapsed >= 1_000_000,
            "should be below threshold long enough, elapsed={elapsed}"
        );

        let candidates = controller.candidates(&simulation).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].synapse_id, SynapseId(10));
        assert_eq!(candidates[0].target, NeuronId(2));
    }

    #[test]
    fn controller_resets_timer_when_utility_recovers() {
        let network = make_network_with_synapse(0.0);
        let mut simulation = make_simulation(network);
        simulation
            .schedule_external_input(SimTime(500_000), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(500_000)).unwrap();

        let mut controller = PruningController::new(PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        })
        .unwrap();
        controller.observe(&simulation).unwrap();
        assert!(
            controller.below_threshold_since(SynapseId(10)).is_some(),
            "timer should start when utility is low"
        );

        simulation
            .update_synapse(SynapseId(10), |s| {
                s.update_utility(0.5, 0.5, SimTime(500_000), 1_000_000.0)?;
                Ok(())
            })
            .unwrap();

        controller.observe(&simulation).unwrap();
        assert_eq!(
            controller.below_threshold_since(SynapseId(10)),
            None,
            "timer must reset when utility recovers above threshold"
        );

        let candidates = controller.candidates(&simulation).unwrap();
        assert!(
            candidates.is_empty(),
            "recovered synapse must not be a candidate"
        );
    }

    #[test]
    fn controller_restarts_timer_on_re_entry_below_threshold() {
        let network = make_network_with_synapse(0.5);
        let mut simulation = make_simulation(network);
        simulation
            .schedule_external_input(SimTime(100_000), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(100_000)).unwrap();

        let mut controller = PruningController::new(PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        })
        .unwrap();
        controller.observe(&simulation).unwrap();
        assert_eq!(
            controller.below_threshold_since(SynapseId(10)),
            None,
            "high utility must not start timer"
        );

        simulation
            .update_synapse(SynapseId(10), |s| {
                s.update_utility(0.0, 1.0, SimTime(100_000), 1_000_000.0)?;
                Ok(())
            })
            .unwrap();
        controller.observe(&simulation).unwrap();
        let first_since = controller.below_threshold_since(SynapseId(10));
        assert!(
            first_since.is_some(),
            "timer must start when utility drops below threshold"
        );

        simulation
            .schedule_external_input(SimTime(200_000), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(200_000)).unwrap();
        simulation
            .update_synapse(SynapseId(10), |s| {
                s.update_utility(0.5, 1.0, SimTime(200_000), 1_000_000.0)?;
                Ok(())
            })
            .unwrap();
        controller.observe(&simulation).unwrap();
        assert_eq!(
            controller.below_threshold_since(SynapseId(10)),
            None,
            "timer must reset on recovery"
        );

        simulation
            .schedule_external_input(SimTime(300_000), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(300_000)).unwrap();
        simulation
            .update_synapse(SynapseId(10), |s| {
                s.update_utility(0.0, 1.0, SimTime(300_000), 1_000_000.0)?;
                Ok(())
            })
            .unwrap();
        controller.observe(&simulation).unwrap();
        let second_since = controller.below_threshold_since(SynapseId(10));
        assert!(
            second_since.is_some(),
            "timer must restart on re-entry below threshold"
        );
        assert!(
            second_since.unwrap() > first_since.unwrap(),
            "re-entry timer must be later than the original"
        );

        simulation
            .schedule_external_input(SimTime(1_500_000), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(1_500_000)).unwrap();
        controller.observe(&simulation).unwrap();
        let candidates = controller.candidates(&simulation).unwrap();
        let elapsed = simulation
            .current_time()
            .duration_since(second_since.unwrap())
            .unwrap_or(0);
        assert!(
            elapsed >= 1_000_000,
            "should have been below threshold for long enough after re-entry, elapsed={elapsed}"
        );
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn controller_does_not_prune_too_soon() {
        let network = make_network_with_synapse(0.0);
        let simulation = run_to(network, 100_000);

        let mut controller = PruningController::new(PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        })
        .unwrap();
        controller.observe(&simulation).unwrap();

        let candidates = controller.candidates(&simulation).unwrap();
        assert!(
            candidates.is_empty(),
            "only 100_000 us below, needs 1_000_000"
        );
    }

    #[test]
    fn controller_forgets_pruned_synapses() {
        let network = make_network_with_synapse(0.0);
        let simulation = run_to(network, 2_000_000);

        let mut controller = PruningController::new(PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        })
        .unwrap();
        controller.observe(&simulation).unwrap();
        assert!(controller.below_threshold_since(SynapseId(10)).is_some());

        controller.forget(SynapseId(10));
        assert_eq!(controller.below_threshold_since(SynapseId(10)), None);
    }

    #[test]
    fn controller_cleans_up_removed_synapses() {
        let network = make_network_with_synapse(0.0);
        let mut simulation = make_simulation(network);
        simulation
            .schedule_external_input(SimTime(100_000), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(100_000)).unwrap();

        let mut controller = PruningController::new(PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        })
        .unwrap();
        controller.observe(&simulation).unwrap();
        assert!(controller.below_threshold_since(SynapseId(10)).is_some());

        simulation.remove_synapse(SynapseId(10)).unwrap();
        controller.observe(&simulation).unwrap();
        assert_eq!(
            controller.below_threshold_since(SynapseId(10)),
            None,
            "removed synapse must be cleaned from tracking"
        );
    }

    #[test]
    fn controller_skips_non_plastic_unless_allowed() {
        let mut network = Network::new();
        network
            .add_neuron(
                Neuron::new(
                    NeuronId(1),
                    Position3D::ORIGIN,
                    Polarity::Excitatory,
                    None,
                    NeuronConfig::default(),
                    SimTime::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
        network
            .add_neuron(
                Neuron::new(
                    NeuronId(2),
                    Position3D::new(1.0, 0.0, 0.0),
                    Polarity::Excitatory,
                    None,
                    NeuronConfig::default(),
                    SimTime::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
        let mut synapse = Synapse::new(
            SynapseId(10),
            NeuronId(1),
            NeuronId(2),
            Weight::new(0.5).unwrap(),
            1,
            false,
        )
        .unwrap();
        synapse
            .update_utility(0.0, 0.5, SimTime::ZERO, 1_000_000.0)
            .unwrap();
        network.add_synapse(synapse).unwrap();

        let simulation = run_to(network, 2_000_000);

        let mut controller = PruningController::new(PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        })
        .unwrap();
        controller.observe(&simulation).unwrap();

        assert_eq!(
            controller.below_threshold_since(SynapseId(10)),
            None,
            "non-plastic synapse must not be tracked when prune_non_plastic is false"
        );

        let candidates = controller.candidates(&simulation).unwrap();
        assert!(candidates.is_empty());
    }

    #[test]
    fn below_threshold_crossing_returns_t0_when_already_below() {
        // U0 = 0.005 already below threshold 0.01 → crossing at t0.
        let crossing = below_threshold_crossing(
            0.005,
            Some(SimTime(1_000)),
            0.01,
            1_000_000.0,
            SimTime(2_000_000),
        );
        assert_eq!(crossing, SimTime(1_000));
    }

    #[test]
    fn below_threshold_crossing_returns_now_when_never_updated() {
        let crossing = below_threshold_crossing(0.0, None, 0.01, 1_000_000.0, SimTime(42_000));
        assert_eq!(crossing, SimTime(42_000));
    }

    #[test]
    fn below_threshold_crossing_computes_analytic_time() {
        // U0 = 0.5, threshold = 0.01, tau = 1_000_000 us.
        // t_cross = 0 + 1_000_000 * ln(50) ≈ 1_000_000 * 3.9120 ≈ 3_912_023 us.
        let crossing = below_threshold_crossing(
            0.5,
            Some(SimTime(0)),
            0.01,
            1_000_000.0,
            SimTime(10_000_000),
        );
        let expected = 1_000_000.0_f64 * (50.0_f64).ln();
        let diff = (crossing.as_micros() as f64 - expected).abs();
        assert!(
            diff <= 2.0,
            "crossing {crossing} should be ~{expected} us, diff={diff}"
        );
    }

    #[test]
    fn below_threshold_crossing_clamps_to_now_when_observed_above() {
        // U0 = 0.5, threshold = 0.01, tau = 1s → crossing ≈ 3.912s.
        // If we observe at t = 1s (before crossing), the decayed utility is
        // still above threshold, so observe() should not start the timer.
        // But if called directly with now < crossing, the clamp returns now.
        let crossing =
            below_threshold_crossing(0.5, Some(SimTime(0)), 0.01, 1_000_000.0, SimTime(1_000));
        assert_eq!(
            crossing,
            SimTime(1_000),
            "when now is before the analytic crossing, clamp to now"
        );
    }

    #[test]
    fn controller_uses_analytic_crossing_not_last_update_time() {
        // U0 = 0.5 at t0 = 0, threshold = 0.01, tau = 1s.
        // Analytic crossing ≈ 3.912s. Observe at t = 4s: decayed utility
        // is below threshold, so the timer must start at ~3.912s, not at 0.
        let network = make_network_with_synapse(0.5);
        let simulation = run_to(network, 4_000_000);

        let mut controller = PruningController::new(PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        })
        .unwrap();
        controller.observe(&simulation).unwrap();

        let since = controller
            .below_threshold_since(SynapseId(10))
            .expect("timer should start when decayed utility is below threshold");
        let expected_crossing = 1_000_000.0_f64 * (50.0_f64).ln();
        let diff = (since.as_micros() as f64 - expected_crossing).abs();
        assert!(
            diff <= 2.0,
            "timer should start at analytic crossing ~{expected_crossing} us, got {since}, diff={diff}"
        );

        // At t = 4s, elapsed below threshold is only ~0.088s, not 4s.
        let elapsed = simulation.current_time().duration_since(since).unwrap();
        assert!(
            elapsed < 200_000,
            "elapsed below threshold should be ~88ms, not 4s, got {elapsed} us"
        );

        // With T_min = 1s, the synapse must NOT be a candidate yet.
        let candidates = controller.candidates(&simulation).unwrap();
        assert!(
            candidates.is_empty(),
            "synapse should not be pruned yet; only ~88ms below threshold"
        );
    }

    #[test]
    fn controller_prunes_after_analytic_crossing_plus_min_duration() {
        // U0 = 0.5 at t0 = 0, threshold = 0.01, tau = 1s.
        // Crossing ≈ 3.912s. With T_min = 1s, pruning eligible at ≈ 4.912s.
        let network = make_network_with_synapse(0.5);
        let simulation = run_to(network, 5_000_000);

        let mut controller = PruningController::new(PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        })
        .unwrap();
        controller.observe(&simulation).unwrap();

        let candidates = controller.candidates(&simulation).unwrap();
        assert_eq!(
            candidates.len(),
            1,
            "synapse should be pruned after crossing + T_min"
        );
    }

    #[test]
    fn controller_detects_silent_recovery_between_observations() {
        // Scenario from the review:
        //   t=0    U < threshold (seeded at 0.0)
        //   t=0    observe → timer starts at 0
        //   t=2s   new positive evidence U=0.8 written (NO observe in between)
        //   t=10s  U has decayed below threshold again
        //   t=10s  observe → timer must be re-computed from the NEW memory
        //          (U=0.8 at t=2s), not the old one (U=0.0 at t=0).
        //
        // With the old code, the timer stayed at 0 because
        // below_threshold_since was already Some(0) and the recovery was
        // never observed. With the fix, the change in utility_updated_at
        // (0 → 2s) invalidates the cache and forces a re-computation.
        let network = make_network_with_synapse(0.0);
        let mut simulation = make_simulation(network);
        simulation
            .schedule_external_input(SimTime(0), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(0)).unwrap();

        let config = PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        };
        let mut controller = PruningController::new(config).unwrap();

        // t=0: U=0 < threshold → timer starts at 0.
        controller.observe(&simulation).unwrap();
        let first_since = controller.below_threshold_since(SynapseId(10));
        assert!(first_since.is_some(), "timer should start at t=0");
        assert_eq!(first_since.unwrap(), SimTime(0));

        // t=2s: positive evidence written WITHOUT calling observe().
        simulation
            .schedule_external_input(SimTime(2_000_000), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(2_000_000)).unwrap();
        simulation
            .update_synapse(SynapseId(10), |s| {
                s.update_utility(0.8, 1.0, SimTime(2_000_000), 1_000_000.0)?;
                Ok(())
            })
            .unwrap();
        // NO observe() here — the recovery happens silently.

        // t=10s: U has decayed from 0.8 at t=2s. With τ=1s:
        // U(10s) = 0.8 * exp(-8) ≈ 0.8 * 3.35e-4 ≈ 2.68e-4 < 0.01.
        // The analytic crossing from the NEW memory is:
        //   t_cross = 2s + 1s * ln(0.8/0.01) ≈ 2 + 4.382 ≈ 6.382s.
        simulation
            .schedule_external_input(SimTime(10_000_000), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(10_000_000)).unwrap();
        controller.observe(&simulation).unwrap();

        let second_since = controller.below_threshold_since(SynapseId(10));
        assert!(
            second_since.is_some(),
            "timer should be running again after re-crossing"
        );
        let second = second_since.unwrap();
        // The timer must be much later than 0 — it should be near 6.382s.
        assert!(
            second > SimTime(5_000_000),
            "recomputed timer {second} should be near 6.382s, not stuck at 0"
        );
        // And it should be close to the expected crossing.
        let expected_crossing_us = 2_000_000.0_f64 + 1_000_000.0_f64 * (0.8_f64 / 0.01_f64).ln();
        let diff = (second.as_micros() as f64 - expected_crossing_us).abs();
        assert!(
            diff <= 2.0,
            "recomputed timer {second} should be ~{expected_crossing_us} us, diff={diff}"
        );

        // At t=10s, elapsed since the NEW crossing is ~3.6s > T_min=1s.
        // So the synapse IS a candidate now. But critically, it would NOT have
        // been a candidate if we had wrongly used the old timer (which would
        // give elapsed=10s, still a candidate — the bug is that the timer
        // would have been wrong if the recovery had happened LATER and the
        // observation had been earlier). The key assertion is that the timer
        // reflects the NEW memory, not the old one.
        let candidates = controller.candidates(&simulation).unwrap();
        // With the correct timer (~6.382s), elapsed at t=10s is ~3.6s > 1s
        // → candidate. With the buggy timer (0), elapsed would be 10s → also
        // candidate. So the candidate count alone doesn't distinguish. The
        // real distinction is the timer value, which we already checked above.
        let _ = candidates;
    }

    #[test]
    fn controller_does_not_reset_timer_when_memory_unchanged_between_observations() {
        // If no new evidence is written between two observations, the timer
        // must NOT be re-computed. This ensures we don't keep moving the
        // crossing time forward on every observe() call.
        let network = make_network_with_synapse(0.5);
        let simulation = run_to(network, 4_000_000);

        let config = PruningConfig {
            pruning_threshold: 0.01,
            min_below_duration_us: 1_000_000,
            prune_non_plastic: false,
        };
        let mut controller = PruningController::new(config).unwrap();

        controller.observe(&simulation).unwrap();
        let first_since = controller.below_threshold_since(SynapseId(10)).unwrap();

        // Observe again at the same time → no memory change → same timer.
        controller.observe(&simulation).unwrap();
        let second_since = controller.below_threshold_since(SynapseId(10)).unwrap();
        assert_eq!(
            first_since, second_since,
            "timer must not change when memory is unchanged"
        );
    }
}
