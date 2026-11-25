//! Fixed postsynaptic budgets for a selected population's excitatory inputs.
//! No task, label, reward or global objective enters this rule.
use crate::core::{Network, NeuronId, Polarity, SimTime, SynapseId};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

/// Invalid pool membership, weight or budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompetitionError(pub &'static str);
impl fmt::Display for CompetitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl Error for CompetitionError {}

/// Immutable pool captured before learning; weights are magnitudes in 0..=2.
#[derive(Clone, Debug, PartialEq)]
pub struct CompetitionPool {
    /// Postsynaptic target, including targets with no eligible incoming edges.
    pub target: NeuronId,
    /// Initially enabled, plastic excitatory edges within the population.
    pub synapses: Vec<SynapseId>,
    /// Sum of initial f32 weights, accumulated in f64 in synapse ID order.
    pub budget: f64,
}

/// Read-only audit of the two distinct causes of weight changes.
#[derive(Clone, Debug, PartialEq)]
pub enum CompetitionEvent {
    /// An ordinary plasticity hook, before timestamp-wide normalization.
    LocalChange {
        /// Hook timestamp.
        time: SimTime,
        /// Changed edge.
        synapse: SynapseId,
        /// Magnitude before the hook.
        before: f32,
        /// Magnitude after the hook.
        after: f32,
    },
    /// One normalization after all hooks of a timestamp.
    Normalized {
        /// Normalization timestamp.
        time: SimTime,
        /// Original immutable pool and budget.
        pool: CompetitionPool,
        /// Tentative weights after all STDP hooks.
        before: Vec<f32>,
        /// Published normalized weights.
        after: Vec<f32>,
        /// Published sum minus budget in f64.
        residual: f64,
    },
}

/// Bounded proportional competition, inactive unless installed in a runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct IncomingCompetition {
    population: BTreeSet<NeuronId>,
    pools: Vec<CompetitionPool>,
    targets: BTreeMap<SynapseId, NeuronId>,
    pending: BTreeSet<NeuronId>,
}
impl IncomingCompetition {
    /// Captures membership and initial budgets without changing any weights.
    pub fn new(network: &Network, population: &[NeuronId]) -> Result<Self, CompetitionError> {
        let selected: BTreeSet<_> = population.iter().copied().collect();
        if selected.len() != population.len()
            || selected.iter().any(|id| network.neuron(*id).is_none())
        {
            return Err(CompetitionError("unknown or duplicate competition neuron"));
        }
        let mut pools = Vec::new();
        let mut targets = BTreeMap::new();
        for &target in &selected {
            let synapses = members(network, &selected, target);
            let mut budget = 0.0;
            for &id in &synapses {
                let weight = network.synapse(id).unwrap().weight().get();
                valid_weight(weight)?;
                budget += f64::from(weight);
                targets.insert(id, target);
            }
            pools.push(CompetitionPool {
                target,
                synapses,
                budget,
            });
        }
        Ok(Self {
            population: selected,
            pools,
            targets,
            pending: BTreeSet::new(),
        })
    }
    /// Pools are ordered by target ID, their edges by synapse ID.
    pub fn pools(&self) -> &[CompetitionPool] {
        &self.pools
    }
    pub(crate) fn changed(&mut self, id: SynapseId) {
        if let Some(target) = self.targets.get(&id) {
            self.pending.insert(*target);
        }
    }
    pub(crate) fn validate(&self, network: &Network) -> Result<(), CompetitionError> {
        for pool in &self.pools {
            if network.neuron(pool.target).is_none()
                || members(network, &self.population, pool.target) != pool.synapses
            {
                return Err(CompetitionError("competition pool membership changed"));
            }
            for &id in &pool.synapses {
                valid_weight(network.synapse(id).unwrap().weight().get())?;
            }
        }
        Ok(())
    }
    pub(crate) fn normalize(
        &mut self,
        network: &mut Network,
        time: SimTime,
    ) -> Result<Vec<CompetitionEvent>, CompetitionError> {
        let mut result = Vec::new();
        for pool in &self.pools {
            if !self.pending.contains(&pool.target) {
                continue;
            }
            let before: Vec<_> = pool
                .synapses
                .iter()
                .map(|id| network.synapse(*id).unwrap().weight().get())
                .collect();
            let after = project(&before, pool.budget)?;
            let residual = after.iter().map(|w| f64::from(*w)).sum::<f64>()
                - effective_budget(pool.budget, &before);
            for (&id, &weight) in pool.synapses.iter().zip(&after) {
                network
                    .synapse_mut(id)
                    .unwrap()
                    .set_weight(crate::primitives::Weight::new(weight).unwrap())
                    .map_err(|_| CompetitionError("competition weight publication failed"))?;
            }
            result.push(CompetitionEvent::Normalized {
                time,
                pool: pool.clone(),
                before,
                after,
                residual,
            });
        }
        self.pending.clear();
        Ok(result)
    }
}
fn members(network: &Network, selected: &BTreeSet<NeuronId>, target: NeuronId) -> Vec<SynapseId> {
    let mut ids: Vec<_> = network
        .incoming_synapse_ids(target)
        .iter()
        .copied()
        .filter(|id| {
            let edge = network.synapse(*id).unwrap();
            edge.is_enabled()
                && edge.is_plastic()
                && selected.contains(&edge.pre())
                && network
                    .neuron(edge.pre())
                    .is_some_and(|n| n.polarity() == Polarity::Excitatory)
        })
        .collect();
    ids.sort();
    ids
}
fn valid_weight(value: f32) -> Result<(), CompetitionError> {
    if value.is_finite() && (0.0..=2.0).contains(&value) {
        Ok(())
    } else {
        Err(CompetitionError("competition weight outside finite 0..=2"))
    }
}
/// Budget that the pool actually distributes.
///
/// The incoming budget never exceeds the capacity of the positive weights
/// (`2` per positive weight). A surplus is therefore not shared with silenced
/// synapses: a weight of `0` stays `0` instead of being revived.
fn effective_budget(budget: f64, weights: &[f32]) -> f64 {
    let positive = weights.iter().filter(|w| **w > 0.0).count();
    budget.min(2.0 * positive as f64)
}

fn project(weights: &[f32], budget: f64) -> Result<Vec<f32>, CompetitionError> {
    for &w in weights {
        valid_weight(w)?;
    }
    if !budget.is_finite() || budget < 0.0 || budget > 2.0 * weights.len() as f64 {
        return Err(CompetitionError("infeasible competition budget"));
    }
    if weights.is_empty() {
        return Ok(Vec::new());
    }
    let mut result = vec![0.0_f64; weights.len()];
    {
        // The budget is capped at the capacity of the positive weights. This
        // keeps the projection feasible and leaves zero weights untouched.
        let mut active: Vec<_> = (0..weights.len()).filter(|i| weights[*i] > 0.0).collect();
        let mut remaining = effective_budget(budget, weights);
        while !active.is_empty() {
            let sum: f64 = active.iter().map(|i| f64::from(weights[*i])).sum();
            let scale = remaining / sum;
            let saturated: Vec<_> = active
                .iter()
                .copied()
                .filter(|i| scale * f64::from(weights[*i]) >= 2.0)
                .collect();
            if saturated.is_empty() {
                for &i in &active {
                    result[i] = scale * f64::from(weights[i]);
                }
                break;
            }
            for &i in &saturated {
                result[i] = 2.0;
                remaining -= 2.0;
            }
            active.retain(|i| !saturated.contains(i));
        }
    }
    let mut rounded: Vec<f32> = result.iter().map(|w| *w as f32).collect();
    // The rounding correction refers to the budget that was actually
    // distributed, not to the incoming budget.
    let distributed = effective_budget(budget, weights);
    let mut residual = distributed - rounded.iter().map(|w| f64::from(*w)).sum::<f64>();
    for (i, value) in rounded.iter_mut().enumerate() {
        // Do not revive a zero outside the explicitly specified fallback.
        if result[i] == 0.0 {
            continue;
        }
        let corrected = (f64::from(*value) + residual).clamp(0.0, 2.0) as f32;
        residual -= f64::from(corrected) - f64::from(*value);
        *value = corrected;
    }
    if residual.abs() > 1e-6 + 1e-6 * distributed {
        return Err(CompetitionError(
            "competition budget residual exceeds tolerance",
        ));
    }
    for &w in &rounded {
        valid_weight(w)?;
    }
    Ok(rounded)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn proportional_saturation_keeps_zero_weights_and_rejects_invalid_budgets() {
        assert_eq!(project(&[0.2, 0.4], 1.5).unwrap(), vec![0.5, 1.0]);
        assert_eq!(project(&[2.0, 0.1, 0.1], 4.0).unwrap(), vec![2.0, 1.0, 1.0]);
        assert_eq!(project(&[], 0.0).unwrap(), Vec::<f32>::new());
        for (w, b) in [
            (vec![0.5], 2.1),
            (vec![f32::NAN], 1.0),
            (vec![2.1], 1.0),
            (vec![], 1.0),
        ] {
            assert!(project(&w, b).is_err());
        }
    }

    #[test]
    fn a_surplus_budget_does_not_revive_silenced_synapses() {
        // The budget exceeds the capacity of the positive weights. It is capped
        // instead of being shared with the weights of zero.
        assert_eq!(project(&[1.0, 0.0, 0.0], 3.0).unwrap(), vec![2.0, 0.0, 0.0]);
        assert_eq!(project(&[0.0, 0.0], 1.5).unwrap(), vec![0.0, 0.0]);
        assert_eq!(project(&[1.0, 0.0], 1.5).unwrap(), vec![1.5, 0.0]);
    }

    #[test]
    fn the_distributed_budget_never_exceeds_the_capacity_of_the_positive_weights() {
        assert_eq!(effective_budget(4.0, &[1.0, 0.0, 0.0]), 2.0);
        assert_eq!(effective_budget(1.5, &[1.0, 0.0, 0.0]), 1.5);
        assert_eq!(effective_budget(4.0, &[1.0, 1.0]), 4.0);
        assert_eq!(effective_budget(4.0, &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn deterministic_rounding_conserves_budgets_across_sparse_pools() {
        for n in 1..64 {
            let values: Vec<_> = (0..n)
                .map(|i| {
                    if i % 5 == 0 {
                        0.0
                    } else {
                        (i as f32 + 1.0) / (n as f32 + 1.0)
                    }
                })
                .collect();
            for fraction in [0.0, 0.01, 0.5, 0.99, 1.0] {
                let budget = 2.0 * n as f64 * fraction;
                let a = project(&values, budget).unwrap();
                assert_eq!(a, project(&values, budget).unwrap());
                // The distributed budget is the incoming budget capped at the
                // capacity of the positive weights.
                let distributed = effective_budget(budget, &values);
                assert!(
                    (a.iter().map(|w| f64::from(*w)).sum::<f64>() - distributed).abs()
                        <= 1e-6 + 1e-6 * distributed
                );
                // Zero weights are never revived.
                for (index, weight) in values.iter().enumerate() {
                    if *weight == 0.0 {
                        assert_eq!(a[index], 0.0);
                    }
                }
            }
        }
    }
}
