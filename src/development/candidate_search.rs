//! Deterministic local search for potential new incoming synapses.
//!
//! A neuron with a positive incoming-growth drive needs more input. The search
//! finds nearby, not-yet-connected cells that could form a new `candidate →
//! target` (incoming) connection. The `target` is the neuron requesting growth;
//! the `candidate` is the prospective presynaptic source.

use std::{error::Error, fmt};

use crate::core::{Network, NeuronId, SimTime};

/// Weights for the deliberately small M1.2 local candidate score.
///
/// The score is `activity_weight * A + temporal_weight * T -
/// distance_weight * d`, where `A` is the product of both current local spike
/// traces and `T` is a two-factor temporal correlation combining spike-time
/// proximity with recency. Redundancy and resource costs are intentionally
/// deferred to later slices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateSearchConfig {
    /// Maximum Euclidean distance at which a source is locally visible.
    pub radius: f32,
    /// Weight of the local activity-trace product.
    pub activity_weight: f32,
    /// Weight of local last-spike temporal correlation.
    pub temporal_weight: f32,
    /// Linear geometric cost per position unit.
    pub distance_weight: f32,
    /// Decay constant in microseconds for the spike-time proximity term.
    pub temporal_tau_us: f32,
    /// Decay constant in microseconds for the recency (age) term.
    pub recency_tau_us: f32,
    /// Minimum total score for a candidate to be eligible; lower scores are
    /// discarded so a "best available" bad candidate is never connected.
    pub min_candidate_score: f32,
}

impl CandidateSearchConfig {
    /// Validates that local score arithmetic is defined and bounded by inputs.
    pub fn validate(&self) -> Result<(), CandidateSearchError> {
        finite_non_negative(self.radius, "candidate_search.radius")?;
        finite_non_negative(self.activity_weight, "candidate_search.activity_weight")?;
        finite_non_negative(self.temporal_weight, "candidate_search.temporal_weight")?;
        finite_non_negative(self.distance_weight, "candidate_search.distance_weight")?;
        finite_non_negative(
            self.min_candidate_score,
            "candidate_search.min_candidate_score",
        )?;
        if !self.temporal_tau_us.is_finite() || self.temporal_tau_us <= 0.0 {
            return Err(CandidateSearchError::NonPositive {
                field: "candidate_search.temporal_tau_us",
                value: self.temporal_tau_us,
            });
        }
        if !self.recency_tau_us.is_finite() || self.recency_tau_us <= 0.0 {
            return Err(CandidateSearchError::NonPositive {
                field: "candidate_search.recency_tau_us",
                value: self.recency_tau_us,
            });
        }
        Ok(())
    }
}

impl Default for CandidateSearchConfig {
    fn default() -> Self {
        Self {
            radius: 1.0,
            activity_weight: 1.0,
            temporal_weight: 1.0,
            distance_weight: 0.1,
            temporal_tau_us: 20_000.0,
            recency_tau_us: 200_000.0,
            min_candidate_score: 0.0,
        }
    }
}

/// One eligible nearby presynaptic source and the local values that ranked it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// Prospective presynaptic source neuron for a new incoming connection.
    pub source: NeuronId,
    /// Euclidean distance from the prospective source to the growth target.
    pub distance: f32,
    /// Product of source and target local activity traces.
    pub activity_correlation: f32,
    /// Two-factor temporal correlation of the two neurons' most recent spikes.
    pub temporal_correlation: f32,
    /// Total score; higher is better.
    pub score: f32,
}

/// Finds all locally visible, not-yet-connected sources in stable score order.
///
/// The `target` is the neuron requesting incoming growth. A candidate `source`
/// is excluded if it is the target itself, or if a directed `source → target`
/// connection already exists. Ties resolve by stable source identity, making
/// the result independent of graph insertion order.
///
/// `now` is the current simulation time, used for the recency factor of the
/// temporal correlation. The caller is responsible for ensuring that neuron
/// traces are current; this function does not advance neuron state.
pub fn local_candidates(
    network: &Network,
    target_id: NeuronId,
    now: SimTime,
    config: &CandidateSearchConfig,
) -> Result<Vec<Candidate>, CandidateSearchError> {
    config.validate()?;
    let target = network
        .neuron(target_id)
        .ok_or(CandidateSearchError::UnknownTarget(target_id))?;
    let incoming: Vec<_> = network
        .incoming_synapse_ids(target_id)
        .iter()
        .map(|id| network.synapse(*id).expect("adjacency ID is valid").pre())
        .collect();

    let mut candidates: Vec<_> = network
        .neurons()
        .filter(|source| source.id() != target_id)
        .filter(|source| !incoming.contains(&source.id()))
        .filter_map(|source| {
            let distance = target.position().distance_to(source.position());
            if distance > config.radius {
                return None;
            }
            let activity_correlation = finite_f32(
                f64::from(source.activity_trace_at(now)) * f64::from(target.activity_trace_at(now)),
            );
            let temporal_correlation = temporal_correlation(
                source.last_spike(),
                target.last_spike(),
                now,
                config.temporal_tau_us,
                config.recency_tau_us,
            );
            let score = finite_f32(
                f64::from(config.activity_weight) * f64::from(activity_correlation)
                    + f64::from(config.temporal_weight) * f64::from(temporal_correlation)
                    - f64::from(config.distance_weight) * f64::from(distance),
            );
            if score < config.min_candidate_score {
                return None;
            }
            Some(Candidate {
                source: source.id(),
                distance,
                activity_correlation,
                temporal_correlation,
                score,
            })
        })
        .collect();
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.source.cmp(&right.source))
    });
    Ok(candidates)
}

/// Two-factor temporal correlation of two neurons' most recent spikes.
///
/// The first factor measures how close the two spikes were in time:
///
/// `exp(-|t_left - t_right| / tau_delta)`
///
/// The second factor measures how recent the more recent of the two spikes is:
///
/// `exp(-(now - max(t_left, t_right)) / tau_recency)`
///
/// Together they ensure that two simultaneous spikes from long ago do not stay
/// at correlation 1 forever, while two simultaneous spikes just now still do.
fn temporal_correlation(
    left: Option<SimTime>,
    right: Option<SimTime>,
    now: SimTime,
    tau_delta_us: f32,
    tau_recency_us: f32,
) -> f32 {
    let Some(left) = left else {
        return 0.0;
    };
    let Some(right) = right else {
        return 0.0;
    };
    let delta_us = if left >= right {
        left.duration_since(right)
            .expect("ordered timestamps have a duration")
    } else {
        right
            .duration_since(left)
            .expect("ordered timestamps have a duration")
    };
    let latest = if left >= right { left } else { right };
    let age_us = now
        .duration_since(latest)
        .expect("now is at least as recent as the latest spike");
    let proximity = (-(delta_us as f64) / f64::from(tau_delta_us)).exp();
    let recency = (-(age_us as f64) / f64::from(tau_recency_us)).exp();
    finite_f32(proximity * recency)
}

fn finite_f32(value: f64) -> f32 {
    value.clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32
}

fn finite_non_negative(value: f32, field: &'static str) -> Result<(), CandidateSearchError> {
    if !value.is_finite() {
        return Err(CandidateSearchError::NonFinite { field, value });
    }
    if value < 0.0 {
        return Err(CandidateSearchError::Negative { field, value });
    }
    Ok(())
}

/// Candidate search could not safely inspect local state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CandidateSearchError {
    /// The requested growth target is absent from the graph.
    UnknownTarget(NeuronId),
    /// A score parameter was NaN or infinite.
    NonFinite { field: &'static str, value: f32 },
    /// A non-negative score parameter was negative.
    Negative { field: &'static str, value: f32 },
    /// A decay constant was zero, negative, NaN, or infinite.
    NonPositive { field: &'static str, value: f32 },
}

impl fmt::Display for CandidateSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTarget(id) => {
                write!(formatter, "unknown candidate-search target {id}")
            }
            Self::NonFinite { field, value } => {
                write!(formatter, "{field} must be finite, got {value}")
            }
            Self::Negative { field, value } => {
                write!(formatter, "{field} must not be negative, got {value}")
            }
            Self::NonPositive { field, value } => {
                write!(formatter, "{field} must be positive, got {value}")
            }
        }
    }
}

impl Error for CandidateSearchError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::NeuronConfig,
        core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
        math::Position3D,
        primitives::Weight,
    };

    fn neuron(id: u64, x: f32) -> Neuron {
        Neuron::new(
            NeuronId(id),
            Position3D::new(x, 0.0, 0.0),
            Polarity::Excitatory,
            None,
            NeuronConfig::default(),
            SimTime::ZERO,
        )
        .unwrap()
    }

    fn network_with_spikes() -> Network {
        let mut network = Network::new();
        network.add_neuron(neuron(1, 0.0)).unwrap();
        network.add_neuron(neuron(2, 0.5)).unwrap();
        network.add_neuron(neuron(3, 3.0)).unwrap();

        for id in [1, 2] {
            network
                .neuron_mut(NeuronId(id))
                .unwrap()
                .integrate_input(SimTime(10), 100.0)
                .unwrap();
        }
        network
            .neuron_mut(NeuronId(3))
            .unwrap()
            .integrate_input(SimTime::ZERO, 100.0)
            .unwrap();
        network
    }

    #[test]
    fn finds_incoming_candidates_for_undersupplied_target() {
        let network = network_with_spikes();
        let candidates = local_candidates(
            &network,
            NeuronId(1),
            SimTime(10),
            &CandidateSearchConfig {
                radius: 2.0,
                activity_weight: 0.0,
                temporal_weight: 1.0,
                distance_weight: 0.0,
                temporal_tau_us: 1.0,
                recency_tau_us: 1_000_000.0,
                min_candidate_score: 0.0,
            },
        )
        .unwrap();

        assert!(candidates.iter().all(|c| c.source != NeuronId(1)));
        assert_eq!(candidates[0].source, NeuronId(2));
    }

    #[test]
    fn excludes_existing_incoming_sources() {
        let mut network = network_with_spikes();
        let synapse = Synapse::new(
            SynapseId(100),
            NeuronId(2),
            NeuronId(1),
            Weight::new(0.1).unwrap(),
            1,
            true,
        )
        .unwrap();
        network.add_synapse(synapse).unwrap();

        let candidates = local_candidates(
            &network,
            NeuronId(1),
            SimTime(10),
            &CandidateSearchConfig::default(),
        )
        .unwrap();

        assert!(candidates.iter().all(|c| c.source != NeuronId(2)));
    }

    #[test]
    fn temporal_correlation_decays_with_age() {
        let network = network_with_spikes();

        let recent = local_candidates(
            &network,
            NeuronId(1),
            SimTime(10),
            &CandidateSearchConfig {
                radius: 2.0,
                activity_weight: 0.0,
                temporal_weight: 1.0,
                distance_weight: 0.0,
                temporal_tau_us: 1.0,
                recency_tau_us: 1_000.0,
                min_candidate_score: 0.0,
            },
        )
        .unwrap();
        let old = local_candidates(
            &network,
            NeuronId(1),
            SimTime(1_000_000),
            &CandidateSearchConfig {
                radius: 2.0,
                activity_weight: 0.0,
                temporal_weight: 1.0,
                distance_weight: 0.0,
                temporal_tau_us: 1.0,
                recency_tau_us: 1_000.0,
                min_candidate_score: 0.0,
            },
        )
        .unwrap();

        let recent_score = recent
            .iter()
            .find(|c| c.source == NeuronId(2))
            .unwrap()
            .temporal_correlation;
        let old_score = old
            .iter()
            .find(|c| c.source == NeuronId(2))
            .unwrap()
            .temporal_correlation;
        assert!(
            recent_score > old_score,
            "recency must reduce old correlations"
        );
    }

    #[test]
    fn min_candidate_score_filters_out_weak_candidates() {
        let network = network_with_spikes();
        let candidates = local_candidates(
            &network,
            NeuronId(1),
            SimTime(10),
            &CandidateSearchConfig {
                radius: 10.0,
                activity_weight: 0.0,
                temporal_weight: 0.0,
                distance_weight: 1.0,
                temporal_tau_us: 1.0,
                recency_tau_us: 1_000_000.0,
                min_candidate_score: 100.0,
            },
        )
        .unwrap();

        assert!(
            candidates.is_empty(),
            "no candidate should beat the threshold"
        );
    }
}
