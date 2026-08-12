//! Timestamp-window measurements for the frozen A-only M0 probe.

use std::{error::Error, fmt};

use crate::{core::SimTime, environment::Pattern};

/// Invalid temporal scoring parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceMetricError {
    /// The expected inter-state delay must be nonzero.
    ZeroDelay,
    /// Computing an expected timestamp exceeded simulation time.
    TimeOverflow,
    /// A signed latency did not fit in the public microsecond representation.
    LatencyOutOfRange,
}

impl fmt::Display for SequenceMetricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDelay => formatter.write_str("transition delay must be greater than zero"),
            Self::TimeOverflow => {
                formatter.write_str("expected probe timestamp exceeds simulation time")
            }
            Self::LatencyOutOfRange => {
                formatter.write_str("probe latency does not fit in signed microseconds")
            }
        }
    }
}

impl Error for SequenceMetricError {}

/// Expected sequence activations versus off-sequence activations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SequenceMetrics {
    /// Expected B, C and D hits inside their individual time windows.
    pub transition_hits: usize,
    /// Unexpected, duplicate, simultaneous, early, or late activations.
    pub false_transitions: usize,
    /// Signed `observed - expected` latency for accepted hits.
    pub latency_errors_us: Vec<i64>,
    /// Observed classified activations in timestamp order.
    pub observed: Vec<(SimTime, Pattern)>,
}

impl SequenceMetrics {
    /// Compatibility helper for order-only callers outside M0 experiments.
    /// M0 itself uses [`Self::from_timed_probe`] and therefore enforces latency.
    pub fn from_probe(observed: impl IntoIterator<Item = (SimTime, Pattern)>) -> Self {
        let mut observed: Vec<_> = observed.into_iter().collect();
        observed.sort_by_key(|(time, pattern)| (*time, *pattern));
        let expected = [Pattern::B, Pattern::C, Pattern::D];
        let mut expected_index = 0;
        let mut hits = 0;
        let mut misses = 0;
        for &(_, pattern) in &observed {
            if expected.get(expected_index) == Some(&pattern) {
                hits += 1;
                expected_index += 1;
            } else if pattern != Pattern::A {
                misses += 1;
            }
        }
        Self {
            transition_hits: hits,
            false_transitions: misses,
            latency_errors_us: Vec::new(),
            observed,
        }
    }

    /// Scores B→C→D around `cue + n × transition_delay_us`.
    ///
    /// A cue spikes are ignored. Every non-cue event may satisfy only its own
    /// pattern window once; all other non-cue events count as false transitions.
    pub fn from_timed_probe(
        observed: impl IntoIterator<Item = (SimTime, Pattern)>,
        cue_time: SimTime,
        transition_delay_us: u64,
        tolerance_us: u64,
    ) -> Result<Self, SequenceMetricError> {
        if transition_delay_us == 0 {
            return Err(SequenceMetricError::ZeroDelay);
        }
        let mut observed: Vec<_> = observed.into_iter().collect();
        observed.sort_by_key(|(time, pattern)| (*time, *pattern));
        let mut expected = Vec::new();
        for (step, pattern) in [Pattern::B, Pattern::C, Pattern::D].into_iter().enumerate() {
            let offset = transition_delay_us
                .checked_mul((step + 1) as u64)
                .ok_or(SequenceMetricError::TimeOverflow)?;
            let time = cue_time
                .checked_add_us(offset)
                .ok_or(SequenceMetricError::TimeOverflow)?;
            expected.push((time, pattern));
        }

        let mut matched = [false; 3];
        let mut cue_ignored = false;
        let mut hits = 0;
        let mut misses = 0;
        let mut latency_errors_us = Vec::new();
        for &(time, pattern) in &observed {
            if pattern == Pattern::A {
                if !cue_ignored && time == cue_time {
                    cue_ignored = true;
                } else {
                    misses += 1;
                }
                continue;
            }
            let Some(index) = [Pattern::B, Pattern::C, Pattern::D]
                .iter()
                .position(|&candidate| candidate == pattern)
            else {
                misses += 1;
                continue;
            };
            let expected_time = expected[index].0;
            let error = i128::from(time.as_micros()) - i128::from(expected_time.as_micros());
            let inside = error.unsigned_abs() <= u128::from(tolerance_us);
            if inside && !matched[index] {
                matched[index] = true;
                hits += 1;
                latency_errors_us.push(
                    i64::try_from(error).map_err(|_| SequenceMetricError::LatencyOutOfRange)?,
                );
            } else {
                misses += 1;
            }
        }

        Ok(Self {
            transition_hits: hits,
            false_transitions: misses,
            latency_errors_us,
            observed,
        })
    }

    /// Hits minus false transitions.
    pub fn score(&self) -> isize {
        self.transition_hits as isize - self.false_transitions as isize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_only_hits_inside_their_pattern_specific_windows() {
        let metrics = SequenceMetrics::from_timed_probe(
            [
                (SimTime(100), Pattern::A),
                (SimTime(111), Pattern::B),
                (SimTime(120), Pattern::C),
                (SimTime(129), Pattern::D),
            ],
            SimTime(100),
            10,
            1,
        )
        .unwrap();

        assert_eq!(metrics.transition_hits, 3);
        assert_eq!(metrics.false_transitions, 0);
        assert_eq!(metrics.latency_errors_us, vec![1, 0, -1]);
    }

    #[test]
    fn simultaneous_late_and_duplicate_patterns_are_false() {
        let metrics = SequenceMetrics::from_timed_probe(
            [
                (SimTime(110), Pattern::B),
                (SimTime(110), Pattern::C),
                (SimTime(110), Pattern::B),
                (SimTime(999), Pattern::D),
            ],
            SimTime(100),
            10,
            0,
        )
        .unwrap();

        assert_eq!(metrics.transition_hits, 1);
        assert_eq!(metrics.false_transitions, 3);
        assert_eq!(metrics.score(), -2);
    }

    #[test]
    fn ignores_only_the_actual_cue_and_counts_recurrent_a_spikes() {
        let metrics = SequenceMetrics::from_timed_probe(
            [
                (SimTime(100), Pattern::A),
                (SimTime(105), Pattern::A),
                (SimTime(110), Pattern::B),
            ],
            SimTime(100),
            10,
            0,
        )
        .unwrap();

        assert_eq!(metrics.transition_hits, 1);
        assert_eq!(metrics.false_transitions, 1);
    }

    #[test]
    fn compatibility_scoring_orders_timestamped_input() {
        let metrics = SequenceMetrics::from_probe([
            (SimTime(30), Pattern::D),
            (SimTime(10), Pattern::B),
            (SimTime(20), Pattern::C),
        ]);

        assert_eq!(metrics.transition_hits, 3);
        assert_eq!(metrics.false_transitions, 0);
        assert_eq!(metrics.observed[0], (SimTime(10), Pattern::B));
    }

    #[test]
    fn extreme_latency_returns_an_error_instead_of_panicking() {
        let result = SequenceMetrics::from_timed_probe(
            [(SimTime(u64::MAX), Pattern::B)],
            SimTime::ZERO,
            1,
            u64::MAX,
        );

        assert_eq!(result, Err(SequenceMetricError::LatencyOutOfRange));
    }
}
