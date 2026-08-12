//! Environment contract and deterministic M0 sequence source.

use std::collections::VecDeque;

use crate::core::SimTime;

/// The four sparse input patterns used by the M0 experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Pattern {
    /// First item of the learned sequence.
    A,
    /// Second item of the learned sequence.
    B,
    /// Third item of the learned sequence.
    C,
    /// Fourth item of the learned sequence.
    D,
}

/// One timestamped value offered by an environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observation {
    /// A symbolic M0 pattern. Translation into spikes happens in `transduction`.
    Pattern { at: SimTime, pattern: Pattern },
    /// A binary sensor value emitted by [`super::BitWorld`].
    Bit { at: SimTime, value: bool },
}

impl Observation {
    /// Timestamp at which this observation becomes available.
    pub fn time(&self) -> SimTime {
        match *self {
            Self::Pattern { at, .. } | Self::Bit { at, .. } => at,
        }
    }
}

/// An action decoded from motor spikes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Set the binary actuator to a value.
    SetBit(bool),
    /// No externally visible change.
    NoOp,
}

/// External systems are deliberately unaware of neurons and synapses.
pub trait Environment {
    /// Timestamp of the next externally available observation, if known.
    ///
    /// This event-time peek lets a closed-loop adapter interleave environment
    /// changes with neural and motor events without introducing a global tick.
    fn next_observation_time(&self) -> Option<SimTime>;

    /// Removes and returns all observations whose timestamp is at most `until`.
    fn observations(&mut self, until: SimTime) -> Vec<Observation>;

    /// Applies one decoded action to the environment.
    fn apply_action(&mut self, action: Action);

    /// Advances an environment-local timestamp before timestamped actions are
    /// applied. Environments without time-dependent observations may keep the
    /// default no-op implementation.
    fn advance_to(&mut self, time: SimTime) {
        let _ = time;
    }
}

/// A deterministic, precomputed input source for sequence experiments.
#[derive(Clone, Debug, Default)]
pub struct SequenceEnvironment {
    pending: VecDeque<Observation>,
    applied_actions: Vec<Action>,
}

impl SequenceEnvironment {
    /// Builds a source from timestamped patterns. Entries are sorted by time while
    /// preserving their original order for equal timestamps.
    pub fn new(patterns: impl IntoIterator<Item = (SimTime, Pattern)>) -> Self {
        let mut indexed: Vec<_> = patterns
            .into_iter()
            .enumerate()
            .map(|(sequence, (at, pattern))| (at, sequence, pattern))
            .collect();
        indexed.sort_by_key(|(at, sequence, _)| (*at, *sequence));

        Self {
            pending: indexed
                .into_iter()
                .map(|(at, _, pattern)| Observation::Pattern { at, pattern })
                .collect(),
            applied_actions: Vec::new(),
        }
    }

    /// Returns actions received so far, for diagnostics and assertions.
    pub fn applied_actions(&self) -> &[Action] {
        &self.applied_actions
    }

    /// Returns whether all scheduled observations were consumed.
    pub fn is_exhausted(&self) -> bool {
        self.pending.is_empty()
    }
}

impl Environment for SequenceEnvironment {
    fn next_observation_time(&self) -> Option<SimTime> {
        self.pending.front().map(Observation::time)
    }

    fn observations(&mut self, until: SimTime) -> Vec<Observation> {
        let count = self
            .pending
            .iter()
            .take_while(|observation| observation.time() <= until)
            .count();
        self.pending.drain(..count).collect()
    }

    fn apply_action(&mut self, action: Action) {
        self.applied_actions.push(action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_source_releases_only_due_observations() {
        let mut source = SequenceEnvironment::new([
            (SimTime(20), Pattern::B),
            (SimTime(10), Pattern::A),
            (SimTime(30), Pattern::C),
        ]);

        assert_eq!(
            source.observations(SimTime(20)),
            vec![
                Observation::Pattern {
                    at: SimTime(10),
                    pattern: Pattern::A,
                },
                Observation::Pattern {
                    at: SimTime(20),
                    pattern: Pattern::B,
                },
            ]
        );
        assert!(!source.is_exhausted());
    }
}
