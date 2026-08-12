//! Deterministic timestamp scheduler without a global simulation tick.

use std::{collections::BTreeMap, error::Error, fmt};

use crate::core::SimTime;

use super::event_batch::{EventBatch, ScheduledEvent};

/// Technical scheduling failures. None of these encode neural behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulerError {
    /// An event was requested before the scheduler's current time.
    TimeWentBackwards {
        /// Timestamp of the most recently removed batch.
        current: SimTime,
        /// Timestamp requested by the caller.
        requested: SimTime,
    },
    /// The timestamp's complete batch has already been removed.
    TimestampAlreadyProcessed {
        /// Timestamp that may no longer accept additional events.
        time: SimTime,
    },
    /// No unique insertion sequence remains.
    InsertionSequenceExhausted,
    /// One exact-timestamp batch exceeds the configured safety bound.
    BatchLimitExceeded {
        /// Timestamp of the oversized batch.
        time: SimTime,
        /// Number of queued events at that timestamp.
        events: usize,
        /// Configured maximum.
        max_events: usize,
    },
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimeWentBackwards { current, requested } => write!(
                formatter,
                "cannot schedule time {} before current time {}",
                requested.as_micros(),
                current.as_micros()
            ),
            Self::TimestampAlreadyProcessed { time } => write!(
                formatter,
                "timestamp {} was already processed as a complete batch",
                time.as_micros()
            ),
            Self::InsertionSequenceExhausted => {
                formatter.write_str("event insertion sequence exhausted")
            }
            Self::BatchLimitExceeded {
                time,
                events,
                max_events,
            } => write!(
                formatter,
                "timestamp {} contains {events} events, exceeding limit {max_events}",
                time.as_micros()
            ),
        }
    }
}

impl Error for SchedulerError {}

/// A deterministic priority queue ordered by `(timestamp, insertion_sequence)`.
#[derive(Debug)]
pub struct EventScheduler<E> {
    /// Events are keyed by their two deterministic ordering keys. A map, in
    /// contrast to a binary heap with tombstones, lets a superseded predicted
    /// event be removed immediately and releases its payload memory.
    events: BTreeMap<(SimTime, u64), E>,
    /// Locates an insertion sequence for `cancel` without changing the
    /// externally stable sequence returned by `schedule`.
    sequence_times: BTreeMap<u64, SimTime>,
    next_sequence: u64,
    current_time: SimTime,
    last_processed_time: Option<SimTime>,
}

impl<E> Default for EventScheduler<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> EventScheduler<E> {
    /// Creates an empty scheduler at simulation time zero.
    pub fn new() -> Self {
        Self {
            events: BTreeMap::new(),
            sequence_times: BTreeMap::new(),
            next_sequence: 0,
            current_time: SimTime::ZERO,
            last_processed_time: None,
        }
    }

    /// Current scheduler time, advanced only when a batch is removed.
    pub fn current_time(&self) -> SimTime {
        self.current_time
    }

    /// Timestamp of the next due batch without removing it.
    pub fn next_time(&self) -> Option<SimTime> {
        self.events.first_key_value().map(|((time, _), _)| *time)
    }

    /// Number of queued events across all timestamps.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether no event is waiting.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns whether any queued payload satisfies `predicate`.
    ///
    /// This read-only inspection is useful for bounded experiment horizons
    /// whose recurring local maintenance events are expected to remain queued.
    pub(crate) fn any_payload(&self, mut predicate: impl FnMut(&E) -> bool) -> bool {
        self.events.values().any(&mut predicate)
    }

    /// Inserts an event and returns its globally stable insertion sequence.
    pub fn schedule(&mut self, execute_at: SimTime, payload: E) -> Result<u64, SchedulerError> {
        if execute_at < self.current_time {
            return Err(SchedulerError::TimeWentBackwards {
                current: self.current_time,
                requested: execute_at,
            });
        }
        if self.last_processed_time == Some(execute_at) {
            return Err(SchedulerError::TimestampAlreadyProcessed { time: execute_at });
        }

        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(SchedulerError::InsertionSequenceExhausted)?;
        self.events.insert((execute_at, sequence), payload);
        self.sequence_times.insert(sequence, execute_at);
        Ok(sequence)
    }

    /// Removes one still-queued event by the insertion sequence returned from
    /// [`Self::schedule`]. Returns `false` when that event was already
    /// processed or had previously been cancelled.
    ///
    /// Cancellation is eager: the event and its payload leave the queue now,
    /// rather than becoming a stale heap entry that must wait for its old
    /// timestamp before being discarded.
    pub fn cancel(&mut self, insertion_sequence: u64) -> bool {
        let Some(time) = self.sequence_times.remove(&insertion_sequence) else {
            return false;
        };
        self.events.remove(&(time, insertion_sequence)).is_some()
    }

    /// Removes every event at the earliest timestamp as one atomic batch.
    ///
    /// The limit is checked before anything is removed, so callers can inspect
    /// or replace an unsuitable configuration without losing queued events.
    pub fn pop_next_batch(
        &mut self,
        max_events: usize,
    ) -> Result<Option<EventBatch<E>>, SchedulerError> {
        let Some(time) = self.next_time() else {
            return Ok(None);
        };

        let first_key = (time, 0);
        let last_key = (time, u64::MAX);
        let event_count = self.events.range(first_key..=last_key).count();
        if event_count > max_events {
            return Err(SchedulerError::BatchLimitExceeded {
                time,
                events: event_count,
                max_events,
            });
        }

        let keys: Vec<_> = self
            .events
            .range(first_key..=last_key)
            .map(|(key, _)| *key)
            .collect();
        let mut events = Vec::with_capacity(event_count);
        for (execute_at, insertion_sequence) in keys {
            let payload = self
                .events
                .remove(&(execute_at, insertion_sequence))
                .expect("ranged queued event must remain present");
            self.sequence_times.remove(&insertion_sequence);
            events.push(ScheduledEvent {
                execute_at,
                insertion_sequence,
                payload,
            });
        }
        self.current_time = time;
        self.last_processed_time = Some(time);

        Ok(Some(EventBatch::from_sorted(time, events)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_batches_by_time_and_events_by_insertion_sequence() {
        let mut scheduler = EventScheduler::new();
        scheduler.schedule(SimTime(20), 'a').unwrap();
        scheduler.schedule(SimTime(10), 'b').unwrap();
        scheduler.schedule(SimTime(20), 'c').unwrap();

        let first = scheduler.pop_next_batch(10).unwrap().unwrap();
        assert_eq!(first.time(), SimTime(10));
        assert_eq!(first.events()[0].payload, 'b');

        let second = scheduler.pop_next_batch(10).unwrap().unwrap();
        assert_eq!(second.time(), SimTime(20));
        assert_eq!(
            second
                .events()
                .iter()
                .map(|event| event.payload)
                .collect::<Vec<_>>(),
            vec!['a', 'c']
        );
    }

    #[test]
    fn oversized_batch_is_left_intact() {
        let mut scheduler = EventScheduler::new();
        scheduler.schedule(SimTime(5), 1).unwrap();
        scheduler.schedule(SimTime(5), 2).unwrap();

        assert_eq!(
            scheduler.pop_next_batch(1),
            Err(SchedulerError::BatchLimitExceeded {
                time: SimTime(5),
                events: 2,
                max_events: 1,
            })
        );
        assert_eq!(scheduler.len(), 2);
        assert_eq!(scheduler.current_time(), SimTime::ZERO);
    }

    #[test]
    fn rejects_events_before_current_time() {
        let mut scheduler = EventScheduler::new();
        scheduler.schedule(SimTime(10), ()).unwrap();
        scheduler.pop_next_batch(1).unwrap();

        assert_eq!(
            scheduler.schedule(SimTime(9), ()),
            Err(SchedulerError::TimeWentBackwards {
                current: SimTime(10),
                requested: SimTime(9),
            })
        );
    }

    #[test]
    fn rejects_late_addition_to_an_already_completed_timestamp() {
        let mut scheduler = EventScheduler::new();
        scheduler.schedule(SimTime::ZERO, ()).unwrap();
        scheduler.pop_next_batch(1).unwrap();

        assert_eq!(
            scheduler.schedule(SimTime::ZERO, ()),
            Err(SchedulerError::TimestampAlreadyProcessed {
                time: SimTime::ZERO,
            })
        );
    }

    #[test]
    fn cancellation_eagerly_removes_a_superseded_event() {
        let mut scheduler = EventScheduler::new();
        let cancelled = scheduler.schedule(SimTime(10), 'a').unwrap();
        scheduler.schedule(SimTime(20), 'b').unwrap();

        assert!(scheduler.cancel(cancelled));
        assert!(!scheduler.cancel(cancelled));
        assert_eq!(scheduler.len(), 1);
        assert_eq!(scheduler.next_time(), Some(SimTime(20)));

        let batch = scheduler.pop_next_batch(10).unwrap().unwrap();
        assert_eq!(batch.events()[0].payload, 'b');
    }
}
