//! Deterministic timestamp scheduler without a global simulation tick.

use std::{cmp::Ordering, collections::BinaryHeap, error::Error, fmt};

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
    heap: BinaryHeap<HeapEntry<E>>,
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
            heap: BinaryHeap::new(),
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
        self.heap.peek().map(|entry| entry.execute_at)
    }

    /// Number of queued events across all timestamps.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether no event is waiting.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
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
        self.heap.push(HeapEntry {
            execute_at,
            insertion_sequence: sequence,
            payload,
        });
        Ok(sequence)
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

        let event_count = self
            .heap
            .iter()
            .filter(|entry| entry.execute_at == time)
            .count();
        if event_count > max_events {
            return Err(SchedulerError::BatchLimitExceeded {
                time,
                events: event_count,
                max_events,
            });
        }

        let mut events = Vec::with_capacity(event_count);
        while self.next_time() == Some(time) {
            let entry = self.heap.pop().expect("peeked event must remain present");
            events.push(ScheduledEvent {
                execute_at: entry.execute_at,
                insertion_sequence: entry.insertion_sequence,
                payload: entry.payload,
            });
        }
        self.current_time = time;
        self.last_processed_time = Some(time);

        Ok(Some(EventBatch::from_sorted(time, events)))
    }
}

/// `BinaryHeap` is a max-heap, so ordering is deliberately reversed for the two
/// scheduling keys. Payloads never participate in ordering.
#[derive(Debug)]
struct HeapEntry<E> {
    execute_at: SimTime,
    insertion_sequence: u64,
    payload: E,
}

impl<E> PartialEq for HeapEntry<E> {
    fn eq(&self, other: &Self) -> bool {
        (self.execute_at, self.insertion_sequence) == (other.execute_at, other.insertion_sequence)
    }
}

impl<E> Eq for HeapEntry<E> {}

impl<E> PartialOrd for HeapEntry<E> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E> Ord for HeapEntry<E> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .execute_at
            .cmp(&self.execute_at)
            .then_with(|| other.insertion_sequence.cmp(&self.insertion_sequence))
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
}
