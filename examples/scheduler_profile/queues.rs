//! Benchmark-only timestamp buckets; the production scheduler stays unchanged.
use std::collections::BTreeMap;

use nerva::{
    core::SimTime,
    runtime::{EventScheduler, ScheduledEvent, SchedulerError},
};

pub trait Queue<E>: Default {
    fn schedule(&mut self, time: SimTime, payload: E) -> Result<u64, SchedulerError>;
    fn cancel(&mut self, sequence: u64) -> bool;
    fn cancel_matching(&mut self, predicate: impl FnMut(&E) -> bool) -> usize;
    fn pop(&mut self, limit: usize) -> Result<Option<Vec<ScheduledEvent<E>>>, SchedulerError>;
    fn state(&self) -> (usize, SimTime, Option<SimTime>);
}

impl<E> Queue<E> for EventScheduler<E> {
    fn schedule(&mut self, time: SimTime, payload: E) -> Result<u64, SchedulerError> {
        self.schedule(time, payload)
    }

    fn cancel(&mut self, sequence: u64) -> bool {
        self.cancel(sequence)
    }

    fn cancel_matching(&mut self, predicate: impl FnMut(&E) -> bool) -> usize {
        self.cancel_matching(predicate)
    }

    fn pop(&mut self, limit: usize) -> Result<Option<Vec<ScheduledEvent<E>>>, SchedulerError> {
        self.pop_next_batch(limit)
            .map(|batch| batch.map(|batch| batch.into_events()))
    }

    fn state(&self) -> (usize, SimTime, Option<SimTime>) {
        (self.len(), self.current_time(), self.next_time())
    }
}

/// Groups equal-time events so extraction removes one bucket instead of
/// searching/removing every event from the outer tree. Both indices remain
/// ordered trees; cancellation frees the payload immediately, with no tombstones.
pub struct BucketQueue<E> {
    buckets: BTreeMap<SimTime, BTreeMap<u64, E>>,
    sequence_times: BTreeMap<u64, SimTime>,
    next_sequence: u64,
    current_time: SimTime,
    last_processed_time: Option<SimTime>,
}

impl<E> Default for BucketQueue<E> {
    fn default() -> Self {
        Self {
            buckets: BTreeMap::new(),
            sequence_times: BTreeMap::new(),
            next_sequence: 0,
            current_time: SimTime::ZERO,
            last_processed_time: None,
        }
    }
}

impl<E> Queue<E> for BucketQueue<E> {
    fn schedule(&mut self, time: SimTime, payload: E) -> Result<u64, SchedulerError> {
        if time < self.current_time {
            return Err(SchedulerError::TimeWentBackwards {
                current: self.current_time,
                requested: time,
            });
        }
        if self.last_processed_time == Some(time) {
            return Err(SchedulerError::TimestampAlreadyProcessed { time });
        }
        let sequence = self.next_sequence;
        self.next_sequence = sequence
            .checked_add(1)
            .ok_or(SchedulerError::InsertionSequenceExhausted)?;
        self.buckets
            .entry(time)
            .or_default()
            .insert(sequence, payload);
        self.sequence_times.insert(sequence, time);
        Ok(sequence)
    }

    fn cancel(&mut self, sequence: u64) -> bool {
        let Some(time) = self.sequence_times.remove(&sequence) else {
            return false;
        };
        let bucket = self.buckets.get_mut(&time).expect("indexed bucket exists");
        let removed = bucket.remove(&sequence).is_some();
        if bucket.is_empty() {
            self.buckets.remove(&time);
        }
        removed
    }

    fn cancel_matching(&mut self, mut predicate: impl FnMut(&E) -> bool) -> usize {
        // Match the production predicate visitation order, including its
        // collect-before-mutate behavior for stateful predicates.
        let sequences: Vec<_> = self
            .buckets
            .values()
            .flat_map(|bucket| bucket.iter())
            .filter_map(|(&sequence, payload)| predicate(payload).then_some(sequence))
            .collect();
        let count = sequences.len();
        for sequence in sequences {
            let cancelled = self.cancel(sequence);
            debug_assert!(cancelled);
        }
        count
    }

    fn pop(&mut self, limit: usize) -> Result<Option<Vec<ScheduledEvent<E>>>, SchedulerError> {
        let Some((&time, bucket)) = self.buckets.first_key_value() else {
            return Ok(None);
        };
        if bucket.len() > limit {
            return Err(SchedulerError::BatchLimitExceeded {
                time,
                events: bucket.len(),
                max_events: limit,
            });
        }
        let (_, bucket) = self.buckets.pop_first().expect("first bucket exists");
        let mut events = Vec::with_capacity(bucket.len());
        for (sequence, payload) in bucket {
            self.sequence_times.remove(&sequence);
            events.push(ScheduledEvent {
                execute_at: time,
                insertion_sequence: sequence,
                payload,
            });
        }
        self.current_time = time;
        self.last_processed_time = Some(time);
        Ok(Some(events))
    }

    fn state(&self) -> (usize, SimTime, Option<SimTime>) {
        (
            self.sequence_times.len(),
            self.current_time,
            self.buckets.first_key_value().map(|(&time, _)| time),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn bucket_errors_preserve_queue_and_sequence() {
        let mut queue = BucketQueue::default();
        assert_eq!(queue.schedule(SimTime(10), 1), Ok(0));
        assert_eq!(queue.schedule(SimTime(10), 2), Ok(1));
        assert_eq!(
            queue.pop(1),
            Err(SchedulerError::BatchLimitExceeded {
                time: SimTime(10),
                events: 2,
                max_events: 1,
            })
        );
        assert_eq!(queue.state(), (2, SimTime::ZERO, Some(SimTime(10))));
        assert_eq!(queue.pop(2).unwrap().unwrap().len(), 2);
        assert_eq!(
            queue.schedule(SimTime(9), 3),
            Err(SchedulerError::TimeWentBackwards {
                current: SimTime(10),
                requested: SimTime(9),
            })
        );
        assert_eq!(
            queue.schedule(SimTime(10), 3),
            Err(SchedulerError::TimestampAlreadyProcessed { time: SimTime(10) })
        );
        assert_eq!(queue.schedule(SimTime(11), 3), Ok(2));
        queue.next_sequence = u64::MAX;
        assert_eq!(
            queue.schedule(SimTime(12), 4),
            Err(SchedulerError::InsertionSequenceExhausted)
        );
        assert_eq!(queue.state(), (1, SimTime(10), Some(SimTime(11))));
        assert_eq!(queue.next_sequence, u64::MAX);
    }

    #[test]
    fn both_queues_release_cancelled_payloads_immediately() {
        fn check<Q: Queue<Rc<()>>>() {
            let payload = Rc::new(());
            let mut queue = Q::default();
            let sequence = queue.schedule(SimTime(100), payload.clone()).unwrap();
            assert_eq!(Rc::strong_count(&payload), 2);
            assert!(queue.cancel(sequence));
            assert_eq!(Rc::strong_count(&payload), 1);
            assert_eq!(queue.state(), (0, SimTime::ZERO, None));
            assert!(!queue.cancel(sequence));
            queue.schedule(SimTime(101), payload.clone()).unwrap();
            assert_eq!(queue.cancel_matching(|_| true), 1);
            assert_eq!(Rc::strong_count(&payload), 1);
            assert!(queue.pop(0).unwrap().is_none());
        }
        check::<EventScheduler<Rc<()>>>();
        check::<BucketQueue<Rc<()>>>();
    }

    #[test]
    fn stateful_cancellation_visits_the_same_order() {
        fn check<Q: Queue<u64>>() -> (Vec<u64>, Vec<ScheduledEvent<u64>>) {
            let mut queue = Q::default();
            for (time, payload) in [(30, 0), (10, 1), (30, 2), (20, 3), (10, 4)] {
                queue.schedule(SimTime(time), payload).unwrap();
            }
            let mut visited = Vec::new();
            assert_eq!(
                queue.cancel_matching(|&payload| {
                    visited.push(payload);
                    visited.len().is_multiple_of(2)
                }),
                2
            );
            let mut delivered = Vec::new();
            while let Some(batch) = queue.pop(10).unwrap() {
                delivered.extend(batch);
            }
            (visited, delivered)
        }
        let expected = check::<EventScheduler<u64>>();
        assert_eq!(expected.0, [1, 4, 3, 0, 2]);
        assert_eq!(check::<BucketQueue<u64>>(), expected);
    }
}
