//! Exact-timestamp event batches used by the deterministic scheduler.

use crate::core::SimTime;

/// One scheduled payload together with its deterministic queue position.
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledEvent<E> {
    /// Simulation timestamp at which the payload becomes due.
    pub execute_at: SimTime,
    /// Monotonic insertion sequence used to order equal-time events.
    pub insertion_sequence: u64,
    /// Runtime-specific event data.
    pub payload: E,
}

/// Every event due at one exact simulation timestamp.
///
/// Events are always stored in ascending insertion-sequence order. Keeping the
/// complete timestamp together prevents callers from accidentally integrating
/// simultaneous inputs one by one.
#[derive(Clone, Debug, PartialEq)]
pub struct EventBatch<E> {
    time: SimTime,
    events: Vec<ScheduledEvent<E>>,
}

impl<E> EventBatch<E> {
    pub(crate) fn from_sorted(time: SimTime, events: Vec<ScheduledEvent<E>>) -> Self {
        debug_assert!(!events.is_empty());
        debug_assert!(events.iter().all(|event| event.execute_at == time));
        debug_assert!(
            events
                .windows(2)
                .all(|pair| pair[0].insertion_sequence < pair[1].insertion_sequence)
        );
        Self { time, events }
    }

    /// Shared timestamp of every event in the batch.
    pub fn time(&self) -> SimTime {
        self.time
    }

    /// Events in deterministic insertion order.
    pub fn events(&self) -> &[ScheduledEvent<E>] {
        &self.events
    }

    /// Number of events in this exact-timestamp batch.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// A scheduler never creates an empty batch.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Consumes the batch and returns its deterministically ordered events.
    pub fn into_events(self) -> Vec<ScheduledEvent<E>> {
        self.events
    }
}
