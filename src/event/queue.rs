//! Deterministic queue for timestamped events.

use crate::common::SimTimeUs;

use super::types::{EventPayload, ScheduledEvent};

/// Events ordered first by timestamp and then by their insertion sequence.
#[derive(Debug, Default)]
pub struct EventQueue {
    /// Ordered scheduled events waiting to execute.
    events: Vec<ScheduledEvent>,
    /// Sequence number assigned to the next inserted event.
    next_sequence: u64,
}

impl EventQueue {
    /// Creates an empty event queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Schedules `payload` for `execute_at` and returns its stable sequence number.
    ///
    /// Events with equal timestamps execute in insertion order.
    pub fn schedule(&mut self, execute_at: SimTimeUs, payload: EventPayload) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.events.push(ScheduledEvent {
            execute_at,
            sequence,
            payload,
        });
        self.events
            .sort_by_key(|event| (event.execute_at, event.sequence));

        sequence
    }

    /// Returns the next event without removing it.
    pub fn peek(&self) -> Option<&ScheduledEvent> {
        self.events.first()
    }

    /// Removes and returns the next event in deterministic execution order.
    pub fn pop_next(&mut self) -> Option<ScheduledEvent> {
        (!self.events.is_empty()).then(|| self.events.remove(0))
    }

    /// Returns the number of scheduled events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns whether no events are scheduled.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        common::{NeuronId, SynapseId},
        event::{EventPayload, SynapticInputEvent},
    };

    use super::*;

    fn input_event() -> EventPayload {
        EventPayload::SynapticInput(SynapticInputEvent {
            synapse: SynapseId(1),
            source: NeuronId(1),
            target: NeuronId(2),
            arrives_at: SimTimeUs(20),
            input_value: 0.5,
        })
    }

    #[test]
    fn orders_events_by_time_then_insertion_sequence() {
        let mut queue = EventQueue::new();
        queue.schedule(SimTimeUs(20), input_event());
        queue.schedule(SimTimeUs(10), input_event());
        queue.schedule(SimTimeUs(20), input_event());

        let first = queue.pop_next().expect("first event");
        let second = queue.pop_next().expect("second event");
        let third = queue.pop_next().expect("third event");

        assert_eq!((first.execute_at, first.sequence), (SimTimeUs(10), 1));
        assert_eq!((second.execute_at, second.sequence), (SimTimeUs(20), 0));
        assert_eq!((third.execute_at, third.sequence), (SimTimeUs(20), 2));
        assert!(queue.is_empty());
    }
}
