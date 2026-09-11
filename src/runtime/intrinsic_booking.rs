//! Runtime-owned bookings for predicted autonomous neuron spikes.

use std::collections::BTreeMap;

use crate::core::{NeuronId, SimTime};

/// One queued intrinsic threshold-crossing prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IntrinsicBooking {
    /// Timestamp at which the crossing is predicted.
    pub(crate) time: SimTime,
    /// Scheduler insertion sequence, used for eager cancellation.
    pub(crate) sequence: u64,
}

/// Runtime-only index of one autonomous-spike booking per neuron.
///
/// Scheduling belongs to the executor, not to a neuron. Keeping this map here
/// prevents core cell state from depending on scheduler internals.
#[derive(Clone, Debug, Default)]
pub(crate) struct IntrinsicBookings(BTreeMap<NeuronId, IntrinsicBooking>);

impl IntrinsicBookings {
    pub(crate) fn get(&self, neuron_id: NeuronId) -> Option<IntrinsicBooking> {
        self.0.get(&neuron_id).copied()
    }

    pub(crate) fn insert(&mut self, neuron_id: NeuronId, booking: IntrinsicBooking) {
        let previous = self.0.insert(neuron_id, booking);
        debug_assert!(
            previous.is_none(),
            "intrinsic booking replaced without cancellation"
        );
    }

    pub(crate) fn remove(&mut self, neuron_id: NeuronId) -> Option<IntrinsicBooking> {
        self.0.remove(&neuron_id)
    }

    pub(crate) fn contains(&self, neuron_id: NeuronId, time: SimTime) -> bool {
        self.get(neuron_id)
            .is_some_and(|booking| booking.time == time)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
