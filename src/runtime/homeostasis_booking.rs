//! Runtime-owned bookings for recurring local homeostasis maintenance.

use std::collections::BTreeMap;

use crate::core::{NeuronId, SimTime};

/// One queued recurring local-maintenance event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HomeostasisBooking {
    /// Timestamp at which maintenance will next run.
    pub(crate) time: SimTime,
    /// Scheduler insertion sequence, used for eager cancellation.
    pub(crate) sequence: u64,
}

/// Runtime-only index of one recurring maintenance booking per neuron.
#[derive(Clone, Debug, Default)]
pub(crate) struct HomeostasisBookings(BTreeMap<NeuronId, HomeostasisBooking>);

impl HomeostasisBookings {
    pub(crate) fn get(&self, neuron_id: NeuronId) -> Option<HomeostasisBooking> {
        self.0.get(&neuron_id).copied()
    }

    pub(crate) fn insert(&mut self, neuron_id: NeuronId, booking: HomeostasisBooking) {
        let previous = self.0.insert(neuron_id, booking);
        debug_assert!(
            previous.is_none(),
            "homeostasis booking replaced without cancellation"
        );
    }

    pub(crate) fn remove(&mut self, neuron_id: NeuronId) -> Option<HomeostasisBooking> {
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
