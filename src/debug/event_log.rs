//! Generic in-memory deterministic event log.

use super::Observer;

/// Append-only copy of immutable observation events.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EventLog<Event> {
    events: Vec<Event>,
}

impl<Event> EventLog<Event> {
    /// Creates an empty log.
    pub const fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Copies an existing event stream in its publication order.
    pub fn from_events(events: impl IntoIterator<Item = Event>) -> Self {
        Self {
            events: events.into_iter().collect(),
        }
    }

    /// Logged events in original publication order.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether no events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Removes diagnostic records without touching simulation state.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Consumes the log and returns its events.
    pub fn into_events(self) -> Vec<Event> {
        self.events
    }
}

impl<Event: Clone> Observer<Event> for EventLog<Event> {
    fn observe(&mut self, event: &Event) {
        self.events.push(event.clone());
    }
}

impl<Event> Extend<Event> for EventLog<Event> {
    fn extend<T: IntoIterator<Item = Event>>(&mut self, iter: T) {
        self.events.extend(iter);
    }
}

impl<Event> FromIterator<Event> for EventLog<Event> {
    fn from_iter<T: IntoIterator<Item = Event>>(iter: T) -> Self {
        Self::from_events(iter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_copies_events_without_reordering_them() {
        let mut log = EventLog::new();
        let first = String::from("first");
        let second = String::from("second");

        log.observe(&first);
        log.observe(&second);

        assert_eq!(log.events(), &["first", "second"]);
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn collection_helpers_preserve_stream_order() {
        let mut log: EventLog<_> = [3, 1].into_iter().collect();
        log.extend([4, 2]);

        assert_eq!(log.events(), &[3, 1, 4, 2]);
        assert_eq!(log.into_events(), vec![3, 1, 4, 2]);
    }

    #[test]
    fn clear_only_changes_the_diagnostic_copy() {
        let source = vec![1, 2];
        let mut log = EventLog::from_events(source.iter().copied());

        log.clear();

        assert!(log.is_empty());
        assert_eq!(source, vec![1, 2]);
    }
}
