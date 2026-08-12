//! Immutable observation boundary.

/// A diagnostics consumer that receives borrowed, read-only observations.
pub trait Observer<Event> {
    /// Observes one event without any path back to simulation state.
    fn observe(&mut self, event: &Event);
}
