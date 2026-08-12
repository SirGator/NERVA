//! Boundary reserved for M1 organizer, field, differentiation and growth logic.
//!
//! M0 intentionally exposes no developmental behavior. Enabling this feature
//! only makes the architectural boundary explicit without altering simulation.

/// Marker proving that developmental mechanisms are disabled for M0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevelopmentDisabled;
