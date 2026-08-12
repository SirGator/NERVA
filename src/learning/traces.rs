//! Analytically decaying local spike traces.

use crate::core::SimTime;

/// Failure while updating a local trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TraceError {
    /// A trace cannot be evaluated before its latest update.
    TimeWentBackwards {
        /// Timestamp currently stored by the trace.
        current: SimTime,
        /// Timestamp requested by the caller.
        requested: SimTime,
    },
    /// A decay time constant must be finite and strictly positive.
    InvalidTimeConstant,
    /// Trace impulses must be finite.
    NonFiniteImpulse,
}

/// A scalar trace with exact exponential decay between events.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecayingTrace {
    value: f32,
    updated_at: SimTime,
}

impl DecayingTrace {
    /// Creates a zero-valued trace at `start_time`.
    pub fn new(start_time: SimTime) -> Self {
        Self {
            value: 0.0,
            updated_at: start_time,
        }
    }

    /// Returns the trace's value at its latest update.
    pub fn value(self) -> f32 {
        self.value
    }

    /// Returns the timestamp of the latest update.
    pub fn updated_at(self) -> SimTime {
        self.updated_at
    }

    /// Decays the trace to `time` and returns the new value.
    ///
    /// On error the trace remains unchanged.
    pub fn advance_to(&mut self, time: SimTime, tau_us: f32) -> Result<f32, TraceError> {
        validate_tau(tau_us)?;
        let elapsed_us =
            time.duration_since(self.updated_at)
                .ok_or(TraceError::TimeWentBackwards {
                    current: self.updated_at,
                    requested: time,
                })?;

        self.value *= decay_factor(elapsed_us, tau_us);
        self.updated_at = time;
        Ok(self.value)
    }

    /// Decays to `time`, adds a local impulse, and returns the new value.
    ///
    /// On error the trace remains unchanged.
    pub fn add_impulse(
        &mut self,
        time: SimTime,
        tau_us: f32,
        impulse: f32,
    ) -> Result<f32, TraceError> {
        if !impulse.is_finite() {
            return Err(TraceError::NonFiniteImpulse);
        }

        self.advance_to(time, tau_us)?;
        self.value += impulse;
        Ok(self.value)
    }
}

/// Analytically decays an already-local scalar value.
pub(crate) fn decay_value(value: f32, elapsed_us: u64, tau_us: f32) -> f32 {
    value * decay_factor(elapsed_us, tau_us)
}

fn decay_factor(elapsed_us: u64, tau_us: f32) -> f32 {
    (-(elapsed_us as f32) / tau_us).exp()
}

fn validate_tau(tau_us: f32) -> Result<(), TraceError> {
    if tau_us.is_finite() && tau_us > 0.0 {
        Ok(())
    } else {
        Err(TraceError::InvalidTimeConstant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_decays_analytically() {
        let mut trace = DecayingTrace::new(SimTime::ZERO);
        trace
            .add_impulse(SimTime::ZERO, 1_000.0, 1.0)
            .expect("valid update");

        let value = trace
            .advance_to(SimTime::from_micros(1_000), 1_000.0)
            .expect("forward time");

        assert!((value - (-1.0_f32).exp()).abs() < 1.0e-6);
    }

    #[test]
    fn time_reversal_is_rejected_without_mutation() {
        let mut trace = DecayingTrace::new(SimTime::from_micros(10));
        trace
            .add_impulse(SimTime::from_micros(20), 100.0, 1.0)
            .expect("forward update");
        let before = trace;

        assert!(matches!(
            trace.advance_to(SimTime::from_micros(19), 100.0),
            Err(TraceError::TimeWentBackwards { .. })
        ));
        assert_eq!(trace, before);
    }
}
