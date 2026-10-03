//! Generic transduction traits connecting I/O signals to root-channel spikes.
//!
//! Concrete encoders (direct, rate, Poisson, population, visual, proprioceptive,
//! ...) implement [`SensoryTransducer`]; concrete decoders implement
//! [`MotorTransducer`]. The NERVA core never knows which coding is in use.

use crate::{
    io::{ChannelId, EffectorSignal, ReceptorSignal},
    primitives::SimTime,
    roots::MotorOutput,
};

use super::TransductionError;

/// A spike on a root-local channel, before nerve routing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelSpike {
    /// Stable root-local channel identity.
    pub channel: ChannelId,
    /// Emission timestamp at the root.
    pub at: SimTime,
    /// Positive stimulus amplitude.
    pub amplitude: f32,
}

/// Converts receptor signals into root-channel activity over simulation time.
///
/// `push` queues a timestamped input; only `advance_until` consumes it and
/// appends output. For continuous encodings, a signal changes channel state at
/// its timestamp and that state persists until a later signal changes it.
/// Advancing without new input must still integrate elapsed time, allowing
/// rate or Poisson encoders to emit spikes during silent receptor intervals.
/// The caller supplies horizons; no global tick or core access is needed.
///
/// Horizons are inclusive and monotonic. Queue all inputs at or before a
/// horizon before advancing to it. Inputs at or before an already completed
/// horizon are rejected. Future inputs may be queued in any order; process
/// them chronologically and preserve insertion order at equal timestamps.
/// Output is appended in time order and never extends beyond the horizon.
/// Repeating a horizon emits nothing; splitting the same timeline into
/// multiple advances must preserve spike times (and random state for seeded
/// encoders). On error, leave state and the caller's output unchanged.
pub trait SensoryTransducer {
    /// Queues a signal without processing it or emitting spikes.
    fn push(&mut self, signal: ReceptorSignal) -> Result<(), TransductionError>;

    /// Processes queued input and elapsed time through `until`, appending spikes.
    fn advance_until(
        &mut self,
        until: SimTime,
        output: &mut Vec<ChannelSpike>,
    ) -> Result<(), TransductionError>;
}

/// Converts observed motor-channel outputs into effector signals.
///
/// Uses the same queueing, inclusive horizons, ordering and error contract as
/// [`SensoryTransducer`]. `advance_until` must integrate elapsed time even
/// without motor spikes, so smoothing or decay can produce updated effector
/// values during silence. A direct decoder only emits one pulse per queued
/// motor output. Continuous decoders may additionally emit their current
/// value at each new horizon; intermediate samples need not match between
/// different horizon partitions, but the state at a shared horizon must agree
/// within numerical precision.
pub trait MotorTransducer {
    /// Queues a motor-channel output without processing it or emitting signals.
    fn push(&mut self, output: MotorOutput) -> Result<(), TransductionError>;

    /// Processes queued output and elapsed time through `until`, appending signals.
    fn advance_until(
        &mut self,
        until: SimTime,
        signals: &mut Vec<EffectorSignal>,
    ) -> Result<(), TransductionError>;
}
