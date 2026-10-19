//! Deterministic rate coding of piecewise-constant receptor values.

use std::collections::BTreeMap;

use crate::{
    io::{ChannelId, ReceptorSignal},
    primitives::SimTime,
};

use super::{ChannelSpike, SensoryTransducer, TransductionError, pending::Pending};

/// Converts receptor values in Hertz into deterministic unit-amplitude spikes.
///
/// Each channel independently integrates `rate_hz * elapsed_us / 1_000_000`
/// and emits a spike whenever this phase reaches one. A new channel starts
/// with zero phase at its first signal. Rate changes preserve fractional
/// phase; zero pauses accumulation until a positive rate arrives. No new
/// receptor input is required for spikes to continue.
///
/// Values must be finite and in `0..=1_000_000` Hz. This upper bound permits at
/// most one spike per channel per microsecond. The adapter is responsible for
/// converting physical values into rates. Channels are preserved and emitted
/// amplitudes are always `1.0`.
///
/// Crossings are rounded up to integer microseconds. Predictions use `f64`
/// arithmetic relative to the last rate change, never absolute timestamps or
/// advance horizons, avoiding rounded-period drift and preserving output
/// across horizon partitions. A crossing outside the `SimTime` domain is not
/// emitted. At a rate-change timestamp, spikes already due under the old rate
/// are emitted before applying the new value. Multiple changes at that time
/// are applied in insertion order. Simultaneous spikes on different channels
/// are ordered by `ChannelId`.
///
/// ```
/// use nerva::{
///     io::{ChannelId, ReceptorSignal},
///     primitives::SimTime,
///     transduction::{RateSensoryTransducer, SensoryTransducer},
/// };
///
/// # fn main() -> Result<(), nerva::transduction::TransductionError> {
/// let mut encoder = RateSensoryTransducer::new();
/// encoder.push(ReceptorSignal {
///     channel: ChannelId(7), at: SimTime(100_000), value: 120.0,
/// })?;
/// let mut spikes = Vec::new();
/// encoder.advance_until(SimTime(110_000), &mut spikes)?;
/// assert_eq!(spikes[0].at, SimTime(108_334));
/// // The held rate continues to emit without another receptor sample.
/// encoder.advance_until(SimTime(120_000), &mut spikes)?;
/// assert_eq!(spikes[1].at, SimTime(116_667));
/// assert_eq!(spikes[1].amplitude, 1.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Default)]
pub struct RateSensoryTransducer {
    pending: Pending<ReceptorSignal>,
    channels: BTreeMap<ChannelId, ChannelRate>,
}

impl RateSensoryTransducer {
    /// Creates an empty transducer; channels become active on their first signal.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SensoryTransducer for RateSensoryTransducer {
    fn push(&mut self, signal: ReceptorSignal) -> Result<(), TransductionError> {
        if !signal.value.is_finite() || !(0.0..=1_000_000.0).contains(&signal.value) {
            return Err(TransductionError::InvalidRate {
                channel: signal.channel,
                rate_hz: signal.value,
            });
        }
        self.pending.push(signal.at, signal)
    }

    fn advance_until(
        &mut self,
        until: SimTime,
        output: &mut Vec<ChannelSpike>,
    ) -> Result<(), TransductionError> {
        let mut updates = Vec::new();
        // Reject a backwards horizon before mutating channel state or output.
        self.pending.advance_until(until, &mut updates)?;
        let start = output.len();
        for update in updates {
            let state = self
                .channels
                .entry(update.channel)
                .or_insert_with(|| ChannelRate::new(update.at));
            state.emit_until(update.channel, update.at, output);
            state.change_rate(update.at, f64::from(update.value));
        }
        for (&channel, state) in &mut self.channels {
            state.emit_until(channel, until, output);
        }
        output[start..].sort_by_key(|spike| (spike.at, spike.channel));
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ChannelRate {
    anchor: SimTime,
    // Phase scaled by one million. Keeping rate * microseconds before division
    // avoids cancellation such as (2.4 - 2.0) shifting an exact crossing by 1 us.
    phase: f64,
    rate_hz: f64,
    emitted: u64,
    next_spike: Option<SimTime>,
}

impl ChannelRate {
    fn new(at: SimTime) -> Self {
        Self {
            anchor: at,
            phase: 0.0,
            rate_hz: 0.0,
            emitted: 0,
            next_spike: None,
        }
    }

    fn change_rate(&mut self, at: SimTime, rate_hz: f64) {
        if rate_hz == self.rate_hz {
            return;
        }
        let elapsed = at
            .duration_since(self.anchor)
            .expect("ordered channel updates");
        self.phase = (self.phase + elapsed as f64 * self.rate_hz
            - self.emitted as f64 * 1_000_000.0)
            .clamp(0.0, 1_000_000.0);
        self.anchor = at;
        self.rate_hz = rate_hz;
        self.emitted = 0;
        self.next_spike = self.predict_after(at);
    }

    fn predict_after(&self, after: SimTime) -> Option<SimTime> {
        if self.rate_hz == 0.0 {
            return None;
        }
        let elapsed =
            (((self.emitted as f64 + 1.0) * 1_000_000.0 - self.phase) / self.rate_hz).ceil();
        // The cast must not saturate a crossing beyond the time domain.
        if elapsed >= u64::MAX as f64 {
            return None;
        }
        let predicted = self.anchor.checked_add_us(elapsed as u64)?;
        Some(predicted.max(after.checked_add_us(1)?))
    }

    fn emit_until(&mut self, channel: ChannelId, until: SimTime, output: &mut Vec<ChannelSpike>) {
        while let Some(at) = self.next_spike.filter(|at| *at <= until) {
            output.push(ChannelSpike {
                channel,
                at,
                amplitude: 1.0,
            });
            self.emitted += 1;
            self.next_spike = self.predict_after(at);
        }
    }
}
