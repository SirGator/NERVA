//! Seeded Poisson coding with independent channel streams.

use std::collections::BTreeMap;

use crate::{
    io::{ChannelId, ReceptorSignal},
    primitives::SimTime,
};

use super::{ChannelSpike, SensoryTransducer, TransductionError, pending::Pending};

/// Converts held receptor rates in Hertz into reproducible Poisson spikes.
///
/// Each channel draws exponential waiting hazards from its own SplitMix64
/// stream, derived from the explicit seed and `ChannelId`. The stream is
/// independent of other channels and their insertion order. Random numbers
/// are consumed only on channel creation and spike emission, so the same seed
/// and input history replay the same spike times across advance horizons.
///
/// Values must be finite and in `0..=1_000_000` Hz. Zero pauses the remaining
/// waiting hazard; rate changes preserve it instead of drawing a new one.
/// Physical values must be converted to rates by the receptor adapter.
/// Channels are preserved and spike amplitudes are always `1.0`.
///
/// Waiting hazards are `-ln(U)` for a uniform `U` strictly inside `(0, 1)`.
/// Time is integrated analytically between rate changes and spikes without a
/// global tick. Continuous arrival times are rounded up to microseconds;
/// multiple Poisson arrivals may therefore share a timestamp. Predictions use
/// `f64` differences between integer timestamps; arrivals outside `SimTime`
/// are not emitted. Old-rate arrivals due at a change timestamp are emitted
/// first. Simultaneous spikes on different channels are ordered by `ChannelId`.
///
/// ```
/// use nerva::{
///     io::{ChannelId, ReceptorSignal},
///     primitives::SimTime,
///     transduction::{PoissonSensoryTransducer, SensoryTransducer},
/// };
/// # fn main() -> Result<(), nerva::transduction::TransductionError> {
/// let mut encoder = PoissonSensoryTransducer::new(42);
/// encoder.push(ReceptorSignal {
///     channel: ChannelId(7), at: SimTime(100_000), value: 120.0,
/// })?;
/// let mut replay = encoder.clone();
/// let mut whole = Vec::new();
/// encoder.advance_until(SimTime(1_100_000), &mut whole)?;
/// let mut split = Vec::new();
/// replay.advance_until(SimTime(500_000), &mut split)?;
/// replay.advance_until(SimTime(1_100_000), &mut split)?;
/// assert_eq!(whole, split);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct PoissonSensoryTransducer {
    seed: u64,
    pending: Pending<ReceptorSignal>,
    channels: BTreeMap<ChannelId, PoissonChannel>,
}

impl PoissonSensoryTransducer {
    /// Creates an empty encoder with reproducible streams derived per channel.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            pending: Pending::default(),
            channels: BTreeMap::new(),
        }
    }
}

impl SensoryTransducer for PoissonSensoryTransducer {
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
        self.pending.advance_until(until, &mut updates)?;
        let start = output.len();
        for update in updates {
            let state = self
                .channels
                .entry(update.channel)
                .or_insert_with(|| PoissonChannel::new(self.seed, update.channel, update.at));
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
struct PoissonChannel {
    rng: ChannelRng,
    anchor: SimTime,
    // Integrated hazard scaled by one million (rate_hz * microseconds).
    // A negative residual represents additional arrivals in the same rounded us.
    remaining: f64,
    rate_hz: f64,
    next_spike: Option<SimTime>,
}

impl PoissonChannel {
    fn new(seed: u64, channel: ChannelId, at: SimTime) -> Self {
        let mut rng = ChannelRng {
            state: mix64(seed ^ mix64(channel.get())),
        };
        let remaining = rng.waiting_hazard();
        Self {
            rng,
            anchor: at,
            remaining,
            rate_hz: 0.0,
            next_spike: None,
        }
    }

    fn change_rate(&mut self, at: SimTime, rate_hz: f64) {
        if self.rate_hz == rate_hz {
            return;
        }
        let elapsed = at
            .duration_since(self.anchor)
            .expect("ordered channel updates");
        self.remaining -= elapsed as f64 * self.rate_hz;
        self.anchor = at;
        self.rate_hz = rate_hz;
        self.next_spike = self.predict();
    }

    fn predict(&self) -> Option<SimTime> {
        if self.rate_hz == 0.0 {
            return None;
        }
        let elapsed = (self.remaining / self.rate_hz).max(0.0).ceil();
        if elapsed >= u64::MAX as f64 {
            return None;
        }
        self.anchor.checked_add_us(elapsed as u64)
    }

    fn emit_until(&mut self, channel: ChannelId, until: SimTime, output: &mut Vec<ChannelSpike>) {
        while let Some(at) = self.next_spike.filter(|at| *at <= until) {
            output.push(ChannelSpike {
                channel,
                at,
                amplitude: 1.0,
            });
            let elapsed = at
                .duration_since(self.anchor)
                .expect("monotonic spike predictions");
            self.remaining -= elapsed as f64 * self.rate_hz;
            self.remaining += self.rng.waiting_hazard();
            self.anchor = at;
            self.next_spike = self.predict();
        }
    }
}

// Specify the integer generator and the open uniform conversion locally so
// coding does not depend on a platform RNG or on another architectural layer.
#[derive(Clone, Debug)]
struct ChannelRng {
    state: u64,
}

impl ChannelRng {
    fn waiting_hazard(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let bits = mix64(self.state) >> 12;
        let uniform = (bits as f64 + 0.5) / ((1_u64 << 52) as f64);
        -uniform.ln() * 1_000_000.0
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
