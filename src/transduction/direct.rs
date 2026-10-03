//! Direct, timestamp-preserving spike and pulse transduction.

use std::collections::VecDeque;

use crate::{
    io::{ChannelId, EffectorSignal, ReceptorSignal},
    primitives::SimTime,
    roots::MotorOutput,
};

use super::{ChannelSpike, MotorTransducer, SensoryTransducer, TransductionError};

/// Converts each finite, positive receptor value into one equal-amplitude spike.
///
/// Channel and timestamp are preserved. Values that cannot represent a
/// positive spike (including zero) are rejected without changing state.
/// Spikes are buffered until an inclusive `advance_until` reaches them.
#[derive(Clone, Debug, Default)]
pub struct DirectSensoryTransducer {
    pending: Pending<ChannelSpike>,
}

impl DirectSensoryTransducer {
    /// Creates an empty transducer with no completed time horizon.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SensoryTransducer for DirectSensoryTransducer {
    fn push(&mut self, signal: ReceptorSignal) -> Result<(), TransductionError> {
        validate_amplitude(signal.channel, signal.value)?;
        self.pending.push(
            signal.at,
            ChannelSpike {
                channel: signal.channel,
                at: signal.at,
                amplitude: signal.value,
            },
        )
    }

    fn advance_until(
        &mut self,
        until: SimTime,
        output: &mut Vec<ChannelSpike>,
    ) -> Result<(), TransductionError> {
        self.pending.advance_until(until, output)
    }
}

/// Converts each finite, positive motor amplitude into one equal-value pulse.
///
/// Channel and timestamp are preserved. Pulses are buffered until an inclusive
/// `advance_until` reaches them; silence produces no additional pulses.
#[derive(Clone, Debug, Default)]
pub struct DirectMotorTransducer {
    pending: Pending<EffectorSignal>,
}

impl DirectMotorTransducer {
    /// Creates an empty transducer with no completed time horizon.
    pub fn new() -> Self {
        Self::default()
    }
}

impl MotorTransducer for DirectMotorTransducer {
    fn push(&mut self, output: MotorOutput) -> Result<(), TransductionError> {
        validate_amplitude(output.channel, output.amplitude)?;
        self.pending.push(
            output.at,
            EffectorSignal {
                channel: output.channel,
                at: output.at,
                value: output.amplitude,
            },
        )
    }

    fn advance_until(
        &mut self,
        until: SimTime,
        signals: &mut Vec<EffectorSignal>,
    ) -> Result<(), TransductionError> {
        self.pending.advance_until(until, signals)
    }
}

#[derive(Clone, Debug)]
struct Pending<T> {
    completed_until: Option<SimTime>,
    events: VecDeque<(SimTime, T)>,
}

impl<T> Default for Pending<T> {
    fn default() -> Self {
        Self {
            completed_until: None,
            events: VecDeque::new(),
        }
    }
}

impl<T> Pending<T> {
    fn push(&mut self, at: SimTime, event: T) -> Result<(), TransductionError> {
        if let Some(completed_until) = self.completed_until
            && at <= completed_until
        {
            return Err(TransductionError::InputAlreadyProcessed {
                completed_until,
                at,
            });
        }
        let index = self.events.partition_point(|(time, _)| *time <= at);
        self.events.insert(index, (at, event));
        Ok(())
    }

    fn advance_until(
        &mut self,
        until: SimTime,
        output: &mut Vec<T>,
    ) -> Result<(), TransductionError> {
        if let Some(current) = self.completed_until
            && until < current
        {
            return Err(TransductionError::TimeWentBackwards {
                current,
                requested: until,
            });
        }
        let due = self.events.partition_point(|(time, _)| *time <= until);
        output.extend(self.events.drain(..due).map(|(_, event)| event));
        self.completed_until = Some(until);
        Ok(())
    }
}

fn validate_amplitude(channel: ChannelId, amplitude: f32) -> Result<(), TransductionError> {
    if amplitude.is_finite() && amplitude > 0.0 {
        Ok(())
    } else {
        Err(TransductionError::InvalidAmplitude { channel, amplitude })
    }
}
