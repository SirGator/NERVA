use nerva::{
    io::{ChannelId, EffectorSignal, ReceptorSignal},
    primitives::SimTime,
    roots::MotorOutput,
    transduction::{
        ChannelSpike, DirectMotorTransducer, DirectSensoryTransducer, MotorTransducer,
        SensoryTransducer, TransductionError,
    },
};

fn input(channel: u64, at: u64, value: f32) -> ReceptorSignal {
    ReceptorSignal {
        channel: ChannelId(channel),
        at: SimTime(at),
        value,
    }
}

fn motor_output(signal: ReceptorSignal) -> MotorOutput {
    MotorOutput {
        channel: signal.channel,
        at: signal.at,
        amplitude: signal.value,
    }
}

#[test]
fn direct_transducers_preserve_order_and_append_only_due_events_once() {
    let mut sensory = DirectSensoryTransducer::new();
    let mut motor = DirectMotorTransducer::new();
    // Queue in time disorder, with two simultaneous events in a deliberate
    // channel order, plus both ends of the integer time domain.
    for signal in [
        input(1, 20, 2.0),
        input(2, 0, 1.0),
        input(9, 10, 0.5),
        input(7, 10, 0.25),
        input(3, u64::MAX, 3.0),
    ] {
        sensory.push(signal).unwrap();
        motor.push(motor_output(signal)).unwrap();
    }

    let mut spikes = vec![ChannelSpike {
        channel: ChannelId(99),
        at: SimTime::ZERO,
        amplitude: 4.0,
    }];
    let mut signals = vec![EffectorSignal {
        channel: ChannelId(99),
        at: SimTime::ZERO,
        value: 4.0,
    }];
    for (until, count) in [
        (0, 2),
        (9, 2),
        (10, 4),
        (10, 4),
        (19, 4),
        (20, 5),
        (u64::MAX, 6),
    ] {
        sensory.advance_until(SimTime(until), &mut spikes).unwrap();
        motor.advance_until(SimTime(until), &mut signals).unwrap();
        assert_eq!(spikes.len(), count);
        assert_eq!(signals.len(), count);
    }

    let expected = [
        input(99, 0, 4.0),
        input(2, 0, 1.0),
        input(9, 10, 0.5),
        input(7, 10, 0.25),
        input(1, 20, 2.0),
        input(3, u64::MAX, 3.0),
    ];
    for ((spike, signal), expected) in spikes.iter().zip(&signals).zip(expected) {
        assert_eq!(spike.channel, expected.channel);
        assert_eq!(spike.at, expected.at);
        assert_eq!(spike.amplitude, expected.value);
        assert_eq!(signal.channel, expected.channel);
        assert_eq!(signal.at, expected.at);
        assert_eq!(signal.value, expected.value);
    }
}

#[test]
fn completed_horizons_reject_late_input_and_backwards_time_without_losing_pending_work() {
    let mut sensory = DirectSensoryTransducer::new();
    let mut motor = DirectMotorTransducer::new();
    let future = input(1, 20, 2.0);
    sensory.push(future).unwrap();
    motor.push(motor_output(future)).unwrap();
    let mut spikes = Vec::new();
    let mut signals = Vec::new();
    sensory.advance_until(SimTime(10), &mut spikes).unwrap();
    motor.advance_until(SimTime(10), &mut signals).unwrap();

    for at in [0, 9, 10] {
        let expected = Err(TransductionError::InputAlreadyProcessed {
            completed_until: SimTime(10),
            at: SimTime(at),
        });
        assert_eq!(sensory.push(input(1, at, 1.0)), expected);
        assert_eq!(motor.push(motor_output(input(1, at, 1.0))), expected);
    }
    let backwards = Err(TransductionError::TimeWentBackwards {
        current: SimTime(10),
        requested: SimTime(9),
    });
    assert_eq!(sensory.advance_until(SimTime(9), &mut spikes), backwards);
    assert_eq!(motor.advance_until(SimTime(9), &mut signals), backwards);
    assert!(spikes.is_empty());
    assert!(signals.is_empty());

    // A rejected operation must leave the frontier and future inputs intact.
    sensory.push(input(2, 11, 0.5)).unwrap();
    motor.push(motor_output(input(2, 11, 0.5))).unwrap();
    sensory.advance_until(SimTime(20), &mut spikes).unwrap();
    motor.advance_until(SimTime(20), &mut signals).unwrap();
    assert_eq!(
        spikes.iter().map(|spike| spike.at).collect::<Vec<_>>(),
        [SimTime(11), SimTime(20)]
    );
    assert_eq!(
        signals.iter().map(|signal| signal.at).collect::<Vec<_>>(),
        [SimTime(11), SimTime(20)]
    );
    assert_eq!(spikes[1].amplitude, 2.0);
    assert_eq!(signals[1].value, 2.0);

    // Check that rejection also preserves an already populated caller buffer.
    let saved_spikes = spikes.clone();
    let saved_signals = signals.clone();
    assert!(sensory.advance_until(SimTime(19), &mut spikes).is_err());
    assert!(motor.advance_until(SimTime(19), &mut signals).is_err());
    assert_eq!(spikes, saved_spikes);
    assert_eq!(signals, saved_signals);
}

#[test]
fn direct_transducers_reject_invalid_amplitudes_without_poisoning_the_queue() {
    let mut sensory = DirectSensoryTransducer::new();
    let mut motor = DirectMotorTransducer::new();
    for value in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            sensory.push(input(7, 10, value)),
            Err(TransductionError::InvalidAmplitude {
                channel: ChannelId(7),
                ..
            })
        ));
        assert!(matches!(
            motor.push(motor_output(input(7, 10, value))),
            Err(TransductionError::InvalidAmplitude {
                channel: ChannelId(7),
                ..
            })
        ));
    }
    sensory.push(input(7, 10, 0.5)).unwrap();
    motor.push(motor_output(input(7, 10, 0.5))).unwrap();
    let mut spikes = Vec::new();
    let mut signals = Vec::new();
    sensory.advance_until(SimTime(10), &mut spikes).unwrap();
    motor.advance_until(SimTime(10), &mut signals).unwrap();
    assert_eq!(spikes.len(), 1);
    assert_eq!(signals.len(), 1);
    assert_eq!(spikes[0].amplitude, 0.5);
    assert_eq!(signals[0].value, 0.5);
}

// Minimal single-channel fixtures exercise continuous time through the public
// traits. They are deliberately test-only, not production coding policies.
#[derive(Clone, Default)]
struct RegularRateFixture {
    inbox: DirectSensoryTransducer,
    channel: ChannelId,
    period_us: u64,
    next_spike: Option<SimTime>,
}

impl RegularRateFixture {
    fn emit_until(&mut self, until: SimTime, output: &mut Vec<ChannelSpike>) {
        while let Some(at) = self.next_spike.filter(|at| *at <= until) {
            output.push(ChannelSpike {
                channel: self.channel,
                at,
                amplitude: 1.0,
            });
            self.next_spike = at.checked_add_us(self.period_us);
        }
    }
}

impl SensoryTransducer for RegularRateFixture {
    fn push(&mut self, signal: ReceptorSignal) -> Result<(), TransductionError> {
        self.inbox.push(signal)
    }

    fn advance_until(
        &mut self,
        until: SimTime,
        output: &mut Vec<ChannelSpike>,
    ) -> Result<(), TransductionError> {
        let mut updates = Vec::new();
        self.inbox.advance_until(until, &mut updates)?;
        for update in updates {
            // A value change replaces the previous rate before any spike at
            // the same time. The test rates divide one second exactly.
            if let Some(before) = update.at.as_micros().checked_sub(1) {
                self.emit_until(SimTime(before), output);
            }
            self.channel = update.channel;
            self.period_us = (1_000_000.0 / update.amplitude) as u64;
            self.next_spike = update.at.checked_add_us(self.period_us);
        }
        self.emit_until(until, output);
        Ok(())
    }
}

#[test]
fn rate_trait_emits_during_receptor_silence_independently_of_horizon_partition() {
    let mut whole = RegularRateFixture::default();
    let rate: &mut dyn SensoryTransducer = &mut whole;
    rate.push(input(1, 100_000, 100.0)).unwrap();
    let mut split = whole.clone();
    let mut all_spikes = Vec::new();
    whole
        .advance_until(SimTime(120_000), &mut all_spikes)
        .unwrap();
    let mut split_spikes = Vec::new();
    split
        .advance_until(SimTime(100_000), &mut split_spikes)
        .unwrap();
    assert!(split_spikes.is_empty());
    split
        .advance_until(SimTime(110_000), &mut split_spikes)
        .unwrap();
    assert_eq!(split_spikes.len(), 1);
    split
        .advance_until(SimTime(120_000), &mut split_spikes)
        .unwrap();
    split
        .advance_until(SimTime(120_000), &mut split_spikes)
        .unwrap();
    assert_eq!(all_spikes, split_spikes);
    assert_eq!(
        all_spikes.iter().map(|spike| spike.at).collect::<Vec<_>>(),
        [SimTime(110_000), SimTime(120_000)]
    );
}

#[test]
fn rate_changes_take_effect_at_their_event_time_even_with_a_large_horizon() {
    let mut whole = RegularRateFixture::default();
    whole.push(input(1, 115_000, 200.0)).unwrap();
    whole.push(input(1, 100_000, 100.0)).unwrap();
    let mut split = whole.clone();
    let mut all_spikes = Vec::new();
    whole
        .advance_until(SimTime(130_000), &mut all_spikes)
        .unwrap();
    let mut split_spikes = Vec::new();
    for until in [100_000, 110_000, 115_000, 120_000, 125_000, 130_000] {
        split
            .advance_until(SimTime(until), &mut split_spikes)
            .unwrap();
    }
    assert_eq!(all_spikes, split_spikes);
    assert_eq!(
        all_spikes.iter().map(|spike| spike.at).collect::<Vec<_>>(),
        [
            SimTime(110_000),
            SimTime(120_000),
            SimTime(125_000),
            SimTime(130_000)
        ]
    );
}

#[derive(Clone, Default)]
struct DecayingMotorFixture {
    inbox: DirectMotorTransducer,
    channel: Option<ChannelId>,
    updated_at: Option<SimTime>,
    rate_hz: f64,
}

impl DecayingMotorFixture {
    const TAU_US: f64 = 10_000.0;

    fn decay_to(&mut self, at: SimTime) {
        if let Some(previous) = self.updated_at {
            let elapsed = at.duration_since(previous).unwrap();
            self.rate_hz *= (-(elapsed as f64) / Self::TAU_US).exp();
        }
        self.updated_at = Some(at);
    }
}

impl MotorTransducer for DecayingMotorFixture {
    fn push(&mut self, output: MotorOutput) -> Result<(), TransductionError> {
        self.inbox.push(output)
    }

    fn advance_until(
        &mut self,
        until: SimTime,
        signals: &mut Vec<EffectorSignal>,
    ) -> Result<(), TransductionError> {
        let mut pulses = Vec::new();
        self.inbox.advance_until(until, &mut pulses)?;
        if self.updated_at == Some(until) {
            return Ok(());
        }
        for pulse in pulses {
            self.decay_to(pulse.at);
            self.channel = Some(pulse.channel);
            self.rate_hz += f64::from(pulse.value) * 1_000_000.0 / Self::TAU_US;
        }
        self.decay_to(until);
        if let Some(channel) = self.channel {
            signals.push(EffectorSignal {
                channel,
                at: until,
                value: self.rate_hz as f32,
            });
        }
        Ok(())
    }
}

#[test]
fn motor_trait_decays_during_silence_and_agrees_at_shared_horizons() {
    let mut whole = DecayingMotorFixture::default();
    let motor: &mut dyn MotorTransducer = &mut whole;
    motor.push(motor_output(input(1, 100_000, 1.0))).unwrap();
    let mut split = whole.clone();
    let mut whole_signals = Vec::new();
    whole
        .advance_until(SimTime(120_000), &mut whole_signals)
        .unwrap();
    let mut split_signals = Vec::new();
    for until in [100_000, 105_000, 110_000, 120_000, 120_000] {
        split
            .advance_until(SimTime(until), &mut split_signals)
            .unwrap();
    }
    assert_eq!(split_signals.len(), 4, "repeating a horizon adds no sample");
    assert_eq!(split_signals[0].value, 100.0);
    assert!(
        split_signals
            .windows(2)
            .all(|pair| pair[1].value < pair[0].value)
    );
    let final_sample = split_signals.last().unwrap();
    assert_eq!(final_sample.at, SimTime(120_000));
    assert!((f64::from(final_sample.value) - 100.0 * (-2.0_f64).exp()).abs() < 1.0e-5);
    assert!((whole_signals[0].value - final_sample.value).abs() < 1.0e-5);
}
