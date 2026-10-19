use nerva::{
    io::{ChannelId, ReceptorSignal},
    primitives::SimTime,
    transduction::{ChannelSpike, RateSensoryTransducer, SensoryTransducer, TransductionError},
};

fn signal(channel: u64, at: u64, rate_hz: f32) -> ReceptorSignal {
    ReceptorSignal {
        channel: ChannelId(channel),
        at: SimTime(at),
        value: rate_hz,
    }
}

fn encode(updates: &[ReceptorSignal], horizons: &[u64]) -> Vec<ChannelSpike> {
    let mut encoder = RateSensoryTransducer::new();
    for &update in updates {
        encoder.push(update).unwrap();
    }
    let mut output = Vec::new();
    for &until in horizons {
        encoder.advance_until(SimTime(until), &mut output).unwrap();
    }
    output
}

#[test]
fn non_integral_periods_are_rounded_up_without_accumulating_period_drift() {
    for (rate_hz, numerator, denominator) in
        [(120.0, 1_000_000_u64, 120_u64), (120.5, 2_000_000, 241)]
    {
        let output = encode(&[signal(7, 100_000, rate_hz)], &[100_000, 1_100_000]);
        assert_eq!(output.len(), 120);
        for (index, spike) in output.iter().enumerate() {
            let elapsed = ((index as u64 + 1) * numerator).div_ceil(denominator);
            assert_eq!(spike.at, SimTime(100_000 + elapsed));
            assert_eq!(spike.channel, ChannelId(7));
            assert_eq!(spike.amplitude, 1.0);
        }
    }
}

#[test]
fn fractional_phase_from_a_non_integral_period_does_not_delay_an_exact_crossing() {
    // 120 Hz for 20 ms accumulates 2.4 spikes. After the two emitted spikes,
    // the remaining 0.4 phase needs exactly 3 ms at 200 Hz to reach one.
    let output = encode(
        &[signal(1, 100_000, 120.0), signal(1, 120_000, 200.0)],
        &[130_000],
    );
    assert_eq!(
        output.iter().map(|spike| spike.at).collect::<Vec<_>>(),
        [
            SimTime(108_334),
            SimTime(116_667),
            SimTime(123_000),
            SimTime(128_000)
        ]
    );
}

#[test]
fn zero_rate_pauses_and_resumption_preserves_fractional_phase() {
    let output = encode(
        &[
            signal(1, 0, 100.0),
            signal(1, 5_000, 0.0),
            signal(1, 20_000, 200.0),
        ],
        &[30_000],
    );
    assert_eq!(
        output.iter().map(|spike| spike.at).collect::<Vec<_>>(),
        [SimTime(22_500), SimTime(27_500)]
    );
    assert!(encode(&[signal(1, 0, 0.0)], &[1_000_000]).is_empty());
}

#[test]
fn channels_are_independent_and_simultaneous_spikes_have_a_stable_order() {
    // Queue in the opposite order from event time.
    let output = encode(&[signal(7, 5_000, 200.0), signal(9, 0, 100.0)], &[20_000]);
    assert_eq!(
        output
            .iter()
            .map(|spike| (spike.at, spike.channel))
            .collect::<Vec<_>>(),
        [
            (SimTime(10_000), ChannelId(7)),
            (SimTime(10_000), ChannelId(9)),
            (SimTime(15_000), ChannelId(7)),
            (SimTime(20_000), ChannelId(7)),
            (SimTime(20_000), ChannelId(9)),
        ]
    );
}

#[test]
fn stopping_at_a_crossing_keeps_the_spike_accumulated_before_the_change() {
    let output = encode(
        &[signal(1, 0, 100.0), signal(1, 10_000, 0.0)],
        &[10_000, 10_000, 50_000],
    );
    assert_eq!(
        output.iter().map(|spike| spike.at).collect::<Vec<_>>(),
        [SimTime(10_000)]
    );
    let last_change_wins = encode(&[signal(1, 0, 50.0), signal(1, 0, 100.0)], &[20_000]);
    assert_eq!(
        last_change_wins
            .iter()
            .map(|spike| spike.at)
            .collect::<Vec<_>>(),
        [SimTime(10_000), SimTime(20_000)]
    );
}

#[test]
fn arbitrary_horizon_partitions_and_streamed_inputs_replay_exactly() {
    let updates = [
        signal(7, 100_000, 120.0),
        signal(9, 104_500, 80.5),
        signal(7, 115_111, 250.0),
        signal(9, 130_000, 0.0),
        signal(7, 137_789, 0.0),
        signal(7, 150_333, 120.5),
        signal(9, 170_000, 79.25),
        signal(7, 190_000, 500.0),
    ];
    let expected = encode(&updates, &[250_000]);
    let mut seed = 7_u64;
    let mut at = 0;
    let mut horizons = Vec::new();
    while at < 250_000 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        at = (at + 1 + seed % 3_000).min(250_000);
        horizons.push(at);
        horizons.push(at); // repeated inclusive advances must be harmless
    }
    assert_eq!(encode(&updates, &horizons), expected);

    let mut streamed = RateSensoryTransducer::new();
    let mut cursor = 0;
    let mut output = Vec::new();
    for until in horizons {
        while cursor < updates.len() && updates[cursor].at <= SimTime(until) {
            streamed.push(updates[cursor]).unwrap();
            cursor += 1;
        }
        streamed.advance_until(SimTime(until), &mut output).unwrap();
    }
    assert_eq!(output, expected);
}

#[test]
fn repeating_the_same_value_does_not_restart_or_round_the_phase() {
    let mut updates = vec![signal(7, 100_000, 120.0)];
    for at in (100_001..200_000).step_by(137) {
        updates.push(signal(7, at, 120.0));
    }
    assert_eq!(
        encode(&updates, &[200_000]),
        encode(&updates[..1], &[200_000])
    );
}

#[test]
fn invalid_rates_and_time_errors_preserve_queued_input_phase_and_caller_output() {
    let mut encoder = RateSensoryTransducer::new();
    encoder.push(signal(7, 100_000, 120.0)).unwrap();
    encoder.push(signal(7, 130_000, 0.0)).unwrap();
    let mut output = vec![ChannelSpike {
        channel: ChannelId(99),
        at: SimTime::ZERO,
        amplitude: 3.0,
    }];
    encoder
        .advance_until(SimTime(110_000), &mut output)
        .unwrap();
    let mut control = encoder.clone();
    let mut expected = output.clone();
    let saved = output.clone();
    for rate in [
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1_000_001.0,
    ] {
        assert!(matches!(
            encoder.push(signal(7, 115_000, rate)),
            Err(TransductionError::InvalidRate {
                channel: ChannelId(7),
                ..
            })
        ));
    }
    for at in [100_000, 110_000] {
        assert_eq!(
            encoder.push(signal(7, at, 1.0)),
            Err(TransductionError::InputAlreadyProcessed {
                completed_until: SimTime(110_000),
                at: SimTime(at)
            })
        );
    }
    assert_eq!(
        encoder.advance_until(SimTime(109_999), &mut output),
        Err(TransductionError::TimeWentBackwards {
            current: SimTime(110_000),
            requested: SimTime(109_999)
        })
    );
    assert_eq!(output, saved);
    encoder
        .advance_until(SimTime(200_000), &mut output)
        .unwrap();
    control
        .advance_until(SimTime(200_000), &mut expected)
        .unwrap();
    assert_eq!(output, expected);
    assert_eq!(output.len(), 4); // sentinel plus 108334, 116667, 125000
}

#[test]
fn absolute_timestamps_remain_exact_above_floating_point_integer_precision() {
    let start = (1_u64 << 53) + 100;
    let output = encode(&[signal(1, start, 120.0)], &[start + 25_000]);
    assert_eq!(
        output.iter().map(|spike| spike.at).collect::<Vec<_>>(),
        [
            SimTime(start + 8_334),
            SimTime(start + 16_667),
            SimTime(start + 25_000)
        ]
    );
    let end = encode(
        &[signal(1, u64::MAX - 2, 1_000_000.0)],
        &[u64::MAX, u64::MAX],
    );
    assert_eq!(
        end.iter().map(|spike| spike.at).collect::<Vec<_>>(),
        [SimTime(u64::MAX - 1), SimTime(u64::MAX)]
    );
}

#[test]
fn extremely_low_rates_do_not_saturate_future_crossings_into_the_time_domain() {
    let output = encode(&[signal(1, 0, f32::from_bits(1))], &[0, u64::MAX]);
    assert!(output.is_empty());
}
