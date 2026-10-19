use nerva::{
    io::{ChannelId, ReceptorSignal},
    primitives::SimTime,
    transduction::{ChannelSpike, PoissonSensoryTransducer, SensoryTransducer, TransductionError},
};

fn signal(channel: u64, at: u64, rate_hz: f32) -> ReceptorSignal {
    ReceptorSignal {
        channel: ChannelId(channel),
        at: SimTime(at),
        value: rate_hz,
    }
}

fn encode(seed: u64, updates: &[ReceptorSignal], horizons: &[u64]) -> Vec<ChannelSpike> {
    let mut encoder = PoissonSensoryTransducer::new(seed);
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
fn seeds_replay_the_same_spike_train_and_distinct_seeds_change_it() {
    let updates = [signal(7, 100_000, 120.0)];
    let output = encode(42, &updates, &[1_100_000]);
    assert_eq!(output, encode(42, &updates, &[1_100_000]));
    assert_ne!(output, encode(43, &updates, &[1_100_000]));
    assert!(output.len() > 50);
    assert!(output.iter().all(|spike| spike.channel == ChannelId(7)
        && spike.amplitude == 1.0
        && spike.at > SimTime(100_000)));
}

#[test]
fn adding_other_channels_and_changing_their_creation_order_preserves_a_channel_stream() {
    let updates = [
        signal(7, 100_000, 120.0),
        signal(7, 400_000, 0.0),
        signal(7, 600_000, 240.0),
    ];
    let expected = encode(42, &updates, &[1_100_000]);
    let mut combined = vec![signal(8, 100_000, 250.0), signal(9, 0, 20.0)];
    combined.extend(updates);
    combined.push(signal(9, 400_000, 1_000.0));
    let with_others = encode(42, &combined, &[1_100_000]);
    assert_eq!(
        with_others
            .iter()
            .filter(|spike| spike.channel == ChannelId(7))
            .copied()
            .collect::<Vec<_>>(),
        expected
    );
    // Distinct channels also get distinct streams even at the same rate and time.
    let equal_rates = encode(
        42,
        &[signal(7, 0, 120.0), signal(8, 0, 120.0)],
        &[1_000_000],
    );
    let times = |channel| {
        equal_rates
            .iter()
            .filter(|spike| spike.channel == ChannelId(channel))
            .map(|spike| spike.at)
            .collect::<Vec<_>>()
    };
    assert_ne!(times(7), times(8));
}

#[test]
fn rate_changes_pauses_and_streamed_inputs_are_independent_of_advance_partition() {
    let updates = [
        signal(7, 100_000, 120.0),
        signal(9, 104_500, 80.5),
        signal(7, 115_111, 250.0),
        signal(7, 115_111, 200.0),
        signal(9, 130_000, 0.0),
        signal(7, 137_789, 0.0),
        signal(7, 150_333, 120.5),
        signal(9, 170_000, 79.25),
        signal(7, 190_000, 500.0),
    ];
    let expected = encode(123, &updates, &[250_000]);
    let mut seed = 7_u64;
    let mut at = 0;
    let mut horizons = Vec::new();
    while at < 250_000 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        at = (at + 1 + seed % 3_000).min(250_000);
        horizons.push(at);
        horizons.push(at);
    }
    assert_eq!(encode(123, &updates, &horizons), expected);
    let mut encoder = PoissonSensoryTransducer::new(123);
    let mut cursor = 0;
    let mut output = Vec::new();
    for until in horizons {
        while cursor < updates.len() && updates[cursor].at <= SimTime(until) {
            encoder.push(updates[cursor]).unwrap();
            cursor += 1;
        }
        encoder.advance_until(SimTime(until), &mut output).unwrap();
    }
    assert_eq!(output, expected);
}

#[test]
fn zero_pauses_the_existing_waiting_hazard_without_consuming_random_draws() {
    let constant = encode(42, &[signal(7, 0, 120.0)], &[900_000]);
    let paused = encode(
        42,
        &[
            signal(7, 0, 120.0),
            signal(7, 100_000, 0.0),
            signal(7, 200_000, 120.0),
        ],
        &[1_000_000],
    );
    let expected: Vec<_> = constant
        .iter()
        .map(|spike| {
            let mut shifted = *spike;
            if spike.at > SimTime(100_000) {
                shifted.at = spike.at.checked_add_us(100_000).unwrap();
            }
            shifted
        })
        .collect();
    assert_eq!(paused, expected);
    assert!(encode(42, &[signal(7, 0, 0.0)], &[u64::MAX]).is_empty());
    assert_eq!(
        encode(
            42,
            &[signal(7, 0, 0.0), signal(7, 100_000, 120.0)],
            &[1_000_000]
        ),
        encode(42, &[signal(7, 100_000, 120.0)], &[1_000_000]),
    );
}

#[test]
fn redundant_samples_do_not_resample_waits_or_change_the_random_stream() {
    let mut updates = vec![signal(7, 100_000, 120.0)];
    for at in (100_001..200_000).step_by(137) {
        updates.push(signal(7, at, 120.0));
    }
    assert_eq!(
        encode(42, &updates, &[200_000]),
        encode(42, &updates[..1], &[200_000])
    );
}

#[test]
fn poisson_waits_have_exponential_mean_and_variance_for_a_fixed_seed() {
    let output = encode(19, &[signal(7, 0, 200.0)], &[100_000_000]);
    assert!((19_000..21_000).contains(&output.len()));
    let mut previous = 0;
    let intervals: Vec<_> = output
        .iter()
        .map(|spike| {
            let interval = spike.at.as_micros() - previous;
            previous = spike.at.as_micros();
            interval as f64
        })
        .collect();
    let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
    let variance = intervals
        .iter()
        .map(|interval| (interval - mean).powi(2))
        .sum::<f64>()
        / intervals.len() as f64;
    assert!(
        (mean / 5_000.0 - 1.0).abs() < 0.05,
        "mean {mean} us differs from 1/r"
    );
    assert!(
        (variance.sqrt() / mean - 1.0).abs() < 0.05,
        "exponential waits have coefficient of variation 1"
    );
}

#[test]
fn quantization_preserves_multiple_poisson_arrivals_in_a_single_microsecond() {
    let output = encode(42, &[signal(7, 0, 1_000_000.0)], &[1_000]);
    assert!((800..1_200).contains(&output.len()));
    assert!(output.windows(2).any(|pair| pair[0].at == pair[1].at));
    assert!(output.windows(2).all(|pair| pair[0].at <= pair[1].at));
    assert!(
        output
            .iter()
            .all(|spike| spike.at > SimTime::ZERO && spike.at <= SimTime(1_000))
    );
}

#[test]
fn invalid_inputs_and_time_errors_preserve_pending_work_output_and_rng_state() {
    let mut encoder = PoissonSensoryTransducer::new(42);
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
            Err(TransductionError::InvalidRate { .. })
        ));
    }
    for at in [100_000, 110_000] {
        assert_eq!(
            encoder.push(signal(7, at, 200.0)),
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
    // A later resumed run also checks the future random stream after the errors.
    encoder.push(signal(7, 210_000, 240.0)).unwrap();
    control.push(signal(7, 210_000, 240.0)).unwrap();
    encoder
        .advance_until(SimTime(1_000_000), &mut output)
        .unwrap();
    control
        .advance_until(SimTime(1_000_000), &mut expected)
        .unwrap();
    assert_eq!(output, expected);
}

#[test]
fn large_timestamps_and_the_end_of_simulation_time_do_not_round_or_wrap() {
    let offset = (1_u64 << 53) + 100;
    let reference = encode(42, &[signal(7, 0, 120.0)], &[100_000]);
    let shifted = encode(42, &[signal(7, offset, 120.0)], &[offset + 100_000]);
    assert_eq!(
        shifted
            .iter()
            .map(|spike| spike.at.as_micros() - offset)
            .collect::<Vec<_>>(),
        reference
            .iter()
            .map(|spike| spike.at.as_micros())
            .collect::<Vec<_>>()
    );
    let end = encode(
        42,
        &[signal(7, u64::MAX - 100, 1_000_000.0)],
        &[u64::MAX, u64::MAX],
    );
    let beginning = encode(42, &[signal(7, 0, 1_000_000.0)], &[100]);
    assert_eq!(
        end.iter()
            .map(|spike| spike.at.as_micros() - (u64::MAX - 100))
            .collect::<Vec<_>>(),
        beginning
            .iter()
            .map(|spike| spike.at.as_micros())
            .collect::<Vec<_>>()
    );
    assert!(encode(42, &[signal(7, 0, f32::from_bits(1))], &[u64::MAX]).is_empty());
    assert!(encode(42, &[signal(7, u64::MAX, 120.0)], &[u64::MAX]).is_empty());
}
