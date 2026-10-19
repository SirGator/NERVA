//! Isolated scheduler comparison on synthetic, prebuilt operation streams.
//! Run with `cargo run --release --example scheduler_profile`.
//! The bucket queue is confined to this example, never selected by Simulation.
#[path = "scheduler_profile/queues.rs"]
mod queues;

use std::{
    error::Error,
    hint::black_box,
    time::{Duration, Instant},
};

use nerva::{
    core::{Event, EventKind, NeuronId, SimTime, SynapseId},
    runtime::{EventScheduler, ScheduledEvent, SchedulerError},
};
use queues::{BucketQueue, Queue};

const REPETITIONS: usize = 7;
const LARGE: usize = 65_536;

enum Stage {
    Enqueue(Vec<Event>),
    Cancel(Vec<u64>),
    Prune,
    Pop(usize),
}

struct Workload {
    name: &'static str,
    stages: Vec<Stage>,
}

fn arrival(index: usize, time: u64) -> Event {
    Event::new(
        SimTime(time),
        EventKind::SynapticArrival {
            synapse_id: SynapseId(index as u64 * 7 + 3),
            target: NeuronId(index as u64 % 2_048),
            amplitude: if index.is_multiple_of(5) {
                -0.125
            } else {
                0.25
            },
        },
    )
}

fn bulk(
    name: &'static str,
    count: usize,
    timestamps: usize,
    cancel_half: bool,
    prune: bool,
) -> Workload {
    let events = (0..count)
        // Odd multiplication permutes power-of-two horizons, so timestamps
        // are inserted out of order. count/timestamps is the exact batch size.
        .map(|index| arrival(index, ((index * 40_503) % timestamps + 1) as u64))
        .collect();
    let mut stages = vec![Stage::Enqueue(events)];
    if cancel_half {
        // Permute handles, too: do not benchmark only sorted deletions.
        stages.push(Stage::Cancel(
            (0..count / 2)
                .map(|index| ((index * 40_503) % count) as u64)
                .collect(),
        ));
    }
    if prune {
        stages.push(Stage::Prune);
    }
    stages.push(Stage::Pop(usize::MAX));
    Workload { name, stages }
}

fn streaming() -> Workload {
    let mut stages = Vec::new();
    let mut predictions = Vec::new();
    let mut sequence = 0;
    for round in 0..16 {
        // Predictions remain queued far beyond the next arrival wave, then
        // get eagerly superseded. Stable handles survive intervening pops.
        if !predictions.is_empty() {
            stages.push(Stage::Cancel(std::mem::take(&mut predictions)));
        }
        let events = (0..4_096)
            .map(|index| {
                arrival(
                    round * 4_096 + index,
                    (round * 100_000 + (index * 40_503) % 128 + 1) as u64,
                )
            })
            .collect();
        stages.push(Stage::Enqueue(events));
        sequence += 4_096;
        let events = (0..128)
            .map(|index| {
                predictions.push(sequence);
                sequence += 1;
                Event::new(
                    SimTime(1_000_000_000 + (index * 101 + round) as u64),
                    EventKind::IntrinsicSpike {
                        neuron_id: NeuronId(index as u64),
                    },
                )
            })
            .collect();
        stages.push(Stage::Enqueue(events));
        stages.push(Stage::Pop(128));
    }
    stages.push(Stage::Pop(usize::MAX));
    Workload {
        name: "streaming-prediction-refresh",
        stages,
    }
}

fn workloads() -> Vec<Workload> {
    vec![
        bulk("small-128-four-times", 128, 4, false, false),
        bulk("synchronous-65536-32-times", LARGE, 32, false, false),
        bulk("distributed-65536-1024-times", LARGE, 1_024, false, false),
        bulk("singleton-65536-times", LARGE, LARGE, false, false),
        bulk("distributed-cancel-half", LARGE, 1_024, true, false),
        bulk("topology-cancel-matching", LARGE, 1_024, false, true),
        streaming(),
    ]
}

#[derive(Debug, PartialEq)]
enum Output {
    Sequence(u64),
    Cancelled(bool),
    Pruned(usize),
    Batch(Vec<ScheduledEvent<EventKind>>),
    Empty,
    State(usize, SimTime, Option<SimTime>),
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Counts {
    scheduled: usize,
    cancelled: usize,
    batches: usize,
    delivered: usize,
    peak_queued: usize,
}

#[derive(Default)]
struct Timings {
    total: Duration,
    enqueue: Duration,
    cancel: Duration,
    matching: Duration,
    dequeue: Duration,
}

struct Run {
    timings: Timings,
    counts: Counts,
    trace: Vec<Output>,
}

/// VERIFY retains complete results only in untimed reference comparisons.
/// PROFILE adds one clock pair per operation block, never per event.
fn run<Q: Queue<EventKind>, const VERIFY: bool, const PROFILE: bool>(
    workload: &Workload,
) -> Result<Run, SchedulerError> {
    let mut queue = Q::default();
    let mut counts = Counts::default();
    let mut timings = Timings::default();
    let mut trace = Vec::new();
    let started = Instant::now();
    for stage in &workload.stages {
        let phase_start = if PROFILE { Some(Instant::now()) } else { None };
        match stage {
            Stage::Enqueue(events) => {
                for event in events {
                    let sequence = queue.schedule(black_box(event.time), black_box(event.kind))?;
                    if VERIFY {
                        trace.push(Output::Sequence(sequence));
                    }
                    black_box(sequence);
                }
                counts.scheduled += events.len();
                if let Some(start) = phase_start {
                    timings.enqueue += start.elapsed();
                }
            }
            Stage::Cancel(sequences) => {
                for &sequence in sequences {
                    let cancelled = queue.cancel(black_box(sequence));
                    counts.cancelled += usize::from(cancelled);
                    if VERIFY {
                        trace.push(Output::Cancelled(cancelled));
                    }
                }
                if let Some(start) = phase_start {
                    timings.cancel += start.elapsed();
                }
            }
            Stage::Prune => {
                let cancelled = queue.cancel_matching(|event| {
                    matches!(event,
                    EventKind::SynapticArrival { synapse_id, .. } if synapse_id.0 % 11 == 3)
                });
                counts.cancelled += cancelled;
                if VERIFY {
                    trace.push(Output::Pruned(cancelled));
                }
                if let Some(start) = phase_start {
                    timings.matching += start.elapsed();
                }
            }
            Stage::Pop(limit) => {
                for _ in 0..*limit {
                    let Some(events) = queue.pop(usize::MAX)? else {
                        if VERIFY {
                            trace.push(Output::Empty);
                        }
                        break;
                    };
                    counts.batches += 1;
                    counts.delivered += events.len();
                    black_box(&events);
                    if VERIFY {
                        trace.push(Output::Batch(events));
                    }
                    // In measured runs the returned batch is dropped here,
                    // including payload/output-vector destruction in dequeue.
                }
                if let Some(start) = phase_start {
                    timings.dequeue += start.elapsed();
                }
            }
        }
        let (len, time, next) = queue.state();
        counts.peak_queued = counts.peak_queued.max(len);
        if VERIFY {
            trace.push(Output::State(len, time, next));
        }
    }
    timings.total = started.elapsed();
    assert_eq!(queue.state().0, 0, "workload must drain the queue");
    assert_eq!(counts.scheduled, counts.cancelled + counts.delivered);
    Ok(Run {
        timings,
        counts,
        trace,
    })
}

fn measure(queue: usize, workload: &Workload, profiling: bool) -> Result<Run, SchedulerError> {
    match (queue, profiling) {
        (0, false) => run::<EventScheduler<EventKind>, false, false>(workload),
        (0, true) => run::<EventScheduler<EventKind>, false, true>(workload),
        (1, false) => run::<BucketQueue<EventKind>, false, false>(workload),
        (1, true) => run::<BucketQueue<EventKind>, false, true>(workload),
        _ => unreachable!(),
    }
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

fn print_run(kind: &str, workload: &str, queue: &str, sample: Option<usize>, run: &Run) {
    let sample = sample.map(|value| value.to_string()).unwrap_or_default();
    // Empty fields distinguish unmeasured phases from measured zero work.
    let phase = |duration| {
        if kind == "phase" {
            format!("{:.3}", micros(duration))
        } else {
            String::new()
        }
    };
    println!(
        "{kind},{workload},{queue},{sample},{:.3},{},{},{},{},{},{},{},{},{}",
        micros(run.timings.total),
        phase(run.timings.enqueue),
        phase(run.timings.cancel),
        phase(run.timings.matching),
        phase(run.timings.dequeue),
        run.counts.scheduled,
        run.counts.cancelled,
        run.counts.batches,
        run.counts.delivered,
        run.counts.peak_queued
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let queues = ["flat-btree", "bucket-btree"];
    println!(
        "kind,workload,queue,sample,total_us,enqueue_us,cancel_us,matching_us,dequeue_us,scheduled,cancelled,batches,delivered,peak_queued"
    );
    for workload in workloads() {
        // Exact operation results, full ordered payloads, batch boundaries,
        // handles, cancellation results and intermediate queue state. No hash.
        let expected = run::<EventScheduler<EventKind>, true, false>(&workload)?;
        let candidate = run::<BucketQueue<EventKind>, true, false>(&workload)?;
        assert_eq!(candidate.trace, expected.trace, "{}", workload.name);
        assert_eq!(candidate.counts, expected.counts, "{}", workload.name);
        let counts = expected.counts;
        drop(expected.trace);
        drop(candidate);

        for queue in 0..queues.len() {
            assert_eq!(measure(queue, &workload, false)?.counts, counts);
        }
        let mut times = [Vec::new(), Vec::new()];
        for sample in 0..REPETITIONS {
            // Alternate order to reduce systematic first/second-run bias.
            for offset in 0..queues.len() {
                let queue = (sample + offset) % queues.len();
                let measured = measure(queue, &workload, false)?;
                assert_eq!(measured.counts, counts);
                times[queue].push(measured.timings.total);
                print_run(
                    "sample",
                    workload.name,
                    queues[queue],
                    Some(sample),
                    &measured,
                );
            }
        }
        for queue in 0..queues.len() {
            times[queue].sort_unstable();
            for (kind, index) in [
                ("median", REPETITIONS / 2),
                ("min", 0),
                ("max", REPETITIONS - 1),
            ] {
                let summary = Run {
                    timings: Timings {
                        total: times[queue][index],
                        ..Default::default()
                    },
                    counts: counts.clone(),
                    trace: Vec::new(),
                };
                print_run(kind, workload.name, queues[queue], None, &summary);
            }
            let profiled = measure(queue, &workload, true)?;
            assert_eq!(profiled.counts, counts);
            print_run("phase", workload.name, queues[queue], None, &profiled);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_workloads_deliver_identical_complete_traces() {
        for workload in workloads() {
            let expected = run::<EventScheduler<EventKind>, true, false>(&workload).unwrap();
            let candidate = run::<BucketQueue<EventKind>, true, false>(&workload).unwrap();
            assert_eq!(candidate.trace, expected.trace, "{}", workload.name);
            assert_eq!(candidate.counts, expected.counts, "{}", workload.name);
            if workload.name == "streaming-prediction-refresh" {
                assert_eq!(candidate.counts.cancelled, 15 * 128);
                assert_eq!(candidate.counts.batches, 16 * 128 + 128);
                assert_eq!(candidate.counts.peak_queued, 4_096 + 128);
            }
        }
    }
}
