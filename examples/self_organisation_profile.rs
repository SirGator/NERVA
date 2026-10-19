//! Spatial, asynchronously stimulated learning networks with local development.
//! This is a reproducible workload model, not a learned application task.
//! cargo run --release --features development --example self_organisation_profile
use std::{
    collections::BTreeSet,
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::{Duration, Instant},
};

use nerva::{
    config::{LearningConfig, NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    development::{
        CandidateSearchConfig, CreatedSynapse, DevelopmentPlan, FormationConfig, PlannedPruning,
        PruningConfig, PruningController,
    },
    learning::{ExcitatoryCausalUtility, PairStdp, UtilityDynamicsConfig},
    math::Position3D,
    primitives::Weight,
    runtime::{Histogram, ObservationEvent, ProfilePhase, RunReport, Simulation},
};

const WINDOWS: usize = 4;
const WINDOW_US: u64 = 100_000;
const FANOUT: usize = 8;
const REPETITIONS: usize = 7;
const HOST_PHASES: [&str; 6] = [
    "InputSchedule",
    "Runtime",
    "PruningObserve",
    "FormationPlan",
    "PruningCandidates",
    "DevelopmentCommit",
];
const PRUNING_THRESHOLD: f32 = 0.03;

// Specified SplitMix64, confined to input generation outside measured time.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

#[derive(Clone, Copy)]
struct Case {
    neurons: usize,
    seed: u64,
    development: bool,
}
impl Case {
    fn mode(self) -> &'static str {
        if self.development {
            "developing"
        } else {
            "fixed-topology"
        }
    }
}

#[derive(Clone, Copy)]
struct Input {
    time: SimTime,
    target: NeuronId,
    amplitude: f32,
}
fn neuron_id(index: usize) -> NeuronId {
    NeuronId(index as u64 * 3 + 17)
}

fn inputs(case: Case) -> Vec<Vec<Input>> {
    let mut rng = Rng(case.seed ^ 0xe4f0_a125_9a77_421b);
    let mut windows = vec![Vec::new(); WINDOWS];
    for index in 0..case.neurons {
        // Independent exponential waiting times, quantized upward to integer
        // microseconds. Background is subthreshold; local packets add drive.
        let mut time = 0_u64;
        loop {
            let uniform = (rng.next() >> 11) as f64 / (1_u64 << 53) as f64;
            let interval = (-(1.0 - uniform).ln() * 1_000_000.0 / 12.0).ceil() as u64;
            time += interval.max(1);
            if time >= WINDOWS as u64 * WINDOW_US {
                break;
            }
            windows[(time / WINDOW_US) as usize].push(Input {
                time: SimTime(time),
                target: neuron_id(index),
                amplitude: 0.7,
            });
        }
    }
    // Small local groups share packets, but each group has an independent
    // phase. Four of eight cells receive the packet, with a causal time lag.
    for group in 0..case.neurons.div_ceil(8) {
        let phase = 2_000 + rng.next() % 40_000;
        for packet in 0..8 {
            for member in 0..4 {
                let index = group * 8 + member;
                if index >= case.neurons {
                    continue;
                }
                let time = phase + packet * 50_000 + member as u64 * 75 + rng.next() % 20;
                windows[(time / WINDOW_US) as usize].push(Input {
                    time: SimTime(time),
                    target: neuron_id(index),
                    amplitude: 1.05,
                });
            }
        }
    }
    for window in &mut windows {
        window.sort_by_key(|input| (input.time, input.target));
    }
    windows
}

fn simulation(case: Case, profiling: bool) -> Result<Simulation<PairStdp>, Box<dyn Error>> {
    assert!(case.neurons >= 64);
    let width = (case.neurons as f64).sqrt().ceil() as usize;
    let mut network = Network::new();
    for index in 0..case.neurons {
        let mut neuron = Neuron::new(
            neuron_id(index),
            Position3D::new(
                (index % width) as f32 * 0.05,
                (index / width) as f32 * 0.05,
                0.0,
            ),
            if (index as u64 + case.seed).is_multiple_of(5) {
                Polarity::Inhibitory
            } else {
                Polarity::Excitatory
            },
            None,
            NeuronConfig {
                resting_potential: 0.0,
                reset_potential: 0.0,
                threshold: 1.0,
                refractory_period_us: 2_000,
                membrane_tau_us: 20_000.0,
                activity_trace_tau_us: 50_000.0,
                ..Default::default()
            },
            SimTime::ZERO,
        )?;
        // Explicit initial local demand, consumed by at most two formations
        // per selected cell. No external connection choices or global reward.
        if index % 16 == 3 {
            neuron.set_structural_drive_clamped(2.5, 0.0, 3.0)?;
        }
        network.add_neuron(neuron)?;
    }
    let offsets = [
        1,
        2,
        width,
        width + 1,
        case.neurons - 1,
        case.neurons - 2,
        case.neurons - width,
        case.neurons - width - 1,
    ];
    let mut rng = Rng(case.seed);
    for pre in 0..case.neurons {
        for (edge, offset) in offsets.iter().enumerate() {
            network.add_synapse(Synapse::new(
                SynapseId((pre * FANOUT + edge) as u64 * 7 + 5),
                neuron_id(pre),
                neuron_id((pre + offset) % case.neurons),
                Weight::new(0.06)?,
                10 + rng.next() % 191,
                true,
            )?)?;
        }
    }
    let mut simulation = Simulation::with_utility_rule(
        network,
        PairStdp::new(&LearningConfig {
            a_plus: 0.001,
            a_minus: 0.0012,
            min_weight: 0.01,
            max_weight: 0.1,
            ..Default::default()
        }),
        Box::new(ExcitatoryCausalUtility),
        RuntimeConfig::default(),
        1.0,
    )?
    .with_utility_dynamics(UtilityDynamicsConfig {
        eligibility_tau_us: 2_000.0,
        eta: 0.2,
        utility_tau_us: 200_000.0,
    })?;
    if profiling {
        simulation.enable_profiling();
    }
    Ok(simulation)
}

fn formation_config() -> FormationConfig {
    FormationConfig {
        initial_weight: Weight::new(0.06).unwrap(),
        delay_us: 100,
        candidate_search: CandidateSearchConfig {
            radius: 0.16,
            temporal_tau_us: 2_000.0,
            recency_tau_us: 100_000.0,
            min_candidate_score: 0.02,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[derive(Debug, PartialEq)]
struct Checkpoint {
    time: SimTime,
    created: Vec<CreatedSynapse>,
    pruned: Vec<PlannedPruning>,
    synapses: usize,
    pending: usize,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    report: RunReport,
    formed: usize,
    pruned: usize,
    final_synapses: usize,
    retained_new: usize,
    useful_new: usize,
    weight_changes: usize,
}

struct Run {
    simulation: Simulation<PairStdp>,
    elapsed: Duration,
    host: [Duration; 6],
    counts: Counts,
    checkpoints: Vec<Checkpoint>,
}

fn include(total: &mut RunReport, report: RunReport) {
    total.batches_processed += report.batches_processed;
    total.events_processed += report.events_processed;
    total.spikes_emitted += report.spikes_emitted;
    if report.last_time.is_some() {
        total.last_time = report.last_time;
    }
}

fn run(case: Case, input_windows: &[Vec<Input>], profiling: bool) -> Result<Run, Box<dyn Error>> {
    let mut simulation = simulation(case, profiling)?;
    let config = formation_config();
    let mut pruning = PruningController::new(PruningConfig {
        pruning_threshold: PRUNING_THRESHOLD,
        min_below_duration_us: 80_000,
        prune_non_plastic: false,
    })?;
    let mut cursor = SynapseId((case.neurons * FANOUT) as u64 * 7 + 100);
    let mut host = [Duration::ZERO; 6];
    let mut counts = Counts::default();
    let mut checkpoints = Vec::with_capacity(WINDOWS);
    let started = Instant::now();
    for (window, inputs) in input_windows.iter().enumerate() {
        let phase = Instant::now();
        for input in inputs {
            simulation.schedule_external_input(input.time, input.target, input.amplitude)?;
        }
        host[0] += phase.elapsed();
        let phase = Instant::now();
        include(
            &mut counts.report,
            simulation.run_until(SimTime((window as u64 + 1) * WINDOW_US - 1))?,
        );
        host[1] += phase.elapsed();
        if case.development {
            let phase = Instant::now();
            pruning.observe(&simulation)?;
            host[2] += phase.elapsed();
            let phase = Instant::now();
            let mut plan = DevelopmentPlan::build(&simulation, &config)?;
            host[3] += phase.elapsed();
            let phase = Instant::now();
            pruning.populate_plan(&mut plan, &simulation)?;
            host[4] += phase.elapsed();
            let phase = Instant::now();
            let created = if plan.is_empty() {
                Vec::new()
            } else {
                plan.commit(&mut simulation, &config, cursor)?
            };
            pruning.forget_plan(&plan);
            host[5] += phase.elapsed();
            if let Some(last) = created.last() {
                cursor = SynapseId(last.synapse_id.0 + 1);
            }
            counts.formed += created.len();
            counts.pruned += plan.prunings.len();
            checkpoints.push(Checkpoint {
                time: simulation.current_time(),
                created,
                pruned: plan.prunings,
                synapses: simulation.network().synapse_count(),
                pending: simulation.pending_event_count(),
            });
        }
    }
    // Drain the short in-flight tail within an explicit bounded horizon.
    let phase = Instant::now();
    include(
        &mut counts.report,
        simulation.run_until(SimTime(WINDOWS as u64 * WINDOW_US + 10_000))?,
    );
    host[1] += phase.elapsed();
    let elapsed = started.elapsed();
    assert_eq!(
        simulation.pending_event_count(),
        0,
        "unexpected activity past the bounded tail"
    );
    // Analysis and exact replay comparisons are outside measured time.
    counts.final_synapses = simulation.network().synapse_count();
    let formed: BTreeSet<_> = checkpoints
        .iter()
        .flat_map(|c| c.created.iter().map(|s| s.synapse_id))
        .collect();
    for synapse in simulation
        .network()
        .synapses()
        .filter(|s| formed.contains(&s.id()))
    {
        counts.retained_new += 1;
        if synapse.utility_at(
            simulation.current_time(),
            simulation.utility_dynamics().utility_tau_us,
        )? >= PRUNING_THRESHOLD
        {
            counts.useful_new += 1;
        }
    }
    counts.weight_changes = simulation
        .event_log()
        .iter()
        .filter(|event| matches!(event, ObservationEvent::WeightChanged { .. }))
        .count();
    assert_eq!(
        counts.final_synapses,
        case.neurons * FANOUT + counts.formed - counts.pruned
    );
    if let Some(profile) = simulation.profile() {
        assert_eq!(
            profile.scheduler_batch_size.total,
            counts.report.events_processed as u64
        );
        assert_eq!(
            profile.scheduler_batch_size.samples,
            counts.report.batches_processed as u64
        );
        assert_eq!(
            profile.phase(ProfilePhase::SchedulerDequeue).calls,
            profile.batches_started
        );
    }
    Ok(Run {
        simulation,
        elapsed,
        host,
        counts,
        checkpoints,
    })
}

fn assert_replay(reference: &Run, measured: &Run) {
    assert_eq!(measured.counts, reference.counts);
    assert_eq!(measured.checkpoints, reference.checkpoints);
    assert_eq!(
        measured.simulation.network(),
        reference.simulation.network()
    );
    assert_eq!(
        measured.simulation.event_log(),
        reference.simulation.event_log()
    );
    assert_eq!(
        measured.simulation.current_time(),
        reference.simulation.current_time()
    );
}

/// Optional full, untimed diagnostic dumps for direct cross-binary comparison.
/// Debug text is tied to this toolchain; it is not a persistence format.
fn write_replay(directory: &Path, case: Case, run: &Run) -> Result<(), Box<dyn Error>> {
    let path = directory.join(format!(
        "{}-{}-{}.txt",
        case.neurons,
        case.seed,
        case.mode()
    ));
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "{:?}", run.counts)?;
    writeln!(writer, "{:?}", run.checkpoints)?;
    writeln!(writer, "{:?}", run.simulation.network())?;
    writeln!(writer, "{:?}", run.simulation.event_log())?;
    writer.flush()?;
    Ok(())
}

fn row(
    kind: &str,
    case: Case,
    sample: &str,
    metric: &str,
    value: impl std::fmt::Display,
    unit: &str,
) {
    println!(
        "{kind},{},{},{},{sample},{metric},{value},{unit}",
        case.neurons,
        case.seed,
        case.mode()
    );
}
fn time_row(kind: &str, case: Case, sample: &str, metric: &str, time: Duration) {
    row(
        kind,
        case,
        sample,
        metric,
        format_args!("{:.3}", time.as_secs_f64() * 1e6),
        "us",
    );
}
fn times(kind: &str, case: Case, sample: &str, run: &Run) {
    time_row(kind, case, sample, "Total", run.elapsed);
    for (name, time) in HOST_PHASES.iter().zip(run.host) {
        time_row(kind, case, sample, name, time);
    }
}
fn histogram(case: Case, name: &str, histogram: &Histogram) {
    for (metric, value) in [
        ("samples", histogram.samples),
        ("total", histogram.total),
        ("max", histogram.max),
        ("singletons", histogram.singletons),
    ] {
        row(
            "histogram",
            case,
            "",
            &format!("{name}.{metric}"),
            value,
            "count",
        );
    }
    for (label, count) in ["0", "1-8", "9-32", "33-128", "129-512", "513+"]
        .iter()
        .zip(histogram.bins)
    {
        row(
            "histogram",
            case,
            "",
            &format!("{name}.bin.{label}"),
            count,
            "count",
        );
    }
}

fn print_profile(case: Case, run: &Run) {
    times("host-phase", case, "", run);
    let counts = &run.counts;
    for (name, count) in [
        ("events", counts.report.events_processed),
        ("batches", counts.report.batches_processed),
        ("spikes", counts.report.spikes_emitted),
        ("formed", counts.formed),
        ("pruned", counts.pruned),
        ("final_synapses", counts.final_synapses),
        ("retained_new", counts.retained_new),
        ("useful_new", counts.useful_new),
        ("weight_changes", counts.weight_changes),
    ] {
        row("count", case, "", name, count, "count");
    }
    for (index, checkpoint) in run.checkpoints.iter().enumerate() {
        for (name, value) in [
            ("time", checkpoint.time.0),
            ("formed", checkpoint.created.len() as u64),
            ("pruned", checkpoint.pruned.len() as u64),
            ("synapses", checkpoint.synapses as u64),
            ("pending", checkpoint.pending as u64),
        ] {
            row(
                "checkpoint",
                case,
                &index.to_string(),
                name,
                value,
                if name == "time" {
                    "simulation_us"
                } else {
                    "count"
                },
            );
        }
    }
    let profile = run.simulation.profile().expect("profiling enabled");
    for (name, histogram_value) in [
        ("scheduler", &profile.scheduler_batch_size),
        ("advance", &profile.advance_batch_size),
        ("integrate", &profile.integrate_batch_size),
        ("arrivals", &profile.synaptic_arrivals_per_timestamp),
        ("spikes", &profile.spikes_per_timestamp),
    ] {
        histogram(case, name, histogram_value);
    }
    for phase in ProfilePhase::ALL {
        let stats = profile.phase(phase);
        time_row(
            "runtime-phase",
            case,
            "",
            &format!("{phase:?}"),
            stats.total,
        );
        row(
            "runtime-calls",
            case,
            "",
            &format!("{phase:?}"),
            stats.calls,
            "count",
        );
        assert_eq!(stats.failures, 0);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let replay_directory = std::env::var_os("NERVA_REPLAY_DIR").map(std::path::PathBuf::from);
    if let Some(directory) = &replay_directory {
        std::fs::create_dir_all(directory)?;
    }
    println!("kind,neurons,seed,mode,sample,metric,value,unit");
    for neurons in [2_048, 8_192] {
        for seed in [7, 42, 2_026] {
            let mut cases = [
                Case {
                    neurons,
                    seed,
                    development: false,
                },
                Case {
                    neurons,
                    seed,
                    development: true,
                },
            ];
            let input_windows = inputs(cases[0]);
            let references = [
                run(cases[0], &input_windows, false)?,
                run(cases[1], &input_windows, false)?,
            ];
            if let Some(directory) = &replay_directory {
                for (case, reference) in cases.iter().zip(&references) {
                    write_replay(directory, *case, reference)?;
                }
            }
            let mut samples: [Vec<Duration>; 2] = [Vec::new(), Vec::new()];
            for sample in 0..REPETITIONS {
                // Interleave paired controls; alternate execution order.
                for offset in 0..2 {
                    let mode = (sample + offset) % 2;
                    let measured = run(cases[mode], &input_windows, false)?;
                    assert_replay(&references[mode], &measured);
                    samples[mode].push(measured.elapsed);
                    times("sample", cases[mode], &sample.to_string(), &measured);
                }
            }
            for (mode, case) in cases.iter_mut().enumerate() {
                samples[mode].sort_unstable();
                for (kind, index) in [
                    ("median", REPETITIONS / 2),
                    ("min", 0),
                    ("max", REPETITIONS - 1),
                ] {
                    time_row(kind, *case, "", "Total", samples[mode][index]);
                }
                let measured = run(*case, &input_windows, true)?;
                assert_replay(&references[mode], &measured);
                print_profile(*case, &measured);
            }
            eprintln!("completed {neurons} neurons, seed {seed}, both modes with exact replay");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_development_changes_topology_and_profiling_preserves_exact_replay() {
        let case = Case {
            neurons: 64,
            seed: 42,
            development: true,
        };
        let input_windows = inputs(case);
        let reference = run(case, &input_windows, false).unwrap();
        let measured = run(case, &input_windows, true).unwrap();
        assert_replay(&reference, &measured);
        assert!(measured.counts.formed > 0);
        assert!(measured.counts.pruned > 0);
        assert!(measured.counts.retained_new > 0);
        assert!(measured.counts.useful_new > 0);
        assert!(measured.counts.weight_changes > 0);
        let profile = measured.simulation.profile().unwrap();
        assert!(profile.scheduler_batch_size.singletons > 0);
        assert!(profile.scheduler_batch_size.singletons < profile.scheduler_batch_size.samples);
        let control = run(
            Case {
                development: false,
                ..case
            },
            &input_windows,
            false,
        )
        .unwrap();
        assert_eq!(control.counts.final_synapses, 64 * FANOUT);
        assert_eq!(control.counts.formed, 0);
        assert_eq!(control.counts.pruned, 0);
        assert!(control.counts.weight_changes > 0);
    }

    #[test]
    fn input_at_window_boundary_joins_its_in_flight_arrival_batch() {
        let case = Case {
            neurons: 64,
            seed: 42,
            development: false,
        };
        let initial = simulation(case, false).unwrap();
        let synapse = initial.network().synapses().next().unwrap();
        let mut windows = vec![Vec::new(); WINDOWS];
        windows[0].push(Input {
            time: SimTime(WINDOW_US - synapse.delay_us()),
            target: synapse.pre(),
            amplitude: 1.05,
        });
        windows[1].push(Input {
            time: SimTime(WINDOW_US),
            target: synapse.post(),
            amplitude: 0.2,
        });
        let measured = run(case, &windows, true).unwrap();
        let reference = run(case, &windows, false).unwrap();
        assert_replay(&reference, &measured);
        assert_eq!(
            measured
                .simulation
                .profile()
                .unwrap()
                .scheduler_batch_size
                .max,
            2
        );
    }

    #[test]
    fn seeds_change_activity_and_inputs_remain_in_their_host_windows() {
        let a = Case {
            neurons: 64,
            seed: 7,
            development: true,
        };
        let b = Case { seed: 42, ..a };
        let input_a = inputs(a);
        let input_b = inputs(b);
        assert_ne!(input_a[0][0].time, input_b[0][0].time);
        for (window, inputs) in input_a.iter().enumerate() {
            assert!(
                inputs
                    .windows(2)
                    .all(|pair| (pair[0].time, pair[0].target) <= (pair[1].time, pair[1].target))
            );
            assert!(
                inputs
                    .iter()
                    .all(|input| input.time.0 >= window as u64 * WINDOW_US
                        && input.time.0 < (window as u64 + 1) * WINDOW_US)
            );
        }
    }
}
