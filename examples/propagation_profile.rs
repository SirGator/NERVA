//! Repeated CPU propagation measurements on explicitly synthetic ring graphs.
//! Run in release mode; construction, input scheduling, checksums and teardown
//! are outside the measured run. One warm-up and seven samples per workload.
use std::{
    collections::hash_map::DefaultHasher,
    error::Error,
    hash::{Hash, Hasher},
    time::{Duration, Instant},
};

use nerva::{
    config::{LearningConfig, NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    learning::{ExcitatoryCausalUtility, NoUtility, PairStdp, UtilityRule},
    math::Position3D,
    primitives::Weight,
    runtime::{ProfilePhase, RunReport, RuntimeProfile, Simulation},
};

const ROUNDS: u64 = 16;
const REPETITIONS: usize = 7;

#[derive(Clone, Copy)]
struct Workload {
    neurons: usize,
    fanout: usize,
    learning: bool,
}

impl Workload {
    fn name(self) -> String {
        format!(
            "synthetic-{}-fanout-{}-{}",
            self.neurons,
            self.fanout,
            if self.learning {
                "stdp-utility"
            } else {
                "frozen"
            }
        )
    }

    fn simulation(self, profiling: bool) -> Result<Simulation<PairStdp>, Box<dyn Error>> {
        let mut network = Network::new();
        // Sparse stable IDs, mixed polarity and non-zero distances also exercise
        // the graph traversal rather than only the coincident excitatory case.
        let neuron_id = |index: usize| NeuronId(index as u64 * 3 + 17);
        for index in 0..self.neurons {
            network.add_neuron(Neuron::new(
                neuron_id(index),
                Position3D::new((index % 32) as f32 * 0.01, 0.0, 0.0),
                if index % 5 == 0 {
                    Polarity::Inhibitory
                } else {
                    Polarity::Excitatory
                },
                None,
                NeuronConfig {
                    resting_potential: 0.0,
                    reset_potential: 0.0,
                    threshold: 1.0,
                    refractory_period_us: 0,
                    ..Default::default()
                },
                SimTime::ZERO,
            )?)?;
        }
        let weight = 0.4 / self.fanout as f32;
        for pre in 0..self.neurons {
            for offset in 1..=self.fanout {
                let mut synapse = Synapse::new(
                    SynapseId((pre * self.fanout + offset) as u64 * 7),
                    neuron_id(pre),
                    neuron_id((pre + offset) % self.neurons),
                    Weight::new(weight)?,
                    10,
                    true,
                )?;
                if offset % 8 == 0 {
                    synapse.set_enabled(false);
                }
                network.add_synapse(synapse)?;
            }
        }
        let utility: Box<dyn UtilityRule> = if self.learning {
            Box::new(ExcitatoryCausalUtility)
        } else {
            Box::new(NoUtility)
        };
        let mut simulation = Simulation::with_utility_rule(
            network,
            PairStdp::new(&LearningConfig {
                // Keep the summed recurrent drive below threshold even as
                // learning changes weights; the workload remains finite.
                a_plus: weight * 0.01,
                a_minus: weight * 0.01,
                min_weight: 0.0,
                max_weight: weight * 1.5,
                ..Default::default()
            }),
            utility,
            RuntimeConfig::default(),
            1.0,
        )?;
        if !self.learning {
            simulation.freeze_learning();
        }
        if profiling {
            simulation.enable_profiling();
        }
        for round in 0..ROUNDS {
            for index in 0..self.neurons {
                simulation.schedule_external_input(
                    SimTime(round * 100_000),
                    neuron_id(index),
                    1.1,
                )?;
            }
        }
        Ok(simulation)
    }
}

struct Measurement {
    elapsed: Duration,
    report: RunReport,
    checksum: u64,
    profile: Option<RuntimeProfile>,
}

fn measure(workload: Workload, profiling: bool) -> Result<Measurement, Box<dyn Error>> {
    let mut simulation = workload.simulation(profiling)?;
    let started = Instant::now();
    let report = simulation.run()?;
    let elapsed = started.elapsed();
    // Fingerprint complete final graph state and ordered observations outside
    // the measured region. Comparable across binaries from this Rust toolchain;
    // this is a benchmark diagnostic, not a persistence format or test oracle.
    let mut hasher = DefaultHasher::new();
    format!("{:?}", simulation.network()).hash(&mut hasher);
    for event in simulation.event_log().iter() {
        format!("{event:?}").hash(&mut hasher);
    }
    Ok(Measurement {
        elapsed,
        report,
        checksum: hasher.finish(),
        profile: simulation.take_profile(),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("kind,workload,sample,run_us,batches,spikes,checksum");
    for workload in [
        Workload {
            neurons: 8,
            fanout: 4,
            learning: false,
        },
        Workload {
            neurons: 2_048,
            fanout: 4,
            learning: false,
        },
        Workload {
            neurons: 2_048,
            fanout: 32,
            learning: false,
        },
        Workload {
            neurons: 2_048,
            fanout: 4,
            learning: true,
        },
        Workload {
            neurons: 2_048,
            fanout: 32,
            learning: true,
        },
    ] {
        let warmup = measure(workload, false)?;
        let mut times = Vec::with_capacity(REPETITIONS);
        for sample in 0..REPETITIONS {
            let run = measure(workload, false)?;
            assert_eq!(run.report, warmup.report);
            assert_eq!(run.checksum, warmup.checksum);
            times.push(run.elapsed);
            println!(
                "sample,{},{sample},{:.3},{},{},{:016x}",
                workload.name(),
                run.elapsed.as_secs_f64() * 1e6,
                run.report.batches_processed,
                run.report.spikes_emitted,
                run.checksum,
            );
        }
        times.sort_unstable();
        println!(
            "summary,{},median_us={:.3},min_us={:.3},max_us={:.3}",
            workload.name(),
            times[REPETITIONS / 2].as_secs_f64() * 1e6,
            times[0].as_secs_f64() * 1e6,
            times[REPETITIONS - 1].as_secs_f64() * 1e6,
        );
        let measured = measure(workload, true)?;
        assert_eq!(measured.report, warmup.report);
        assert_eq!(measured.checksum, warmup.checksum);
        let profile = measured.profile.expect("profiling enabled");
        for phase in ProfilePhase::ALL {
            let stats = profile.phase(phase);
            println!(
                "phase,{},{phase:?},calls={},failures={},total_us={:.3}",
                workload.name(),
                stats.calls,
                stats.failures,
                stats.total.as_secs_f64() * 1e6,
            );
        }
    }
    Ok(())
}
