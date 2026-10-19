//! Measures actual M0 runs and explicitly synthetic synchronized workloads.
use nerva::{
    config::{NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    experiment::{M0Experiment, M0ExperimentConfig, M0Group, M0StudyConfig},
    learning::NoPlasticity,
    math::Position3D,
    primitives::Weight,
    runtime::{HISTOGRAM_LABELS, Histogram, ProfilePhase, RuntimeProfile, Simulation},
};

fn histogram(name: &str, hist: &Histogram) {
    println!(
        "histogram,{name},samples={},total={},max={},singletons={}",
        hist.samples, hist.total, hist.max, hist.singletons
    );
    for (label, count) in HISTOGRAM_LABELS.iter().zip(hist.bins) {
        println!("bin,{name},{label},{count}");
    }
}

fn print_profile(name: &str, profile: &RuntimeProfile) {
    println!(
        "workload,{name},started={},completed={},failed={}",
        profile.batches_started, profile.batches_completed, profile.batches_failed
    );
    for (label, hist) in [
        ("scheduler_batch", &profile.scheduler_batch_size),
        ("advance", &profile.advance_batch_size),
        ("integrate", &profile.integrate_batch_size),
        ("inputs_per_neuron", &profile.inputs_per_neuron),
        (
            "synaptic_arrivals",
            &profile.synaptic_arrivals_per_timestamp,
        ),
        ("spikes", &profile.spikes_per_timestamp),
        ("outgoing_per_spike", &profile.outgoing_synapses_per_spike),
    ] {
        histogram(label, hist);
    }
    for phase in ProfilePhase::ALL {
        let stats = profile.phase(phase);
        println!(
            "phase,{phase:?},calls={},failures={},total_us={:.3},max_us={:.3}",
            stats.calls,
            stats.failures,
            stats.total.as_secs_f64() * 1e6,
            stats.max.as_secs_f64() * 1e6
        );
    }
}

fn synchronized(size: usize) -> Result<RuntimeProfile, Box<dyn std::error::Error>> {
    let mut network = Network::new();
    for id in 0..size {
        network.add_neuron(Neuron::new(
            NeuronId(id as u64),
            Position3D::ORIGIN,
            Polarity::Excitatory,
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
    for pre in 0..size {
        for offset in 1..=4.min(size - 1) {
            network.add_synapse(Synapse::new(
                SynapseId((pre * 4 + offset) as u64),
                NeuronId(pre as u64),
                NeuronId(((pre + offset) % size) as u64),
                Weight::new(0.1)?,
                10,
                false,
            )?)?;
        }
    }
    let mut simulation =
        Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0)?.with_profiling();
    for round in 0..16 {
        for id in 0..size {
            simulation.schedule_external_input(
                SimTime(round * 100_000),
                NeuronId(id as u64),
                1.1,
            )?;
        }
    }
    simulation.run()?;
    Ok(simulation.take_profile().expect("profiling enabled"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut profile = RuntimeProfile::default();
    for seed in M0StudyConfig::default().seeds {
        let mut config = M0ExperimentConfig::default();
        config.nerva.network.seed = seed;
        for group in [
            M0Group::OrderedLearning,
            M0Group::OrderedFixed,
            M0Group::RandomLearning,
            M0Group::OrderedHomeostasis,
        ] {
            let (_, run) = M0Experiment::new(config.clone(), group)?.run_profiled()?;
            profile.merge(&run);
        }
    }
    print_profile("M0-six-seeds-four-groups-training-and-two-probes", &profile);
    for size in [1, 8, 128, 2_048] {
        print_profile(
            &format!("synthetic-synchronized-{size}"),
            &synchronized(size)?,
        );
    }
    Ok(())
}
