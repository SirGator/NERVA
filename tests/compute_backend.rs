use std::{
    cell::RefCell,
    fs,
    path::Path,
    process::{self, Command},
    rc::Rc,
};

use nerva::{
    compute::{ComputeBackend, ComputeError, CpuCompute, DenseNeuronBatch, FlatNeuronInputs},
    config::{NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Spike},
    learning::NoPlasticity,
    math::Position3D,
    runtime::{Simulation, SimulationError},
};

#[derive(Debug, PartialEq)]
enum Call {
    Advance(SimTime, Vec<NeuronId>),
    Integrate(SimTime, FlatNeuronInputs),
}

struct RecordingCompute(Rc<RefCell<Vec<Call>>>);

impl ComputeBackend for RecordingCompute {
    fn advance_neurons(
        &mut self,
        neurons: &mut DenseNeuronBatch,
        time: SimTime,
    ) -> Result<(), ComputeError> {
        self.0
            .borrow_mut()
            .push(Call::Advance(time, neurons.neuron_ids().to_vec()));
        CpuCompute.advance_neurons(neurons, time)
    }

    fn integrate_neurons(
        &mut self,
        neurons: &mut DenseNeuronBatch,
        inputs: &mut FlatNeuronInputs,
        time: SimTime,
    ) -> Result<Vec<Spike>, ComputeError> {
        for state in neurons.states() {
            assert_eq!(state.last_update, time);
        }
        self.0
            .borrow_mut()
            .push(Call::Integrate(time, inputs.clone()));
        CpuCompute.integrate_neurons(neurons, inputs, time)
    }
}

fn neuron(id: u64) -> Neuron {
    Neuron::new(
        NeuronId(id),
        Position3D::ORIGIN,
        Polarity::Excitatory,
        None,
        NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            ..Default::default()
        },
        SimTime::ZERO,
    )
    .unwrap()
}

#[test]
fn runtime_batches_compute_work_and_observes_live_neuron_edits() {
    let mut network = Network::new();
    for id in [91, 2, 17] {
        network.add_neuron(neuron(id)).unwrap();
    }
    let calls = Rc::new(RefCell::new(Vec::new()));
    let mut simulation = Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0)
        .unwrap()
        .with_compute_backend(RecordingCompute(calls.clone()));
    for (id, amplitude) in [(17, 0.6), (2, 1.1), (17, 0.5)] {
        simulation
            .schedule_external_input(SimTime(10), NeuronId(id), amplitude)
            .unwrap();
    }
    for _ in 0..2 {
        simulation
            .schedule_homeostasis(SimTime(10), NeuronId(91))
            .unwrap();
    }
    assert_eq!(simulation.run_until(SimTime(10)).unwrap().spikes_emitted, 2);
    assert_eq!(
        *calls.borrow(),
        vec![
            Call::Advance(SimTime(10), vec![NeuronId(2), NeuronId(17), NeuronId(91)]),
            Call::Integrate(
                SimTime(10),
                FlatNeuronInputs::from_rows([
                    (NeuronId(2), vec![1.1]),
                    (NeuronId(17), vec![0.6, 0.5]),
                ])
                .unwrap()
            ),
        ]
    );
    // A backend must see new/removed cells and current local state on its next call.
    simulation.remove_neuron(NeuronId(91)).unwrap();
    simulation.add_neuron(neuron(105)).unwrap();
    simulation
        .update_neuron(NeuronId(105), |cell| cell.set_threshold(0.25))
        .unwrap();
    simulation
        .schedule_external_input(SimTime(5_000), NeuronId(105), 0.3)
        .unwrap();
    assert_eq!(simulation.run().unwrap().spikes_emitted, 1);
    assert_eq!(
        &calls.borrow()[2..],
        &[
            Call::Advance(SimTime(5_000), vec![NeuronId(105)]),
            Call::Integrate(
                SimTime(5_000),
                FlatNeuronInputs::from_rows([(NeuronId(105), vec![0.3])]).unwrap()
            ),
        ]
    );
    simulation.network().validate_indices().unwrap();
}

#[test]
fn numerical_overflow_keeps_the_existing_runtime_error_and_poison_contract() {
    let mut network = Network::new();
    network.add_neuron(neuron(2)).unwrap();
    let mut simulation =
        Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0).unwrap();
    for _ in 0..2 {
        simulation
            .schedule_external_input(SimTime(10), NeuronId(2), f32::MAX)
            .unwrap();
    }
    assert!(matches!(
        simulation.step(),
        Err(SimulationError::SimulationPoisoned { .. })
    ));
    assert_eq!(
        simulation.poison_reason(),
        Some(&SimulationError::NonFiniteSummedInput {
            target: NeuronId(2)
        })
    );
    assert!(simulation.is_poisoned());
    assert_eq!(
        simulation
            .network()
            .neuron(NeuronId(2))
            .unwrap()
            .spike_count(),
        0
    );
    assert!(matches!(
        simulation.run(),
        Err(SimulationError::SimulationPoisoned { .. })
    ));
}

#[test]
fn compute_compiles_without_runtime_learning_io_or_development() {
    for include_compute in [false, true] {
        assert_lower_layers_compile(include_compute);
    }
}

fn assert_lower_layers_compile(include_compute: bool) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let base = std::env::temp_dir().join(format!(
        "nerva-compute-boundary-{}-{include_compute}",
        process::id()
    ));
    let source_path = base.with_extension("rs");
    let metadata_path = base.with_extension("rmeta");
    let source = format!(
        r#"
#[path = {primitives:?}] pub mod primitives;
#[path = {math:?}] pub mod math;
#[path = {config:?}] pub mod config;
#[path = {core:?}] pub mod core;
{compute_module}
"#,
        primitives = root.join("primitives/mod.rs"),
        math = root.join("math/mod.rs"),
        config = root.join("config/mod.rs"),
        core = root.join("core/mod.rs"),
        compute_module = if include_compute {
            format!(
                "#[path = {:?}] pub mod compute;",
                root.join("compute/mod.rs")
            )
        } else {
            String::new()
        },
    );
    fs::write(&source_path, source).unwrap();
    let output = Command::new("rustc")
        .args([
            "--crate-name",
            "nerva_compute_boundary",
            "--crate-type",
            "lib",
            "--edition=2024",
            "--emit=metadata",
            "-o",
        ])
        .arg(&metadata_path)
        .arg(&source_path)
        .output()
        .expect("rustc is available to Cargo tests");
    fs::remove_file(source_path).unwrap();
    if metadata_path.exists() {
        fs::remove_file(metadata_path).unwrap();
    }
    assert!(
        output.status.success(),
        "compute must compile using only the lower layers:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
