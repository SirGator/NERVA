use std::{
    fs,
    path::Path,
    process::{self, Command},
};

use nerva::{
    core::{Event, EventKind, NeuronId, SimTime},
    runtime::EventScheduler,
};

const RETIRED_LEGACY_MODULES: [&str; 7] = [
    "area", "common", "event", "network", "neuron", "presets", "synapse",
];

#[test]
fn retired_parallel_modules_do_not_return() {
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");

    for module in RETIRED_LEGACY_MODULES {
        assert!(
            !source_root.join(module).exists(),
            "retired parallel module src/{module}/ must not be reintroduced"
        );
        assert!(
            !source_root.join(format!("{module}.rs")).exists(),
            "retired parallel module src/{module}.rs must not be reintroduced"
        );
    }
}

#[test]
fn primitives_compile_without_any_higher_architectural_layer() {
    let manifest_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let metadata_path =
        std::env::temp_dir().join(format!("nerva-primitives-boundary-{}.rmeta", process::id()));
    let output = Command::new("rustc")
        .current_dir(manifest_root)
        .args([
            "--crate-name",
            "nerva_primitives_boundary",
            "--crate-type",
            "lib",
            "--edition=2024",
            "src/primitives/mod.rs",
            "--emit=metadata",
            "-o",
        ])
        .arg(&metadata_path)
        .output()
        .expect("rustc is available to Cargo tests");

    if output.status.success() {
        fs::remove_file(&metadata_path).expect("temporary metadata can be removed");
    }

    assert!(
        output.status.success(),
        "primitives must compile as a standalone lower layer:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn io_boundary_modules_do_not_depend_on_experiment_or_environment_types() {
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");

    for module in ["io", "transduction", "roots", "nerves"] {
        assert_device_neutral_sources(&source_root.join(module));
    }
}

fn assert_device_neutral_sources(directory: &Path) {
    for entry in fs::read_dir(directory).expect("boundary directory is readable") {
        let path = entry.expect("boundary source entry is readable").path();
        if path.is_dir() {
            assert_device_neutral_sources(&path);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            let source = fs::read_to_string(&path).expect("boundary source is UTF-8 Rust text");
            for forbidden in [
                "experiment",
                "environment::",
                "Pattern",
                "Observation",
                "Action",
                "Bit",
                "M0",
            ] {
                assert!(
                    !source.contains(forbidden),
                    "{} references experiment-specific type or module {forbidden}",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn experiment_m0_reexports_its_boundary_types() {
    let pattern: nerva::experiment::Pattern = nerva::experiment::Pattern::A;
    let observation: nerva::experiment::Observation = nerva::experiment::Observation::Pattern {
        at: SimTime::ZERO,
        pattern,
    };
    let action: nerva::experiment::Action = nerva::experiment::Action::NoOp;

    assert_eq!(observation.time(), SimTime::ZERO);
    assert_eq!(action, nerva::experiment::Action::NoOp);
}

#[test]
fn runtime_commands_keep_their_payload_through_timestamping_and_batching() {
    let command = EventKind::ExternalInput {
        target: NeuronId(5),
        amplitude: 0.75,
    };
    let timestamped_command = Event::new(SimTime(10), command);

    let mut scheduler = EventScheduler::new();
    scheduler
        .schedule(timestamped_command.time, timestamped_command.kind)
        .expect("runtime command can be scheduled");
    let batch = scheduler
        .pop_next_batch(1)
        .expect("batch is valid")
        .expect("one batch is due");

    assert_eq!(batch.time(), SimTime(10));
    assert_eq!(batch.events()[0].payload, command);
}
