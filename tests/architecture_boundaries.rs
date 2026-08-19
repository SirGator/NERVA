use std::{
    fs,
    path::Path,
    process::{self, Command},
};

use dsvlm_rust::{
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
        std::env::temp_dir().join(format!("dsvlm-primitives-boundary-{}.rmeta", process::id()));
    let output = Command::new("rustc")
        .current_dir(manifest_root)
        .args([
            "--crate-name",
            "dsvlm_primitives_boundary",
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
fn transduction_does_not_depend_on_the_environment_layer() {
    let transduction_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("transduction");

    for entry in fs::read_dir(transduction_root).expect("transduction directory is readable") {
        let path = entry.expect("transduction source entry is readable").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }

        let source = fs::read_to_string(&path).expect("transduction source is UTF-8 Rust text");
        assert!(
            !source.contains("crate::environment"),
            "{} depends upward on the environment layer",
            path.display()
        );
    }
}

#[test]
fn environment_paths_reexport_transduction_boundary_types() {
    let pattern: dsvlm_rust::transduction::Pattern = dsvlm_rust::environment::Pattern::A;
    let observation: dsvlm_rust::transduction::Observation =
        dsvlm_rust::environment::Observation::Pattern {
            at: SimTime::ZERO,
            pattern,
        };
    let action: dsvlm_rust::transduction::Action = dsvlm_rust::environment::Action::NoOp;

    assert_eq!(observation.time(), SimTime::ZERO);
    assert_eq!(action, dsvlm_rust::transduction::Action::NoOp);
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
