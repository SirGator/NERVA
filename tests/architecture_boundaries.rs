use std::{
    fs,
    path::Path,
    process::{self, Command},
};

use nerva::{
    core::{Event, EventKind, NeuronId, SimTime},
    io::ChannelId,
    nerves::{Fiber, FiberDirection, FiberId, Mapping, MappingError, Routing},
    roots::{Root, RootChannel, RootDirection, RootId, SensoryRoot},
    runtime::EventScheduler,
    transduction::ChannelSpike,
};

const RETIRED_LEGACY_MODULES: [&str; 7] = [
    "area", "common", "event", "network", "neuron", "presets", "synapse",
];

fn declared_root(id: u64, direction: RootDirection) -> Root {
    Root::new(
        RootId(id),
        "mapping test",
        direction,
        vec![RootChannel {
            channel: ChannelId(7),
        }],
    )
    .unwrap()
}

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
fn roots_compile_without_neurons_or_nerves_and_channels_only_declare_identity() {
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let temporary_base =
        std::env::temp_dir().join(format!("nerva-roots-boundary-{}", process::id()));
    let source_path = temporary_base.with_extension("rs");
    let metadata_path = temporary_base.with_extension("rmeta");
    let source = format!(
        r#"
#[path = {primitives:?}] pub mod primitives;
#[path = {io:?}] pub mod io;
#[path = {roots:?}] pub mod roots;

pub fn channel_identity() -> io::ChannelId {{
    // Construction and exhaustive destructuring protect the single-field API.
    let roots::RootChannel {{ channel }} = roots::RootChannel {{ channel: io::ChannelId(7) }};
    channel
}}
"#,
        primitives = source_root.join("primitives/mod.rs"),
        io = source_root.join("io/mod.rs"),
        roots = source_root.join("roots/mod.rs"),
    );
    fs::write(&source_path, source).expect("temporary boundary source can be written");
    let output = Command::new("rustc")
        .args([
            "--crate-name",
            "nerva_roots_boundary",
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
    fs::remove_file(&source_path).expect("temporary source can be removed");
    if metadata_path.exists() {
        fs::remove_file(&metadata_path).expect("temporary metadata can be removed");
    }
    assert!(
        output.status.success(),
        "roots must declare channels without owning physical associations:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn mapping_is_the_only_physical_channel_association_and_rejects_a_second_fiber() {
    let root = SensoryRoot::new(
        RootId(1),
        "sensor",
        vec![RootChannel {
            channel: ChannelId(7),
        }],
    )
    .unwrap();
    let mut mapping = Mapping::new();
    for (fiber, neuron) in [(FiberId(10), NeuronId(100)), (FiberId(99), NeuronId(200))] {
        mapping
            .add_fiber(Fiber::new(fiber, FiberDirection::Sensory, neuron, 2, 1.0).unwrap())
            .unwrap();
    }
    mapping
        .map_sensory(root.root(), ChannelId(7), FiberId(10))
        .unwrap();
    assert_eq!(
        mapping.map_sensory(root.root(), ChannelId(7), FiberId(99)),
        Err(MappingError::DuplicateChannel {
            root: RootId(1),
            channel: ChannelId(7)
        })
    );
    let impulse = Routing::sensory(
        &mapping,
        root.root().id,
        ChannelSpike {
            channel: root.root().channels[0].channel,
            at: SimTime(5),
            amplitude: 1.0,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(impulse.fiber, FiberId(10));
    assert_eq!(impulse.neuron, NeuronId(100));
    assert_eq!(impulse.arrives_at, SimTime(7));
}

#[test]
fn motor_channel_cannot_be_assigned_twice_even_across_different_neurons() {
    let root = declared_root(1, RootDirection::Motor);
    let mut mapping = Mapping::new();
    for (fiber, neuron) in [(FiberId(10), NeuronId(100)), (FiberId(99), NeuronId(200))] {
        mapping
            .add_fiber(Fiber::new(fiber, FiberDirection::Motor, neuron, 2, 1.0).unwrap())
            .unwrap();
    }
    mapping.map_motor(&root, ChannelId(7), FiberId(10)).unwrap();
    assert_eq!(
        mapping.map_motor(&root, ChannelId(7), FiberId(99)),
        Err(MappingError::DuplicateChannel {
            root: RootId(1),
            channel: ChannelId(7)
        })
    );
    let impulses = Routing::motor(&mapping, NeuronId(100), SimTime(5)).unwrap();
    assert_eq!(impulses.len(), 1);
    assert_eq!(impulses[0].fiber, FiberId(10));
    assert!(
        Routing::motor(&mapping, NeuronId(200), SimTime(5))
            .unwrap()
            .is_empty()
    );
    // The same channel identity may still exist in a different root.
    mapping
        .map_motor(
            &declared_root(2, RootDirection::Motor),
            ChannelId(7),
            FiberId(99),
        )
        .unwrap();
}

#[test]
fn a_root_channel_cannot_be_mapped_in_both_transport_directions() {
    let sensory = declared_root(1, RootDirection::Sensory);
    let motor = declared_root(1, RootDirection::Motor);
    for sensory_first in [true, false] {
        let mut mapping = Mapping::new();
        mapping
            .add_fiber(
                Fiber::new(FiberId(10), FiberDirection::Sensory, NeuronId(100), 1, 1.0).unwrap(),
            )
            .unwrap();
        mapping
            .add_fiber(
                Fiber::new(FiberId(99), FiberDirection::Motor, NeuronId(200), 1, 1.0).unwrap(),
            )
            .unwrap();
        let rejected = if sensory_first {
            mapping
                .map_sensory(&sensory, ChannelId(7), FiberId(10))
                .unwrap();
            mapping.map_motor(&motor, ChannelId(7), FiberId(99))
        } else {
            mapping
                .map_motor(&motor, ChannelId(7), FiberId(99))
                .unwrap();
            mapping.map_sensory(&sensory, ChannelId(7), FiberId(10))
        };
        assert_eq!(
            rejected,
            Err(MappingError::DuplicateChannel {
                root: RootId(1),
                channel: ChannelId(7)
            })
        );
    }
}

#[test]
fn mapping_validates_declared_channels_and_root_directions_before_binding() {
    let sensory = declared_root(1, RootDirection::Sensory);
    let motor = declared_root(2, RootDirection::Motor);
    let mut mapping = Mapping::new();
    mapping
        .add_fiber(Fiber::new(FiberId(10), FiberDirection::Sensory, NeuronId(100), 1, 1.0).unwrap())
        .unwrap();
    mapping
        .add_fiber(Fiber::new(FiberId(20), FiberDirection::Motor, NeuronId(200), 1, 1.0).unwrap())
        .unwrap();
    assert_eq!(
        mapping.map_sensory(&sensory, ChannelId(999), FiberId(10)),
        Err(MappingError::UnknownChannel {
            root: RootId(1),
            channel: ChannelId(999)
        })
    );
    assert_eq!(
        mapping.map_motor(&motor, ChannelId(999), FiberId(20)),
        Err(MappingError::UnknownChannel {
            root: RootId(2),
            channel: ChannelId(999)
        })
    );
    assert_eq!(
        mapping.map_sensory(&motor, ChannelId(7), FiberId(10)),
        Err(MappingError::RootDirectionMismatch {
            root: RootId(2),
            expected: RootDirection::Sensory,
            actual: RootDirection::Motor
        })
    );
    assert_eq!(
        mapping.map_motor(&sensory, ChannelId(7), FiberId(20)),
        Err(MappingError::RootDirectionMismatch {
            root: RootId(1),
            expected: RootDirection::Motor,
            actual: RootDirection::Sensory
        })
    );
    assert_eq!(
        mapping.map_sensory(&sensory, ChannelId(7), FiberId(99)),
        Err(MappingError::UnknownFiber(FiberId(99)))
    );
    assert_eq!(
        mapping.map_sensory(&sensory, ChannelId(7), FiberId(20)),
        Err(MappingError::DirectionMismatch(FiberId(20)))
    );
    assert_eq!(
        mapping.map_motor(&motor, ChannelId(7), FiberId(10)),
        Err(MappingError::DirectionMismatch(FiberId(10)))
    );
    assert!(mapping.sensory_fiber(RootId(1), ChannelId(7)).is_none());
    assert!(mapping.motor_fibers(NeuronId(200)).next().is_none());
    // All failed attempts leave the intended channels available for valid bindings.
    mapping
        .map_sensory(&sensory, ChannelId(7), FiberId(10))
        .unwrap();
    mapping
        .map_motor(&motor, ChannelId(7), FiberId(20))
        .unwrap();
    assert_eq!(
        mapping.sensory_fiber(RootId(1), ChannelId(7)).unwrap().id,
        FiberId(10)
    );
    assert_eq!(mapping.motor_fibers(NeuronId(200)).count(), 1);
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
