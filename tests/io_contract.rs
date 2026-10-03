use std::collections::VecDeque;

use nerva::{
    config::{NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, NeuronRole, Polarity, SimTime, Synapse, SynapseId},
    io::{ChannelId, Effector, EffectorSignal, Receptor, ReceptorSignal},
    learning::NoPlasticity,
    math::Position3D,
    nerves::{Fiber, FiberDirection, FiberId, Mapping, Routing},
    primitives::Weight,
    roots::{MotorOutput, MotorRoot, RootChannel, RootId, SensoryRoot},
    runtime::{ObservationEvent, Simulation},
    transduction::{
        DirectMotorTransducer, DirectSensoryTransducer, MotorTransducer, SensoryTransducer,
    },
};

const SENSORY_CHANNEL: ChannelId = ChannelId(7);
const MOTOR_CHANNEL: ChannelId = ChannelId(9);
const SENSORY_NEURON: NeuronId = NeuronId(100);
const MOTOR_NEURON: NeuronId = NeuronId(200);

struct DummyReceptor {
    pending: VecDeque<ReceptorSignal>,
}

impl Receptor for DummyReceptor {
    fn observations_until(&mut self, until: SimTime) -> Vec<ReceptorSignal> {
        let due = self
            .pending
            .iter()
            .take_while(|signal| signal.at <= until)
            .count();
        self.pending.drain(..due).collect()
    }
}

#[derive(Default)]
struct DummyEffector {
    received: Vec<EffectorSignal>,
}

impl Effector for DummyEffector {
    fn apply(&mut self, signal: EffectorSignal) {
        self.received.push(signal);
    }
}

fn neuron(id: NeuronId, role: NeuronRole) -> Neuron {
    Neuron::new(
        id,
        Position3D::ORIGIN,
        Polarity::Excitatory,
        Some(role),
        NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            membrane_tau_us: 10_000.0,
            refractory_period_us: 0,
            activity_trace_tau_us: 10_000.0,
            intrinsic: Default::default(),
        },
        SimTime::ZERO,
    )
    .expect("valid I/O integration neuron")
}

#[test]
fn receptor_reaches_effector_through_real_nerva_dynamics() {
    let sensory_root_id = RootId(1);
    let motor_root_id = RootId(2);
    let sensory_fiber = FiberId(10);
    let motor_fiber = FiberId(20);

    let sensory_root = SensoryRoot::new(
        sensory_root_id,
        "dummy receptor",
        vec![RootChannel {
            channel: SENSORY_CHANNEL,
            fiber: sensory_fiber,
        }],
    )
    .expect("valid sensory root");
    let mut motor_root = MotorRoot::new(
        motor_root_id,
        "dummy effector",
        vec![RootChannel {
            channel: MOTOR_CHANNEL,
            fiber: motor_fiber,
        }],
    )
    .expect("valid motor root");

    let mut mapping = Mapping::new();
    mapping
        .add_fiber(
            Fiber::new(
                sensory_fiber,
                FiberDirection::Sensory,
                SENSORY_NEURON,
                1,
                1.0,
            )
            .expect("valid sensory fiber"),
        )
        .expect("unique sensory fiber");
    mapping
        .map_sensory(sensory_root_id, SENSORY_CHANNEL, sensory_fiber)
        .expect("unique sensory channel");
    mapping
        .add_fiber(
            Fiber::new(motor_fiber, FiberDirection::Motor, MOTOR_NEURON, 1, 1.0)
                .expect("valid motor fiber"),
        )
        .expect("unique motor fiber");
    mapping
        .map_motor(motor_root_id, MOTOR_CHANNEL, motor_fiber)
        .expect("unique motor channel");

    let mut network = Network::new();
    network
        .add_neuron(neuron(SENSORY_NEURON, NeuronRole::Sensory))
        .expect("unique sensory neuron");
    network
        .add_neuron(neuron(MOTOR_NEURON, NeuronRole::Motor))
        .expect("unique motor neuron");
    network
        .add_synapse(
            Synapse::new(
                SynapseId(1),
                SENSORY_NEURON,
                MOTOR_NEURON,
                Weight::new(1.0).expect("valid weight"),
                1,
                false,
            )
            .expect("valid synapse"),
        )
        .expect("unique synapse");

    let mut simulation = Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 1.0)
        .expect("valid simulation");
    let mut receptor = DummyReceptor {
        pending: VecDeque::from([ReceptorSignal {
            channel: SENSORY_CHANNEL,
            at: SimTime(10),
            value: 1.0,
        }]),
    };
    let mut effector = DummyEffector::default();
    let mut sensory_transducer = DirectSensoryTransducer::new();
    let sensory: &mut dyn SensoryTransducer = &mut sensory_transducer;
    let mut motor_transducer = DirectMotorTransducer::new();
    let motor: &mut dyn MotorTransducer = &mut motor_transducer;

    for signal in receptor.observations_until(SimTime(10)) {
        sensory.push(signal).expect("valid receptor signal");
    }
    let mut spikes = Vec::new();
    sensory
        .advance_until(SimTime(10), &mut spikes)
        .expect("sensory transduction reaches the input horizon");
    for spike in spikes {
        let impulse = Routing::sensory(&mapping, sensory_root.root().id, spike)
            .expect("valid sensory routing")
            .expect("mapped receptor channel");
        simulation
            .schedule_external_input(impulse.arrives_at, impulse.neuron, impulse.amplitude)
            .expect("valid external input");
    }

    let report = simulation
        .run_until(SimTime(12))
        .expect("real neural dynamics complete");
    assert_eq!(report.spikes_emitted, 2);

    let motor_spike = simulation
        .event_log()
        .iter()
        .find_map(|event| match event {
            ObservationEvent::SpikeEmitted(spike) if spike.neuron_id == MOTOR_NEURON => {
                Some(*spike)
            }
            _ => None,
        })
        .expect("motor neuron emitted a real spike");
    assert_eq!(motor_spike.time, SimTime(12));

    for impulse in Routing::motor(&mapping, motor_spike.neuron_id, motor_spike.time)
        .expect("valid motor routing")
    {
        assert_eq!(impulse.root, motor_root.root().id);
        motor_root.observe(MotorOutput {
            channel: impulse.channel,
            at: impulse.arrives_at,
            amplitude: impulse.amplitude,
        });
    }
    for output in motor_root.drain_outputs() {
        motor.push(output).expect("valid motor output");
    }
    let mut signals = Vec::new();
    motor
        .advance_until(SimTime(12), &mut signals)
        .expect("motor transduction reaches the neural horizon");
    assert!(signals.is_empty(), "motor fiber is still conducting");
    motor
        .advance_until(SimTime(13), &mut signals)
        .expect("motor transduction reaches the arrival horizon");
    for signal in signals.drain(..) {
        effector.apply(signal);
    }

    assert_eq!(
        effector.received,
        [EffectorSignal {
            channel: MOTOR_CHANNEL,
            at: SimTime(13),
            value: 1.0,
        }]
    );
    assert!(receptor.pending.is_empty());
    motor
        .advance_until(SimTime(20), &mut signals)
        .expect("motor transduction advances through silence");
    assert!(signals.is_empty(), "one spike produces only one pulse");
}
