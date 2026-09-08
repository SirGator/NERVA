//! Neutral orchestration of one environment-to-core-to-environment cycle.

use std::{collections::BTreeMap, error::Error, fmt};

use crate::{
    core::{SimTime, Spike},
    environment::{Environment, Observation},
    learning::PlasticityRule,
    nerves::{FiberImpulse, Mapping, Routing, RoutingError},
    roots::{MotorOutput, MotorRoot, RootId},
    runtime::{ObservationEvent, RunReport, Simulation, SimulationError},
    transduction::{Decoder, Encoder, EncodingError},
};

/// Counts and runtime progress produced by one closed-loop boundary cycle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClosedLoopReport {
    /// Environment observations consumed in this cycle.
    pub observations: usize,
    /// Root-local spikes emitted by the encoder.
    pub encoded_spikes: usize,
    /// Sensory impulses scheduled into the runtime.
    pub sensory_impulses: usize,
    /// Core batches/events/spikes processed through the requested horizon.
    pub runtime: RunReport,
    /// Due impulses delivered to the configured motor root.
    pub motor_outputs: usize,
    /// Decoded actions passed to [`Environment::apply_action`].
    pub actions_applied: usize,
    /// Immutable motor outputs delivered during this call.
    pub motor_output_events: Vec<MotorOutput>,
}

/// Failure at one explicit slice boundary of [`ClosedLoop`].
#[derive(Clone, Debug, PartialEq)]
pub enum ClosedLoopError {
    /// Observation-to-channel transduction failed.
    Encoding(EncodingError),
    /// Fixed nerve transport failed.
    Routing(RoutingError),
    /// Scheduling or executing core events failed.
    Simulation(SimulationError),
    /// The encoder emitted a channel absent from the configured sensory root.
    UnmappedSensoryChannel {
        /// Sensory root used by this adapter.
        root: RootId,
        /// Channel emitted by the encoder.
        channel: u16,
    },
    /// The supplied mapping routes a motor spike to a root not owned by this
    /// single-root adapter.
    UnexpectedMotorRoot {
        /// Root owned by the adapter.
        expected: RootId,
        /// Root found in the fixed mapping.
        actual: RootId,
    },
    /// A motor mapping targets a channel/fiber pair not declared by the root.
    UnexpectedMotorChannel {
        /// Motor root receiving the impulse.
        root: RootId,
        /// Mapped root-local channel.
        channel: u16,
        /// Mapped fixed fiber.
        fiber: crate::nerves::FiberId,
    },
    /// Closed-loop horizons must stay monotonic even if no runtime event was due.
    TimeWentBackwards {
        /// Most recent successful horizon or processed runtime time.
        current: SimTime,
        /// Earlier requested horizon.
        requested: SimTime,
    },
    /// The supplied runtime already contains events that the adapter cannot
    /// place on its external causal timeline.
    PendingRuntimeEvents {
        /// Number of events already queued at construction time.
        count: usize,
    },
    /// An environment returned an observation outside the requested monotonic
    /// interval.
    ObservationTimeOutOfRange {
        /// Timestamp returned by the environment.
        observation_time: SimTime,
        /// Earliest timestamp still causally admissible.
        earliest: SimTime,
        /// Latest timestamp requested from the environment.
        latest: SimTime,
    },
    /// An encoder attempted to place a spike before its source observation.
    EncodedSpikeBeforeObservation {
        /// Source observation timestamp.
        observation_time: SimTime,
        /// Invalid encoded spike timestamp.
        spike_time: SimTime,
    },
    /// The adapter's shadow schedule and the runtime batch disagreed.
    RuntimeScheduleDiverged {
        /// Timestamp expected by the adapter.
        expected_time: SimTime,
        /// Timestamp actually processed by the runtime, if it had a batch.
        actual_time: Option<SimTime>,
        /// Number of events expected in the timestamp batch.
        expected_events: usize,
        /// Number actually processed.
        actual_events: usize,
    },
    /// A successfully emitted core spike could not be mirrored on the adapter's
    /// causal schedule because its synaptic delay overflowed.
    RuntimePropagationTimeOverflow {
        /// Spike emission timestamp.
        spike_time: SimTime,
        /// Synaptic delay that did not fit.
        delay_us: u64,
    },
    /// `next_observation_time` promised a due item but the environment did not
    /// release one at that timestamp.
    EnvironmentScheduleDiverged {
        /// Timestamp advertised by the environment.
        expected_time: SimTime,
    },
}

impl fmt::Display for ClosedLoopError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encoding(error) => write!(formatter, "observation encoding failed: {error}"),
            Self::Routing(error) => write!(formatter, "nerve routing failed: {error}"),
            Self::Simulation(error) => write!(formatter, "core simulation failed: {error}"),
            Self::UnmappedSensoryChannel { root, channel } => write!(
                formatter,
                "sensory root {root:?} has no mapped fiber for encoded channel {channel}"
            ),
            Self::UnexpectedMotorRoot { expected, actual } => write!(
                formatter,
                "motor mapping targets root {actual:?}, but this adapter owns {expected:?}"
            ),
            Self::UnexpectedMotorChannel {
                root,
                channel,
                fiber,
            } => write!(
                formatter,
                "motor mapping targets undeclared channel {channel} and fiber {fiber:?} on root {root:?}"
            ),
            Self::TimeWentBackwards { current, requested } => write!(
                formatter,
                "cannot move closed-loop horizon backwards from {current} to {requested}"
            ),
            Self::PendingRuntimeEvents { count } => write!(
                formatter,
                "closed-loop runtime must start with an empty scheduler, found {count} pending events"
            ),
            Self::ObservationTimeOutOfRange {
                observation_time,
                earliest,
                latest,
            } => write!(
                formatter,
                "environment observation at {observation_time} lies outside causal interval [{earliest}, {latest}]"
            ),
            Self::EncodedSpikeBeforeObservation {
                observation_time,
                spike_time,
            } => write!(
                formatter,
                "encoder placed spike at {spike_time} before its observation at {observation_time}"
            ),
            Self::RuntimeScheduleDiverged {
                expected_time,
                actual_time,
                expected_events,
                actual_events,
            } => write!(
                formatter,
                "runtime schedule diverged at {expected_time}: processed {actual_events} events at {actual_time:?}, expected {expected_events}"
            ),
            Self::RuntimePropagationTimeOverflow {
                spike_time,
                delay_us,
            } => write!(
                formatter,
                "adding core synaptic delay {delay_us} us to spike time {spike_time} overflows the closed-loop schedule"
            ),
            Self::EnvironmentScheduleDiverged { expected_time } => write!(
                formatter,
                "environment advertised an observation at {expected_time} but released none"
            ),
        }
    }
}

impl Error for ClosedLoopError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Encoding(error) => Some(error),
            Self::Routing(error) => Some(error),
            Self::Simulation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<EncodingError> for ClosedLoopError {
    fn from(error: EncodingError) -> Self {
        Self::Encoding(error)
    }
}

impl From<RoutingError> for ClosedLoopError {
    fn from(error: RoutingError) -> Self {
        Self::Routing(error)
    }
}

impl From<SimulationError> for ClosedLoopError {
    fn from(error: SimulationError) -> Self {
        Self::Simulation(error)
    }
}

/// Boundary-only adapter connecting existing slices without touching core state
/// directly or assigning meaning to neuronal activity.
///
/// A call to [`Self::cycle_until`] advances one shared causal frontier. Actions
/// may expose new observations at the same timestamp; those are consumed in the
/// same call before later work. Delayed motor impulses remain pending until
/// their exact arrival timestamp becomes due.
pub struct ClosedLoop<E, C, D, R>
where
    E: Environment,
    C: Encoder,
    D: Decoder,
    R: PlasticityRule,
{
    environment: E,
    encoder: C,
    mapping: Mapping,
    sensory_root: RootId,
    motor_root: MotorRoot,
    decoder: D,
    simulation: Simulation<R>,
    runtime_event_cursor: usize,
    pending_observations: Vec<Observation>,
    pending_sensory: Vec<FiberImpulse>,
    pending_motor: Vec<FiberImpulse>,
    /// Exact timestamp/count mirror of events scheduled through this adapter or
    /// causally emitted by the runtime.
    runtime_schedule: BTreeMap<SimTime, usize>,
    last_boundary_time: SimTime,
}

impl<E, C, D, R> ClosedLoop<E, C, D, R>
where
    E: Environment,
    C: Encoder,
    D: Decoder,
    R: PlasticityRule,
{
    /// Connects an environment and transducers to fixed nerves and one runtime.
    pub fn new(
        environment: E,
        encoder: C,
        mapping: Mapping,
        sensory_root: RootId,
        motor_root: MotorRoot,
        decoder: D,
        simulation: Simulation<R>,
    ) -> Result<Self, ClosedLoopError> {
        let pending_events = simulation.pending_event_count();
        if pending_events != 0 {
            return Err(ClosedLoopError::PendingRuntimeEvents {
                count: pending_events,
            });
        }
        let runtime_event_cursor = simulation.event_log().len();
        let last_boundary_time = simulation.current_time();
        Ok(Self {
            environment,
            encoder,
            mapping,
            sensory_root,
            motor_root,
            decoder,
            simulation,
            runtime_event_cursor,
            pending_observations: Vec::new(),
            pending_sensory: Vec::new(),
            pending_motor: Vec::new(),
            runtime_schedule: BTreeMap::new(),
            last_boundary_time,
        })
    }

    /// Executes one neutral observation/input/runtime/output/action pass through
    /// `inclusive` in exact causal order.
    ///
    /// Observations, sensory arrivals, runtime batches and motor arrivals share
    /// one ordered frontier. In particular, a motor action at `t` is applied and
    /// its environment feedback is encoded before the runtime advances past
    /// `t`, even when `inclusive` is much later.
    pub fn cycle_until(&mut self, inclusive: SimTime) -> Result<ClosedLoopReport, ClosedLoopError> {
        let current = self.last_boundary_time.max(self.simulation.current_time());
        if inclusive < current {
            return Err(ClosedLoopError::TimeWentBackwards {
                current,
                requested: inclusive,
            });
        }

        let mut report = ClosedLoopReport::default();

        while let Some(next_time) = self.next_boundary_time() {
            if next_time > inclusive {
                break;
            }
            if next_time < current {
                return Err(ClosedLoopError::TimeWentBackwards {
                    current,
                    requested: next_time,
                });
            }

            let environment_due = self.environment.next_observation_time() == Some(next_time);
            let pending_before = self.pending_observations.len();
            self.collect_environment_observations(next_time, next_time)?;
            if environment_due && self.pending_observations.len() == pending_before {
                return Err(ClosedLoopError::EnvironmentScheduleDiverged {
                    expected_time: next_time,
                });
            }

            self.encode_observations_at(next_time, &mut report)?;
            self.schedule_sensory_at(next_time, &mut report)?;
            self.process_runtime_at(next_time, &mut report)?;
            self.deliver_motor_at(next_time, &mut report);
        }

        self.last_boundary_time = inclusive;
        Ok(report)
    }

    /// Read-only access to the external state.
    pub fn environment(&self) -> &E {
        &self.environment
    }

    /// Explicit mutable access for environment setup between cycles.
    pub fn environment_mut(&mut self) -> &mut E {
        &mut self.environment
    }

    /// Read-only access to the runtime and neural state.
    pub fn simulation(&self) -> &Simulation<R> {
        &self.simulation
    }

    /// Passive motor root owned by the adapter.
    pub fn motor_root(&self) -> &MotorRoot {
        &self.motor_root
    }

    /// Motor impulses waiting for their exact conduction-arrival time.
    pub fn pending_motor_count(&self) -> usize {
        self.pending_motor.len()
    }

    fn next_boundary_time(&self) -> Option<SimTime> {
        [
            self.environment.next_observation_time(),
            self.pending_observations.first().map(Observation::time),
            self.pending_sensory
                .first()
                .map(|impulse| impulse.arrives_at),
            self.runtime_schedule
                .first_key_value()
                .map(|(time, _)| *time),
            self.pending_motor.first().map(|impulse| impulse.arrives_at),
        ]
        .into_iter()
        .flatten()
        .min()
    }

    fn collect_environment_observations(
        &mut self,
        earliest: SimTime,
        latest: SimTime,
    ) -> Result<(), ClosedLoopError> {
        let observations = self.environment.observations(latest);
        for observation in &observations {
            let observation_time = observation.time();
            if observation_time < earliest || observation_time > latest {
                return Err(ClosedLoopError::ObservationTimeOutOfRange {
                    observation_time,
                    earliest,
                    latest,
                });
            }
        }
        self.pending_observations.extend(observations);
        self.pending_observations.sort_by_key(Observation::time);
        Ok(())
    }

    fn encode_observations_at(
        &mut self,
        time: SimTime,
        report: &mut ClosedLoopReport,
    ) -> Result<(), ClosedLoopError> {
        let due_count = self
            .pending_observations
            .partition_point(|observation| observation.time() <= time);
        let due: Vec<_> = self.pending_observations.drain(..due_count).collect();

        for observation in due {
            let observation_time = observation.time();
            let spikes = self.encoder.encode(&observation)?;
            report.observations = report.observations.saturating_add(1);
            report.encoded_spikes = report.encoded_spikes.saturating_add(spikes.len());
            for spike in spikes {
                if spike.at < observation_time {
                    return Err(ClosedLoopError::EncodedSpikeBeforeObservation {
                        observation_time,
                        spike_time: spike.at,
                    });
                }
                let channel = spike.channel;
                let impulse = Routing::sensory(&self.mapping, self.sensory_root, spike)?.ok_or(
                    ClosedLoopError::UnmappedSensoryChannel {
                        root: self.sensory_root,
                        channel,
                    },
                )?;
                self.pending_sensory.push(impulse);
            }
        }
        self.pending_sensory.sort_by_key(|impulse| {
            (
                impulse.arrives_at,
                impulse.root,
                impulse.channel,
                impulse.fiber,
            )
        });
        Ok(())
    }

    fn schedule_sensory_at(
        &mut self,
        time: SimTime,
        report: &mut ClosedLoopReport,
    ) -> Result<(), ClosedLoopError> {
        let due_count = self
            .pending_sensory
            .partition_point(|impulse| impulse.arrives_at <= time);
        let due: Vec<_> = self.pending_sensory.drain(..due_count).collect();
        for impulse in due {
            self.simulation.schedule_external_input(
                impulse.arrives_at,
                impulse.neuron,
                impulse.amplitude,
            )?;
            *self.runtime_schedule.entry(time).or_default() += 1;
            report.sensory_impulses = report.sensory_impulses.saturating_add(1);
        }
        Ok(())
    }

    fn process_runtime_at(
        &mut self,
        time: SimTime,
        report: &mut ClosedLoopReport,
    ) -> Result<(), ClosedLoopError> {
        let Some(expected_events) = self.runtime_schedule.remove(&time) else {
            return Ok(());
        };

        let batch = self.simulation.step()?;
        let actual_time = batch.map(|batch| batch.time);
        let actual_events = batch.map_or(0, |batch| batch.events_processed);
        if actual_time != Some(time) || actual_events != expected_events {
            return Err(ClosedLoopError::RuntimeScheduleDiverged {
                expected_time: time,
                actual_time,
                expected_events,
                actual_events,
            });
        }
        let batch = batch.expect("the successful schedule comparison proves a batch exists");
        report.runtime.batches_processed = report.runtime.batches_processed.saturating_add(1);
        report.runtime.events_processed = report
            .runtime
            .events_processed
            .saturating_add(batch.events_processed);
        report.runtime.spikes_emitted = report
            .runtime
            .spikes_emitted
            .saturating_add(batch.spikes_emitted);
        report.runtime.last_time = Some(batch.time);

        self.collect_runtime_effects()
    }

    fn collect_runtime_effects(&mut self) -> Result<(), ClosedLoopError> {
        let new_spikes: Vec<Spike> = self.simulation.event_log()[self.runtime_event_cursor..]
            .iter()
            .filter_map(|event| match event {
                ObservationEvent::SpikeEmitted(spike) => Some(*spike),
                _ => None,
            })
            .collect();

        let expected_root = self.motor_root.root().id;
        let mut new_impulses = Vec::new();
        for spike in new_spikes {
            for &synapse_id in self
                .simulation
                .network()
                .outgoing_synapse_ids(spike.neuron_id)
            {
                let synapse = self
                    .simulation
                    .network()
                    .synapse(synapse_id)
                    .expect("validated adjacency references a stored synapse");
                if synapse.is_enabled() {
                    let arrives_at = spike.time.checked_add_us(synapse.delay_us()).ok_or(
                        ClosedLoopError::RuntimePropagationTimeOverflow {
                            spike_time: spike.time,
                            delay_us: synapse.delay_us(),
                        },
                    )?;
                    *self.runtime_schedule.entry(arrives_at).or_default() += 1;
                }
            }

            for impulse in Routing::motor(&self.mapping, spike.neuron_id, spike.time)? {
                if impulse.root != expected_root {
                    return Err(ClosedLoopError::UnexpectedMotorRoot {
                        expected: expected_root,
                        actual: impulse.root,
                    });
                }
                let channel_is_declared = self.motor_root.root().channels.iter().any(|declared| {
                    declared.channel == impulse.channel && declared.fiber == impulse.fiber
                });
                if !channel_is_declared {
                    return Err(ClosedLoopError::UnexpectedMotorChannel {
                        root: expected_root,
                        channel: impulse.channel,
                        fiber: impulse.fiber,
                    });
                }
                new_impulses.push(impulse);
            }
        }

        self.runtime_event_cursor = self.simulation.event_log().len();
        self.pending_motor.extend(new_impulses);
        self.pending_motor.sort_by_key(|impulse| {
            (
                impulse.arrives_at,
                impulse.root,
                impulse.channel,
                impulse.fiber,
            )
        });
        Ok(())
    }

    fn deliver_motor_at(&mut self, time: SimTime, report: &mut ClosedLoopReport) -> bool {
        let due_count = self
            .pending_motor
            .partition_point(|impulse| impulse.arrives_at <= time);
        let due: Vec<_> = self.pending_motor.drain(..due_count).collect();

        for impulse in due {
            self.motor_root.observe(MotorOutput {
                channel: impulse.channel,
                at: impulse.arrives_at,
                amplitude: impulse.amplitude,
            });
        }
        let outputs = self.motor_root.drain_outputs();
        report.motor_outputs = report.motor_outputs.saturating_add(outputs.len());
        report.motor_output_events.extend(outputs.iter().copied());
        let actions = self.decoder.decode(&outputs);
        let actions_applied = !actions.is_empty();
        if actions_applied {
            self.environment.advance_to(time);
            for action in actions {
                self.environment.apply_action(action);
                report.actions_applied = report.actions_applied.saturating_add(1);
            }
        }
        actions_applied
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        config::{NeuronConfig, RuntimeConfig},
        core::{Network, Neuron, NeuronId, Polarity},
        environment::{Action, BitWorld},
        learning::NoPlasticity,
        math::Position3D,
        nerves::{Fiber, FiberDirection, FiberId},
        roots::RootChannel,
        transduction::{BitDecoder, BitEncoder},
    };

    use super::*;

    fn simulation() -> Simulation<NoPlasticity> {
        let mut network = Network::new();
        network
            .add_neuron(
                Neuron::new(
                    NeuronId(1),
                    Position3D::ORIGIN,
                    Polarity::Excitatory,
                    None,
                    NeuronConfig {
                        resting_potential: 0.0,
                        reset_potential: 0.0,
                        threshold: 1.0,
                        membrane_tau_us: 1_000.0,
                        refractory_period_us: 1,
                        activity_trace_tau_us: 10_000.0,
                        intrinsic: Default::default(),
                    },
                    SimTime::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
        Simulation::new(
            network,
            NoPlasticity,
            RuntimeConfig {
                max_events_per_batch: 100,
            },
            1.0,
        )
        .unwrap()
    }

    #[derive(Debug)]
    struct RecordingBitWorld {
        inner: BitWorld,
        now: SimTime,
        observation_times: Vec<SimTime>,
        action_times: Vec<SimTime>,
    }

    impl RecordingBitWorld {
        fn new(initial: bool) -> Self {
            Self {
                inner: BitWorld::new(initial),
                now: SimTime::ZERO,
                observation_times: Vec::new(),
                action_times: Vec::new(),
            }
        }
    }

    impl Environment for RecordingBitWorld {
        fn next_observation_time(&self) -> Option<SimTime> {
            self.inner.next_observation_time()
        }

        fn observations(&mut self, until: SimTime) -> Vec<Observation> {
            let observations = self.inner.observations(until);
            self.observation_times
                .extend(observations.iter().map(Observation::time));
            observations
        }

        fn apply_action(&mut self, action: Action) {
            self.action_times.push(self.now);
            self.inner.apply_action(action);
        }

        fn advance_to(&mut self, time: SimTime) {
            self.now = self.now.max(time);
            self.inner.advance_to(time);
        }
    }

    fn closed_loop_with_environment<E: Environment>(
        environment: E,
    ) -> ClosedLoop<E, BitEncoder, BitDecoder, NoPlasticity> {
        let sensory_root = RootId(1);
        let motor_root_id = RootId(2);
        let mut mapping = Mapping::new();
        for (id, channel) in [(10, 0), (11, 1)] {
            mapping
                .add_fiber(
                    Fiber::new(FiberId(id), FiberDirection::Sensory, NeuronId(1), 1, 1.0).unwrap(),
                )
                .unwrap();
            mapping
                .map_sensory(sensory_root, channel, FiberId(id))
                .unwrap();
        }
        mapping
            .add_fiber(Fiber::new(FiberId(20), FiberDirection::Motor, NeuronId(1), 1, 1.0).unwrap())
            .unwrap();
        mapping.map_motor(motor_root_id, 1, FiberId(20)).unwrap();

        let motor_root = MotorRoot::new(
            motor_root_id,
            "bit actuator",
            vec![RootChannel {
                channel: 1,
                fiber: FiberId(20),
            }],
        )
        .unwrap();

        ClosedLoop::new(
            environment,
            BitEncoder::default(),
            mapping,
            sensory_root,
            motor_root,
            BitDecoder,
            simulation(),
        )
        .unwrap()
    }

    fn closed_loop() -> ClosedLoop<BitWorld, BitEncoder, BitDecoder, NoPlasticity> {
        closed_loop_with_environment(BitWorld::new(false))
    }

    #[test]
    fn bit_world_round_trip_uses_every_boundary_slice() {
        let mut closed_loop = closed_loop();

        let input = closed_loop.cycle_until(SimTime(1)).unwrap();
        assert_eq!(input.observations, 1);
        assert_eq!(input.encoded_spikes, 1);
        assert_eq!(input.sensory_impulses, 1);
        assert_eq!(input.runtime.spikes_emitted, 1);
        assert_eq!(input.actions_applied, 0);
        assert_eq!(closed_loop.pending_motor_count(), 1);
        assert!(!closed_loop.environment().value());

        let output = closed_loop.cycle_until(SimTime(2)).unwrap();
        assert_eq!(output.motor_outputs, 1);
        assert_eq!(output.motor_output_events.len(), 1);
        assert_eq!(output.motor_output_events[0].at, SimTime(2));
        assert_eq!(output.actions_applied, 1);
        assert_eq!(output.observations, 1);
        assert!(closed_loop.environment().value());

        // The feedback observation was already encoded at action time 2; its
        // fixed sensory delay makes it arrive at the core at time 3.
        let feedback = closed_loop.cycle_until(SimTime(3)).unwrap();
        assert_eq!(feedback.observations, 0);
        assert_eq!(feedback.sensory_impulses, 1);
    }

    #[test]
    fn large_horizon_interleaves_action_and_feedback_before_later_runtime_work() {
        let mut closed_loop = closed_loop_with_environment(RecordingBitWorld::new(false));

        let report = closed_loop.cycle_until(SimTime(100)).unwrap();

        // Initial false observation: observation@0 -> runtime spike@1 ->
        // SetBit(true)@2 -> feedback observation@2 -> runtime spike@3 ->
        // idempotent SetBit(true)@4. All of it closes in this single call.
        assert_eq!(report.observations, 2);
        assert_eq!(report.encoded_spikes, 2);
        assert_eq!(report.sensory_impulses, 2);
        assert_eq!(report.runtime.spikes_emitted, 2);
        assert_eq!(report.motor_outputs, 2);
        assert_eq!(report.actions_applied, 2);
        assert!(closed_loop.environment().inner.value());
        assert_eq!(
            closed_loop.environment().observation_times,
            vec![SimTime(0), SimTime(2)]
        );
        assert_eq!(
            closed_loop.environment().action_times,
            vec![SimTime(2), SimTime(4)]
        );
        assert_eq!(closed_loop.pending_motor_count(), 0);

        let spike_times: Vec<_> = closed_loop
            .simulation()
            .event_log()
            .iter()
            .filter_map(|event| match event {
                ObservationEvent::SpikeEmitted(spike) => Some(spike.time),
                _ => None,
            })
            .collect();
        assert_eq!(spike_times, vec![SimTime(1), SimTime(3)]);
    }

    #[test]
    fn unmapped_encoder_channel_is_not_silently_dropped() {
        // Remove the matching map by constructing an isolated adapter instead;
        // mappings are intentionally immutable once the loop is connected.
        let empty_mapping = Mapping::new();
        let motor_root = MotorRoot::new(RootId(2), "motor", Vec::new()).unwrap();
        let mut isolated = ClosedLoop::new(
            BitWorld::new(false),
            BitEncoder::default(),
            empty_mapping,
            RootId(1),
            motor_root,
            BitDecoder,
            simulation(),
        )
        .unwrap();

        assert_eq!(
            isolated.cycle_until(SimTime::ZERO),
            Err(ClosedLoopError::UnmappedSensoryChannel {
                root: RootId(1),
                channel: 0,
            })
        );
    }

    #[test]
    fn motor_mapping_must_match_the_declared_root_channel_and_fiber() {
        let sensory_root = RootId(1);
        let motor_root_id = RootId(2);
        let mut mapping = Mapping::new();
        mapping
            .add_fiber(
                Fiber::new(FiberId(10), FiberDirection::Sensory, NeuronId(1), 1, 1.0).unwrap(),
            )
            .unwrap();
        mapping.map_sensory(sensory_root, 0, FiberId(10)).unwrap();
        mapping
            .add_fiber(Fiber::new(FiberId(20), FiberDirection::Motor, NeuronId(1), 1, 1.0).unwrap())
            .unwrap();
        mapping.map_motor(motor_root_id, 1, FiberId(20)).unwrap();
        let motor_root = MotorRoot::new(
            motor_root_id,
            "mismatched actuator",
            vec![RootChannel {
                channel: 0,
                fiber: FiberId(20),
            }],
        )
        .unwrap();
        let mut loop_with_mismatch = ClosedLoop::new(
            BitWorld::new(false),
            BitEncoder::default(),
            mapping,
            sensory_root,
            motor_root,
            BitDecoder,
            simulation(),
        )
        .unwrap();

        assert_eq!(
            loop_with_mismatch.cycle_until(SimTime(1)),
            Err(ClosedLoopError::UnexpectedMotorChannel {
                root: motor_root_id,
                channel: 1,
                fiber: FiberId(20),
            })
        );
    }

    #[test]
    fn boundary_horizon_cannot_move_backwards() {
        let mut closed_loop = closed_loop();
        closed_loop.cycle_until(SimTime(5)).unwrap();

        assert_eq!(
            closed_loop.cycle_until(SimTime(4)),
            Err(ClosedLoopError::TimeWentBackwards {
                current: SimTime(5),
                requested: SimTime(4),
            })
        );
    }

    #[test]
    fn constructor_rejects_runtime_events_outside_its_causal_schedule() {
        let mut pre_scheduled = simulation();
        pre_scheduled
            .schedule_external_input(SimTime(10), NeuronId(1), 1.0)
            .unwrap();
        let result = ClosedLoop::new(
            BitWorld::new(false),
            BitEncoder::default(),
            Mapping::new(),
            RootId(1),
            MotorRoot::new(RootId(2), "motor", Vec::new()).unwrap(),
            BitDecoder,
            pre_scheduled,
        );

        assert!(matches!(
            result,
            Err(ClosedLoopError::PendingRuntimeEvents { count: 1 })
        ));
    }
}
