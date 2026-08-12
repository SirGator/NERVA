//! Single-threaded deterministic execution of exact-timestamp event batches.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use crate::{
    config::{ConfigError, DsvlmConfig, HomeostasisConfig, RuntimeConfig},
    core::{
        Event, EventKind, Network, NetworkError, NeuronError, NeuronId, SimTime, Spike, Synapse,
        SynapseId,
    },
    learning::{HomeostasisError, LocalHomeostasis, PlasticityRule},
    math::{DecayError, try_distance_attenuation},
};

use super::{
    event_batch::EventBatch,
    propagation::{PropagationError, plan_spike_propagation},
    scheduler::{EventScheduler, SchedulerError},
};

/// Immutable diagnostic output published by the runtime.
#[derive(Clone, Debug, PartialEq)]
pub enum ObservationEvent {
    /// A root- or nerve-provided impulse reached a core neuron.
    ExternalInput {
        /// Exact arrival timestamp.
        time: SimTime,
        /// Receiving neuron.
        target: NeuronId,
        /// Signed contribution included in the timestamp batch.
        amplitude: f32,
    },
    /// A core impulse reached the far end of a synapse.
    SynapticArrival {
        /// Exact arrival timestamp.
        time: SimTime,
        /// Connection carrying the impulse.
        synapse_id: SynapseId,
        /// Presynaptic neuron.
        source: NeuronId,
        /// Postsynaptic neuron.
        target: NeuronId,
        /// Signed, distance-attenuated contribution captured at emission.
        amplitude: f32,
    },
    /// A neuron crossed threshold after all simultaneous inputs were summed.
    SpikeEmitted(Spike),
    /// A local plasticity hook changed one weight magnitude.
    WeightChanged {
        /// Exact local learning timestamp.
        time: SimTime,
        /// Synapse modified by the rule.
        synapse_id: SynapseId,
        /// Weight before the local hook.
        old_weight: f32,
        /// Weight after the local hook.
        new_weight: f32,
    },
    /// Local homeostasis changed one neuron's threshold.
    ThresholdChanged {
        /// Exact maintenance timestamp.
        time: SimTime,
        /// Neuron modified by local maintenance.
        neuron_id: NeuronId,
        /// Threshold before maintenance.
        old_threshold: f32,
        /// Threshold after maintenance.
        new_threshold: f32,
        /// Neuron-local rate estimate used for the change.
        estimated_rate_hz: f32,
    },
}

/// Append-only observation log whose contents cannot mutate network state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EventLog {
    events: Vec<ObservationEvent>,
}

impl EventLog {
    /// Ordered immutable observations.
    pub fn events(&self) -> &[ObservationEvent] {
        &self.events
    }

    /// Number of observations in the log.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log contains no observations.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Consumes the log and returns its ordered observations.
    pub fn into_events(self) -> Vec<ObservationEvent> {
        self.events
    }

    fn push(&mut self, event: ObservationEvent) {
        self.events.push(event);
    }
}

/// Result of processing one exact-timestamp batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchReport {
    /// Timestamp processed atomically.
    pub time: SimTime,
    /// Number of queued inputs/maintenance events consumed.
    pub events_processed: usize,
    /// Number of neurons that emitted a spike.
    pub spikes_emitted: usize,
}

/// Aggregate result of `run` or `run_until`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunReport {
    /// Exact-timestamp batches processed.
    pub batches_processed: usize,
    /// Queued events consumed.
    pub events_processed: usize,
    /// Neuron spikes emitted.
    pub spikes_emitted: usize,
    /// Timestamp of the final processed batch, if any.
    pub last_time: Option<SimTime>,
}

impl RunReport {
    fn include(&mut self, batch: BatchReport) {
        self.batches_processed = self.batches_processed.saturating_add(1);
        self.events_processed = self.events_processed.saturating_add(batch.events_processed);
        self.spikes_emitted = self.spikes_emitted.saturating_add(batch.spikes_emitted);
        self.last_time = Some(batch.time);
    }
}

/// A fatal runtime or event-validation failure.
#[derive(Clone, Debug, PartialEq)]
pub enum SimulationError {
    /// A supplied aggregate configuration is invalid.
    InvalidConfig(ConfigError),
    /// The supplied network's objects and adjacency indices disagree.
    InvalidNetwork(NetworkError),
    /// The spatial decay length is invalid.
    InvalidDistanceDecayLength(DecayError),
    /// Event queue invariant or safety limit failure.
    Scheduler(SchedulerError),
    /// Spike propagation could not be planned.
    Propagation(PropagationError),
    /// A neuron rejected time evolution or integration.
    Neuron(NeuronError),
    /// Local homeostasis rejected an update.
    Homeostasis(HomeostasisError),
    /// An event references no such neuron.
    UnknownNeuron(NeuronId),
    /// A synaptic arrival references no such synapse.
    UnknownSynapse(SynapseId),
    /// A captured arrival target disagrees with the authoritative synapse.
    SynapticTargetMismatch {
        /// Referenced synapse.
        synapse_id: SynapseId,
        /// Target captured in the event.
        event_target: NeuronId,
        /// Target currently stored by the network.
        synapse_target: NeuronId,
    },
    /// A learning rule attempted to change immutable synapse identity/topology.
    InvalidLearningMutation {
        /// Synapse slot whose identity or endpoints were changed.
        synapse_id: SynapseId,
    },
    /// A scheduled input contains NaN or infinity.
    NonFiniteInput {
        /// Target receiving the invalid value.
        target: NeuronId,
        /// Rejected value.
        amplitude: f32,
    },
    /// Finite simultaneous inputs overflowed while being accumulated.
    NonFiniteSummedInput {
        /// Target whose input sum overflowed.
        target: NeuronId,
    },
}

impl fmt::Display for SimulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => {
                write!(formatter, "invalid simulation configuration: {error}")
            }
            Self::InvalidNetwork(error) => write!(formatter, "invalid simulation network: {error}"),
            Self::InvalidDistanceDecayLength(error) => {
                write!(formatter, "invalid propagation distance scale: {error}")
            }
            Self::Scheduler(error) => write!(formatter, "scheduler error: {error}"),
            Self::Propagation(error) => write!(formatter, "propagation error: {error}"),
            Self::Neuron(error) => write!(formatter, "neuron runtime error: {error}"),
            Self::Homeostasis(error) => write!(formatter, "homeostasis error: {error:?}"),
            Self::UnknownNeuron(id) => write!(formatter, "event references unknown neuron {id}"),
            Self::UnknownSynapse(id) => write!(formatter, "event references unknown synapse {id}"),
            Self::SynapticTargetMismatch {
                synapse_id,
                event_target,
                synapse_target,
            } => write!(
                formatter,
                "arrival for synapse {synapse_id} targets neuron {event_target}, but the synapse targets {synapse_target}"
            ),
            Self::InvalidLearningMutation { synapse_id } => write!(
                formatter,
                "learning rule changed immutable identity or endpoints of synapse {synapse_id}"
            ),
            Self::NonFiniteInput { target, amplitude } => {
                write!(
                    formatter,
                    "input for neuron {target} must be finite, got {amplitude}"
                )
            }
            Self::NonFiniteSummedInput { target } => {
                write!(
                    formatter,
                    "simultaneous input sum for neuron {target} became non-finite"
                )
            }
        }
    }
}

impl Error for SimulationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidConfig(error) => Some(error),
            Self::InvalidNetwork(error) => Some(error),
            Self::InvalidDistanceDecayLength(error) => Some(error),
            Self::Scheduler(error) => Some(error),
            Self::Propagation(error) => Some(error),
            Self::Neuron(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SchedulerError> for SimulationError {
    fn from(error: SchedulerError) -> Self {
        Self::Scheduler(error)
    }
}

impl From<PropagationError> for SimulationError {
    fn from(error: PropagationError) -> Self {
        Self::Propagation(error)
    }
}

impl From<NeuronError> for SimulationError {
    fn from(error: NeuronError) -> Self {
        Self::Neuron(error)
    }
}

impl From<HomeostasisError> for SimulationError {
    fn from(error: HomeostasisError) -> Self {
        Self::Homeostasis(error)
    }
}

/// Single-threaded event runtime parameterized by one local plasticity rule.
pub struct Simulation<R: PlasticityRule> {
    network: Network,
    plasticity_rule: R,
    homeostasis: LocalHomeostasis,
    scheduler: EventScheduler<EventKind>,
    event_log: EventLog,
    max_events_per_batch: usize,
    distance_decay_length: f32,
    learning_enabled: bool,
}

impl<R: PlasticityRule> Simulation<R> {
    /// Creates a runtime with local homeostasis disabled by default.
    pub fn new(
        network: Network,
        plasticity_rule: R,
        runtime_config: RuntimeConfig,
        distance_decay_length: f32,
    ) -> Result<Self, SimulationError> {
        runtime_config
            .validate()
            .map_err(SimulationError::InvalidConfig)?;
        network
            .validate_indices()
            .map_err(SimulationError::InvalidNetwork)?;
        try_distance_attenuation(0.0, distance_decay_length)
            .map_err(SimulationError::InvalidDistanceDecayLength)?;

        Ok(Self {
            network,
            plasticity_rule,
            homeostasis: LocalHomeostasis::new(&HomeostasisConfig::default()),
            scheduler: EventScheduler::new(),
            event_log: EventLog::default(),
            max_events_per_batch: runtime_config.max_events_per_batch,
            distance_decay_length,
            learning_enabled: true,
        })
    }

    /// Constructs a runtime from a complete, validated experiment config.
    pub fn from_config(
        network: Network,
        plasticity_rule: R,
        config: &DsvlmConfig,
    ) -> Result<Self, SimulationError> {
        config.validate().map_err(SimulationError::InvalidConfig)?;
        let mut simulation = Self::new(
            network,
            plasticity_rule,
            config.runtime,
            config.network.distance_decay_length,
        )?;
        simulation
            .network
            .validate_against_config(&config.network, &config.neuron)
            .map_err(SimulationError::InvalidNetwork)?;
        simulation.learning_enabled = config.learning.enabled;
        simulation.homeostasis = LocalHomeostasis::from_learning_config(&config.learning);
        Ok(simulation)
    }

    /// Replaces the default-disabled local homeostasis state.
    pub fn with_homeostasis(mut self, homeostasis: LocalHomeostasis) -> Self {
        self.homeostasis = homeostasis;
        self
    }

    /// Read-only access to authoritative network state.
    pub fn network(&self) -> &Network {
        &self.network
    }

    /// Explicit mutable access for experiment setup and controlled interventions.
    pub fn network_mut(&mut self) -> &mut Network {
        &mut self.network
    }

    /// Current time of the most recently removed batch.
    pub fn current_time(&self) -> SimTime {
        self.scheduler.current_time()
    }

    /// Number of queued events.
    pub fn pending_event_count(&self) -> usize {
        self.scheduler.len()
    }

    /// Immutable observations accumulated so far.
    pub fn event_log(&self) -> &[ObservationEvent] {
        self.event_log.events()
    }

    /// Removes and returns the current observation log without touching neural state.
    pub fn drain_event_log(&mut self) -> EventLog {
        std::mem::take(&mut self.event_log)
    }

    /// Enables or freezes calls into the synaptic learning rule.
    pub fn set_learning_enabled(&mut self, enabled: bool) {
        self.learning_enabled = enabled;
    }

    /// Permanently freezes weights until explicitly re-enabled.
    pub fn freeze_learning(&mut self) {
        self.set_learning_enabled(false);
    }

    /// Whether synaptic learning hooks currently run.
    pub fn learning_enabled(&self) -> bool {
        self.learning_enabled
    }

    /// Enables or freezes local activity recording and threshold maintenance.
    pub fn set_homeostasis_enabled(&mut self, enabled: bool) {
        self.homeostasis.set_enabled(enabled);
    }

    /// Whether local homeostasis is active.
    pub fn homeostasis_enabled(&self) -> bool {
        self.homeostasis.is_enabled()
    }

    /// Schedules one trusted, validated core event.
    ///
    /// Kept crate-private so callers cannot forge synaptic arrivals. Public
    /// external input and homeostasis helpers are the supported boundaries.
    pub(crate) fn schedule_event(&mut self, event: Event) -> Result<u64, SimulationError> {
        self.validate_event(event.kind)?;
        self.scheduler
            .schedule(event.time, event.kind)
            .map_err(SimulationError::Scheduler)
    }

    /// Schedules one root/nerve input without directly mutating a neuron.
    pub fn schedule_external_input(
        &mut self,
        at: SimTime,
        target: NeuronId,
        amplitude: f32,
    ) -> Result<u64, SimulationError> {
        self.schedule_event(Event::new(
            at,
            EventKind::ExternalInput { target, amplitude },
        ))
    }

    /// Schedules one local threshold-maintenance event.
    pub fn schedule_homeostasis(
        &mut self,
        at: SimTime,
        neuron_id: NeuronId,
    ) -> Result<u64, SimulationError> {
        self.schedule_event(Event::new(at, EventKind::Homeostasis { neuron_id }))
    }

    /// Processes the next complete timestamp batch.
    pub fn step(&mut self) -> Result<Option<BatchReport>, SimulationError> {
        let Some(batch) = self.scheduler.pop_next_batch(self.max_events_per_batch)? else {
            return Ok(None);
        };
        self.process_batch(batch).map(Some)
    }

    /// Runs until no event remains.
    pub fn run(&mut self) -> Result<RunReport, SimulationError> {
        let mut report = RunReport::default();
        while let Some(batch) = self.step()? {
            report.include(batch);
        }
        Ok(report)
    }

    /// Runs every event whose timestamp is at most `inclusive`.
    pub fn run_until(&mut self, inclusive: SimTime) -> Result<RunReport, SimulationError> {
        let mut report = RunReport::default();
        while self
            .scheduler
            .next_time()
            .is_some_and(|next_time| next_time <= inclusive)
        {
            let batch = self
                .step()?
                .expect("a scheduler time implies a pending event batch");
            report.include(batch);
        }
        Ok(report)
    }

    /// Consumes the runtime into authoritative state, rule state, homeostasis, and log.
    pub fn into_parts(self) -> (Network, R, LocalHomeostasis, EventLog) {
        (
            self.network,
            self.plasticity_rule,
            self.homeostasis,
            self.event_log,
        )
    }

    fn validate_event(&self, kind: EventKind) -> Result<(), SimulationError> {
        match kind {
            EventKind::ExternalInput { target, amplitude } => {
                self.require_neuron(target)?;
                validate_input(target, amplitude)
            }
            EventKind::SynapticArrival {
                synapse_id,
                target,
                amplitude,
            } => {
                let synapse = self
                    .network
                    .synapse(synapse_id)
                    .ok_or(SimulationError::UnknownSynapse(synapse_id))?;
                if synapse.post() != target {
                    return Err(SimulationError::SynapticTargetMismatch {
                        synapse_id,
                        event_target: target,
                        synapse_target: synapse.post(),
                    });
                }
                self.require_neuron(synapse.pre())?;
                self.require_neuron(target)?;
                validate_input(target, amplitude)
            }
            EventKind::Homeostasis { neuron_id } => self.require_neuron(neuron_id),
        }
    }

    fn require_neuron(&self, id: NeuronId) -> Result<(), SimulationError> {
        self.network
            .neuron(id)
            .map(|_| ())
            .ok_or(SimulationError::UnknownNeuron(id))
    }

    fn process_batch(
        &mut self,
        batch: EventBatch<EventKind>,
    ) -> Result<BatchReport, SimulationError> {
        let time = batch.time();
        let events = batch.into_events();

        // Validate the whole batch before the first state mutation. This also
        // catches controlled network edits made after an event was scheduled.
        for scheduled in &events {
            self.validate_event(scheduled.payload)?;
        }

        let mut affected_neurons = BTreeSet::new();
        let mut maintenance_neurons = BTreeSet::new();
        for scheduled in &events {
            match scheduled.payload {
                EventKind::ExternalInput { target, .. }
                | EventKind::SynapticArrival { target, .. } => {
                    affected_neurons.insert(target);
                }
                EventKind::Homeostasis { neuron_id } => {
                    affected_neurons.insert(neuron_id);
                    maintenance_neurons.insert(neuron_id);
                }
            }
        }

        // Every affected neuron first reaches the shared timestamp analytically.
        for neuron_id in affected_neurons {
            self.network
                .neuron_mut(neuron_id)
                .ok_or(SimulationError::UnknownNeuron(neuron_id))?
                .advance_to(time)?;
        }

        let mut simultaneous_inputs = BTreeMap::<NeuronId, Vec<f32>>::new();
        for scheduled in &events {
            match scheduled.payload {
                EventKind::ExternalInput { target, amplitude } => {
                    self.event_log.push(ObservationEvent::ExternalInput {
                        time,
                        target,
                        amplitude,
                    });
                    collect_input(&mut simultaneous_inputs, target, amplitude);
                }
                EventKind::SynapticArrival {
                    synapse_id,
                    target,
                    amplitude,
                } => {
                    let (source, accepted_by_rule) = {
                        let synapse = self
                            .network
                            .synapse(synapse_id)
                            .ok_or(SimulationError::UnknownSynapse(synapse_id))?;
                        let source = synapse.pre();
                        let polarity = self
                            .network
                            .neuron(source)
                            .ok_or(SimulationError::UnknownNeuron(source))?
                            .polarity();
                        (source, self.plasticity_rule.accepts(synapse, polarity))
                    };

                    self.event_log.push(ObservationEvent::SynapticArrival {
                        time,
                        synapse_id,
                        source,
                        target,
                        amplitude,
                    });
                    self.network
                        .synapse_mut(synapse_id)
                        .ok_or(SimulationError::UnknownSynapse(synapse_id))?
                        .record_transmission();

                    if self.learning_enabled && accepted_by_rule {
                        let post = self
                            .network
                            .neuron(target)
                            .ok_or(SimulationError::UnknownNeuron(target))?
                            .clone();
                        let (old_weight, new_weight) = {
                            let synapse = self
                                .network
                                .synapse_mut(synapse_id)
                                .ok_or(SimulationError::UnknownSynapse(synapse_id))?;
                            let old_weight = synapse.weight();
                            self.plasticity_rule.on_pre_arrival(synapse, &post, time);
                            (old_weight, synapse.weight())
                        };
                        self.log_weight_change(time, synapse_id, old_weight, new_weight);
                    }

                    collect_input(&mut simultaneous_inputs, target, amplitude);
                }
                EventKind::Homeostasis { .. } => {}
            }
        }

        // Target order and a canonical magnitude/value ordering make membrane
        // integration independent of the insertion order of simultaneous
        // arrivals, including floating-point rounding behavior.
        let mut spikes = Vec::new();
        for (target, inputs) in simultaneous_inputs {
            let summed_input = canonical_input_sum(target, inputs)?;
            if let Some(spike) = self
                .network
                .neuron_mut(target)
                .ok_or(SimulationError::UnknownNeuron(target))?
                .integrate_input(time, summed_input)?
            {
                spikes.push(spike);
            }
        }

        for &spike in &spikes {
            self.event_log.push(ObservationEvent::SpikeEmitted(spike));
            let neuron = self
                .network
                .neuron(spike.neuron_id)
                .ok_or(SimulationError::UnknownNeuron(spike.neuron_id))?;
            self.homeostasis.record_spike(neuron, time)?;
        }

        if self.learning_enabled {
            for &spike in &spikes {
                self.apply_post_spike_learning(spike)?;
            }
        }

        // Duplicate maintenance events for one cell and timestamp collapse into
        // one local update, just as duplicate arrivals collapse into one firing check.
        for neuron_id in maintenance_neurons {
            let change = {
                let neuron = self
                    .network
                    .neuron_mut(neuron_id)
                    .ok_or(SimulationError::UnknownNeuron(neuron_id))?;
                self.homeostasis.maintain(neuron, time)?
            };
            if let Some(change) = change {
                self.event_log.push(ObservationEvent::ThresholdChanged {
                    time,
                    neuron_id: change.neuron_id,
                    old_threshold: change.old_threshold,
                    new_threshold: change.new_threshold,
                    estimated_rate_hz: change.estimated_rate_hz,
                });
            }
        }

        // Post hooks and maintenance finish before any outgoing weights are read.
        // Positive validated delays ensure these events cannot re-enter this batch.
        for &spike in &spikes {
            for transmission in
                plan_spike_propagation(&self.network, spike, self.distance_decay_length)?
            {
                self.schedule_event(transmission.event)?;
            }
        }

        Ok(BatchReport {
            time,
            events_processed: events.len(),
            spikes_emitted: spikes.len(),
        })
    }

    fn apply_post_spike_learning(&mut self, spike: Spike) -> Result<(), SimulationError> {
        let neuron = self
            .network
            .neuron(spike.neuron_id)
            .ok_or(SimulationError::UnknownNeuron(spike.neuron_id))?
            .clone();
        let incoming_ids = self.network.incoming_synapse_ids(spike.neuron_id).to_vec();
        let mut accepted_ids = Vec::new();
        for synapse_id in incoming_ids {
            let synapse = self
                .network
                .synapse(synapse_id)
                .ok_or(SimulationError::UnknownSynapse(synapse_id))?;
            let polarity = self
                .network
                .neuron(synapse.pre())
                .ok_or(SimulationError::UnknownNeuron(synapse.pre()))?
                .polarity();
            if self.plasticity_rule.accepts(synapse, polarity) {
                accepted_ids.push(synapse_id);
            }
        }

        let mut incoming = Vec::with_capacity(accepted_ids.len());
        for &synapse_id in &accepted_ids {
            incoming.push(
                self.network
                    .synapse(synapse_id)
                    .ok_or(SimulationError::UnknownSynapse(synapse_id))?
                    .clone(),
            );
        }
        let old_weights: Vec<_> = incoming.iter().map(Synapse::weight).collect();
        self.plasticity_rule
            .on_post_spike(&neuron, &mut incoming, spike.time);

        for (&synapse_id, updated) in accepted_ids.iter().zip(&incoming) {
            if updated.id() != synapse_id
                || updated.post() != spike.neuron_id
                || self.network.synapse(synapse_id).map(Synapse::pre) != Some(updated.pre())
            {
                return Err(SimulationError::InvalidLearningMutation { synapse_id });
            }
            updated.validate().map_err(|error| {
                SimulationError::InvalidNetwork(NetworkError::InvalidSynapse(error))
            })?;
        }

        for ((synapse_id, old_weight), updated) in
            accepted_ids.into_iter().zip(old_weights).zip(incoming)
        {
            let new_weight = updated.weight();
            *self
                .network
                .synapse_mut(synapse_id)
                .ok_or(SimulationError::UnknownSynapse(synapse_id))? = updated;
            self.log_weight_change(spike.time, synapse_id, old_weight, new_weight);
        }
        Ok(())
    }

    fn log_weight_change(
        &mut self,
        time: SimTime,
        synapse_id: SynapseId,
        old_weight: f32,
        new_weight: f32,
    ) {
        if old_weight != new_weight {
            self.event_log.push(ObservationEvent::WeightChanged {
                time,
                synapse_id,
                old_weight,
                new_weight,
            });
        }
    }
}

fn validate_input(target: NeuronId, amplitude: f32) -> Result<(), SimulationError> {
    if amplitude.is_finite() {
        Ok(())
    } else {
        Err(SimulationError::NonFiniteInput { target, amplitude })
    }
}

fn collect_input(inputs: &mut BTreeMap<NeuronId, Vec<f32>>, target: NeuronId, amplitude: f32) {
    inputs.entry(target).or_default().push(amplitude);
}

fn canonical_input_sum(target: NeuronId, mut amplitudes: Vec<f32>) -> Result<f32, SimulationError> {
    amplitudes.sort_by(|left, right| {
        right
            .abs()
            .total_cmp(&left.abs())
            .then_with(|| left.total_cmp(right))
    });
    let sum = amplitudes
        .into_iter()
        .fold(0.0_f64, |sum, amplitude| sum + f64::from(amplitude));
    let sum = sum as f32;
    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(SimulationError::NonFiniteSummedInput { target })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        config::{HomeostasisConfig, NeuronConfig},
        core::{Neuron, Polarity, Synapse},
        learning::{NoPlasticity, PlasticityRule},
        math::Position3D,
    };

    use super::*;

    fn neuron_params() -> NeuronConfig {
        NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            membrane_tau_us: 1_000.0,
            refractory_period_us: 1,
            activity_trace_tau_us: 10_000.0,
        }
    }

    fn neuron(id: u64, polarity: Polarity) -> Neuron {
        Neuron::new(
            NeuronId(id),
            Position3D::ORIGIN,
            polarity,
            None,
            neuron_params(),
            SimTime::ZERO,
        )
        .unwrap()
    }

    fn one_neuron_network() -> Network {
        let mut network = Network::new();
        network.add_neuron(neuron(1, Polarity::Excitatory)).unwrap();
        network
    }

    fn connected_network(source_polarity: Polarity) -> Network {
        let mut network = Network::new();
        network.add_neuron(neuron(1, source_polarity)).unwrap();
        network.add_neuron(neuron(2, Polarity::Excitatory)).unwrap();
        network
            .add_synapse(
                Synapse::new(SynapseId(1), NeuronId(1), NeuronId(2), 1.0, 5, true).unwrap(),
            )
            .unwrap();
        network
    }

    fn runtime_config(max_events_per_batch: usize) -> RuntimeConfig {
        RuntimeConfig {
            max_events_per_batch,
        }
    }

    fn simulation<R: PlasticityRule>(network: Network, rule: R) -> Simulation<R> {
        Simulation::new(network, rule, runtime_config(100), 1.0).unwrap()
    }

    fn network_with_inconsistent_indices() -> Network {
        let mut network = connected_network(Polarity::Excitatory);
        network
            .synapse_mut(SynapseId(1))
            .expect("test synapse exists")
            .pre = NeuronId(2);
        network
    }

    #[derive(Debug, Default)]
    struct AdditiveRule {
        pre_calls: usize,
        post_synapses: usize,
    }

    impl PlasticityRule for AdditiveRule {
        fn accepts(&self, synapse: &Synapse, polarity: Polarity) -> bool {
            synapse.is_enabled() && synapse.is_plastic() && polarity == Polarity::Excitatory
        }

        fn on_pre_arrival(&mut self, synapse: &mut Synapse, _post: &Neuron, _time: SimTime) {
            self.pre_calls += 1;
            synapse.set_weight(synapse.weight() + 0.1).unwrap();
        }

        fn on_post_spike(&mut self, _neuron: &Neuron, incoming: &mut [Synapse], _time: SimTime) {
            self.post_synapses += incoming.len();
            for synapse in incoming {
                synapse.set_weight(synapse.weight() + 0.2).unwrap();
            }
        }
    }

    #[test]
    fn constructors_reject_inconsistent_network_indices() {
        let network = network_with_inconsistent_indices();

        assert!(matches!(
            Simulation::new(network.clone(), NoPlasticity, runtime_config(100), 1.0),
            Err(SimulationError::InvalidNetwork(
                NetworkError::InconsistentIndices
            ))
        ));
        assert!(matches!(
            Simulation::from_config(network, NoPlasticity, &DsvlmConfig::default()),
            Err(SimulationError::InvalidNetwork(
                NetworkError::InconsistentIndices
            ))
        ));
    }

    #[test]
    fn aggregate_constructor_rejects_network_config_mismatch() {
        let network = connected_network(Polarity::Excitatory);
        let error = Simulation::from_config(network, NoPlasticity, &DsvlmConfig::default())
            .err()
            .expect("default config describes another population");

        assert!(matches!(
            error,
            SimulationError::InvalidNetwork(NetworkError::PopulationMismatch { .. })
        ));
    }

    #[test]
    fn simultaneous_inputs_are_summed_before_one_firing_check() {
        let mut simulation = simulation(one_neuron_network(), NoPlasticity);
        simulation
            .schedule_external_input(SimTime(10), NeuronId(1), 0.6)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(10), NeuronId(1), 0.6)
            .unwrap();

        let report = simulation.step().unwrap().unwrap();

        assert_eq!(
            report,
            BatchReport {
                time: SimTime(10),
                events_processed: 2,
                spikes_emitted: 1,
            }
        );
        assert_eq!(
            simulation
                .network()
                .neuron(NeuronId(1))
                .unwrap()
                .spike_count(),
            1
        );
    }

    #[test]
    fn simultaneous_sum_is_independent_of_event_insertion_order() {
        fn run(amplitudes: [f32; 3]) -> Vec<ObservationEvent> {
            let mut simulation = simulation(one_neuron_network(), NoPlasticity);
            for amplitude in amplitudes {
                simulation
                    .schedule_external_input(SimTime(1), NeuronId(1), amplitude)
                    .unwrap();
            }
            simulation.run().unwrap();
            simulation.event_log().to_vec()
        }

        let first = run([f32::MAX, -f32::MAX, 1.0]);
        let second = run([1.0, f32::MAX, -f32::MAX]);
        let first_spikes = first
            .iter()
            .filter(|event| matches!(event, ObservationEvent::SpikeEmitted(_)))
            .count();
        let second_spikes = second
            .iter()
            .filter(|event| matches!(event, ObservationEvent::SpikeEmitted(_)))
            .count();

        assert_eq!(first_spikes, 1);
        assert_eq!(first_spikes, second_spikes);
    }

    #[test]
    fn spike_is_delayed_propagated_and_logged() {
        let mut simulation = simulation(connected_network(Polarity::Excitatory), NoPlasticity);
        simulation
            .schedule_external_input(SimTime(10), NeuronId(1), 1.0)
            .unwrap();

        let report = simulation.run().unwrap();

        assert_eq!(report.batches_processed, 2);
        assert_eq!(report.events_processed, 2);
        assert_eq!(report.spikes_emitted, 2);
        assert_eq!(report.last_time, Some(SimTime(15)));
        assert_eq!(
            simulation
                .network()
                .synapse(SynapseId(1))
                .unwrap()
                .transmission_count(),
            1
        );
        assert!(simulation.event_log().iter().any(|event| matches!(
            event,
            ObservationEvent::SynapticArrival {
                time: SimTime(15),
                synapse_id: SynapseId(1),
                ..
            }
        )));
    }

    #[test]
    fn excitatory_learning_hooks_run_and_freeze_prevents_all_weight_mutation() {
        let network = connected_network(Polarity::Excitatory);
        let mut learning = simulation(network.clone(), AdditiveRule::default());
        learning
            .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
            .unwrap();
        learning.run().unwrap();
        let learned_weight = learning.network().synapse(SynapseId(1)).unwrap().weight();
        assert!((learned_weight - 1.3).abs() < 1.0e-6);
        assert_eq!(
            learning
                .event_log()
                .iter()
                .filter(|event| matches!(event, ObservationEvent::WeightChanged { .. }))
                .count(),
            2
        );

        let mut frozen = simulation(network, AdditiveRule::default());
        frozen.freeze_learning();
        frozen
            .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
            .unwrap();
        frozen.run().unwrap();
        let (network, rule, _, _) = frozen.into_parts();
        assert_eq!(network.synapse(SynapseId(1)).unwrap().weight(), 1.0);
        assert_eq!(rule.pre_calls, 0);
        assert_eq!(rule.post_synapses, 0);
    }

    #[test]
    fn inhibitory_presynaptic_cell_is_never_passed_to_stdp() {
        let mut simulation = simulation(
            connected_network(Polarity::Inhibitory),
            AdditiveRule::default(),
        );
        simulation
            .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
            .unwrap();
        simulation.run().unwrap();

        let (network, rule, _, _) = simulation.into_parts();
        assert_eq!(network.synapse(SynapseId(1)).unwrap().weight(), 1.0);
        assert_eq!(rule.pre_calls, 0);
        assert_eq!(rule.post_synapses, 0);
    }

    #[test]
    fn same_inputs_produce_identical_observation_log() {
        fn run_once() -> EventLog {
            let mut simulation = simulation(connected_network(Polarity::Excitatory), NoPlasticity);
            simulation
                .schedule_external_input(SimTime(4), NeuronId(1), 0.4)
                .unwrap();
            simulation
                .schedule_external_input(SimTime(4), NeuronId(1), 0.6)
                .unwrap();
            simulation.run().unwrap();
            simulation.drain_event_log()
        }

        assert_eq!(run_once(), run_once());
    }

    #[test]
    fn max_events_per_batch_fails_before_batch_mutation() {
        let mut simulation =
            Simulation::new(one_neuron_network(), NoPlasticity, runtime_config(1), 1.0).unwrap();
        simulation
            .schedule_external_input(SimTime(1), NeuronId(1), 0.1)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(1), NeuronId(1), 0.2)
            .unwrap();

        assert!(matches!(
            simulation.step(),
            Err(SimulationError::Scheduler(
                SchedulerError::BatchLimitExceeded { .. }
            ))
        ));
        assert_eq!(simulation.pending_event_count(), 2);
        assert_eq!(
            simulation
                .network()
                .neuron(NeuronId(1))
                .unwrap()
                .last_update(),
            SimTime::ZERO
        );
    }

    #[test]
    fn maintenance_uses_neuron_local_activity_after_same_time_spike() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            target_rate_hz: 0.0,
            adjustment_rate: 0.001,
            min_threshold: 0.1,
            max_threshold: 2.0,
        });
        let mut simulation =
            simulation(one_neuron_network(), NoPlasticity).with_homeostasis(homeostasis);
        simulation
            .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
            .unwrap();
        simulation
            .schedule_homeostasis(SimTime::ZERO, NeuronId(1))
            .unwrap();

        simulation.run().unwrap();

        assert!(
            (simulation
                .network()
                .neuron(NeuronId(1))
                .unwrap()
                .threshold()
                - 1.1)
                .abs()
                < 1.0e-6
        );
        assert!(simulation.event_log().iter().any(|event| matches!(
            event,
            ObservationEvent::ThresholdChanged {
                neuron_id: NeuronId(1),
                ..
            }
        )));
    }

    #[test]
    fn invalid_external_input_is_rejected_before_scheduling() {
        let mut simulation = simulation(one_neuron_network(), NoPlasticity);

        assert!(matches!(
            simulation.schedule_external_input(SimTime::ZERO, NeuronId(1), f32::NAN),
            Err(SimulationError::NonFiniteInput { .. })
        ));
        assert_eq!(simulation.pending_event_count(), 0);
    }
}
