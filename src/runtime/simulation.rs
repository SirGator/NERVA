//! Deterministic orchestration of exact-timestamp batches and numerical compute.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    time::Instant,
};

use crate::{
    compute::{ComputeBackend, ComputeError, CpuCompute, DenseNeuronBatch, FlatNeuronInputs},
    config::{ConfigError, HomeostasisConfig, NervaConfig, RuntimeConfig},
    core::{
        Event, EventKind, Network, NetworkError, Neuron, NeuronError, NeuronId, SimTime, Spike,
        Synapse, SynapseError, SynapseId,
    },
    learning::{
        HomeostasisError, LocalHomeostasis, NoUtility, PlasticityRule, UtilityContext,
        UtilityDynamicsConfig, UtilityRule, UtilityTrigger,
    },
    math::{DecayError, try_distance_attenuation},
    primitives::Weight,
};

use super::{
    event_batch::{EventBatch, ScheduledEvent},
    homeostasis_booking::{HomeostasisBooking, HomeostasisBookings},
    intrinsic_booking::{IntrinsicBooking, IntrinsicBookings},
    profile::{ProfilePhase, RuntimeProfile},
    propagation::{PropagationError, plan_spike_propagation_into},
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
        old_weight: Weight,
        /// Weight after the local hook.
        new_weight: Weight,
    },
    /// Local homeostasis changed one neuron's cellular or structural state.
    HomeostasisChanged {
        /// Exact maintenance timestamp.
        time: SimTime,
        /// Neuron modified by local maintenance.
        neuron_id: NeuronId,
        /// Neuron-local firing estimate used for the change.
        firing_avg: f32,
        /// Neuron-local input estimate used for the change.
        input_avg: f32,
        /// Intrinsic current before maintenance.
        old_intrinsic_current: f32,
        /// Intrinsic current after maintenance.
        new_intrinsic_current: f32,
        /// Structural drive before maintenance.
        old_structural_drive: f32,
        /// Structural drive after maintenance.
        new_structural_drive: f32,
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
    /// Dense numerical data or synchronization violated the compute contract.
    Compute(ComputeError),
    /// Local homeostasis rejected an update.
    Homeostasis(HomeostasisError),
    /// `run` has no natural end while local clocks or autonomous firing are active.
    RunRequiresDeadline,
    /// A local maintenance deadline could not be represented in simulation time.
    HomeostasisTimeOverflow {
        /// Last successfully processed maintenance timestamp.
        time: SimTime,
        /// Interval or initial offset that overflowed the timestamp.
        duration_us: u64,
    },
    /// A manually supplied local-maintenance controller had an invalid period.
    InvalidHomeostasisInterval,
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
    /// Finite simultaneous input magnitudes overflowed while being accumulated.
    NonFiniteInputMagnitude {
        /// Target whose absolute-input sum overflowed.
        target: NeuronId,
    },
    /// A controlled live mutation of a synapse was rejected.
    InvalidNetworkMutation(SynapseError),
    /// No unused `SynapseId` remains at or above a development cursor.
    NoSynapseIdsAvailable,
    /// A utility parameter was invalid.
    InvalidUtilityParameter {
        /// Parameter name.
        parameter: &'static str,
        /// Rejected value.
        value: f32,
    },
    /// The runtime already failed fatally and can no longer be advanced.
    SimulationPoisoned {
        /// Display text of the original fatal error.
        reason: String,
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
            Self::Compute(error) => write!(formatter, "compute error: {error}"),
            Self::Homeostasis(error) => write!(formatter, "homeostasis error: {error:?}"),
            Self::RunRequiresDeadline => formatter.write_str(
                "local clocks or autonomous firing are active; use run_until with an explicit horizon",
            ),
            Self::HomeostasisTimeOverflow { time, duration_us } => write!(
                formatter,
                "adding local maintenance duration {duration_us} us to time {time} overflows simulation time"
            ),
            Self::InvalidHomeostasisInterval => {
                formatter.write_str("local maintenance interval must be greater than zero")
            }
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
            Self::NonFiniteInputMagnitude { target } => write!(
                formatter,
                "simultaneous input magnitude for neuron {target} became non-finite"
            ),
            Self::InvalidNetworkMutation(error) => {
                write!(formatter, "controlled network mutation failed: {error}")
            }
            Self::NoSynapseIdsAvailable => {
                formatter.write_str("no synapse identities remain for development")
            }
            Self::InvalidUtilityParameter { parameter, value } => {
                write!(formatter, "invalid utility parameter {parameter}={value}")
            }
            Self::SimulationPoisoned { reason } => write!(
                formatter,
                "this simulation failed fatally earlier and can no longer be advanced: {reason}"
            ),
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
            Self::Compute(error) => Some(error),
            Self::InvalidNetworkMutation(error) => Some(error),
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

/// Runtime-only roles collected before any numerical mutation.
struct BatchNeurons {
    affected: Vec<NeuronId>,
    maintenance: BTreeSet<NeuronId>,
    due_intrinsic: BTreeSet<NeuronId>,
}

impl From<ComputeError> for SimulationError {
    fn from(error: ComputeError) -> Self {
        match error {
            error @ (ComputeError::InvalidBatch { .. } | ComputeError::Backend { .. }) => {
                Self::Compute(error)
            }
            ComputeError::UnknownNeuron(id) => Self::UnknownNeuron(id),
            ComputeError::Neuron(error) => Self::Neuron(error),
            ComputeError::NonFiniteInput { target, amplitude } => {
                Self::NonFiniteInput { target, amplitude }
            }
            ComputeError::NonFiniteSummedInput { target } => Self::NonFiniteSummedInput { target },
            ComputeError::NonFiniteInputMagnitude { target } => {
                Self::NonFiniteInputMagnitude { target }
            }
        }
    }
}

impl From<HomeostasisError> for SimulationError {
    fn from(error: HomeostasisError) -> Self {
        Self::Homeostasis(error)
    }
}

/// Deterministic event orchestration parameterized by one local plasticity rule.
///
/// Numerical cell advancement and input integration use a synchronous
/// [`ComputeBackend`], defaulting to [`CpuCompute`]. Scheduling, graph edits,
/// learning hooks, observations and local-clock bookings remain runtime-owned.
///
/// The utility evidence rule is held as a trait object (`Box<dyn UtilityRule>`)
/// so that the second concern does not propagate as a second generic parameter
/// through every call site. The default is [`NoUtility`], which leaves the
/// utility memory of every synapse untouched.
pub struct Simulation<R: PlasticityRule> {
    network: Network,
    compute: Box<dyn ComputeBackend>,
    profile: Option<RuntimeProfile>,
    plasticity_rule: R,
    utility_rule: Box<dyn UtilityRule>,
    homeostasis: LocalHomeostasis,
    /// None preserves the default: every neuron receives local maintenance.
    homeostasis_population: Option<BTreeSet<NeuronId>>,
    scheduler: EventScheduler<EventKind>,
    /// Scheduler bookkeeping intentionally lives in the runtime rather than
    /// in core neuron state.
    homeostasis_bookings: HomeostasisBookings,
    intrinsic_bookings: IntrinsicBookings,
    event_log: EventLog,
    max_events_per_batch: usize,
    distance_decay_length: f32,
    learning_enabled: bool,
    /// Canonical, validated parameters for the utility memory and eligibility
    /// dynamics. This is the single source of truth; the pruning controller
    /// and utility rules read from it instead of carrying their own copies.
    utility_dynamics: UtilityDynamicsConfig,
    /// Set when a fatal batch error leaves the runtime in an inconsistent
    /// partial state. Every further mutation or execution is rejected.
    poisoned: Option<Box<SimulationError>>,
}

impl<R: PlasticityRule> Simulation<R> {
    /// Creates a runtime with local homeostasis disabled by default and the
    /// no-op [`NoUtility`] utility rule.
    pub fn new(
        network: Network,
        plasticity_rule: R,
        runtime_config: RuntimeConfig,
        distance_decay_length: f32,
    ) -> Result<Self, SimulationError> {
        Self::with_utility_rule(
            network,
            plasticity_rule,
            Box::new(NoUtility),
            runtime_config,
            distance_decay_length,
        )
    }

    /// Creates a runtime with an explicit utility rule and default utility
    /// dynamics.
    ///
    /// Use [`Self::with_utility_dynamics`] to override the canonical
    /// [`UtilityDynamicsConfig`] defaults.
    pub fn with_utility_rule(
        network: Network,
        plasticity_rule: R,
        utility_rule: Box<dyn UtilityRule>,
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

        let mut simulation = Self {
            network,
            compute: Box::new(CpuCompute),
            profile: None,
            plasticity_rule,
            utility_rule,
            homeostasis: LocalHomeostasis::new(&HomeostasisConfig::default()),
            homeostasis_population: None,
            scheduler: EventScheduler::new(),
            homeostasis_bookings: HomeostasisBookings::default(),
            intrinsic_bookings: IntrinsicBookings::default(),
            event_log: EventLog::default(),
            max_events_per_batch: runtime_config.max_events_per_batch,
            distance_decay_length,
            learning_enabled: true,
            utility_dynamics: UtilityDynamicsConfig::default(),
            poisoned: None,
        };
        simulation.refresh_all_intrinsic_spikes()?;
        Ok(simulation)
    }

    /// Selects a synchronous numerical backend; constructors default to CpuCompute.
    ///
    /// The network remains authoritative. The backend receives only exported
    /// numerical records and flat inputs; the runtime publishes results before
    /// learning hooks and retains event ordering and scheduling.
    pub fn with_compute_backend(mut self, backend: impl ComputeBackend + 'static) -> Self {
        self.compute = Box::new(backend);
        self
    }

    /// Enables bounded host profiling without changing the event stream.
    /// Disabled by default; no wall-clock reads occur while disabled.
    pub fn with_profiling(mut self) -> Self {
        self.enable_profiling();
        self
    }

    /// Starts profiling if it is not already active, retaining existing samples.
    pub fn enable_profiling(&mut self) {
        self.profile.get_or_insert_with(RuntimeProfile::default);
    }

    /// Read-only host diagnostics, separate from deterministic observations.
    pub fn profile(&self) -> Option<&RuntimeProfile> {
        self.profile.as_ref()
    }

    /// Returns the current diagnostics and disables further measurement.
    pub fn take_profile(&mut self) -> Option<RuntimeProfile> {
        self.profile.take()
    }

    fn measure<T, E>(
        &mut self,
        phase: ProfilePhase,
        operation: impl FnOnce(&mut Self) -> Result<T, E>,
    ) -> Result<T, E> {
        let started = self.profile.as_ref().map(|_| Instant::now());
        let result = operation(self);
        if let (Some(started), Some(profile)) = (started, &mut self.profile) {
            profile.record_phase(phase, started.elapsed(), result.is_err());
        }
        result
    }

    /// Sets the canonical utility dynamics parameters.
    ///
    /// Must be called before the simulation is advanced. This is the single
    /// source of truth — the pruning controller and utility rules read from
    /// this config instead of carrying their own copies.
    pub fn with_utility_dynamics(
        mut self,
        dynamics: UtilityDynamicsConfig,
    ) -> Result<Self, SimulationError> {
        dynamics
            .validate()
            .map_err(|_| SimulationError::InvalidUtilityParameter {
                parameter: "utility_dynamics",
                value: 0.0,
            })?;
        self.utility_dynamics = dynamics;
        Ok(self)
    }

    /// Immutable reference to the canonical utility dynamics configuration.
    pub fn utility_dynamics(&self) -> &UtilityDynamicsConfig {
        &self.utility_dynamics
    }

    /// Immutable reference to the utility rule.
    pub fn utility_rule(&self) -> &dyn UtilityRule {
        self.utility_rule.as_ref()
    }

    /// Constructs a runtime from a complete, validated experiment config.
    pub fn from_config(
        network: Network,
        plasticity_rule: R,
        config: &NervaConfig,
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
        simulation.reset_homeostasis_time_anchors()?;
        simulation.start_homeostasis_clocks()?;
        Ok(simulation)
    }

    /// Replaces the default-disabled local homeostasis state and starts every
    /// enabled neuron's independently phased local clock.
    pub fn with_homeostasis(
        mut self,
        homeostasis: LocalHomeostasis,
    ) -> Result<Self, SimulationError> {
        self.homeostasis = homeostasis;
        self.reset_homeostasis_time_anchors()?;
        self.start_homeostasis_clocks()?;
        Ok(self)
    }

    /// Read-only access to authoritative network state.
    pub fn network(&self) -> &Network {
        &self.network
    }

    /// Inserts one neuron and integrates it into every runtime structure.
    ///
    /// The new cell's homeostasis clock starts when local maintenance is
    /// enabled, and its first autonomous threshold crossing is predicted and
    /// scheduled immediately. This is the only supported way to add a neuron
    /// to a live simulation because the scheduler, the homeostasis clocks,
    /// and the intrinsic spike predictions must stay consistent.
    pub fn add_neuron(&mut self, neuron: Neuron) -> Result<(), SimulationError> {
        self.ensure_not_poisoned()?;
        self.transaction(|simulation| {
            let neuron_id = neuron.id();
            simulation
                .network
                .add_neuron(neuron)
                .map_err(SimulationError::InvalidNetwork)?;
            simulation.integrate_new_neuron(neuron_id)
        })
    }

    /// Removes one isolated neuron together with every queued runtime event
    /// that names it.
    ///
    /// The cell must be disconnected first so topology deletion stays
    /// explicit. Pending local maintenance of the removed cell is cancelled;
    /// its predicted autonomous crossing is discarded. This is the only
    /// supported way to remove a neuron from a live simulation.
    pub fn remove_neuron(&mut self, neuron_id: NeuronId) -> Result<Neuron, SimulationError> {
        self.ensure_not_poisoned()?;
        self.transaction(|simulation| {
            // Remove from the authoritative graph first. If this is rejected
            // because a caller has not explicitly disconnected the cell, the
            // transaction restores every existing queue booking unchanged.
            let removed = simulation
                .network
                .remove_neuron(neuron_id)
                .map_err(SimulationError::InvalidNetwork)?;
            simulation.cancel_neuron_events(neuron_id);
            Ok(removed)
        })
    }

    /// Adds one validated synapse to the live network atomically.
    ///
    /// Both endpoints must already exist. Topology is owned by the runtime so
    /// future structural-plasticity rules cannot bypass adjacency invariants.
    pub fn add_synapse(&mut self, synapse: Synapse) -> Result<(), SimulationError> {
        self.ensure_not_poisoned()?;
        self.transaction(|simulation| {
            simulation
                .network
                .add_synapse(synapse)
                .map_err(SimulationError::InvalidNetwork)
        })
    }

    /// Removes one synapse and eagerly cancels every in-flight arrival that
    /// was scheduled through it.
    ///
    /// An arrival captures its amplitude at emission, but it cannot be
    /// delivered after the topology that identifies its synapse is removed:
    /// retaining it would fail batch validation later. This operation is the
    /// live pruning boundary used by structural plasticity.
    pub fn remove_synapse(&mut self, synapse_id: SynapseId) -> Result<Synapse, SimulationError> {
        self.ensure_not_poisoned()?;
        self.transaction(|simulation| {
            let removed = simulation
                .network
                .remove_synapse(synapse_id)
                .map_err(SimulationError::InvalidNetwork)?;
            simulation.cancel_synaptic_arrivals(synapse_id);
            Ok(removed)
        })
    }

    /// Applies one mutation to a neuron's local state and refreshes every
    /// runtime booking that depends on it.
    ///
    /// Intrinsic drive changes move the predicted autonomous crossing;
    /// threshold changes move the crossing too. Because the refresh eagerly
    /// cancels a superseded queue entry, the scheduler stays bounded and
    /// consistent. This is the supported path for controlled interventions
    /// such as homeostatic-style external edits during a paused experiment.
    pub fn update_neuron(
        &mut self,
        neuron_id: NeuronId,
        operation: impl FnOnce(&mut Neuron) -> Result<(), NeuronError>,
    ) -> Result<(), SimulationError> {
        self.ensure_not_poisoned()?;
        self.transaction(|simulation| simulation.update_neuron_inner(neuron_id, operation))
    }

    /// Crate-internal synapse addition without its own transaction.
    ///
    /// Used by the development plan commit so all mutations share one outer
    /// transaction. Caller must already hold an outer transaction.
    #[cfg(feature = "development")]
    pub(crate) fn add_synapse_inner(&mut self, synapse: Synapse) -> Result<(), SimulationError> {
        self.network
            .add_synapse(synapse)
            .map_err(SimulationError::InvalidNetwork)
    }

    /// Crate-internal synapse removal without its own transaction.
    ///
    /// Used by the development plan commit so all mutations share one outer
    /// transaction. Caller must already hold an outer transaction.
    #[cfg(feature = "development")]
    pub(crate) fn remove_synapse_inner(
        &mut self,
        synapse_id: SynapseId,
    ) -> Result<Synapse, SimulationError> {
        let removed = self
            .network
            .remove_synapse(synapse_id)
            .map_err(SimulationError::InvalidNetwork)?;
        self.cancel_synaptic_arrivals(synapse_id);
        Ok(removed)
    }

    /// Crate-internal neuron mutation without its own transaction.
    ///
    /// Used by the development plan commit so all mutations share one outer
    /// transaction. Caller must already hold an outer transaction.
    pub(crate) fn update_neuron_inner(
        &mut self,
        neuron_id: NeuronId,
        operation: impl FnOnce(&mut Neuron) -> Result<(), NeuronError>,
    ) -> Result<(), SimulationError> {
        let neuron = self
            .network
            .neuron_mut(neuron_id)
            .ok_or(SimulationError::UnknownNeuron(neuron_id))?;
        operation(neuron)?;
        let now = self.current_time();
        self.refresh_intrinsic_spike(neuron_id, now)
    }

    /// Applies one mutation to a synapse's local state.
    ///
    /// Only weight, delay, plastic, and enabled flags are mutable; identity
    /// and endpoints cannot change. Weight edits do not require a scheduler
    /// refresh because a travelling arrival captured its amplitude at
    /// emission time.
    pub fn update_synapse(
        &mut self,
        synapse_id: SynapseId,
        operation: impl FnOnce(&mut Synapse) -> Result<(), SynapseError>,
    ) -> Result<(), SimulationError> {
        self.ensure_not_poisoned()?;
        self.transaction(|simulation| {
            let synapse = simulation
                .network
                .synapse_mut(synapse_id)
                .ok_or(SimulationError::UnknownSynapse(synapse_id))?;
            operation(synapse).map_err(SimulationError::InvalidNetworkMutation)?;
            synapse.validate().map_err(|error| {
                SimulationError::InvalidNetwork(NetworkError::InvalidSynapse(error))
            })
        })
    }

    /// Whether a fatal error poisoned this simulation.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.is_some()
    }

    /// The fatal error that made this simulation unusable, if any.
    pub fn poison_reason(&self) -> Option<&SimulationError> {
        self.poisoned.as_deref()
    }

    fn ensure_not_poisoned(&self) -> Result<(), SimulationError> {
        if let Some(reason) = &self.poisoned {
            Err(SimulationError::SimulationPoisoned {
                reason: reason.to_string(),
            })
        } else {
            Ok(())
        }
    }

    fn poison(&mut self, reason: SimulationError) -> SimulationError {
        let stored = reason.to_string();
        if self.poisoned.is_none() {
            self.poisoned = Some(Box::new(reason));
        }
        SimulationError::SimulationPoisoned { reason: stored }
    }

    /// Runs an externally visible runtime mutation atomically.
    ///
    /// `Neuron` and `Network` are core state while event queues and their
    /// booking indices are execution state; all three must commit or roll
    /// back together. The learning-rule and observation log cannot be touched
    /// by these controlled mutation paths and therefore need no snapshot.
    pub(crate) fn transaction<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> Result<T, SimulationError>,
    ) -> Result<T, SimulationError> {
        let network = self.network.clone();
        let homeostasis = self.homeostasis.clone();
        let homeostasis_population = self.homeostasis_population.clone();
        let scheduler = self.scheduler.clone();
        let homeostasis_bookings = self.homeostasis_bookings.clone();
        let intrinsic_bookings = self.intrinsic_bookings.clone();
        match operation(self) {
            Ok(value) => Ok(value),
            Err(error) => {
                self.network = network;
                self.homeostasis = homeostasis;
                self.homeostasis_population = homeostasis_population;
                self.scheduler = scheduler;
                self.homeostasis_bookings = homeostasis_bookings;
                self.intrinsic_bookings = intrinsic_bookings;
                Err(error)
            }
        }
    }

    fn integrate_new_neuron(&mut self, neuron_id: NeuronId) -> Result<(), SimulationError> {
        let now = self.current_time();
        self.refresh_intrinsic_spike(neuron_id, now)?;
        if self.homeostasis.is_enabled() && self.homeostasis_selects(neuron_id) {
            self.network
                .neuron_mut(neuron_id)
                .ok_or(SimulationError::UnknownNeuron(neuron_id))?
                .reset_homeostasis_time_anchor(now)
                .map_err(SimulationError::Neuron)?;
            let interval = self.homeostasis.update_interval_us();
            if interval == 0 {
                return Err(SimulationError::InvalidHomeostasisInterval);
            }
            let initial_offset = homeostasis_initial_phase(neuron_id, interval) + 1;
            let first = now.checked_add_us(initial_offset).ok_or(
                SimulationError::HomeostasisTimeOverflow {
                    time: now,
                    duration_us: initial_offset,
                },
            )?;
            self.schedule_next_homeostasis(neuron_id, first)?;
        }
        Ok(())
    }

    /// Cancels all events that would require a removed neuron's state.
    fn cancel_neuron_events(&mut self, neuron_id: NeuronId) {
        if let Some(booking) = self.homeostasis_bookings.remove(neuron_id) {
            self.scheduler.cancel(booking.sequence);
        }
        if let Some(booking) = self.intrinsic_bookings.remove(neuron_id) {
            self.scheduler.cancel(booking.sequence);
        }
        self.scheduler.cancel_matching(|event| match event {
            EventKind::ExternalInput { target, .. } | EventKind::SynapticArrival { target, .. } => {
                *target == neuron_id
            }
            EventKind::Homeostasis { neuron_id: target }
            | EventKind::IntrinsicSpike { neuron_id: target } => *target == neuron_id,
        });
    }

    /// Cancels every still-queued arrival for a synapse that was pruned.
    fn cancel_synaptic_arrivals(&mut self, synapse_id: SynapseId) {
        self.scheduler.cancel_matching(|event| {
            matches!(
                event,
                EventKind::SynapticArrival {
                    synapse_id: event_synapse_id,
                    ..
                } if *event_synapse_id == synapse_id
            )
        });
    }

    /// Current time of the most recently removed batch.
    pub fn current_time(&self) -> SimTime {
        self.scheduler.current_time()
    }

    /// Number of queued events.
    pub fn pending_event_count(&self) -> usize {
        self.scheduler.len()
    }

    /// Read-only comparison of runtime-owned replay state, including pending
    /// events, local clocks and rule-owned STDP traces. Opaque compute backends
    /// and utility-rule objects are excluded: callers must ensure identical,
    /// stateless implementations. Profiling and observation logs are excluded.
    pub fn same_replay_state(&self, other: &Self) -> bool
    where
        R: PartialEq,
    {
        self.poisoned.is_none()
            && other.poisoned.is_none()
            && self.network == other.network
            && self.scheduler == other.scheduler
            && self.plasticity_rule == other.plasticity_rule
            && self.homeostasis == other.homeostasis
            && self.homeostasis_population == other.homeostasis_population
            && self.homeostasis_bookings == other.homeostasis_bookings
            && self.intrinsic_bookings == other.intrinsic_bookings
            && self.max_events_per_batch == other.max_events_per_batch
            && self.distance_decay_length == other.distance_decay_length
            && self.learning_enabled == other.learning_enabled
            && self.utility_dynamics == other.utility_dynamics
    }

    /// Runtime-owned deadline of this neuron's recurring maintenance clock.
    ///
    /// This is execution bookkeeping, deliberately separate from [`Neuron`]
    /// state. `None` means that no recurring clock is currently queued.
    pub fn next_homeostasis_update(&self, neuron_id: NeuronId) -> Option<SimTime> {
        self.homeostasis_bookings
            .get(neuron_id)
            .map(|booking| booking.time)
    }

    /// Runtime-owned prediction of this neuron's next autonomous spike.
    ///
    /// This is a scheduled execution prediction rather than intrinsic cell
    /// state; it is recomputed whenever a relevant local state change occurs.
    pub fn next_intrinsic_spike(&self, neuron_id: NeuronId) -> Option<SimTime> {
        self.intrinsic_bookings
            .get(neuron_id)
            .map(|booking| booking.time)
    }

    /// Whether an event other than recurring local maintenance remains queued.
    ///
    /// A bounded `run_until` deliberately leaves the next per-neuron
    /// maintenance deadline in the scheduler. Experiments can use this method
    /// to distinguish that expected local clock from unresolved neural work.
    pub fn has_pending_non_homeostasis_events(&self) -> bool {
        self.scheduler
            .any_payload(|event| !matches!(event, EventKind::Homeostasis { .. }))
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

    /// Enables or freezes local activity recording and cellular maintenance.
    ///
    /// Enabling starts any missing neuron-local clocks and resets their elapsed
    /// maintenance anchors. A disabled interval therefore cannot be applied as
    /// one oversized adjustment after re-enabling. Disabling stops rescheduling
    /// after a currently queued maintenance event, so no global maintenance
    /// lifecycle needs to be managed by the caller.
    pub fn set_homeostasis_enabled(&mut self, enabled: bool) -> Result<(), SimulationError> {
        self.ensure_not_poisoned()?;
        self.transaction(|simulation| {
            let was_enabled = simulation.homeostasis.is_enabled();
            simulation.homeostasis.set_enabled(enabled);
            if enabled && !was_enabled {
                simulation.reset_homeostasis_time_anchors()?;
            }
            if enabled {
                simulation.start_homeostasis_clocks()?;
            }
            Ok(())
        })
    }

    /// Whether local homeostasis is active.
    pub fn homeostasis_enabled(&self) -> bool {
        self.homeostasis.is_enabled()
    }

    /// Selects the cells maintained by the existing local rule. None means all;
    /// an empty set means none. Excluded clocks are cancelled, newly included
    /// cells start at the current time, and unknown IDs leave all state intact.
    /// This selects a population; it does not alter the homeostatic rule.
    pub fn set_homeostasis_population(
        &mut self,
        population: Option<BTreeSet<NeuronId>>,
    ) -> Result<(), SimulationError> {
        self.ensure_not_poisoned()?;
        if let Some(ids) = &population {
            for &id in ids {
                if self.network.neuron(id).is_none() {
                    return Err(SimulationError::UnknownNeuron(id));
                }
            }
        }
        self.transaction(|simulation| {
            let previous = simulation.homeostasis_population.clone();
            simulation.homeostasis_population = population;
            let now = simulation.current_time();
            for id in simulation.network.neuron_ids().collect::<Vec<_>>() {
                if !simulation.homeostasis_selects(id) {
                    if let Some(booking) = simulation.homeostasis_bookings.remove(id) {
                        simulation.scheduler.cancel(booking.sequence);
                    }
                } else if previous.as_ref().is_some_and(|ids| !ids.contains(&id)) {
                    simulation
                        .network
                        .neuron_mut(id)
                        .unwrap()
                        .reset_homeostasis_time_anchor(now)?;
                }
            }
            simulation.start_homeostasis_clocks()
        })
    }

    fn homeostasis_selects(&self, id: NeuronId) -> bool {
        self.homeostasis_population
            .as_ref()
            .is_none_or(|ids| ids.contains(&id))
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
        self.ensure_not_poisoned()?;
        self.schedule_event(Event::new(
            at,
            EventKind::ExternalInput { target, amplitude },
        ))
    }

    /// Schedules one one-shot local maintenance event.
    ///
    /// For recurring local clocks, use [`Self::start_homeostasis_clocks`].
    pub fn schedule_homeostasis(
        &mut self,
        at: SimTime,
        neuron_id: NeuronId,
    ) -> Result<u64, SimulationError> {
        self.ensure_not_poisoned()?;
        self.schedule_event(Event::new(at, EventKind::Homeostasis { neuron_id }))
    }

    /// Starts one recurring, independently scheduled maintenance clock per
    /// neuron when local homeostasis is enabled.
    ///
    /// The first deadlines are deterministically staggered by stable neuron
    /// identity. This schedules local events only; it never scans and updates
    /// all cells in a global tick.
    pub fn start_homeostasis_clocks(&mut self) -> Result<(), SimulationError> {
        if !self.homeostasis.is_enabled() {
            return Ok(());
        }

        let now = self.current_time();
        let interval = self.homeostasis.update_interval_us();
        if interval == 0 {
            return Err(SimulationError::InvalidHomeostasisInterval);
        }
        let neuron_ids: Vec<_> = self.network.neuron_ids().collect();
        for neuron_id in neuron_ids {
            if !self.homeostasis_selects(neuron_id) {
                continue;
            }
            if self.homeostasis_bookings.get(neuron_id).is_some() {
                continue;
            }
            let initial_offset = homeostasis_initial_phase(neuron_id, interval) + 1;
            let first = now.checked_add_us(initial_offset).ok_or(
                SimulationError::HomeostasisTimeOverflow {
                    time: now,
                    duration_us: initial_offset,
                },
            )?;
            self.schedule_next_homeostasis(neuron_id, first)?;
        }
        Ok(())
    }

    /// Processes the next complete timestamp batch.
    ///
    /// A batch is validated completely before the first mutation. If a fatal
    /// error still occurs midway, the simulation is poisoned: partially
    /// applied state may remain, and every further use is rejected.
    pub fn step(&mut self) -> Result<Option<BatchReport>, SimulationError> {
        self.ensure_not_poisoned()?;
        let Some(batch) = self.measure(ProfilePhase::SchedulerDequeue, |simulation| {
            simulation
                .scheduler
                .pop_next_batch(simulation.max_events_per_batch)
        })?
        else {
            return Ok(None);
        };
        if let Some(profile) = &mut self.profile {
            profile.batches_started = profile.batches_started.saturating_add(1);
            profile.scheduler_batch_size.observe(batch.len());
            profile.synaptic_arrivals_per_timestamp.observe(
                batch
                    .events()
                    .iter()
                    .filter(|event| matches!(event.payload, EventKind::SynapticArrival { .. }))
                    .count(),
            );
        }
        let result = self.measure(ProfilePhase::Batch, |simulation| {
            simulation.process_batch(batch)
        });
        if let Some(profile) = &mut self.profile {
            if let Ok(report) = &result {
                profile.batches_completed = profile.batches_completed.saturating_add(1);
                profile.spikes_per_timestamp.observe(report.spikes_emitted);
            } else {
                profile.batches_failed = profile.batches_failed.saturating_add(1);
            }
        }
        match result {
            Ok(report) => Ok(Some(report)),
            Err(error) => Err(self.poison(error)),
        }
    }

    /// Runs a finite event stream until no event remains.
    ///
    /// Recurring local maintenance or autonomous intrinsic firing have no
    /// implicit end time, so those simulations must use [`Self::run_until`].
    pub fn run(&mut self) -> Result<RunReport, SimulationError> {
        self.ensure_not_poisoned()?;
        if self.has_open_ended_local_events() {
            return Err(SimulationError::RunRequiresDeadline);
        }
        let mut report = RunReport::default();
        while let Some(batch) = self.step()? {
            report.include(batch);
            if self.has_open_ended_local_events() {
                return Err(SimulationError::RunRequiresDeadline);
            }
        }
        Ok(report)
    }

    /// Runs every event whose timestamp is at most `inclusive`.
    pub fn run_until(&mut self, inclusive: SimTime) -> Result<RunReport, SimulationError> {
        self.ensure_not_poisoned()?;
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
            EventKind::Homeostasis { neuron_id } | EventKind::IntrinsicSpike { neuron_id } => {
                self.require_neuron(neuron_id)
            }
        }
    }

    fn require_neuron(&self, id: NeuronId) -> Result<(), SimulationError> {
        self.network
            .neuron(id)
            .map(|_| ())
            .ok_or(SimulationError::UnknownNeuron(id))
    }

    fn has_open_ended_local_events(&self) -> bool {
        (self.homeostasis.is_enabled() && !self.homeostasis_bookings.is_empty())
            || !self.intrinsic_bookings.is_empty()
    }

    fn reset_homeostasis_time_anchors(&mut self) -> Result<(), SimulationError> {
        let now = self.current_time();
        let population = self.homeostasis_population.as_ref();
        for neuron in self.network.neurons_mut() {
            if population.is_some_and(|ids| !ids.contains(&neuron.id())) {
                continue;
            }
            neuron.reset_homeostasis_time_anchor(now)?;
        }
        Ok(())
    }

    fn schedule_next_homeostasis(
        &mut self,
        neuron_id: NeuronId,
        time: SimTime,
    ) -> Result<(), SimulationError> {
        let sequence =
            self.schedule_event(Event::new(time, EventKind::Homeostasis { neuron_id }))?;
        self.homeostasis_bookings
            .insert(neuron_id, HomeostasisBooking { time, sequence });
        Ok(())
    }

    fn refresh_all_intrinsic_spikes(&mut self) -> Result<(), SimulationError> {
        let neuron_ids: Vec<_> = self.network.neuron_ids().collect();
        let now = self.current_time();
        for neuron_id in neuron_ids {
            self.refresh_intrinsic_spike(neuron_id, now)?;
        }
        Ok(())
    }

    fn refresh_intrinsic_spike(
        &mut self,
        neuron_id: NeuronId,
        now: SimTime,
    ) -> Result<(), SimulationError> {
        let (current, predicted) = {
            let neuron = self
                .network
                .neuron(neuron_id)
                .ok_or(SimulationError::UnknownNeuron(neuron_id))?;
            (
                self.intrinsic_bookings.get(neuron_id),
                neuron.predicted_intrinsic_spike_at(now),
            )
        };
        if current.map(|booking| booking.time) == predicted {
            return Ok(());
        }
        if let Some(booking) = current {
            // A prediction is replaced rather than left as a stale future
            // event. This keeps queue space bounded by the live prediction
            // count even when frequent inputs continually move crossings.
            self.scheduler.cancel(booking.sequence);
            self.intrinsic_bookings.remove(neuron_id);
        }
        if let Some(predicted) = predicted {
            let sequence = self.schedule_event(Event::new(
                predicted,
                EventKind::IntrinsicSpike { neuron_id },
            ))?;
            self.intrinsic_bookings.insert(
                neuron_id,
                IntrinsicBooking {
                    time: predicted,
                    sequence,
                },
            );
        }
        Ok(())
    }

    fn process_batch(
        &mut self,
        batch: EventBatch<EventKind>,
    ) -> Result<BatchReport, SimulationError> {
        let time = batch.time();
        let events = batch.into_events();
        self.validate_batch(&events)?;
        let neurons = self.collect_batch_neurons(time, &events);

        // Pre-arrival learning observes advanced cells before input integration.
        if let Some(profile) = &mut self.profile {
            profile.advance_batch_size.observe(neurons.affected.len());
        }
        let mut state = self.measure(ProfilePhase::AdvanceExport, |simulation| {
            DenseNeuronBatch::from_network(&simulation.network, &neurons.affected)
        })?;
        let result = self.measure(ProfilePhase::AdvanceCompute, |simulation| {
            simulation.compute.advance_neurons(&mut state, time)
        });
        self.measure(ProfilePhase::AdvancePublish, |simulation| {
            state.complete(&mut simulation.network, result)
        })?;
        let inputs = self.measure(ProfilePhase::Arrivals, |simulation| {
            simulation.process_arrivals(time, &events)
        })?;
        let spikes = self.integrate_batch_neurons(time, neurons.due_intrinsic, inputs)?;
        self.record_batch_spikes(time, &spikes)?;
        self.apply_batch_learning(&spikes)?;
        self.measure(ProfilePhase::Maintenance, |simulation| {
            simulation.maintain_batch_neurons(time, neurons.maintenance)
        })?;
        self.measure(ProfilePhase::Propagation, |simulation| {
            simulation.propagate_batch_spikes(&spikes)
        })?;
        self.measure(ProfilePhase::Prediction, |simulation| {
            simulation.refresh_batch_predictions(time, neurons.affected)
        })?;

        Ok(BatchReport {
            time,
            events_processed: events.len(),
            spikes_emitted: spikes.len(),
        })
    }

    fn validate_batch(&self, events: &[ScheduledEvent<EventKind>]) -> Result<(), SimulationError> {
        // Validate the whole batch before the first state mutation. This also
        // catches controlled network edits made after an event was scheduled.
        for scheduled in events {
            self.validate_event(scheduled.payload)?;
        }

        Ok(())
    }

    fn collect_batch_neurons(
        &self,
        time: SimTime,
        events: &[ScheduledEvent<EventKind>],
    ) -> BatchNeurons {
        let mut affected_neurons = BTreeSet::new();
        let mut maintenance_neurons = BTreeSet::new();
        let mut due_intrinsic_neurons = BTreeSet::new();
        for scheduled in events {
            match scheduled.payload {
                EventKind::ExternalInput { target, .. }
                | EventKind::SynapticArrival { target, .. } => {
                    affected_neurons.insert(target);
                }
                EventKind::Homeostasis { neuron_id } => {
                    affected_neurons.insert(neuron_id);
                    maintenance_neurons.insert(neuron_id);
                }
                EventKind::IntrinsicSpike { neuron_id } => {
                    // Cancellation keeps this normally exact. Retaining this
                    // identity check also protects deterministic behavior if a
                    // caller supplied an older serialized runtime state.
                    if self.intrinsic_bookings.contains(neuron_id, time) {
                        affected_neurons.insert(neuron_id);
                        due_intrinsic_neurons.insert(neuron_id);
                    }
                }
            }
        }

        BatchNeurons {
            affected: affected_neurons.into_iter().collect(),
            maintenance: maintenance_neurons,
            due_intrinsic: due_intrinsic_neurons,
        }
    }

    fn process_arrivals(
        &mut self,
        time: SimTime,
        events: &[ScheduledEvent<EventKind>],
    ) -> Result<BTreeMap<NeuronId, Vec<f32>>, SimulationError> {
        let mut simultaneous_inputs = BTreeMap::<NeuronId, Vec<f32>>::new();
        for scheduled in events {
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
                    let synapse = self
                        .network
                        .synapse_mut(synapse_id)
                        .ok_or(SimulationError::UnknownSynapse(synapse_id))?;
                    synapse.record_transmission();

                    // Record utility eligibility `e_ij` for every enabled
                    // synapse, independent of whether the STDP rule accepts
                    // it and independent of whether the synapse is weight-
                    // plastic. Structural utility is a separate concern from
                    // weight learning: even a non-plastic synapse may be
                    // structurally useful and should be eligible for
                    // utility-based pruning.
                    if self.learning_enabled && synapse.is_enabled() {
                        synapse
                            .record_utility_arrival(time, self.utility_dynamics.eligibility_tau_us)
                            .map_err(SimulationError::InvalidNetworkMutation)?;
                    }

                    if self.learning_enabled && accepted_by_rule {
                        self.measure(ProfilePhase::PreStdp, |simulation| {
                            let post = simulation
                                .network
                                .neuron(target)
                                .ok_or(SimulationError::UnknownNeuron(target))?
                                .clone();
                            let (old_weight, new_weight) = {
                                let synapse = simulation
                                    .network
                                    .synapse_mut(synapse_id)
                                    .ok_or(SimulationError::UnknownSynapse(synapse_id))?;
                                let old_weight = synapse.weight();
                                simulation
                                    .plasticity_rule
                                    .on_pre_arrival(synapse, &post, time);
                                (old_weight, synapse.weight())
                            };
                            simulation.log_weight_change(time, synapse_id, old_weight, new_weight);
                            Ok::<_, SimulationError>(())
                        })?;
                    }

                    collect_input(&mut simultaneous_inputs, target, amplitude);
                }
                EventKind::Homeostasis { .. } | EventKind::IntrinsicSpike { .. } => {}
            }
        }

        Ok(simultaneous_inputs)
    }

    fn integrate_batch_neurons(
        &mut self,
        time: SimTime,
        mut firing_candidates: BTreeSet<NeuronId>,
        mut simultaneous_inputs: BTreeMap<NeuronId, Vec<f32>>,
    ) -> Result<Vec<Spike>, SimulationError> {
        firing_candidates.extend(simultaneous_inputs.keys().copied());
        for &neuron_id in &firing_candidates {
            if self.intrinsic_bookings.contains(neuron_id, time) {
                self.intrinsic_bookings.remove(neuron_id);
            }
        }
        let (mut inputs, mut state) =
            self.measure(ProfilePhase::IntegratePrepare, |simulation| {
                let inputs =
                    FlatNeuronInputs::from_rows(firing_candidates.into_iter().map(|neuron_id| {
                        (
                            neuron_id,
                            simultaneous_inputs.remove(&neuron_id).unwrap_or_default(),
                        )
                    }))?;
                let state =
                    DenseNeuronBatch::from_network(&simulation.network, inputs.neuron_ids())?;
                Ok::<_, ComputeError>((inputs, state))
            })?;
        if let Some(profile) = &mut self.profile {
            profile
                .integrate_batch_size
                .observe(inputs.neuron_ids().len());
            for offsets in inputs.offsets().windows(2) {
                profile
                    .inputs_per_neuron
                    .observe((offsets[1] - offsets[0]) as usize);
            }
        }
        let result = self.measure(ProfilePhase::IntegrateCompute, |simulation| {
            simulation
                .compute
                .integrate_neurons(&mut state, &mut inputs, time)
        });
        Ok(self.measure(ProfilePhase::IntegratePublish, |simulation| {
            state.complete(&mut simulation.network, result)
        })?)
    }

    fn record_batch_spikes(
        &mut self,
        time: SimTime,
        spikes: &[Spike],
    ) -> Result<(), SimulationError> {
        for &spike in spikes {
            self.event_log.push(ObservationEvent::SpikeEmitted(spike));
            let neuron = self
                .network
                .neuron(spike.neuron_id)
                .ok_or(SimulationError::UnknownNeuron(spike.neuron_id))?;
            self.homeostasis.record_spike(neuron, time)?;
        }

        Ok(())
    }

    fn apply_batch_learning(&mut self, spikes: &[Spike]) -> Result<(), SimulationError> {
        if self.learning_enabled {
            self.measure(ProfilePhase::PostStdp, |simulation| {
                for &spike in spikes {
                    simulation.apply_post_spike_learning(spike)?;
                }
                Ok::<_, SimulationError>(())
            })?;
            self.measure(ProfilePhase::Utility, |simulation| {
                for &spike in spikes {
                    simulation.apply_post_spike_utility(spike)?;
                }
                Ok::<_, SimulationError>(())
            })?;
        }

        Ok(())
    }

    fn maintain_batch_neurons(
        &mut self,
        time: SimTime,
        maintenance_neurons: BTreeSet<NeuronId>,
    ) -> Result<(), SimulationError> {
        // Duplicate maintenance events for one cell and timestamp collapse into
        // one local update, just as duplicate arrivals collapse into one firing check.
        for neuron_id in maintenance_neurons {
            let is_recurring_clock = self.homeostasis_bookings.contains(neuron_id, time);
            if is_recurring_clock {
                self.homeostasis_bookings.remove(neuron_id);
            }
            if !self.homeostasis_selects(neuron_id) {
                continue;
            }
            let change = {
                let neuron = self
                    .network
                    .neuron_mut(neuron_id)
                    .ok_or(SimulationError::UnknownNeuron(neuron_id))?;
                self.homeostasis.maintain(neuron, time)?
            };
            if let Some(change) = change {
                self.event_log.push(ObservationEvent::HomeostasisChanged {
                    time,
                    neuron_id: change.neuron_id,
                    firing_avg: change.firing_avg,
                    input_avg: change.input_avg,
                    old_intrinsic_current: change.old_intrinsic_current,
                    new_intrinsic_current: change.new_intrinsic_current,
                    old_structural_drive: change.old_structural_drive,
                    new_structural_drive: change.new_structural_drive,
                });
            }
            if is_recurring_clock && self.homeostasis.is_enabled() {
                let next = time
                    .checked_add_us(self.homeostasis.update_interval_us())
                    .ok_or(SimulationError::HomeostasisTimeOverflow {
                        time,
                        duration_us: self.homeostasis.update_interval_us(),
                    })?;
                self.schedule_next_homeostasis(neuron_id, next)?;
            }
        }

        Ok(())
    }

    fn propagate_batch_spikes(&mut self, spikes: &[Spike]) -> Result<(), SimulationError> {
        // Post hooks and maintenance finish before any outgoing weights are read.
        // Positive validated delays ensure these events cannot re-enter this batch.
        let mut transmissions = Vec::new();
        for &spike in spikes {
            if let Some(profile) = &mut self.profile {
                profile
                    .outgoing_synapses_per_spike
                    .observe(self.network.outgoing_synapse_ids(spike.neuron_id).len());
            }

            self.measure(ProfilePhase::PropagationPlan, |simulation| {
                plan_spike_propagation_into(
                    &simulation.network,
                    spike,
                    simulation.distance_decay_length,
                    &mut transmissions,
                )
            })?;
            self.measure(ProfilePhase::PropagationEnqueue, |simulation| {
                // The planner has checked endpoints and finite amplitudes;
                // no topology or learning mutation occurs before insertion.
                // Scheduler time/sequence checks remain authoritative, and
                // arrivals are validated again when their batch is dequeued.
                for transmission in &transmissions {
                    simulation
                        .scheduler
                        .schedule(transmission.event.time, transmission.event.kind)
                        .map_err(SimulationError::Scheduler)?;
                }
                Ok::<_, SimulationError>(())
            })?;
        }

        Ok(())
    }

    fn refresh_batch_predictions(
        &mut self,
        time: SimTime,
        affected_neurons: Vec<NeuronId>,
    ) -> Result<(), SimulationError> {
        // Input, spike reset, and maintenance can all change the next
        // autonomous crossing. Each affected neuron refreshes only its own
        // prediction and eagerly cancels a superseded queue entry.
        for neuron_id in affected_neurons {
            self.refresh_intrinsic_spike(neuron_id, time)?;
        }

        Ok(())
    }

    fn apply_post_spike_learning(&mut self, spike: Spike) -> Result<(), SimulationError> {
        let neuron = self
            .network
            .neuron(spike.neuron_id)
            .ok_or(SimulationError::UnknownNeuron(spike.neuron_id))?;
        let mut originals = Vec::new();
        let mut incoming = Vec::new();
        for &synapse_id in self.network.incoming_synapse_ids(spike.neuron_id) {
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
                originals.push((synapse_id, synapse.pre(), synapse.weight()));
                incoming.push(synapse.clone());
            }
        }

        self.plasticity_rule
            .on_post_spike(neuron, &mut incoming, spike.time);

        // The rule only receives staged synapse copies. Validate the complete
        // result against the captured identities before publishing any copy.
        for (&(synapse_id, pre, _), updated) in originals.iter().zip(&incoming) {
            if updated.id() != synapse_id
                || updated.post() != spike.neuron_id
                || updated.pre() != pre
            {
                return Err(SimulationError::InvalidLearningMutation { synapse_id });
            }
            updated.validate().map_err(|error| {
                SimulationError::InvalidNetwork(NetworkError::InvalidSynapse(error))
            })?;
        }

        for ((synapse_id, _, old_weight), updated) in originals.into_iter().zip(incoming) {
            let new_weight = updated.weight();
            *self
                .network
                .synapse_mut(synapse_id)
                .ok_or(SimulationError::UnknownSynapse(synapse_id))? = updated;
            self.log_weight_change(spike.time, synapse_id, old_weight, new_weight);
        }
        Ok(())
    }

    /// Evaluates utility evidence for every incoming synapse of a neuron that
    /// just fired and updates the synapse's utility memory when the rule
    /// returns `Some(sample)`.
    ///
    /// This is the structural-utility analogue of [`Self::apply_post_spike_learning`].
    /// It iterates over **all** incoming synapses of the spiking neuron, not
    /// only those accepted by the plasticity rule: utility evidence is a
    /// separate concern from STDP and must not depend on which connections the
    /// weight-learning rule chooses to update.
    fn apply_post_spike_utility(&mut self, spike: Spike) -> Result<(), SimulationError> {
        let post_neuron = self
            .network
            .neuron(spike.neuron_id)
            .ok_or(SimulationError::UnknownNeuron(spike.neuron_id))?
            .clone();
        let incoming_ids = self.network.incoming_synapse_ids(spike.neuron_id).to_vec();
        let eligibility_tau = self.utility_dynamics.eligibility_tau_us;
        let eta = self.utility_dynamics.eta;
        let utility_tau = self.utility_dynamics.utility_tau_us;

        for synapse_id in incoming_ids {
            let synapse = self
                .network
                .synapse(synapse_id)
                .ok_or(SimulationError::UnknownSynapse(synapse_id))?;
            let polarity = self
                .network
                .neuron(synapse.pre())
                .ok_or(SimulationError::UnknownNeuron(spike.neuron_id))?
                .polarity();

            // Skip synapses the active utility rule cannot judge: their
            // memory must stay untouched so "unknown" is not conflated
            // with "bad".
            if !self.utility_rule.supports(polarity) {
                continue;
            }

            let sample = {
                // Compute the decayed eligibility with the canonical τ_e so
                // the rule does not need its own copy.
                let eligibility = synapse
                    .utility_eligibility_at(spike.time, eligibility_tau)
                    .map_err(SimulationError::InvalidNetworkMutation)?;
                let context = UtilityContext {
                    synapse,
                    post_neuron: &post_neuron,
                    presynaptic_polarity: polarity,
                    now: spike.time,
                    trigger: UtilityTrigger::PostSpike,
                    eligibility,
                };
                self.utility_rule.evidence(&context)
            };

            if let Some(sample) = sample {
                self.network
                    .synapse_mut(synapse_id)
                    .ok_or(SimulationError::UnknownSynapse(synapse_id))?
                    .update_utility(sample, eta, spike.time, utility_tau)
                    .map_err(SimulationError::InvalidNetworkMutation)?;
            }
        }
        Ok(())
    }

    fn log_weight_change(
        &mut self,
        time: SimTime,
        synapse_id: SynapseId,
        old_weight: Weight,
        new_weight: Weight,
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

/// Deterministically mixes a stable neuron ID into a local-clock phase.
///
/// Sequential IDs would otherwise make low-numbered phases line up in the
/// same order on every interval. SplitMix64 is fixed arithmetic (not the
/// process-randomized standard hasher), so replay is stable across runs and
/// platforms while phases remain well spread over `0..interval`.
fn homeostasis_initial_phase(neuron_id: NeuronId, interval: u64) -> u64 {
    debug_assert!(interval > 0);
    let mut value = neuron_id.get().wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (value ^ (value >> 31)) % interval
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

#[cfg(test)]
mod tests {
    use crate::{
        config::{HomeostasisConfig, NervaConfig, NeuronConfig},
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
            intrinsic: Default::default(),
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
                Synapse::new(
                    SynapseId(1),
                    NeuronId(1),
                    NeuronId(2),
                    Weight::new(1.0).unwrap(),
                    5,
                    true,
                )
                .unwrap(),
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
            synapse
                .set_weight(Weight::new(synapse.weight().get() + 0.1).unwrap())
                .unwrap();
        }

        fn on_post_spike(&mut self, _neuron: &Neuron, incoming: &mut [Synapse], _time: SimTime) {
            self.post_synapses += incoming.len();
            for synapse in incoming {
                synapse
                    .set_weight(Weight::new(synapse.weight().get() + 0.2).unwrap())
                    .unwrap();
            }
        }
    }

    #[test]
    fn buffered_propagation_matches_public_plans_and_validated_scheduling() {
        let mut network = connected_network(Polarity::Inhibitory);
        network
            .add_neuron(neuron(91, Polarity::Excitatory))
            .unwrap();
        // Insert out of ID order, include a disabled edge and an empty source.
        for (id, pre, post, delay, enabled) in [
            (90, 2, 91, 9, true),
            (8, 1, 91, 5, false),
            (5, 1, 2, 5, true),
        ] {
            let mut synapse = Synapse::new(
                SynapseId(id),
                NeuronId(pre),
                NeuronId(post),
                Weight::new(0.1).unwrap(),
                delay,
                false,
            )
            .unwrap();
            synapse.set_enabled(enabled);
            network.add_synapse(synapse).unwrap();
        }
        let mut reference = simulation(network.clone(), NoPlasticity);
        let mut buffered = simulation(network, NoPlasticity);
        // Exercise buffer clearing for different degrees and repeated spikes.
        let spikes: Vec<_> = [(1, 0), (91, 0), (2, 0), (1, 20), (2, 20)]
            .into_iter()
            .map(|(id, time)| Spike::new(NeuronId(id), SimTime(time)))
            .collect();
        for &spike in &spikes {
            for transmission in
                super::super::propagation::plan_spike_propagation(reference.network(), spike, 1.0)
                    .unwrap()
            {
                reference.schedule_event(transmission.event).unwrap();
            }
        }
        buffered.propagate_batch_spikes(&spikes).unwrap();
        assert_eq!(buffered.run().unwrap(), reference.run().unwrap());
        assert_eq!(buffered.network(), reference.network());
        assert_eq!(buffered.event_log(), reference.event_log());
    }

    #[test]
    fn a_failed_plan_never_enqueues_its_successful_prefix() {
        let mut network = connected_network(Polarity::Excitatory);
        network
            .add_synapse(
                Synapse::new(
                    SynapseId(2),
                    NeuronId(1),
                    NeuronId(2),
                    Weight::new(0.1).unwrap(),
                    10,
                    false,
                )
                .unwrap(),
            )
            .unwrap();
        let mut simulation = simulation(network, NoPlasticity).with_profiling();
        simulation
            .schedule_external_input(SimTime(u64::MAX - 6), NeuronId(1), 1.1)
            .unwrap();
        assert!(simulation.step().is_err());
        assert!(simulation.is_poisoned());
        assert!(matches!(
            simulation.poison_reason(),
            Some(SimulationError::Propagation(
                PropagationError::ArrivalTimeOverflow { .. }
            ))
        ));
        assert_eq!(simulation.pending_event_count(), 0);
        let profile = simulation.profile().unwrap();
        assert_eq!(profile.phase(ProfilePhase::PropagationPlan).failures, 1);
        assert_eq!(profile.phase(ProfilePhase::PropagationEnqueue).calls, 0);
    }

    #[test]
    fn post_learning_validates_all_staged_synapses_before_publishing_any() {
        struct CorruptPost(u8);
        impl PlasticityRule for CorruptPost {
            fn accepts(&self, _: &Synapse, _: Polarity) -> bool {
                true
            }
            fn on_pre_arrival(&mut self, _: &mut Synapse, _: &Neuron, _: SimTime) {}
            fn on_post_spike(&mut self, _: &Neuron, incoming: &mut [Synapse], _: SimTime) {
                incoming[0].set_weight(Weight::new(0.9).unwrap()).unwrap();
                match self.0 {
                    0 => incoming[1].id = SynapseId(999),
                    1 => incoming[1].pre = NeuronId(2),
                    2 => incoming[1].post = NeuronId(1),
                    _ => incoming[1].delay_us = 0,
                }
            }
        }
        for corruption in 0..4 {
            let mut network = connected_network(Polarity::Excitatory);
            network
                .add_synapse(
                    Synapse::new(
                        SynapseId(2),
                        NeuronId(1),
                        NeuronId(2),
                        Weight::new(0.2).unwrap(),
                        5,
                        true,
                    )
                    .unwrap(),
                )
                .unwrap();
            let original = network.clone();
            let mut simulation = simulation(network, CorruptPost(corruption));
            assert!(
                simulation
                    .apply_post_spike_learning(Spike::new(NeuronId(2), SimTime::ZERO))
                    .is_err()
            );
            assert_eq!(simulation.network(), &original);
            assert!(simulation.event_log().is_empty());
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
            Simulation::from_config(network, NoPlasticity, &NervaConfig::default()),
            Err(SimulationError::InvalidNetwork(
                NetworkError::InconsistentIndices
            ))
        ));
    }

    #[test]
    fn aggregate_constructor_rejects_network_config_mismatch() {
        let network = connected_network(Polarity::Excitatory);
        let error = Simulation::from_config(network, NoPlasticity, &NervaConfig::default())
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
        let learned_weight = learning
            .network()
            .synapse(SynapseId(1))
            .unwrap()
            .weight()
            .get();
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
        assert_eq!(
            network.synapse(SynapseId(1)).unwrap().weight(),
            Weight::new(1.0).unwrap()
        );
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
        assert_eq!(
            network.synapse(SynapseId(1)).unwrap().weight(),
            Weight::new(1.0).unwrap()
        );
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
        // A scheduler safety failure leaves the runtime usable.
        assert!(!simulation.is_poisoned());
    }

    #[test]
    fn controlled_add_neuron_integrates_homeostasis_and_intrinsic_prediction() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 1_000,
            ..HomeostasisConfig::default()
        });
        let mut simulation = simulation(one_neuron_network(), NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        let driven = Neuron::new(
            NeuronId(2),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            None,
            NeuronConfig {
                intrinsic: crate::config::IntrinsicDynamicsConfig {
                    intrinsic_drive: 2_000_000.0,
                    ..crate::config::IntrinsicDynamicsConfig::default()
                },
                ..neuron_params()
            },
            SimTime::ZERO,
        )
        .unwrap();

        simulation.add_neuron(driven).unwrap();

        // The new cell's autonomous crossing was predicted and scheduled.
        assert!(simulation.next_intrinsic_spike(NeuronId(2)).is_some());
        // Its local maintenance clock started with the shared phase rule.
        assert!(simulation.next_homeostasis_update(NeuronId(2)).is_some());
        assert!(simulation.run_until(SimTime(10_000)).is_ok());
    }

    #[test]
    fn remove_neuron_cancels_intrinsic_homeostasis_and_external_bookings() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 1_000,
            ..HomeostasisConfig::default()
        });
        let mut simulation = simulation(one_neuron_network(), NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        let driven = Neuron::new(
            NeuronId(9),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            None,
            NeuronConfig {
                intrinsic: crate::config::IntrinsicDynamicsConfig {
                    intrinsic_drive: 2_000_000.0,
                    ..crate::config::IntrinsicDynamicsConfig::default()
                },
                ..neuron_params()
            },
            SimTime::ZERO,
        )
        .unwrap();
        simulation.add_neuron(driven).unwrap();
        simulation
            .schedule_external_input(SimTime(5_000), NeuronId(9), 0.5)
            .unwrap();
        assert!(simulation.next_intrinsic_spike(NeuronId(9)).is_some());
        assert!(simulation.next_homeostasis_update(NeuronId(9)).is_some());
        assert_eq!(simulation.pending_event_count(), 4);

        let removed = simulation.remove_neuron(NeuronId(9)).unwrap();
        assert_eq!(removed.id(), NeuronId(9));
        assert!(simulation.network().neuron(NeuronId(9)).is_none());
        assert!(simulation.next_intrinsic_spike(NeuronId(9)).is_none());
        assert!(simulation.next_homeostasis_update(NeuronId(9)).is_none());
        // Only neuron 1's recurring local clock remains. No stale event can
        // later fail validation because it refers to the removed cell.
        assert_eq!(simulation.pending_event_count(), 1);
        assert!(simulation.run_until(SimTime(5_000)).is_ok());
    }

    #[test]
    fn failed_connected_removal_preserves_all_runtime_bookings() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 1_000,
            ..HomeostasisConfig::default()
        });
        let mut simulation = simulation(connected_network(Polarity::Excitatory), NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(5_000), NeuronId(1), 0.5)
            .unwrap();
        let before_events = simulation.pending_event_count();
        let before_clock = simulation.next_homeostasis_update(NeuronId(1));

        assert!(matches!(
            simulation.remove_neuron(NeuronId(1)),
            Err(SimulationError::InvalidNetwork(
                NetworkError::NeuronStillConnected { .. }
            ))
        ));
        assert!(simulation.network().neuron(NeuronId(1)).is_some());
        assert_eq!(simulation.pending_event_count(), before_events);
        assert_eq!(
            simulation.next_homeostasis_update(NeuronId(1)),
            before_clock
        );
        assert!(simulation.run_until(SimTime(5_000)).is_ok());
    }

    #[test]
    fn live_synapse_pruning_cancels_in_flight_arrivals_and_unblocks_neuron_removal() {
        let mut network = Network::new();
        network.add_neuron(neuron(1, Polarity::Excitatory)).unwrap();
        network.add_neuron(neuron(2, Polarity::Excitatory)).unwrap();
        let mut simulation = simulation(network, NoPlasticity);

        simulation
            .add_synapse(
                Synapse::new(
                    SynapseId(7),
                    NeuronId(1),
                    NeuronId(2),
                    Weight::new(1.0).unwrap(),
                    100,
                    true,
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(simulation.network().synapse_count(), 1);

        simulation
            .schedule_external_input(SimTime::ZERO, NeuronId(1), 1.0)
            .unwrap();
        simulation.run_until(SimTime::ZERO).unwrap();
        assert_eq!(simulation.pending_event_count(), 1);

        let removed = simulation.remove_synapse(SynapseId(7)).unwrap();
        assert_eq!(removed.id(), SynapseId(7));
        assert_eq!(simulation.network().synapse_count(), 0);
        assert_eq!(simulation.pending_event_count(), 0);
        // No delayed arrival can reach batch validation after pruning.
        assert!(simulation.run_until(SimTime(100)).is_ok());
        assert_eq!(
            simulation
                .network()
                .neuron(NeuronId(2))
                .unwrap()
                .spike_count(),
            0
        );

        // The public live-topology path now permits explicit dismantling.
        assert_eq!(
            simulation.remove_neuron(NeuronId(1)).unwrap().id(),
            NeuronId(1)
        );
        assert_eq!(
            simulation.remove_neuron(NeuronId(2)).unwrap().id(),
            NeuronId(2)
        );
    }

    #[test]
    fn failed_live_synapse_removal_preserves_queue_and_topology() {
        let mut simulation = simulation(connected_network(Polarity::Excitatory), NoPlasticity);
        simulation
            .schedule_external_input(SimTime(5_000), NeuronId(1), 0.5)
            .unwrap();
        let before_events = simulation.pending_event_count();

        assert!(matches!(
            simulation.remove_synapse(SynapseId(999)),
            Err(SimulationError::InvalidNetwork(
                NetworkError::MissingSynapse(SynapseId(999))
            ))
        ));
        assert!(simulation.network().synapse(SynapseId(1)).is_some());
        assert_eq!(simulation.pending_event_count(), before_events);
    }

    #[test]
    fn failed_live_synapse_addition_preserves_topology() {
        let mut simulation = simulation(one_neuron_network(), NoPlasticity);

        assert!(matches!(
            simulation.add_synapse(
                Synapse::new(
                    SynapseId(7),
                    NeuronId(1),
                    NeuronId(2),
                    Weight::new(1.0).unwrap(),
                    1,
                    true,
                )
                .unwrap(),
            ),
            Err(SimulationError::InvalidNetwork(
                NetworkError::MissingPostsynapticNeuron(NeuronId(2))
            ))
        ));
        assert_eq!(simulation.network().synapse_count(), 0);
        assert_eq!(simulation.pending_event_count(), 0);
    }

    #[test]
    fn update_neuron_refreshes_the_autonomous_prediction() {
        let mut simulation = simulation(one_neuron_network(), NoPlasticity);
        assert!(simulation.next_intrinsic_spike(NeuronId(1)).is_none());

        simulation
            .update_neuron(NeuronId(1), |neuron| {
                neuron.set_intrinsic_current(1_000_000.0)
            })
            .unwrap();

        assert!(simulation.next_intrinsic_spike(NeuronId(1)).is_some());
        let report = simulation.run_until(SimTime(5_000)).unwrap();
        assert!(report.spikes_emitted >= 1);
    }

    #[test]
    fn failed_neuron_mutation_rolls_back_cell_and_booking_state() {
        let mut simulation = simulation(one_neuron_network(), NoPlasticity);
        let before = simulation.network().neuron(NeuronId(1)).unwrap().clone();
        let before_events = simulation.pending_event_count();

        let result = simulation.update_neuron(NeuronId(1), |neuron| {
            neuron.set_intrinsic_current(1_000_000.0)?;
            Err(NeuronError::InvalidHomeostaticValue {
                field: "test rollback",
                value: f32::NAN,
            })
        });

        assert!(matches!(result, Err(SimulationError::Neuron(_))));
        assert_eq!(simulation.network().neuron(NeuronId(1)), Some(&before));
        assert_eq!(simulation.pending_event_count(), before_events);
        assert_eq!(simulation.next_intrinsic_spike(NeuronId(1)), None);
    }

    #[test]
    fn failed_homeostasis_enable_rolls_back_controller_and_cell_anchors() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: false,
            update_interval_us: 10,
            ..HomeostasisConfig::default()
        });
        let mut simulation = simulation(one_neuron_network(), NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(u64::MAX), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(u64::MAX)).unwrap();
        let before = simulation.network().neuron(NeuronId(1)).unwrap().clone();

        assert!(matches!(
            simulation.set_homeostasis_enabled(true),
            Err(SimulationError::HomeostasisTimeOverflow { .. })
        ));
        assert!(!simulation.homeostasis_enabled());
        assert_eq!(simulation.network().neuron(NeuronId(1)), Some(&before));
        assert_eq!(simulation.next_homeostasis_update(NeuronId(1)), None);
    }

    #[test]
    fn poisoned_simulation_rejects_every_further_operation() {
        // Schedule an event that will fail fatally during batch processing:
        // the batch targets a neuron whose local state rejects the batch's
        // timestamp, while the scheduler itself accepts the booking.
        let mut oversize =
            Simulation::new(one_neuron_network(), NoPlasticity, runtime_config(1), 1.0).unwrap();
        oversize
            .schedule_external_input(SimTime(1), NeuronId(1), 0.1)
            .unwrap();
        oversize
            .schedule_external_input(SimTime(1), NeuronId(1), 0.2)
            .unwrap();

        // A scheduler batch-limit failure is a pre-mutation safety stop and
        // leaves the runtime usable.
        assert!(matches!(
            oversize.step(),
            Err(SimulationError::Scheduler(_))
        ));
        assert!(!oversize.is_poisoned());

        // A neuron-integration failure poisons the runtime instead.
        // Build a network whose neuron rejects time evolution by starting
        // already advanced; supplying an event earlier than the neuron's
        // last-update anchor makes integration fail fatally.
        let mut network = one_neuron_network();
        network
            .neuron_mut(NeuronId(1))
            .unwrap()
            .advance_to(SimTime(100))
            .unwrap();
        let mut poisoned =
            Simulation::new(network, NoPlasticity, runtime_config(100), 1.0).unwrap();
        // The scheduler accepts the timestamp, but advancing the neuron to a
        // time before its anchor fails inside the batch.
        poisoned
            .scheduler
            .schedule(
                SimTime(50),
                EventKind::Homeostasis {
                    neuron_id: NeuronId(1),
                },
            )
            .unwrap();

        let result = poisoned.step();
        assert!(result.is_err());
        assert!(poisoned.is_poisoned());

        // Every further use is rejected with the original reason.
        assert!(matches!(
            poisoned.step(),
            Err(SimulationError::SimulationPoisoned { .. })
        ));
        assert!(matches!(
            poisoned.run_until(SimTime(1_000)),
            Err(SimulationError::SimulationPoisoned { .. })
        ));
        assert!(matches!(
            poisoned.schedule_external_input(SimTime(1_000), NeuronId(1), 1.0),
            Err(SimulationError::SimulationPoisoned { .. })
        ));
        assert!(matches!(
            poisoned.add_neuron(neuron(3, Polarity::Excitatory)),
            Err(SimulationError::SimulationPoisoned { .. })
        ));
        assert!(matches!(
            poisoned.update_neuron(NeuronId(1), |_| Ok(())),
            Err(SimulationError::SimulationPoisoned { .. })
        ));
        assert!(poisoned.poison_reason().is_some());
    }

    #[test]
    fn maintenance_uses_neuron_local_input_and_activity_after_same_time_spike() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 1_000,
            target_rate_hz: 0.0,
            target_input_rate: 0.0,
            intrinsic_adjustment_rate: 1.0,
            structural_adjustment_rate: 0.0,
            min_intrinsic_current: -2.0,
            max_intrinsic_current: 2.0,
            min_structural_drive: 0.0,
            max_structural_drive: 2.0,
        });
        let mut simulation = simulation(one_neuron_network(), NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        simulation
            .schedule_external_input(SimTime(1_000), NeuronId(1), 1.0)
            .unwrap();
        simulation
            .schedule_homeostasis(SimTime(1_000), NeuronId(1))
            .unwrap();

        simulation.run_until(SimTime(1_000)).unwrap();

        assert!(
            simulation
                .network()
                .neuron(NeuronId(1))
                .unwrap()
                .intrinsic_current()
                < 0.0
        );
        assert!(simulation.event_log().iter().any(|event| matches!(
            event,
            ObservationEvent::HomeostasisChanged {
                neuron_id: NeuronId(1),
                ..
            }
        )));
    }

    #[test]
    fn local_homeostasis_clock_maintains_a_silent_neuron_without_global_tick() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 10,
            target_rate_hz: 1.0,
            target_input_rate: 1.0,
            intrinsic_adjustment_rate: 1.0,
            structural_adjustment_rate: 1.0,
            min_intrinsic_current: -10.0,
            max_intrinsic_current: 10.0,
            min_structural_drive: 0.0,
            max_structural_drive: 10.0,
        });
        let mut simulation = simulation(one_neuron_network(), NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        let first = simulation
            .next_homeostasis_update(NeuronId(1))
            .expect("enabled controller starts a local clock");
        assert!((1..=10).contains(&first.as_micros()));
        assert_eq!(simulation.run(), Err(SimulationError::RunRequiresDeadline));

        let report = simulation
            .run_until(first.checked_add_us(10).unwrap())
            .unwrap();
        let neuron = simulation.network().neuron(NeuronId(1)).unwrap();
        assert_eq!(report.spikes_emitted, 0);
        assert!(neuron.structural_drive() > 0.0);
        assert_eq!(
            simulation.next_homeostasis_update(NeuronId(1)),
            first.checked_add_us(20)
        );
        assert_eq!(
            simulation
                .event_log()
                .iter()
                .filter(|event| matches!(event, ObservationEvent::HomeostasisChanged { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn aggregate_configuration_starts_local_clocks_automatically() {
        let mut config = NervaConfig::default();
        config.network.excitatory_neurons = 1;
        config.network.inhibitory_neurons = 0;
        config.neuron = neuron_params();
        config.learning.homeostasis.enabled = true;
        config.learning.homeostasis.update_interval_us = 10;

        let simulation = Simulation::from_config(one_neuron_network(), NoPlasticity, &config)
            .expect("validated configuration starts runtime");

        assert!(simulation.next_homeostasis_update(NeuronId(1)).is_some());
    }

    #[test]
    fn intrinsic_current_emits_a_predicted_deterministic_spike_without_input() {
        let mut network = one_neuron_network();
        network
            .neuron_mut(NeuronId(1))
            .unwrap()
            .set_intrinsic_current(2_000.0)
            .unwrap();
        let mut simulation = simulation(network, NoPlasticity);

        assert_eq!(
            simulation.run_until(SimTime(693)).unwrap().spikes_emitted,
            0
        );
        assert_eq!(
            simulation.run_until(SimTime(694)).unwrap().spikes_emitted,
            1
        );
        assert!(simulation.event_log().iter().any(|event| matches!(
            event,
            ObservationEvent::SpikeEmitted(spike) if spike.neuron_id == NeuronId(1) && spike.time == SimTime(694)
        )));
    }

    #[test]
    fn superseded_intrinsic_prediction_is_cancelled_eagerly() {
        let mut network = one_neuron_network();
        network
            .neuron_mut(NeuronId(1))
            .unwrap()
            .set_intrinsic_current(2_000.0)
            .unwrap();
        let mut simulation = simulation(network, NoPlasticity);
        assert_eq!(simulation.pending_event_count(), 1);

        simulation
            .schedule_external_input(SimTime(100), NeuronId(1), -0.5)
            .unwrap();
        simulation.run_until(SimTime(100)).unwrap();

        // The original crossing at 694 us was removed, rather than retained
        // as an inert stale event alongside the replacement prediction.
        assert_eq!(simulation.pending_event_count(), 1);
        assert!(
            simulation
                .next_intrinsic_spike(NeuronId(1))
                .is_some_and(|time| time > SimTime(694))
        );
    }

    #[test]
    fn enabling_homeostasis_starts_and_disabling_stops_local_clocks() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: false,
            update_interval_us: 100,
            target_rate_hz: 1.0,
            target_input_rate: 1.0,
            intrinsic_adjustment_rate: 1.0,
            structural_adjustment_rate: 1.0,
            min_intrinsic_current: -10.0,
            max_intrinsic_current: 10.0,
            min_structural_drive: 0.0,
            max_structural_drive: 10.0,
        });
        let mut simulation = simulation(one_neuron_network(), NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        assert_eq!(simulation.next_homeostasis_update(NeuronId(1)), None);

        simulation.set_homeostasis_enabled(true).unwrap();
        let first = simulation
            .next_homeostasis_update(NeuronId(1))
            .expect("enabling starts the local clock");
        simulation.set_homeostasis_enabled(false).unwrap();
        simulation.run_until(first).unwrap();

        assert_eq!(simulation.next_homeostasis_update(NeuronId(1)), None);
        assert_eq!(simulation.pending_event_count(), 0);

        simulation.set_homeostasis_enabled(true).unwrap();
        assert!(simulation.next_homeostasis_update(NeuronId(1)).is_some());
    }

    #[test]
    fn reenabled_homeostasis_does_not_integrate_the_disabled_interval() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 10,
            target_rate_hz: 1.0,
            target_input_rate: 1.0,
            intrinsic_adjustment_rate: 1.0,
            structural_adjustment_rate: 1.0,
            min_intrinsic_current: -10.0,
            max_intrinsic_current: 10.0,
            min_structural_drive: 0.0,
            max_structural_drive: 10.0,
        });
        let mut simulation = simulation(one_neuron_network(), NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        let first = simulation.next_homeostasis_update(NeuronId(1)).unwrap();
        simulation.run_until(first).unwrap();
        let drive_before_pause = simulation
            .network()
            .neuron(NeuronId(1))
            .unwrap()
            .structural_drive();

        simulation.set_homeostasis_enabled(false).unwrap();
        simulation
            .schedule_external_input(SimTime(100_000), NeuronId(1), 0.0)
            .unwrap();
        simulation.run_until(SimTime(100_000)).unwrap();

        simulation.set_homeostasis_enabled(true).unwrap();
        let first_after_reenable = simulation.next_homeostasis_update(NeuronId(1)).unwrap();
        simulation.run_until(first_after_reenable).unwrap();
        let drive_after_reenable = simulation
            .network()
            .neuron(NeuronId(1))
            .unwrap()
            .structural_drive();

        // Only the freshly enabled local phase (at most 10 us), not the
        // nearly 100 ms disabled interval, contributes to this update.
        assert!(drive_after_reenable - drive_before_pause < 0.000_02);
    }

    #[test]
    fn many_neuron_clocks_have_deterministic_spread_without_a_global_phase() {
        let mut network = Network::new();
        for id in 1..=1_000 {
            network
                .add_neuron(neuron(id, Polarity::Excitatory))
                .unwrap();
        }
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 10_000,
            target_rate_hz: 1.0,
            target_input_rate: 1.0,
            intrinsic_adjustment_rate: 1.0,
            structural_adjustment_rate: 1.0,
            min_intrinsic_current: -10.0,
            max_intrinsic_current: 10.0,
            min_structural_drive: 0.0,
            max_structural_drive: 10.0,
        });
        let simulation = simulation(network, NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        let phases: BTreeSet<_> = simulation
            .network()
            .neuron_ids()
            .map(|neuron_id| simulation.next_homeostasis_update(neuron_id).unwrap())
            .collect();

        assert_eq!(simulation.pending_event_count(), 1_000);
        assert!(phases.len() > 900, "hash phases unexpectedly collided");
        assert!(phases.first().unwrap().as_micros() < 100);
        assert!(phases.last().unwrap().as_micros() > 9_900);
    }

    #[test]
    fn multi_minute_local_homeostasis_stays_bounded_across_input_regimes() {
        let params = NeuronConfig {
            resting_potential: 0.0,
            reset_potential: 0.0,
            threshold: 1.0,
            membrane_tau_us: 1_000.0,
            refractory_period_us: 1,
            // A one-second trace makes the 10 Hz long-run target observable
            // despite the two deliberately different input temporal patterns.
            activity_trace_tau_us: 1_000_000.0,
            intrinsic: Default::default(),
        };
        let mut network = Network::new();
        for id in 1..=3 {
            network
                .add_neuron(
                    Neuron::new(
                        NeuronId(id),
                        Position3D::ORIGIN,
                        Polarity::Excitatory,
                        None,
                        params,
                        SimTime::ZERO,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 100_000,
            target_rate_hz: 10.0,
            target_input_rate: 1.0,
            intrinsic_adjustment_rate: 0.01,
            structural_adjustment_rate: 0.001,
            min_intrinsic_current: -1_000.0,
            max_intrinsic_current: 1_000.0,
            min_structural_drive: 0.0,
            max_structural_drive: 10.0,
        });
        let mut simulation = simulation(network, NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        const HORIZON_US: u64 = 180_000_000;

        // Neuron 1 gets regular 10 Hz drive. Neuron 3 receives the same mean
        // rate in short bursts; neuron 2 remains isolated and must request
        // structure without manufacturing output through intrinsic current.
        for time in (0..=HORIZON_US).step_by(100_000) {
            simulation
                .schedule_external_input(SimTime(time), NeuronId(1), 1.1)
                .unwrap();
        }
        for burst_start in (0..=HORIZON_US).step_by(500_000) {
            for offset in [0, 20_000, 40_000, 60_000, 80_000] {
                let time = burst_start + offset;
                if time <= HORIZON_US {
                    simulation
                        .schedule_external_input(SimTime(time), NeuronId(3), 1.1)
                        .unwrap();
                }
            }
        }

        simulation.run_until(SimTime(HORIZON_US)).unwrap();
        let regular = simulation.network().neuron(NeuronId(1)).unwrap();
        let isolated = simulation.network().neuron(NeuronId(2)).unwrap();
        let bursty = simulation.network().neuron(NeuronId(3)).unwrap();

        assert!(regular.intrinsic_current().abs() < 100.0);
        assert!(bursty.intrinsic_current().abs() < 100.0);
        assert_eq!(isolated.intrinsic_current(), 0.0);
        assert!(
            (0.1..0.2).contains(&isolated.structural_drive()),
            "isolated structural drive should remain below its clamp: {}",
            isolated.structural_drive()
        );
        assert!(simulation.event_log().iter().any(|event| matches!(
            event,
            ObservationEvent::SpikeEmitted(spike) if spike.neuron_id == NeuronId(1)
        )));
        assert!(simulation.event_log().iter().any(|event| matches!(
            event,
            ObservationEvent::SpikeEmitted(spike) if spike.neuron_id == NeuronId(3)
        )));
    }

    #[test]
    fn sustained_input_raises_current_then_autonomous_spikes_apply_negative_feedback() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 1_000,
            target_rate_hz: 90.0,
            target_input_rate: 1.0,
            intrinsic_adjustment_rate: 5_000.0,
            structural_adjustment_rate: 0.0,
            min_intrinsic_current: 0.0,
            max_intrinsic_current: 2_000.0,
            min_structural_drive: 0.0,
            max_structural_drive: 10.0,
        });
        let mut simulation = simulation(one_neuron_network(), NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        for time in (0..=100_000).step_by(1_300) {
            simulation
                .schedule_external_input(SimTime(time), NeuronId(1), 0.1)
                .unwrap();
        }

        simulation.run_until(SimTime(100_000)).unwrap();
        let events = simulation.event_log();
        assert!(events.iter().any(|event| matches!(
            event,
            ObservationEvent::HomeostasisChanged {
                old_intrinsic_current,
                new_intrinsic_current,
                ..
            } if new_intrinsic_current > old_intrinsic_current
        )));
        let spike_times: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ObservationEvent::SpikeEmitted(spike) => Some(spike.time),
                _ => None,
            })
            .collect();
        assert!(
            spike_times.iter().any(|time| time.as_micros() % 1_300 != 0),
            "{spike_times:?}"
        );
        assert!(events.iter().any(|event| matches!(
            event,
            ObservationEvent::HomeostasisChanged {
                firing_avg,
                old_intrinsic_current,
                new_intrinsic_current,
                ..
            } if (firing_avg - 90.0).abs() <= 5.0
                && new_intrinsic_current < old_intrinsic_current
        )));
        let current = simulation
            .network()
            .neuron(NeuronId(1))
            .unwrap()
            .intrinsic_current();
        assert!(
            (0.0..2_000.0).contains(&current),
            "feedback must settle below its safety clamp, got {current}"
        );
    }

    #[test]
    fn isolated_cell_accumulates_structural_drive_without_artificial_spikes_over_time() {
        let homeostasis = LocalHomeostasis::new(&HomeostasisConfig {
            enabled: true,
            update_interval_us: 1_000,
            target_rate_hz: 20.0,
            target_input_rate: 1.0,
            intrinsic_adjustment_rate: 5_000.0,
            structural_adjustment_rate: 1.0,
            min_intrinsic_current: 0.0,
            max_intrinsic_current: 2_000.0,
            min_structural_drive: 0.0,
            max_structural_drive: 10.0,
        });
        let mut simulation = simulation(one_neuron_network(), NoPlasticity)
            .with_homeostasis(homeostasis)
            .unwrap();
        let report = simulation.run_until(SimTime(100_000)).unwrap();
        let neuron = simulation.network().neuron(NeuronId(1)).unwrap();

        assert_eq!(report.spikes_emitted, 0);
        assert_eq!(neuron.intrinsic_current(), 0.0);
        assert!(neuron.structural_drive() > 0.09);
    }

    #[test]
    fn homeostasis_population_excludes_clocks_and_one_shot_maintenance() {
        let mut network = one_neuron_network();
        network.add_neuron(neuron(2, Polarity::Excitatory)).unwrap();
        let mut runtime = simulation(network, NoPlasticity)
            .with_homeostasis(LocalHomeostasis::new(&HomeostasisConfig {
                enabled: true,
                update_interval_us: 10_000,
                target_input_rate: 50.0,
                structural_adjustment_rate: 1.0,
                max_structural_drive: 100.0,
                ..HomeostasisConfig::default()
            }))
            .unwrap();
        runtime
            .set_homeostasis_population(Some(BTreeSet::from([NeuronId(1)])))
            .unwrap();
        assert!(runtime.next_homeostasis_update(NeuronId(1)).is_some());
        assert!(runtime.next_homeostasis_update(NeuronId(2)).is_none());
        runtime
            .schedule_homeostasis(SimTime(50_000), NeuronId(2))
            .unwrap();
        runtime
            .schedule_external_input(SimTime(100_000), NeuronId(2), 0.0)
            .unwrap();
        runtime.run_until(SimTime(100_000)).unwrap();
        assert_eq!(
            runtime
                .network()
                .neuron(NeuronId(2))
                .unwrap()
                .structural_drive(),
            0.0
        );
        assert!(
            runtime
                .network()
                .neuron(NeuronId(1))
                .unwrap()
                .structural_drive()
                > 0.0
        );
        assert!(runtime.event_log().iter().all(|event| !matches!(
            event,
            ObservationEvent::HomeostasisChanged {
                neuron_id: NeuronId(2),
                ..
            }
        )));
        runtime
            .set_homeostasis_population(Some(BTreeSet::from([NeuronId(2)])))
            .unwrap();
        let first = runtime.next_homeostasis_update(NeuronId(2)).unwrap();
        runtime.run_until(first).unwrap();
        assert!(
            runtime
                .network()
                .neuron(NeuronId(2))
                .unwrap()
                .structural_drive()
                <= 0.5001
        );
        assert!(runtime.next_homeostasis_update(NeuronId(1)).is_none());
        let before = runtime.network().clone();
        let booking = runtime.next_homeostasis_update(NeuronId(2));
        assert!(
            runtime
                .set_homeostasis_population(Some(BTreeSet::from([NeuronId(999)])))
                .is_err()
        );
        assert_eq!(runtime.network(), &before);
        assert_eq!(runtime.next_homeostasis_update(NeuronId(2)), booking);
        runtime.add_neuron(neuron(3, Polarity::Excitatory)).unwrap();
        assert!(runtime.next_homeostasis_update(NeuronId(3)).is_none());
        runtime.set_homeostasis_population(None).unwrap();
        assert!(runtime.next_homeostasis_update(NeuronId(3)).is_some());
        runtime
            .set_homeostasis_population(Some(BTreeSet::new()))
            .unwrap();
        assert!((1..=3).all(|id| runtime.next_homeostasis_update(NeuronId(id)).is_none()));
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
