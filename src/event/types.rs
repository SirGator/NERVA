//! Event payloads exchanged by the network runtime.

use crate::common::{NeuronId, SimTimeUs, SynapseId};

/// A spike emitted by one neuron at a simulation timestamp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpikeEvent {
    /// Neuron that emitted the spike.
    pub source: NeuronId,
    /// Timestamp at which the spike was emitted.
    pub emitted_at: SimTimeUs,
}

/// Input transmitted by a synapse and delivered to a target neuron.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynapticInputEvent {
    /// Synapse that generated this input.
    pub synapse: SynapseId,
    /// Presynaptic neuron that emitted the source spike.
    pub source: NeuronId,
    /// Postsynaptic neuron receiving the input.
    pub target: NeuronId,
    /// Timestamp at which the input reaches the target.
    pub arrives_at: SimTimeUs,
    /// Signed input contribution delivered by the synapse.
    pub input_value: f32,
}

/// All inputs delivered to one neuron at a single simulation timestamp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuronInputBatch {
    /// Neuron receiving the aggregated input.
    pub target: NeuronId,
    /// Shared arrival timestamp of the aggregated inputs.
    pub arrives_at: SimTimeUs,
    /// Sum of all synaptic input contributions.
    pub summed_input: f32,
    /// Noise sample to apply while integrating the batch.
    pub noise_sample: f32,
}

/// An event together with its deterministic execution position.
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledEvent {
    /// Timestamp at which the queue executes this event.
    pub execute_at: SimTimeUs,
    /// Insertion order used to break ties at equal timestamps.
    pub sequence: u64,
    /// Concrete event data.
    pub payload: EventPayload,
}

/// Data carried by a scheduled event.
#[derive(Clone, Debug, PartialEq)]
pub enum EventPayload {
    /// A delayed synaptic input that will be delivered to a neuron.
    SynapticInput(SynapticInputEvent),
}
