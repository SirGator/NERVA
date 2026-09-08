//! Pure planning of delayed, spatially attenuated synaptic arrivals.

use std::{error::Error, fmt};

use crate::{
    core::{Event, EventKind, Network, NeuronId, SimTime, Spike, SynapseError, SynapseId},
    math::{DecayError, try_distance_attenuation},
};

/// One authoritative arrival produced from an enabled outgoing synapse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedTransmission {
    /// Synapse whose diagnostics counter must be advanced once scheduling succeeds.
    pub synapse_id: SynapseId,
    /// Delayed event to insert into the scheduler.
    pub event: Event,
}

/// Why a spike could not be converted into delayed arrivals.
#[derive(Clone, Debug, PartialEq)]
pub enum PropagationError {
    /// The spike source is absent from the network.
    UnknownSourceNeuron(NeuronId),
    /// An outgoing adjacency index references no synapse.
    UnknownSynapse(SynapseId),
    /// A synapse endpoint is absent from the network.
    UnknownTargetNeuron(NeuronId),
    /// An outgoing adjacency index points at a synapse with another source.
    SourceMismatch {
        /// Emitter being propagated.
        spike_source: NeuronId,
        /// Referenced synapse.
        synapse_id: SynapseId,
        /// Source stored in that synapse.
        synapse_source: NeuronId,
    },
    /// The configured distance-decay length or computed distance is invalid.
    InvalidAttenuation(DecayError),
    /// The synapse rejected the computed attenuation.
    InvalidSynapse(SynapseError),
    /// Adding the strictly positive synaptic delay overflowed simulation time.
    ArrivalTimeOverflow {
        /// Emission timestamp.
        emitted_at: SimTime,
        /// Synaptic delay that could not be represented.
        delay_us: u64,
    },
}

impl fmt::Display for PropagationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSourceNeuron(id) => write!(formatter, "unknown spike source neuron {id}"),
            Self::UnknownSynapse(id) => write!(formatter, "unknown outgoing synapse {id}"),
            Self::UnknownTargetNeuron(id) => write!(formatter, "unknown target neuron {id}"),
            Self::SourceMismatch {
                spike_source,
                synapse_id,
                synapse_source,
            } => write!(
                formatter,
                "outgoing index for neuron {spike_source} contains synapse {synapse_id} owned by neuron {synapse_source}"
            ),
            Self::InvalidAttenuation(error) => {
                write!(formatter, "invalid spatial attenuation: {error}")
            }
            Self::InvalidSynapse(error) => {
                write!(formatter, "invalid synaptic propagation: {error}")
            }
            Self::ArrivalTimeOverflow {
                emitted_at,
                delay_us,
            } => write!(
                formatter,
                "adding delay {delay_us} us to emission time {emitted_at} overflows simulation time"
            ),
        }
    }
}

impl Error for PropagationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidAttenuation(error) => Some(error),
            Self::InvalidSynapse(error) => Some(error),
            _ => None,
        }
    }
}

/// Plans every enabled outgoing transmission in stable synapse-ID order.
///
/// The effective amplitude is captured at emission time. Subsequent learning
/// therefore affects future spikes, not an impulse already travelling through a
/// synapse.
pub fn plan_spike_propagation(
    network: &Network,
    spike: Spike,
    distance_decay_length: f32,
) -> Result<Vec<PlannedTransmission>, PropagationError> {
    let source = network
        .neuron(spike.neuron_id)
        .ok_or(PropagationError::UnknownSourceNeuron(spike.neuron_id))?;
    let source_position = source.position();
    let source_polarity = source.polarity();
    let outgoing = network.outgoing_synapse_ids(spike.neuron_id);
    let mut transmissions = Vec::with_capacity(outgoing.len());

    for &synapse_id in outgoing {
        let synapse = network
            .synapse(synapse_id)
            .ok_or(PropagationError::UnknownSynapse(synapse_id))?;
        if synapse.pre() != spike.neuron_id {
            return Err(PropagationError::SourceMismatch {
                spike_source: spike.neuron_id,
                synapse_id,
                synapse_source: synapse.pre(),
            });
        }
        if !synapse.is_enabled() {
            continue;
        }

        let target = network
            .neuron(synapse.post())
            .ok_or(PropagationError::UnknownTargetNeuron(synapse.post()))?;
        let distance = source_position.distance_to(target.position());
        let attenuation = try_distance_attenuation(distance, distance_decay_length)
            .map_err(PropagationError::InvalidAttenuation)?;
        let amplitude = synapse
            .effective_weight(source_polarity, attenuation)
            .map_err(PropagationError::InvalidSynapse)?;
        let arrives_at = spike.time.checked_add_us(synapse.delay_us()).ok_or(
            PropagationError::ArrivalTimeOverflow {
                emitted_at: spike.time,
                delay_us: synapse.delay_us(),
            },
        )?;

        transmissions.push(PlannedTransmission {
            synapse_id,
            event: Event::new(
                arrives_at,
                EventKind::SynapticArrival {
                    synapse_id,
                    target: synapse.post(),
                    amplitude,
                },
            ),
        });
    }

    Ok(transmissions)
}

#[cfg(test)]
mod tests {
    use crate::{
        config::NeuronConfig,
        core::{Network, Neuron, Polarity, Synapse},
        math::Position3D,
    };

    use super::*;

    fn params() -> NeuronConfig {
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

    fn neuron(id: u64, x: f32, polarity: Polarity) -> Neuron {
        Neuron::new(
            NeuronId(id),
            Position3D::new(x, 0.0, 0.0),
            polarity,
            None,
            params(),
            SimTime::ZERO,
        )
        .unwrap()
    }

    fn network(polarity: Polarity) -> Network {
        let mut network = Network::new();
        network.add_neuron(neuron(1, 0.0, polarity)).unwrap();
        network
            .add_neuron(neuron(2, 1.0, Polarity::Excitatory))
            .unwrap();
        network
            .add_synapse(
                Synapse::new(SynapseId(7), NeuronId(1), NeuronId(2), 2.0, 5, true).unwrap(),
            )
            .unwrap();
        network
    }

    #[test]
    fn applies_delay_distance_attenuation_and_sender_polarity() {
        for (polarity, sign) in [
            (Polarity::Excitatory, 1.0_f32),
            (Polarity::Inhibitory, -1.0_f32),
        ] {
            let transmissions = plan_spike_propagation(
                &network(polarity),
                Spike::new(NeuronId(1), SimTime(10)),
                1.0,
            )
            .unwrap();

            assert_eq!(transmissions.len(), 1);
            assert_eq!(transmissions[0].synapse_id, SynapseId(7));
            assert_eq!(transmissions[0].event.time, SimTime(15));
            let EventKind::SynapticArrival {
                synapse_id,
                target,
                amplitude,
            } = transmissions[0].event.kind
            else {
                panic!("propagation must produce a synaptic arrival")
            };
            assert_eq!(synapse_id, SynapseId(7));
            assert_eq!(target, NeuronId(2));
            assert!((amplitude - sign * 2.0 / std::f32::consts::E).abs() < 1.0e-6);
        }
    }

    #[test]
    fn disabled_synapse_does_not_propagate() {
        let mut network = network(Polarity::Excitatory);
        network
            .synapse_mut(SynapseId(7))
            .unwrap()
            .set_enabled(false);

        let transmissions =
            plan_spike_propagation(&network, Spike::new(NeuronId(1), SimTime(10)), 1.0).unwrap();

        assert!(transmissions.is_empty());
    }

    #[test]
    fn arrival_time_overflow_is_rejected_instead_of_wrapping() {
        let error = plan_spike_propagation(
            &network(Polarity::Excitatory),
            Spike::new(NeuronId(1), SimTime(u64::MAX - 1)),
            1.0,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            PropagationError::ArrivalTimeOverflow { .. }
        ));
    }
}
