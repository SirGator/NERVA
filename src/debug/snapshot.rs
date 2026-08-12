//! Neutral copied state suitable for replay diagnostics and visualization.

use crate::{
    core::{Network, NeuronId, SimTime, SynapseId},
    math::Position3D,
};

/// Copied neuron state with no mutation path back to the simulation.
#[derive(Clone, Debug, PartialEq)]
pub struct NeuronSnapshot {
    /// Stable identity.
    pub id: NeuronId,
    /// Geometric location.
    pub position: Position3D,
    /// Current membrane potential.
    pub membrane_potential: f32,
    /// Current local threshold.
    pub threshold: f32,
    /// Total emitted spikes.
    pub spike_count: u64,
}

/// Copied synaptic state.
#[derive(Clone, Debug, PartialEq)]
pub struct SynapseSnapshot {
    /// Stable identity.
    pub id: SynapseId,
    /// Presynaptic neuron.
    pub pre: NeuronId,
    /// Postsynaptic neuron.
    pub post: NeuronId,
    /// Non-negative magnitude.
    pub weight: f32,
    /// Whether transmission is active.
    pub enabled: bool,
}

/// Complete copied network state at one exact timestamp.
#[derive(Clone, Debug, PartialEq)]
pub struct NetworkSnapshot {
    /// Snapshot timestamp.
    pub at: SimTime,
    /// Neurons in stable ID order.
    pub neurons: Vec<NeuronSnapshot>,
    /// Synapses in stable ID order.
    pub synapses: Vec<SynapseSnapshot>,
}

impl NetworkSnapshot {
    /// Copies the observable state of a network at an exact timestamp.
    pub fn capture(at: SimTime, network: &Network) -> Self {
        let neurons = network
            .neurons()
            .map(|neuron| NeuronSnapshot {
                id: neuron.id(),
                position: neuron.position(),
                membrane_potential: neuron.membrane_potential(),
                threshold: neuron.threshold(),
                spike_count: neuron.spike_count(),
            })
            .collect();
        let synapses = network
            .synapses()
            .map(|synapse| SynapseSnapshot {
                id: synapse.id(),
                pre: synapse.pre(),
                post: synapse.post(),
                weight: synapse.weight(),
                enabled: synapse.is_enabled(),
            })
            .collect();

        Self::new(at, neurons, synapses)
    }

    /// Constructs a neutral snapshot and stabilizes its ordering.
    pub fn new(
        at: SimTime,
        mut neurons: Vec<NeuronSnapshot>,
        mut synapses: Vec<SynapseSnapshot>,
    ) -> Self {
        neurons.sort_by_key(|neuron| neuron.id);
        synapses.sort_by_key(|synapse| synapse.id);
        Self {
            at,
            neurons,
            synapses,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        config::NeuronConfig,
        core::{Neuron, NeuronRole, Polarity, Synapse},
    };

    use super::*;

    fn neuron(id: u64, x: f32) -> Neuron {
        Neuron::new(
            NeuronId(id),
            Position3D::new(x, 0.0, 0.0),
            Polarity::Excitatory,
            Some(NeuronRole::Processing),
            NeuronConfig::default(),
            SimTime::ZERO,
        )
        .unwrap()
    }

    #[test]
    fn capture_copies_current_network_state_in_stable_id_order() {
        let mut network = Network::new();
        network.add_neuron(neuron(2, 2.0)).unwrap();
        network.add_neuron(neuron(1, 1.0)).unwrap();
        network
            .add_synapse(
                Synapse::new(SynapseId(7), NeuronId(2), NeuronId(1), 0.4, 10, true).unwrap(),
            )
            .unwrap();
        network
            .synapse_mut(SynapseId(7))
            .unwrap()
            .set_enabled(false);

        let snapshot = NetworkSnapshot::capture(SimTime(99), &network);

        assert_eq!(snapshot.at, SimTime(99));
        assert_eq!(
            snapshot
                .neurons
                .iter()
                .map(|neuron| neuron.id)
                .collect::<Vec<_>>(),
            [NeuronId(1), NeuronId(2)]
        );
        assert_eq!(snapshot.neurons[0].position.x, 1.0);
        assert_eq!(snapshot.synapses[0].id, SynapseId(7));
        assert_eq!(snapshot.synapses[0].weight, 0.4);
        assert!(!snapshot.synapses[0].enabled);
    }

    #[test]
    fn constructor_stabilizes_caller_supplied_order() {
        let snapshot = NetworkSnapshot::new(
            SimTime::ZERO,
            vec![
                NeuronSnapshot {
                    id: NeuronId(2),
                    position: Position3D::ORIGIN,
                    membrane_potential: 0.0,
                    threshold: 1.0,
                    spike_count: 0,
                },
                NeuronSnapshot {
                    id: NeuronId(1),
                    position: Position3D::ORIGIN,
                    membrane_potential: 0.0,
                    threshold: 1.0,
                    spike_count: 0,
                },
            ],
            Vec::new(),
        );

        assert_eq!(snapshot.neurons[0].id, NeuronId(1));
        assert_eq!(snapshot.neurons[1].id, NeuronId(2));
    }
}
