//! Embeds NERVA in a small application with two connected neurons.

use nerva::{
    config::{NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    learning::NoPlasticity,
    math::Position3D,
    primitives::Weight,
    runtime::{ObservationEvent, Simulation},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut network = Network::new();
    for (id, position) in [
        (NeuronId(1), Position3D::ORIGIN),
        (NeuronId(2), Position3D::new(1.0, 0.0, 0.0)),
    ] {
        network.add_neuron(Neuron::new(
            id,
            position,
            Polarity::Excitatory,
            None,
            NeuronConfig::default(),
            SimTime::ZERO,
        )?)?;
    }
    // Positive weight and delay: neuron 1 excites neuron 2 after 1,000 us.
    network.add_synapse(Synapse::new(
        SynapseId(1),
        NeuronId(1),
        NeuronId(2),
        Weight::new(20.0)?,
        1_000,
        false,
    )?)?;

    let mut simulation = Simulation::new(network, NoPlasticity, RuntimeConfig::default(), 10.0)?;
    simulation.schedule_external_input(SimTime::ZERO, NeuronId(1), 20.0)?;
    let report = simulation.run_until(SimTime(2_000))?;

    for event in simulation.event_log() {
        if let ObservationEvent::SpikeEmitted(spike) = event {
            println!(
                "Neuron {} spiked at {} us",
                spike.neuron_id,
                spike.time.as_micros(),
            );
        }
    }
    println!(
        "NERVA: {} spikes across {} timestamp batches",
        report.spikes_emitted, report.batches_processed,
    );
    Ok(())
}
