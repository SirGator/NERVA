//! One shared network with graded construction profiles and endogenous release.
//! Run: cargo run --offline --release --example neuromodulated_tissue
use nerva::{
    cell_profiles::{AreaField, electrical_profile, receptor_profile, short_term_profile},
    config::{LearningConfig, NeuronConfig, RuntimeConfig},
    core::{Network, Neuron, NeuronId, Polarity, SimTime, Synapse, SynapseId},
    learning::{PairStdp, ThreeFactorConfig},
    math::Position3D,
    neuromodulation::{ChemicalConfig, ReleaseProfile},
    primitives::Weight,
    runtime::{ObservationEvent, Simulation},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let field = AreaField::new(0., 1., 0.16)?;
    let base = NeuronConfig {
        resting_potential: 0.,
        reset_potential: 0.,
        threshold: 1.,
        ..NeuronConfig::default()
    };
    let mut network = Network::new();
    for id in 0..16 {
        let position = Position3D::new(id as f32 / 15., 0., 0.);
        let area = field.at(position)?;
        let polarity = if id % 5 == 4 {
            Polarity::Inhibitory
        } else {
            Polarity::Excitatory
        };
        let mut cell = Neuron::new(
            NeuronId(id),
            position,
            polarity,
            None,
            electrical_profile(base, area, polarity)?,
            SimTime::ZERO,
        )?;
        cell.set_receptor_profile(receptor_profile(area)?)?;
        if id == 0 {
            cell.set_release_profile(ReleaseProfile {
                dopamine: 0.001,
                acetylcholine: 0.001,
                noradrenaline: 0.001,
            })?;
        }
        network.add_neuron(cell)?;
    }
    let mut next = 0;
    for pre in 0_u64..16 {
        for post in 0_u64..16 {
            if pre == post || pre.abs_diff(post) > 3 {
                continue;
            }
            let source = network.neuron(NeuronId(pre)).unwrap();
            let target = network.neuron(NeuronId(post)).unwrap();
            let mut edge = Synapse::new(
                SynapseId(next),
                NeuronId(pre),
                NeuronId(post),
                Weight::new(0.15)?,
                1_000,
                true,
            )?;
            edge.set_short_term(short_term_profile(
                field.at(source.position())?,
                field.at(target.position())?,
                source.polarity(),
            )?)?;
            network.add_synapse(edge)?;
            next += 1;
        }
    }
    let rule =
        PairStdp::new(&LearningConfig::default()).with_three_factor(ThreeFactorConfig::default())?;
    let mut sim = Simulation::new(network, rule, RuntimeConfig::default(), 1.)?
        .with_neuromodulation(ChemicalConfig::default())?;
    for index in 0..10 {
        for id in [0, 1, 5, 6, 9, 10, 13, 14] {
            sim.schedule_external_input(
                SimTime(1_000 + index * 20_000 + (id % 2) * 5_000),
                NeuronId(id),
                1.5,
            )?;
        }
    }
    let report = sim.run_until(SimTime(500_000))?;
    let updates = sim
        .event_log()
        .iter()
        .filter(|e| matches!(e, ObservationEvent::WeightChanged { .. }))
        .count();
    println!(
        "neurons={} synapses={} spikes={} weight_updates={} local_pulses={}",
        sim.network().neuron_count(),
        sim.network().synapse_count(),
        report.spikes_emitted,
        updates,
        sim.chemical_field().unwrap().pulse_count()
    );
    println!("Mechanism demonstration only; no pattern-completion or memory claim.");
    Ok(())
}
