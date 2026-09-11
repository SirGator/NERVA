//! Neutral CSV and JSON serialization without renderer-specific assumptions.

use std::{borrow::Borrow, fmt::Write as _};

use crate::{
    core::{Network, NeuronId, SimTime, Spike, SynapseId},
    math::Position3D,
};

/// One neuron's stable identity and geometric position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionRecord {
    /// Stable neuron identity.
    pub neuron_id: NeuronId,
    /// Geometric position; it is not used as an identity.
    pub position: Position3D,
}

/// One directed edge and its non-weight transmission properties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConnectionRecord {
    /// Stable connection identity.
    pub synapse_id: SynapseId,
    /// Presynaptic endpoint.
    pub pre: NeuronId,
    /// Postsynaptic endpoint.
    pub post: NeuronId,
    /// Propagation delay in integer microseconds.
    pub delay_us: u64,
    /// Whether a local learning rule may update the connection.
    pub plastic: bool,
    /// Whether the connection currently transmits spikes.
    pub enabled: bool,
}

/// One current, non-negative synaptic weight magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightRecord {
    /// Stable connection identity.
    pub synapse_id: SynapseId,
    /// Current non-negative magnitude. Its sign comes from the source neuron.
    pub weight: f32,
}

/// One exact neuron spike timestamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpikeTimeRecord {
    /// Emitting neuron.
    pub neuron_id: NeuronId,
    /// Exact emission time.
    pub time: SimTime,
}

/// Copied visualization data with canonical ordering and no network access.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VisualizationExport {
    positions: Vec<PositionRecord>,
    connections: Vec<ConnectionRecord>,
    weights: Vec<WeightRecord>,
    spike_times: Vec<SpikeTimeRecord>,
}

impl VisualizationExport {
    /// Copies a network and spike collection into deterministic export order.
    ///
    /// Network records are ordered by their stable IDs. Spikes are ordered by
    /// `(time, neuron_id)`, making the text output independent of collection
    /// iteration order while retaining duplicate spike records.
    pub fn capture<I, S>(network: &Network, spikes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Borrow<Spike>,
    {
        let mut positions: Vec<_> = network
            .neurons()
            .map(|neuron| PositionRecord {
                neuron_id: neuron.id(),
                position: neuron.position(),
            })
            .collect();
        let mut connections: Vec<_> = network
            .synapses()
            .map(|synapse| ConnectionRecord {
                synapse_id: synapse.id(),
                pre: synapse.pre(),
                post: synapse.post(),
                delay_us: synapse.delay_us(),
                plastic: synapse.is_plastic(),
                enabled: synapse.is_enabled(),
            })
            .collect();
        let mut weights: Vec<_> = network
            .synapses()
            .map(|synapse| WeightRecord {
                synapse_id: synapse.id(),
                weight: synapse.weight().get(),
            })
            .collect();
        let mut spike_times: Vec<_> = spikes
            .into_iter()
            .map(|spike| {
                let spike = spike.borrow();
                SpikeTimeRecord {
                    neuron_id: spike.neuron_id,
                    time: spike.time,
                }
            })
            .collect();

        // Network currently supplies ordered iterators, but sorting here keeps
        // the export contract local if the graph's backing store ever changes.
        positions.sort_by_key(|record| record.neuron_id);
        connections.sort_by_key(|record| record.synapse_id);
        weights.sort_by_key(|record| record.synapse_id);
        spike_times.sort_by_key(|record| (record.time, record.neuron_id));

        Self {
            positions,
            connections,
            weights,
            spike_times,
        }
    }

    /// Position records in ascending neuron-ID order.
    pub fn positions(&self) -> &[PositionRecord] {
        &self.positions
    }

    /// Connection records in ascending synapse-ID order.
    pub fn connections(&self) -> &[ConnectionRecord] {
        &self.connections
    }

    /// Weight records in ascending synapse-ID order.
    pub fn weights(&self) -> &[WeightRecord] {
        &self.weights
    }

    /// Spike records in ascending `(time, neuron_id)` order.
    pub fn spike_times(&self) -> &[SpikeTimeRecord] {
        &self.spike_times
    }

    /// Serializes all four tables into separate CSV documents.
    pub fn to_csv(&self) -> CsvExport {
        CsvExport {
            positions: positions_to_csv(&self.positions),
            connections: connections_to_csv(&self.connections),
            weights: weights_to_csv(&self.weights),
            spike_times: spike_times_to_csv(&self.spike_times),
        }
    }

    /// Serializes all four datasets into one deterministic JSON object.
    pub fn to_json(&self) -> String {
        let mut output = String::from("{\"positions\":[");
        for (index, record) in self.positions.iter().enumerate() {
            separate_json_item(&mut output, index);
            write!(
                output,
                "{{\"neuron_id\":{},\"x\":{},\"y\":{},\"z\":{}}}",
                record.neuron_id.get(),
                json_f32(record.position.x),
                json_f32(record.position.y),
                json_f32(record.position.z),
            )
            .expect("writing to String cannot fail");
        }

        output.push_str("],\"connections\":[");
        for (index, record) in self.connections.iter().enumerate() {
            separate_json_item(&mut output, index);
            write!(
                output,
                "{{\"synapse_id\":{},\"pre\":{},\"post\":{},\"delay_us\":{},\"plastic\":{},\"enabled\":{}}}",
                record.synapse_id.get(),
                record.pre.get(),
                record.post.get(),
                record.delay_us,
                record.plastic,
                record.enabled,
            )
            .expect("writing to String cannot fail");
        }

        output.push_str("],\"weights\":[");
        for (index, record) in self.weights.iter().enumerate() {
            separate_json_item(&mut output, index);
            write!(
                output,
                "{{\"synapse_id\":{},\"weight\":{}}}",
                record.synapse_id.get(),
                json_f32(record.weight),
            )
            .expect("writing to String cannot fail");
        }

        output.push_str("],\"spike_times\":[");
        for (index, record) in self.spike_times.iter().enumerate() {
            separate_json_item(&mut output, index);
            write!(
                output,
                "{{\"neuron_id\":{},\"time_us\":{}}}",
                record.neuron_id.get(),
                record.time.as_micros(),
            )
            .expect("writing to String cannot fail");
        }
        output.push_str("]}");
        output
    }
}

/// The four schema-specific CSV documents produced from one capture.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CsvExport {
    /// `neuron_id,x,y,z` rows.
    pub positions: String,
    /// Topology and transmission-property rows.
    pub connections: String,
    /// `synapse_id,weight` rows.
    pub weights: String,
    /// `neuron_id,time_us` rows.
    pub spike_times: String,
}

/// Captures all datasets and returns four schema-specific CSV documents.
pub fn export_csv<I, S>(network: &Network, spikes: I) -> CsvExport
where
    I: IntoIterator<Item = S>,
    S: Borrow<Spike>,
{
    VisualizationExport::capture(network, spikes).to_csv()
}

/// Captures all datasets and returns one deterministic JSON document.
pub fn export_json<I, S>(network: &Network, spikes: I) -> String
where
    I: IntoIterator<Item = S>,
    S: Borrow<Spike>,
{
    VisualizationExport::capture(network, spikes).to_json()
}

/// Exports network positions in ascending neuron-ID order.
pub fn positions_csv(network: &Network) -> String {
    let records = VisualizationExport::capture(network, std::iter::empty::<Spike>());
    positions_to_csv(records.positions())
}

/// Exports network connections in ascending synapse-ID order.
pub fn connections_csv(network: &Network) -> String {
    let records = VisualizationExport::capture(network, std::iter::empty::<Spike>());
    connections_to_csv(records.connections())
}

/// Exports current weights in ascending synapse-ID order.
pub fn weights_csv(network: &Network) -> String {
    let records = VisualizationExport::capture(network, std::iter::empty::<Spike>());
    weights_to_csv(records.weights())
}

/// Exports spikes in ascending `(time, neuron_id)` order.
pub fn spike_times_csv<I, S>(spikes: I) -> String
where
    I: IntoIterator<Item = S>,
    S: Borrow<Spike>,
{
    let empty = Network::new();
    let records = VisualizationExport::capture(&empty, spikes);
    spike_times_to_csv(records.spike_times())
}

fn positions_to_csv(records: &[PositionRecord]) -> String {
    let mut output = String::from("neuron_id,x,y,z\n");
    for record in records {
        writeln!(
            output,
            "{},{},{},{}",
            record.neuron_id.get(),
            record.position.x,
            record.position.y,
            record.position.z,
        )
        .expect("writing to String cannot fail");
    }
    output
}

fn connections_to_csv(records: &[ConnectionRecord]) -> String {
    let mut output = String::from("synapse_id,pre,post,delay_us,plastic,enabled\n");
    for record in records {
        writeln!(
            output,
            "{},{},{},{},{},{}",
            record.synapse_id.get(),
            record.pre.get(),
            record.post.get(),
            record.delay_us,
            record.plastic,
            record.enabled,
        )
        .expect("writing to String cannot fail");
    }
    output
}

fn weights_to_csv(records: &[WeightRecord]) -> String {
    let mut output = String::from("synapse_id,weight\n");
    for record in records {
        writeln!(output, "{},{}", record.synapse_id.get(), record.weight,)
            .expect("writing to String cannot fail");
    }
    output
}

fn spike_times_to_csv(records: &[SpikeTimeRecord]) -> String {
    let mut output = String::from("neuron_id,time_us\n");
    for record in records {
        writeln!(
            output,
            "{},{}",
            record.neuron_id.get(),
            record.time.as_micros(),
        )
        .expect("writing to String cannot fail");
    }
    output
}

fn separate_json_item(output: &mut String, index: usize) {
    if index != 0 {
        output.push(',');
    }
}

fn json_f32(value: f32) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        // Core constructors reject non-finite positions and weights. Keeping
        // this defensive fallback nevertheless guarantees syntactically valid
        // JSON if a public synapse field was modified without revalidation.
        String::from("null")
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        config::NeuronConfig,
        core::{Neuron, NeuronRole, Polarity, Synapse},
        primitives::Weight,
    };

    use super::*;

    fn network() -> Network {
        let mut network = Network::new();
        for (id, position) in [
            (2, Position3D::new(1.0, 2.0, 3.0)),
            (1, Position3D::new(-1.0, 0.5, 0.0)),
        ] {
            network
                .add_neuron(
                    Neuron::new(
                        NeuronId(id),
                        position,
                        Polarity::Excitatory,
                        Some(NeuronRole::Processing),
                        NeuronConfig::default(),
                        SimTime::ZERO,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        network
            .add_synapse(
                Synapse::new(
                    SynapseId(9),
                    NeuronId(2),
                    NeuronId(1),
                    Weight::new(0.75).unwrap(),
                    12,
                    true,
                )
                .unwrap(),
            )
            .unwrap();
        network
    }

    #[test]
    fn csv_tables_have_neutral_schemas_and_canonical_order() {
        let network = network();
        let spikes = [
            Spike::new(NeuronId(2), SimTime(20)),
            Spike::new(NeuronId(2), SimTime(10)),
            Spike::new(NeuronId(1), SimTime(10)),
        ];

        let csv = export_csv(&network, spikes.iter());

        assert_eq!(csv.positions, "neuron_id,x,y,z\n1,-1,0.5,0\n2,1,2,3\n");
        assert_eq!(
            csv.connections,
            "synapse_id,pre,post,delay_us,plastic,enabled\n9,2,1,12,true,true\n"
        );
        assert_eq!(csv.weights, "synapse_id,weight\n9,0.75\n");
        assert_eq!(csv.spike_times, "neuron_id,time_us\n1,10\n2,10\n2,20\n");
    }

    #[test]
    fn json_contains_all_datasets_in_canonical_order() {
        let network = network();
        let spikes = [
            Spike::new(NeuronId(2), SimTime(20)),
            Spike::new(NeuronId(1), SimTime(10)),
        ];

        let json = export_json(&network, spikes);

        assert_eq!(
            json,
            "{\"positions\":[{\"neuron_id\":1,\"x\":-1,\"y\":0.5,\"z\":0},{\"neuron_id\":2,\"x\":1,\"y\":2,\"z\":3}],\"connections\":[{\"synapse_id\":9,\"pre\":2,\"post\":1,\"delay_us\":12,\"plastic\":true,\"enabled\":true}],\"weights\":[{\"synapse_id\":9,\"weight\":0.75}],\"spike_times\":[{\"neuron_id\":1,\"time_us\":10},{\"neuron_id\":2,\"time_us\":20}]}"
        );
    }

    #[test]
    fn empty_exports_retain_headers_and_json_shape() {
        let network = Network::new();
        let export = VisualizationExport::capture(&network, std::iter::empty::<Spike>());

        assert_eq!(export.to_csv().positions, "neuron_id,x,y,z\n");
        assert_eq!(
            export.to_json(),
            "{\"positions\":[],\"connections\":[],\"weights\":[],\"spike_times\":[]}"
        );
    }

    #[test]
    fn spike_only_helper_accepts_owned_and_borrowed_records() {
        let spikes = vec![
            Spike::new(NeuronId(2), SimTime(1)),
            Spike::new(NeuronId(1), SimTime(1)),
        ];

        assert_eq!(spike_times_csv(&spikes), spike_times_csv(spikes.clone()));
        assert_eq!(spike_times_csv(spikes), "neuron_id,time_us\n1,1\n2,1\n");
    }
}
