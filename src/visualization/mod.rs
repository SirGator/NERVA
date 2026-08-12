//! Deterministic, read-only exports for external visualization tools.
//!
//! This slice deliberately returns neutral text/data records. It neither owns
//! a renderer nor has a mutation path back into the simulated network.

mod export;

pub use export::{
    ConnectionRecord, CsvExport, PositionRecord, SpikeTimeRecord, VisualizationExport,
    WeightRecord, connections_csv, export_csv, export_json, positions_csv, spike_times_csv,
    weights_csv,
};
