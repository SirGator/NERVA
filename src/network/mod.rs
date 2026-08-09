//! Network construction and connectivity.

/// Directed network graph representation.
pub mod graph;
/// Aggregate network data model.
pub mod model;
/// Network routing behavior.
pub mod routing;
/// Network runtime coordination.
pub mod runtime;

pub use graph::NetworkGraph;
pub use model::NeuralNetwork;
