//! Reproducible construction and execution of NERVA experiments.

mod bit_world;
mod comparison;
pub mod m0;
mod rng;
mod runner;
mod sequence_m0;

pub use bit_world::{ClosedLoop, ClosedLoopError, ClosedLoopReport};
pub use comparison::{
    Comparison, M0Group, M0GroupResult, M0Metrics, M0StudyConfig, M0StudyReport, M0SuccessReport,
};
pub use m0::{
    Action, BitDecoder, BitEncoder, BitEncoderError, BitWorld, Decoder, EncodingError, Encoder,
    Environment, Observation, Pattern, PatternEncoder, PatternEncoderError, SequenceEnvironment,
};
pub use runner::{ExperimentError, ExperimentRecord, ExperimentRunner};
pub use sequence_m0::{M0Experiment, M0ExperimentConfig, run_m0_comparison, run_m0_study};
