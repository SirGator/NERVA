//! Errors emitted while configuring or advancing a neuron.

use crate::common::SimTimeUs;

use super::{CellType, PrimaryTransmitter};

#[derive(Clone, Debug, PartialEq, Eq)]
/// A violation of the static invariants required to construct a neuron.
pub enum NeuronConfigError {
    /// A floating-point parameter is NaN or infinite.
    NonFiniteParameter(&'static str),
    /// A parameter that must be strictly positive is zero or negative.
    NonPositiveParameter(&'static str),
    /// A parameter that must be non-negative is negative.
    NegativeParameter(&'static str),
    /// The firing threshold is not higher than the resting potential.
    ThresholdNotAboveRestingPotential,
    /// The reset potential is not lower than the firing threshold.
    ResetNotBelowThreshold,
    /// The selected cell type is incompatible with its primary transmitter.
    InvalidCellTransmitter {
        /// Requested cell type.
        cell_type: CellType,
        /// Requested transmitter.
        transmitter: PrimaryTransmitter,
    },
    /// A modulatory role was selected without a modulatory transmitter.
    ModulatoryRoleRequiresModulator,
}

#[derive(Clone, Debug, PartialEq)]
/// A failure while evolving a neuron during simulation.
pub enum NeuronRuntimeError {
    /// An update was requested before the neuron's last update time.
    TimeWentBackwards {
        /// Current timestamp stored by the neuron.
        current: SimTimeUs,
        /// Earlier timestamp requested by the caller.
        requested: SimTimeUs,
    },
    /// The supplied integrated input is NaN or infinite.
    NonFiniteInput,
    /// The supplied noise sample is NaN or infinite.
    NonFiniteNoise,
    /// A named modulator level is NaN or infinite.
    InvalidModulatorLevel {
        /// Name of the invalid modulator.
        name: &'static str,
        /// Invalid level supplied by the caller.
        value: f32,
    },
}
