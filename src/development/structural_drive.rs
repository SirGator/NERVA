//! Neuron-local incoming-growth demand.
//!
//! A positive structural drive `D_i^{in,+}` means neuron *i* needs more
//! **incoming** input. Formation creates `source → i` (incoming). Negative
//! structural drive values carry no pruning semantics; they are clamped to
//! zero. Pruning is a synapse-local decision owned by the `PruningController`
//! based on smoothed utility (see [`super::pruning`]).

use std::{error::Error, fmt};

use crate::core::Neuron;

/// Incoming-growth demand extracted from a neuron's local structural signal.
///
/// Only the growth direction is represented here. Pruning is a synapse-local
/// decision owned by the `PruningController`, not a neuron-level "delete
/// something" signal.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IncomingGrowthDrive {
    /// Demand for new incoming connections.
    pub demand: f32,
}

impl IncomingGrowthDrive {
    /// Extracts the non-negative incoming-growth demand from a neuron.
    ///
    /// Negative structural drive values are clamped to zero: they do not
    /// carry pruning semantics in this model.
    pub fn from_neuron(neuron: &Neuron) -> Result<Self, StructuralDriveError> {
        let raw = neuron.structural_drive();
        if !raw.is_finite() {
            return Err(StructuralDriveError::NonFinite(raw));
        }
        Ok(Self {
            demand: raw.max(0.0),
        })
    }
}

/// A neuron carried a structural signal that cannot safely drive topology.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StructuralDriveError {
    /// The signal was NaN or infinite.
    NonFinite(f32),
}

impl fmt::Display for StructuralDriveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(value) => {
                write!(formatter, "structural drive must be finite, got {value}")
            }
        }
    }
}

impl Error for StructuralDriveError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::NeuronConfig,
        core::{NeuronId, Polarity, SimTime},
        math::Position3D,
    };

    fn neuron_with_drive(drive: f32) -> Neuron {
        let mut neuron = Neuron::new(
            NeuronId(1),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            None,
            NeuronConfig::default(),
            SimTime::ZERO,
        )
        .unwrap();
        neuron
            .set_structural_drive_clamped(drive, -10.0, 10.0)
            .unwrap();
        neuron
    }

    #[test]
    fn positive_drive_becomes_growth_demand() {
        let drive = IncomingGrowthDrive::from_neuron(&neuron_with_drive(2.0)).unwrap();
        assert_eq!(drive.demand, 2.0);
    }

    #[test]
    fn negative_drive_clamps_to_zero_not_pruning() {
        let drive = IncomingGrowthDrive::from_neuron(&neuron_with_drive(-3.0)).unwrap();
        assert_eq!(drive.demand, 0.0);
    }

    #[test]
    fn finite_drive_is_extracted_correctly() {
        let mut neuron = Neuron::new(
            NeuronId(1),
            Position3D::ORIGIN,
            Polarity::Excitatory,
            None,
            NeuronConfig::default(),
            SimTime::ZERO,
        )
        .unwrap();
        neuron
            .set_structural_drive_clamped(5.0, -10.0, 10.0)
            .unwrap();
        let drive = IncomingGrowthDrive::from_neuron(&neuron).unwrap();
        assert_eq!(drive.demand, 5.0);
    }
}
