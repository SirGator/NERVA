//! Passive local chemical identities, receptors and held cellular effects.
//! No runtime, field solver or learning dependency belongs to these core types.
use super::SimTime;
use std::{error::Error, fmt};

/// Chemical identity, distinct from the stable identity of a releasing source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChemicalSpecies {
    Dopamine,
    Acetylcholine,
    Noradrenaline,
}
impl ChemicalSpecies {
    pub const ALL: [Self; 3] = [Self::Dopamine, Self::Acetylcholine, Self::Noradrenaline];
    pub const fn index(self) -> usize {
        match self {
            Self::Dopamine => 0,
            Self::Acetylcholine => 1,
            Self::Noradrenaline => 2,
        }
    }
}

/// Local concentrations in abstract mass per geometry-volume units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Concentrations {
    pub dopamine: f32,
    pub acetylcholine: f32,
    pub noradrenaline: f32,
}
impl Concentrations {
    pub const ZERO: Self = Self {
        dopamine: 0.,
        acetylcholine: 0.,
        noradrenaline: 0.,
    };
    pub const fn amount(self, species: ChemicalSpecies) -> f32 {
        match species {
            ChemicalSpecies::Dopamine => self.dopamine,
            ChemicalSpecies::Acetylcholine => self.acetylcholine,
            ChemicalSpecies::Noradrenaline => self.noradrenaline,
        }
    }
    pub fn validate(self) -> Result<(), ChemicalError> {
        if ChemicalSpecies::ALL
            .iter()
            .all(|&s| self.amount(s).is_finite() && self.amount(s) >= 0.)
        {
            Ok(())
        } else {
            Err(ChemicalError::Invalid(
                "concentrations must be finite and nonnegative",
            ))
        }
    }
}

/// Per-spike mass released by a source cell; zero is a non-releasing cell.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReleaseProfile {
    pub dopamine: f32,
    pub acetylcholine: f32,
    pub noradrenaline: f32,
}
impl ReleaseProfile {
    pub const fn amount(self, species: ChemicalSpecies) -> f32 {
        match species {
            ChemicalSpecies::Dopamine => self.dopamine,
            ChemicalSpecies::Acetylcholine => self.acetylcholine,
            ChemicalSpecies::Noradrenaline => self.noradrenaline,
        }
    }
    pub fn validate(self) -> Result<(), ChemicalError> {
        if ChemicalSpecies::ALL
            .iter()
            .all(|&s| self.amount(s).is_finite() && (0. ..=1_000_000.).contains(&self.amount(s)))
        {
            Ok(())
        } else {
            Err(ChemicalError::Invalid(
                "release mass must be finite and in [0, 1000000]",
            ))
        }
    }
}

/// Receptor sensitivities, not concentrations or categorical cell identities.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReceptorProfile {
    pub dopamine_d1_like: f32,
    pub dopamine_d2_like: f32,
    pub acetylcholine: f32,
    pub noradrenaline: f32,
}
impl ReceptorProfile {
    pub fn validate(self) -> Result<(), ChemicalError> {
        if [
            self.dopamine_d1_like,
            self.dopamine_d2_like,
            self.acetylcholine,
            self.noradrenaline,
        ]
        .iter()
        .all(|v| v.is_finite() && (0. ..=8.).contains(v))
        {
            Ok(())
        } else {
            Err(ChemicalError::Invalid(
                "receptor sensitivities must be finite and in [0, 8]",
            ))
        }
    }
    /// Synthetic saturating receptor coupling shared by all cells. Opposing D1/D2
    /// responses are local; no reward or task label enters this calculation.
    pub fn response(
        self,
        concentrations: Concentrations,
    ) -> Result<ModulationState, ChemicalError> {
        self.validate()?;
        concentrations.validate()?;
        let occupancy = |c: f32| (f64::from(c) / (1. + f64::from(c))) as f32;
        let dopamine = occupancy(concentrations.dopamine);
        let d1 = self.dopamine_d1_like * dopamine;
        let d2 = self.dopamine_d2_like * dopamine;
        let ach = self.acetylcholine * occupancy(concentrations.acetylcholine);
        let na = self.noradrenaline * occupancy(concentrations.noradrenaline);
        Ok(ModulationState {
            threshold_shift: (-0.15 * d1 + 0.15 * d2 - 0.12 * na).clamp(-2., 2.),
            intrinsic_drive: (10. * na + 4. * ach).clamp(0., 112.),
            learning_signal: (d1 - d2 + 0.15 * ach + 0.2 * na).clamp(-4., 4.),
            transmission_gain: (1. - 0.45 * ach + 0.15 * na).clamp(0.1, 3.),
        })
    }
}

/// Applied receptor effects, kept independent of homeostatic base state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModulationState {
    pub threshold_shift: f32,
    pub intrinsic_drive: f32,
    /// Signed local third factor. Zero leaves marked synapses unconsolidated.
    pub learning_signal: f32,
    pub transmission_gain: f32,
}
impl Default for ModulationState {
    fn default() -> Self {
        Self {
            threshold_shift: 0.,
            intrinsic_drive: 0.,
            learning_signal: 0.,
            transmission_gain: 1.,
        }
    }
}
impl ModulationState {
    pub fn validate(self) -> Result<(), ChemicalError> {
        if self.threshold_shift.is_finite()
            && (-2. ..=2.).contains(&self.threshold_shift)
            && self.intrinsic_drive.is_finite()
            && (-112. ..=112.).contains(&self.intrinsic_drive)
            && self.learning_signal.is_finite()
            && (-4. ..=4.).contains(&self.learning_signal)
            && self.transmission_gain.is_finite()
            && (0.1..=3.).contains(&self.transmission_gain)
        {
            Ok(())
        } else {
            Err(ChemicalError::Invalid(
                "modulation effects outside finite supported bounds",
            ))
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ChemicalError {
    Invalid(&'static str),
    TimeWentBackwards {
        current: SimTime,
        requested: SimTime,
    },
    CapacityExceeded {
        limit: usize,
    },
}
impl fmt::Display for ChemicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => f.write_str(reason),
            Self::TimeWentBackwards { current, requested } => write!(
                f,
                "chemical time went backwards from {current} to {requested}"
            ),
            Self::CapacityExceeded { limit } => {
                write!(f, "chemical pulse capacity {limit} exceeded")
            }
        }
    }
}
impl Error for ChemicalError {}
