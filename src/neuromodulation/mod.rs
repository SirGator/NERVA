//! Local volume transmission with receptor-specific effects.
//!
//! Pulses are finite-width Gaussian masses in unbounded three-dimensional space.
//! Their diffusion/linear clearance has an analytical solution, so releases and
//! queries need no spatial grid or per-synapse chemical solver. The configured
//! retention cutoff drops each remaining mass below `exp(-retention_tau_multiple)`
//! of its original amount; this is an explicit bounded-memory approximation.
//! Names are biological inspirations, not calibrated biological units or claims.

pub use crate::core::{
    ChemicalError, ChemicalSpecies, Concentrations, ModulationState, ReceptorProfile,
    ReleaseProfile,
};

use crate::{core::SimTime, math::Position3D, primitives::ModulatorId};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeciesKinetics {
    /// Geometry units squared per second; zero disables diffusion.
    pub diffusion: f64,
    pub decay_tau_us: f64,
    /// Initial Gaussian standard deviation, avoiding singular point sources.
    pub initial_sigma: f64,
}
impl SpeciesKinetics {
    fn validate(self) -> Result<(), ChemicalError> {
        if self.diffusion.is_finite()
            && (0. ..=1_000_000.).contains(&self.diffusion)
            && self.decay_tau_us.is_finite()
            && (1. ..=1e12).contains(&self.decay_tau_us)
            && self.initial_sigma.is_finite()
            && (1e-6..=1e6).contains(&self.initial_sigma)
        {
            Ok(())
        } else {
            Err(ChemicalError::Invalid(
                "invalid diffusion, clearance time or initial spread",
            ))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChemicalConfig {
    pub species: [SpeciesKinetics; 3],
    /// Receptor sample-and-hold interval, owned/scheduled by the runtime.
    pub update_interval_us: u64,
    pub max_pulses: usize,
    /// Explicit tail truncation in clearance time constants (default exp(-20)).
    pub retention_tau_multiple: f64,
}
impl Default for ChemicalConfig {
    fn default() -> Self {
        Self {
            species: [
                SpeciesKinetics {
                    diffusion: 0.025,
                    decay_tau_us: 400_000.,
                    initial_sigma: 0.08,
                },
                SpeciesKinetics {
                    diffusion: 0.040,
                    decay_tau_us: 200_000.,
                    initial_sigma: 0.08,
                },
                SpeciesKinetics {
                    diffusion: 0.030,
                    decay_tau_us: 500_000.,
                    initial_sigma: 0.08,
                },
            ],
            update_interval_us: 5_000,
            max_pulses: 16_384,
            retention_tau_multiple: 20.,
        }
    }
}
impl ChemicalConfig {
    pub fn validate(self) -> Result<(), ChemicalError> {
        for kinetics in self.species {
            kinetics.validate()?;
        }
        if self.update_interval_us == 0
            || self.max_pulses == 0
            || self.max_pulses > 1_000_000
            || !self.retention_tau_multiple.is_finite()
            || !(1. ..=100.).contains(&self.retention_tau_multiple)
        {
            return Err(ChemicalError::Invalid(
                "invalid chemical sampling, capacity or retention budget",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Pulse {
    source: ModulatorId,
    species: ChemicalSpecies,
    position: Position3D,
    amount: f64,
    at: SimTime,
    expires: SimTime,
}

/// Bounded list of local releases; querying never alters concentrations/state.
#[derive(Clone, Debug, PartialEq)]
pub struct ChemicalField {
    config: ChemicalConfig,
    clock: SimTime,
    pulses: Vec<Pulse>,
}
impl ChemicalField {
    pub fn new(config: ChemicalConfig) -> Result<Self, ChemicalError> {
        config.validate()?;
        Ok(Self {
            config,
            clock: SimTime::ZERO,
            pulses: Vec::new(),
        })
    }
    pub const fn config(&self) -> &ChemicalConfig {
        &self.config
    }
    pub const fn clock(&self) -> SimTime {
        self.clock
    }
    pub fn is_empty(&self) -> bool {
        self.pulses.is_empty()
    }
    pub fn pulse_count(&self) -> usize {
        self.pulses.len()
    }
    pub fn active_until(&self) -> Option<SimTime> {
        self.pulses.iter().map(|p| p.expires).max()
    }

    pub fn advance_to(&mut self, time: SimTime) -> Result<(), ChemicalError> {
        self.validate_time(time)?;
        self.pulses.retain(|p| time < p.expires);
        self.clock = time;
        Ok(())
    }

    /// Add a finite nonnegative mass. Same-source/species/time/position pulses
    /// coalesce; invalid/capacity-rejected releases leave the field unchanged.
    pub fn release(
        &mut self,
        source: ModulatorId,
        species: ChemicalSpecies,
        position: Position3D,
        amount: f32,
        time: SimTime,
    ) -> Result<(), ChemicalError> {
        self.validate_time(time)?;
        position
            .validate()
            .map_err(|_| ChemicalError::Invalid("non-finite release position"))?;
        if !amount.is_finite() || !(0. ..=1_000_000.).contains(&amount) {
            return Err(ChemicalError::Invalid(
                "release mass must be finite and in [0, 1000000]",
            ));
        }
        if amount == 0. {
            return self.advance_to(time);
        }
        let same = self.pulses.iter().position(|p| {
            p.at == time && p.source == source && p.species == species && p.position == position
        });
        let combined = same.map_or(f64::from(amount), |i| {
            self.pulses[i].amount + f64::from(amount)
        });
        if combined > 1e12 {
            return Err(ChemicalError::Invalid(
                "coalesced release mass exceeds 1e12",
            ));
        }
        if same.is_none()
            && self.pulses.iter().filter(|p| time < p.expires).count() >= self.config.max_pulses
        {
            return Err(ChemicalError::CapacityExceeded {
                limit: self.config.max_pulses,
            });
        }
        let duration = (self.config.species[species.index()].decay_tau_us
            * self.config.retention_tau_multiple)
            .ceil() as u64;
        let expires = time.checked_add_us(duration).ok_or(ChemicalError::Invalid(
            "release lifetime overflows simulation time",
        ))?;
        // Coalescing applies only at the current exact timestamp, so pruning
        // cannot remove this pulse. Refind after retain to avoid stale indices.
        self.advance_to(time)?;
        if let Some(pulse) = self.pulses.iter_mut().find(|p| {
            p.at == time && p.source == source && p.species == species && p.position == position
        }) {
            pulse.amount = combined;
        } else {
            self.pulses.push(Pulse {
                source,
                species,
                position,
                amount: combined,
                at: time,
                expires,
            });
        }
        Ok(())
    }

    /// Exact Gaussian diffusion/linear-clearance solution before tail cutoff.
    pub fn sample(
        &self,
        position: Position3D,
        time: SimTime,
    ) -> Result<Concentrations, ChemicalError> {
        self.validate_time(time)?;
        position
            .validate()
            .map_err(|_| ChemicalError::Invalid("non-finite query position"))?;
        let mut result = [0_f64; 3];
        for pulse in &self.pulses {
            if time < pulse.at || time >= pulse.expires {
                continue;
            }
            let kinetics = self.config.species[pulse.species.index()];
            let age_us = (time.0 - pulse.at.0) as f64;
            let variance = kinetics.initial_sigma.powi(2) + 2. * kinetics.diffusion * age_us * 1e-6;
            let dx = f64::from(position.x) - f64::from(pulse.position.x);
            let dy = f64::from(position.y) - f64::from(pulse.position.y);
            let dz = f64::from(position.z) - f64::from(pulse.position.z);
            let distance_squared = dx * dx + dy * dy + dz * dz;
            let density = pulse.amount / (std::f64::consts::TAU * variance).powf(1.5)
                * (-age_us / kinetics.decay_tau_us - distance_squared / (2. * variance)).exp();
            result[pulse.species.index()] += density;
        }
        let concentrations = Concentrations {
            dopamine: result[0] as f32,
            acetylcholine: result[1] as f32,
            noradrenaline: result[2] as f32,
        };
        concentrations.validate()?;
        Ok(concentrations)
    }

    fn validate_time(&self, time: SimTime) -> Result<(), ChemicalError> {
        if time < self.clock {
            Err(ChemicalError::TimeWentBackwards {
                current: self.clock,
                requested: time,
            })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn relative(actual: f32, expected: f64) {
        assert!(
            (f64::from(actual) - expected).abs() <= 2e-7 * expected.abs().max(1.),
            "{actual} vs {expected}"
        );
    }
    #[test]
    fn gaussian_matches_independent_local_and_spatial_decay_solution() {
        let mut config = ChemicalConfig::default();
        config.species[0] = SpeciesKinetics {
            diffusion: 0.2,
            decay_tau_us: 1_000_000.,
            initial_sigma: 0.1,
        };
        let mut field = ChemicalField::new(config).unwrap();
        field
            .release(
                ModulatorId(7),
                ChemicalSpecies::Dopamine,
                Position3D::ORIGIN,
                0.01,
                SimTime::ZERO,
            )
            .unwrap();
        let initial = 0.01 / (2. * std::f64::consts::PI * 0.01).powf(1.5);
        relative(
            field
                .sample(Position3D::ORIGIN, SimTime::ZERO)
                .unwrap()
                .dopamine,
            initial,
        );
        let near = field
            .sample(Position3D::new(0.1, 0., 0.), SimTime::ZERO)
            .unwrap();
        relative(near.dopamine, initial * (-0.5_f64).exp());
        let later = field.sample(Position3D::ORIGIN, SimTime(500_000)).unwrap();
        relative(
            later.dopamine,
            0.01 / (2. * std::f64::consts::PI * 0.21).powf(1.5) * (-0.5_f64).exp(),
        );
        assert_eq!(later.acetylcholine, 0.);
        assert_eq!(later.noradrenaline, 0.);
    }
    #[test]
    fn diffusion_transports_mass_outward_and_zero_diffusion_has_exact_clearance() {
        let mut config = ChemicalConfig::default();
        config.species[0].diffusion = 0.;
        let mut no_diffusion = ChemicalField::new(config).unwrap();
        config.species[0].diffusion = 0.1;
        let mut diffusion = ChemicalField::new(config).unwrap();
        for field in [&mut no_diffusion, &mut diffusion] {
            field
                .release(
                    ModulatorId(1),
                    ChemicalSpecies::Dopamine,
                    Position3D::ORIGIN,
                    1.,
                    SimTime::ZERO,
                )
                .unwrap();
        }
        let target = Position3D::new(0.5, 0., 0.);
        let before = no_diffusion.sample(target, SimTime::ZERO).unwrap().dopamine;
        let at = SimTime(100_000);
        relative(
            no_diffusion.sample(target, at).unwrap().dopamine,
            f64::from(before) * (-0.25_f64).exp(),
        );
        assert!(diffusion.sample(target, at).unwrap().dopamine > before);
    }
    #[test]
    fn species_are_independent_and_same_source_releases_coalesce() {
        let mut field = ChemicalField::new(ChemicalConfig::default()).unwrap();
        for species in ChemicalSpecies::ALL {
            field
                .release(
                    ModulatorId(3),
                    species,
                    Position3D::ORIGIN,
                    0.1,
                    SimTime(10),
                )
                .unwrap();
            field
                .release(
                    ModulatorId(3),
                    species,
                    Position3D::ORIGIN,
                    0.2,
                    SimTime(10),
                )
                .unwrap();
        }
        assert_eq!(field.pulse_count(), 3);
        let concentration = field.sample(Position3D::ORIGIN, SimTime(10)).unwrap();
        assert_eq!(concentration.dopamine, concentration.acetylcholine);
        assert_eq!(concentration.dopamine, concentration.noradrenaline);
        // Different releasing identities never become a chemical species ID.
        field
            .release(
                ModulatorId(4),
                ChemicalSpecies::Dopamine,
                Position3D::ORIGIN,
                0.3,
                SimTime(10),
            )
            .unwrap();
        assert_eq!(field.pulse_count(), 4);
        relative(
            field
                .sample(Position3D::ORIGIN, SimTime(10))
                .unwrap()
                .dopamine,
            2. * f64::from(concentration.dopamine),
        );
    }
    #[test]
    fn cutoff_is_explicit_and_invalid_or_capacity_releases_are_atomic() {
        let config = ChemicalConfig {
            max_pulses: 1,
            ..ChemicalConfig::default()
        };
        let mut field = ChemicalField::new(config).unwrap();
        field
            .release(
                ModulatorId(1),
                ChemicalSpecies::Dopamine,
                Position3D::ORIGIN,
                1.,
                SimTime(10),
            )
            .unwrap();
        let before = field.clone();
        assert!(
            field
                .release(
                    ModulatorId(2),
                    ChemicalSpecies::Dopamine,
                    Position3D::ORIGIN,
                    1.,
                    SimTime(11)
                )
                .is_err()
        );
        assert_eq!(field, before);
        assert!(
            field
                .release(
                    ModulatorId(1),
                    ChemicalSpecies::Dopamine,
                    Position3D::ORIGIN,
                    f32::NAN,
                    SimTime(10)
                )
                .is_err()
        );
        assert_eq!(field, before);
        let expiry = field.active_until().unwrap();
        assert_eq!(
            field.sample(Position3D::ORIGIN, expiry).unwrap(),
            Concentrations::ZERO
        );
        field.advance_to(expiry).unwrap();
        assert!(field.is_empty());
        assert!(field.sample(Position3D::ORIGIN, SimTime(9)).is_err());
    }
    #[test]
    fn receptor_zero_is_neutral_d1_d2_oppose_and_effects_saturate() {
        let high = Concentrations {
            dopamine: 1e20,
            acetylcholine: 1e20,
            noradrenaline: 1e20,
        };
        assert_eq!(
            ReceptorProfile::default().response(high).unwrap(),
            ModulationState::default()
        );
        let d1 = ReceptorProfile {
            dopamine_d1_like: 1.,
            ..ReceptorProfile::default()
        }
        .response(high)
        .unwrap();
        let d2 = ReceptorProfile {
            dopamine_d2_like: 1.,
            ..ReceptorProfile::default()
        }
        .response(high)
        .unwrap();
        assert!(d1.learning_signal > 0. && d2.learning_signal < 0.);
        assert_eq!(d1.threshold_shift, -d2.threshold_shift);
        let all = ReceptorProfile {
            dopamine_d1_like: 8.,
            dopamine_d2_like: 8.,
            acetylcholine: 8.,
            noradrenaline: 8.,
        }
        .response(high)
        .unwrap();
        all.validate().unwrap();
        assert!(
            ReceptorProfile {
                dopamine_d1_like: f32::NAN,
                ..ReceptorProfile::default()
            }
            .response(high)
            .is_err()
        );
        assert!(
            ReceptorProfile::default()
                .response(Concentrations {
                    dopamine: -1.,
                    ..Concentrations::ZERO
                })
                .is_err()
        );
    }
}
