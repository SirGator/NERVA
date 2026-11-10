//! Synthetic continuous construction profiles over one shared neuron model.
//! These settings are not fitted physiology, cell identities, or memory algorithms.
use crate::{
    config::{ConfigError, NeuronConfig},
    core::{Polarity, ShortTermConfig},
    math::Position3D,
    neuromodulation::ReceptorProfile,
};
use std::{error::Error, fmt};

/// Convex developmental influences; they are not neural activity or rewards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AreaWeights {
    pub cortex: f32,
    pub dg: f32,
    pub ca3: f32,
    pub ca1: f32,
}
impl AreaWeights {
    pub const CORTICAL: Self = Self {
        cortex: 1.,
        dg: 0.,
        ca3: 0.,
        ca1: 0.,
    };
    pub fn as_array(self) -> [f32; 4] {
        [self.cortex, self.dg, self.ca3, self.ca1]
    }
    pub fn validate(self) -> Result<(), ProfileError> {
        let values = self.as_array();
        if values
            .iter()
            .all(|v| v.is_finite() && (0. ..=1.).contains(v))
            && (values.iter().map(|&v| f64::from(v)).sum::<f64>() - 1.).abs() <= 1e-6
        {
            Ok(())
        } else {
            Err(ProfileError::InvalidWeights)
        }
    }
    fn mix(self, values: [f32; 4]) -> f32 {
        self.as_array()
            .iter()
            .zip(values)
            .map(|(&a, p)| a * p)
            .sum()
    }
}

/// Normalized overlapping Gaussian influences along a spatial x-axis.
/// Width zero is an explicit sharp-boundary control, not the default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AreaField {
    x_min: f32,
    x_max: f32,
    width: f32,
}
impl AreaField {
    pub fn new(x_min: f32, x_max: f32, width: f32) -> Result<Self, ProfileError> {
        if !x_min.is_finite()
            || !x_max.is_finite()
            || x_min >= x_max
            || !width.is_finite()
            || !(0. ..=1.).contains(&width)
        {
            return Err(ProfileError::InvalidField);
        }
        Ok(Self {
            x_min,
            x_max,
            width,
        })
    }
    pub fn at(self, position: Position3D) -> Result<AreaWeights, ProfileError> {
        position
            .validate()
            .map_err(|_| ProfileError::InvalidCoordinate)?;
        let x = ((f64::from(position.x) - f64::from(self.x_min))
            / (f64::from(self.x_max) - f64::from(self.x_min)))
        .clamp(0., 1.);
        self.at_normalized(x as f32)
    }
    pub fn at_normalized(self, x: f32) -> Result<AreaWeights, ProfileError> {
        if !x.is_finite() {
            return Err(ProfileError::InvalidCoordinate);
        }
        let x = f64::from(x.clamp(0., 1.));
        let distances = [0., 1. / 3., 2. / 3., 1.].map(|center| (x - center).powi(2));
        let closest = (0..4)
            .min_by(|&a, &b| distances[a].total_cmp(&distances[b]))
            .unwrap();
        let values = if self.width == 0. {
            let mut v = [0.; 4];
            v[closest] = 1.;
            v
        } else {
            let denominator = 2. * f64::from(self.width).powi(2);
            let raw = distances.map(|d| ((distances[closest] - d) / denominator).exp());
            let sum = raw.iter().sum::<f64>();
            raw.map(|v| (v / sum) as f32)
        };
        Ok(AreaWeights {
            cortex: values[0],
            dg: values[1],
            ca3: values[2],
            ca1: values[3],
        })
    }
}

/// Blends suitable continuous electrical parameters. Polarity stays discrete.
/// Thresholds are scaled relative to rest, so negative-potential LIF models work too.
pub fn electrical_profile(
    base: NeuronConfig,
    area: AreaWeights,
    polarity: Polarity,
) -> Result<NeuronConfig, ProfileError> {
    area.validate()?;
    base.validate().map_err(ProfileError::InvalidNeuron)?;
    if area == AreaWeights::CORTICAL && polarity == Polarity::Excitatory {
        return Ok(base);
    }
    let mut next = base;
    let gap = base.threshold - base.resting_potential;
    let factor = if polarity == Polarity::Inhibitory {
        0.8
    } else {
        1.
    };
    next.threshold = base.resting_potential + gap * area.mix([1., 1.45, 0.9, 1.05]) * factor;
    next.membrane_tau_us *= area.mix([1., 0.65, 1.6, 1.1])
        * if polarity == Polarity::Inhibitory {
            0.55
        } else {
            1.
        };
    next.refractory_period_us =
        (base.refractory_period_us as f64 * f64::from(area.mix([1., 1.3, 0.8, 1.]))).round() as u64;
    next.intrinsic.burst_gain += gap * area.mix([0., 0., 4., 0.]);
    next.intrinsic.adaptation_gain += gap * area.mix([0., 8., 2., 4.]);
    next.intrinsic.threshold_adaptation_gain += gap * area.mix([0., 0.12, 0.025, 0.04]);
    if polarity == Polarity::Inhibitory {
        next.refractory_period_us = (base.refractory_period_us as f64 * 0.24).round() as u64;
        next.intrinsic.burst_gain = 0.;
    }
    next.validate().map_err(ProfileError::InvalidNeuron)?;
    Ok(next)
}

/// Positive connection tendencies; no area pair is prohibited by this function.
/// The host separately applies geometry, candidate limits and seeded sampling.
pub fn connection_affinity(
    pre: AreaWeights,
    post: AreaWeights,
    polarity: Polarity,
) -> Result<f32, ProfileError> {
    pre.validate()?;
    post.validate()?;
    Ok(if polarity == Polarity::Inhibitory {
        1. + 0.8 * post.dg
    } else {
        0.35 + pre.cortex * (0.8 * post.cortex + 0.55 * post.dg + 0.45 * post.ca1)
            + pre.dg * (1.2 * post.ca3 + 0.2 * post.ca1)
            + pre.ca3 * (0.8 * post.ca3 + post.ca1 + 0.15 * post.dg)
            + pre.ca1 * (0.4 * post.cortex + 0.3 * post.ca3 + 0.3 * post.ca1)
    })
}

/// Receptor sensitivities blended from the same local construction influences.
/// The host chooses source cells separately; this never creates reward signals.
pub fn receptor_profile(area: AreaWeights) -> Result<ReceptorProfile, ProfileError> {
    area.validate()?;
    Ok(ReceptorProfile {
        dopamine_d1_like: area.mix([0.5, 0.7, 1.1, 0.9]),
        dopamine_d2_like: area.mix([0.35, 0.65, 0.3, 0.45]),
        acetylcholine: area.mix([0.7, 1.1, 0.8, 0.9]),
        noradrenaline: area.mix([0.6, 0.9, 0.7, 0.8]),
    })
}

/// An explicit synapse-class tendency for excitatory DG→CA3-like connections.
/// Choosing this discrete release class is distinct from smooth area influences.
pub fn short_term_profile(
    pre: AreaWeights,
    post: AreaWeights,
    polarity: Polarity,
) -> Result<Option<ShortTermConfig>, ProfileError> {
    pre.validate()?;
    post.validate()?;
    let strength = pre.dg * post.ca3;
    Ok(if polarity == Polarity::Excitatory && strength >= 0.15 {
        Some(ShortTermConfig {
            utilization: 0.25 - 0.2 * strength,
            facilitation_increment: 0.18 * strength,
            recovery_tau_us: 200_000.,
            facilitation_tau_us: 500_000.,
        })
    } else {
        None
    })
}

/// Invalid construction data is rejected before creating a live cell.
#[derive(Clone, Debug, PartialEq)]
pub enum ProfileError {
    InvalidField,
    InvalidWeights,
    InvalidCoordinate,
    InvalidNeuron(ConfigError),
}
impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid local cell profile: {self:?}")
    }
}
impl Error for ProfileError {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smooth_and_sharp_fields_have_valid_limits_and_no_nan_coordinates() {
        let field = AreaField::new(-1., 1., 0.16).unwrap();
        for i in 0..=1000 {
            field
                .at_normalized(i as f32 / 1000.)
                .unwrap()
                .validate()
                .unwrap();
        }
        let left = field.at_normalized(0.4999).unwrap().as_array();
        let right = field.at_normalized(0.5001).unwrap().as_array();
        assert!(left.iter().zip(right).all(|(a, b)| (a - b).abs() < 0.001));
        assert_eq!(
            AreaField::new(-1., 1., 0.)
                .unwrap()
                .at_normalized(1. / 3.)
                .unwrap()
                .as_array(),
            [0., 1., 0., 0.]
        );
        for width in [f32::MIN_POSITIVE, 1e-20, 1.] {
            AreaField::new(-1., 1., width)
                .unwrap()
                .at_normalized(0.5)
                .unwrap()
                .validate()
                .unwrap();
        }
        assert!(field.at_normalized(f32::NAN).is_err());
        assert!(field.at(Position3D::new(f32::NAN, 0., 0.)).is_err());
    }
    #[test]
    fn profiles_work_for_negative_and_zero_rest_and_all_paths_remain_open() {
        let field = AreaField::new(0., 1., 0.).unwrap();
        for base in [
            NeuronConfig::default(),
            NeuronConfig {
                resting_potential: 0.,
                reset_potential: 0.,
                threshold: 1.,
                ..NeuronConfig::default()
            },
        ] {
            assert_eq!(
                electrical_profile(base, AreaWeights::CORTICAL, Polarity::Excitatory).unwrap(),
                base
            );
            for x in [0., 1. / 3., 2. / 3., 1.] {
                let pre = field.at_normalized(x).unwrap();
                for polarity in [Polarity::Excitatory, Polarity::Inhibitory] {
                    electrical_profile(base, pre, polarity)
                        .unwrap()
                        .validate()
                        .unwrap();
                    receptor_profile(pre).unwrap().validate().unwrap();
                    for y in [0., 1. / 3., 2. / 3., 1.] {
                        assert!(
                            connection_affinity(pre, field.at_normalized(y).unwrap(), polarity)
                                .unwrap()
                                > 0.
                        );
                    }
                }
            }
        }
        let dg = field.at_normalized(1. / 3.).unwrap();
        let ca3 = field.at_normalized(2. / 3.).unwrap();
        assert!(
            short_term_profile(dg, ca3, Polarity::Excitatory)
                .unwrap()
                .is_some()
        );
        assert!(
            short_term_profile(dg, ca3, Polarity::Inhibitory)
                .unwrap()
                .is_none()
        );
        let invalid = AreaWeights {
            cortex: 1.,
            dg: 1.,
            ca3: 0.,
            ca1: 0.,
        };
        assert!(
            electrical_profile(NeuronConfig::default(), invalid, Polarity::Excitatory).is_err()
        );
    }
}
