//! Weight publication policy, independent of pair timing and neuronal dynamics.
use super::WeightBounds;
use crate::primitives::Weight;

/// How a local rule publishes its requested delta to a transmitted f32 weight.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightUpdateMode {
    /// Historical rounded delta and f32 addition, without compensation.
    #[default]
    DirectF32,
    /// Preserve unpublished signed mass locally until f32 publication can change.
    Accumulated,
}

/// Diagnostic filter of individual requested contributions before carry and bounds.
/// Eligibility, chemistry, pair timing and neural dynamics remain unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightContribution {
    #[default]
    All,
    PotentiatingOnly,
    DepressingOnly,
}
impl WeightContribution {
    pub fn retain(self, delta: f64) -> f64 {
        match self {
            Self::All => delta,
            Self::PotentiatingOnly => delta.max(0.),
            Self::DepressingOnly => delta.min(0.),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Publication {
    pub weight: Weight,
    pub residual: f64,
    pub clipped_delta: f64,
    pub rounding_error: f64,
}

pub(crate) fn publish(
    weight: Weight,
    residual: f64,
    delta: f64,
    bounds: WeightBounds,
    mode: WeightUpdateMode,
) -> Publication {
    let old = f64::from(weight.get());
    let low = f64::from(bounds.min().get()) - old;
    let high = f64::from(bounds.max().get()) - old;
    match mode {
        WeightUpdateMode::DirectF32 => {
            let cast = delta.clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32;
            let next = bounds.apply_delta(weight, cast);
            let clipped = delta - delta.clamp(low, high);
            Publication {
                weight: next,
                residual,
                clipped_delta: clipped,
                rounding_error: delta - (f64::from(next.get()) - old) - clipped,
            }
        }
        WeightUpdateMode::Accumulated => {
            // Sum near zero first. Contributions below even an f64 ULP of
            // the visible weight still survive in the independent carry.
            let pending = residual + delta;
            let allowed = pending.clamp(low, high);
            // Reapply typed bounds after publication: a very wide subtraction
            // can lose a small positive lower bound even in f64.
            let next = bounds.clamp(
                Weight::new((old + allowed) as f32).expect("validated finite weight bounds"),
            );
            let actual = f64::from(next.get()) - old;
            let carry = allowed - actual;
            Publication {
                weight: next,
                residual: carry,
                clipped_delta: pending - allowed,
                rounding_error: delta - actual - (carry - residual) - (pending - allowed),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wide_dynamic_ranges_still_obey_the_exact_f32_lower_bound() {
        let bounds = WeightBounds::new(0.2, f32::MAX).unwrap();
        let result = publish(
            Weight::new(f32::MAX).unwrap(),
            0.,
            -f64::from(f32::MAX),
            bounds,
            WeightUpdateMode::Accumulated,
        );
        assert_eq!(result.weight, bounds.min());
        assert!(result.residual.is_finite());
    }
    #[test]
    fn signed_carry_cancels_and_clipped_mass_never_becomes_future_debt() {
        let bounds = WeightBounds::new(0., 2.).unwrap();
        let start = Weight::new(0.5).unwrap();
        let up = publish(start, 0., 1e-12, bounds, WeightUpdateMode::Accumulated);
        let down = publish(
            up.weight,
            up.residual,
            -1e-12,
            bounds,
            WeightUpdateMode::Accumulated,
        );
        assert_eq!(down.weight, start);
        assert_eq!(down.residual, 0.);
        let top = publish(
            Weight::new(2.).unwrap(),
            0.,
            1e-12,
            bounds,
            WeightUpdateMode::Accumulated,
        );
        assert_eq!(top.residual, 0.);
        assert_eq!(top.clipped_delta, 1e-12);
        let reverse = publish(
            top.weight,
            top.residual,
            -1e-12,
            bounds,
            WeightUpdateMode::Accumulated,
        );
        assert!(reverse.residual < 0.);
        assert_eq!(reverse.clipped_delta, 0.);
    }
}
