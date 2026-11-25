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
            let actual = f64::from(next.get()) - old;
            // The part of the step that the bounds reject.
            let clipped = delta - delta.clamp(low, high);
            // Everything else that did not reach the weight: the rounding of the
            // f32 publication and the damping of the soft landing. It is
            // rejected and never retained as hidden debt.
            let rounding = delta - actual - clipped;
            Publication {
                weight: next,
                residual,
                clipped_delta: clipped,
                rounding_error: rounding,
            }
        }
        WeightUpdateMode::Accumulated => {
            // Sum near zero first. Contributions below even an f64 ULP of
            // the visible weight still survive in the independent carry.
            let pending = residual + delta;
            let allowed = pending.clamp(low, high);
            // Publish with the soft landing: the weight approaches the bound
            // continuously instead of jumping onto it.
            let next = bounds.soft_update(weight, allowed as f32);
            let actual = f64::from(next.get()) - old;
            // The part of the allowed step that the soft landing damped stays
            // in the carry and is offered again with the next publication.
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
    fn wide_dynamic_ranges_stay_inside_the_f32_bounds() {
        let bounds = WeightBounds::new(0.2, f32::MAX).unwrap();
        let result = publish(
            Weight::new(f32::MAX).unwrap(),
            0.,
            -f64::from(f32::MAX),
            bounds,
            WeightUpdateMode::Accumulated,
        );
        // The soft landing approaches the lower bound; it never leaves it.
        assert!(result.weight.get() >= bounds.min().get());
        assert!(result.weight.get().is_finite());
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
