//! Weight-distribution measurements.

/// Aggregate non-mutating measurements of synaptic magnitudes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WeightMetrics {
    /// Number of sampled weights.
    pub count: usize,
    /// Arithmetic mean, or zero for an empty sample.
    pub mean: f32,
    /// Fraction at the inclusive configured minimum.
    pub at_min_fraction: f32,
    /// Fraction at the inclusive configured maximum.
    pub at_max_fraction: f32,
}

impl WeightMetrics {
    /// Derives measurements from copied weights.
    pub fn from_weights(
        weights: impl IntoIterator<Item = f32>,
        min_weight: f32,
        max_weight: f32,
    ) -> Self {
        let mut weights: Vec<_> = weights.into_iter().collect();
        if weights.is_empty() {
            return Self::default();
        }

        // Canonical ordering makes the floating-point aggregate independent of
        // the input collection's iteration order.
        weights.sort_by(f32::total_cmp);
        let at_min = weights
            .iter()
            .filter(|&&weight| approximately_equal(weight, min_weight))
            .count();
        let at_max = weights
            .iter()
            .filter(|&&weight| approximately_equal(weight, max_weight))
            .count();
        let sum: f64 = weights.iter().map(|&weight| f64::from(weight)).sum();
        let count = weights.len();

        Self {
            count,
            mean: (sum / count as f64) as f32,
            at_min_fraction: at_min as f32 / count as f32,
            at_max_fraction: at_max as f32 / count as f32,
        }
    }
}

fn approximately_equal(left: f32, right: f32) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= f32::EPSILON * 8.0 * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_sample_has_zero_metrics() {
        assert_eq!(
            WeightMetrics::from_weights([], 0.0, 1.0),
            WeightMetrics::default()
        );
    }

    #[test]
    fn computes_mean_and_inclusive_bound_fractions() {
        let metrics = WeightMetrics::from_weights([1.0, 0.0, 0.5, 1.0], 0.0, 1.0);

        assert_eq!(metrics.count, 4);
        assert_eq!(metrics.mean, 0.625);
        assert_eq!(metrics.at_min_fraction, 0.25);
        assert_eq!(metrics.at_max_fraction, 0.5);
    }

    #[test]
    fn result_does_not_depend_on_input_iteration_order() {
        let forward = WeightMetrics::from_weights([0.1, 1_000.0, 0.2], 0.0, 1_000.0);
        let reverse = WeightMetrics::from_weights([0.2, 1_000.0, 0.1], 0.0, 1_000.0);

        assert_eq!(forward, reverse);
    }
}
