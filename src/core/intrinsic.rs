//! Continuous intrinsic neuron state and bounded threshold-crossing helpers.
//!
//! This module contains no scheduler. It supplies the analytical local terms
//! used by [`crate::core::Neuron`]; the runtime only schedules the timestamp
//! that the neuron predicts.

/// Read-only snapshot of one neuron's continuous intrinsic state.
///
/// All drive fields are currents in potential units per second. The effective
/// threshold is the mutable base threshold plus `threshold_adaptation`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IntrinsicState {
    /// Baseline capability drive plus the current homeostatic adjustment.
    pub intrinsic_drive: f32,
    /// Short-lived positive after-current produced by recent spikes.
    pub burst_drive: f32,
    /// Short-lived negative after-current produced by recent spikes.
    pub adaptation_drive: f32,
    /// Positive after-current produced by recent inhibitory input.
    pub rebound_drive: f32,
    /// Temporary amount added to the base firing threshold.
    pub threshold_adaptation: f32,
    /// Threshold currently used for firing decisions.
    pub effective_threshold: f32,
}

/// Exact LIF response to an exponentially decaying current.
pub(crate) fn exponential_current_response(
    current: f32,
    current_tau_us: f32,
    membrane_tau_us: f32,
    elapsed_us: u64,
) -> f64 {
    if current == 0.0 || elapsed_us == 0 {
        return 0.0;
    }

    let elapsed_seconds = elapsed_us as f64 / 1_000_000.0;
    let current_tau_seconds = f64::from(current_tau_us) / 1_000_000.0;
    let membrane_tau_seconds = f64::from(membrane_tau_us) / 1_000_000.0;
    let current = f64::from(current);
    let relative_difference =
        (current_tau_seconds - membrane_tau_seconds).abs() / membrane_tau_seconds;
    if relative_difference <= 1.0e-9 {
        current * elapsed_seconds * (-elapsed_seconds / membrane_tau_seconds).exp()
    } else {
        current * membrane_tau_seconds * current_tau_seconds
            / (current_tau_seconds - membrane_tau_seconds)
            * ((-elapsed_seconds / current_tau_seconds).exp()
                - (-elapsed_seconds / membrane_tau_seconds).exp())
    }
}

#[derive(Clone, Copy, Debug)]
struct ExponentialGapTerm {
    /// `(constant + linear * t_us) * exp(-t_us / tau_us)`.
    constant: f64,
    linear: f64,
    tau_us: f64,
}

impl ExponentialGapTerm {
    fn value_at(self, elapsed_us: f64) -> f64 {
        (self.constant + self.linear * elapsed_us) * (-elapsed_us / self.tau_us).exp()
    }

    fn range(self, start: u64, end: u64) -> GapRange {
        let start = start as f64;
        let end = end as f64;
        let mut range = GapRange::from_value(self.value_at(start));
        range.include(self.value_at(end));
        if self.linear != 0.0 {
            let stationary = self.tau_us - self.constant / self.linear;
            if stationary > start && stationary < end {
                range.include(self.value_at(stationary));
            }
        }
        range
    }

    fn derivative(self) -> Self {
        Self {
            constant: self.linear - self.constant / self.tau_us,
            linear: -self.linear / self.tau_us,
            tau_us: self.tau_us,
        }
    }

    fn tail_absolute_bound(self, start: u64) -> f64 {
        let start = start as f64;
        let mut bound = self.value_at(start).abs();
        if self.linear != 0.0 {
            let stationary = self.tau_us - self.constant / self.linear;
            if stationary > start {
                bound = bound.max(self.value_at(stationary).abs());
            }
        }
        bound
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GapRange {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    absolute_scale: f64,
}

impl GapRange {
    fn from_value(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
            absolute_scale: value.abs(),
        }
    }

    fn include(&mut self, value: f64) {
        self.lower = self.lower.min(value);
        self.upper = self.upper.max(value);
        self.absolute_scale = self.absolute_scale.max(value.abs());
    }

    fn add(&mut self, other: Self) {
        self.lower += other.lower;
        self.upper += other.upper;
        self.absolute_scale += other.absolute_scale;
    }

    pub(crate) fn rounding_margin(self) -> f64 {
        self.absolute_scale.max(1.0) * f64::from(f32::EPSILON) * 16.0
    }
}

/// Exponential-polynomial representation of `V(t) - threshold(t)`.
#[derive(Clone, Debug)]
pub(crate) struct IntrinsicGapModel {
    constant: f64,
    terms: Vec<ExponentialGapTerm>,
}

impl IntrinsicGapModel {
    pub(crate) fn new(constant: f64) -> Self {
        Self {
            constant,
            terms: Vec::with_capacity(5),
        }
    }

    pub(crate) fn add_term(&mut self, tau_us: f32, constant: f64, linear: f64) {
        if constant == 0.0 && linear == 0.0 {
            return;
        }
        if let Some(existing) = self
            .terms
            .iter_mut()
            .find(|term| term.tau_us == f64::from(tau_us))
        {
            existing.constant += constant;
            existing.linear += linear;
        } else {
            self.terms.push(ExponentialGapTerm {
                constant,
                linear,
                tau_us: f64::from(tau_us),
            });
        }
    }

    pub(crate) fn add_current_response(
        &mut self,
        current: f32,
        current_tau_us: f32,
        membrane_tau_us: f32,
        sign: f64,
    ) {
        if current == 0.0 {
            return;
        }
        let current_tau_seconds = f64::from(current_tau_us) / 1_000_000.0;
        let membrane_tau_seconds = f64::from(membrane_tau_us) / 1_000_000.0;
        let signed_current = sign * f64::from(current);
        let relative_difference =
            (current_tau_seconds - membrane_tau_seconds).abs() / membrane_tau_seconds;
        if relative_difference <= 1.0e-9 {
            self.add_term(membrane_tau_us, 0.0, signed_current / 1_000_000.0);
        } else {
            let factor = signed_current * membrane_tau_seconds * current_tau_seconds
                / (current_tau_seconds - membrane_tau_seconds);
            self.add_term(current_tau_us, factor, 0.0);
            self.add_term(membrane_tau_us, -factor, 0.0);
        }
    }

    pub(crate) fn remove_zero_terms(&mut self) {
        self.terms
            .retain(|term| term.constant != 0.0 || term.linear != 0.0);
    }

    pub(crate) fn range(&self, start: u64, end: u64) -> GapRange {
        let mut range = GapRange::from_value(self.constant);
        for term in &self.terms {
            range.add(term.range(start, end));
        }
        range
    }

    pub(crate) fn derivative_range(&self, start: u64, end: u64) -> GapRange {
        let mut range = GapRange::from_value(0.0);
        for term in &self.terms {
            range.add(term.derivative().range(start, end));
        }
        range
    }

    pub(crate) fn tail_is_strictly_negative(&self, start: u64) -> bool {
        let scale = self
            .terms
            .iter()
            .map(|term| term.tail_absolute_bound(start))
            .sum::<f64>();
        let margin = (self.constant.abs() + scale).max(1.0) * f64::from(f32::EPSILON) * 16.0;
        if self.constant < 0.0 {
            return scale + margin < -self.constant;
        }
        if self.constant != 0.0 || self.terms.is_empty() {
            return false;
        }

        // At exact equilibrium, factor out the slowest exponential. Its
        // polynomial eventually dominates all faster terms.
        let leading = self
            .terms
            .iter()
            .max_by(|left, right| left.tau_us.total_cmp(&right.tau_us))
            .expect("non-empty terms were checked above");
        let leading_polynomial = leading.constant + leading.linear * start as f64;
        if leading_polynomial >= 0.0 || leading.linear > 0.0 {
            return false;
        }

        let mut scaled_rest_bound = 0.0;
        for term in self
            .terms
            .iter()
            .filter(|term| term.tau_us != leading.tau_us)
        {
            let rate_difference = 1.0 / term.tau_us - 1.0 / leading.tau_us;
            debug_assert!(rate_difference > 0.0);
            let scaled = ExponentialGapTerm {
                constant: term.constant,
                linear: term.linear,
                tau_us: 1.0 / rate_difference,
            };
            scaled_rest_bound += scaled.tail_absolute_bound(start);
        }
        let scaled_margin = (leading_polynomial.abs() + scaled_rest_bound).max(1.0)
            * f64::from(f32::EPSILON)
            * 16.0;
        scaled_rest_bound + scaled_margin < -leading_polynomial
    }
}
