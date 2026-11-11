//! Bounded optional observations; never part of neural or replay state.
use crate::{
    core::{Concentrations, ModulationState, NeuronId, SimTime, Synapse, SynapseId},
    learning::DelayedLearningUpdate,
    primitives::Weight,
};
use std::{collections::BTreeMap, error::Error, fmt};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LearningPhase {
    Delayed,
    PreArrival,
    PostSpike,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LearningAuditEvent {
    Trace {
        time: SimTime,
        synapse: SynapseId,
        phase: LearningPhase,
        tag_before: f32,
        tag_after: f32,
        tag_anchor_before: Option<SimTime>,
        held_signal: f32,
        old_weight: Weight,
        new_weight: Weight,
        residual_before: f64,
        residual_after: f64,
        delayed: Option<DelayedLearningUpdate>,
    },
    ChemicalSample {
        time: SimTime,
        neuron: NeuronId,
        concentrations: Concentrations,
        response: ModulationState,
    },
}

/// Per-drain exact sums in native integration order; no neural state is changed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DelayedLearningSummary {
    pub integrations: usize,
    pub first_us: Option<u64>,
    pub last_us: Option<u64>,
    pub peak_abs_tag: f32,
    pub positive_tag_integral: f64,
    pub negative_tag_integral: f64,
    pub positive_requested: f64,
    pub negative_requested: f64,
    pub positive_requested_no_change: f64,
    pub negative_requested_no_change: f64,
    pub max_abs_requested: f64,
    pub min_weight: Option<f32>,
    pub max_weight: Option<f32>,
    pub positive_actual: f64,
    pub negative_actual: f64,
    pub actual_changes: usize,
    pub nonzero_requested: usize,
    pub nonzero_requested_without_weight_change: usize,
    pub zero_signal_integrations: usize,
    pub min_signal: Option<f32>,
    pub max_signal: Option<f32>,
    pub first_nonzero_signal_us: Option<u64>,
    pub last_nonzero_signal_us: Option<u64>,
    pub first_residual: Option<f64>,
    pub last_residual: Option<f64>,
    pub positive_clipped: f64,
    pub negative_clipped: f64,
    pub positive_rounding_error: f64,
    pub negative_rounding_error: f64,
}
impl DelayedLearningSummary {
    fn record(&mut self, event: &LearningAuditEvent) {
        let LearningAuditEvent::Trace {
            time,
            tag_before,
            held_signal,
            old_weight,
            new_weight,
            residual_before,
            residual_after,
            delayed,
            ..
        } = event
        else {
            return;
        };
        self.integrations += 1;
        self.first_residual.get_or_insert(*residual_before);
        self.last_residual = Some(*residual_after);
        self.first_us.get_or_insert(time.0);
        self.last_us = Some(time.0);
        self.peak_abs_tag = self.peak_abs_tag.max(tag_before.abs());
        self.min_signal = Some(
            self.min_signal
                .map_or(*held_signal, |v| v.min(*held_signal)),
        );
        self.max_signal = Some(
            self.max_signal
                .map_or(*held_signal, |v| v.max(*held_signal)),
        );
        self.zero_signal_integrations += usize::from(*held_signal == 0.);
        if *held_signal != 0. {
            self.first_nonzero_signal_us.get_or_insert(time.0);
            self.last_nonzero_signal_us = Some(time.0);
        }
        if let Some(update) = delayed {
            self.positive_clipped += update.clipped_delta.max(0.);
            self.negative_clipped += update.clipped_delta.min(0.);
            self.positive_rounding_error += update.rounding_error.max(0.);
            self.negative_rounding_error += update.rounding_error.min(0.);
            self.positive_tag_integral += update.tag_integral_seconds.max(0.);
            self.negative_tag_integral += update.tag_integral_seconds.min(0.);
            self.positive_requested += update.unrounded_delta.max(0.);
            self.negative_requested += update.unrounded_delta.min(0.);
            self.max_abs_requested = self.max_abs_requested.max(update.unrounded_delta.abs());
            if old_weight == new_weight {
                self.positive_requested_no_change += update.unrounded_delta.max(0.);
                self.negative_requested_no_change += update.unrounded_delta.min(0.);
            }
            self.nonzero_requested += usize::from(update.unrounded_delta != 0.);
            self.nonzero_requested_without_weight_change +=
                usize::from(update.unrounded_delta != 0. && old_weight == new_weight);
        }
        let low = old_weight.get().min(new_weight.get());
        let high = old_weight.get().max(new_weight.get());
        self.min_weight = Some(self.min_weight.map_or(low, |v| v.min(low)));
        self.max_weight = Some(self.max_weight.map_or(high, |v| v.max(high)));
        let delta = f64::from(new_weight.get()) - f64::from(old_weight.get());
        self.positive_actual += delta.max(0.);
        self.negative_actual += delta.min(0.);
        self.actual_changes += usize::from(delta != 0.);
    }
}

/// Overflow is explicit. Consumers must reject incomplete scientific records.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LearningAuditLog {
    pub events: Vec<LearningAuditEvent>,
    /// Optional exact integration boundaries, enabled only for local-rule replay.
    pub boundaries: Vec<SimTime>,
    pub dropped_events: usize,
    /// Every nonzero-tag time integration is counted, including sub-ULP attempts.
    /// Raw events retain actual weight changes; idle decay is aggregated here.
    pub delayed: BTreeMap<SynapseId, DelayedLearningSummary>,
}
impl LearningAuditLog {
    pub fn is_complete(&self) -> bool {
        self.dropped_events == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LearningAuditError {
    InvalidBudget,
    AlreadyEnabled,
}
impl fmt::Display for LearningAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidBudget => "learning audit budget must be in 1..=1000000",
            Self::AlreadyEnabled => "learning audit already enabled; drain/disable first",
        })
    }
}
impl Error for LearningAuditError {}

pub(super) struct LearningAudit {
    limit: usize,
    pub record_boundaries: bool,
    pub log: LearningAuditLog,
}
impl LearningAudit {
    pub fn new(limit: usize) -> Result<Self, LearningAuditError> {
        if !(1..=1_000_000).contains(&limit) {
            return Err(LearningAuditError::InvalidBudget);
        }
        Ok(Self {
            limit,
            record_boundaries: false,
            log: LearningAuditLog::default(),
        })
    }
    pub fn boundary(&mut self, time: SimTime) {
        if !self.record_boundaries {
            return;
        }
        if self.log.events.len() + self.log.delayed.len() + self.log.boundaries.len() < self.limit {
            self.log.boundaries.push(time);
        } else {
            self.log.dropped_events = self.log.dropped_events.saturating_add(1);
        }
    }
    pub fn record(&mut self, event: LearningAuditEvent) {
        if let LearningAuditEvent::Trace {
            synapse,
            phase: LearningPhase::Delayed,
            old_weight,
            new_weight,
            ..
        } = &event
        {
            if self.log.delayed.contains_key(synapse)
                || self.log.events.len() + self.log.delayed.len() + self.log.boundaries.len()
                    < self.limit
            {
                self.log.delayed.entry(*synapse).or_default().record(&event);
            } else {
                self.log.dropped_events = self.log.dropped_events.saturating_add(1);
            }
            if old_weight == new_weight {
                return;
            }
        }
        if self.log.events.len() + self.log.delayed.len() + self.log.boundaries.len() < self.limit {
            self.log.events.push(event);
        } else {
            self.log.dropped_events = self.log.dropped_events.saturating_add(1);
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct TraceState {
    id: SynapseId,
    tag: f32,
    anchor: Option<SimTime>,
    weight: Weight,
    residual: f64,
}
impl TraceState {
    pub fn read(edge: &Synapse) -> Self {
        Self {
            id: edge.id(),
            tag: edge.learning_eligibility(),
            anchor: edge.learning_eligibility_updated_at(),
            weight: edge.weight(),
            residual: edge.learning_weight_residual(),
        }
    }
    pub fn finish(
        self,
        edge: &Synapse,
        time: SimTime,
        phase: LearningPhase,
        held_signal: f32,
        delayed: Option<DelayedLearningUpdate>,
    ) -> Option<LearningAuditEvent> {
        if self.tag == 0.
            && edge.learning_eligibility() == 0.
            && self.weight == edge.weight()
            && self.residual == edge.learning_weight_residual()
        {
            return None;
        }
        Some(LearningAuditEvent::Trace {
            time,
            synapse: self.id,
            phase,
            tag_before: self.tag,
            tag_after: edge.learning_eligibility(),
            tag_anchor_before: self.anchor,
            held_signal,
            old_weight: self.weight,
            new_weight: edge.weight(),
            residual_before: self.residual,
            residual_after: edge.learning_weight_residual(),
            delayed,
        })
    }
}
