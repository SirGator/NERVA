//! Reproducible experiment entry point and record envelope.

use std::{error::Error, fmt};

use super::{
    Comparison, M0ExperimentConfig, M0StudyConfig, M0StudyReport, run_m0_comparison, run_m0_study,
};

/// Failure while validating, assembling or executing an experiment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExperimentError {
    context: &'static str,
    detail: String,
}

impl ExperimentError {
    /// Wraps an error while retaining which experiment stage rejected it.
    pub fn new(context: &'static str, detail: impl Into<String>) -> Self {
        Self {
            context,
            detail: detail.into(),
        }
    }

    /// Experiment stage that failed.
    pub fn context(&self) -> &'static str {
        self.context
    }

    /// Underlying human-readable reason.
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for ExperimentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.context, self.detail)
    }
}

impl Error for ExperimentError {}

/// Complete reproducible output envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentRecord {
    /// Validated configuration used for every group.
    pub config: M0ExperimentConfig,
    /// G1 through G4 results.
    pub comparison: Comparison,
}

/// Small orchestration object with no neuron or learning logic of its own.
#[derive(Clone, Debug)]
pub struct ExperimentRunner {
    config: M0ExperimentConfig,
}

impl ExperimentRunner {
    /// Validates and stores an immutable M0 configuration.
    pub fn new(config: M0ExperimentConfig) -> Result<Self, ExperimentError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Executes all controlled groups and returns a self-describing record.
    pub fn run(&self) -> Result<ExperimentRecord, ExperimentError> {
        Ok(ExperimentRecord {
            config: self.config.clone(),
            comparison: run_m0_comparison(&self.config)?,
        })
    }

    /// Executes paired comparisons for all configured study seeds.
    pub fn run_study(&self, study: &M0StudyConfig) -> Result<M0StudyReport, ExperimentError> {
        run_m0_study(&self.config, study)
    }

    /// Stored validated configuration.
    pub fn config(&self) -> &M0ExperimentConfig {
        &self.config
    }
}
