//! Shared inclusive event-time queue for transducers.

use std::collections::VecDeque;

use crate::primitives::SimTime;

use super::TransductionError;

#[derive(Clone, Debug)]
pub(super) struct Pending<T> {
    completed_until: Option<SimTime>,
    events: VecDeque<(SimTime, T)>,
}

impl<T> Default for Pending<T> {
    fn default() -> Self {
        Self {
            completed_until: None,
            events: VecDeque::new(),
        }
    }
}

impl<T> Pending<T> {
    pub(super) fn push(&mut self, at: SimTime, event: T) -> Result<(), TransductionError> {
        if let Some(completed_until) = self.completed_until
            && at <= completed_until
        {
            return Err(TransductionError::InputAlreadyProcessed {
                completed_until,
                at,
            });
        }
        let index = self.events.partition_point(|(time, _)| *time <= at);
        self.events.insert(index, (at, event));
        Ok(())
    }

    pub(super) fn advance_until(
        &mut self,
        until: SimTime,
        output: &mut Vec<T>,
    ) -> Result<(), TransductionError> {
        if let Some(current) = self.completed_until
            && until < current
        {
            return Err(TransductionError::TimeWentBackwards {
                current,
                requested: until,
            });
        }
        let due = self.events.partition_point(|(time, _)| *time <= until);
        output.extend(self.events.drain(..due).map(|(_, event)| event));
        self.completed_until = Some(until);
        Ok(())
    }
}
