//! Small closed sensor/motor loop used after the open-loop M0 sequence test.

use crate::core::SimTime;

use super::{Action, Environment, Observation};

/// A deterministic one-bit world.
#[derive(Clone, Debug)]
pub struct BitWorld {
    value: bool,
    now: SimTime,
    observation_pending: bool,
}

impl BitWorld {
    /// Creates a world with one initial observation at time zero.
    pub fn new(initial: bool) -> Self {
        Self {
            value: initial,
            now: SimTime::ZERO,
            observation_pending: true,
        }
    }

    /// Current actuator/sensor state.
    pub fn value(&self) -> bool {
        self.value
    }

    /// Advances the timestamp used for the next changed-state observation.
    pub fn set_time(&mut self, now: SimTime) {
        self.now = self.now.max(now);
    }
}

impl Environment for BitWorld {
    fn next_observation_time(&self) -> Option<SimTime> {
        self.observation_pending.then_some(self.now)
    }

    fn observations(&mut self, until: SimTime) -> Vec<Observation> {
        if self.observation_pending && self.now <= until {
            self.observation_pending = false;
            vec![Observation::Bit {
                at: self.now,
                value: self.value,
            }]
        } else {
            Vec::new()
        }
    }

    fn apply_action(&mut self, action: Action) {
        if let Action::SetBit(value) = action
            && value != self.value
        {
            self.value = value;
            self.observation_pending = true;
        }
    }

    fn advance_to(&mut self, time: SimTime) {
        // A boundary adapter may revisit an equal timestamp while closing a
        // same-time batch, but it must never move the external clock backwards.
        self.now = self.now.max(time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changed_action_produces_a_timestamped_follow_up_observation() {
        let mut world = BitWorld::new(false);
        assert_eq!(
            world.observations(SimTime::ZERO),
            vec![Observation::Bit {
                at: SimTime::ZERO,
                value: false,
            }]
        );

        world.advance_to(SimTime(9));
        world.apply_action(Action::SetBit(true));

        assert_eq!(
            world.observations(SimTime(9)),
            vec![Observation::Bit {
                at: SimTime(9),
                value: true,
            }]
        );
    }

    #[test]
    fn adapter_clock_cannot_move_world_backwards() {
        let mut world = BitWorld::new(false);
        world.advance_to(SimTime(10));
        world.advance_to(SimTime(5));
        world.apply_action(Action::SetBit(true));

        assert!(world.observations(SimTime(9)).is_empty());
        assert_eq!(world.observations(SimTime(10)).len(), 1);
    }
}
