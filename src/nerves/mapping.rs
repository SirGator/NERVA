//! Fixed lookup from root channels to fibers and back.

use std::collections::HashMap;

use crate::{core::NeuronId, roots::RootId};

use super::{Fiber, FiberDirection, FiberId};

/// A rejected or inconsistent nerve mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MappingError {
    /// A stable fiber ID was inserted twice.
    DuplicateFiber(FiberId),
    /// A root channel was mapped twice.
    DuplicateChannel { root: RootId, channel: u16 },
    /// The requested fiber does not exist.
    UnknownFiber(FiberId),
    /// The requested fiber has the wrong transport direction.
    DirectionMismatch(FiberId),
}

/// Fiber registry and fixed root-channel mapping.
#[derive(Clone, Debug, Default)]
pub struct Mapping {
    fibers: HashMap<FiberId, Fiber>,
    sensory_channels: HashMap<(RootId, u16), FiberId>,
    motor_channels: HashMap<NeuronId, Vec<(RootId, u16, FiberId)>>,
}

impl Mapping {
    /// Creates an empty mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a fiber to the registry.
    pub fn add_fiber(&mut self, fiber: Fiber) -> Result<(), MappingError> {
        if self.fibers.contains_key(&fiber.id) {
            return Err(MappingError::DuplicateFiber(fiber.id));
        }
        self.fibers.insert(fiber.id, fiber);
        Ok(())
    }

    /// Connects one sensory root channel to a sensory fiber.
    pub fn map_sensory(
        &mut self,
        root: RootId,
        channel: u16,
        fiber: FiberId,
    ) -> Result<(), MappingError> {
        let registered = self
            .fibers
            .get(&fiber)
            .ok_or(MappingError::UnknownFiber(fiber))?;
        if registered.direction != FiberDirection::Sensory {
            return Err(MappingError::DirectionMismatch(fiber));
        }
        if self.sensory_channels.contains_key(&(root, channel)) {
            return Err(MappingError::DuplicateChannel { root, channel });
        }
        self.sensory_channels.insert((root, channel), fiber);
        Ok(())
    }

    /// Connects a motor neuron and root channel to a motor fiber.
    pub fn map_motor(
        &mut self,
        root: RootId,
        channel: u16,
        fiber: FiberId,
    ) -> Result<(), MappingError> {
        let registered = self
            .fibers
            .get(&fiber)
            .ok_or(MappingError::UnknownFiber(fiber))?;
        if registered.direction != FiberDirection::Motor {
            return Err(MappingError::DirectionMismatch(fiber));
        }
        let mappings = self.motor_channels.entry(registered.neuron).or_default();
        if mappings.iter().any(|(mapped_root, mapped_channel, _)| {
            *mapped_root == root && *mapped_channel == channel
        }) {
            return Err(MappingError::DuplicateChannel { root, channel });
        }
        mappings.push((root, channel, fiber));
        mappings.sort_by_key(|(root, channel, fiber)| (*root, *channel, *fiber));
        Ok(())
    }

    /// Resolves a sensory root channel.
    pub fn sensory_fiber(&self, root: RootId, channel: u16) -> Option<&Fiber> {
        self.sensory_channels
            .get(&(root, channel))
            .and_then(|id| self.fibers.get(id))
    }

    /// Resolves all motor outputs of one neuron.
    pub fn motor_fibers(&self, neuron: NeuronId) -> impl Iterator<Item = (RootId, u16, &Fiber)> {
        self.motor_channels
            .get(&neuron)
            .into_iter()
            .flatten()
            .filter_map(|(root, channel, fiber)| {
                self.fibers.get(fiber).map(|fiber| (*root, *channel, fiber))
            })
    }

    /// Returns one registered fiber.
    pub fn fiber(&self, id: FiberId) -> Option<&Fiber> {
        self.fibers.get(&id)
    }
}
