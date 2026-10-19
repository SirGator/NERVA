//! Fixed lookup from root channels to fibers and back.

use std::collections::HashMap;

use crate::{
    core::NeuronId,
    io::ChannelId,
    roots::{Root, RootDirection, RootId},
};

use super::{Fiber, FiberDirection, FiberId};

/// A rejected or inconsistent nerve mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MappingError {
    /// A stable fiber ID was inserted twice.
    DuplicateFiber(FiberId),
    /// A root channel was mapped twice.
    DuplicateChannel {
        /// Root containing the duplicate channel.
        root: RootId,
        /// Duplicate channel identity.
        channel: ChannelId,
    },
    /// The requested fiber does not exist.
    UnknownFiber(FiberId),
    /// The requested fiber has the wrong transport direction.
    DirectionMismatch(FiberId),
    /// The requested channel is not declared by the supplied root.
    UnknownChannel {
        /// Root whose channel layout was checked.
        root: RootId,
        /// Undeclared channel identity.
        channel: ChannelId,
    },
    /// The supplied root has the wrong direction for the requested association.
    RootDirectionMismatch {
        /// Root whose direction was checked.
        root: RootId,
        /// Required direction.
        expected: RootDirection,
        /// Supplied direction.
        actual: RootDirection,
    },
}

/// Sole owner of physical root-channel associations and the fiber registry.
///
/// Each `(RootId, ChannelId)` identifies at most one fiber.
/// Roots declare channel existence; fibers own endpoints, delays, and gains.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mapping {
    fibers: HashMap<FiberId, Fiber>,
    sensory_channels: HashMap<(RootId, ChannelId), FiberId>,
    motor_channels: HashMap<NeuronId, Vec<(RootId, ChannelId, FiberId)>>,
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

    /// Connects one declared sensory root channel to a sensory fiber.
    ///
    /// Validates root direction and channel existence before changing the mapping.
    /// Root identities and channel layouts must remain stable after binding.
    pub fn map_sensory(
        &mut self,
        root: &Root,
        channel: ChannelId,
        fiber: FiberId,
    ) -> Result<(), MappingError> {
        Self::validate_root(root, channel, RootDirection::Sensory)?;
        let registered = self
            .fibers
            .get(&fiber)
            .ok_or(MappingError::UnknownFiber(fiber))?;
        if registered.direction != FiberDirection::Sensory {
            return Err(MappingError::DirectionMismatch(fiber));
        }
        if self.has_channel(root.id, channel) {
            return Err(MappingError::DuplicateChannel {
                root: root.id,
                channel,
            });
        }
        self.sensory_channels.insert((root.id, channel), fiber);
        Ok(())
    }

    /// Connects a motor neuron and declared motor root channel to a motor fiber.
    ///
    /// Validates root direction and channel existence before changing the mapping.
    /// Root identities and channel layouts must remain stable after binding.
    pub fn map_motor(
        &mut self,
        root: &Root,
        channel: ChannelId,
        fiber: FiberId,
    ) -> Result<(), MappingError> {
        Self::validate_root(root, channel, RootDirection::Motor)?;
        let registered = self
            .fibers
            .get(&fiber)
            .ok_or(MappingError::UnknownFiber(fiber))?;
        if registered.direction != FiberDirection::Motor {
            return Err(MappingError::DirectionMismatch(fiber));
        }
        if self.has_channel(root.id, channel) {
            return Err(MappingError::DuplicateChannel {
                root: root.id,
                channel,
            });
        }
        let mappings = self.motor_channels.entry(registered.neuron).or_default();
        mappings.push((root.id, channel, fiber));
        mappings.sort_by_key(|(root, channel, fiber)| (*root, *channel, *fiber));
        Ok(())
    }

    /// Resolves a sensory root channel.
    pub fn sensory_fiber(&self, root: RootId, channel: ChannelId) -> Option<&Fiber> {
        self.sensory_channels
            .get(&(root, channel))
            .and_then(|id| self.fibers.get(id))
    }

    /// Resolves all motor outputs of one neuron.
    pub fn motor_fibers(
        &self,
        neuron: NeuronId,
    ) -> impl Iterator<Item = (RootId, ChannelId, &Fiber)> {
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

    fn validate_root(
        root: &Root,
        channel: ChannelId,
        expected: RootDirection,
    ) -> Result<(), MappingError> {
        if root.direction != expected {
            return Err(MappingError::RootDirectionMismatch {
                root: root.id,
                expected,
                actual: root.direction,
            });
        }
        if !root
            .channels
            .iter()
            .any(|declared| declared.channel == channel)
        {
            return Err(MappingError::UnknownChannel {
                root: root.id,
                channel,
            });
        }
        Ok(())
    }

    fn has_channel(&self, root: RootId, channel: ChannelId) -> bool {
        self.sensory_channels.contains_key(&(root, channel))
            || self
                .motor_channels
                .values()
                .flatten()
                .any(|(mapped_root, mapped_channel, _)| {
                    *mapped_root == root && *mapped_channel == channel
                })
    }
}
