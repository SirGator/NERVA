//! Passive, observable motor root.

use crate::core::SimTime;

use super::{Root, RootChannel, RootDirection, RootId};

/// One spike arriving at a motor-root channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotorOutput {
    /// Channel that received the spike.
    pub channel: u16,
    /// Arrival time after nerve conduction.
    pub at: SimTime,
    /// Transported amplitude.
    pub amplitude: f32,
}

/// Motor attachment that only records outputs; decoding remains separate.
#[derive(Clone, Debug, PartialEq)]
pub struct MotorRoot {
    root: Root,
    outputs: Vec<MotorOutput>,
}

impl MotorRoot {
    /// Builds a motor root from fixed channels.
    pub fn new(
        id: RootId,
        name: impl Into<String>,
        channels: Vec<RootChannel>,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            root: Root::new(id, name, RootDirection::Motor, channels)?,
            outputs: Vec::new(),
        })
    }

    /// Shared root metadata.
    pub fn root(&self) -> &Root {
        &self.root
    }

    /// Records a transported motor spike without interpreting it.
    pub fn observe(&mut self, output: MotorOutput) {
        self.outputs.push(output);
    }

    /// Recorded outputs in arrival order.
    pub fn outputs(&self) -> &[MotorOutput] {
        &self.outputs
    }

    /// Removes and returns all recorded outputs.
    pub fn drain_outputs(&mut self) -> Vec<MotorOutput> {
        std::mem::take(&mut self.outputs)
    }
}
