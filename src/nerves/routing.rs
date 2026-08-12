//! Pure conversion between channel spikes and delayed fiber impulses.

use std::{error::Error, fmt};

use crate::{
    core::{NeuronId, SimTime},
    roots::RootId,
    transduction::ChannelSpike,
};

use super::{FiberId, Mapping};

/// A fixed nerve impulse could not be represented safely.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoutingError {
    /// Adding the fiber delay would exceed [`SimTime`].
    TimeOverflow {
        /// Fiber whose delay was applied.
        fiber: FiberId,
        /// Timestamp at the transmitting endpoint.
        departed_at: SimTime,
        /// Configured delay that did not fit.
        delay_us: u64,
    },
    /// Input amplitude or fixed-gain multiplication was not finite and positive.
    InvalidAmplitude {
        /// Fiber carrying the rejected impulse.
        fiber: FiberId,
        /// Rejected transported amplitude.
        amplitude: f32,
    },
}

impl fmt::Display for RoutingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimeOverflow {
                fiber,
                departed_at,
                delay_us,
            } => write!(
                formatter,
                "adding fiber {fiber:?} delay {delay_us} us to departure time {departed_at} overflows simulation time"
            ),
            Self::InvalidAmplitude { fiber, amplitude } => write!(
                formatter,
                "fiber {fiber:?} amplitude must be finite and greater than zero, got {amplitude}"
            ),
        }
    }
}

impl Error for RoutingError {}

/// A spike transported along one fixed nerve fiber.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiberImpulse {
    /// Fiber carrying the impulse.
    pub fiber: FiberId,
    /// Core endpoint.
    pub neuron: NeuronId,
    /// Root endpoint and its channel.
    pub root: RootId,
    /// Root channel number.
    pub channel: u16,
    /// Arrival timestamp at the receiving endpoint.
    pub arrives_at: SimTime,
    /// Positive unsigned amplitude. Neuronal polarity is a core concern.
    pub amplitude: f32,
}

/// Stateless nerve routing helpers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Routing;

impl Routing {
    /// Routes an encoded root-channel spike toward its mapped core neuron.
    pub fn sensory(
        mapping: &Mapping,
        root: RootId,
        spike: ChannelSpike,
    ) -> Result<Option<FiberImpulse>, RoutingError> {
        let Some(fiber) = mapping.sensory_fiber(root, spike.channel) else {
            return Ok(None);
        };
        let arrives_at =
            spike
                .at
                .checked_add_us(fiber.delay_us)
                .ok_or(RoutingError::TimeOverflow {
                    fiber: fiber.id,
                    departed_at: spike.at,
                    delay_us: fiber.delay_us,
                })?;
        let amplitude = spike.amplitude * fiber.gain;
        validate_amplitude(fiber.id, amplitude)?;

        Ok(Some(FiberImpulse {
            fiber: fiber.id,
            neuron: fiber.neuron,
            root,
            channel: spike.channel,
            arrives_at,
            amplitude,
        }))
    }

    /// Routes one core spike to every mapped motor channel.
    pub fn motor(
        mapping: &Mapping,
        neuron: NeuronId,
        emitted_at: SimTime,
    ) -> Result<Vec<FiberImpulse>, RoutingError> {
        mapping
            .motor_fibers(neuron)
            .map(|(root, channel, fiber)| {
                let arrives_at = emitted_at.checked_add_us(fiber.delay_us).ok_or(
                    RoutingError::TimeOverflow {
                        fiber: fiber.id,
                        departed_at: emitted_at,
                        delay_us: fiber.delay_us,
                    },
                )?;
                validate_amplitude(fiber.id, fiber.gain)?;
                Ok(FiberImpulse {
                    fiber: fiber.id,
                    neuron,
                    root,
                    channel,
                    arrives_at,
                    amplitude: fiber.gain,
                })
            })
            .collect()
    }
}

fn validate_amplitude(fiber: FiberId, amplitude: f32) -> Result<(), RoutingError> {
    if amplitude.is_finite() && amplitude > 0.0 {
        Ok(())
    } else {
        Err(RoutingError::InvalidAmplitude { fiber, amplitude })
    }
}

#[cfg(test)]
mod tests {
    use crate::{core::NeuronId, roots::RootId, transduction::ChannelSpike};

    use super::*;
    use crate::nerves::{Fiber, FiberDirection};

    #[test]
    fn sensory_routing_applies_only_fixed_delay_and_gain() {
        let mut mapping = Mapping::new();
        mapping
            .add_fiber(
                Fiber::new(FiberId(1), FiberDirection::Sensory, NeuronId(7), 5, 2.0).unwrap(),
            )
            .unwrap();
        mapping.map_sensory(RootId(3), 4, FiberId(1)).unwrap();

        let impulse = Routing::sensory(
            &mapping,
            RootId(3),
            ChannelSpike {
                channel: 4,
                at: SimTime(10),
                amplitude: 0.5,
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(impulse.neuron, NeuronId(7));
        assert_eq!(impulse.arrives_at, SimTime(15));
        assert_eq!(impulse.amplitude, 1.0);
    }

    #[test]
    fn sensory_routing_reports_timestamp_overflow() {
        let mut mapping = Mapping::new();
        mapping
            .add_fiber(
                Fiber::new(FiberId(1), FiberDirection::Sensory, NeuronId(7), 2, 1.0).unwrap(),
            )
            .unwrap();
        mapping.map_sensory(RootId(3), 4, FiberId(1)).unwrap();

        assert_eq!(
            Routing::sensory(
                &mapping,
                RootId(3),
                ChannelSpike {
                    channel: 4,
                    at: SimTime(u64::MAX - 1),
                    amplitude: 1.0,
                },
            ),
            Err(RoutingError::TimeOverflow {
                fiber: FiberId(1),
                departed_at: SimTime(u64::MAX - 1),
                delay_us: 2,
            })
        );
    }

    #[test]
    fn motor_routing_reports_timestamp_overflow() {
        let mut mapping = Mapping::new();
        mapping
            .add_fiber(Fiber::new(FiberId(1), FiberDirection::Motor, NeuronId(7), 2, 1.0).unwrap())
            .unwrap();
        mapping.map_motor(RootId(3), 1, FiberId(1)).unwrap();

        assert_eq!(
            Routing::motor(&mapping, NeuronId(7), SimTime(u64::MAX - 1)),
            Err(RoutingError::TimeOverflow {
                fiber: FiberId(1),
                departed_at: SimTime(u64::MAX - 1),
                delay_us: 2,
            })
        );
    }

    #[test]
    fn sensory_routing_rejects_non_finite_fixed_gain_product() {
        let mut mapping = Mapping::new();
        mapping
            .add_fiber(
                Fiber::new(
                    FiberId(1),
                    FiberDirection::Sensory,
                    NeuronId(7),
                    1,
                    f32::MAX,
                )
                .unwrap(),
            )
            .unwrap();
        mapping.map_sensory(RootId(3), 4, FiberId(1)).unwrap();

        assert!(matches!(
            Routing::sensory(
                &mapping,
                RootId(3),
                ChannelSpike {
                    channel: 4,
                    at: SimTime::ZERO,
                    amplitude: f32::MAX,
                },
            ),
            Err(RoutingError::InvalidAmplitude { .. })
        ));
    }
}
