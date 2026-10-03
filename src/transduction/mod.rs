//! Device-neutral translation between receptor/effector signals and spikes.
//!
//! Queue timestamped inputs with `push`, then append output through an
//! inclusive simulation horizon with `advance_until`. The same traits support
//! activity or decay during intervals without new inputs.
//!
//! ```
//! use nerva::{
//!     io::{ChannelId, ReceptorSignal},
//!     primitives::SimTime,
//!     roots::MotorOutput,
//!     transduction::{
//!         DirectMotorTransducer, DirectSensoryTransducer,
//!         MotorTransducer, SensoryTransducer,
//!     },
//! };
//!
//! # fn main() -> Result<(), nerva::transduction::TransductionError> {
//! let mut sensory = DirectSensoryTransducer::new();
//! sensory.push(ReceptorSignal {
//!     channel: ChannelId(7), at: SimTime(100), value: 1.0,
//! })?;
//! let mut spikes = Vec::new();
//! sensory.advance_until(SimTime(99), &mut spikes)?;
//! assert!(spikes.is_empty());
//! sensory.advance_until(SimTime(100), &mut spikes)?;
//! assert_eq!(spikes[0].amplitude, 1.0);
//!
//! // MotorOutput normally comes from MotorRoot after neural activity and
//! // nerve routing. The direct decoder preserves its channel and time.
//! let mut motor = DirectMotorTransducer::new();
//! motor.push(MotorOutput {
//!     channel: ChannelId(9), at: SimTime(200), amplitude: 1.0,
//! })?;
//! let mut signals = Vec::new();
//! motor.advance_until(SimTime(200), &mut signals)?;
//! assert_eq!(signals[0].value, 1.0);
//! assert_eq!(signals[0].at, SimTime(200));
//! # Ok(())
//! # }
//! ```

mod direct;
mod error;
mod transducer;

pub use direct::{DirectMotorTransducer, DirectSensoryTransducer};
pub use error::TransductionError;
pub use transducer::{ChannelSpike, MotorTransducer, SensoryTransducer};
