//! Fixed transport structures between roots and the neuronal core.

mod bundle;
mod fiber;
mod mapping;
mod routing;

pub use bundle::Bundle;
pub use fiber::{Fiber, FiberDirection, FiberId};
pub use mapping::{Mapping, MappingError};
pub use routing::{FiberImpulse, Routing, RoutingError};
