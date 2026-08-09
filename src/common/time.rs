/// Monotonic simulation timestamp measured in microseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SimTimeUs(pub u64);

impl SimTimeUs {
    /// The beginning of simulation time.
    pub const ZERO: Self = Self(0);

    /// Returns a timestamp advanced by `duration_us`, saturating at `u64::MAX`.
    pub fn saturating_add_us(self, duration_us: u64) -> Self {
        Self(self.0.saturating_add(duration_us))
    }

    /// Returns elapsed microseconds since `earlier`, or `None` if time went backwards.
    pub fn duration_since(self, earlier: Self) -> Option<u64> {
        self.0.checked_sub(earlier.0)
    }
}
