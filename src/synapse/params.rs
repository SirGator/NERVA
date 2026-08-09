/// Fixed constraints and timing for a synapse.
pub struct SynapseParams {
    /// Inclusive lower weight bound.
    pub min_weight: f32,
    /// Inclusive upper weight bound.
    pub max_weight: f32,
    /// Delay between a presynaptic spike and target delivery in microseconds.
    pub transmission_delay_us: u64,
}
