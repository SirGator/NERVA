/// Local levels of neuromodulators affecting an area.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModulatorLevels {
    /// Dopamine level.
    pub dopamine: f32,
    /// Acetylcholine level.
    pub acetylcholine: f32,
    /// Noradrenaline level.
    pub noradrenaline: f32,
    /// Serotonin level.
    pub serotonin: f32,
}
