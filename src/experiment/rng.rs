//! Small specified PRNG used to make M0 construction and shuffling replayable.

/// SplitMix64 with an algorithm fixed by this crate, independent of dependencies.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub(crate) fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let selected = (self.next_u64() % (index as u64 + 1)) as usize;
            values.swap(index, selected);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_is_identical_for_an_identical_seed() {
        let mut left = [1, 2, 3, 4, 5, 6];
        let mut right = left;
        DeterministicRng::new(42).shuffle(&mut left);
        DeterministicRng::new(42).shuffle(&mut right);
        assert_eq!(left, right);
    }
}
