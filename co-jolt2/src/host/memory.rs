use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use tracer::emulator::memory::Memory;

use crate::utils::transpose;

/// Rep3 secret-shared main memory.
#[derive(Clone, Debug, Default)]
pub struct Rep3Memory {
    pub data: Vec<Rep3RingShare<u64>>,
}

impl Rep3Memory {
    /// Generate 3-party binary secret shares of the memory state.
    pub fn generate_secret_shares<R: rand::Rng>(memory: Memory, rng: &mut R) -> Vec<Self> {
        if memory.data.is_empty() {
            return vec![Self::default(); 3];
        }

        let shares_per_word: Vec<Vec<Rep3RingShare<u64>>> = memory
            .data
            .into_iter()
            .map(|word| rep3_ring::binary::generate_shares_rep3(word, rng))
            .collect();

        let transposed = transpose(shares_per_word);

        transposed.into_iter().map(|data| Self { data }).collect()
    }
}
