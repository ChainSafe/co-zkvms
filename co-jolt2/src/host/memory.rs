use jolt_common::constants::{RAM_START_ADDRESS, RAM_WORD_SIZE};
use jolt_common::jolt_device::MemoryLayout;
use jolt_core::zkvm::ram::remap_address;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use serde::{Deserialize, Serialize};
use tracer::emulator::memory::Memory;

use crate::utils::transpose;

/// Rep3 secret-shared main memory.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Rep3Memory {
    pub data: Vec<Rep3RingShare<u64>>,
}

impl Rep3Memory {
    /// Generate 3-party arithmetic secret shares of the memory state.
    ///
    /// Only the DRAM words that fall within the `ram_K`-length address space
    /// are secret-shared. The emulator allocates the full DRAM capacity
    /// (e.g. 128 MB = 16M u64 words), but `ram_K` is typically much smaller
    /// (derived from actual memory accesses). Trimming avoids expensive a2b
    /// conversion on millions of unused words during proving.
    ///
    /// The resulting `data` vector contains exactly the DRAM words that map
    /// to indices `[dram_start_index .. ram_K)` in the K-length address space.
    /// `Rep3RamDagWorker::new` overlays these onto the initial memory state.
    pub fn generate_secret_shares<R: rand::Rng>(
        memory: Memory,
        memory_layout: &MemoryLayout,
        ram_K: usize,
        rng: &mut R,
    ) -> Vec<Self> {
        if memory.data.is_empty() {
            return vec![Self::default(); 3];
        }

        let dram_start_index = remap_address(RAM_START_ADDRESS, memory_layout).unwrap() as usize;
        let dram_words_needed = ram_K.saturating_sub(dram_start_index);
        let ws = RAM_WORD_SIZE;
        let total_dram_words = (memory.data.len() as u64 * 8).div_ceil(ws) as usize;
        let share_len = dram_words_needed.min(total_dram_words);

        let shares_per_word: Vec<Vec<Rep3RingShare<u64>>> = (0..share_len)
            .map(|word_idx| {
                let address = word_idx as u64 * ws;
                let word = memory.read_bytes(address, ws);
                rep3_ring::binary::generate_shares_rep3(word, rng)
            })
            .collect();

        let transposed = transpose(shares_per_word);

        transposed.into_iter().map(|data| Self { data }).collect()
    }
}
