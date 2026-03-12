use jolt_common::constants::{RAM_START_ADDRESS, RAM_WORD_SIZE};
use jolt_common::jolt_device::MemoryLayout;
use jolt_core::zkvm::ram::remap_address;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use serde::{Deserialize, Serialize};
use tracer::emulator::memory::Memory;

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
    pub fn generate_secret_shares<R: rand::Rng + rand::CryptoRng>(
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

        let mut transposed: [Vec<Rep3RingShare<u64>>; 3] = std::array::from_fn(|_| Vec::with_capacity(share_len));
        for word_idx in 0..share_len {
            let address = word_idx as u64 * ws;
            let word = memory.read_bytes(address, ws);
            let [s0, s1, s2] = rep3_ring::share_ring_element_binary(rep3_ring::ring::ring_impl::RingElement(word), rng);
            transposed[0].push(s0);
            transposed[1].push(s1);
            transposed[2].push(s2);
        }

        transposed.into_iter().map(|data| Self { data }).collect()
    }
}
