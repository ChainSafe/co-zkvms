#![allow(clippy::too_many_arguments)]

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::{
    constants::{BYTES_PER_INSTRUCTION, RAM_WORD_SIZE},
    jolt_device::MemoryLayout,
};

pub mod booleanity;
pub mod hamming_booleanity;
pub mod hamming_weight;
pub mod ra_virtual;
pub mod raf_evaluation;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct RAMPreprocessing {
    pub min_bytecode_address: u64,
    pub bytecode_words: Vec<u64>,
}

impl RAMPreprocessing {
    pub fn preprocess(memory_init: Vec<(u64, u8)>) -> Self {
        let min_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .min()
            .unwrap_or(0);

        let max_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .max()
            .unwrap_or(0)
            + (BYTES_PER_INSTRUCTION as u64 - 1);

        let ws = RAM_WORD_SIZE;
        let num_words = max_bytecode_address.next_multiple_of(ws) / ws - min_bytecode_address / ws + 1;
        let mut bytecode_words = vec![0u64; num_words as usize];
        // Convert bytes into words and populate `bytecode_words`
        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / ws == address_b / ws)
        {
            let mut word = [0u8; 8];
            for (address, byte) in chunk {
                word[(address % ws) as usize] = *byte;
            }
            let word = u64::from_le_bytes(word);
            let remapped_index = (chunk[0].0 / ws - min_bytecode_address / ws) as usize;
            bytecode_words[remapped_index] = word;
        }

        Self {
            min_bytecode_address,
            bytecode_words,
        }
    }
}

/// Convert a chunk of bytes (up to RAM_WORD_SIZE bytes) into a u64 word (little-endian).
pub fn bytes_to_ram_word(bytes: &[u8]) -> u64 {
    let mut word = [0u8; 8];
    for (i, byte) in bytes.iter().enumerate() {
        word[i] = *byte;
    }
    u64::from_le_bytes(word)
}

/// Returns Some(address) if there was read/write
/// Returns None if there was no read/write
pub fn remap_address(address: u64, memory_layout: &MemoryLayout) -> Option<u64> {
    if address == 0 {
        return None;
    }

    if address >= memory_layout.trusted_advice_start {
        Some((address - memory_layout.trusted_advice_start) / common::constants::RAM_WORD_SIZE + 1)
    } else {
        panic!("Unexpected address {address}")
    }
}
