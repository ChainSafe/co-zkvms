#![allow(clippy::too_many_arguments)]

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::{
    constants::{BYTES_PER_INSTRUCTION, RAM_WORD_SIZE},
    jolt_device::MemoryLayout,
};

pub mod booleanity;
pub mod hamming_booleanity;
pub mod hamming_weight;
pub mod output_check;
pub mod ra_virtual;
pub mod raf_evaluation;
pub mod read_write_checking;
pub mod val_evaluation;

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
        let num_words =
            max_bytecode_address.next_multiple_of(ws) / ws - min_bytecode_address / ws + 1;
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

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use crate::utils::math::Math;
use tracer::JoltDevice;

pub fn build_initial_memory_state(
    ram_preprocessing: &RAMPreprocessing,
    program_io: &JoltDevice,
    K: usize,
) -> Vec<u64> {
    let memory_layout = &program_io.memory_layout;
    let mut initial_memory_state: Vec<u64> = vec![0; K];

    // Copy bytecode
    let mut index =
        remap_address(ram_preprocessing.min_bytecode_address, memory_layout).unwrap() as usize;
    for word in &ram_preprocessing.bytecode_words {
        initial_memory_state[index] = *word;
        index += 1;
    }

    // Copy trusted advice
    index = remap_address(memory_layout.trusted_advice_start, memory_layout).unwrap() as usize;
    for chunk in program_io.trusted_advice.chunks(8) {
        let mut word = [0u8; 8];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        initial_memory_state[index] = u64::from_le_bytes(word);
        index += 1;
    }

    // Copy untrusted advice
    index = remap_address(memory_layout.untrusted_advice_start, memory_layout).unwrap() as usize;
    for chunk in program_io.untrusted_advice.chunks(8) {
        let mut word = [0u8; 8];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        initial_memory_state[index] = u64::from_le_bytes(word);
        index += 1;
    }

    // Copy inputs
    index = remap_address(memory_layout.input_start, memory_layout).unwrap() as usize;
    for chunk in program_io.inputs.chunks(8) {
        let mut word = [0u8; 8];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        initial_memory_state[index] = u64::from_le_bytes(word);
        index += 1;
    }

    initial_memory_state
}

/// Compute the contribution of an advice region to the full initial memory MLE at `r_address`.
///
/// The advice polynomial covers addresses [start_index, start_index + 2^log_advice_size),
/// so its contribution to the full memory MLE evaluated at r_address is:
///   eq(r_address_high, binary(start_index >> log_advice_size)) * advice_eval
/// where advice_eval is the evaluation of the advice polynomial at the low bits of r_address.
pub fn calculate_advice_memory_evaluation<F: JoltField>(
    advice_opening: Option<(OpeningPoint<BIG_ENDIAN, F>, F)>,
    log_advice_size: usize,
    advice_start_address: u64,
    memory_layout: &MemoryLayout,
    r_address: &[F::Challenge],
    total_memory_vars: usize,
) -> F {
    let (_, advice_eval) = match advice_opening {
        Some(opening) => opening,
        None => return F::zero(),
    };

    let start_index = remap_address(advice_start_address, memory_layout).unwrap() as usize;
    let block_index = start_index >> log_advice_size;

    // The high bits of r_address select which block we're in
    let high_bits = total_memory_vars - log_advice_size;
    let r_high = &r_address[..high_bits];

    // Compute eq(r_high, binary(block_index))
    let block_bits: Vec<F::Challenge> = (0..high_bits)
        .map(|i| {
            if (block_index >> i) & 1 == 1 {
                F::Challenge::from(1u128)
            } else {
                F::Challenge::from(0u128)
            }
        })
        .collect();
    let eq_val = EqPolynomial::<F>::mle(&block_bits, r_high);

    eq_val * advice_eval
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
