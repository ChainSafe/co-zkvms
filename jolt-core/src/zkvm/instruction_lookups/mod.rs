use common::constants::XLEN;

pub mod booleanity;
pub mod hamming_weight;
pub mod ra_virtual;
pub mod read_raf_checking;

pub const LOG_K: usize = XLEN * 2;

#[cfg(any(feature = "rv64", not(feature = "fewer-phases")))]
pub const PHASES: usize = 8;
#[cfg(all(not(feature = "rv64"), feature = "fewer-phases"))]
pub const PHASES: usize = 4;

pub const LOG_M: usize = LOG_K / PHASES;
pub const M: usize = 1 << LOG_M;
pub const D: usize = 16;
pub const CHUNKS_PER_PHASE: usize = D / PHASES;
pub const LOG_K_CHUNK: usize = LOG_K / D;
pub const K_CHUNK: usize = 1 << LOG_K_CHUNK;

/// Computes the bit-length of the suffix, for the current (`j`th) round
/// of sumcheck.
pub fn current_suffix_len(j: usize) -> usize {
    LOG_K - (j / LOG_M + 1) * LOG_M
}
