//! Shared types for coordinator↔worker communication.

use jolt_common::jolt_device::MemoryLayout;
use serde::{Deserialize, Serialize};
use tracer::instruction::Instruction;

/// Public data sent by workers to the coordinator at the start of each proof.
///
/// All 3 workers send identical copies (they all have the same public data).
/// The coordinator uses this to compute preprocessing and drive the proof.
///
/// NOTE: No advice or secret data is included — only public transcript fields.
#[derive(Serialize, Deserialize)]
pub struct ProofRequest {
    /// Decoded bytecode instructions (from `program.decode()`).
    pub bytecode: Vec<Instruction>,
    /// Initial memory state (from `program.decode()`).
    pub memory_init: Vec<(u64, u8)>,
    /// Padded trace length (next power of 2).
    pub padded_len: usize,
    /// RAM address space parameter.
    pub ram_k: usize,
    /// Memory layout (public, for preprocessing).
    pub memory_layout: MemoryLayout,
    /// Program inputs (public, for Fiat-Shamir transcript).
    pub inputs: Vec<u8>,
    /// Program outputs (public, for Fiat-Shamir transcript).
    pub outputs: Vec<u8>,
    /// Whether the guest panicked (public, for transcript).
    pub panic: bool,
}
