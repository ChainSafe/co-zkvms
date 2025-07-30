pub mod coordinator;
pub mod witness;
pub mod worker;

use crate::jolt::instruction::JoltInstructionSet;
use jolt_common::rv_trace::ELFInstruction;
use jolt_core::field::JoltField;

pub use jolt_core::jolt::vm::bytecode::BytecodeRow;

use jolt_tracer::RV32IM;
use rayon::prelude::*;

pub trait BytecodeRowExt {
    fn bitflags_ext<InstructionSet, F: JoltField>(instruction: &ELFInstruction) -> u64
    where
        InstructionSet: JoltInstructionSet<F>;

    fn from_instruction_ext<F: JoltField, InstructionSet>(instruction: &ELFInstruction) -> Self
    where
        InstructionSet: JoltInstructionSet<F>;
}

impl BytecodeRowExt for BytecodeRow {
    /// Packs the instruction's circuit flags and instruction flags into a single u64 bitvector.
    /// The layout is:
    ///     circuit flags || instruction flags
    /// where instruction flags is a one-hot bitvector corresponding to the instruction's
    /// index in the `InstructionSet` enum.
    fn bitflags_ext<InstructionSet, F: JoltField>(instruction: &ELFInstruction) -> u64
    where
        InstructionSet: JoltInstructionSet<F>,
    {
        let mut bitvector = 0;
        for flag in instruction.to_circuit_flags() {
            bitvector |= flag as u64;
            bitvector <<= 1;
        }

        // instruction flag
        if let Ok(jolt_instruction) = InstructionSet::try_from(instruction) {
            let instruction_index = InstructionSet::enum_index(&jolt_instruction);
            bitvector <<= instruction_index;
            bitvector |= 1;
            bitvector <<= InstructionSet::COUNT - instruction_index - 1;
        } else {
            bitvector <<= InstructionSet::COUNT - 1;
        }

        bitvector
    }

    fn from_instruction_ext<F: JoltField, InstructionSet>(instruction: &ELFInstruction) -> Self
    where
        InstructionSet: JoltInstructionSet<F>,
    {
        let imm = match instruction.opcode {
            RV32IM::LW
            | RV32IM::SW
            | RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU => instruction.imm.unwrap_or(0),
            _ => instruction.imm.unwrap_or(0) & u32::MAX as i64,
        };

        Self {
            address: instruction.address as usize,
            bitflags: Self::bitflags_ext::<InstructionSet, F>(instruction),
            rd: instruction.rd.unwrap_or(0) as u8,
            rs1: instruction.rs1.unwrap_or(0) as u8,
            rs2: instruction.rs2.unwrap_or(0) as u8,
            imm,
            virtual_sequence_remaining: instruction.virtual_sequence_remaining,
        }
    }
}
