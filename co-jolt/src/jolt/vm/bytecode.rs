pub mod witness;

use crate::jolt::instruction::JoltInstructionSet;
use jolt_core::utils::transcript::Transcript;
// use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
// use crate::poly::eq_poly::EqPolynomial;
// use jolt_core::field::JoltField;
// use jolt_core::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use jolt_common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS, REGISTER_COUNT};
use jolt_common::rv_trace::ELFInstruction;
use jolt_core::lasso::memory_checking::Initializable;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;

use jolt_core::field::JoltField;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
// use std::{collections::HashMap, marker::PhantomData};
use crate::lasso::memory_checking::{
    MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, StructuredPolynomialData,
    VerifierComputedOpening,
};

pub use jolt_core::jolt::vm::bytecode::BytecodeRow;

// use crate::{
//     poly::{
//         dense_mlpoly::DensePolynomial,
//         identity_poly::IdentityPolynomial,
//         structured_poly::{StructuredCommitment, StructuredOpeningProof},
//     },
//     utils::errors::ProofVerifyError,
// };

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// pub type BytecodeCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
//     BytecodeStuff<PCS::Commitment>;

// pub type BytecodeProof<F, C> = MemoryCheckingProof<
//     F,
//     C,
//     BytecodePolynomials<F, C>,
//     BytecodeReadWriteOpenings<F>,
//     BytecodeInitFinalOpenings<F>,
// >;

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
        Self {
            address: instruction.address as usize,
            bitflags: Self::bitflags_ext::<InstructionSet, F>(instruction),
            rd: instruction.rd.unwrap_or(0) as u8,
            rs1: instruction.rs1.unwrap_or(0) as u8,
            rs2: instruction.rs2.unwrap_or(0) as u8,
            imm: instruction.imm.unwrap_or(0) as i64, // imm is always cast to its 32-bit repr, signed or unsigned
            virtual_sequence_remaining: instruction.virtual_sequence_remaining,
        }
    }
}
