use crate::{
    field::JoltField,
    jolt::{
        instruction::JoltInstructionSet,
        vm::{
            read_write_memory::witness::Rep3ProgramIO,
            witness::{Rep3Polynomials, WorkerInitializable},
        },
    },
    poly::Rep3MultilinearPolynomial,
    utils::{
        future_ring::{FutureRep3Ring, Rep3RingFutureExt},
        types::Either,
    },
};
use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS};
#[cfg(feature = "debug")]
use jolt_core::jolt::vm::bytecode::BytecodePolynomials;
use jolt_core::jolt::vm::bytecode::{BytecodePreprocessing, BytecodeStuff};
use jolt_tracer::{ELFInstruction, RV32IM};
use mpc_core::protocols::{
    rep3::{
        network::{IoContextPool, Rep3NetworkWorker},
        Rep3PrimeFieldShare,
    },
    rep3_ring::Rep3RingSignedShare,
};
use serde::{Deserialize, Serialize};

use rayon::prelude::*;
use snarks_core::math::Math;

pub type Rep3BytecodePolynomials<F> = BytecodeStuff<Rep3MultilinearPolynomial<F>>;

impl<F: JoltField> Rep3Polynomials<F, BytecodePreprocessing<F>> for Rep3BytecodePolynomials<F> {
    #[cfg(feature = "debug")]
    type PublicPolynomials = BytecodePolynomials<F>;

    #[tracing::instrument(skip_all, name = "Bytecode::generate_witness_rep3", level = "info")]
    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &BytecodePreprocessing<F>,
        trace: &mut [crate::jolt::vm::JoltTraceStep<Instructions>],
        _: &Rep3ProgramIO<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: crate::jolt::instruction::Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker,
    {
        let num_workers = io_ctx.num_workers();
        let worker_idx = io_ctx.worker_idx();

        let _half_net_log_num_workers = (num_workers >> 1).max(1).log_2();
        let half_net_worker_idx = worker_idx % (num_workers >> 1).max(1);

        let num_ops = trace.len();
        let num_ops_worker = num_ops / (num_workers >> 1).max(1); // when num_workers >= 4 worker generates sharded polys

        tracing::info!(
            "bytecode num_ops: {:?} num_ops_worker {} code_size {}",
            num_ops,
            num_ops_worker,
            preprocessing.code_size
        );

        let mut a_read_write: Vec<u32> = vec![0; num_ops];
        let mut read_cts: Vec<u32> = vec![0; num_ops];
        let mut final_cts: Vec<u32> = vec![0; preprocessing.code_size];

        let trace_range = if num_workers >= 4 {
            num_ops_worker * half_net_worker_idx..num_ops_worker * (half_net_worker_idx + 1)
        } else {
            0..num_ops
        };

        for (step_index, step) in trace.iter_mut().enumerate() {
            if !step.bytecode_row.address.is_zero() {
                assert!(step.bytecode_row.address >= RAM_START_ADDRESS as usize);
                assert!(step.bytecode_row.address % BYTES_PER_INSTRUCTION == 0);
                // Compress instruction address for more efficient commitment:
                step.bytecode_row.address = 1
                    + (step.bytecode_row.address - RAM_START_ADDRESS as usize)
                        / BYTES_PER_INSTRUCTION;
            }

            let virtual_address = preprocessing
                .virtual_address_map
                .get(&(
                    step.bytecode_row.address,
                    step.bytecode_row.virtual_sequence_remaining.unwrap_or(0),
                ))
                .unwrap();
            a_read_write[step_index] = *virtual_address as u32;
            let counter = final_cts[*virtual_address];
            read_cts[step_index] = counter;
            final_cts[*virtual_address] = counter + 1;
        }

        let mut address = vec![0; num_ops_worker];
        let mut bitflags = vec![0; num_ops_worker];
        let mut rd = vec![0; num_ops_worker];
        let mut rs1 = vec![0; num_ops_worker];
        let mut rs2 = vec![0; num_ops_worker];
        let mut imm =
            vec![FutureRep3Ring::Ready(Rep3PrimeFieldShare::<F>::zero_share()); num_ops_worker];

        trace[trace_range]
            .into_par_iter()
            .zip(address.par_iter_mut())
            .zip(bitflags.par_iter_mut())
            .zip(rd.par_iter_mut())
            .zip(rs1.par_iter_mut())
            .zip(rs2.par_iter_mut())
            .zip(imm.par_iter_mut())
            .for_each(|((((((step, addr), bit), rd), rs1), rs2), imm)| {
                *addr = step.bytecode_row.address as u64;
                *bit = step.bytecode_row.bitflags;
                *rd = step.bytecode_row.rd;
                *rs1 = step.bytecode_row.rs1;
                *rs2 = step.bytecode_row.rs2;
                *imm = match step.bytecode_row.imm {
                    Either::Shared(imm) => FutureRep3Ring::cast_to_field_signed_b2a(imm),
                    Either::Public(_) => {
                        // zero pad
                        FutureRep3Ring::Ready(Rep3PrimeFieldShare::<F>::zero_share())
                    }
                };
            });

        let _guard = tracing::trace_span!("cast_imm").entered();
        let imm: Vec<Rep3PrimeFieldShare<F>> = imm.fulfill_batched(io_ctx, |res, _: ()| res)?;
        drop(_guard);

        assert!(num_workers < 4, "unimplemented");

        // TODO: for worker < num_workers - 1 v_read_write[0] and a_read_write polys must have extra trace row for cross-step constraints

        let log_num_workers = io_ctx.log_num_workers();
        // TODO: when num_worker >= 4, shard ranges will be incorrect if we pass half_net_log_num_workers/half_net_worker_idx
        let v_read_write = [
            Rep3MultilinearPolynomial::new_shard_public_u64(
                address,
                num_ops,
                log_num_workers,
                worker_idx,
            ),
            Rep3MultilinearPolynomial::new_shard_public_u64(
                bitflags,
                num_ops,
                log_num_workers,
                worker_idx,
            ),
            Rep3MultilinearPolynomial::new_shard_public_u8(
                rd,
                num_ops,
                log_num_workers,
                worker_idx,
            ),
            Rep3MultilinearPolynomial::new_shard_public_u8(
                rs1,
                num_ops,
                log_num_workers,
                worker_idx,
            ),
            Rep3MultilinearPolynomial::new_shard_public_u8(
                rs2,
                num_ops,
                log_num_workers,
                worker_idx,
            ),
            Rep3MultilinearPolynomial::new_shard_shared(imm, num_ops, log_num_workers, worker_idx),
        ];
        let t_read = Rep3MultilinearPolynomial::new_shard_public_u32(
            read_cts,
            num_ops,
            log_num_workers,
            worker_idx,
        );
        let t_final = Rep3MultilinearPolynomial::new_shard_public_u32(
            final_cts,
            preprocessing.code_size,
            log_num_workers,
            worker_idx,
        );
        let a_read_write = Rep3MultilinearPolynomial::new_shard_public_u32(
            a_read_write,
            num_ops,
            log_num_workers,
            worker_idx,
        );

        Ok(Self {
            a_read_write,
            v_read_write,
            t_read,
            t_final,
            a_init_final: None,
            v_init_final: None,
        })
    }

    #[cfg(feature = "debug")]
    fn combine_polynomials(
        _: &BytecodePreprocessing<F>,
        polynomials_shares: Vec<Self>,
    ) -> Self::PublicPolynomials {
        Self::PublicPolynomials {
            a_read_write: polynomials_shares[0]
                .a_read_write
                .clone()
                .try_into()
                .unwrap(),
            v_read_write: [
                polynomials_shares[0].v_read_write[0]
                    .clone()
                    .try_into()
                    .unwrap(),
                polynomials_shares[0].v_read_write[1]
                    .clone()
                    .try_into()
                    .unwrap(),
                polynomials_shares[0].v_read_write[2]
                    .clone()
                    .try_into()
                    .unwrap(),
                polynomials_shares[0].v_read_write[3]
                    .clone()
                    .try_into()
                    .unwrap(),
                polynomials_shares[0].v_read_write[4]
                    .clone()
                    .try_into()
                    .unwrap(),
                Rep3MultilinearPolynomial::combine_shares(vec![
                    polynomials_shares[0].v_read_write[5].clone(),
                    polynomials_shares[1].v_read_write[5].clone(),
                    polynomials_shares[2].v_read_write[5].clone(),
                ]),
            ],
            t_read: polynomials_shares[0].t_read.clone().try_into().unwrap(),
            t_final: polynomials_shares[0].t_final.clone().try_into().unwrap(),
            a_init_final: None,
            v_init_final: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BytecodeRow {
    /// Memory address as read from the ELF.
    pub address: usize,
    /// Packed instruction/circuit flags, used for r1cs
    pub bitflags: u64,
    /// Index of the destination register for this instruction (0 if register is unused).
    pub rd: u8,
    /// Index of the first source register for this instruction (0 if register is unused).
    pub rs1: u8,
    /// Index of the second source register for this instruction (0 if register is unused).
    pub rs2: u8,
    /// "Immediate" value for this instruction (0 if unused).
    pub imm: Either<i64, Rep3RingSignedShare<u64>>,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl BytecodeRow {
    pub fn new(address: usize, bitflags: u64, rd: u8, rs1: u8, rs2: u8, imm: i64) -> Self {
        Self {
            address,
            bitflags,
            rd,
            rs1,
            rs2,
            imm: imm.into(),
            virtual_sequence_remaining: None,
        }
    }

    pub fn no_op(address: usize) -> Self {
        Self {
            address,
            bitflags: 0,
            rd: 0,
            rs1: 0,
            rs2: 0,
            imm: 0.into(),
            virtual_sequence_remaining: None,
        }
    }

    /// Packs the instruction's circuit flags and instruction flags into a single u64 bitvector.
    /// The layout is:
    ///     circuit flags || instruction flags
    /// where instruction flags is a one-hot bitvector corresponding to the instruction's
    /// index in the `InstructionSet` enum.
    fn bitflags<InstructionSet>(instruction: &ELFInstruction) -> u64
    where
        InstructionSet: JoltInstructionSet,
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

    pub fn from_instruction<InstructionSet>(instruction: &ELFInstruction) -> Self
    where
        InstructionSet: JoltInstructionSet,
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
            bitflags: Self::bitflags::<InstructionSet>(instruction),
            rd: instruction.rd.unwrap_or(0) as u8,
            rs1: instruction.rs1.unwrap_or(0) as u8,
            rs2: instruction.rs2.unwrap_or(0) as u8,
            imm: imm.into(),
            virtual_sequence_remaining: instruction.virtual_sequence_remaining,
        }
    }
}

impl Into<jolt_core::jolt::vm::bytecode::BytecodeRow> for BytecodeRow {
    fn into(self) -> jolt_core::jolt::vm::bytecode::BytecodeRow {
        jolt_core::jolt::vm::bytecode::BytecodeRow {
            address: self.address,
            bitflags: self.bitflags,
            rd: self.rd,
            rs1: self.rs1,
            rs2: self.rs2,
            imm: *self.imm.as_public(),
            virtual_sequence_remaining: self.virtual_sequence_remaining,
        }
    }
}

impl<F: JoltField, T: CanonicalSerialize + CanonicalDeserialize + Default>
    WorkerInitializable<T, BytecodePreprocessing<F>> for BytecodeStuff<T>
{
    type VerifierPreprocessing = BytecodePreprocessing<F>;

    fn worker_initialize(preprocessing: &BytecodePreprocessing<F>) -> Self {
        <Self as jolt_core::lasso::memory_checking::Initializable<T, BytecodePreprocessing<F>>>::initialize(preprocessing)
    }
}
