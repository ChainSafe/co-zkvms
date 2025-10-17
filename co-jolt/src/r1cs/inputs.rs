#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use jolt_core::jolt::vm::JoltStuff;
use jolt_core::lasso::memory_checking::Initializable;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::r1cs::inputs::{
    AuxVariable, AuxVariableStuff, ConstraintInput, R1CSPolynomials, R1CSStuff,
};

use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::impl_r1cs_input_lc_conversions;
use crate::jolt::instruction::{JoltInstructionSet, Rep3JoltInstructionSet};
use crate::jolt::vm::read_write_memory::witness::Rep3ProgramIO;
use crate::jolt::vm::rv32i_vm::RV32I;
use crate::jolt::vm::witness::Rep3Polynomials;
use crate::jolt::vm::JoltTraceStep;
use crate::poly::Rep3MultilinearPolynomial;
use crate::utils::future_ring::{FutureRep3Ring, Rep3RingFutureExt};
use crate::utils::{transpose, transpose_par_from_flat};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use jolt_common::rv_trace::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use std::fmt::Debug;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

pub struct ConstantPreprocessing<const C: usize>;

pub type Rep3R1CSPolynomials<F> = R1CSStuff<Rep3MultilinearPolynomial<F>>;

impl<const C: usize, F> Rep3Polynomials<F, ConstantPreprocessing<C>> for Rep3R1CSPolynomials<F>
where
    F: JoltField,
{
    #[cfg(feature = "debug")]
    type PublicPolynomials = R1CSPolynomials<F>;

    #[tracing::instrument(skip_all, name = "R1CS::generate_witness_rep3")]
    fn generate_witness_rep3<Instructions, Network>(
        _: &ConstantPreprocessing<C>,
        trace: &mut [JoltTraceStep<Instructions>],
        _: &Rep3ProgramIO<F>,
        M: usize,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker,
    {
        let m = trace.len();
        let log_M = log2(M) as usize;

        let mut chunks_x =
            vec![FutureRep3Ring::Ready(Rep3PrimeFieldShare::<F>::zero_share()); C * m];
        let mut chunks_y =
            vec![FutureRep3Ring::Ready(Rep3PrimeFieldShare::<F>::zero_share()); C * m];
        let mut circuit_flags = vec![vec![0u8; NUM_CIRCUIT_FLAGS]; m];

        let id = io_ctx.party_id();
        trace
            .into_par_iter()
            .zip(chunks_x.par_chunks_mut(C))
            .zip(chunks_y.par_chunks_mut(C))
            .zip(circuit_flags.par_iter_mut())
            .for_each(|(((step, chunks_x), chunks_y), circuit_flags)| {
                if let Some(instr) = &step.instruction_lookup {
                    let (x, y) = instr.operand_chunks_rep3(C, log_M, id);
                    for i in 0..C {
                        chunks_x[i] = FutureRep3Ring::cast_to_field_b2a(x[i]);
                        chunks_y[i] = FutureRep3Ring::cast_to_field_b2a(y[i]);
                    }
                }

                for j in 0..NUM_CIRCUIT_FLAGS {
                    if step.circuit_flags[j] {
                        circuit_flags[j] = 1;
                    }
                }
            });

        let _guard = tracing::trace_span!("cast_chunks_x").entered();
        let chunks_x = transpose_par_from_flat::<Rep3PrimeFieldShare<F>>(
            chunks_x.fulfill_batched(io_ctx, |res, _: ()| res)?,
            m,
            C,
        )
        .into_iter()
        .map(Rep3MultilinearPolynomial::from)
        .collect();
        drop(_guard);

        let _guard = tracing::trace_span!("cast_chunks_y").entered();

        let chunks_y = transpose_par_from_flat::<Rep3PrimeFieldShare<F>>(
            chunks_y.fulfill_batched(io_ctx, |res, _: ()| res)?,
            m,
            C,
        )
        .into_iter()
        .map(Rep3MultilinearPolynomial::from)
        .collect();
        drop(_guard);

        let circuit_flags = transpose(circuit_flags)
            .into_iter()
            .map(Rep3MultilinearPolynomial::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Ok(Self {
            chunks_x: chunks_x,
            chunks_y: chunks_y,
            circuit_flags: circuit_flags,
            // Actual aux variable polynomials will be computed afterwards
            aux: AuxVariableStuff::initialize(&C),
        })
    }

    #[cfg(feature = "debug")]
    fn combine_polynomials(
        _: &ConstantPreprocessing<C>,
        polynomials_shares: Vec<Self>,
    ) -> Self::PublicPolynomials {
        use itertools::{multizip, Itertools};

        let [share1, share2, share3] = polynomials_shares
            .try_into()
            .map_err(|_| "expected 3 shares")
            .unwrap();

        let chunks_x = multizip((share1.chunks_x, share2.chunks_x, share3.chunks_x))
            .map(|(p1, p2, p3)| Rep3MultilinearPolynomial::combine_shares(vec![p1, p2, p3]))
            .collect_vec();

        let chunks_y = multizip((share1.chunks_y, share2.chunks_y, share3.chunks_y))
            .map(|(p1, p2, p3)| Rep3MultilinearPolynomial::combine_shares(vec![p1, p2, p3]))
            .collect_vec();

        let circuit_flags = share1.circuit_flags.map(|p| p.try_into().unwrap());

        let mut aux = AuxVariableStuff::initialize(&C);

        aux.left_lookup_operand = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.aux.left_lookup_operand,
            share2.aux.left_lookup_operand,
            share3.aux.left_lookup_operand,
        ]);
        aux.right_lookup_operand = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.aux.right_lookup_operand,
            share2.aux.right_lookup_operand,
            share3.aux.right_lookup_operand,
        ]);
        aux.product = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.aux.product,
            share2.aux.product,
            share3.aux.product,
        ]);
        aux.relevant_y_chunks = multizip((
            share1.aux.relevant_y_chunks,
            share2.aux.relevant_y_chunks,
            share3.aux.relevant_y_chunks,
        ))
        .map(|(p1, p2, p3)| Rep3MultilinearPolynomial::combine_shares(vec![p1, p2, p3]))
        .collect_vec();
        aux.write_lookup_output_to_rd = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.aux.write_lookup_output_to_rd,
            share2.aux.write_lookup_output_to_rd,
            share3.aux.write_lookup_output_to_rd,
        ]);
        aux.write_pc_to_rd = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.aux.write_pc_to_rd,
            share2.aux.write_pc_to_rd,
            share3.aux.write_pc_to_rd,
        ]);
        aux.next_pc_jump = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.aux.next_pc_jump,
            share2.aux.next_pc_jump,
            share3.aux.next_pc_jump,
        ]);
        aux.should_branch = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.aux.should_branch,
            share2.aux.should_branch,
            share3.aux.should_branch,
        ]);
        aux.next_pc = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.aux.next_pc,
            share2.aux.next_pc,
            share3.aux.next_pc,
        ]);

        Self::PublicPolynomials {
            chunks_x,
            chunks_y,
            circuit_flags,
            aux,
        }
    }
}

pub trait R1CSPolynomialsExt<F: JoltField> {
    #[tracing::instrument(skip_all, name = "R1CSPolynomials::generate_witness")]
    fn generate_witness<const C: usize, const M: usize, InstructionSet: JoltInstructionSet>(
        trace: &[JoltTraceStep<InstructionSet>],
    ) -> R1CSPolynomials<F> {
        let log_M = log2(M) as usize;

        let mut chunks_x = vec![vec![0u8; trace.len()]; C];
        let mut chunks_y = vec![vec![0u8; trace.len()]; C];
        let mut circuit_flags = vec![vec![0u8; trace.len()]; NUM_CIRCUIT_FLAGS];

        for (step_index, step) in trace.iter().enumerate() {
            if let Some(instr) = &step.instruction_lookup {
                let (x, y) = instr.operand_chunks(C, log_M);
                for i in 0..C {
                    chunks_x[i][step_index] = x[i];
                    chunks_y[i][step_index] = y[i];
                }
            }

            for j in 0..NUM_CIRCUIT_FLAGS {
                if step.circuit_flags[j] {
                    circuit_flags[j][step_index] = 1;
                }
            }
        }

        R1CSPolynomials {
            chunks_x: chunks_x
                .into_iter()
                .map(MultilinearPolynomial::from)
                .collect(),
            chunks_y: chunks_y
                .into_iter()
                .map(MultilinearPolynomial::from)
                .collect(),
            circuit_flags: circuit_flags
                .into_iter()
                .map(MultilinearPolynomial::from)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            // Actual aux variable polynomials will be computed afterwards
            aux: AuxVariableStuff::initialize(&C),
        }
    }
}

impl<F: JoltField> R1CSPolynomialsExt<F> for R1CSPolynomials<F> {}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, PartialEq, EnumIter)]
pub enum JoltR1CSInputs {
    Bytecode_A, // Virtual address
    // Bytecode_V
    Bytecode_ELFAddress,
    Bytecode_Bitflags,
    Bytecode_RS1,
    Bytecode_RS2,
    Bytecode_RD,
    Bytecode_Imm,

    RAM_Address,
    RS1_Read,
    RS2_Read,
    RD_Read,
    RAM_Read,
    RD_Write,
    RAM_Write,

    ChunksQuery(usize),
    LookupOutput,
    ChunksX(usize),
    ChunksY(usize),

    OpFlags(CircuitFlags),
    InstructionFlags(RV32I),
    Aux(AuxVariable),
}

impl_r1cs_input_lc_conversions!(JoltR1CSInputs, 4);

impl ConstraintInput for JoltR1CSInputs {
    fn flatten<const C: usize>() -> Vec<Self> {
        JoltR1CSInputs::iter()
            .flat_map(|variant| match variant {
                Self::ChunksQuery(_) => (0..C).map(Self::ChunksQuery).collect(),
                Self::ChunksX(_) => (0..C).map(Self::ChunksX).collect(),
                Self::ChunksY(_) => (0..C).map(Self::ChunksY).collect(),
                Self::OpFlags(_) => CircuitFlags::iter().map(Self::OpFlags).collect(),
                Self::InstructionFlags(_) => RV32I::iter().map(Self::InstructionFlags).collect(),
                Self::Aux(_) => AuxVariable::iter()
                    .flat_map(|aux| match aux {
                        AuxVariable::RelevantYChunk(_) => (0..C)
                            .map(|i| Self::Aux(AuxVariable::RelevantYChunk(i)))
                            .collect(),
                        _ => vec![Self::Aux(aux)],
                    })
                    .collect(),
                _ => vec![variant],
            })
            .collect()
    }

    fn get_ref<'a, T: CanonicalSerialize + CanonicalDeserialize + Sync>(
        &self,
        jolt: &'a JoltStuff<T>,
    ) -> &'a T {
        let aux_polynomials = &jolt.r1cs.aux;
        match self {
            JoltR1CSInputs::Bytecode_A => &jolt.bytecode.a_read_write,
            JoltR1CSInputs::Bytecode_ELFAddress => &jolt.bytecode.v_read_write[0],
            JoltR1CSInputs::Bytecode_Bitflags => &jolt.bytecode.v_read_write[1],
            JoltR1CSInputs::Bytecode_RD => &jolt.bytecode.v_read_write[2],
            JoltR1CSInputs::Bytecode_RS1 => &jolt.bytecode.v_read_write[3],
            JoltR1CSInputs::Bytecode_RS2 => &jolt.bytecode.v_read_write[4],
            JoltR1CSInputs::Bytecode_Imm => &jolt.bytecode.v_read_write[5],
            JoltR1CSInputs::RAM_Address => &jolt.read_write_memory.a_ram,
            JoltR1CSInputs::RS1_Read => &jolt.read_write_memory.v_read_rs1,
            JoltR1CSInputs::RS2_Read => &jolt.read_write_memory.v_read_rs2,
            JoltR1CSInputs::RD_Read => &jolt.read_write_memory.v_read_rd,
            JoltR1CSInputs::RAM_Read => &jolt.read_write_memory.v_read_ram,
            JoltR1CSInputs::RD_Write => &jolt.read_write_memory.v_write_rd,
            JoltR1CSInputs::RAM_Write => &jolt.read_write_memory.v_write_ram,
            JoltR1CSInputs::ChunksQuery(i) => &jolt.instruction_lookups.dim[*i],
            JoltR1CSInputs::LookupOutput => &jolt.instruction_lookups.lookup_outputs,
            JoltR1CSInputs::ChunksX(i) => &jolt.r1cs.chunks_x[*i],
            JoltR1CSInputs::ChunksY(i) => &jolt.r1cs.chunks_y[*i],
            JoltR1CSInputs::OpFlags(i) => &jolt.r1cs.circuit_flags[*i as usize],
            JoltR1CSInputs::InstructionFlags(i) => {
                &jolt.instruction_lookups.instruction_flags
                    [<RV32I as JoltInstructionSet>::enum_index(i)]
            }
            Self::Aux(aux) => match aux {
                AuxVariable::LeftLookupOperand => &aux_polynomials.left_lookup_operand,
                AuxVariable::RightLookupOperand => &aux_polynomials.right_lookup_operand,
                AuxVariable::Product => &aux_polynomials.product,
                AuxVariable::RelevantYChunk(i) => &aux_polynomials.relevant_y_chunks[*i],
                AuxVariable::WriteLookupOutputToRD => &aux_polynomials.write_lookup_output_to_rd,
                AuxVariable::WritePCtoRD => &aux_polynomials.write_pc_to_rd,
                AuxVariable::NextPCJump => &aux_polynomials.next_pc_jump,
                AuxVariable::ShouldBranch => &aux_polynomials.should_branch,
                AuxVariable::NextPC => &aux_polynomials.next_pc,
            },
        }
    }

    fn get_ref_mut<'a, T: CanonicalSerialize + CanonicalDeserialize + Sync>(
        &self,
        jolt: &'a mut JoltStuff<T>,
    ) -> &'a mut T {
        let aux_polynomials = &mut jolt.r1cs.aux;
        match self {
            Self::Aux(aux) => match aux {
                AuxVariable::LeftLookupOperand => &mut aux_polynomials.left_lookup_operand,
                AuxVariable::RightLookupOperand => &mut aux_polynomials.right_lookup_operand,
                AuxVariable::Product => &mut aux_polynomials.product,
                AuxVariable::RelevantYChunk(i) => &mut aux_polynomials.relevant_y_chunks[*i],
                AuxVariable::WriteLookupOutputToRD => {
                    &mut aux_polynomials.write_lookup_output_to_rd
                }
                AuxVariable::WritePCtoRD => &mut aux_polynomials.write_pc_to_rd,
                AuxVariable::NextPCJump => &mut aux_polynomials.next_pc_jump,
                AuxVariable::ShouldBranch => &mut aux_polynomials.should_branch,
                AuxVariable::NextPC => &mut aux_polynomials.next_pc,
            },
            _ => panic!("get_ref_mut should only be invoked when computing aux polynomials"),
        }
    }
}
