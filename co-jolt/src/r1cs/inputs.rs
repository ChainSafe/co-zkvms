#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use jolt_core::impl_r1cs_input_lc_conversions;
use jolt_core::jolt::vm::{JoltCommitments, JoltStuff};
use jolt_core::lasso::memory_checking::{Initializable, NoPreprocessing, StructuredPolynomialData};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::opening_proof::VerifierOpeningAccumulator;
use jolt_core::r1cs::inputs::{AuxVariable, AuxVariableStuff, ConstraintInput, R1CSPolynomials, R1CSStuff};
use jolt_core::utils::transcript::Transcript;

use jolt_core::r1cs::key::UniformSpartanKey;
use jolt_core::r1cs::spartan::{SpartanError, UniformSpartanProof};
use mpc_core::protocols::rep3::network::IoContext;

use crate::field::JoltField;
use crate::jolt::instruction::{JoltInstructionSet, Rep3JoltInstructionSet};
use crate::jolt::vm::rv32i_vm::RV32I;
use crate::jolt::vm::witness::Rep3Polynomials;
use crate::jolt::vm::JoltTraceStep;
use crate::poly::Rep3MultilinearPolynomial;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use jolt_common::rv_trace::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use std::fmt::Debug;
use std::marker::PhantomData;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

pub type Rep3R1CSPolynomials<F: JoltField> = R1CSStuff<Rep3MultilinearPolynomial<F>>;

pub struct R1CSPreprocessing<F: JoltField, const C: usize, I: ConstraintInput>(PhantomData<(F, I)>);

impl<F, const C: usize, I> Rep3Polynomials<F, R1CSPreprocessing<F, C, I>> for Rep3R1CSPolynomials<F>
where
    F: JoltField,
    I: ConstraintInput,
{
    type PublicPolynomials = R1CSPolynomials<F>;

    fn generate_secret_shares<R: rand::Rng>(
        _preprocessing: &R1CSPreprocessing<F, C, I>,
        polynomials: Self::PublicPolynomials,
        rng: &mut R,
    ) -> Vec<Self> {
        todo!()
    }

    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &R1CSPreprocessing<F, C, I>,
        trace: &mut [JoltTraceStep<F, Instructions>],
        M: usize,
        network: IoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
        Network: mpc_core::protocols::rep3::network::Rep3Network,
    {
        todo!()
    }

    fn combine_polynomials(
        preprocessing: &R1CSPreprocessing<F, C, I>,
        polynomials_shares: Vec<Self>,
    ) -> eyre::Result<Self::PublicPolynomials> {
        todo!()
    }
}

pub trait R1CSPolynomialsExt<F: JoltField> {
    fn new<const C: usize, const M: usize, InstructionSet: JoltInstructionSet<F>>(
        trace: &[JoltTraceStep<F, InstructionSet>],
    ) -> R1CSPolynomials<F> {
        let log_M = log2(M) as usize;

        let mut chunks_x = vec![vec![0u8; trace.len()]; C];
        let mut chunks_y = vec![vec![0u8; trace.len()]; C];
        let mut circuit_flags = vec![vec![0u8; trace.len()]; NUM_CIRCUIT_FLAGS];

        // TODO(moodlezoup): Can be parallelized
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
pub enum JoltR1CSInputs<F: JoltField> {
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
    InstructionFlags(RV32I<F>),
    Aux(AuxVariable),
}


// impl_r1cs_input_lc_conversions!(JoltR1CSInputs<F>, 4);

impl<F: JoltField> ConstraintInput for JoltR1CSInputs<F> {
    fn flatten<const C: usize>() -> Vec<Self> {
        JoltR1CSInputs::iter()
            .flat_map(|variant| match variant {
                Self::ChunksQuery(_) => (0..C).map(Self::ChunksQuery).collect(),
                Self::ChunksX(_) => (0..C).map(Self::ChunksX).collect(),
                Self::ChunksY(_) => (0..C).map(Self::ChunksY).collect(),
                Self::OpFlags(_) => CircuitFlags::iter().map(Self::OpFlags).collect(),
                Self::InstructionFlags(_) => RV32I::<F>::iter().map(Self::InstructionFlags).collect(),
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
                &jolt.instruction_lookups.instruction_flags[<RV32I::<F> as JoltInstructionSet<F>>::enum_index(i)]
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
