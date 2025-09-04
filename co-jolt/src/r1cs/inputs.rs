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

use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker, WorkerIoContext,
};
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use rayon::prelude::*;

use crate::field::JoltField;
use crate::impl_r1cs_input_lc_conversions;
use crate::jolt::instruction::{JoltInstructionSet, Rep3JoltInstructionSet};
use crate::jolt::vm::read_write_memory::witness::Rep3ProgramIO;
use crate::jolt::vm::rv32i_vm::RV32I;
use crate::jolt::vm::witness::Rep3Polynomials;
use crate::jolt::vm::JoltTraceStep;
use crate::poly::{
    generate_poly_shares_rep3, generate_poly_shares_rep3_vec, Rep3MultilinearPolynomial,
};
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
    type PublicPolynomials = R1CSPolynomials<F>;

    #[tracing::instrument(
        skip_all,
        name = "Rep3R1CSPolynomials::stream_secret_shares",
        level = "trace"
    )]
    fn stream_secret_shares<R: rand::Rng, Network: Rep3NetworkCoordinator>(
        _preprocessing: &ConstantPreprocessing<C>,
        polynomials: Self::PublicPolynomials,
        rng: &mut R,
        network: &mut Network,
    ) -> eyre::Result<()> {
        let AuxVariableStuff {
            left_lookup_operand,
            right_lookup_operand,
            product,
            relevant_y_chunks,
            write_lookup_output_to_rd,
            write_pc_to_rd,
            next_pc_jump,
            should_branch,
            next_pc,
        } = polynomials.aux;

        let public_polys = (0..3)
            .map(|_| Rep3R1CSPolynomials {
                chunks_x: Default::default(),
                chunks_y: Default::default(),
                circuit_flags: Rep3MultilinearPolynomial::public_vec(
                    polynomials.circuit_flags.to_vec(),
                )
                .try_into()
                .unwrap(),
                aux: AuxVariableStuff {
                    left_lookup_operand: Default::default(),
                    right_lookup_operand: Default::default(),
                    product: Default::default(),
                    relevant_y_chunks: Default::default(),
                    write_lookup_output_to_rd: Rep3MultilinearPolynomial::public(
                        write_lookup_output_to_rd.clone(),
                    ),
                    write_pc_to_rd: Rep3MultilinearPolynomial::public(write_pc_to_rd.clone()),
                    next_pc_jump: Default::default(),
                    should_branch: Default::default(),
                    next_pc: Default::default(),
                },
            })
            .collect();
        network.send_requests(public_polys)?;
        let chunks_x_shares = generate_poly_shares_rep3_vec(&polynomials.chunks_x, rng);
        network.send_requests(chunks_x_shares)?;
        let chunks_y_shares = generate_poly_shares_rep3_vec(&polynomials.chunks_y, rng);
        network.send_requests(chunks_y_shares)?;
        let left_lookup_operand_shares = generate_poly_shares_rep3(&left_lookup_operand, rng);
        network.send_requests(left_lookup_operand_shares)?;
        let right_lookup_operand_shares = generate_poly_shares_rep3(&right_lookup_operand, rng);
        network.send_requests(right_lookup_operand_shares)?;
        let product_shares = generate_poly_shares_rep3(&product, rng);
        network.send_requests(product_shares)?;
        let relevant_y_chunks_shares = generate_poly_shares_rep3_vec(&relevant_y_chunks, rng);
        network.send_requests(relevant_y_chunks_shares)?;
        let next_pc_jump_shares = generate_poly_shares_rep3(&next_pc_jump, rng);
        network.send_requests(next_pc_jump_shares)?;
        let should_branch_shares = generate_poly_shares_rep3(&should_branch, rng);
        network.send_requests(should_branch_shares)?;
        let next_pc_shares = generate_poly_shares_rep3(&next_pc, rng);
        network.send_requests(next_pc_shares)?;

        Ok(())
    }

    #[tracing::instrument(
        skip_all,
        name = "Rep3R1CSPolynomials::receive_witness_share",
        level = "trace"
    )]
    fn receive_witness_share<Network: Rep3NetworkWorker>(
        _: &ConstantPreprocessing<C>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Self> {
        let mut partial_polys: Self = io_ctx.network().receive_request()?;
        partial_polys.chunks_x = io_ctx.network().receive_request()?;
        partial_polys.chunks_y = io_ctx.network().receive_request()?;
        partial_polys.aux.left_lookup_operand = io_ctx.network().receive_request()?;
        partial_polys.aux.right_lookup_operand = io_ctx.network().receive_request()?;
        partial_polys.aux.product = io_ctx.network().receive_request()?;
        partial_polys.aux.relevant_y_chunks = io_ctx.network().receive_request()?;
        partial_polys.aux.next_pc_jump = io_ctx.network().receive_request()?;
        partial_polys.aux.should_branch = io_ctx.network().receive_request()?;
        partial_polys.aux.next_pc = io_ctx.network().receive_request()?;

        Ok(partial_polys)
    }

    fn generate_witness_rep3<Instructions, Network>(
        _: &ConstantPreprocessing<C>,
        trace: &mut [JoltTraceStep<Instructions>],
        _: &Rep3ProgramIO<F>,
        M: usize,
        io_ctx: &mut WorkerIoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker,
    {
        let m = trace.len();
        let log_M = log2(M) as usize;

        let mut chunks_x = vec![Rep3RingShare::zero_share(); C * m];
        let mut chunks_y = vec![Rep3RingShare::zero_share(); C * m];
        let mut circuit_flags = vec![vec![0u8; NUM_CIRCUIT_FLAGS]; m];

        trace
            .into_par_iter()
            .zip(chunks_x.par_chunks_mut(C))
            .zip(chunks_y.par_chunks_mut(C))
            .zip(circuit_flags.par_iter_mut())
            .for_each(|(((step, chunks_x), chunks_y), circuit_flags)| {
                if let Some(instr) = &step.instruction_lookup {
                    let (x, y) = instr.operand_chunks_rep3(C, log_M);
                    for i in 0..C {
                        chunks_x[i] = x[i];

                        chunks_y[i] = y[i];
                    }
                }

                for j in 0..NUM_CIRCUIT_FLAGS {
                    if step.circuit_flags[j] {
                        circuit_flags[j] = 1;
                    }
                }
            });

        let chunks_x = transpose_par_from_flat(
            io_ctx.par_chunks(chunks_x, None, |chunk, io_ctx| {
                rep3_ring::casts::ring_to_field_many_selector(&chunk, io_ctx)
            })?,
            m,
            C,
        )
        .into_iter()
        .map(Rep3MultilinearPolynomial::from)
        .collect();

        let chunks_y = transpose_par_from_flat(
            io_ctx.par_chunks(chunks_y, None, |chunk, io_ctx| {
                rep3_ring::casts::ring_to_field_many_selector(&chunk, io_ctx)
            })?,
            m,
            C,
        )
        .into_iter()
        .map(Rep3MultilinearPolynomial::from)
        .collect();

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

    fn combine_polynomials(
        _preprocessing: &ConstantPreprocessing<C>,
        _: Vec<Self>,
    ) -> eyre::Result<Self::PublicPolynomials> {
        todo!()
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
