pub mod coordinator;
pub mod witness;
pub mod worker;

use std::{marker::PhantomData, sync::Arc};

use crate::{
    jolt::{
        trace::mem_op::MemoryOp,
        vm::{
            bytecode::witness::BytecodeRow,
            instruction_lookups::witness::InstructionLookupsPreprocessingExt,
        },
    },
    lasso::memory_checking::StructuredPolynomialData,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{ReducedOpeningProof, VerifierOpeningAccumulator},
    },
    r1cs::inputs::R1CSPreprocessing,
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::Context;
use jolt_common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{MemoryLayout, NUM_CIRCUIT_FLAGS},
};
use jolt_tracer::{ELFInstruction, JoltDevice};
use serde::{Deserialize, Serialize};
use strum::EnumCount;

use crate::field::JoltField;
use crate::jolt::{
    instruction::JoltInstructionSet, vm::instruction_lookups::InstructionLookupsProof,
};
use crate::r1cs::inputs::R1CSPolynomialsExt;
use jolt_core::{
    jolt::subtable::JoltSubtableSet,
    jolt::vm::{bytecode::BytecodePreprocessing, read_write_memory::ReadWriteMemoryPreprocessing},
    utils::transcript::AppendToTranscript,
};
use jolt_core::{
    jolt::{
        instruction::{
            div::DIVInstruction, divu::DIVUInstruction, lb::LBInstruction, lbu::LBUInstruction,
            lh::LHInstruction, lhu::LHUInstruction, mulh::MULHInstruction,
            mulhsu::MULHSUInstruction, rem::REMInstruction, remu::REMUInstruction,
            sb::SBInstruction, sh::SHInstruction, VirtualInstructionSequence,
        },
        vm::{
            bytecode::BytecodeProof,
            instruction_lookups::InstructionLookupsPreprocessing,
            read_write_memory::{ReadWriteMemoryPolynomials, ReadWriteMemoryProof},
            timestamp_range_check::TimestampValidityProof,
            JoltStuff, JoltVerifierPreprocessing,
        },
    },
    lasso::memory_checking::MemoryCheckingVerifier,
    poly::multilinear_polynomial::MultilinearPolynomial,
    r1cs::{
        constraints::R1CSConstraints,
        inputs::{ConstraintInput, R1CSPolynomials, R1CSProof},
        spartan::{self, UniformSpartanProof},
    },
};

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltWorkerPreprocessing<const C: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub generators: PCS::Setup,
    pub instruction_lookups: Arc<InstructionLookupsPreprocessingExt<C, F>>,
    pub bytecode: BytecodePreprocessing<F>,
    pub read_write_memory: ReadWriteMemoryPreprocessing,
    pub memory_layout: MemoryLayout,
    pub r1cs: R1CSPreprocessing<C>,
    pub field: F::SmallValueLookupTables,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoltTraceStep<InstructionSet: JoltInstructionSet> {
    pub instruction_lookup: Option<InstructionSet>,
    pub bytecode_row: BytecodeRow,
    pub memory_ops: [MemoryOp; MEMORY_OPS_PER_INSTRUCTION],
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
}

impl<InstructionSet: JoltInstructionSet> JoltTraceStep<InstructionSet> {
    fn no_op() -> Self {
        JoltTraceStep {
            instruction_lookup: None,
            bytecode_row: BytecodeRow::no_op(0),
            memory_ops: [
                MemoryOp::noop_read(),  // rs1
                MemoryOp::noop_read(),  // rs2
                MemoryOp::noop_write(), // rd is write-only
                MemoryOp::noop_read(),  // RAM
            ],
            circuit_flags: [false; NUM_CIRCUIT_FLAGS],
        }
    }

    pub fn pad(trace: &mut Vec<Self>) {
        let unpadded_length = trace.len();
        let padded_length = unpadded_length.next_power_of_two();
        trace.resize(padded_length, Self::no_op());
    }
}

type JoltTraceStepNative = jolt_core::jolt::vm::JoltTraceStep<jolt_core::jolt::vm::rv32i_vm::RV32I>;

impl<InstructionSet: JoltInstructionSet> Into<JoltTraceStepNative>
    for JoltTraceStep<InstructionSet>
{
    fn into(self) -> JoltTraceStepNative {
        jolt_core::jolt::vm::JoltTraceStep {
            instruction_lookup: None,
            bytecode_row: self.bytecode_row.into(),
            memory_ops: self.memory_ops.map(|op| op.into()),
            circuit_flags: self.circuit_flags,
        }
    }
}

// #[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProof<
    const C: usize,
    const M: usize,
    I,
    F,
    PCS,
    InstructionSet,
    Subtables,
    ProofTranscript,
> where
    I: ConstraintInput,
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    InstructionSet: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    pub trace_length: usize,
    pub bytecode: BytecodeProof<F, PCS, ProofTranscript>,
    pub read_write_memory: ReadWriteMemoryProof<F, PCS, ProofTranscript>,
    pub instruction_lookups:
        InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>,
    pub r1cs: UniformSpartanProof<C, I, F, ProofTranscript>,
    pub opening_proof: ReducedOpeningProof<F, PCS, ProofTranscript>,
    // pub opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript>,
    _marker: PhantomData<I>,
}

pub type JoltPolynomials<F> = JoltStuff<MultilinearPolynomial<F>>;

pub type JoltCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
    JoltStuff<PCS::Commitment>;

pub trait Jolt<F, PCS, const C: usize, const M: usize, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type InstructionSet: JoltInstructionSet;
    type Subtables: JoltSubtableSet<F>;
    type Constraints: R1CSConstraints<C, F>;

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn verifier_preprocess(
        bytecode: Vec<ELFInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_bytecode_size: usize,
        max_memory_size: usize,
        max_trace_length: usize,
    ) -> JoltVerifierPreprocessing<C, F, PCS, ProofTranscript> {
        // icicle::icicle_init();
        let instruction_lookups_preprocessing = InstructionLookupsProof::<
            C,
            M,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
            ProofTranscript,
        >::preprocess();

        let read_write_memory_preprocessing = ReadWriteMemoryPreprocessing::preprocess(memory_init);

        use jolt_tracer as tracer;
        let bytecode_rows: Vec<_> = bytecode
            .into_iter()
            .flat_map(|instruction| match instruction.opcode {
                tracer::RV32IM::MULH => MULHInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::MULHSU => MULHSUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::DIV => DIVInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::DIVU => DIVUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::REM => REMInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::REMU => REMUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::SH => SHInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::SB => SBInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LBU => LBUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LHU => LHUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LB => LBInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LH => LHInstruction::<32>::virtual_sequence(instruction),
                _ => vec![instruction],
            })
            .map(|instruction| {
                BytecodeRow::from_instruction::<Self::InstructionSet>(&instruction).into()
            })
            .collect();
        let bytecode_preprocessing = BytecodePreprocessing::<F>::preprocess(bytecode_rows);

        let max_poly_len: usize = [
            (max_bytecode_size + 1).next_power_of_two(), // Account for no-op prepended to bytecode
            max_trace_length.next_power_of_two(),
            max_memory_size.next_power_of_two(),
            M,
        ]
        .into_iter()
        .max()
        .unwrap();

        tracing::info!("max_poly_len: {:?}", max_poly_len);
        let generators = PCS::setup(max_poly_len);

        JoltVerifierPreprocessing {
            generators,
            memory_layout,
            instruction_lookups: instruction_lookups_preprocessing,
            bytecode: bytecode_preprocessing,
            read_write_memory: read_write_memory_preprocessing,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::generate_witness")]
    fn generate_witness(
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        trace: Vec<JoltTraceStep<Self::InstructionSet>>,
        program_io: &JoltDevice,
    ) -> JoltPolynomials<F> {
        let instruction_lookups =
            InstructionLookupsProof::<
                C,
                M,
                F,
                PCS,
                Self::InstructionSet,
                Self::Subtables,
                ProofTranscript,
            >::generate_witness(&preprocessing.instruction_lookups, &trace);

        let r1cs = R1CSPolynomials::generate_witness::<C, M, Self::InstructionSet>(&trace);

        let mut trace: Vec<JoltTraceStepNative> =
            trace.into_iter().map(|step| step.into()).collect();

        let read_write_memory = ReadWriteMemoryPolynomials::generate_witness(
            program_io,
            &preprocessing.read_write_memory,
            &trace,
        );
        let timestamp_range_check =
            TimestampValidityProof::<F, PCS, ProofTranscript>::generate_witness(&read_write_memory);

        let bytecode = BytecodeProof::<F, PCS, ProofTranscript>::generate_witness(
            &preprocessing.bytecode,
            &mut trace,
        );

        JoltPolynomials {
            instruction_lookups,
            read_write_memory,
            timestamp_range_check,
            r1cs,
            bytecode,
        }
    }

    #[tracing::instrument(skip_all)]
    fn verify(
        mut preprocessing: JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        proof: JoltProof<
            C,
            M,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
            ProofTranscript,
        >,
        commitments: JoltCommitments<PCS, ProofTranscript>,
        program_io: JoltDevice,
        // _debug_info: Option<ProverDebugInfo<F, ProofTranscript>>,
    ) -> eyre::Result<()> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();

        // opening_accumulator.compare_to(proof.opening_accumulator, &preprocessing.generators);

        // Self::fiat_shamir_preamble(
        //     &mut transcript,
        //     &program_io,
        //     &preprocessing.memory_layout,
        //     proof.trace_length,
        // );

        // Regenerate the uniform Spartan key
        let padded_trace_length = proof.trace_length.next_power_of_two();
        let memory_start = preprocessing.memory_layout.input_start;
        let r1cs_builder =
            Self::Constraints::construct_constraints(padded_trace_length, memory_start);
        let spartan_key = spartan::UniformSpartanProof::<C, _, F, ProofTranscript>::setup(
            &r1cs_builder,
            padded_trace_length,
        );
        transcript.append_scalar(&spartan_key.vk_digest);

        let r1cs_proof = R1CSProof {
            key: spartan_key,
            proof: proof.r1cs,
            _marker: PhantomData,
        };

        commitments
            .read_write_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));

        commitments
            .init_final_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));

        Self::verify_bytecode(
            &preprocessing.bytecode,
            &preprocessing.generators,
            proof.bytecode,
            &commitments,
            &mut opening_accumulator,
            &mut transcript,
        )?;

        Self::verify_instruction_lookups(
            &preprocessing.instruction_lookups,
            &preprocessing.generators,
            proof.instruction_lookups,
            &commitments,
            &mut opening_accumulator,
            &mut transcript,
        )
        .map_err(|e| eyre::eyre!(e))
        .context("failed to verify instruction lookups")?;

        Self::verify_memory(
            &mut preprocessing.read_write_memory,
            &preprocessing.generators,
            &preprocessing.memory_layout,
            proof.read_write_memory,
            &commitments,
            program_io,
            &mut opening_accumulator,
            &mut transcript,
        )?;

        Self::verify_r1cs(
            r1cs_proof,
            &commitments,
            &mut opening_accumulator,
            &mut transcript,
        )
        .map_err(|e| eyre::eyre!(e))
        .context("failed to verify r1cs")?;

        // Batch-verify all openings
        opening_accumulator
            .reduce_and_verify(
                &preprocessing.generators,
                &proof.opening_proof,
                &mut transcript,
            )
            .map_err(|e| eyre::eyre!(e))
            .context("failed to verify reduced openings")?;

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    fn verify_instruction_lookups<'a>(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        generators: &PCS::Setup,
        proof: InstructionLookupsProof<
            C,
            M,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
            ProofTranscript,
        >,
        commitments: &'a JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        InstructionLookupsProof::verify(
            preprocessing,
            generators,
            proof,
            commitments,
            opening_accumulator,
            transcript,
        )
    }

    #[tracing::instrument(skip_all)]
    fn verify_bytecode<'a>(
        preprocessing: &BytecodePreprocessing<F>,
        generators: &PCS::Setup,
        proof: BytecodeProof<F, PCS, ProofTranscript>,
        commitments: &'a JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        BytecodeProof::verify_memory_checking(
            preprocessing,
            generators,
            proof,
            &commitments.bytecode,
            commitments,
            opening_accumulator,
            transcript,
        )
    }

    // #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all)]
    fn verify_memory<'a>(
        preprocessing: &mut ReadWriteMemoryPreprocessing,
        generators: &PCS::Setup,
        memory_layout: &MemoryLayout,
        proof: ReadWriteMemoryProof<F, PCS, ProofTranscript>,
        commitment: &'a JoltCommitments<PCS, ProofTranscript>,
        program_io: JoltDevice,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        assert!(program_io.inputs.len() <= memory_layout.max_input_size as usize);
        assert!(program_io.outputs.len() <= memory_layout.max_output_size as usize);
        // pair the memory layout with the program io from the proof
        preprocessing.program_io = Some(JoltDevice {
            inputs: program_io.inputs,
            outputs: program_io.outputs,
            panic: program_io.panic,
            memory_layout: memory_layout.clone(),
        });

        ReadWriteMemoryProof::verify(
            proof,
            generators,
            preprocessing,
            commitment,
            opening_accumulator,
            transcript,
        )
    }

    #[tracing::instrument(skip_all)]
    fn verify_r1cs<'a>(
        proof: R1CSProof<
            C,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            ProofTranscript,
        >,
        commitments: &'a JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        proof
            .verify(commitments, opening_accumulator, transcript)
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))
    }

    fn fiat_shamir_preamble(
        transcript: &mut ProofTranscript,
        program_io: &JoltDevice,
        memory_layout: &MemoryLayout,
        trace_length: usize,
    ) {
        transcript.append_u64(trace_length as u64);
        transcript.append_u64(C as u64);
        transcript.append_u64(M as u64);
        transcript.append_u64(Self::InstructionSet::COUNT as u64);
        transcript.append_u64(Self::Subtables::COUNT as u64);
        transcript.append_u64(memory_layout.max_input_size);
        transcript.append_u64(memory_layout.max_output_size);
        transcript.append_bytes(&program_io.inputs);
        transcript.append_bytes(&program_io.outputs);
        transcript.append_u64(program_io.panic as u64);
    }
}
