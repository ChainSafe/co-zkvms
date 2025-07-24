pub mod coordinator;
pub mod witness;
pub mod worker;

use std::{collections::BTreeMap, iter, marker::PhantomData};

use crate::{
    jolt::vm::bytecode::BytecodeRowExt,
    lasso::memory_checking::StructuredPolynomialData,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{
            ProverOpeningAccumulator, ReducedOpeningProof, VerifierOpeningAccumulator,
        },
    },
    utils::{errors::ProofVerifyError, thread::drop_in_background_thread, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{MemoryLayout, MemoryOp, NUM_CIRCUIT_FLAGS},
};
use jolt_tracer::{ELFInstruction, JoltDevice};
use serde::{Deserialize, Serialize};
use snarks_core::math::Math;
use strum::EnumCount;

use super::bytecode::BytecodeRow;
use crate::jolt::{
    instruction::JoltInstructionSet,
    subtable::JoltSubtableSet,
    vm::{instruction_lookups::InstructionLookupsProof, rv32i_vm::RV32I},
};
use crate::r1cs::inputs::R1CSPolynomialsExt;
use jolt_core::{
    field::JoltField,
    jolt::{instruction::VirtualInstructionSequence, vm::{
        bytecode::BytecodeProof, instruction_lookups::InstructionLookupsPreprocessing, read_write_memory::ReadWriteMemoryPolynomials, timestamp_range_check::TimestampValidityProof, JoltProverPreprocessing, JoltStuff, JoltVerifierPreprocessing
    }},
    poly::multilinear_polynomial::MultilinearPolynomial,
    r1cs::{
        constraints::R1CSConstraints,
        inputs::{ConstraintInput, R1CSPolynomials},
        spartan::{self, UniformSpartanProof},
    },
};
use jolt_core::{
    jolt::vm::{bytecode::BytecodePreprocessing, read_write_memory::ReadWriteMemoryPreprocessing},
    utils::transcript::AppendToTranscript,
};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoltTraceStep<F: JoltField, InstructionSet: JoltInstructionSet<F>> {
    pub instruction_lookup: Option<InstructionSet>,
    pub bytecode_row: BytecodeRow,
    pub memory_ops: [MemoryOp; MEMORY_OPS_PER_INSTRUCTION],
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
    pub(crate) _field: PhantomData<F>,
}

impl<F: JoltField, InstructionSet: JoltInstructionSet<F>> JoltTraceStep<F, InstructionSet> {
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
            _field: PhantomData,
        }
    }

    fn pad(trace: &mut Vec<Self>) {
        let unpadded_length = trace.len();
        let padded_length = unpadded_length.next_power_of_two();
        trace.resize(padded_length, Self::no_op());
    }
}

type JoltTraceStepNative = jolt_core::jolt::vm::JoltTraceStep<jolt_core::jolt::vm::rv32i_vm::RV32I>;

impl<F: JoltField, InstructionSet: JoltInstructionSet<F>> Into<JoltTraceStepNative>
    for JoltTraceStep<F, InstructionSet>
{
    fn into(self) -> JoltTraceStepNative {
        jolt_core::jolt::vm::JoltTraceStep {
            instruction_lookup: None,
            bytecode_row: self.bytecode_row,
            memory_ops: self.memory_ops,
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
    InstructionSet: JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    pub trace_length: usize,
    // pub bytecode: BytecodeProof<F, PCS, ProofTranscript>,
    // pub read_write_memory: ReadWriteMemoryProof<F, PCS, ProofTranscript>,
    pub instruction_lookups:
        InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>,
    pub r1cs: UniformSpartanProof<C, I, F, ProofTranscript>,
    pub opening_proof: ReducedOpeningProof<F, PCS, ProofTranscript>,
    // pub opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript>,
}

pub type JoltPolynomials<F: JoltField> = JoltStuff<MultilinearPolynomial<F>>;

pub type JoltCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
    JoltStuff<PCS::Commitment>;

pub trait Jolt<F, PCS, const C: usize, const M: usize, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type InstructionSet: JoltInstructionSet<F>;
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

        use jolt_core::jolt::instruction;
        use jolt_tracer as tracer;
        let bytecode_rows: Vec<BytecodeRow> = bytecode
            .into_iter()
            .flat_map(|instruction| match instruction.opcode {
                tracer::RV32IM::MULH => instruction::mulh::MULHInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::MULHSU => instruction::mulhsu::MULHSUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::DIV => instruction::div::DIVInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::DIVU => instruction::divu::DIVUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::REM => instruction::rem::REMInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::REMU => instruction::remu::REMUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::SH => instruction::sh::SHInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::SB => instruction::sb::SBInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LBU => instruction::lbu::LBUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LHU => instruction::lhu::LHUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LB => instruction::lb::LBInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LH => instruction::lh::LHInstruction::<32>::virtual_sequence(instruction),
                _ => vec![instruction],
            })
            .map(|instruction| {
                BytecodeRow::from_instruction_ext::<F, Self::InstructionSet>(&instruction)
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

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn prover_preprocess(
        bytecode: Vec<ELFInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_bytecode_size: usize,
        max_memory_size: usize,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<C, F, PCS, ProofTranscript> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());
        let shared = Self::verifier_preprocess(
            bytecode,
            memory_layout,
            memory_init,
            max_bytecode_size,
            max_memory_size,
            max_trace_length,
        );
        JoltProverPreprocessing {
            shared,
            field: small_value_lookup_tables,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::generate_witness")]
    fn generate_witness(
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        trace: Vec<JoltTraceStep<F, Self::InstructionSet>>,
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

        let mut trace: Vec<JoltTraceStepNative> = trace.into_iter().map(|step| step.into()).collect();

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

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    fn prove(
        program_io: JoltDevice,
        mut trace: Vec<JoltTraceStep<F, Self::InstructionSet>>,
        preprocessing: JoltProverPreprocessing<C, F, PCS, ProofTranscript>,
    ) -> (
        JoltProof<
            C,
            M,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
            ProofTranscript,
        >,
        JoltCommitments<PCS, ProofTranscript>,
        // JoltDevice,
        // Option<ProverDebugInfo<F, ProofTranscript>>,
    ) {
        // icicle::icicle_init();
        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();
        let srs_size = PCS::srs_size(&preprocessing.shared.generators);
        let padded_log2 = padded_trace_length.log_2();
        let srs_log2 = srs_size.log_2();

        // println!(
        //     "Trace length: {trace_length} (2^{})",
        //     trace_length.next_power_of_two().log_2()
        // );

        if padded_trace_length > srs_size {
            panic!(
                "Padded trace length {padded_trace_length} (2^{padded_log2}) exceeds SRS size {srs_size} (2^{srs_log2}). Consider increasing the max_trace_length."
            );
        }

        // F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));

        // TODO(moodlezoup): Truncate generators

        // TODO(JP): Drop padding on number of steps
        JoltTraceStep::pad(&mut trace);

        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        // Self::fiat_shamir_preamble(
        //     &mut transcript,
        //     &program_io,
        //     &program_io.memory_layout,
        //     trace_length,
        // );

        let instruction_polynomials =
            InstructionLookupsProof::<
                C,
                M,
                F,
                PCS,
                Self::InstructionSet,
                Self::Subtables,
                ProofTranscript,
            >::generate_witness(&preprocessing.shared.instruction_lookups, &trace);

        // let memory_polynomials = ReadWriteMemoryPolynomials::generate_witness(
        //     &program_io,
        //     &preprocessing.shared.read_write_memory,
        //     &trace,
        // );

        // let (bytecode_polynomials, range_check_polys) = rayon::join(
        //     || {
        //         BytecodeProof::<F, PCS, ProofTranscript>::generate_witness(
        //             &preprocessing.shared.bytecode,
        //             &mut trace,
        //         )
        //     },
        //     || {
        //         TimestampValidityProof::<F, PCS, ProofTranscript>::generate_witness(
        //             &memory_polynomials,
        //         )
        //     },
        // );

        let r1cs_builder = Self::Constraints::construct_constraints(
            padded_trace_length,
            program_io.memory_layout.input_start,
        );
        let spartan_key = spartan::UniformSpartanProof::<
            C,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            ProofTranscript,
        >::setup(&r1cs_builder, padded_trace_length);

        let r1cs_polynomials =
            R1CSPolynomials::generate_witness::<C, M, Self::InstructionSet>(&trace);

        let mut jolt_polynomials = JoltPolynomials {
            // bytecode: bytecode_polynomials,
            // read_write_memory: memory_polynomials,
            // timestamp_range_check: range_check_polys,
            instruction_lookups: instruction_polynomials,
            r1cs: r1cs_polynomials,
            ..Default::default()
        };

        r1cs_builder.compute_aux(&mut jolt_polynomials);

        let jolt_commitments =
            jolt_polynomials.commit::<C, PCS, ProofTranscript>(&preprocessing.shared);

        // transcript.append_scalar(&spartan_key.vk_digest);

        // jolt_commitments
        //     .read_write_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));
        // jolt_commitments
        //     .init_final_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));

        let mut opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
            ProverOpeningAccumulator::new();

        // let bytecode_proof = BytecodeProof::prove_memory_checking(
        //     &preprocessing.shared.generators,
        //     &preprocessing.shared.bytecode,
        //     &jolt_polynomials.bytecode,
        //     &jolt_polynomials,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // );

        let instruction_proof = InstructionLookupsProof::prove(
            &preprocessing.shared.generators,
            &mut jolt_polynomials,
            &preprocessing.shared.instruction_lookups,
            &mut opening_accumulator,
            &mut transcript,
        );

        // let memory_proof = ReadWriteMemoryProof::prove(
        //     &preprocessing.shared.generators,
        //     &preprocessing.shared.read_write_memory,
        //     &jolt_polynomials,
        //     &program_io,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // );

        let spartan_proof = UniformSpartanProof::<
            C,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            ProofTranscript,
        >::prove::<PCS>(
            &r1cs_builder,
            &spartan_key,
            &jolt_polynomials,
            &mut opening_accumulator,
            &mut transcript,
        )
        .expect("r1cs proof failed");

        // Batch-prove all openings
        let opening_proof = opening_accumulator
            .reduce_and_prove::<PCS>(&preprocessing.shared.generators, &mut transcript);

        drop_in_background_thread(jolt_polynomials);

        let jolt_proof = JoltProof {
            trace_length,
            // bytecode: bytecode_proof,
            // read_write_memory: memory_proof,
            instruction_lookups: instruction_proof,
            r1cs: spartan_proof,
            opening_proof,
            // opening_accumulator,
        };

        // #[cfg(test)]
        // let debug_info = Some(ProverDebugInfo {
        //     transcript,
        //     opening_accumulator,
        // });
        // #[cfg(not(test))]
        // let debug_info = None;
        (jolt_proof, jolt_commitments)
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
        // program_io: JoltDevice,
        // _debug_info: Option<ProverDebugInfo<F, ProofTranscript>>,
    ) -> Result<(), ProofVerifyError> {
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
        // let memory_start = preprocessing.memory_layout.input_start;
        // let r1cs_builder =
        //     Self::Constraints::construct_constraints(padded_trace_length, memory_start);
        // let spartan_key = spartan::UniformSpartanProof::<C, _, F, ProofTranscript>::setup(
        //     &r1cs_builder,
        //     padded_trace_length,
        // );
        // transcript.append_scalar(&spartan_key.vk_digest);

        // let r1cs_proof = R1CSProof {
        //     key: spartan_key,
        //     proof: proof.r1cs,
        //     _marker: PhantomData,
        // };

        // commitments
        //     .read_write_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));
        // commitments
        //     .init_final_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));

        // Self::verify_bytecode(
        //     &preprocessing.bytecode,
        //     &preprocessing.generators,
        //     proof.bytecode,
        //     &commitments,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // )?;
        Self::verify_instruction_lookups(
            &preprocessing.instruction_lookups,
            &preprocessing.generators,
            proof.instruction_lookups,
            &commitments,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        // Self::verify_memory(
        //     &mut preprocessing.read_write_memory,
        //     &preprocessing.generators,
        //     &preprocessing.memory_layout,
        //     proof.read_write_memory,
        //     &commitments,
        //     program_io,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // )?;
        // Self::verify_r1cs(
        //     r1cs_proof,
        //     &commitments,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // )?;

        // Batch-verify all openings
        opening_accumulator.reduce_and_verify(
            &preprocessing.generators,
            &proof.opening_proof,
            &mut transcript,
        )?;

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

    // #[tracing::instrument(skip_all)]
    // fn verify_bytecode<'a>(
    //     preprocessing: &BytecodePreprocessing<F>,
    //     generators: &PCS::Setup,
    //     proof: BytecodeProof<F, PCS, ProofTranscript>,
    //     commitments: &'a JoltCommitments<PCS, ProofTranscript>,
    //     opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
    //     transcript: &mut ProofTranscript,
    // ) -> Result<(), ProofVerifyError> {
    //     BytecodeProof::verify_memory_checking(
    //         preprocessing,
    //         generators,
    //         proof,
    //         &commitments.bytecode,
    //         commitments,
    //         opening_accumulator,
    //         transcript,
    //     )
    // }

    // #[allow(clippy::too_many_arguments)]
    // #[tracing::instrument(skip_all)]
    // fn verify_memory<'a>(
    //     preprocessing: &mut ReadWriteMemoryPreprocessing,
    //     generators: &PCS::Setup,
    //     memory_layout: &MemoryLayout,
    //     proof: ReadWriteMemoryProof<F, PCS, ProofTranscript>,
    //     commitment: &'a JoltCommitments<PCS, ProofTranscript>,
    //     program_io: JoltDevice,
    //     opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
    //     transcript: &mut ProofTranscript,
    // ) -> Result<(), ProofVerifyError> {
    //     assert!(program_io.inputs.len() <= memory_layout.max_input_size as usize);
    //     assert!(program_io.outputs.len() <= memory_layout.max_output_size as usize);
    //     // pair the memory layout with the program io from the proof
    //     preprocessing.program_io = Some(JoltDevice {
    //         inputs: program_io.inputs,
    //         outputs: program_io.outputs,
    //         panic: program_io.panic,
    //         memory_layout: memory_layout.clone(),
    //     });

    //     ReadWriteMemoryProof::verify(
    //         proof,
    //         generators,
    //         preprocessing,
    //         commitment,
    //         opening_accumulator,
    //         transcript,
    //     )
    // }

    // #[tracing::instrument(skip_all)]
    // fn verify_r1cs<'a>(
    //     proof: R1CSProof<
    //         C,
    //         <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
    //         F,
    //         ProofTranscript,
    //     >,
    //     commitments: &'a JoltCommitments<PCS, ProofTranscript>,
    //     opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
    //     transcript: &mut ProofTranscript,
    // ) -> Result<(), ProofVerifyError> {
    //     proof
    //         .verify(commitments, opening_accumulator, transcript)
    //         .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))
    // }

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
