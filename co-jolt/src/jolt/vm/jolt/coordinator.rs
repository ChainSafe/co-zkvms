use std::marker::PhantomData;

use crate::{
    jolt::vm::{
        read_write_memory::{coordinator::Rep3ReadWriteMemoryCoordinator, witness::Rep3ProgramIO},
        witness::JoltWitnessMeta,
    },
    lasso::memory_checking::{Rep3MemoryCheckingProver, StructuredPolynomialData},
    poly::{commitment::Rep3CommitmentScheme, opening_proof::Rep3ProverOpeningAccumulator},
    r1cs::spartan::coordinator::Rep3UniformSpartanCoordinator,
    utils::transcript::TranscriptExt,
};
use ark_std::test_rng;
use jolt_core::{
    jolt::vm::{
        bytecode::BytecodeProof, read_write_memory::ReadWriteMemoryProof, JoltVerifierPreprocessing,
    },
    r1cs::{constraints::R1CSConstraints, key::UniformSpartanKey, spartan::UniformSpartanProof},
};
use jolt_tracer::JoltDevice;
use mpc_core::protocols::rep3::{network::Rep3NetworkCoordinator, PartyID};
use snarks_core::math::Math;

use crate::jolt::vm::{jolt::witness::Rep3Polynomials, witness::Rep3JoltPolynomialsExt};
use crate::jolt::{
    instruction::Rep3JoltInstructionSet,
    vm::{
        instruction_lookups::InstructionLookupsProof, rv32i_vm::RV32IJoltVM,
        witness::Rep3JoltPolynomials, Jolt, JoltCommitments, JoltProof, JoltTraceStep,
    },
};
use jolt_core::field::JoltField;
use jolt_core::utils::transcript::AppendToTranscript;

pub trait JoltRep3<F, PCS, const C: usize, const M: usize, ProofTranscript>:
    Jolt<F, PCS, C, M, ProofTranscript>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: TranscriptExt,
    Self::InstructionSet: Rep3JoltInstructionSet<F>,
{
    #[tracing::instrument(skip_all, name = "Rep3Jolt::init")]
    fn init_rep3<Network: Rep3NetworkCoordinator>(
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        witness: Option<(Vec<JoltTraceStep<F, Self::InstructionSet>>, JoltDevice)>,
        network: &mut Network,
    ) -> eyre::Result<(
        UniformSpartanKey<C, <Self::Constraints as R1CSConstraints<C, F>>::Inputs, F>,
        JoltWitnessMeta,
    )> {
        let (spartan_key, meta) = match witness {
            Some((mut trace, program_io)) => {
                let trace_length = trace.len();
                let padded_trace_length = trace_length.next_power_of_two();
                JoltTraceStep::pad(&mut trace);
                let mut rng = test_rng();
                let mut polynomials = Self::generate_witness(&preprocessing, trace, &program_io);
                let r1cs_builder = Self::Constraints::construct_constraints(
                    padded_trace_length,
                    program_io.memory_layout.input_start,
                );
                let spartan_key = UniformSpartanKey::from_builder(&r1cs_builder);
                r1cs_builder.compute_aux(&mut polynomials);
                let read_write_memory_size = polynomials.read_write_memory.v_final.len();
                let memory_layout = program_io.memory_layout;
                let num_ops = padded_trace_length;
                assert_eq!(polynomials.instruction_lookups.dim[0].len(), num_ops);
                assert_eq!(polynomials.read_write_memory.a_ram.len(), num_ops);
                assert_eq!(polynomials.bytecode.a_read_write.len(), num_ops);

                let polynomials_shares = Rep3JoltPolynomials::generate_secret_shares(
                    &preprocessing,
                    polynomials,
                    &mut rng,
                );
                let program_io_shares = Rep3ProgramIO::<F>::generate_secret_shares(
                    program_io,
                    read_write_memory_size,
                    &mut rng,
                );
                let witness_shares: Vec<_> = polynomials_shares
                    .into_iter()
                    .zip(program_io_shares)
                    .map(|(polynomials, program_io)| (polynomials, program_io, trace_length))
                    .collect();

                tracing::trace_span!("send_witness_shares")
                    .in_scope(|| network.send_requests_blocking(witness_shares))?;
                (
                    spartan_key,
                    JoltWitnessMeta {
                        padded_trace_length,
                        read_write_memory_size,
                        memory_layout,
                    },
                )
            }
            None => {
                let meta = network.receive_response::<JoltWitnessMeta>(PartyID::ID0, 0)?;
                let r1cs_builder = Self::Constraints::construct_constraints(
                    meta.padded_trace_length.next_power_of_two(),
                    meta.memory_layout.input_start,
                );
                let spartan_key = UniformSpartanKey::from_builder(&r1cs_builder);
                (spartan_key, meta)
            }
        };

        tracing::info!("padded_trace_length: {}", meta.padded_trace_length);

        Ok((spartan_key, meta))
    }

    #[tracing::instrument(skip_all, name = "Rep3Jolt::prove")]
    fn prove_rep3<Network: Rep3NetworkCoordinator>(
        meta: JoltWitnessMeta,
        spartan_key: &UniformSpartanKey<C, <Self::Constraints as R1CSConstraints<C, F>>::Inputs, F>,
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        network: &mut Network,
    ) -> eyre::Result<(
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
        // Option<ProverDebugInfo<F, ProofTranscript>>,
    )> {
        // icicle::icicle_init();
        network.sync_with_parties()?;

        let trace_length = meta.padded_trace_length;
        let padded_trace_length = trace_length.next_power_of_two();
        let srs_size = PCS::srs_size(&preprocessing.generators);
        let padded_log2 = padded_trace_length.log_2();
        let srs_log2 = srs_size.log_2();

        if padded_trace_length > srs_size {
            panic!(
                "Padded trace length {padded_trace_length} (2^{padded_log2}) exceeds SRS size {srs_size} (2^{srs_log2}). Consider increasing the max_trace_length."
            );
        }

        // F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));

        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        // TODO: why does this work different from jolt_core?
        // Self::fiat_shamir_preamble(
        //     &mut transcript,
        //     &program_io,
        //     &program_io.memory_layout,
        //     trace_length,
        // );

        let jolt_commitments = Rep3JoltPolynomials::receive_commitments(&preprocessing, network)?;

        transcript.append_scalar(&spartan_key.vk_digest);

        jolt_commitments
            .read_write_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));
        jolt_commitments
            .init_final_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));

        network.sync_with_parties()?;

        let span = tracing::span!(tracing::Level::INFO, "BytecodeProof::prove");
        let _guard = span.enter();
        let bytecode_proof = BytecodeProof::coordinate_memory_checking(
            &preprocessing.bytecode,
            meta.padded_trace_length,
            preprocessing.bytecode.v_init_final[0].len(),
            &mut transcript,
            network,
        )?;
        drop(_guard);
        drop(span);

        let instruction_lookups_proof = InstructionLookupsProof::prove_rep3(
            trace_length,
            &preprocessing.instruction_lookups,
            network,
            &mut transcript,
        )?;

        let memory_proof = ReadWriteMemoryProof::prove_rep3(
            meta.padded_trace_length,
            meta.read_write_memory_size,
            &preprocessing.read_write_memory,
            &mut transcript,
            network,
        )?;

        let r1cs_proof =
            UniformSpartanProof::prove_rep3::<PCS>(&spartan_key, &mut transcript, network)
                .expect("r1cs proof failed");

        let opening_proof =
            Rep3ProverOpeningAccumulator::<F>::reduce_and_prove(&mut transcript, network)?;

        let jolt_proof = JoltProof {
            bytecode: bytecode_proof,
            trace_length,
            read_write_memory: memory_proof,
            instruction_lookups: instruction_lookups_proof,
            r1cs: r1cs_proof,
            opening_proof,
            _marker: PhantomData,
        };

        Ok((jolt_proof, jolt_commitments))
    }
}

const C: usize = crate::jolt::vm::rv32i_vm::C;
const M: usize = crate::jolt::vm::rv32i_vm::M;

impl<F, PCS, ProofTranscript> JoltRep3<F, PCS, C, M, ProofTranscript> for RV32IJoltVM
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: TranscriptExt,
{
}
