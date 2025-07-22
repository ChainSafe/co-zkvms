use crate::{
    jolt::vm::witness::JoltWitnessMeta,
    lasso::memory_checking::StructuredPolynomialData,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{
            ProverOpeningAccumulator, ReducedOpeningProof, Rep3ProverOpeningAccumulator,
        },
    },
    subprotocols::commitment::DistributedCommitmentScheme,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::test_rng;
use jolt_core::{
    jolt::vm::JoltVerifierPreprocessing,
    utils::{thread::drop_in_background_thread, transcript::Transcript},
};
use jolt_tracer::JoltDevice;
use mpc_core::protocols::rep3::{network::Rep3NetworkCoordinator, PartyID};
use snarks_core::math::Math;
use strum::EnumCount;

use crate::jolt::vm::{jolt::witness::Rep3Polynomials, witness::Rep3JoltPolynomialsExt};
use crate::jolt::{
    instruction::{JoltInstructionSet, Rep3JoltInstructionSet},
    subtable::JoltSubtableSet,
    vm::{
        instruction_lookups::InstructionLookupsProof,
        rv32i_vm::{RV32IJoltVM, RV32I},
        witness::Rep3JoltPolynomials,
        Jolt, JoltCommitments, JoltPolynomials, JoltProof, JoltTraceStep,
    },
};
use jolt_core::utils::transcript::AppendToTranscript;
use jolt_core::{
    field::JoltField,
    jolt::vm::{JoltProverPreprocessing, JoltStuff, ProverDebugInfo},
    poly::multilinear_polynomial::MultilinearPolynomial,
    r1cs::inputs::ConstraintInput,
};

pub trait JoltRep3<F, PCS, const C: usize, const M: usize, ProofTranscript>:
    Jolt<F, PCS, C, M, ProofTranscript>
where
    F: JoltField,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Self::InstructionSet: Rep3JoltInstructionSet<F>,
{
    fn init_rep3<Network: Rep3NetworkCoordinator>(
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        trace: Option<Vec<JoltTraceStep<F, Self::InstructionSet>>>,
        network: &mut Network,
    ) -> eyre::Result<JoltWitnessMeta> {
        let meta = match trace {
            Some(mut trace) => {
                let trace_length = trace.len();
                JoltTraceStep::pad(&mut trace);
                let mut rng = test_rng();
                let polynomials = Self::generate_witness(&preprocessing, &trace);

                let polynomials_shares = Rep3JoltPolynomials::generate_secret_shares(
                    &preprocessing,
                    &polynomials,
                    &mut rng,
                );
                network.send_requests(polynomials_shares)?;
                network.broadcast_request(trace_length)?;
                JoltWitnessMeta {
                    trace_length,
                    read_write_memory_size: polynomials.read_write_memory.v_final.len(),
                }
            }
            None => {
                network.receive_response::<JoltWitnessMeta>(PartyID::ID0, 0, Default::default())?
            }
        };

        Ok(meta)
    }

    #[tracing::instrument(skip_all, name = "Rep3Jolt::prove")]
    fn prove_rep3<Network: Rep3NetworkCoordinator>(
        meta: JoltWitnessMeta,
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        network: &mut Network,
    ) -> eyre::Result<(
        JoltProof<
            C,
            M,
            // <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
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

        let trace_length = meta.trace_length;
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
        // Self::fiat_shamir_preamble(
        //     &mut transcript,
        //     &program_io,
        //     &program_io.memory_layout,
        //     trace_length,
        // );

        // r1cs_builder.compute_aux(&mut jolt_polynomials);

        let jolt_commitments = Rep3JoltPolynomials::receive_commitments(&preprocessing, network)?;

        // transcript.append_scalar(&spartan_key.vk_digest);

        // jolt_commitments
        //     .read_write_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));
        // jolt_commitments
        //     .init_final_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));

        let instruction_lookups = InstructionLookupsProof::prove_rep3(
            trace_length,
            &preprocessing.instruction_lookups,
            network,
            &mut transcript,
        )?;

        let opening_proof =
            Rep3ProverOpeningAccumulator::<F>::reduce_and_prove(&mut transcript, network)?;

        let jolt_proof = JoltProof {
            trace_length: 0,
            instruction_lookups,
            opening_proof,
        };

        Ok((jolt_proof, jolt_commitments))
    }
}

const C: usize = crate::jolt::vm::rv32i_vm::C;
const M: usize = crate::jolt::vm::rv32i_vm::M;

impl<F, PCS, ProofTranscript> JoltRep3<F, PCS, C, M, ProofTranscript> for RV32IJoltVM
where
    F: JoltField,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
{
}
