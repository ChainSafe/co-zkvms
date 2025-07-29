use std::marker::PhantomData;

use crate::jolt::vm::jolt::witness::JoltWitnessMeta;
use crate::lasso::memory_checking::{MemoryCheckingProver, Rep3MemoryCheckingProver};
use crate::poly::opening_proof::Rep3ProverOpeningAccumulator;
use crate::subprotocols::grand_product::{Rep3BatchedDenseGrandProduct, Rep3BatchedGrandProduct};
use crate::subprotocols::sumcheck;
use crate::utils::transcript::TranscriptExt;
use jolt_core::field::JoltField;
use jolt_core::jolt::vm::read_write_memory::{OutputSumcheckProof, ReadWriteMemoryPreprocessing};
use jolt_core::poly::compact_polynomial::CompactPolynomial;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::opening_proof::ProverOpeningAccumulator;
use jolt_core::subprotocols::grand_product::BatchedDenseGrandProduct;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_core::protocols::rep3::PartyID;
use rayon::prelude::*;

use crate::poly::commitment::{commitment_scheme::CommitmentScheme, Rep3CommitmentScheme};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_common::constants::{
    BYTES_PER_INSTRUCTION, MEMORY_OPS_PER_INSTRUCTION, RAM_START_ADDRESS, REGISTER_COUNT,
};
use jolt_common::rv_trace::{JoltDevice, MemoryLayout, MemoryOp};
use jolt_core::jolt::vm::{
    read_write_memory::{ReadWriteMemoryPolynomials, ReadWriteMemoryProof},
    timestamp_range_check::TimestampValidityProof,
    JoltCommitments, JoltPolynomials, JoltStuff, JoltTraceStep,
};
use jolt_core::utils::transcript::Transcript;
use jolt_core::{
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, identity_poly::IdentityPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math},
};

use crate::jolt::vm::witness::Rep3JoltPolynomials;

pub trait Rep3ReadWriteMemoryCoordinator<F, PCS, ProofTranscript, Network>:
    Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn prove_rep3(
        pcs_setup: &PCS::Setup,
        meta: JoltWitnessMeta,
        preprocessing: &ReadWriteMemoryPreprocessing,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<ReadWriteMemoryProof<F, PCS, ProofTranscript>>;
}

impl<F, PCS, ProofTranscript, Network>
    Rep3ReadWriteMemoryCoordinator<F, PCS, ProofTranscript, Network>
    for ReadWriteMemoryProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: TranscriptExt,
    Network: Rep3NetworkCoordinator,
{
    fn prove_rep3(
        pcs_setup: &PCS::Setup,
        meta: JoltWitnessMeta,
        preprocessing: &ReadWriteMemoryPreprocessing,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<ReadWriteMemoryProof<F, PCS, ProofTranscript>> {
        let memory_size = meta.read_write_memory_size;

        let memory_checking_proof =
            Self::coordinate_memory_checking(preprocessing, memory_size, transcript, network)?;

        let output_proof = coordinate_prove_outputs(memory_size, transcript, network)?;

        network.send_requests(vec![Some(transcript.state()), None, None])?;

        let (timestamp_validity_proof, transcript_state) =
            network.receive_response(PartyID::ID0, 0)?;
        transcript.update_state(transcript_state);

        Ok(ReadWriteMemoryProof {
            memory_checking_proof,
            output_proof,
            timestamp_validity_proof,
        })
    }
}

fn coordinate_prove_outputs<F, PCS, ProofTranscript, Network>(
    memory_size: usize,
    transcript: &mut ProofTranscript,
    network: &mut Network,
) -> eyre::Result<OutputSumcheckProof<F, PCS, ProofTranscript>>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    let num_rounds = memory_size.log_2();
    let r_eq: Vec<F> = transcript.challenge_vector(num_rounds);
    network.broadcast_request(r_eq)?;
    let (sumcheck_proof, _) =
        sumcheck::coordinate_prove_arbitrary::<F, _, Network>(num_rounds, transcript, network)?;

    let sumcheck_openings = Rep3ProverOpeningAccumulator::receive_claims(transcript, network)?;

    Ok(OutputSumcheckProof {
        num_rounds,
        sumcheck_proof,
        opening: sumcheck_openings[0],
        _pcs: PhantomData,
    })
}

impl<F, PCS, ProofTranscript, Network> Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>
    for ReadWriteMemoryProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    type Rep3ReadWriteGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    type Rep3InitFinalGrandProduct = Rep3BatchedDenseGrandProduct<F>;
}
