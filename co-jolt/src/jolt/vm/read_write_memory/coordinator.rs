use std::marker::PhantomData;

use crate::field::JoltField;
use crate::jolt::vm::timestamp_range_check::coordinator::TimestampValidityProver;
use crate::lasso::memory_checking::Rep3MemoryCheckingProver;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::opening_proof::Rep3OpeningAccumulatorCoordinator;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use crate::subprotocols::sumcheck;
use crate::utils::transcript::TranscriptExt;
use jolt_core::jolt::vm::read_write_memory::ReadWriteMemoryProof;
use jolt_core::jolt::vm::read_write_memory::{OutputSumcheckProof, ReadWriteMemoryPreprocessing};
use jolt_core::jolt::vm::timestamp_range_check::TimestampValidityProof;

use jolt_core::utils::math::Math;
use jolt_core::utils::transcript::Transcript;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

pub trait Rep3ReadWriteMemoryCoordinator<F, PCS, ProofTranscript, Network>:
    Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn prove_rep3(
        num_ops: usize,
        memory_size: usize,
        preprocessing: &ReadWriteMemoryPreprocessing,
        opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
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
    #[tracing::instrument(skip_all, name = "Rep3ReadWriteMemory::prove")]
    fn prove_rep3(
        num_ops: usize,
        memory_size: usize,
        preprocessing: &ReadWriteMemoryPreprocessing,
        opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<ReadWriteMemoryProof<F, PCS, ProofTranscript>> {
        let memory_checking_proof = Self::coordinate_memory_checking(
            preprocessing,
            num_ops,
            memory_size,
            opening_accumulator,
            transcript,
            network,
        )?;

        let output_proof =
            coordinate_prove_outputs(memory_size, opening_accumulator, transcript, network)?;

        let timestamp_validity_proof = TimestampValidityProof::prove_distributed(
            num_ops,
            memory_size,
            opening_accumulator,
            transcript,
            network,
        )?;

        Ok(ReadWriteMemoryProof {
            memory_checking_proof,
            output_proof,
            timestamp_validity_proof,
        })
    }
}

#[tracing::instrument(skip_all, name = "prove_outputs", level = "info")]
fn coordinate_prove_outputs<F, PCS, ProofTranscript, Network>(
    memory_size: usize,
    opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
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
    let mut claim = F::zero();

    // eq * io_witness_range * (v_final - v_io)
    let output_check_fn = |vals: &[F]| -> F { vals[0] * vals[1] * (vals[2] - vals[3]) };

    let (sumcheck_proof, _, sumcheck_openings) = sumcheck::coordinate_distributed_prove_arbitrary(
        &mut claim,
        num_rounds,
        4,
        3,
        output_check_fn,
        transcript,
        network,
    )?;

    opening_accumulator.append_with_claims(
        num_rounds,
        &[sumcheck_openings[2]],
        transcript,
        network,
    )?;

    Ok(OutputSumcheckProof {
        num_rounds,
        sumcheck_proof,
        opening: sumcheck_openings[2],
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
