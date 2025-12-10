use eyre::Context;
use jolt_common::constants::MEMORY_OPS_PER_INSTRUCTION;
use jolt_core::{
    jolt::vm::{
        rv32i_vm::ProofTranscript,
        timestamp_range_check::{
            ReadTimestampOpenings, TimestampRangeCheckOpenings, TimestampValidityProof,
        },
    },
    lasso::memory_checking::{
        ExogenousOpenings, MemoryCheckingProver, MultisetHashes, NoPreprocessing,
        StructuredPolynomialData,
    },
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::grand_product::{BatchedDenseGrandProduct, BatchedGrandProductProof},
    utils::transcript::Transcript,
};
use mpc_core::protocols::rep3::{network::Rep3NetworkCoordinator, PartyID};
use snarks_core::math::Math;
use tokio::net;

use crate::{
    field::JoltField, poly::opening_proof::Rep3OpeningAccumulatorCoordinator,
    subprotocols::grand_product::Rep3BatchedGrandProduct,
};

pub trait TimestampValidityProver<F: JoltField, PCS, ProofTranscript, Network>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn prove_distributed(
        num_ops: usize,
        memory_size: usize,
        opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<TimestampValidityProof<F, PCS, ProofTranscript>>;

    fn prove_grand_products_distributed(
        num_ops: usize,
        _memory_size: usize,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(
        BatchedGrandProductProof<PCS, ProofTranscript>,
        MultisetHashes<F>,
        Vec<F>,
    )>;
}

impl<F, PCS, ProofTranscript, Network> TimestampValidityProver<F, PCS, ProofTranscript, Network>
    for TimestampValidityProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    #[tracing::instrument(skip_all, name = "TimestampValidityProof::prove_distributed")]
    fn prove_distributed(
        num_ops: usize,
        memory_size: usize,
        opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Self> {
        // temporary hack: switch to worker 0 for timestamp range check grand products
        network.set_num_workers(1);
        let (batched_grand_product, multiset_hashes, r_grand_product) =
            Self::prove_grand_products_distributed(num_ops, memory_size, transcript, network)?;

        // send `r_grand_product` to other workers
        network.reset_num_workers();
        network.broadcast_request(r_grand_product)?;

        let mut openings = TimestampRangeCheckOpenings::default();
        let mut timestamp_openings = ReadTimestampOpenings::<F>::default();

        let read_write_evals: Vec<F> =
            opening_accumulator.append(num_ops.log_2(), transcript, network)?;

        let read_write_openings: Vec<_> = openings
            .read_write_values_grand_product_mut()
            .into_iter()
            .chain(timestamp_openings.openings_mut())
            .collect();

        for (opening, eval) in read_write_openings.into_iter().zip(read_write_evals.iter()) {
            *opening = *eval;
        }

        Ok(Self {
            multiset_hashes,
            openings,
            exogenous_openings: timestamp_openings,
            batched_grand_product,
        })
    }

    #[tracing::instrument(
        skip_all,
        name = "TimestampValidityProof::prove_grand_products_distributed"
    )]
    fn prove_grand_products_distributed(
        num_ops: usize,
        _memory_size: usize,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(
        BatchedGrandProductProof<PCS, ProofTranscript>,
        MultisetHashes<F>,
        Vec<F>,
    )> {
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        network.send_request_to_workers(PartyID::ID0, (gamma, tau))?;

        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let hashes = network
            .receive_response_from_workers::<Vec<F>>(PartyID::ID0)
            .context("while receiving hashes")?
            .concat();

        let (read_write_hashes, init_final_hashes) =
            hashes.split_at(4 * MEMORY_OPS_PER_INSTRUCTION);

        let multiset_hashes =
            TimestampValidityProof::<F, PCS, ProofTranscript>::uninterleave_hashes(
                &NoPreprocessing,
                read_write_hashes.to_vec(),
                init_final_hashes.to_vec(),
            );

        multiset_hashes.append_to_transcript(transcript);

        let batch_size = MEMORY_OPS_PER_INSTRUCTION * 6 + 1;

        let (batched_grand_product, r_grand_product) =
            <BatchedDenseGrandProduct<F> as Rep3BatchedGrandProduct<
                F,
                PCS,
                ProofTranscript,
                Network,
            >>::construct(num_ops.log_2(), batch_size)
            .cooridinate_prove_grand_product(hashes, transcript, network)?;

        Ok((batched_grand_product, multiset_hashes, r_grand_product))
    }
}
