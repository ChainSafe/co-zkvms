use eyre::Context;
pub use jolt_core::lasso::memory_checking::{
    MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes, StructuredPolynomialData,
};
use jolt_core::{
    lasso::memory_checking::{ExogenousOpenings, Initializable},
    poly::dense_interleaved_poly::DenseInterleavedPolynomial,
    subprotocols::grand_product::{
        BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductProof,
    },
};
use mpc_core::protocols::{
    additive::{self, AdditiveShare},
    rep3::network::Rep3NetworkCoordinator,
};
use rayon::prelude::*;

use crate::{field::JoltField, poly::opening_proof::Rep3OpeningAccumulatorCoordinator};
use crate::{
    poly::commitment::Rep3CommitmentScheme,
    subprotocols::grand_product::Rep3BatchedGrandProduct,
    utils::{math::Math, transcript::Transcript},
};
pub use jolt_core::lasso::memory_checking::MemoryCheckingProof;
pub mod worker;

pub trait Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>:
    MemoryCheckingProver<F, PCS, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
    Self::Openings: Initializable<F, Self::Preprocessing>,
{
    type Rep3ReadWriteGrandProduct: Rep3BatchedGrandProduct<F, PCS, ProofTranscript, Network>
        + Send
        + 'static;
    type Rep3InitFinalGrandProduct: Rep3BatchedGrandProduct<F, PCS, ProofTranscript, Network>
        + Send
        + 'static;

    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::prove_memory_checking")]
    fn coordinate_memory_checking(
        preprocessing: &Self::Preprocessing,
        num_lookups: usize,
        memory_size: usize,
        opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<
        MemoryCheckingProof<F, PCS, Self::Openings, Self::ExogenousOpenings, ProofTranscript>,
    > {
        let (read_write_grand_product, init_final_grand_product, multiset_hashes) =
            Self::prove_grand_products_rep3(
                preprocessing,
                num_lookups,
                memory_size,
                network,
                transcript,
            )
            .context("while proving grand products")?;

        let (openings, exogenous_openings) = Self::receive_openings(
            num_lookups,
            memory_size,
            preprocessing,
            opening_accumulator,
            transcript,
            network,
        )?;

        Ok(MemoryCheckingProof {
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
            openings,
            exogenous_openings,
        })
    }

    fn prove_grand_products_rep3(
        preprocessing: &Self::Preprocessing,
        num_lookups: usize,
        memory_size: usize,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> eyre::Result<(
        BatchedGrandProductProof<PCS, ProofTranscript>,
        BatchedGrandProductProof<PCS, ProofTranscript>,
        MultisetHashes<F>,
    )> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();
        network.broadcast_request((gamma, tau))?;
        transcript.append_message(Self::protocol_name());

        let (read_write_hashes, init_final_hashes): (Vec<_>, Vec<_>) = network
            .receive_responses_from_subnets::<(Vec<AdditiveShare<F>>, Vec<AdditiveShare<F>>)>()
            .context("while receiving hashes")?
            .into_iter()
            .map(|worker_hashes| {
                let (rw_shares, if_shares) = worker_hashes.into_iter().unzip();

                (
                    additive::combine_additive_vec(rw_shares),
                    additive::combine_additive_vec(if_shares),
                )
            })
            .unzip();

        let read_write_hashes = read_write_hashes.concat();
        let init_final_hashes = init_final_hashes.concat();

        let read_write_batch_size = read_write_hashes.len();
        let init_final_batch_size = init_final_hashes.len();

        let multiset_hashes = Self::uninterleave_hashes(
            preprocessing,
            read_write_hashes.clone(),
            init_final_hashes.clone(),
        );

        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let read_write_circuit =
            Self::read_write_grand_product_rep3(preprocessing, num_lookups, read_write_batch_size);
        let init_final_circuit =
            Self::init_final_grand_product_rep3(preprocessing, memory_size, init_final_batch_size);

        let (read_write_grand_product, _) = read_write_circuit.cooridinate_prove_grand_product(
            read_write_hashes,
            transcript,
            network,
        )?;

        let (init_final_grand_product, _) = init_final_circuit.cooridinate_prove_grand_product(
            init_final_hashes,
            transcript,
            network,
        )?;

        if network.is_distributed() {
            network.broadcast_request((read_write_batch_size, init_final_batch_size))?;
        }

        Ok((
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
        ))
    }

    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::receive_openings")]
    fn receive_openings(
        read_write_chunk_size: usize,
        init_final_chunk_size: usize,
        preprocessing: &Self::Preprocessing,
        opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(Self::Openings, Self::ExogenousOpenings)> {
        receive_openings::<F, _, Self::Openings, Self::ExogenousOpenings, _, _>(
            read_write_chunk_size,
            init_final_chunk_size,
            preprocessing,
            opening_accumulator,
            transcript,
            network,
        )
    }

    fn read_write_grand_product_rep3(
        _preprocessing: &Self::Preprocessing,
        num_lookups: usize,
        batch_size: usize,
    ) -> Self::Rep3ReadWriteGrandProduct {
        Self::Rep3ReadWriteGrandProduct::construct(num_lookups.log_2(), batch_size)
    }

    fn init_final_grand_product_rep3(
        _preprocessing: &Self::Preprocessing,
        memory_size: usize,
        batch_size: usize,
    ) -> Self::Rep3InitFinalGrandProduct {
        Self::Rep3InitFinalGrandProduct::construct(memory_size.log_2(), batch_size)
    }

    fn construct_remaining_layers(
        hashes: &mut Vec<F>,
        num_workers: usize,
    ) -> Vec<DenseInterleavedPolynomial<F>> {
        let grand_product = <BatchedDenseGrandProduct<F> as BatchedGrandProduct<
            F,
            PCS,
            ProofTranscript,
        >>::construct((hashes.clone(), hashes.len() / num_workers));

        *hashes = BatchedGrandProduct::<F, PCS, ProofTranscript>::claimed_outputs(&grand_product);

        grand_product.into_layers()
    }
}

/// This type, used within a `StructuredPolynomialData` struct, indicates that the
/// field has a corresponding opening but no corresponding polynomial or commitment ––
/// the prover doesn't need to compute a witness polynomial or commitment because
/// the verifier can compute the opening on its own.
pub type VerifierComputedOpening<T> = Option<T>;

pub(crate) fn receive_openings<F, Preprocessing, Openings, ExoOpenings, ProofTranscript, Network>(
    read_write_chunk_size: usize,
    init_final_chunk_size: usize,
    preprocessing: &Preprocessing,
    opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
    transcript: &mut ProofTranscript,
    network: &mut Network,
) -> eyre::Result<(Openings, ExoOpenings)>
where
    F: JoltField,
    Openings: StructuredPolynomialData<F> + Sync + Initializable<F, Preprocessing>,
    ExoOpenings: ExogenousOpenings<F> + Sync,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    let mut exogenous_openings = ExoOpenings::default();
    let mut openings = Openings::initialize(preprocessing);

    let read_write_evals: Vec<F> =
        opening_accumulator.append(read_write_chunk_size.log_2(), transcript, network)?;

    let read_write_openings: Vec<_> = openings
        .read_write_values_grand_product_mut()
        .into_iter()
        .chain(exogenous_openings.openings_mut())
        .collect();

    read_write_openings
        .into_par_iter()
        .zip(read_write_evals.par_iter())
        .for_each(|(opening, eval)| {
            *opening = *eval;
        });

    let init_final_evals: Vec<F> =
        opening_accumulator.append(init_final_chunk_size.log_2(), transcript, network)?;

    openings
        .init_final_values_mut()
        .into_par_iter()
        .zip(init_final_evals.par_iter())
        .for_each(|(opening, eval)| {
            *opening = *eval;
        });

    Ok((openings, exogenous_openings))
}
