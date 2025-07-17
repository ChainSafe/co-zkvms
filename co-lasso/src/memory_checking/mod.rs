use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::Context;
use jolt_core::lasso::memory_checking::{ExogenousOpenings, Initializable};
pub use jolt_core::lasso::memory_checking::{
    MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes, StructuredPolynomialData,
};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use mpc_core::protocols::{additive, rep3::network::Rep3NetworkCoordinator};

use crate::{
    field::JoltField,
    poly::opening_proof::Rep3ProverOpeningAccumulator,
    subprotocols::{
        commitment::DistributedCommitmentScheme,
        grand_product::{BatchedGrandProductProof, BatchedGrandProductProver},
    },
    utils::{math::Math, transcript::Transcript},
};

pub mod worker;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryCheckingProof<F, PCS, Openings, OtherOpenings, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    Openings: StructuredPolynomialData<F> + Sync + CanonicalSerialize + CanonicalDeserialize,
    OtherOpenings: ExogenousOpenings<F> + Sync,
    ProofTranscript: Transcript,
{
    /// Read/write/init/final multiset hashes for each memory
    pub multiset_hashes: MultisetHashes<F>,
    /// The read and write grand products for every memory has the same size,
    /// so they can be batched.
    pub read_write_grand_product: BatchedGrandProductProof<PCS, ProofTranscript>,
    /// The init and final grand products for every memory has the same size,
    /// so they can be batched.
    pub init_final_grand_product: BatchedGrandProductProof<PCS, ProofTranscript>,
    /// The openings associated with the grand products.
    pub openings: Openings,
    pub exogenous_openings: OtherOpenings,
}

pub trait Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>:
    MemoryCheckingProver<F, PCS, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
    Self::Openings: Initializable<F, Self::Preprocessing>,
{
    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::prove_memory_checking")]
    fn prove_memory_checking(
        memory_size: usize,
        preprocessing: &Self::Preprocessing,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> eyre::Result<
        MemoryCheckingProof<F, PCS, Self::Openings, Self::ExogenousOpenings, ProofTranscript>,
    > {
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = Self::prove_grand_products_rep3(memory_size, preprocessing, network, transcript)
            .context("while proving grand products")?;

        let read_write_batch_size =
            multiset_hashes.read_hashes.len() + multiset_hashes.write_hashes.len();
        let init_final_batch_size =
            multiset_hashes.init_hashes.len() + multiset_hashes.final_hashes.len();

        tracing::info!("read_write_batch_size: {}", read_write_batch_size);
        tracing::info!("init_final_batch_size: {}", init_final_batch_size);

        let (openings, exogenous_openings) =
            Self::receive_openings(preprocessing, transcript, network)?;

        Ok(MemoryCheckingProof {
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
            openings,
            exogenous_openings,
        })
    }

    fn prove_grand_products_rep3(
        M: usize,
        preprocessing: &Self::Preprocessing,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> eyre::Result<(
        BatchedGrandProductProof<PCS, ProofTranscript>,
        BatchedGrandProductProof<PCS, ProofTranscript>,
        MultisetHashes<F>,
        Vec<F>,
        Vec<F>,
    )> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();
        network.broadcast_request((gamma, tau))?;
        transcript.append_message(Self::protocol_name());

        let num_lookups = network.receive_responses(0usize)?[0];

        let (read_write_hashes_shares, init_final_hashes_shares): (Vec<Vec<_>>, Vec<Vec<_>>) =
            network
                .receive_responses((Vec::default(), Vec::default()))
                .context("while receiving hashes")?
                .into_iter()
                .unzip();

        assert_eq!(read_write_hashes_shares.len(), 3);
        assert_eq!(init_final_hashes_shares.len(), 3);

        let read_write_hashes = additive::combine_field_elements(
            &read_write_hashes_shares[0],
            &read_write_hashes_shares[1],
            &read_write_hashes_shares[2],
        );
        let init_final_hashes = additive::combine_field_elements(
            &init_final_hashes_shares[0],
            &init_final_hashes_shares[1],
            &init_final_hashes_shares[2],
        );

        tracing::info!("read_write_hashes: {:?}", read_write_hashes);
        tracing::info!("init_final_hashes: {:?}", init_final_hashes);

        let multiset_hashes = Self::uninterleave_hashes(
            preprocessing,
            read_write_hashes.clone(),
            init_final_hashes.clone(),
        );
        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let num_layers_read_write = (num_lookups).log_2() + 1; // +1 for the flag layer
        let num_layers_init_final = M.log_2();

        let (read_write_grand_product, r_read_write) = BatchedGrandProductProver::prove(
            read_write_hashes,
            num_layers_read_write,
            network,
            transcript,
        )?;
        let (init_final_grand_product, r_init_final) = BatchedGrandProductProver::prove(
            init_final_hashes,
            num_layers_init_final,
            network,
            transcript,
        )?;

        Ok((
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ))
    }

    fn receive_openings(
        preprocessing: &Self::Preprocessing,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(Self::Openings, Self::ExogenousOpenings)> {
        let mut exogenous_openings = Self::ExogenousOpenings::default();
        let mut openings = Self::Openings::initialize(preprocessing);

        let read_write_evals: Vec<F> =
            Rep3ProverOpeningAccumulator::receive_claims(transcript, network)?;

        let read_write_openings: Vec<&mut F> = openings
            .read_write_values_mut()
            .into_iter()
            .chain(exogenous_openings.openings_mut())
            .collect();

        for (opening, eval) in read_write_openings.into_iter().zip(read_write_evals.iter()) {
            *opening = *eval;
        }

        let init_final_evals: Vec<F> =
            Rep3ProverOpeningAccumulator::receive_claims(transcript, network)?;

        for (opening, eval) in openings
            .init_final_values_mut()
            .into_iter()
            .zip(init_final_evals.iter())
        {
            *opening = *eval;
        }

        Ok((openings, exogenous_openings))
    }
}

/// This type, used within a `StructuredPolynomialData` struct, indicates that the
/// field has a corresponding opening but no corresponding polynomial or commitment ––
/// the prover doesn't need to compute a witness polynomial or commitment because
/// the verifier can compute the opening on its own.
pub type VerifierComputedOpening<T> = Option<T>;
