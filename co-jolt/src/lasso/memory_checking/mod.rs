use eyre::Context;
pub use jolt_core::lasso::memory_checking::{
    MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes, StructuredPolynomialData,
};
use jolt_core::{
    field::JoltField,
    lasso::memory_checking::{ExogenousOpenings, Initializable},
    subprotocols::grand_product::BatchedGrandProductProof,
};
use mpc_core::protocols::{
    additive,
    rep3::network::Rep3NetworkCoordinator,
};

use crate::{
    poly::commitment::Rep3CommitmentScheme,
    poly::opening_proof::Rep3ProverOpeningAccumulator,
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

        let (read_write_hashes_shares, init_final_hashes_shares): (Vec<Vec<_>>, Vec<Vec<_>>) =
            network
                .receive_responses()
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

        let multiset_hashes = Self::uninterleave_hashes(
            preprocessing,
            read_write_hashes.clone(),
            init_final_hashes.clone(),
        );
        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let read_write_circuit = Self::read_write_grand_product_rep3(preprocessing, num_lookups);
        let init_final_circuit = Self::init_final_grand_product_rep3(preprocessing, memory_size);

        let read_write_grand_product = read_write_circuit.cooridinate_prove_grand_product(
            read_write_hashes,
            transcript,
            network,
        )?;
        let init_final_grand_product = init_final_circuit.cooridinate_prove_grand_product(
            init_final_hashes,
            transcript,
            network,
        )?;

        Ok((
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
        ))
    }

    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::receive_openings")]
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

    fn read_write_grand_product_rep3(
        _preprocessing: &Self::Preprocessing,
        num_lookups: usize,
    ) -> Self::Rep3ReadWriteGrandProduct {
        Self::Rep3ReadWriteGrandProduct::construct(num_lookups.log_2())
    }

    fn init_final_grand_product_rep3(
        _preprocessing: &Self::Preprocessing,
        memory_size: usize,
    ) -> Self::Rep3InitFinalGrandProduct {
        Self::Rep3InitFinalGrandProduct::construct(memory_size.log_2())
    }
}

/// This type, used within a `StructuredPolynomialData` struct, indicates that the
/// field has a corresponding opening but no corresponding polynomial or commitment ––
/// the prover doesn't need to compute a witness polynomial or commitment because
/// the verifier can compute the opening on its own.
pub type VerifierComputedOpening<T> = Option<T>;
