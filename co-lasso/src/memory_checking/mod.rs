use co_spartan::mpc::additive;
use eyre::Context;
use jolt_core::{
    poly::{field::JoltField, structured_poly::StructuredCommitment},
    subprotocols::grand_product::BatchedGrandProductArgument,
    utils::{math::Math, transcript::ProofTranscript},
};
use mpc_net::mpc_star::MpcStarNetCoordinator;
use std::marker::PhantomData;

pub use jolt_core::lasso::memory_checking::{
    MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
};

use crate::{
    poly::Rep3StructuredOpeningProof,
    subprotocols::{
        commitment::DistributedCommitmentScheme, grand_product::BatchedGrandProductProver,
    },
};

pub mod worker;

pub trait Rep3MemoryCheckingProver<F, CS, Polynomials, Network>:
    MemoryCheckingProver<F, CS, Polynomials>
where
    F: JoltField,
    CS: DistributedCommitmentScheme<F>,
    Polynomials: StructuredCommitment<CS>,
    Network: MpcStarNetCoordinator,
    Self::ReadWriteOpenings: Rep3StructuredOpeningProof<F, CS, Polynomials>,
    Self::InitFinalOpenings: Rep3StructuredOpeningProof<F, CS, Polynomials>,
{
    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::prove_memory_checking")]
    fn prove_memory_checking(
        memory_size: usize,
        preprocessing: &Self::Preprocessing,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> eyre::Result<
        MemoryCheckingProof<F, CS, Polynomials, Self::ReadWriteOpenings, Self::InitFinalOpenings>,
    > {
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = Self::prove_grand_products_rep3(memory_size, preprocessing, network, transcript)
            .context("while proving grand products")?;

        let read_write_openings = Self::ReadWriteOpenings::open_rep3(&r_read_write, network)?;
        let read_write_opening_proof =
            Self::ReadWriteOpenings::prove_openings_rep3(transcript, network)?;
        let init_final_openings = Self::InitFinalOpenings::open_rep3(&r_init_final, network)?;
        let init_final_opening_proof =
            Self::InitFinalOpenings::prove_openings_rep3(transcript, network)?;

        Ok(MemoryCheckingProof {
            _polys: PhantomData,
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
            read_write_openings,
            read_write_opening_proof,
            init_final_openings,
            init_final_opening_proof,
        })
    }

    fn prove_grand_products_rep3(
        M: usize,
        preprocessing: &Self::Preprocessing,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> eyre::Result<(
        BatchedGrandProductArgument<F>,
        BatchedGrandProductArgument<F>,
        MultisetHashes<F>,
        Vec<F>,
        Vec<F>,
    )> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar::<F>(b"Memory checking gamma");
        let tau: F = transcript.challenge_scalar::<F>(b"Memory checking tau");
        network.broadcast_request((gamma, tau))?;
        transcript.append_protocol_name(Self::protocol_name());

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
}
