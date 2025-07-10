use std::marker::PhantomData;

use ark_ec::pairing::Pairing;
use ark_ff::{biginteger::arithmetic, Field};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use co_spartan::mpc::additive;
use eyre::Context;
use itertools::{interleave, multizip, Itertools};
use jolt_core::{
    lasso::memory_checking::MultisetHashes,
    poly::{
        commitment::commitment_scheme::CommitmentScheme, dense_mlpoly::DensePolynomial,
        field::JoltField, structured_poly::StructuredCommitment,
    },
    subprotocols::grand_product::{
        BatchedGrandProductArgument, BatchedGrandProductCircuit, GrandProductCircuit,
    },
    utils::{
        errors::ProofVerifyError, math::Math, mul_0_1_optimized, split_poly_flagged,
        transcript::ProofTranscript,
    },
};
use mpc_core::protocols::rep3;
use mpc_net::mpc_star::MpcStarNetCoordinator;
// use mpc_net::mpc_star::MpcStarNetCoordinator;
use color_eyre::eyre::Result;
use spartan::transcript::Transcript;
use tracing::trace_span;

use crate::{
    grand_product::BatchedGrandProductProver,
    lasso::{LassoPolynomials, MemoryCheckingProof},
    poly::Rep3DensePolynomial,
    LassoPreprocessing, Rep3LassoPolynomials,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct Rep3MemoryCheckingProver<
    const C: usize,
    const M: usize,
    F: JoltField,
    N: MpcStarNetCoordinator,
    // C: CommitmentScheme<Field = F>,
> {
    pub _marker: PhantomData<(F, N)>,
}

type Preprocessing<F> = LassoPreprocessing<F>;

impl<F: JoltField, const C: usize, const M: usize, N: MpcStarNetCoordinator>
    Rep3MemoryCheckingProver<C, M, F, N>
{
    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::prove")]
    pub fn prove(
        preprocessing: &Preprocessing<F>,
        network: &mut N,
        transcript: &mut ProofTranscript,
    ) -> Result<MemoryCheckingProof<F, LassoPolynomials<F>>> {
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = Self::prove_grand_products(preprocessing, network, transcript)
            .context("while proving grand products")?;

        Ok(MemoryCheckingProof {
            _polys: PhantomData,
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
        })
    }

    fn prove_grand_products(
        preprocessing: &Preprocessing<F>,
        network: &mut N,
        transcript: &mut ProofTranscript,
    ) -> Result<(
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
        // tracing::info!("gamma: {} tau: {}", gamma, tau);
        let num_lookups = network.receive_responses(0usize)?[0];
        println!("num_lookups: {}", num_lookups);

        {
            let (read_write_leaves, init_final_leaves): (
                Vec<Vec<Rep3DensePolynomial<F>>>,
                Vec<Vec<Rep3DensePolynomial<F>>>,
            ) = network
                .receive_responses((Vec::default(), Vec::default()))?
                .into_iter()
                .unzip();
            let [rw_share1, rw_share2, rw_share3] = read_write_leaves.try_into().unwrap();
            let [if_share1, if_share2, if_share3] = init_final_leaves.try_into().unwrap();
            let read_write_leaves = multizip((rw_share1, rw_share2, rw_share3))
                .map(|(s1, s2, s3)| {
                    rep3::combine_field_elements(s1.evals_ref(), s2.evals_ref(), s3.evals_ref())
                })
                .collect_vec();
            tracing::info!("read_write_leaves: {:?}", read_write_leaves[0].len());
            let init_final_leaves = multizip((if_share1, if_share2, if_share3))
                .map(|(s1, s2, s3)| {
                    rep3::combine_field_elements(s1.evals_ref(), s2.evals_ref(), s3.evals_ref())
                })
                .collect_vec();
            // tracing::info!("read_leaves: {:?}", (&read_write_leaves[0][..2], &read_write_leaves[0][read_write_leaves[0].len() - 2..]));
            // tracing::info!("write_leaves: {:?}", (&read_write_leaves[1][..2], &read_write_leaves[1][read_write_leaves[1].len() - 2..]));
            // tracing::info!("init_leaves: {:?}", (&init_final_leaves[0][..2], &init_final_leaves[0][init_final_leaves[0].len() - 2..]));
            // tracing::info!("final_leaves: {:?}", (&init_final_leaves[1][..2], &init_final_leaves[1][init_final_leaves[1].len() - 2..]));
        }

        // transcript.append_protocol_name(Self::protocol_name());

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
        tracing::info!("read_write_hashes: {:?}", &read_write_hashes[..2]);
        let init_final_hashes = additive::combine_field_elements(
            &init_final_hashes_shares[0],
            &init_final_hashes_shares[1],
            &init_final_hashes_shares[2],
        );

        let multiset_hashes =
            Self::uninterleave_hashes(preprocessing, &read_write_hashes, &init_final_hashes);
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

    fn uninterleave_hashes(
        _preprocessing: &Preprocessing<F>,
        read_write_hashes: &[F],
        init_final_hashes: &[F],
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len() % 2, 0);
        let num_memories = read_write_hashes.len() / 2;

        let mut read_hashes = Vec::with_capacity(num_memories);
        let mut write_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(num_memories);
        let mut final_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            init_hashes.push(init_final_hashes[2 * i]);
            final_hashes.push(init_final_hashes[2 * i + 1]);
        }

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(
        _preprocessing: &Preprocessing<F>,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        let num_memories = multiset_hashes.read_hashes.len();
        assert_eq!(multiset_hashes.final_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.write_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.init_hashes.len(), num_memories);

        (0..num_memories).into_par_iter().for_each(|i| {
            let read_hash = multiset_hashes.read_hashes[i];
            let write_hash = multiset_hashes.write_hashes[i];
            let init_hash = multiset_hashes.init_hashes[i];
            let final_hash = multiset_hashes.final_hashes[i];
            // tracing::info!("init_hash {} write_hash: {}", init_hash, write_hash);
            // tracing::info!("final_hash {} read_hash: {}", final_hash, read_hash);
            assert_eq!(
                init_hash * write_hash,
                final_hash * read_hash,
                "Multiset hashes don't match"
            );
        });
    }

    fn interleave_hashes(
        _preprocessing: &Preprocessing<F>,
        multiset_hashes: &MultisetHashes<F>,
    ) -> (Vec<F>, Vec<F>) {
        let read_write_hashes = interleave(
            multiset_hashes.read_hashes.clone(),
            multiset_hashes.write_hashes.clone(),
        )
        .collect();
        let init_final_hashes = interleave(
            multiset_hashes.init_hashes.clone(),
            multiset_hashes.final_hashes.clone(),
        )
        .collect();

        (read_write_hashes, init_final_hashes)
    }
}
