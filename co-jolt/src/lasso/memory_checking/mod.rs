use eyre::Context;
use itertools::{interleave, Itertools};
pub use jolt_core::lasso::memory_checking::{
    MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes, StructuredPolynomialData,
};
use jolt_core::{
    lasso::memory_checking::{ExogenousOpenings, Initializable},
    poly::{
        dense_interleaved_poly::DenseInterleavedPolynomial,
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    subprotocols::grand_product::{
        BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductProof,
    },
};
use mpc_core::protocols::{
    additive::{self, AdditiveShare},
    rep3::network::Rep3NetworkCoordinator,
};
use rayon::prelude::*;

use crate::field::JoltField;
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
        let (
            read_write_grand_product,
            init_final_grand_product,
            r_read_write,
            r_init_final,
            multiset_hashes,
        ) = Self::prove_grand_products_rep3(
            preprocessing,
            num_lookups,
            memory_size,
            network,
            transcript,
        )
        .context("while proving grand products")?;

        let (openings, exogenous_openings) = Self::receive_openings(
            r_read_write,
            r_init_final,
            preprocessing,
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
        Vec<F>,
        Vec<F>,
        MultisetHashes<F>,
    )> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();
        network.broadcast_request((gamma, tau))?;
        transcript.append_message(Self::protocol_name());

        let (mut read_write_hashes, mut init_final_hashes) = network
            .receive_responses_from_subnets::<(Vec<AdditiveShare<F>>, Vec<AdditiveShare<F>>)>()
            .context("while receiving hashes")?
            .into_iter()
            .map(|worker_shares| {
                let (rw_hashes_shares, if_hashes_shares): (Vec<Vec<_>>, Vec<Vec<_>>) =
                    worker_shares.into_iter().unzip();
                (
                    additive::combine_additive_vec(rw_hashes_shares),
                    additive::combine_additive_vec(if_hashes_shares),
                )
            })
            .reduce(|(rw_hashes, if_hashes), (rw_hashes_next, if_hashes_next)| {
                (
                    interleave(rw_hashes, rw_hashes_next).collect_vec(),
                    interleave(if_hashes, if_hashes_next).collect_vec(),
                )
            })
            .unwrap();

        let num_workers = 1 << network.log_num_workers_per_party();
        let rw_remaining_layers = if network.log_num_workers_per_party() > 0 {
            Some(Self::construct_remaining_layers(
                &mut read_write_hashes,
                num_workers,
            ))
        } else {
            None
        };

        let init_final_remaining_layers = if network.log_num_workers_per_party() > 0 {
            Some(Self::construct_remaining_layers(
                &mut init_final_hashes,
                num_workers,
            ))
        } else {
            None
        };

        // init_final_hashes = DenseInterleavedPolynomial::new(init_final_hashes)
        //     .par_chunks(2)
        //     .map(|chunk| chunk[0] * chunk[1])
        //     .collect();

        let multiset_hashes = Self::uninterleave_hashes(
            preprocessing,
            read_write_hashes.clone(),
            init_final_hashes.clone(),
        );

        // println!("Multiset read_hashes: {:?}", multiset_hashes.read_hashes);
        // println!("Multiset write_hashes: {:?}", multiset_hashes.write_hashes);
        // println!("Multiset init_hashes: {:?}", multiset_hashes.init_hashes);
        // println!("Multiset final_hashes: {:?}", multiset_hashes.final_hashes);

        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        println!("Multiset equality check passed");
        multiset_hashes.append_to_transcript(transcript);

        let read_write_circuit = Self::read_write_grand_product_rep3(preprocessing, num_lookups);
        let init_final_circuit = Self::init_final_grand_product_rep3(preprocessing, memory_size);

        let (read_write_grand_product, r_read_write) = read_write_circuit
            .cooridinate_prove_grand_product(
                read_write_hashes,
                rw_remaining_layers,
                transcript,
                network,
            )?;
        println!("Read-write grand product proved");
        let (init_final_grand_product, r_init_final) = init_final_circuit
            .cooridinate_prove_grand_product(
                init_final_hashes,
                init_final_remaining_layers,
                transcript,
                network,
            )?;
        println!("Init-final grand product proved");

        Ok((
            read_write_grand_product,
            init_final_grand_product,
            r_read_write,
            r_init_final,
            multiset_hashes,
        ))
    }

    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::receive_openings")]
    fn receive_openings(
        mut r_read_write: Vec<F>,
        mut r_init_final: Vec<F>,
        preprocessing: &Self::Preprocessing,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(Self::Openings, Self::ExogenousOpenings)> {
        let mut exogenous_openings = Self::ExogenousOpenings::default();
        let mut openings = Self::Openings::initialize(preprocessing);

        let log_num_workers = network.log_num_workers_per_party();
        let remaining_r_read_write = r_read_write.split_off(r_read_write.len() - log_num_workers);
        let remaining_r_init_final = r_init_final.split_off(r_init_final.len() - log_num_workers);

        let read_write_evals: Vec<_> = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
            .into_iter()
            .map(additive::combine_additive_vec)
            .fold(
                vec![vec![]; openings.read_write_values().len()],
                |mut evals, eval| {
                    evals.push(eval);
                    evals
                },
            )
            .into_par_iter()
            .map(|evals| {
                DensePolynomial::new(evals).evaluate_at_chi_low_optimized(&remaining_r_read_write)
            })
            .collect();

        Rep3ProverOpeningAccumulator::coordinate_with_known_claims(
            &read_write_evals,
            transcript,
            network,
        )?;

        let read_write_openings: Vec<&mut F> = openings
            .read_write_values_mut()
            .into_iter()
            .chain(exogenous_openings.openings_mut())
            .collect();

        for (opening, eval) in read_write_openings.into_iter().zip(read_write_evals.iter()) {
            *opening = *eval;
        }

        let init_final_evals: Vec<_> = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
            .into_iter()
            .map(additive::combine_additive_vec)
            .fold(
                vec![vec![]; openings.init_final_values().len()],
                |mut evals, eval| {
                    evals.push(eval);
                    evals
                },
            )
            .into_par_iter()
            .map(|evals| {
                DensePolynomial::new(evals).evaluate_at_chi_low_optimized(&remaining_r_init_final)
            })
            .collect();

        Rep3ProverOpeningAccumulator::coordinate_with_known_claims(
            &init_final_evals,
            transcript,
            network,
        )?;

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

        // // not sure if this is sufficient for workers > 2
        // read_write_hashes = DenseInterleavedPolynomial::new(read_write_hashes)
        //     .par_chunks(2)
        //     .map(|chunk| chunk[0] * chunk[1])
        //     .collect();

        // assert_eq!(read_write_hashes, read_write_hashes_);
        grand_product.into_layers()
    }
}

/// This type, used within a `StructuredPolynomialData` struct, indicates that the
/// field has a corresponding opening but no corresponding polynomial or commitment ––
/// the prover doesn't need to compute a witness polynomial or commitment because
/// the verifier can compute the opening on its own.
pub type VerifierComputedOpening<T> = Option<T>;
