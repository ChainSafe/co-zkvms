use std::panic;

use eyre::Context;
use itertools::{interleave, izip, Itertools};
pub use jolt_core::lasso::memory_checking::{
    MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes, StructuredPolynomialData,
};
use jolt_core::{
    lasso::memory_checking::{ExogenousOpenings, Initializable},
    poly::{
        dense_interleaved_poly::DenseInterleavedPolynomial,
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
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

        let log_num_workers = network.log_num_workers_per_party();
        let num_workers = 1 << log_num_workers;

        let read_write_effective_workers =
            Self::read_write_effective_workers(preprocessing, log_num_workers);
        let init_final_effective_workers =
            Self::init_final_effective_workers(preprocessing, log_num_workers);

        println!(
            "read_write_effective_workers {} init_final_effective_workers {}",
            read_write_effective_workers, init_final_effective_workers
        );

        network.set_worker_subnets(read_write_effective_workers);
        let mut read_write_hashes = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()
            .context("while receiving hashes")?
            .into_iter()
            .flat_map(additive::combine_additive_vec)
            .collect::<Vec<_>>();

        println!("read_write_hashes: {:?}", read_write_hashes.len());

        network.set_worker_subnets(init_final_effective_workers);
        let mut init_final_hashes = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()
            .context("while receiving hashes")?
            .into_iter()
            .flat_map(additive::combine_additive_vec)
            .collect::<Vec<_>>();

        println!("init_final_hashes: {:?}", init_final_hashes.len());

        // let rw_remaining_layers = (log_num_workers > 0).then(|| {
        //     Self::construct_remaining_layers(&mut read_write_hashes, read_write_effective_workers)
        // });

        // let init_final_remaining_layers = (log_num_workers > 0).then(|| {
        //     Self::construct_remaining_layers(&mut init_final_hashes, init_final_effective_workers)
        // });

        let multiset_hashes = Self::uninterleave_hashes(
            preprocessing,
            read_write_hashes.clone(),
            init_final_hashes.clone(),
        );

        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let read_write_circuit =
            Self::read_write_grand_product_rep3(preprocessing, num_lookups, log_num_workers);
        let init_final_circuit =
            Self::init_final_grand_product_rep3(preprocessing, memory_size, log_num_workers);

        network.set_worker_subnets(read_write_effective_workers);
        let (read_write_grand_product, r_read_write) = read_write_circuit
            .cooridinate_prove_grand_product(read_write_hashes, None, transcript, network)?;

        network.set_worker_subnets(init_final_effective_workers);
        let (init_final_grand_product, r_init_final) = init_final_circuit
            .cooridinate_prove_grand_product(init_final_hashes, None, transcript, network)?;

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
        r_read_write: Vec<F>,
        r_init_final: Vec<F>,
        preprocessing: &Self::Preprocessing,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(Self::Openings, Self::ExogenousOpenings)> {
        let mut exogenous_openings = Self::ExogenousOpenings::default();
        let mut openings = Self::Openings::initialize(preprocessing);

        let log_num_workers = network.log_num_workers_per_party();

        let read_write_openings: Vec<_> = openings
            .read_write_values_mut()
            .into_iter()
            .chain(exogenous_openings.openings_mut())
            .collect();

        let read_write_evals = if log_num_workers == 0 {
            additive::combine_additive_vec(network.receive_responses()?)
        } else {
            Self::compute_remaining_openings(r_read_write, read_write_openings.len(), network)?
        };

        read_write_openings
            .into_par_iter()
            .zip(read_write_evals.par_iter())
            .for_each(|(opening, eval)| {
                *opening = *eval;
            });

        Rep3ProverOpeningAccumulator::coordinate_with_known_claims(
            &read_write_evals,
            transcript,
            network,
        )?;

        let init_final_evals = if log_num_workers == 0 {
            additive::combine_additive_vec(network.receive_responses()?)
        } else {
            Self::compute_remaining_openings(
                r_init_final,
                openings.init_final_values().len(),
                network,
            )?
        };

        openings
            .init_final_values_mut()
            .into_par_iter()
            .zip(init_final_evals.par_iter())
            .for_each(|(opening, eval)| {
                *opening = *eval;
            });

        Rep3ProverOpeningAccumulator::coordinate_with_known_claims(
            &init_final_evals,
            transcript,
            network,
        )?;

        Ok((openings, exogenous_openings))
    }

    fn read_write_grand_product_rep3(
        _preprocessing: &Self::Preprocessing,
        num_lookups: usize,
        log_num_workers: usize,
    ) -> Self::Rep3ReadWriteGrandProduct {
        Self::Rep3ReadWriteGrandProduct::construct(num_lookups.log_2() - log_num_workers)
    }

    fn init_final_grand_product_rep3(
        _preprocessing: &Self::Preprocessing,
        memory_size: usize,
        log_num_workers: usize,
    ) -> Self::Rep3InitFinalGrandProduct {
        Self::Rep3InitFinalGrandProduct::construct(memory_size.log_2() - log_num_workers)
    }

    fn read_write_effective_workers(
        _preprocessing: &Self::Preprocessing,
        log_num_workers: usize,
    ) -> usize {
        1 << log_num_workers
    }

    fn init_final_effective_workers(
        _preprocessing: &Self::Preprocessing,
        log_num_workers: usize,
    ) -> usize {
        1 << log_num_workers
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

    fn compute_remaining_openings(
        mut r: Vec<F>,
        num_openings: usize,
        network: &mut Network,
    ) -> eyre::Result<Vec<F>> {
        let log_num_workers = network.log_num_workers_per_party();
        let r_remaining = r.split_off(r.len() - log_num_workers);

        let chi = EqPolynomial::evals(&r_remaining);

        let read_write_evals: Vec<_> = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
            .into_iter()
            .map(additive::combine_additive_vec)
            .fold(vec![vec![]; num_openings], |mut evals, eval| {
                izip!(evals.iter_mut(), eval).for_each(|(a, b)| a.push(b));
                evals
            })
            .into_par_iter()
            .map(|evals| DensePolynomial::new(evals).evaluate_at_chi_low_optimized(&chi))
            .collect();

        Ok(read_write_evals)
    }
}

/// This type, used within a `StructuredPolynomialData` struct, indicates that the
/// field has a corresponding opening but no corresponding polynomial or commitment ––
/// the prover doesn't need to compute a witness polynomial or commitment because
/// the verifier can compute the opening on its own.
pub type VerifierComputedOpening<T> = Option<T>;
