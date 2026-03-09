use eyre::Context;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::{AppendToTranscript, Transcript};
use jolt_core::zkvm::instruction_lookups::{D, LOG_K_CHUNK};

const LOG_K: usize = D * LOG_K_CHUNK;
use jolt_core::zkvm::witness::CommittedPolynomial;
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use rayon::prelude::*;
use co_jolt2::zkvm::instruction_lookups::ra_virtual::Rep3InstructionRaSumcheckWorker;

use crate::field::JoltField;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::poly::ra_poly::{shifted_table_from_rand_ohv, Rep3RaPolynomial};
use crate::subprotocols::mles_product_sum::compute_mles_product_16_rep3;
use crate::zkvm::dag::stage::Rep3SumcheckInstance;
use std::sync::Arc;
use tracing::trace_span;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3InstructionRaSumcheck<F: JoltField> {
    input_claim: F,
    r_cycle: Vec<F::Challenge>,
    r_address_chunks: Vec<Vec<F::Challenge>>,
}

impl<F: JoltField> Rep3InstructionRaSumcheck<F> {
    pub fn new(
        input_claim: F,
        r_cycle: Vec<F::Challenge>,
        r_address_chunks: Vec<Vec<F::Challenge>>,
    ) -> Self {
        Self {
            input_claim,
            r_cycle,
            r_address_chunks,
        }
    }

    pub fn degree(&self) -> usize {
        D + 1
    }

    pub fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    pub fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    pub fn cache_openings<T: Transcript>(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        for (i, r_address_chunk) in self.r_address_chunks.iter().enumerate() {
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                r_address_chunk,
                &opening_point.r,
                vec![claims[i]],
            );
        }
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3InstructionRaSumcheck<F> {
    fn degree(&self) -> usize {
        D + 1
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim_public(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let eq_eval = EqPolynomial::<F>::mle(&self.r_cycle, r);
        let ra_claim_prod: F = (0..D)
            .map(|i| {
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::InstructionRa(i),
                        SumcheckId::InstructionRaVirtualization,
                    )
                    .1
            })
            .product();
        eq_eval * ra_claim_prod
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        for (i, r_address_chunk) in self.r_address_chunks.iter().enumerate() {
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                r_address_chunk,
                &opening_point.r,
                vec![claims[i]],
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Dedicated stage 4 proving loops
// ---------------------------------------------------------------------------

/// Worker-side stage 4 proving loop for the RA virtualization sumcheck.
///
/// This runs a single-instance sumcheck with `io_ctx` access for resharing.
pub fn prove_worker<F, N>(
    worker: &mut Rep3InstructionRaSumcheckWorker<F>,
    accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
    io_ctx: &mut IoContextPool<N>,
) -> eyre::Result<Vec<F::Challenge>>
where
    F: JoltField,
    N: Rep3NetworkWorker,
{
    let party_id = io_ctx.party_id();
    let num_rounds = worker.num_rounds_inner();
    let degree = worker.degree_inner();

    let mut claim: AdditiveShare<F> =
        additive::promote_to_trivial_share(worker.input_claim_public(), party_id);
    let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let msg = worker.compute_prover_message_share(round, claim, io_ctx)?;

        let r_j: F::Challenge = io_ctx
            .network()
            .exchange(msg.clone())
            .context("exchange RA round evals")?;
        r_sumcheck.push(r_j);

        worker.bind_inner(r_j);

        // Update claim: reconstruct UniPoly from {0,1,...,degree} and evaluate at r_j
        claim = evaluate_univariate_at_share::<F>(degree, claim, &msg, r_j)?;
    }

    // Cache openings and send claims to coordinator.
    let opening_point = worker.normalize_opening_point_inner(&r_sumcheck);
    let rep3_claims = worker.cache_openings_worker_inner(accumulator, opening_point);
    let additive_claims: Vec<AdditiveShare<F>> = rep3_claims
        .into_iter()
        .map(Rep3PrimeFieldShare::into_additive)
        .collect();
    io_ctx
        .network()
        .send_response(vec![additive_claims])
        .context("send RA opening claims")?;

    Ok(r_sumcheck)
}

/// Coordinator-side stage 4 proving loop for the RA virtualization sumcheck.
pub fn prove_coordinator<F, ProofTranscript, N>(
    coord: &Rep3InstructionRaSumcheck<F>,
    accumulator: &mut Rep3OpeningAccumulator<F>,
    transcript: &mut ProofTranscript,
    network: &mut N,
) -> eyre::Result<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>)>
where
    F: JoltField,
    ProofTranscript: Transcript,
    N: Rep3NetworkCoordinator,
{
    let num_rounds = coord.num_rounds();
    let degree = coord.degree();

    let mut batched_claim = coord.input_claim;
    transcript.append_scalar(&batched_claim);

    let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    for _round in 0..num_rounds {
        let round_evals = receive_round_evals::<F, N>(network)?;
        eyre::ensure!(
            round_evals.len() == degree,
            "round evals len mismatch: expected {degree}, got {}",
            round_evals.len()
        );

        // Convert {0,2,3,...,D+1} to {0,1,2,...,D+1}.
        let mut full_evals = Vec::with_capacity(degree + 1);
        full_evals.push(round_evals[0]);
        full_evals.push(batched_claim - round_evals[0]); // eval at 1
        full_evals.extend(round_evals.into_iter().skip(1));

        let round_poly = UniPoly::<F>::from_evals(&full_evals);
        let compressed_poly = round_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar_optimized::<F>();
        r_sumcheck.push(r_j);

        batched_claim = round_poly.evaluate(&r_j);

        network
            .broadcast_request(r_j)
            .context("broadcast RA round challenge")?;
    }

    // Receive opening claims.
    let opening_claims = receive_opening_claims::<F, N>(network)?;
    eyre::ensure!(
        opening_claims.len() == 1,
        "expected 1 instance, got {}",
        opening_claims.len()
    );
    let claims = opening_claims.into_iter().next().unwrap();

    let opening_point = coord.normalize_opening_point(&r_sumcheck);
    coord.cache_openings(accumulator, transcript, opening_point, claims);

    Ok((SumcheckInstanceProof::new(compressed_polys), r_sumcheck))
}

// ---------------------------------------------------------------------------
// Network helpers
// ---------------------------------------------------------------------------

fn receive_round_evals<F, N>(network: &mut N) -> eyre::Result<Vec<F>>
where
    F: JoltField,
    N: Rep3NetworkCoordinator,
{
    let shares = network
        .receive_responses::<Vec<AdditiveShare<F>>>()
        .context("receive RA round eval shares")?;
    Ok(additive::combine_additive_vec(shares))
}

fn receive_opening_claims<F, N>(network: &mut N) -> eyre::Result<Vec<Vec<F>>>
where
    F: JoltField,
    N: Rep3NetworkCoordinator,
{
    let shares = network
        .receive_responses::<Vec<Vec<AdditiveShare<F>>>>()
        .context("receive RA opening claim shares")?;

    // shares: 3 parties, each Vec<Vec<AdditiveShare<F>>>
    // For the RA sumcheck there is 1 instance with D claims.
    let [s0, s1, s2]: [Vec<Vec<AdditiveShare<F>>>; 3] = shares.try_into().unwrap();
    let num_instances = s0.len();
    let mut result = Vec::with_capacity(num_instances);
    for i in 0..num_instances {
        let claims = additive::combine_additive_shares(&s0[i], &s1[i], &s2[i]);
        result.push(claims);
    }
    Ok(result)
}

/// Evaluate a degree-d univariate polynomial (given by round-message evals at {0,2,...,d})
/// combined with the claim-derived eval at 1, at challenge point x.
fn evaluate_univariate_at_share<F: JoltField>(
    degree: usize,
    previous_claim: AdditiveShare<F>,
    msg_evals: &[AdditiveShare<F>],
    x: F::Challenge,
) -> eyre::Result<AdditiveShare<F>> {
    eyre::ensure!(degree >= 1, "sumcheck degree must be >= 1");
    eyre::ensure!(
        msg_evals.len() >= degree,
        "msg evals length must be >= degree"
    );

    let mut full_evals: Vec<AdditiveShare<F>> = Vec::with_capacity(degree + 1);
    full_evals.push(msg_evals[0]);
    full_evals.push(previous_claim - msg_evals[0]);
    full_evals.extend((2..=degree).map(|k| msg_evals[k - 1]));

    let evals_as_fe: Vec<F> = AdditiveShare::into_fe_vec(full_evals);
    let poly = UniPoly::<F>::from_evals(&evals_as_fe);
    Ok(AdditiveShare::from_fe(poly.evaluate(&x)))
}
