use eyre::Context;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::{AppendToTranscript, Transcript};
use jolt_core::zkvm::instruction_lookups::D;
use jolt_core::zkvm::witness::CommittedPolynomial;
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

use crate::field::JoltField;
use crate::poly::opening_proof::Rep3OpeningAccumulator;
use crate::zkvm::dag::stage::Rep3SumcheckInstance;

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
