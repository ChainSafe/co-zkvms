#![allow(clippy::type_complexity)]

use eyre::Context;
use jolt_core::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof};
use jolt_core::transcripts::{AppendToTranscript, KeccakTranscript, Transcript};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::utils::types::Rep3Value;

// ---------------------------------------------------------------------------
// Sumcheck instance traits (per-instance interface)
// ---------------------------------------------------------------------------

/// Worker-side sumcheck instance. Computes shared evaluations at each round
/// and accumulates openings for the final batch opening proof.
pub trait Rep3SumcheckInstanceWorker<F: JoltField, N: Rep3NetworkWorker>: Send {
    fn degree(&self) -> usize;
    fn num_rounds(&self) -> usize;

    /// The input claim for this sumcheck instance.
    ///
    /// Returns `Rep3Value::Public(F)` for instances where the input claim is publicly
    /// known (initialized via `promote_to_trivial_share`), or `Rep3Value::Shared(share)`
    /// for instances with secret-shared input claims (initialized via `into_additive()`).
    fn input_claim(&self) -> Rep3Value<F>;

    /// Compute the worker's share of the round polynomial evaluations.
    ///
    /// Returns `Vec<AdditivePrimeFieldShare<F>>` of length `max_degree`:
    /// evaluations at points {0, 2, 3, ..., max_degree}. (Max-degree padded.)
    fn compute_prover_message_share(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
        io_ctx: &mut IoContextPool<N>,
    ) -> Vec<AdditiveShare<F>>;

    /// Bind the sumcheck variable for this round to challenge `r_j`.
    fn bind(
        &mut self,
        r_j: F::Challenge,
        round: usize,
        io_ctx: &mut IoContextPool<N>,
        edabits_pool: &mut PreprocessingPool<F>,
    );

    /// Normalize the low-to-high sumcheck opening point to big-endian form.
    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F>;

    /// After the sumcheck completes, cache polynomial openings in the accumulator and
    /// return the claim shares that were appended (in a stable, deterministic order).
    fn cache_openings_worker(
        &mut self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>>;
}

/// Coordinator-side sumcheck instance. Drives the Fiat-Shamir transcript
/// and verifies claims at the end of each sumcheck.
pub trait Rep3SumcheckInstance<F: JoltField, T: Transcript> {
    fn degree(&self) -> usize;
    fn num_rounds(&self) -> usize;

    /// The public input claim for this sumcheck instance.
    fn input_claim_public(&self) -> F;

    /// Compute the expected output claim after all rounds are complete,
    /// using the accumulated opening values.
    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F;

    /// Normalize the low-to-high sumcheck opening point to big-endian form.
    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F>;

    /// Cache polynomial openings into the coordinator accumulator (with transcript).
    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    );
}

// ---------------------------------------------------------------------------
// Public sumcheck instance traits (single-worker public computation)
// ---------------------------------------------------------------------------

/// Worker-side public sumcheck instance.
///
/// This is for sumchecks whose prover messages are computable from public data.
/// Only a single designated worker (PartyID::ID0, per subnet) should execute these
/// instances and send public evaluations/claims to the coordinator.
pub trait PublicSumcheckInstanceWorker<F: JoltField>: Send {
    fn degree(&self) -> usize;
    fn num_rounds(&self) -> usize;

    /// The public input claim for this sumcheck instance.
    fn input_claim_public(&self) -> F;

    /// Compute public round evaluations at points {0, 2, 3, ..., max_degree}.
    ///
    /// Returns a vector of length `max_degree`.
    fn compute_prover_message_public(
        &mut self,
        round: usize,
        previous_claim: F,
        max_degree: usize,
    ) -> Vec<F>;

    /// Bind the sumcheck variable for this round to challenge `r_j`.
    fn bind(&mut self, r_j: F::Challenge, round: usize);

    /// Normalize the low-to-high sumcheck opening point to big-endian form.
    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F>;

    /// Cache polynomial openings into the worker accumulator and return the PUBLIC
    /// claims appended (stable order).
    ///
    /// ID0 computes actual claims from prover state; non-ID0 stores zero shares
    /// (they lack prover state but need matching accumulator structure for the
    /// opening-proof reduction).
    fn cache_openings_public(
        &self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        party_id: PartyID,
    ) -> Vec<F>;
}

/// Coordinator-side public sumcheck instance.
///
/// Mirrors `Rep3SumcheckInstance` but is used for instances whose worker-side
/// computation is done publicly by a single designated worker.
pub trait PublicSumcheckInstance<F: JoltField, T: Transcript> {
    fn degree(&self) -> usize;
    fn num_rounds(&self) -> usize;
    fn input_claim_public(&self) -> F;

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F;

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F>;

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    );
}

// ---------------------------------------------------------------------------
// Batched instance wrappers
// ---------------------------------------------------------------------------

pub enum BatchedSumcheckWorkerInstance<F: JoltField, N: Rep3NetworkWorker> {
    Secret(Box<dyn Rep3SumcheckInstanceWorker<F, N>>),
    Public(Box<dyn PublicSumcheckInstanceWorker<F>>),
}

impl<F: JoltField, N: Rep3NetworkWorker> BatchedSumcheckWorkerInstance<F, N> {
    fn degree(&self) -> usize {
        match self {
            BatchedSumcheckWorkerInstance::Secret(s) => s.degree(),
            BatchedSumcheckWorkerInstance::Public(s) => s.degree(),
        }
    }
    fn num_rounds(&self) -> usize {
        match self {
            BatchedSumcheckWorkerInstance::Secret(s) => s.num_rounds(),
            BatchedSumcheckWorkerInstance::Public(s) => s.num_rounds(),
        }
    }
    fn input_claim(&self) -> Rep3Value<F> {
        match self {
            BatchedSumcheckWorkerInstance::Secret(s) => s.input_claim(),
            BatchedSumcheckWorkerInstance::Public(s) => Rep3Value::Public(s.input_claim_public()),
        }
    }
}

pub enum BatchedSumcheckInstance<F: JoltField, T: Transcript> {
    Secret(Box<dyn Rep3SumcheckInstance<F, T>>),
    Public(Box<dyn PublicSumcheckInstance<F, T>>),
}

impl<F: JoltField, T: Transcript> BatchedSumcheckInstance<F, T> {
    fn degree(&self) -> usize {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.degree(),
            BatchedSumcheckInstance::Public(s) => s.degree(),
        }
    }
    fn num_rounds(&self) -> usize {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.num_rounds(),
            BatchedSumcheckInstance::Public(s) => s.num_rounds(),
        }
    }
    fn input_claim_public(&self) -> F {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.input_claim_public(),
            BatchedSumcheckInstance::Public(s) => s.input_claim_public(),
        }
    }
    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.normalize_opening_point(opening_point),
            BatchedSumcheckInstance::Public(s) => s.normalize_opening_point(opening_point),
        }
    }
    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        match self {
            BatchedSumcheckInstance::Secret(s) => {
                s.cache_openings(accumulator, transcript, opening_point, claims)
            }
            BatchedSumcheckInstance::Public(s) => {
                s.cache_openings(accumulator, transcript, opening_point, claims)
            }
        }
    }
}

pub struct Rep3BatchedSumcheck;

impl Rep3BatchedSumcheck {
    #[tracing::instrument(skip_all, name = "Rep3BatchedSumcheck::prove", level = "trace")]
    pub fn prove<F, ProofTranscript, N>(
        instances: &[Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>],
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        network: &mut N,
    ) -> eyre::Result<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>)>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        N: Rep3NetworkCoordinator,
    {
        eyre::ensure!(
            !instances.is_empty(),
            "Batched sumcheck requires >= 1 instance"
        );

        let max_num_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
        let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(instances.len());
        network
            .broadcast_request(batching_coeffs.clone())
            .context("broadcast batching coeffs")?;

        let individual_claims: Vec<F> = instances
            .iter()
            .map(|instance| {
                let input_claim = instance.input_claim_public();
                transcript.append_scalar(&input_claim);
                input_claim.mul_pow_2(max_num_rounds - instance.num_rounds())
            })
            .collect();

        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            let round_evals = receive_batched_round_evals::<F, N>(network)?;
            eyre::ensure!(
                round_evals.len() == max_degree,
                "round evals len mismatch: expected {max_degree}, got {}",
                round_evals.len()
            );

            // Convert {0,2,3,...,D} to {0,1,2,...,D}.
            let mut full_evals = Vec::with_capacity(max_degree + 1);
            full_evals.push(round_evals[0]);
            full_evals.push(batched_claim - round_evals[0]); // eval at 1
            full_evals.extend(round_evals.into_iter().skip(1));

            let mut round_poly = UniPoly::<F>::from_evals(&full_evals);
            // Trim trailing zero coefficients so the polynomial degree matches
            // vanilla's per-round adaptive degree (vanilla produces variable-degree
            // polys because each instance contributes its own degree, while MPC
            // always pads to max_degree).
            while round_poly.coeffs.len() > 1 && round_poly.coeffs.last() == Some(&F::zero()) {
                round_poly.coeffs.pop();
            }
            let compressed_poly = round_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            batched_claim = round_poly.evaluate(&r_j);

            network
                .broadcast_request(r_j)
                .context("broadcast round challenge")?;
        }

        // Receive opening-claim shares from workers and cache openings per instance.
        let opening_claims = receive_opening_claims::<F, N>(network)?;
        eyre::ensure!(
            opening_claims.len() == instances.len(),
            "opening claims instance count mismatch: expected {}, got {}",
            instances.len(),
            opening_claims.len()
        );

        for (i, instance) in instances.iter().enumerate() {
            let num_rounds = instance.num_rounds();
            let r_slice = &r_sumcheck[max_num_rounds - num_rounds..];
            let opening_point = instance.normalize_opening_point(r_slice);
            instance.cache_openings(
                accumulator,
                transcript,
                opening_point,
                opening_claims[i].clone(),
            );
        }

        Ok((SumcheckInstanceProof::new(compressed_polys), r_sumcheck))
    }
}

pub struct Rep3BatchedSumcheckWorker;

impl Rep3BatchedSumcheckWorker {
    #[tracing::instrument(skip_all, name = "prove_batched_sumcheck")]
    pub fn prove<F, N>(
        instances: &mut [Box<dyn Rep3SumcheckInstanceWorker<F, N>>],
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<N>,
        edabits_pool: &mut PreprocessingPool<F>,
    ) -> eyre::Result<Vec<F::Challenge>>
    where
        F: JoltField,
        N: Rep3NetworkWorker,
    {
        eyre::ensure!(
            !instances.is_empty(),
            "Batched sumcheck requires >= 1 instance"
        );

        let party_id = io_ctx.party_id();

        let max_num_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
        let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();

        let batching_coeffs: Vec<F> = io_ctx
            .network()
            .receive_request()
            .context("receive batching coeffs")?;
        eyre::ensure!(
            batching_coeffs.len() == instances.len(),
            "batching coeffs len mismatch: expected {}, got {}",
            instances.len(),
            batching_coeffs.len()
        );

        let inv2 = F::TWO_INV;

        // Per-instance additive claim shares, initialized with front-loaded scaling
        // (vanilla batching semantics). Public claims are promoted to trivial shares;
        // secret-shared claims are converted to additive shares directly.
        let mut individual_claims: Vec<AdditiveShare<F>> = instances
            .iter()
            .map(|instance| {
                let padding = max_num_rounds - instance.num_rounds();
                match instance.input_claim() {
                    Rep3Value::Public(f) => {
                        additive::promote_to_trivial_share(f.mul_pow_2(padding), party_id)
                    }
                    Rep3Value::Shared(share) => Rep3PrimeFieldShare::new(
                        share.a.mul_pow_2(padding),
                        share.b.mul_pow_2(padding),
                    )
                    .into_additive(),
                    Rep3Value::Additive(a) => {
                        AdditiveShare::from_fe(a.into_fe().mul_pow_2(padding))
                    }
                }
            })
            .collect();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            let remaining_rounds = max_num_rounds - round;

            let mut batched_evals = vec![AdditiveShare::<F>::zero(); max_degree];
            let mut active_round_msgs: Vec<Option<Vec<AdditiveShare<F>>>> =
                vec![None; instances.len()];

            for (i, instance) in instances.iter_mut().enumerate() {
                let num_rounds = instance.num_rounds();

                if remaining_rounds > num_rounds {
                    // Inactive instance: constant polynomial of value claim/2.
                    let c = individual_claims[i] * inv2;
                    individual_claims[i] = c;

                    for eval in batched_evals.iter_mut() {
                        *eval += c * batching_coeffs[i];
                    }
                    continue;
                }

                let offset = max_num_rounds - num_rounds;
                let local_round = round - offset;

                let msg = instance.compute_prover_message_share(
                    local_round,
                    individual_claims[i],
                    max_degree,
                    io_ctx,
                );
                eyre::ensure!(
                    msg.len() == max_degree,
                    "instance message len mismatch: expected {max_degree}, got {}",
                    msg.len()
                );
                active_round_msgs[i] = Some(msg.clone());

                for (dst, src) in batched_evals.iter_mut().zip(msg.iter()) {
                    *dst += *src * batching_coeffs[i];
                }
            }

            let r_j: F::Challenge = io_ctx
                .network()
                .exchange(batched_evals)
                .context("exchange round evals")?;
            r_sumcheck.push(r_j);

            for (i, instance) in instances.iter_mut().enumerate() {
                let num_rounds = instance.num_rounds();
                if remaining_rounds > num_rounds {
                    continue;
                }
                let offset = max_num_rounds - num_rounds;
                let local_round = round - offset;

                instance.bind(r_j, local_round, io_ctx, edabits_pool);

                let msg = active_round_msgs[i]
                    .take()
                    .unwrap_or_else(|| unreachable!("active msg missing"));

                individual_claims[i] = evaluate_univariate_at_share::<F>(
                    instance.degree(),
                    individual_claims[i],
                    &msg,
                    r_j,
                )?;
            }
        }

        // Cache openings and send opening-claim shares to coordinator.
        let mut opening_claims_by_instance: Vec<Vec<AdditiveShare<F>>> =
            Vec::with_capacity(instances.len());
        for instance in instances.iter_mut() {
            let num_rounds = instance.num_rounds();
            let r_slice = &r_sumcheck[max_num_rounds - num_rounds..];
            let opening_point = instance.normalize_opening_point(r_slice);
            let rep3_claims = instance.cache_openings_worker(accumulator, opening_point);
            opening_claims_by_instance.push(
                rep3_claims
                    .into_iter()
                    .map(Rep3PrimeFieldShare::into_additive)
                    .collect(),
            );
        }

        io_ctx
            .network()
            .send_response(opening_claims_by_instance)
            .context("send opening claim shares")?;

        Ok(r_sumcheck)
    }
}

// ---------------------------------------------------------------------------
// Hybrid batched sumcheck (secret + public-only instances)
// ---------------------------------------------------------------------------

pub struct HybridBatchedSumcheck;

type HybridRoundMsg<F> = (Vec<AdditiveShare<F>>, Option<Vec<F>>);
type HybridOpeningsMsg<F> = Vec<(Vec<AdditiveShare<F>>, Option<Vec<F>>)>;

impl HybridBatchedSumcheck {
    #[tracing::instrument(skip_all, name = "HybridBatchedSumcheck::prove", level = "trace")]
    #[allow(unused_variables)]
    pub fn prove<F, ProofTranscript, N>(
        instances: &[BatchedSumcheckInstance<F, ProofTranscript>],
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        network: &mut N,
    ) -> eyre::Result<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>)>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        N: Rep3NetworkCoordinator,
    {
        eyre::ensure!(
            !instances.is_empty(),
            "Batched sumcheck requires >= 1 instance"
        );

        let max_num_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
        let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(instances.len());
        network
            .broadcast_request(batching_coeffs.clone())
            .context("broadcast batching coeffs")?;

        let individual_claims: Vec<F> = instances
            .iter()
            .map(|instance| {
                let input_claim = instance.input_claim_public();
                transcript.append_scalar(&input_claim);
                input_claim.mul_pow_2(max_num_rounds - instance.num_rounds())
            })
            .collect();

        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);

        for _round in 0..max_num_rounds {
            let round_evals = receive_hybrid_round_evals::<F, N>(network)?;
            eyre::ensure!(
                round_evals.len() == max_degree,
                "round evals len mismatch: expected {max_degree}, got {}",
                round_evals.len()
            );

            // Convert {0,2,3,...,D} to {0,1,2,...,D}.
            let mut full_evals = Vec::with_capacity(max_degree + 1);
            full_evals.push(round_evals[0]);
            full_evals.push(batched_claim - round_evals[0]); // eval at 1
            full_evals.extend(round_evals.into_iter().skip(1));

            let mut round_poly = UniPoly::<F>::from_evals(&full_evals);
            // Trim trailing zero coefficients so the polynomial degree matches
            // vanilla's per-round adaptive degree (vanilla produces variable-degree
            // polys because each instance contributes its own degree, while MPC
            // always pads to max_degree).
            while round_poly.coeffs.len() > 1 && round_poly.coeffs.last() == Some(&F::zero()) {
                round_poly.coeffs.pop();
            }
            let compressed_poly = round_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            batched_claim = round_poly.evaluate(&r_j);

            network
                .broadcast_request(r_j)
                .context("broadcast round challenge")?;
        }

        let opening_claims = receive_hybrid_opening_claims::<F, N>(network)?;
        eyre::ensure!(
            opening_claims.len() == instances.len(),
            "opening claims instance count mismatch: expected {}, got {}",
            instances.len(),
            opening_claims.len()
        );

        for (i, instance) in instances.iter().enumerate() {
            let num_rounds = instance.num_rounds();
            let r_slice = &r_sumcheck[max_num_rounds - num_rounds..];
            let opening_point = instance.normalize_opening_point(r_slice);
            instance.cache_openings(
                accumulator,
                transcript,
                opening_point,
                opening_claims[i].clone(),
            );
        }

        Ok((SumcheckInstanceProof::new(compressed_polys), r_sumcheck))
    }
}

pub struct HybridBatchedSumcheckWorker;

impl HybridBatchedSumcheckWorker {
    #[tracing::instrument(skip_all, name = "HybridBatchedSumcheckWorker::prove", level = "trace")]
    pub fn prove<F, N>(
        instances: &mut [BatchedSumcheckWorkerInstance<F, N>],
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<N>,
        edabits_pool: &mut PreprocessingPool<F>,
    ) -> eyre::Result<Vec<F::Challenge>>
    where
        F: JoltField,
        N: Rep3NetworkWorker,
    {
        eyre::ensure!(
            !instances.is_empty(),
            "Batched sumcheck requires >= 1 instance"
        );

        let party_id = io_ctx.party_id();
        let is_public_worker = party_id == mpc_core::protocols::rep3::PartyID::ID0;

        let max_num_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
        let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();

        let batching_coeffs: Vec<F> = io_ctx
            .network()
            .receive_request()
            .context("receive batching coeffs")?;
        eyre::ensure!(
            batching_coeffs.len() == instances.len(),
            "batching coeffs len mismatch: expected {}, got {}",
            instances.len(),
            batching_coeffs.len()
        );

        let inv2 = F::TWO_INV;

        let mut secret_claims: Vec<Option<AdditiveShare<F>>> = Vec::with_capacity(instances.len());
        let mut public_claims: Vec<Option<F>> = Vec::with_capacity(instances.len());
        for instance in instances.iter() {
            match instance {
                BatchedSumcheckWorkerInstance::Secret(s) => {
                    let padding = max_num_rounds - s.num_rounds();
                    let claim = match s.input_claim() {
                        Rep3Value::Public(f) => {
                            additive::promote_to_trivial_share(f.mul_pow_2(padding), party_id)
                        }
                        Rep3Value::Shared(share) => Rep3PrimeFieldShare::new(
                            share.a.mul_pow_2(padding),
                            share.b.mul_pow_2(padding),
                        )
                        .into_additive(),
                        Rep3Value::Additive(a) => {
                            AdditiveShare::from_fe(a.into_fe().mul_pow_2(padding))
                        }
                    };
                    secret_claims.push(Some(claim));
                    public_claims.push(None);
                }
                BatchedSumcheckWorkerInstance::Public(s) => {
                    secret_claims.push(None);
                    if is_public_worker {
                        let scaled = s
                            .input_claim_public()
                            .mul_pow_2(max_num_rounds - s.num_rounds());
                        public_claims.push(Some(scaled));
                    } else {
                        public_claims.push(None);
                    }
                }
            }
        }

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            let remaining_rounds = max_num_rounds - round;

            let mut batched_secret_evals = vec![AdditiveShare::<F>::zero(); max_degree];
            let mut batched_public_evals = if is_public_worker {
                Some(vec![F::zero(); max_degree])
            } else {
                None
            };

            let mut active_secret_msgs: Vec<Option<Vec<AdditiveShare<F>>>> =
                vec![None; instances.len()];
            let mut active_public_msgs: Vec<Option<Vec<F>>> = vec![None; instances.len()];

            for (i, instance) in instances.iter_mut().enumerate() {
                let num_rounds = instance.num_rounds();

                if remaining_rounds > num_rounds {
                    // Inactive instance: constant polynomial of value claim/2.
                    match instance {
                        BatchedSumcheckWorkerInstance::Secret(_) => {
                            let c = secret_claims[i].unwrap() * inv2;
                            secret_claims[i] = Some(c);
                            for eval in batched_secret_evals.iter_mut() {
                                *eval += c * batching_coeffs[i];
                            }
                        }
                        BatchedSumcheckWorkerInstance::Public(_) => {
                            if is_public_worker {
                                let c = public_claims[i].unwrap() * inv2;
                                public_claims[i] = Some(c);
                                if let Some(ref mut v) = batched_public_evals {
                                    for eval in v.iter_mut() {
                                        *eval += c * batching_coeffs[i];
                                    }
                                }
                            }
                        }
                    }
                    continue;
                }

                let offset = max_num_rounds - num_rounds;
                let local_round = round - offset;

                match instance {
                    BatchedSumcheckWorkerInstance::Secret(s) => {
                        let prev = secret_claims[i].unwrap();
                        let msg =
                            s.compute_prover_message_share(local_round, prev, max_degree, io_ctx);
                        eyre::ensure!(
                            msg.len() == max_degree,
                            "instance message len mismatch: expected {max_degree}, got {}",
                            msg.len()
                        );
                        active_secret_msgs[i] = Some(msg.clone());
                        for (dst, src) in batched_secret_evals.iter_mut().zip(msg.iter()) {
                            *dst += *src * batching_coeffs[i];
                        }
                    }
                    BatchedSumcheckWorkerInstance::Public(p) => {
                        if !is_public_worker {
                            continue;
                        }
                        let prev = public_claims[i].unwrap();
                        let msg = p.compute_prover_message_public(local_round, prev, max_degree);
                        eyre::ensure!(
                            msg.len() == max_degree,
                            "public instance message len mismatch: expected {max_degree}, got {}",
                            msg.len()
                        );
                        active_public_msgs[i] = Some(msg.clone());
                        if let Some(ref mut v) = batched_public_evals {
                            for (dst, src) in v.iter_mut().zip(msg.iter()) {
                                *dst += *src * batching_coeffs[i];
                            }
                        }
                    }
                }
            }

            let r_j: F::Challenge = io_ctx
                .network()
                .exchange::<HybridRoundMsg<F>, F::Challenge>((
                    batched_secret_evals,
                    batched_public_evals,
                ))
                .context("exchange round evals")?;
            r_sumcheck.push(r_j);

            for (i, instance) in instances.iter_mut().enumerate() {
                let num_rounds = instance.num_rounds();
                if remaining_rounds > num_rounds {
                    continue;
                }
                let offset = max_num_rounds - num_rounds;
                let local_round = round - offset;

                match instance {
                    BatchedSumcheckWorkerInstance::Secret(s) => {
                        s.bind(r_j, local_round, io_ctx, edabits_pool);
                        let msg = active_secret_msgs[i]
                            .take()
                            .unwrap_or_else(|| unreachable!("active msg missing"));
                        secret_claims[i] = Some(evaluate_univariate_at_share::<F>(
                            s.degree(),
                            secret_claims[i].unwrap(),
                            &msg,
                            r_j,
                        )?);
                    }
                    BatchedSumcheckWorkerInstance::Public(p) => {
                        if !is_public_worker {
                            continue;
                        }
                        p.bind(r_j, local_round);
                        let msg = active_public_msgs[i]
                            .take()
                            .unwrap_or_else(|| unreachable!("active msg missing"));
                        public_claims[i] = Some(evaluate_univariate_at_public::<F>(
                            p.degree(),
                            public_claims[i].unwrap(),
                            &msg,
                            r_j,
                        ));
                    }
                }
            }
        }

        let mut openings_by_instance: HybridOpeningsMsg<F> = Vec::with_capacity(instances.len());
        for (i, instance) in instances.iter_mut().enumerate() {
            let num_rounds = instance.num_rounds();
            let r_slice = &r_sumcheck[max_num_rounds - num_rounds..];

            match instance {
                BatchedSumcheckWorkerInstance::Secret(s) => {
                    let opening_point = s.normalize_opening_point(r_slice);
                    let rep3_claims = s.cache_openings_worker(accumulator, opening_point);
                    openings_by_instance.push((
                        rep3_claims
                            .into_iter()
                            .map(Rep3PrimeFieldShare::into_additive)
                            .collect(),
                        None,
                    ));
                }
                BatchedSumcheckWorkerInstance::Public(p) => {
                    let opening_point = p.normalize_opening_point(r_slice);
                    let claims = p.cache_openings_public(accumulator, opening_point, party_id);
                    if is_public_worker {
                        openings_by_instance.push((vec![], Some(claims)));
                    } else {
                        openings_by_instance.push((vec![], None));
                    }
                }
            }
        }

        io_ctx
            .network()
            .send_response(openings_by_instance)
            .context("send opening claims")?;

        Ok(r_sumcheck)
    }
}

fn evaluate_univariate_at_public<F: JoltField>(
    degree: usize,
    previous_claim: F,
    msg_evals: &[F],
    x: F::Challenge,
) -> F {
    debug_assert!(degree >= 1);
    debug_assert!(msg_evals.len() >= degree);
    let mut full_evals: Vec<F> = Vec::with_capacity(degree + 1);
    full_evals.push(msg_evals[0]);
    full_evals.push(previous_claim - msg_evals[0]);
    full_evals.extend((2..=degree).map(|k| msg_evals[k - 1]));
    let poly = UniPoly::<F>::from_evals(&full_evals);
    poly.evaluate(&x)
}

fn receive_hybrid_round_evals<F, N>(network: &mut N) -> eyre::Result<Vec<F>>
where
    F: JoltField,
    N: Rep3NetworkCoordinator,
{
    if network.is_distributed() {
        let subnet_responses = network
            .receive_responses_from_subnets::<HybridRoundMsg<F>>()
            .context("receive round evals from subnets")?;

        let max_degree = subnet_responses
            .get(0)
            .and_then(|s| s.get(0))
            .map(|msg| msg.0.len())
            .unwrap_or(0);

        let mut total = vec![F::zero(); max_degree];

        for shares_by_party in subnet_responses {
            eyre::ensure!(shares_by_party.len() == 3, "expected 3 parties per subnet");
            let secret_parts: Vec<Vec<AdditiveShare<F>>> =
                shares_by_party.iter().map(|m| m.0.clone()).collect();
            let secret_open = additive::combine_additive_vec(secret_parts);
            total
                .iter_mut()
                .zip(secret_open.into_iter())
                .for_each(|(dst, src)| *dst += src);

            if let Some(ref public) = shares_by_party[0].1 {
                eyre::ensure!(public.len() == max_degree, "public evals len mismatch");
                total
                    .iter_mut()
                    .zip(public.iter())
                    .for_each(|(dst, src)| *dst += *src);
            }
        }

        Ok(total)
    } else {
        let responses = network
            .receive_responses::<HybridRoundMsg<F>>()
            .context("receive round evals")?;
        eyre::ensure!(responses.len() == 3, "expected 3 parties");

        let secret_parts: Vec<Vec<AdditiveShare<F>>> =
            responses.iter().map(|m| m.0.clone()).collect();
        let mut total = additive::combine_additive_vec(secret_parts);

        if let Some(ref public) = responses[0].1 {
            eyre::ensure!(public.len() == total.len(), "public evals len mismatch");
            total
                .iter_mut()
                .zip(public.iter())
                .for_each(|(dst, src)| *dst += *src);
        }
        Ok(total)
    }
}

fn receive_hybrid_opening_claims<F, N>(network: &mut N) -> eyre::Result<Vec<Vec<F>>>
where
    F: JoltField,
    N: Rep3NetworkCoordinator,
{
    if network.is_distributed() {
        let subnet_responses = network
            .receive_responses_from_subnets::<HybridOpeningsMsg<F>>()
            .context("receive opening claims from subnets")?;

        let mut acc: Option<Vec<Vec<F>>> = None;

        for shares_by_party in subnet_responses {
            eyre::ensure!(shares_by_party.len() == 3, "expected 3 parties per subnet");
            let m = shares_by_party[0].len();

            // Open secret shares instance-by-instance.
            let mut opened: Vec<Vec<F>> = Vec::with_capacity(m);
            for i in 0..m {
                let secret_lists: Vec<Vec<AdditiveShare<F>>> = shares_by_party
                    .iter()
                    .map(|party| party[i].0.clone())
                    .collect();
                let mut claims = if secret_lists.iter().all(|v| v.is_empty()) {
                    vec![]
                } else {
                    additive::combine_additive_vec(secret_lists)
                };
                if let Some(ref public_claims) = shares_by_party[0][i].1 {
                    if claims.is_empty() {
                        claims = public_claims.clone();
                    } else {
                        eyre::ensure!(
                            claims.len() == public_claims.len(),
                            "opening claims len mismatch"
                        );
                        for (c, p) in claims.iter_mut().zip(public_claims.iter()) {
                            *c += *p;
                        }
                    }
                }
                opened.push(claims);
            }

            match &mut acc {
                None => acc = Some(opened),
                Some(total) => {
                    eyre::ensure!(
                        total.len() == opened.len(),
                        "opening claims instance count mismatch across subnets"
                    );
                    for (t, o) in total.iter_mut().zip(opened.into_iter()) {
                        if t.is_empty() {
                            *t = o;
                            continue;
                        }
                        eyre::ensure!(
                            t.len() == o.len(),
                            "opening claim len mismatch across subnets"
                        );
                        for (tv, ov) in t.iter_mut().zip(o.into_iter()) {
                            *tv += ov;
                        }
                    }
                }
            }
        }
        Ok(acc.unwrap_or_default())
    } else {
        let responses = network
            .receive_responses::<HybridOpeningsMsg<F>>()
            .context("receive opening claims")?;
        eyre::ensure!(responses.len() == 3, "expected 3 parties");
        let m = responses[0].len();

        let mut opened: Vec<Vec<F>> = Vec::with_capacity(m);
        for i in 0..m {
            let secret_lists: Vec<Vec<AdditiveShare<F>>> =
                responses.iter().map(|party| party[i].0.clone()).collect();
            let mut claims = if secret_lists.iter().all(|v| v.is_empty()) {
                vec![]
            } else {
                additive::combine_additive_vec(secret_lists)
            };
            if let Some(ref public_claims) = responses[0][i].1 {
                if claims.is_empty() {
                    claims = public_claims.clone();
                } else {
                    eyre::ensure!(
                        claims.len() == public_claims.len(),
                        "opening claim len mismatch"
                    );
                    for (c, p) in claims.iter_mut().zip(public_claims.iter()) {
                        *c += *p;
                    }
                }
            }
            opened.push(claims);
        }
        Ok(opened)
    }
}

fn receive_batched_round_evals<F, N>(network: &mut N) -> eyre::Result<Vec<F>>
where
    F: JoltField,
    N: Rep3NetworkCoordinator,
{
    if network.is_distributed() {
        let subnet_responses = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()
            .context("receive round eval shares from subnets")?;
        let degree = subnet_responses
            .get(0)
            .and_then(|s| s.get(0))
            .map(|v| v.len())
            .unwrap_or(0);

        Ok(subnet_responses
            .into_iter()
            .map(additive::combine_additive_vec)
            .fold(vec![F::zero(); degree], |mut acc, evals| {
                acc.iter_mut()
                    .zip(evals.into_iter())
                    .for_each(|(dst, src)| *dst += src);
                acc
            }))
    } else {
        let shares = network
            .receive_responses::<Vec<AdditiveShare<F>>>()
            .context("receive round eval shares")?;
        Ok(additive::combine_additive_vec(shares))
    }
}

fn receive_opening_claims<F, N>(network: &mut N) -> eyre::Result<Vec<Vec<F>>>
where
    F: JoltField,
    N: Rep3NetworkCoordinator,
{
    if network.is_distributed() {
        let subnet_responses = network
            .receive_responses_from_subnets::<Vec<Vec<AdditiveShare<F>>>>()
            .context("receive opening claim shares from subnets")?;

        let mut acc: Option<Vec<Vec<F>>> = None;
        for shares_by_party in subnet_responses {
            let m = shares_by_party.get(0).map(|v| v.len()).unwrap_or_default();
            let opened: Vec<Vec<F>> = (0..m)
                .map(|i| {
                    let shares_for_instance: Vec<Vec<AdditiveShare<F>>> = shares_by_party
                        .iter()
                        .map(|party| party[i].clone())
                        .collect();
                    additive::combine_additive_vec(shares_for_instance)
                })
                .collect();

            match &mut acc {
                None => acc = Some(opened),
                Some(total) => {
                    eyre::ensure!(
                        total.len() == opened.len(),
                        "opening claims instance count mismatch across subnets"
                    );
                    for (t, o) in total.iter_mut().zip(opened.into_iter()) {
                        eyre::ensure!(
                            t.len() == o.len(),
                            "opening claim len mismatch across subnets"
                        );
                        for (tv, ov) in t.iter_mut().zip(o.into_iter()) {
                            *tv += ov;
                        }
                    }
                }
            }
        }
        Ok(acc.unwrap_or_default())
    } else {
        let responses = network
            .receive_responses::<Vec<Vec<AdditiveShare<F>>>>()
            .context("receive opening claim shares")?;
        let m = responses.get(0).map(|v| v.len()).unwrap_or_default();
        Ok((0..m)
            .map(|i| {
                let shares_for_instance: Vec<Vec<AdditiveShare<F>>> =
                    responses.iter().map(|party| party[i].clone()).collect();
                additive::combine_additive_vec(shares_for_instance)
            })
            .collect())
    }
}

fn evaluate_univariate_at_share<F: JoltField>(
    degree: usize,
    previous_claim: AdditiveShare<F>,
    msg_evals: &[AdditiveShare<F>],
    x: F::Challenge,
) -> eyre::Result<AdditiveShare<F>> {
    eyre::ensure!(degree >= 1, "sumcheck degree must be >= 1");
    eyre::ensure!(
        msg_evals.len() >= degree,
        "msg evals length must be >= degree (need points up to {degree})"
    );

    // Nodes are consecutive x = 0..degree:
    // - y(0) = msg_evals[0]
    // - y(1) = previous_claim - y(0)
    // - y(k) for k>=2 is msg_evals[k-1] (since msg is {0,2,3,...})
    let mut full_evals: Vec<AdditiveShare<F>> = Vec::with_capacity(degree + 1);
    full_evals.push(msg_evals[0]);
    full_evals.push(previous_claim - msg_evals[0]);
    full_evals.extend((2..=degree).map(|k| msg_evals[k - 1]));

    let evals_as_fe: Vec<F> = AdditiveShare::into_fe_vec(full_evals);
    let poly = UniPoly::<F>::from_evals(&evals_as_fe);
    Ok(AdditiveShare::from_fe(poly.evaluate(&x)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::Zero;
    use jolt_core::transcripts::KeccakTranscript;
    use mpc_core::protocols::rep3::test_utils::run_rep3_local_test_with_coordinator;
    use mpc_core::protocols::rep3::PartyID;

    struct ConstSecretWorker {
        input_claim: Fr,
        rounds: usize,
    }

    impl<N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<Fr, N> for ConstSecretWorker {
        fn degree(&self) -> usize {
            2
        }
        fn num_rounds(&self) -> usize {
            self.rounds
        }
        fn input_claim(&self) -> Rep3Value<Fr> {
            Rep3Value::Public(self.input_claim)
        }

        fn compute_prover_message_share(
            &mut self,
            _round: usize,
            previous_claim: AdditiveShare<Fr>,
            max_degree: usize,
            _io_ctx: &mut IoContextPool<N>,
        ) -> Vec<AdditiveShare<Fr>> {
            let c = previous_claim * Fr::TWO_INV;
            vec![c; max_degree]
        }

        fn bind(
            &mut self,
            _r_j: <Fr as jolt_core::field::JoltField>::Challenge,
            _round: usize,
            _io_ctx: &mut IoContextPool<N>,
            _edabits_pool: &mut PreprocessingPool<Fr>,
        ) {
        }

        fn normalize_opening_point(
            &self,
            opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
        ) -> OpeningPoint<BIG_ENDIAN, Fr> {
            OpeningPoint::new(opening_point.to_vec())
        }

        fn cache_openings_worker(
            &mut self,
            _accumulator: &mut Rep3OpeningAccumulatorWorker<Fr>,
            _opening_point: OpeningPoint<BIG_ENDIAN, Fr>,
        ) -> Vec<Rep3PrimeFieldShare<Fr>> {
            vec![Rep3PrimeFieldShare::zero_share()]
        }
    }

    struct ConstSecretCoord {
        input_claim: Fr,
        rounds: usize,
    }

    impl Rep3SumcheckInstance<Fr, KeccakTranscript> for ConstSecretCoord {
        fn degree(&self) -> usize {
            2
        }
        fn num_rounds(&self) -> usize {
            self.rounds
        }
        fn input_claim_public(&self) -> Fr {
            self.input_claim
        }

        fn expected_output_claim(
            &self,
            _accumulator: &Rep3OpeningAccumulator<Fr>,
            _r: &[<Fr as jolt_core::field::JoltField>::Challenge],
        ) -> Fr {
            Fr::zero()
        }

        fn normalize_opening_point(
            &self,
            opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
        ) -> OpeningPoint<BIG_ENDIAN, Fr> {
            OpeningPoint::new(opening_point.to_vec())
        }

        fn cache_openings(
            &self,
            _accumulator: &mut Rep3OpeningAccumulator<Fr>,
            transcript: &mut KeccakTranscript,
            _opening_point: OpeningPoint<BIG_ENDIAN, Fr>,
            claims: Vec<Fr>,
        ) {
            for c in claims {
                transcript.append_scalar(&c);
            }
        }
    }

    struct ConstPublicWorker {
        input_claim: Fr,
        rounds: usize,
    }

    impl PublicSumcheckInstanceWorker<Fr> for ConstPublicWorker {
        fn degree(&self) -> usize {
            2
        }
        fn num_rounds(&self) -> usize {
            self.rounds
        }
        fn input_claim_public(&self) -> Fr {
            self.input_claim
        }

        fn compute_prover_message_public(
            &mut self,
            _round: usize,
            previous_claim: Fr,
            max_degree: usize,
        ) -> Vec<Fr> {
            let c = previous_claim * Fr::TWO_INV;
            vec![c; max_degree]
        }

        fn bind(&mut self, _r_j: <Fr as jolt_core::field::JoltField>::Challenge, _round: usize) {}

        fn normalize_opening_point(
            &self,
            opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
        ) -> OpeningPoint<BIG_ENDIAN, Fr> {
            OpeningPoint::new(opening_point.to_vec())
        }

        fn cache_openings_public(
            &self,
            _accumulator: &mut Rep3OpeningAccumulatorWorker<Fr>,
            _opening_point: OpeningPoint<BIG_ENDIAN, Fr>,
            _party_id: PartyID,
        ) -> Vec<Fr> {
            vec![Fr::from_u64(123)]
        }
    }

    struct PanicPublicWorker;

    impl PublicSumcheckInstanceWorker<Fr> for PanicPublicWorker {
        fn degree(&self) -> usize {
            2
        }
        fn num_rounds(&self) -> usize {
            2
        }
        fn input_claim_public(&self) -> Fr {
            Fr::zero()
        }
        fn compute_prover_message_public(
            &mut self,
            _round: usize,
            _previous_claim: Fr,
            _max_degree: usize,
        ) -> Vec<Fr> {
            panic!("public worker should not run on non-ID0 parties")
        }
        fn bind(&mut self, _r_j: <Fr as jolt_core::field::JoltField>::Challenge, _round: usize) {
            panic!("public worker should not bind on non-ID0 parties")
        }
        fn normalize_opening_point(
            &self,
            opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
        ) -> OpeningPoint<BIG_ENDIAN, Fr> {
            OpeningPoint::new(opening_point.to_vec())
        }
        fn cache_openings_public(
            &self,
            _accumulator: &mut Rep3OpeningAccumulatorWorker<Fr>,
            _opening_point: OpeningPoint<BIG_ENDIAN, Fr>,
            _party_id: PartyID,
        ) -> Vec<Fr> {
            vec![]
        }
    }

    struct ConstPublicCoord {
        input_claim: Fr,
        rounds: usize,
    }

    impl PublicSumcheckInstance<Fr, KeccakTranscript> for ConstPublicCoord {
        fn degree(&self) -> usize {
            2
        }
        fn num_rounds(&self) -> usize {
            self.rounds
        }
        fn input_claim_public(&self) -> Fr {
            self.input_claim
        }

        fn expected_output_claim(
            &self,
            _accumulator: &Rep3OpeningAccumulator<Fr>,
            _r: &[<Fr as jolt_core::field::JoltField>::Challenge],
        ) -> Fr {
            Fr::zero()
        }

        fn normalize_opening_point(
            &self,
            opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
        ) -> OpeningPoint<BIG_ENDIAN, Fr> {
            OpeningPoint::new(opening_point.to_vec())
        }

        fn cache_openings(
            &self,
            _accumulator: &mut Rep3OpeningAccumulator<Fr>,
            transcript: &mut KeccakTranscript,
            _opening_point: OpeningPoint<BIG_ENDIAN, Fr>,
            claims: Vec<Fr>,
        ) {
            for c in claims {
                transcript.append_scalar(&c);
            }
        }
    }

    #[test]
    fn hybrid_batched_sumcheck_public_worker_id0_only() {
        let rounds = 2;
        let secret_claim = Fr::from_u64(10);
        let public_claim = Fr::from_u64(20);

        let (worker_out, coordinator_out) = run_rep3_local_test_with_coordinator(
            0,
            |_party_idx| (),
            || (),
            move |(), mut io_ctx| {
                let party_id = io_ctx.party_id();
                let mut acc = Rep3OpeningAccumulatorWorker::<Fr>::new(party_id);
                let mut instances: Vec<BatchedSumcheckWorkerInstance<Fr, _>> = vec![
                    BatchedSumcheckWorkerInstance::Secret(Box::new(ConstSecretWorker {
                        input_claim: secret_claim,
                        rounds,
                    })),
                    if party_id == PartyID::ID0 {
                        BatchedSumcheckWorkerInstance::Public(Box::new(ConstPublicWorker {
                            input_claim: public_claim,
                            rounds,
                        }))
                    } else {
                        BatchedSumcheckWorkerInstance::Public(Box::new(PanicPublicWorker))
                    },
                ];
                let mut pool = PreprocessingPool::empty(party_id);
                let r = HybridBatchedSumcheckWorker::prove(&mut instances, &mut acc, &mut io_ctx, &mut pool)?;
                Ok(r.len())
            },
            move |(), net| {
                let mut acc = Rep3OpeningAccumulator::<Fr>::new();
                let mut transcript = KeccakTranscript::new(b"hybrid_batched_test");
                let instances: Vec<BatchedSumcheckInstance<Fr, KeccakTranscript>> = vec![
                    BatchedSumcheckInstance::Secret(Box::new(ConstSecretCoord {
                        input_claim: secret_claim,
                        rounds,
                    })),
                    BatchedSumcheckInstance::Public(Box::new(ConstPublicCoord {
                        input_claim: public_claim,
                        rounds,
                    })),
                ];
                let (_proof, r) =
                    HybridBatchedSumcheck::prove(&instances, &mut acc, &mut transcript, net)?;
                Ok(r.len())
            },
        );

        assert_eq!(worker_out, [rounds, rounds, rounds]);
        assert_eq!(coordinator_out, rounds);
    }
}
