#![allow(clippy::type_complexity)]

use eyre::Context;
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use jolt_core::transcripts::{AppendToTranscript, Transcript};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};

// ---------------------------------------------------------------------------
// Sumcheck instance traits (per-instance interface)
// ---------------------------------------------------------------------------

/// Worker-side sumcheck instance. Computes shared evaluations at each round
/// and accumulates openings for the final batch opening proof.
pub trait Rep3SumcheckInstanceWorker<F: JoltField>: Send {
    fn degree(&self) -> usize;
    fn num_rounds(&self) -> usize;

    /// The public input claim for this sumcheck instance.
    ///
    /// NOTE: This must be public for `Rep3BatchedSumcheckWorker` to deterministically
    /// initialize per-instance claim shares without extra communication.
    fn input_claim_public(&self) -> F;

    /// Compute the worker's share of the round polynomial evaluations.
    ///
    /// Returns `Vec<AdditivePrimeFieldShare<F>>` of length `max_degree`:
    /// evaluations at points {0, 2, 3, ..., max_degree}. (Max-degree padded.)
    fn compute_prover_message_share(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
    ) -> Vec<AdditiveShare<F>>;

    /// Bind the sumcheck variable for this round to challenge `r_j`.
    fn bind(&mut self, r_j: F::Challenge, round: usize);

    /// Normalize the low-to-high sumcheck opening point to big-endian form.
    fn normalize_opening_point(&self, opening_point: &[F::Challenge])
        -> OpeningPoint<BIG_ENDIAN, F>;

    /// After the sumcheck completes, cache polynomial openings in the accumulator and
    /// return the claim shares that were appended (in a stable, deterministic order).
    fn cache_openings_worker(
        &self,
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
    fn normalize_opening_point(&self, opening_point: &[F::Challenge])
        -> OpeningPoint<BIG_ENDIAN, F>;

    /// Cache polynomial openings into the coordinator accumulator (with transcript).
    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    );
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
        eyre::ensure!(!instances.is_empty(), "Batched sumcheck requires >= 1 instance");

        let max_num_rounds = instances
            .iter()
            .map(|s| s.num_rounds())
            .max()
            .unwrap();
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

        {
            let per_instance: Vec<String> = instances
                .iter()
                .enumerate()
                .map(|(i, s)| format!("i{}(rounds={}, raw_claim={:?}, scaled={:?})", i, s.num_rounds(), s.input_claim_public(), individual_claims[i]))
                .collect();
            eprintln!(
                "[MPC-coord] batched_claim={:?}, max_rounds={}, coeffs[0]={:?}, instances=[{}]",
                batched_claim, max_num_rounds,
                batching_coeffs.first(),
                per_instance.join(", "),
            );
        }

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

            let round_poly = UniPoly::<F>::from_evals(&full_evals);
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
    #[tracing::instrument(skip_all, name = "Rep3BatchedSumcheckWorker::prove", level = "trace")]
    pub fn prove<F, N>(
        instances: &mut [Box<dyn Rep3SumcheckInstanceWorker<F>>],
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<Vec<F::Challenge>>
    where
        F: JoltField,
        N: Rep3NetworkWorker,
    {
        eyre::ensure!(!instances.is_empty(), "Batched sumcheck requires >= 1 instance");

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

        // Per-instance additive claim shares, initialized from public input claims with
        // front-loaded scaling (vanilla batching semantics). This is equivalent to starting
        // with the unscaled input claim and applying the inactive-round `claim := claim/2`
        // update for the first `(max_num_rounds - num_rounds_i)` rounds.
        let mut individual_claims: Vec<AdditiveShare<F>> = instances
            .iter()
            .map(|instance| {
                let scaled = instance
                    .input_claim_public()
                    .mul_pow_2(max_num_rounds - instance.num_rounds());
                additive::promote_to_trivial_share(scaled, party_id)
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

                instance.bind(r_j, local_round);

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
        for instance in instances.iter() {
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
            let m = shares_by_party
                .get(0)
                .map(|v| v.len())
                .unwrap_or_default();
            let opened: Vec<Vec<F>> = (0..m)
                .map(|i| {
                    let shares_for_instance: Vec<Vec<AdditiveShare<F>>> =
                        shares_by_party.iter().map(|party| party[i].clone()).collect();
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
                        eyre::ensure!(t.len() == o.len(), "opening claim len mismatch across subnets");
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
