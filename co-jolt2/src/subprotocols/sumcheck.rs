#![allow(clippy::type_complexity)]

use eyre::Context;
use jolt_core::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::transcripts::Transcript;
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;

use jolt_core::field::JoltField;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
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
        preproc: &mut PreprocessingPool<F>,
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
}

pub struct Rep3BatchedSumcheckWorker;

impl Rep3BatchedSumcheckWorker {
    #[tracing::instrument(skip_all, name = "BatchedSumcheck::prove")]
    pub fn prove<F, N>(
        instances: &mut [Box<dyn Rep3SumcheckInstanceWorker<F, N>>],
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
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

                instance.bind(r_j, local_round, io_ctx, preproc);

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

type HybridRoundMsg<F> = (Vec<AdditiveShare<F>>, Option<Vec<F>>);
type HybridOpeningsMsg<F> = Vec<(Vec<AdditiveShare<F>>, Option<Vec<F>>)>;

pub struct HybridBatchedSumcheckWorker;

impl HybridBatchedSumcheckWorker {
    #[tracing::instrument(skip_all, name = "HybridSumcheck::prove", level = "trace")]
    pub fn prove<F, N>(
        mut instances: Vec<BatchedSumcheckWorkerInstance<F, N>>,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
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
                        s.bind(r_j, local_round, io_ctx, preproc);
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
        for instance in instances.iter_mut() {
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

pub(crate) fn evaluate_univariate_at_share<F: JoltField>(
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
