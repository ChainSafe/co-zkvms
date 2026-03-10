use eyre::Context;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::{AppendToTranscript, Transcript};
use jolt_core::zkvm::instruction_lookups::{D, K_CHUNK, LOG_K_CHUNK};

const LOG_K: usize = D * LOG_K_CHUNK;
use jolt_core::zkvm::witness::CommittedPolynomial;
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use rayon::prelude::*;

use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::poly::ra_poly::{shifted_table_from_rand_ohv, Rep3RaPolynomial};
use crate::subprotocols::mles_product_sum::compute_mles_product_16_rep3;
use jolt_core::field::JoltField;
use std::sync::Arc;
use tracing::trace_span;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct Rep3InstructionRaSumcheckWorker<F: JoltField> {
    ra_i_polys: Vec<Rep3RaPolynomial<u8, F>>,
    r_cycle: Vec<F::Challenge>,
    r_sumcheck: Vec<F::Challenge>,
    input_claim: F,
    r_address_chunks: Vec<Vec<F::Challenge>>,
}

impl<F: JoltField> Rep3InstructionRaSumcheckWorker<F> {
    pub fn new(
        one_hot_polys: Arc<[Rep3OneHotPolynomial<F>; D]>,
        r_address: &[F::Challenge],
        r_cycle: Vec<F::Challenge>,
        input_claim: F,
    ) -> Self {
        assert_eq!(r_address.len(), LOG_K);

        let r_address_chunks: Vec<Vec<F::Challenge>> =
            r_address.chunks(LOG_K_CHUNK).map(|c| c.to_vec()).collect();

        let ra_i_polys: Vec<Rep3RaPolynomial<u8, F>> = (0..D)
            .into_par_iter()
            .map(|i| {
                let eq_u = EqPolynomial::evals(&r_address_chunks[i]);
                let shifted_table =
                    shifted_table_from_rand_ohv(&eq_u, &one_hot_polys[i].rand_ohv_e_field);
                Rep3RaPolynomial::new(one_hot_polys[i].masked_indices_c.clone(), shifted_table)
            })
            .collect();

        Self {
            ra_i_polys,
            r_cycle,
            r_sumcheck: vec![],
            input_claim,
            r_address_chunks,
        }
    }

    /// Compute the prover's share of the round polynomial evaluations.
    ///
    /// Returns evaluations at {0, 2, 3, ..., degree} as `Vec<AdditiveShare<F>>` of length degree.
    /// Uses `io_ctx` for resharing intermediate products in the 16-fold multiplication tree.
    pub fn compute_prover_message_share<N: Rep3NetworkWorker>(
        &mut self,
        _round: usize,
        previous_claim: AdditiveShare<F>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<Vec<AdditiveShare<F>>> {
        let degree = D + 1; // 17
        let ra_i_polys = &self.ra_i_polys;
        let r_cycle = &self.r_cycle;
        let r_sumcheck = &self.r_sumcheck;

        // Split-Eq optimization (all public).
        // See https://eprint.iacr.org/2025/1117.pdf section 5.2.
        let w = &r_cycle[r_sumcheck.len() + 1..];
        let (wr, wl) = w.split_at(w.len() / 2);
        let eq_constant_factor = EqPolynomial::mle(r_sumcheck, &r_cycle[..r_sumcheck.len()]);
        let eq_wl_evals = EqPolynomial::evals_parallel(wl, Some(eq_constant_factor));
        let eq_wr_evals = EqPolynomial::evals_parallel(wr, None);

        let round = r_sumcheck.len();
        let half = 1usize << w.len();

        // Compute the 16-fold product tree via 4 levels with 3 reshares.
        let n_wl = eq_wl_evals.len();
        let n_wr = eq_wr_evals.len();
        let level1_len = n_wr * n_wl * (8 * 3);
        let level2_len = n_wr * n_wl * (4 * 5);
        let level3_len = n_wr * n_wl * (2 * 9);
        let _span = trace_span!(
            "compute_mles_product_16_rep3",
            round,
            w_len = w.len(),
            n_wl,
            n_wr,
            half,
            level1_len,
            level2_len,
            level3_len
        )
        .entered();
        let sum_evals = compute_mles_product_16_rep3(
            &eq_wl_evals,
            &eq_wr_evals,
            ra_i_polys,
            half,
            wl.len(),
            io_ctx,
        )?;
        drop(_span);

        // sum_evals[0..D] are evaluations at {1, 2, ..., 15, ∞} as AdditiveShare<F>.
        // This is the product polynomial WITHOUT the eq(X, r[round]) factor.

        // Recover eval at 0 from the claim.
        let eq_eval_at_0: F = EqPolynomial::mle(&[F::zero()], &[r_cycle[round]]);
        let eq_eval_at_1: F = EqPolynomial::mle(&[F::one()], &[r_cycle[round]]);

        let eval_at_1 = sum_evals[0]; // point 1
        let eval_at_0 =
            (previous_claim - eval_at_1 * eq_eval_at_1) * eq_eval_at_0.inverse().unwrap();

        // Toom-Cook interpolation: from evals at {0, 1, 2, ..., D-1, ∞} to coefficients.
        // The intermediate polynomial has degree D-1 = 15, evaluated at D+1 = 17 points.
        let toom_evals: Vec<F> = {
            let mut v = Vec::with_capacity(D + 1);
            v.push(eval_at_0.into_fe());
            for i in 0..D {
                v.push(sum_evals[i].into_fe());
            }
            v
        };
        let tmp_coeffs = UniPoly::from_evals_toom(&toom_evals).coeffs;

        // Multiply by eq(X, r[round]) = (1 - r[round]) + (2*r[round] - 1)*X
        let r_round: F = r_cycle[round].into();
        let constant_coeff = F::one() - r_round;
        let x_coeff = r_round + r_round - F::one();

        let mut coeffs_fe = vec![F::zero(); tmp_coeffs.len() + 1];
        for (i, coeff) in tmp_coeffs.into_iter().enumerate() {
            coeffs_fe[i] += coeff * constant_coeff;
            coeffs_fe[i + 1] += coeff * x_coeff;
        }

        // Evaluate the final polynomial at {0, 2, 3, ..., degree}.
        let final_poly = UniPoly::from_coeff(coeffs_fe);
        let mut result = Vec::with_capacity(degree);
        result.push(AdditiveShare::from_fe(final_poly.evaluate::<F>(&F::zero())));
        for x in 2..=degree {
            result.push(AdditiveShare::from_fe(
                final_poly.evaluate::<F>(&F::from(x as u64)),
            ));
        }

        Ok(result)
    }
}

impl<F: JoltField> Rep3InstructionRaSumcheckWorker<F> {
    pub fn degree_inner(&self) -> usize {
        D + 1
    }

    pub fn num_rounds_inner(&self) -> usize {
        self.r_cycle.len()
    }

    pub fn input_claim_public(&self) -> F {
        self.input_claim
    }

    pub fn bind_inner(&mut self, r_j: F::Challenge) {
        self.ra_i_polys
            .par_iter_mut()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::HighToLow));
        self.r_sumcheck.push(r_j);
    }

    pub fn normalize_opening_point_inner(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    pub fn cache_openings_worker_inner(
        &self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let ra_claims: Vec<Rep3PrimeFieldShare<F>> = self
            .ra_i_polys
            .iter()
            .map(|ra| ra.final_sumcheck_claim())
            .collect();

        for (i, r_address_chunk) in self.r_address_chunks.iter().enumerate() {
            accumulator.append_sparse(
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                r_address_chunk,
                &opening_point.r,
                vec![ra_claims[i]],
            );
        }

        ra_claims
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> crate::zkvm::dag::stage::Rep3SumcheckInstanceWorker<F, N>
    for Rep3InstructionRaSumcheckWorker<F>
{
    fn degree(&self) -> usize {
        self.degree_inner()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds_inner()
    }

    fn input_claim(&self) -> crate::utils::types::Rep3Value<F> {
        crate::utils::types::Rep3Value::Public(self.input_claim_public())
    }

    fn compute_prover_message_share(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
        io_ctx: &mut IoContextPool<N>,
    ) -> Vec<AdditiveShare<F>> {
        let own_degree = self.degree_inner();
        let mut evals = self
            .compute_prover_message_share(round, previous_claim, io_ctx)
            .expect("RA sumcheck round computation failed");

        // Pad with zeros if batched with higher-degree instances.
        // evals has length own_degree; trait expects length max_degree.
        if max_degree > own_degree {
            evals.resize(max_degree, AdditiveShare::zero());
        }
        evals
    }

    fn bind(
        &mut self,
        r_j: F::Challenge,
        _round: usize,
        _io_ctx: &mut IoContextPool<N>,
        _preproc: &mut PreprocessingPool<F>,
    ) {
        self.bind_inner(r_j);
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        self.normalize_opening_point_inner(opening_point)
    }

    fn cache_openings_worker(
        &mut self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        self.cache_openings_worker_inner(accumulator, opening_point)
    }
}

// ---------------------------------------------------------------------------
// Dedicated stage 4 proving loops
// ---------------------------------------------------------------------------

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
        claim = crate::subprotocols::sumcheck::evaluate_univariate_at_share::<F>(
            degree, claim, &msg, r_j,
        )?;
    }

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

// ---------------------------------------------------------------------------
