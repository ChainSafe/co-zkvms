use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction_lookups::{D, LOG_K_CHUNK};
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use rayon::prelude::*;
use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::utils::types::Rep3Value;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};

const DEGREE: usize = 1;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct Rep3HammingWeightSumcheckWorker<F: JoltField> {
    gamma: [F; D],
    ra: [Rep3DensePolynomial<F>; D],
}

impl<F: JoltField> Rep3HammingWeightSumcheckWorker<F> {
    pub fn new(G: [Arc<Vec<Rep3PrimeFieldShare<F>>>; D], gamma: [F; D]) -> Self {
        Self {
            gamma,
            ra: G.map(Rep3DensePolynomial::from_coeffs_arc),
        }
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N>
    for Rep3HammingWeightSumcheckWorker<F>
{
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK
    }

    fn input_claim(&self) -> Rep3Value<F> {
        Rep3Value::Public(self.gamma.iter().copied().sum())
    }

    fn compute_prover_message_share(
        &mut self,
        _round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Vec<AdditiveShare<F>> {
        // Degree 1: g(x) is linear, but batching may require evaluations at x=2,3,...,max_degree.
        // We return evaluations at points {0,2,3,...,max_degree}.
        let eval_0: AdditiveShare<F> = self
            .ra
            .par_iter()
            .zip(self.gamma.par_iter())
            .map(|(ra, gamma)| {
                let ra_sum: Rep3PrimeFieldShare<F> = (0..ra.len() / 2)
                    .into_par_iter()
                    .map(|j| ra.get_bound_coeff(2 * j))
                    .reduce(Rep3PrimeFieldShare::zero_share, |a, b| a + b);
                ra_sum.into_additive() * *gamma
            })
            .reduce(AdditiveShare::zero, |a, b| a + b);

        let eval_1 = previous_claim - eval_0;
        let slope = eval_1 - eval_0;

        let mut evals = vec![AdditiveShare::zero(); max_degree];
        evals[0] = eval_0;
        for x in 2..=max_degree {
            evals[x - 1] = eval_0 + slope * F::from(x as u64);
        }
        evals
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize, _io_ctx: &mut IoContextPool<N>) {
        let r: F = r_j.into();
        self.ra
            .par_iter_mut()
            .for_each(|ra| ra.bind(r, BindingOrder::LowToHigh));
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_worker(
        &mut self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let ra_claims: Vec<Rep3PrimeFieldShare<F>> =
            self.ra.iter().map(|ra| ra.final_sumcheck_claim()).collect();

        // Get r_cycle from the accumulator (stored during Spartan outer sumcheck).
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;

        accumulator.append_sparse(
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionHammingWeight,
            &opening_point.r,
            &r_cycle,
            ra_claims.clone(),
        );

        ra_claims
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3HammingWeightSumcheck<F: JoltField> {
    gamma: [F; D],
}

impl<F: JoltField> Rep3HammingWeightSumcheck<F> {
    pub fn new<T: Transcript>(transcript: &mut T) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        Self {
            gamma: gamma_powers,
        }
    }

    /// Return gamma powers so the worker can use them.
    pub fn gamma(&self) -> [F; D] {
        self.gamma
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK
    }

    fn input_claim_public(&self) -> F {
        self.gamma.iter().copied().sum()
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        _r: &[F::Challenge],
    ) -> F {
        let ra_claims = (0..D).map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionHammingWeight,
                )
                .1
        });

        self.gamma
            .iter()
            .zip(ra_claims)
            .map(|(gamma, ra)| ra * gamma)
            .sum()
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        // Get r_cycle from the accumulator (stored during Spartan outer sumcheck).
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();

        accumulator.append_sparse(
            transcript,
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionHammingWeight,
            &opening_point.r,
            &r_cycle,
            claims,
        );
    }
}
