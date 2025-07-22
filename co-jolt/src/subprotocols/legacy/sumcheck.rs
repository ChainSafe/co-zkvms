#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::marker::PhantomData;

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::utils::errors::ProofVerifyError;
use crate::utils::thread::drop_in_background_thread;
use jolt_core::utils::transcript::{AppendToTranscript, Transcript};
use ark_serialize::*;
use itertools::multizip;
use rayon::prelude::*;
use tracing::trace_span;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CubicSumcheckType {
    // eq * A * B
    Prod,

    // eq * A * B, optimized for high probability (A, B) = 1
    ProdOnes,

    // eq *(A * flags + (1 - flags))
    Flags,
}

impl Into<u8> for CubicSumcheckType {
    fn into(self) -> u8 {
        match self {
            CubicSumcheckType::Prod => 0,
            CubicSumcheckType::ProdOnes => 1,
            CubicSumcheckType::Flags => 2,
        }
    }
}

impl From<u8> for CubicSumcheckType {
    fn from(value: u8) -> Self {
        match value {
            0 => CubicSumcheckType::Prod,
            1 => CubicSumcheckType::ProdOnes,
            2 => CubicSumcheckType::Flags,
            _ => panic!("Invalid sumcheck type"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CubicSumcheckParams<F: JoltField> {
    poly_As: Vec<DensePolynomial<F>>,
    poly_Bs: Vec<DensePolynomial<F>>,

    poly_eq: DensePolynomial<F>,

    pub num_rounds: usize,

    pub sumcheck_type: CubicSumcheckType,
}

impl<F: JoltField> CubicSumcheckParams<F> {
    pub fn new_prod(
        poly_lefts: Vec<DensePolynomial<F>>,
        poly_rights: Vec<DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_lefts.len(), poly_rights.len());
        debug_assert_eq!(poly_lefts[0].len(), poly_rights[0].len());
        debug_assert_eq!(poly_lefts[0].len(), poly_eq.len());

        CubicSumcheckParams {
            poly_As: poly_lefts,
            poly_Bs: poly_rights,
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::Prod,
        }
    }

    pub fn new_prod_ones(
        poly_lefts: Vec<DensePolynomial<F>>,
        poly_rights: Vec<DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_lefts.len(), poly_rights.len());
        debug_assert_eq!(poly_lefts[0].len(), poly_rights[0].len());
        debug_assert_eq!(poly_lefts[0].len(), poly_eq.len());

        CubicSumcheckParams {
            poly_As: poly_lefts,
            poly_Bs: poly_rights,
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::ProdOnes,
        }
    }

    pub fn new_flags(
        poly_leaves: Vec<DensePolynomial<F>>,
        poly_flags: Vec<DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_leaves[0].len(), poly_flags[0].len());
        debug_assert_eq!(poly_leaves[0].len(), poly_eq.len());

        CubicSumcheckParams {
            poly_As: poly_leaves,
            poly_Bs: poly_flags,
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::Flags,
        }
    }

    #[inline]
    pub fn combine_prod(l: &F, r: &F, eq: &F) -> F {
        if *l == F::one() && *r == F::one() {
            *eq
        } else if *l == F::one() {
            *r * eq
        } else if *r == F::one() {
            *l * eq
        } else {
            *l * r * eq
        }
    }

    #[inline]
    pub fn combine_flags(h: &F, flag: &F, eq: &F) -> F {
        if *flag == F::zero() {
            *eq
        } else if *flag == F::one() {
            *eq * *h
        } else {
            *eq * (*flag * h + (F::one() + flag.neg()))
        }
    }

    pub fn get_final_evals(&self) -> (Vec<F>, Vec<F>, F) {
        debug_assert_eq!(self.poly_As[0].len(), 1);
        debug_assert_eq!(self.poly_Bs[0].len(), 1);
        debug_assert_eq!(self.poly_eq.len(), 1);

        let poly_A_final: Vec<F> = (0..self.poly_As.len())
            .map(|i| self.poly_As[i][0])
            .collect();

        let poly_B_final: Vec<F> = (0..self.poly_Bs.len())
            .map(|i| self.poly_Bs[i][0])
            .collect();

        let poly_eq_final = self.poly_eq[0];

        (poly_A_final, poly_B_final, poly_eq_final)
    }
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    /// Create a sumcheck proof for polynomial(s) of arbitrary degree.
    ///
    /// Params
    /// - `claim`: Claimed sumcheck evaluation (note: currently unused)
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `polys`: Dense polynomials to combine and sumcheck
    /// - `comb_func`: Function used to combine each polynomial evaluation
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (SumcheckInstanceProof, r_eval_point, final_evals)
    /// - `r_eval_point`: Final random point of evaluation
    /// - `final_evals`: Each of the polys evaluated at `r_eval_point`
    #[tracing::instrument(skip_all, name = "Sumcheck.prove", level = "trace")]
    pub fn prove_arbitrary<Func>(
        _claim: &F,
        num_rounds: usize,
        polys: &mut Vec<DensePolynomial<F>>,
        comb_func: Func,
        combined_degree: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func: Fn(&[F]) -> F + std::marker::Sync,
    {
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _round in 0..num_rounds {
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            let mut eval_points = vec![F::zero(); combined_degree + 1];

            let mle_half = polys[0].len() / 2;

            let accum: Vec<Vec<F>> = (0..mle_half)
                .into_par_iter()
                .map(|poly_term_i| {
                    let mut accum = vec![F::zero(); combined_degree + 1];
                    // Evaluate P({0, ..., |g(r)|})

                    // TODO(#28): Optimize
                    // Tricks can be used here for low order bits {0,1} but general premise is a running sum for each
                    // of the m terms in the Dense multilinear polynomials. Formula is:
                    // half = | D_{n-1} | / 2
                    // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] - D_{n-1}[index])

                    // eval 0: bound_func is A(low)
                    let params_zero: Vec<F> = polys.iter().map(|poly| poly[poly_term_i]).collect();
                    accum[0] += comb_func(&params_zero);

                    // TODO(#28): Can be computed from prev_round_claim - eval_point_0
                    let params_one: Vec<F> = polys
                        .iter()
                        .map(|poly| poly[mle_half + poly_term_i])
                        .collect();
                    accum[1] += comb_func(&params_one);

                    // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] - D_{n-1}[index])
                    // D_n(index, 0) = D_{n-1}[LOW]
                    // D_n(index, 1) = D_{n-1}[HIGH]
                    // D_n(index, 2) = D_{n-1}[HIGH] + (D_{n-1}[HIGH] - D_{n-1}[LOW])
                    // D_n(index, 3) = D_{n-1}[HIGH] + (D_{n-1}[HIGH] - D_{n-1}[LOW]) + (D_{n-1}[HIGH] - D_{n-1}[LOW])
                    // ...
                    let mut existing_term = params_one;
                    for eval_i in 2..(combined_degree + 1) {
                        let mut poly_evals = vec![F::zero(); polys.len()];
                        for poly_i in 0..polys.len() {
                            let poly = &polys[poly_i];
                            poly_evals[poly_i] = existing_term[poly_i]
                                + poly[mle_half + poly_term_i]
                                - poly[poly_term_i];
                        }

                        accum[eval_i] += comb_func(&poly_evals);
                        existing_term = poly_evals;
                    }
                    accum
                })
                .collect();

            eval_points
                .par_iter_mut()
                .enumerate()
                .for_each(|(poly_i, eval_point)| {
                    *eval_point = accum
                        .par_iter()
                        .take(mle_half)
                        .map(|mle| mle[poly_i])
                        .sum::<F>();
                });

            let round_uni_poly = UniPoly::from_evals(&eval_points);

            // append the prover's message to the transcript
            round_uni_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenege
            polys
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top(&r_j));
            compressed_polys.push(round_uni_poly.compress());
        }

        let final_evals = polys.iter().map(|poly| poly[0]).collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_batched", level = "trace")]
    pub fn prove_cubic_batched(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F)) {
        match params.sumcheck_type {
            CubicSumcheckType::Prod => {
                Self::prove_cubic_batched_prod(claim, params, coeffs, transcript)
            }
            CubicSumcheckType::ProdOnes => {
                Self::prove_cubic_batched_prod_ones(claim, params, coeffs, transcript)
            }
            CubicSumcheckType::Flags => {
                Self::prove_cubic_batched_flags(claim, params, coeffs, transcript)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_cubic_batched_prod", level = "trace")]
    pub fn prove_cubic_batched_prod(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F)) {
        assert_eq!(params.poly_As.len(), params.poly_Bs.len());
        assert_eq!(params.poly_As.len(), coeffs.len());

        let mut params = params;

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _j in 0..params.num_rounds {
            let len = params.poly_As[0].len() / 2;
            let eq = &params.poly_eq;

            let _span = trace_span!("eval_loop");
            let _enter = _span.enter();
            let evals = (0..len)
                .into_par_iter()
                .map(|low_index| {
                    let high_index = low_index + len;

                    let eq_evals = {
                        let eval_point_0 = eq[low_index];
                        let m_eq = eq[high_index] - eq[low_index];
                        let eval_point_2 = eq[high_index] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        (eval_point_0, eval_point_2, eval_point_3)
                    };

                    let mut evals = (F::zero(), F::zero(), F::zero());

                    for (coeff, poly_A, poly_B) in
                        multizip((coeffs, &params.poly_As, &params.poly_Bs))
                    {
                        // We want to compute:
                        //     evals.0 += coeff * poly_A[low_index] * poly_B[low_index]
                        //     evals.1 += coeff * (2 * poly_A[high_index] - poly_A[low_index]) * (2 * poly_B[high_index] - poly_B[low_index])
                        //     evals.0 += coeff * (3 * poly_A[high_index] - 2 * poly_A[low_index]) * (3 * poly_B[high_index] - 2 * poly_B[low_index])
                        // which naively requires 3 multiplications by `coeff`.
                        // By computing these values `A_low` and `A_high`, we only use 2 multiplications by `coeff`.
                        let A_low = *coeff * poly_A[low_index];
                        let A_high = *coeff * poly_A[high_index];

                        let m_a = A_high - A_low;
                        let m_b = poly_B[high_index] - poly_B[low_index];

                        let point_2_A = A_high + m_a;
                        let point_3_A = point_2_A + m_a;

                        let point_2_B = poly_B[high_index] + m_b;
                        let point_3_B = point_2_B + m_b;

                        evals.0 += A_low * poly_B[low_index];
                        evals.1 += point_2_A * point_2_B;
                        evals.2 += point_3_A * point_3_B;
                    }

                    evals.0 *= eq_evals.0;
                    evals.1 *= eq_evals.1;
                    evals.2 *= eq_evals.2;
                    evals
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );
            drop(_enter);
            drop(_span);

            let evals = [evals.0, e - evals.0, evals.1, evals.2];
            let poly = UniPoly::from_evals(&evals);

            // append the prover's message to the transcript
            poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenege
            let _span = trace_span!("binding");
            let _enter = _span.enter();

            let poly_iter = params
                .poly_As
                .par_iter_mut()
                .chain(params.poly_Bs.par_iter_mut());

            rayon::join(
                || poly_iter.for_each(|poly| poly.bound_poly_var_top(&r_j)),
                || params.poly_eq.bound_poly_var_top(&r_j),
            );

            drop(_enter);
            drop(_span);

            e = poly.evaluate(&r_j);
            cubic_polys.push(poly.compress());
        }

        let claims_prod = params.get_final_evals();

        drop_in_background_thread(params);

        (SumcheckInstanceProof::new(cubic_polys), r, claims_prod)
    }

    #[tracing::instrument(
        skip_all,
        name = "Sumcheck.prove_cubic_batched_prod_ones",
        level = "trace"
    )]
    pub fn prove_cubic_batched_prod_ones(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F)) {
        let mut params = params;

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _j in 0..params.num_rounds {
            let len = params.poly_As[0].len() / 2;
            let eq = &params.poly_eq;
            let eq_evals: Vec<(F, F, F)> = (0..len)
                .into_par_iter()
                .map(|i| {
                    let low = i;
                    let high = len + i;

                    let eval_point_0 = eq[low];
                    let m_eq = eq[high] - eq[low];
                    let eval_point_2 = eq[high] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            let _span = trace_span!("eval_loop");
            let _enter = _span.enter();
            let evals: Vec<(F, F, F)> = (0..params.poly_As.len())
                .into_par_iter()
                .with_max_len(4)
                .map(|batch_index| {
                    let poly_A = &params.poly_As[batch_index];
                    let poly_B = &params.poly_Bs[batch_index];
                    let len = poly_A.len() / 2;

                    // In the case of a flagged tree, the majority of the leaves will be 1s, optimize for this case.
                    let (eval_point_0, eval_point_2, eval_point_3) = (0..len)
                        .map(|mle_index| {
                            let low = mle_index;
                            let high = len + mle_index;

                            // Optimized version of the product for the high probability that A[low], A[high], B[low], B[high] == 1

                            let a_low_one = poly_A[low].is_one();
                            let a_high_one = poly_A[high].is_one();
                            let b_low_one = poly_B[low].is_one();
                            let b_high_one = poly_B[high].is_one();

                            let eval_point_0: F = if a_low_one && b_low_one {
                                eq_evals[low].0
                            } else if a_low_one {
                                poly_B[low] * eq_evals[low].0
                            } else if b_low_one {
                                poly_A[low] * eq_evals[low].0
                            } else {
                                poly_A[low] * poly_B[low] * eq_evals[low].0
                            };

                            let m_a_zero = a_low_one && a_high_one;
                            let m_b_zero = b_low_one && b_high_one;

                            let (eval_point_2, eval_point_3) = if m_a_zero && m_b_zero {
                                (eq_evals[low].1, eq_evals[low].2)
                            } else if m_a_zero {
                                let m_b = poly_B[high] - poly_B[low];
                                let point_2_B = poly_B[high] + m_b;
                                let point_3_B = point_2_B + m_b;

                                let eval_point_2 = eq_evals[low].1 * point_2_B;
                                let eval_point_3 = eq_evals[low].2 * point_3_B;
                                (eval_point_2, eval_point_3)
                            } else if m_b_zero {
                                let m_a = poly_A[high] - poly_A[low];
                                let point_2_A = poly_A[high] + m_a;
                                let point_3_A = point_2_A + m_a;

                                let eval_point_2 = eq_evals[low].1 * point_2_A;
                                let eval_point_3 = eq_evals[low].2 * point_3_A;
                                (eval_point_2, eval_point_3)
                            } else {
                                let m_a = poly_A[high] - poly_A[low];
                                let m_b = poly_B[high] - poly_B[low];

                                let point_2_A = poly_A[high] + m_a;
                                let point_3_A = point_2_A + m_a;

                                let point_2_B = poly_B[high] + m_b;
                                let point_3_B = point_2_B + m_b;

                                let eval_point_2 = eq_evals[low].1 * point_2_A * point_2_B;
                                let eval_point_3 = eq_evals[low].2 * point_3_A * point_3_B;
                                (eval_point_2, eval_point_3)
                            };

                            (
                                eval_point_0 * coeffs[batch_index],
                                eval_point_2 * coeffs[batch_index],
                                eval_point_3 * coeffs[batch_index],
                            )
                        })
                        // For parallel
                        // .reduce(
                        //     || (F::zero(), F::zero(), F::zero()),
                        //     |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                        // );
                        // For normal
                        .fold(
                            (F::zero(), F::zero(), F::zero()),
                            |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                        );

                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
            drop(_enter);
            drop(_span);

            let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0).sum();
            let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1).sum();
            let evals_combined_3 = (0..evals.len()).map(|i| evals[i].2).sum();

            let evals = [
                evals_combined_0,
                e - evals_combined_0,
                evals_combined_2,
                evals_combined_3,
            ];
            let poly = UniPoly::from_evals(&evals);

            // append the prover's message to the transcript
            poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenege
            let _span = trace_span!("binding (ones)");
            let _enter = _span.enter();

            let poly_iter = params
                .poly_As
                .par_iter_mut()
                .chain(params.poly_Bs.par_iter_mut());

            rayon::join(
                || poly_iter.for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j)),
                || params.poly_eq.bound_poly_var_top(&r_j),
            );

            drop(_enter);
            drop(_span);

            e = poly.evaluate(&r_j);
            cubic_polys.push(poly.compress());
        }

        let claims_prod = params.get_final_evals();

        drop_in_background_thread(params);

        (SumcheckInstanceProof::new(cubic_polys), r, claims_prod)
    }

    #[tracing::instrument(
        skip_all,
        name = "SumcheckInstanceProof::compute_cubic_evals_flags",
        level = "trace"
    )]
    fn compute_cubic_evals_flags(
        flags: &DensePolynomial<F>,
        leaves: &DensePolynomial<F>,
        eq_evals: &Vec<(F, F, F)>,
        len: usize,
    ) -> (F, F, F) {
        let (flags_low, flags_high) = flags.split_evals(len);
        let (leaves_low, leaves_high) = leaves.split_evals(len);

        let mut evals = (F::zero(), F::zero(), F::zero());
        for (&flag_low, &flag_high, &leaf_low, &leaf_high, eq_eval) in
            multizip((flags_low, flags_high, leaves_low, leaves_high, eq_evals))
        {
            let m_eq: F = flag_high - flag_low;
            let (flag_eval_point_2, flag_eval_point_3) = if m_eq.is_zero() {
                (flag_high, flag_high)
            } else {
                let eval_point_2 = flag_high + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                (eval_point_2, eval_point_3)
            };

            let flag_eval = (flag_low, flag_eval_point_2, flag_eval_point_3);

            if flag_eval.0.is_zero() {
                evals.0 += eq_eval.0
            } else if flag_eval.0.is_one() {
                evals.0 += eq_eval.0 * leaf_low
            } else {
                evals.0 += eq_eval.0 * (flag_eval.0 * leaf_low + (F::one() - flag_eval.0))
            };

            let opt_poly_2_res: Option<(F, F)> = if flag_eval.1.is_zero() {
                evals.1 += eq_eval.1;
                None
            } else if flag_eval.1.is_one() {
                let poly_m = leaf_high - leaf_low;
                let poly_2 = leaf_high + poly_m;
                evals.1 += eq_eval.1 * poly_2;
                Some((poly_2, poly_m))
            } else {
                let poly_m = leaf_high - leaf_low;
                let poly_2 = leaf_high + poly_m;
                evals.1 += eq_eval.1 * (flag_eval.1 * poly_2 + (F::one() - flag_eval.1));
                Some((poly_2, poly_m))
            };

            if let Some((poly_2, poly_m)) = opt_poly_2_res {
                if flag_eval.2.is_zero() {
                    evals.2 += eq_eval.2; // TODO(sragss): Path may never happen
                } else if flag_eval.2.is_one() {
                    let poly_3 = poly_2 + poly_m;
                    evals.2 += eq_eval.2 * poly_3;
                } else {
                    let poly_3 = poly_2 + poly_m;
                    evals.2 += eq_eval.2 * (flag_eval.2 * poly_3 + (F::one() - flag_eval.2));
                }
            } else {
                evals.2 += eq_eval.2;
            };

            // Above is just a more complicated form of the following, optimizing for 0 / 1 flags.
            // let poly_m = poly_eval[high] - poly_eval[low];
            // let poly_2 = poly_eval[high] + poly_m;
            // let poly_3 = poly_2 + poly_m;

            // let eval_0 += params.combine(&poly_eval[low], &flag_eval.0, &eq_eval.0);
            // let eval_2 += params.combine(&poly_2, &flag_eval.1, &eq_eval.1);
            // let eval_3 += params.combine(&poly_3, &flag_eval.2, &eq_eval.2);
        }
        evals
    }

    #[tracing::instrument(
        skip_all,
        name = "Sumcheck.prove_batched_special_fork_flags",
        level = "trace"
    )]
    pub fn prove_cubic_batched_flags(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F)) {
        let mut params = params;

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        let mut eq_evals: Vec<(F, F, F)> = Vec::with_capacity(params.poly_As[0].len() / 2);

        for _j in 0..params.num_rounds {
            let len = params.poly_As[0].len() / 2;
            let eq_span = trace_span!("eq_evals");
            let _eq_enter = eq_span.enter();
            (0..len)
                .into_par_iter()
                .map(|i| {
                    let low = i;
                    let high = len + i;

                    let eq = &params.poly_eq;

                    let eval_point_0 = eq[low];
                    let m_eq = eq[high] - eq[low];
                    let eval_point_2 = eq[high] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect_into_vec(&mut eq_evals);
            drop(_eq_enter);
            drop(eq_span);

            let _span = trace_span!("eval_loop");
            let _enter = _span.enter();
            let evals: Vec<(F, F, F)> = params
                .poly_Bs
                .par_iter()
                .enumerate()
                .flat_map(|(memory_index, memory_flag_poly)| {
                    let read_leaves = &params.poly_As[2 * memory_index];
                    let write_leaves = &params.poly_As[2 * memory_index + 1];

                    let (read_evals, write_evals) = rayon::join(
                        || {
                            Self::compute_cubic_evals_flags(
                                memory_flag_poly,
                                read_leaves,
                                &eq_evals,
                                len,
                            )
                        },
                        || {
                            Self::compute_cubic_evals_flags(
                                memory_flag_poly,
                                write_leaves,
                                &eq_evals,
                                len,
                            )
                        },
                    );

                    [read_evals, write_evals]
                })
                .collect();
            drop(_enter);
            drop(_span);

            let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
            let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
            let evals_combined_3 = (0..evals.len()).map(|i| evals[i].2 * coeffs[i]).sum();

            let cubic_evals = [
                evals_combined_0,
                e - evals_combined_0,
                evals_combined_2,
                evals_combined_3,
            ];
            let poly = UniPoly::from_evals(&cubic_evals);

            // append the prover's message to the transcript
            poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            let poly_As_span = trace_span!("Bind leaves");
            let _poly_As_enter = poly_As_span.enter();
            params
                .poly_As
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top(&r_j));
            drop(_poly_As_enter);
            drop(poly_As_span);

            let poly_other_span = trace_span!("Bind EQ and flags");
            let _poly_other_enter = poly_other_span.enter();
            rayon::join(
                || params.poly_eq.bound_poly_var_top(&r_j),
                || {
                    params
                        .poly_Bs
                        .par_iter_mut()
                        .for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j))
                },
            );
            drop(_poly_other_enter);
            drop(poly_other_span);

            e = poly.evaluate(&r_j);
            cubic_polys.push(poly.compress());
        }

        let leaves_claims: Vec<F> = (0..params.poly_As.len())
            .map(|i| params.poly_As[i][0])
            .collect();

        let flags_claims: Vec<F> = (0..params.poly_As.len())
            .map(|i| params.poly_Bs[i / 2][0])
            .collect();

        let poly_eq_final = params.poly_eq[0];

        let claims_prod = (leaves_claims, flags_claims, poly_eq_final);

        drop_in_background_thread(params);

        (SumcheckInstanceProof::new(cubic_polys), r, claims_prod)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct SumcheckInstanceProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    pub fn new(
        compressed_polys: Vec<CompressedUniPoly<F>>,
    ) -> SumcheckInstanceProof<F, ProofTranscript> {
        SumcheckInstanceProof {
            compressed_polys,
            _marker: PhantomData,
        }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            let poly = self.compressed_polys[i].decompress(&e);

            // verify degree bound
            if poly.degree() != degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    poly.degree(),
                ));
            }

            // check if G_k(0) + G_k(1) = e
            assert_eq!(poly.eval_at_zero() + poly.eval_at_one(), e);

            // append the prover's message to the transcript
            poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i = transcript.challenge_scalar();

            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i
            e = poly.evaluate(&r_i);
        }

        Ok((e, r))
    }
}
