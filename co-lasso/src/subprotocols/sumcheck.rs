#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::os::unix::net;

use co_spartan::mpc::additive;
use co_spartan::mpc::rep3::Rep3PrimeFieldShare;
use itertools::multizip;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::field::JoltField;
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::subprotocols::sumcheck::CubicSumcheckType;
use jolt_core::utils::thread::drop_in_background_thread;
use mpc_core::protocols::rep3::{self, PartyID};
use mpc_net::mpc_star::MpcStarNetWorker;
use rayon::prelude::*;
use tracing::trace_span;

use crate::poly::Rep3DensePolynomial;

#[derive(Debug, Clone)]
pub struct Rep3CubicSumcheckParams<F: JoltField> {
    poly_As: Vec<Rep3DensePolynomial<F>>,
    poly_Bs: Vec<Rep3DensePolynomial<F>>,
    poly_flags: Vec<DensePolynomial<F>>,

    poly_eq: DensePolynomial<F>,

    pub num_rounds: usize,

    pub sumcheck_type: CubicSumcheckType,
}

impl<F: JoltField> Rep3CubicSumcheckParams<F> {
    pub fn new_prod(
        poly_lefts: Vec<Rep3DensePolynomial<F>>,
        poly_rights: Vec<Rep3DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_lefts.len(), poly_rights.len());
        debug_assert_eq!(poly_lefts[0].len(), poly_rights[0].len());
        debug_assert_eq!(poly_lefts[0].len(), poly_eq.len());

        Rep3CubicSumcheckParams {
            poly_As: poly_lefts,
            poly_Bs: poly_rights,
            poly_flags: vec![],
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::Prod,
        }
    }

    pub fn new_prod_ones(
        poly_lefts: Vec<Rep3DensePolynomial<F>>,
        poly_rights: Vec<Rep3DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_lefts.len(), poly_rights.len());
        debug_assert_eq!(poly_lefts[0].len(), poly_rights[0].len());
        debug_assert_eq!(poly_lefts[0].len(), poly_eq.len());

        Rep3CubicSumcheckParams {
            poly_As: poly_lefts,
            poly_Bs: poly_rights,
            poly_flags: vec![],
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::ProdOnes,
        }
    }

    pub fn new_flags(
        poly_leaves: Vec<Rep3DensePolynomial<F>>,
        poly_flags: Vec<DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_leaves[0].len(), poly_flags[0].len());
        debug_assert_eq!(poly_leaves[0].len(), poly_eq.len());

        Self {
            poly_As: poly_leaves,
            poly_Bs: vec![],
            poly_flags,
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

    pub fn get_final_evals(&self) -> (Vec<Rep3PrimeFieldShare<F>>, Vec<Rep3PrimeFieldShare<F>>, F) {
        debug_assert_eq!(self.poly_As[0].len(), 1);
        debug_assert_eq!(self.poly_Bs[0].len(), 1);
        debug_assert_eq!(self.poly_eq.len(), 1);

        let poly_A_final = (0..self.poly_As.len())
            .map(|i| self.poly_As[i][0])
            .collect();

        let poly_B_final = (0..self.poly_Bs.len())
            .map(|i| self.poly_Bs[i][0])
            .collect();

        let poly_eq_final = self.poly_eq[0];

        (poly_A_final, poly_B_final, poly_eq_final)
    }
}

pub fn rep3_prove_cubic_batched<F: JoltField, N: MpcStarNetWorker>(
    claim: &F,
    params: Rep3CubicSumcheckParams<F>,
    coeffs: &[F],
    network: &mut N,
) -> eyre::Result<(
    Vec<F>,
    (Vec<Rep3PrimeFieldShare<F>>, Vec<Rep3PrimeFieldShare<F>>, F),
)> {
    match params.sumcheck_type {
        CubicSumcheckType::Prod => prove_cubic_batched_prod(claim, params, coeffs, network),
        CubicSumcheckType::ProdOnes => {
            prove_cubic_batched_prod_ones(claim, params, coeffs, network)
        }
        CubicSumcheckType::Flags => prove_cubic_batched_flags(claim, params, coeffs, network),
    }
}

#[tracing::instrument(skip_all, name = "Sumcheck.prove_cubic_batched_prod", level = "trace")]
pub fn prove_cubic_batched_prod<F: JoltField, N: MpcStarNetWorker>(
    claim: &F,
    params: Rep3CubicSumcheckParams<F>,
    coeffs: &[F],
    network: &mut N,
) -> eyre::Result<(
    Vec<F>,
    (Vec<Rep3PrimeFieldShare<F>>, Vec<Rep3PrimeFieldShare<F>>, F),
)> {
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

                for (coeff, poly_A, poly_B) in multizip((coeffs, &params.poly_As, &params.poly_Bs))
                {
                    // We want to compute:
                    //     evals.0 += coeff * poly_A[low_index] * poly_B[low_index]
                    //     evals.1 += coeff * (2 * poly_A[high_index] - poly_A[low_index]) * (2 * poly_B[high_index] - poly_B[low_index])
                    //     evals.0 += coeff * (3 * poly_A[high_index] - 2 * poly_A[low_index]) * (3 * poly_B[high_index] - 2 * poly_B[low_index])
                    // which naively requires 3 multiplications by `coeff`.
                    // By computing these values `A_low` and `A_high`, we only use 2 multiplications by `coeff`.
                    let A_low = poly_A[low_index] * *coeff;
                    let A_high = poly_A[high_index] * *coeff;

                    let m_a = A_high - A_low;
                    let m_b = poly_B[high_index] - poly_B[low_index];

                    let point_2_A = A_high + m_a;
                    let point_3_A = point_2_A + m_a;

                    let point_2_B = poly_B[high_index] + m_b;
                    let point_3_B = point_2_B + m_b;

                    // evals.0 += A_low * poly_B[low_index];
                    // evals.1 += point_2_A * point_2_B;
                    // evals.2 += point_3_A * point_3_B;

                    // we multiply each term of sum by public eq_evals.* to avoid resharing
                    // TODO: check if overhead of doing len(coeffs) * 3 more multiplications is worth it
                    evals.0 += A_low * rep3::arithmetic::mul_public(poly_B[low_index], eq_evals.0);
                    evals.1 += point_2_A * rep3::arithmetic::mul_public(point_2_B, eq_evals.1);
                    evals.2 += point_3_A * rep3::arithmetic::mul_public(point_3_B, eq_evals.2);
                }

                // since evals here are additive shares, we can't multiply by public eq_evals.*
                // evals.0 *= eq_evals.0;
                // evals.1 *= eq_evals.1;
                // evals.2 *= eq_evals.2;
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
        network.send_response(poly.as_vec())?;

        //derive the verifier's challenge for the next round
        let r_j = network.receive_request()?;
        r.push(r_j);

        // bound all tables to the verifier's challenege
        let _span = trace_span!("binding");
        let _enter = _span.enter();

        let poly_iter = params
            .poly_As
            .par_iter_mut()
            .chain(params.poly_Bs.par_iter_mut());

        rayon::join(
            || poly_iter.for_each(|poly| poly.fix_var_top(&r_j)),
            || params.poly_eq.bound_poly_var_top(&r_j),
        );
        
        drop(_enter);
        drop(_span);

        // poly coeffs are additive shares but evaluation requires multiplication
        // e = poly.evaluate(&r_j);
        // since we sent coeffs shares earlier, we can just receive the evaluation from coordinator
        e = additive::promote_to_trivial_share(network.receive_request()?, network.party_id());
        cubic_polys.push(poly.compress());
    }

    // network.send_response(e)?;

    let claims_prod = params.get_final_evals();

    drop_in_background_thread(params);

    Ok((r, claims_prod))
}

#[tracing::instrument(
    skip_all,
    name = "Sumcheck.prove_cubic_batched_prod_ones",
    level = "trace"
)]
pub fn prove_cubic_batched_prod_ones<F: JoltField, N: MpcStarNetWorker>(
    claim: &F,
    params: Rep3CubicSumcheckParams<F>,
    coeffs: &[F],
    network: &mut N,
) -> eyre::Result<(
    Vec<F>,
    (Vec<Rep3PrimeFieldShare<F>>, Vec<Rep3PrimeFieldShare<F>>, F),
)> {
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

                        // order of multiplications here is important,
                        // since poly_A[*] and poly_B[*] are Rep3 shares whose multiplication is additive share,
                        // which cannot be multiplied on public value eq_evals[*] without reshare.
                        // To avoid resharing, we first multiply poly_B[low] * eq_evals[low].0, as such commutativity holds.
                        let eval_point_0: F = poly_A[low]
                            * rep3::arithmetic::mul_public(
                                poly_B[low],
                                eq_evals[low].0 * coeffs[batch_index],
                            );

                        let m_a = poly_A[high] - poly_A[low];
                        let m_b = poly_B[high] - poly_B[low];

                        let point_2_A = poly_A[high] + m_a;
                        let point_3_A = point_2_A + m_a;

                        let point_2_B = poly_B[high] + m_b;
                        let point_3_B = point_2_B + m_b;

                        let eval_point_2 = point_2_A
                            * rep3::arithmetic::mul_public(
                                point_2_B,
                                eq_evals[low].1 * coeffs[batch_index],
                            );
                        let eval_point_3 = point_3_A
                            * rep3::arithmetic::mul_public(
                                point_3_B,
                                eq_evals[low].2 * coeffs[batch_index],
                            );

                        (eval_point_0, eval_point_2, eval_point_3)
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


        // network.send_response(evals.to_vec())?;
        let poly = UniPoly::from_evals(&evals); // TODO: may not work on additive shares

        network.send_response(poly.as_vec())?;

        //derive the verifier's challenge for the next round
        let r_j = network.receive_request()?;
        r.push(r_j);

        // bound all tables to the verifier's challenege
        let _span = trace_span!("binding (ones)");
        let _enter = _span.enter();

        let poly_iter = params
            .poly_As
            .par_iter_mut()
            .chain(params.poly_Bs.par_iter_mut());


        rayon::join(
            || poly_iter.for_each(|poly| poly.fix_var_top_many_ones(&r_j)),
            || params.poly_eq.bound_poly_var_top(&r_j),
        );

        drop(_enter);
        drop(_span);

        // poly coeffs are additive shares but evaluation requires multiplication
        // e = poly.evaluate(&r_j);
        // since we sent coeffs shares earlier, we can just receive the evaluation from coordinator
        e = additive::promote_to_trivial_share(network.receive_request()?, network.party_id());
        cubic_polys.push(poly.compress());
    }

    let claims_prod = params.get_final_evals();

    drop_in_background_thread(params);

    Ok((r, claims_prod))
}

#[tracing::instrument(
    skip_all,
    name = "Sumcheck.prove_batched_special_fork_flags",
    level = "trace"
)]
pub fn prove_cubic_batched_flags<F: JoltField, N: MpcStarNetWorker>(
    claim: &F,
    params: Rep3CubicSumcheckParams<F>,
    coeffs: &[F],
    network: &mut N,
) -> eyre::Result<(
    Vec<F>,
    (Vec<Rep3PrimeFieldShare<F>>, Vec<Rep3PrimeFieldShare<F>>, F),
)> {
    let mut params = params;

    let mut e = *claim;
    let mut r: Vec<F> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

    let mut eq_evals: Vec<[F; 3]> = Vec::with_capacity(params.poly_As[0].len() / 2);

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
                [eval_point_0, eval_point_2, eval_point_3]
            })
            .collect_into_vec(&mut eq_evals);
        drop(_eq_enter);
        drop(eq_span);

        let party_id = network.party_id();
        let _span = trace_span!("eval_loop");
        let _enter = _span.enter();
        let evals: Vec<[_; 3]> = params
            .poly_flags
            .par_iter()
            .enumerate()
            .flat_map(|(memory_index, memory_flag_poly)| {
                let read_leaves = &params.poly_As[2 * memory_index];
                let write_leaves = &params.poly_As[2 * memory_index + 1];

                let (read_evals, write_evals) = rayon::join(
                    || {
                        compute_cubic_evals_flags(
                            memory_flag_poly,
                            read_leaves,
                            &eq_evals,
                            len,
                            party_id,
                        )
                    },
                    || {
                        compute_cubic_evals_flags(
                            memory_flag_poly,
                            write_leaves,
                            &eq_evals,
                            len,
                            party_id,
                        )
                    },
                );

                [read_evals, write_evals]
            })
            .collect();
        drop(_enter);
        drop(_span);

        let evals_combined_0 = (0..evals.len())
            .map(|i| (evals[i][0] * coeffs[i]).into_additive())
            .sum();
        let evals_combined_2 = (0..evals.len())
            .map(|i| (evals[i][1] * coeffs[i]).into_additive())
            .sum();
        let evals_combined_3 = (0..evals.len())
            .map(|i| (evals[i][2] * coeffs[i]).into_additive())
            .sum();

        let cubic_evals = [
            evals_combined_0,
            e - evals_combined_0,
            evals_combined_2,
            evals_combined_3,
        ];
        let poly = UniPoly::from_evals(&cubic_evals);

        // append the prover's message to the transcript
        network.send_response(poly.as_vec())?;

        //derive the verifier's challenge for the next round
        let r_j = network.receive_request()?;
        r.push(r_j);

        let poly_As_span = trace_span!("Bind leaves");
        let _poly_As_enter = poly_As_span.enter();
        params
            .poly_As
            .par_iter_mut()
            .for_each(|poly| poly.fix_var_top(&r_j));
        drop(_poly_As_enter);
        drop(poly_As_span);

        let poly_other_span = trace_span!("Bind EQ and flags");
        let _poly_other_enter = poly_other_span.enter();
        rayon::join(
            || params.poly_eq.bound_poly_var_top(&r_j),
            || {
                params
                    .poly_flags
                    .par_iter_mut()
                    .for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j))
            },
        );
        drop(_poly_other_enter);
        drop(poly_other_span);

        e = additive::promote_to_trivial_share(network.receive_request()?, network.party_id());
        cubic_polys.push(poly.compress());
    }

    let leaves_claims: Vec<Rep3PrimeFieldShare<F>> = (0..params.poly_As.len())
        .map(|i| params.poly_As[i][0])
        .collect();

    let flags_claims: Vec<F> = (0..params.poly_As.len())
        .map(|i| params.poly_flags[i / 2][0])
        .collect();

    let poly_eq_final = params.poly_eq[0];

    let claims_prod = (
        leaves_claims,
        rep3::arithmetic::promote_to_trivial_shares(flags_claims, network.party_id()),
        poly_eq_final,
    );

    drop_in_background_thread(params);

    Ok((r, claims_prod))
}

#[tracing::instrument(
    skip_all,
    name = "SumcheckInstanceProof::compute_cubic_evals_flags",
    level = "trace"
)]
fn compute_cubic_evals_flags<F: JoltField>(
    flags: &DensePolynomial<F>,
    leaves: &Rep3DensePolynomial<F>,
    eq_evals: &Vec<[F; 3]>,
    len: usize,
    id: PartyID,
) -> [Rep3PrimeFieldShare<F>; 3] {
    let (flags_low, flags_high) = flags.split_evals(len);
    let (leaves_low, leaves_high) = leaves.split_evals(len);

    let mut evals = [Rep3PrimeFieldShare::zero_share(); 3];
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

        let flag_eval = [flag_low, flag_eval_point_2, flag_eval_point_3];

        if flag_eval[0].is_zero() {
            rep3::arithmetic::add_assign_public(&mut evals[0], eq_eval[0], id)
        } else if flag_eval[0].is_one() {
            evals[0] += rep3::arithmetic::mul_public(leaf_low, eq_eval[0])
        } else {
            evals[0] += rep3::arithmetic::mul_public(
                rep3::arithmetic::add_public(leaf_low * flag_eval[0], F::one() - flag_eval[0], id),
                eq_eval[0],
            )
        };

        let opt_poly_2_res: Option<(_, _)> = if flag_eval[1].is_zero() {
            rep3::arithmetic::add_assign_public(&mut evals[1], eq_eval[1], id);
            None
        } else if flag_eval[1].is_one() {
            let poly_m = leaf_high - leaf_low;
            let poly_2 = leaf_high + poly_m;
            evals[1] += rep3::arithmetic::mul_public(poly_2, eq_eval[1]);
            Some((poly_2, poly_m))
        } else {
            let poly_m = leaf_high - leaf_low;
            let poly_2 = leaf_high + poly_m;
            evals[1] += rep3::arithmetic::mul_public(
                rep3::arithmetic::add_public(poly_2 * flag_eval[1], F::one() - flag_eval[1], id),
                eq_eval[1],
            );
            Some((poly_2, poly_m))
        };

        if let Some((poly_2, poly_m)) = opt_poly_2_res {
            if flag_eval[2].is_zero() {
                rep3::arithmetic::add_assign_public(&mut evals[2], eq_eval[2], id);
            } else if flag_eval[2].is_one() {
                let poly_3 = poly_2 + poly_m;
                evals[2] += rep3::arithmetic::mul_public(poly_3, eq_eval[2]);
            } else {
                let poly_3 = poly_2 + poly_m;
                evals[2] += rep3::arithmetic::mul_public(
                    rep3::arithmetic::add_public(
                        poly_3 * flag_eval[2],
                        F::one() - flag_eval[2],
                        id,
                    ),
                    eq_eval[2],
                );
            }
        } else {
            rep3::arithmetic::add_assign_public(&mut evals[2], eq_eval[2], id);
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
