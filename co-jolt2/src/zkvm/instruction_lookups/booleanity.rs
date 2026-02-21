use std::sync::Arc;

use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction_lookups::{D, K_CHUNK, LOG_K_CHUNK};
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use rayon::prelude::*;
use snarks_core::math::Math;

use crate::field::JoltField;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::poly::ra_poly::{shifted_table_from_rand_ohv, Rep3RaPolynomial};
use crate::utils::types::Rep3Value;
use crate::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};

const DEGREE: usize = 3;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

struct BooleanityProverStateWorker<F: JoltField> {
    eq_r_address: GruenSplitEqPolynomial<F>,
    eq_r_cycle: GruenSplitEqPolynomial<F>,
    G: [Vec<Rep3PrimeFieldShare<F>>; D],
    /// Public masked indices from RandOHV (= masked_indices_c from witness gen).
    masked_H_indices: [Arc<Vec<Option<u8>>>; D],
    H: [Rep3RaPolynomial<u8, F>; D],
    F_table: Vec<F>,
    eq_r_r: F,
    /// Shared one-hot vectors e(r_i) for shifted table construction at phase transition.
    one_hot_e_fields: [Arc<Vec<Rep3PrimeFieldShare<F>>>; D],
}

pub struct Rep3BooleanitySumcheckWorker<F: JoltField> {
    party_id: PartyID,
    gamma: [F; D],
    r_address: Vec<F::Challenge>,
    log_T: usize,
    state: BooleanityProverStateWorker<F>,
}

impl<F: JoltField> Rep3BooleanitySumcheckWorker<F> {
    /// Construct the worker. `gamma` and `r_address` are public challenges
    /// received from the coordinator.
    pub fn new(
        gamma: [F; D],
        r_address: Vec<F::Challenge>,
        G: [Vec<Rep3PrimeFieldShare<F>>; D],
        one_hot_polys: &[Rep3OneHotPolynomial<F>; D],
        r_cycle: &[F::Challenge],
        trace_len: usize,
        party_id: PartyID,
    ) -> Self {
        Self {
            party_id,
            gamma,
            r_address: r_address.clone(),
            log_T: trace_len.log_2(),
            state: BooleanityProverStateWorker {
                eq_r_address: GruenSplitEqPolynomial::new(&r_address, BindingOrder::LowToHigh),
                eq_r_cycle: GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh),
                G,
                masked_H_indices: std::array::from_fn(|i| {
                    one_hot_polys[i].masked_indices_c.clone()
                }),
                H: std::array::from_fn(|_| Rep3RaPolynomial::None),
                F_table: {
                    let mut f = vec![F::zero(); K_CHUNK];
                    f[0] = F::one();
                    f
                },
                eq_r_r: F::zero(),
                one_hot_e_fields: std::array::from_fn(|i| {
                    one_hot_polys[i].rand_ohv_e_field.clone()
                }),
            },
        }
    }

    /// Phase 1: address-variable rounds (0..LOG_K_CHUNK).
    /// All ops are `shared * public` — no MPC communication needed.
    fn compute_phase1_message(
        &self,
        round: usize,
        previous_claim: AdditiveShare<F>,
    ) -> Vec<AdditiveShare<F>> {
        let p = &self.state;
        let m = round + 1;
        let B = &p.eq_r_address;

        // Compute quadratic coefficients for Gruen optimization.
        // Structure mirrors vanilla lines 310-458 but G values are Rep3PrimeFieldShare.
        let quadratic_coeffs: [Rep3PrimeFieldShare<F>; DEGREE - 1] = if B.E_in_current_len() == 1 {
            (0..B.len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let B_eval = B.E_out_current()[k_prime];

                    let inner_sum: [Rep3PrimeFieldShare<F>; DEGREE - 1] = (0..1 << m)
                        .into_par_iter()
                        .map(|k| {
                            let k_m = k >> (m - 1);
                            let F_k = p.F_table[k % (1 << (m - 1))];
                            let k_G = (k_prime << m) + k;

                            // Sum gamma-weighted G values (shared * public)
                            let G_times_F: Rep3PrimeFieldShare<F> =
                                p.G.iter()
                                    .zip(self.gamma.iter())
                                    .map(|(g, gamma)| g[k_G] * *gamma)
                                    .fold(Rep3PrimeFieldShare::zero_share(), |acc, x| acc + x)
                                    * F_k;

                            let eval_infty = G_times_F * F_k;
                            let eval_0 = if k_m == 0 {
                                eval_infty - G_times_F
                            } else {
                                Rep3PrimeFieldShare::zero_share()
                            };
                            [eval_0, eval_infty]
                        })
                        .reduce(
                            || [Rep3PrimeFieldShare::zero_share(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    // Multiply by public B_eval
                    [inner_sum[0] * B_eval, inner_sum[1] * B_eval]
                })
                .reduce(
                    || [Rep3PrimeFieldShare::zero_share(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            let num_x_in_bits = B.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;
            let chunk_size = 1 << num_x_in_bits;

            (0..B.len() / 2)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(x_out, chunk)| {
                    let B_E_out_eval = B.E_out_current()[x_out];

                    let chunk_evals: [Rep3PrimeFieldShare<F>; DEGREE - 1] = chunk
                        .par_iter()
                        .map(|k_prime| {
                            let x_in = k_prime & x_bitmask;
                            let B_E_in_eval = B.E_in_current()[x_in];

                            let inner_sum: [Rep3PrimeFieldShare<F>; DEGREE - 1] = (0..1 << m)
                                .into_par_iter()
                                .map(|k| {
                                    let k_m = k >> (m - 1);
                                    let F_k = p.F_table[k % (1 << (m - 1))];
                                    let k_G = (k_prime << m) + k;

                                    let G_times_F: Rep3PrimeFieldShare<F> = p
                                        .G
                                        .iter()
                                        .zip(self.gamma.iter())
                                        .map(|(g, gamma)| g[k_G] * *gamma)
                                        .fold(Rep3PrimeFieldShare::zero_share(), |acc, x| acc + x)
                                        * F_k;

                                    let eval_infty = G_times_F * F_k;
                                    let eval_0 = if k_m == 0 {
                                        eval_infty - G_times_F
                                    } else {
                                        Rep3PrimeFieldShare::zero_share()
                                    };
                                    [eval_0, eval_infty]
                                })
                                .reduce(
                                    || [Rep3PrimeFieldShare::zero_share(); DEGREE - 1],
                                    |running, new| [running[0] + new[0], running[1] + new[1]],
                                );

                            [inner_sum[0] * B_E_in_eval, inner_sum[1] * B_E_in_eval]
                        })
                        .reduce(
                            || [Rep3PrimeFieldShare::zero_share(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [chunk_evals[0] * B_E_out_eval, chunk_evals[1] * B_E_out_eval]
                })
                .reduce(
                    || [Rep3PrimeFieldShare::zero_share(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        // Apply Gruen expansion: public algebra on shared quadratic coefficients.
        let [q0, q_inf] = quadratic_coeffs;
        gruen_evals_deg_3(
            &self.state.eq_r_address,
            q0.into(),
            q_inf.into(),
            previous_claim,
            self.party_id,
        )
    }

    /// Phase 2: cycle-variable rounds (LOG_K_CHUNK..LOG_K_CHUNK+log_T).
    /// h_0^2 uses rep3 * rep3 → additive share (local, no network).
    fn compute_phase2_message(
        &self,
        _round: usize,
        previous_claim: AdditiveShare<F>,
    ) -> Vec<AdditiveShare<F>> {
        let p = &self.state;
        let D_poly = &p.eq_r_cycle;

        let quadratic_coeffs: [AdditiveShare<F>; DEGREE - 1] = if D_poly.E_in_current_len() == 1 {
            (0..D_poly.len() / 2)
                .into_par_iter()
                .map(|j_prime| {
                    let D_eval: F = D_poly.E_out_current()[j_prime];
                    let coeffs: [AdditiveShare<F>; 2] =
                        p.H.iter()
                            .zip(self.gamma.iter())
                            .map(|(h, gamma)| {
                                let h_0 = h.get_bound_coeff(2 * j_prime);
                                let h_1 = h.get_bound_coeff(2 * j_prime + 1);
                                let b = h_1 - h_0;
                                // h_0^2 - h_0: rep3 * rep3 → additive, then subtract rep3→additive
                                let h0_sq: AdditiveShare<F> = h_0 * h_0;
                                let h0_add: AdditiveShare<F> = h_0.into_additive();
                                let booleanity = (h0_sq - h0_add) * *gamma;
                                // b^2: rep3 * rep3 → additive
                                let b_sq: AdditiveShare<F> = b * b;
                                let quadratic = b_sq * *gamma;
                                [booleanity, quadratic]
                            })
                            .fold(
                                [AdditiveShare::zero(), AdditiveShare::zero()],
                                |running, new| [running[0] + new[0], running[1] + new[1]],
                            );

                    [coeffs[0] * D_eval, coeffs[1] * D_eval]
                })
                .reduce(
                    || [AdditiveShare::zero(), AdditiveShare::zero()],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            let num_x_in_bits = D_poly.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;
            let chunk_size = 1 << num_x_in_bits;

            (0..D_poly.len() / 2)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(x_out, chunk)| {
                    let D_E_out_eval: F = D_poly.E_out_current()[x_out];

                    let chunk_evals: [AdditiveShare<F>; DEGREE - 1] = chunk
                        .par_iter()
                        .map(|j_prime| {
                            let x_in = j_prime & x_bitmask;
                            let D_E_in_eval: F = D_poly.E_in_current()[x_in];
                            let coeffs: [AdditiveShare<F>; 2] =
                                p.H.iter()
                                    .zip(self.gamma.iter())
                                    .map(|(h, gamma)| {
                                        let h_0 = h.get_bound_coeff(2 * j_prime);
                                        let h_1 = h.get_bound_coeff(2 * j_prime + 1);
                                        let b = h_1 - h_0;
                                        let h0_sq: AdditiveShare<F> = h_0 * h_0;
                                        let h0_add: AdditiveShare<F> = h_0.into_additive();
                                        let booleanity = (h0_sq - h0_add) * *gamma;
                                        let b_sq: AdditiveShare<F> = b * b;
                                        let quadratic = b_sq * *gamma;
                                        [booleanity, quadratic]
                                    })
                                    .fold(
                                        [AdditiveShare::zero(), AdditiveShare::zero()],
                                        |running, new| [running[0] + new[0], running[1] + new[1]],
                                    );

                            [coeffs[0] * D_E_in_eval, coeffs[1] * D_E_in_eval]
                        })
                        .reduce(
                            || [AdditiveShare::zero(), AdditiveShare::zero()],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [chunk_evals[0] * D_E_out_eval, chunk_evals[1] * D_E_out_eval]
                })
                .reduce(
                    || [AdditiveShare::zero(), AdditiveShare::zero()],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        // Apply Gruen expansion with additive coefficients.
        let adjusted_claim = previous_claim * p.eq_r_r.inverse().unwrap();
        let gruen_evals = gruen_evals_deg_3(
            D_poly,
            quadratic_coeffs[0].into(),
            quadratic_coeffs[1].into(),
            adjusted_claim,
            self.party_id,
        );
        vec![
            gruen_evals[0] * p.eq_r_r,
            gruen_evals[1] * p.eq_r_r,
            gruen_evals[2] * p.eq_r_r,
        ]
    }
}

impl<F: JoltField> Rep3SumcheckInstanceWorker<F> for Rep3BooleanitySumcheckWorker<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK + self.log_T
    }

    fn input_claim_public(&self) -> F {
        F::zero()
    }

    fn compute_prover_message_share(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
    ) -> Vec<AdditiveShare<F>> {
        let base = if round < LOG_K_CHUNK {
            self.compute_phase1_message(round, previous_claim)
        } else {
            self.compute_phase2_message(round, previous_claim)
        };
        extend_degree_3_evals::<F>(previous_claim, &base, max_degree)
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let ps = &mut self.state;

        if round < LOG_K_CHUNK {
            // Phase 1: Bind address eq and update F_table (all public)
            ps.eq_r_address.bind(r_j);
            let r_j_f: F = r_j.into();
            let size = 1 << round;
            let (F_left, F_right) = ps.F_table.split_at_mut(size);
            F_left
                .par_iter_mut()
                .zip(F_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j_f;
                    *x -= *y;
                });

            if round == LOG_K_CHUNK - 1 {
                ps.eq_r_r = ps.eq_r_address.get_current_scalar();
                let f_table = std::mem::take(&mut ps.F_table);

                // Initialize H polynomials using shifted_table_from_rand_ohv
                for i in 0..D {
                    let shifted_table =
                        shifted_table_from_rand_ohv(&f_table, &ps.one_hot_e_fields[i]);
                    ps.H[i] = Rep3RaPolynomial::new(ps.masked_H_indices[i].clone(), shifted_table);
                }

                // Drop G (no longer needed)
                for i in 0..D {
                    ps.G[i] = vec![];
                }
            }
        } else {
            // Phase 2: Bind cycle eq and H polynomials
            ps.eq_r_cycle.bind(r_j);
            ps.H.par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address, r_cycle) = opening_point.split_at(LOG_K_CHUNK);
        let mut r_big_endian: Vec<F::Challenge> = r_address.iter().rev().copied().collect();
        r_big_endian.extend(r_cycle.iter().copied().rev());
        OpeningPoint::new(r_big_endian)
    }

    fn cache_openings_worker(
        &self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let ra_claims: Vec<Rep3PrimeFieldShare<F>> = self
            .state
            .H
            .iter()
            .map(|ra| ra.final_sumcheck_claim())
            .collect();

        accumulator.append_sparse(
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionBooleanity,
            &opening_point.r[..LOG_K_CHUNK],
            &opening_point.r[LOG_K_CHUNK..],
            ra_claims.clone(),
        );

        ra_claims
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3BooleanitySumcheck<F: JoltField> {
    gamma: [F; D],
    r_address: Vec<F::Challenge>,
    log_T: usize,
}

impl<F: JoltField> Rep3BooleanitySumcheck<F> {
    pub fn new<T: Transcript>(transcript: &mut T, log_T: usize) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let r_address: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(LOG_K_CHUNK);

        Self {
            gamma: gamma_powers,
            r_address,
            log_T,
        }
    }

    /// Return gamma powers so the worker can use them.
    pub fn gamma(&self) -> [F; D] {
        self.gamma
    }

    /// Return r_address so the worker can use them.
    pub fn r_address(&self) -> &[F::Challenge] {
        &self.r_address
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK + self.log_T
    }

    fn input_claim_public(&self) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r_prime: &[F::Challenge],
    ) -> F {
        let ra_claims = (0..D).map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionBooleanity,
                )
                .1
        });
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();

        EqPolynomial::<F>::mle(
            r_prime,
            &self
                .r_address
                .iter()
                .cloned()
                .rev()
                .chain(r_cycle.iter().cloned().rev())
                .collect::<Vec<F::Challenge>>(),
        ) * self
            .gamma
            .iter()
            .zip(ra_claims)
            .fold(F::zero(), |acc, (gamma, ra)| {
                (ra.square() - ra) * gamma + acc
            })
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address, r_cycle) = opening_point.split_at(LOG_K_CHUNK);
        let mut r_big_endian: Vec<F::Challenge> = r_address.iter().rev().copied().collect();
        r_big_endian.extend(r_cycle.iter().copied().rev());
        OpeningPoint::new(r_big_endian)
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        accumulator.append_sparse(
            transcript,
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionBooleanity,
            &r_sumcheck.r[..LOG_K_CHUNK],
            &r_sumcheck.r[LOG_K_CHUNK..],
            claims,
        );
    }
}

// ---------------------------------------------------------------------------
// Gruen helpers for shared/additive coefficients
// ---------------------------------------------------------------------------

/// Gruen degree-3 expansion with Rep3PrimeFieldShare quadratic coefficients.
///
/// Uses the linearity of `gruen_evals_deg_3(q0, q_inf, prev)` in (q0, q_inf)
/// to extract public coefficients via basis vector calls, then applies them
/// to the shared quadratic coefficients. Returns `[eval_0, eval_2, eval_3]`
/// as `AdditiveShare`.
pub(crate) fn gruen_evals_deg_3<F: JoltField>(
    eq: &GruenSplitEqPolynomial<F>,
    q0: Rep3Value<F>,
    q_inf: Rep3Value<F>,
    previous_claim: AdditiveShare<F>,
    party_id: PartyID,
) -> Vec<AdditiveShare<F>> {
    // gruen_evals_deg_3 is affine-linear in (q0, q_inf, previous_claim):
    //   f(q0, q_inf, prev) = A*q0 + B*q_inf + C*prev + D
    // where:
    //   D = f(0,0,0)
    //   A = f(1,0,0) - D
    //   B = f(0,1,0) - D
    //   C = f(0,0,1) - D
    let d = eq.gruen_evals_deg_3(F::zero(), F::zero(), F::zero());
    let a_plus_d = eq.gruen_evals_deg_3(F::one(), F::zero(), F::zero());
    let b_plus_d = eq.gruen_evals_deg_3(F::zero(), F::one(), F::zero());
    let c_plus_d = eq.gruen_evals_deg_3(F::zero(), F::zero(), F::one());

    let prev = Rep3Value::Additive(previous_claim);
    (0..3)
        .map(|i| {
            let a_i = a_plus_d[i] - d[i];
            let b_i = b_plus_d[i] - d[i];
            let c_i = c_plus_d[i] - d[i];

            let t = q0.mul_public(a_i).add(&q_inf.mul_public(b_i), party_id);
            let t = t.add(&prev.mul_public(c_i), party_id);
            t.add_public(d[i], party_id).into_additive(party_id)
        })
        .collect()
}

pub(crate) fn extend_degree_3_evals<F: JoltField>(
    previous_claim: AdditiveShare<F>,
    base: &[AdditiveShare<F>],
    max_degree: usize,
) -> Vec<AdditiveShare<F>> {
    debug_assert_eq!(base.len(), DEGREE);
    debug_assert!(max_degree >= DEGREE);

    if max_degree == DEGREE {
        return base.to_vec();
    }

    // Nodes for degree-3 polynomial at x=0..3.
    let y0 = base[0];
    let y1 = previous_claim - y0;
    let y2 = base[1]; // eval at 2
    let y3 = base[2]; // eval at 3

    let mut evals = vec![AdditiveShare::<F>::zero(); max_degree];
    evals[0] = y0;
    evals[1] = y2;
    evals[2] = y3;

    // Evaluate at x = 4..=max_degree via Lagrange on nodes 0..3.
    for x in 4..=max_degree {
        let xf = F::from(x as u64);
        let coeffs = lagrange_coeffs_consecutive_3::<F>(xf);
        evals[x - 1] = y0 * coeffs[0] + y1 * coeffs[1] + y2 * coeffs[2] + y3 * coeffs[3];
    }

    evals
}

fn lagrange_coeffs_consecutive_3<F: JoltField>(x: F) -> [F; 4] {
    // degree=3 nodes {0,1,2,3}. Precompute denominators and compute numerators on the fly.
    // denom(k) = Π_{m!=k} (k - m).
    let den0 = (F::from(0u64) - F::from(1u64))
        * (F::from(0u64) - F::from(2u64))
        * (F::from(0u64) - F::from(3u64));
    let den1 = (F::from(1u64) - F::from(0u64))
        * (F::from(1u64) - F::from(2u64))
        * (F::from(1u64) - F::from(3u64));
    let den2 = (F::from(2u64) - F::from(0u64))
        * (F::from(2u64) - F::from(1u64))
        * (F::from(2u64) - F::from(3u64));
    let den3 = (F::from(3u64) - F::from(0u64))
        * (F::from(3u64) - F::from(1u64))
        * (F::from(3u64) - F::from(2u64));

    let num0 = (x - F::from(1u64)) * (x - F::from(2u64)) * (x - F::from(3u64));
    let num1 = (x - F::from(0u64)) * (x - F::from(2u64)) * (x - F::from(3u64));
    let num2 = (x - F::from(0u64)) * (x - F::from(1u64)) * (x - F::from(3u64));
    let num3 = (x - F::from(0u64)) * (x - F::from(1u64)) * (x - F::from(2u64));

    [
        num0 * den0.inverse().unwrap(),
        num1 * den1.inverse().unwrap(),
        num2 * den2.inverse().unwrap(),
        num3 * den3.inverse().unwrap(),
    ]
}
