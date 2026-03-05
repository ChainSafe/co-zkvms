use eyre::Context;
use jolt_core::zkvm::instruction_lookups::D;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{self, Rep3PrimeFieldShare};
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::ra_poly::Rep3RaPolynomial;

// ---------------------------------------------------------------------------
// Per-level product tree functions (mirrors vanilla eval_inter{2,4,8,16})
// ---------------------------------------------------------------------------

/// Level 1: multiply two degree-1 Rep3-shared polynomials.
///
/// Given `p = (p(0), p(1))` and `q = (q(0), q(1))` as Rep3 shares, returns
/// evaluations of `p*q` at `{1, 2, ∞}` as additive shares.
///
/// No network required — rep3 × rep3 → additive is local.
#[inline]
pub fn eval_inter2_rep3<F: JoltField>(
    (p0, p1): (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>),
    (q0, q1): (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>),
) -> [AdditiveShare<F>; 3] {
    let p_inf = p1 - p0;
    let p2 = p_inf + p1;
    let q_inf = q1 - q0;
    let q2 = q_inf + q1;
    [
        p1 * q1,       // eval at 1
        p2 * q2,       // eval at 2
        p_inf * q_inf, // eval at ∞
    ]
}

/// Level 2: given two level-1 products as Rep3 triples at `{1, 2, ∞}`,
/// extrapolate each to `{1, 2, 3, 4, ∞}` and multiply pointwise.
///
/// Returns 5 additive shares at `{1, 2, 3, 4, ∞}`.
pub fn eval_inter4_rep3<F: JoltField>(
    a: &[Rep3PrimeFieldShare<F>; 3],
    b: &[Rep3PrimeFieldShare<F>; 3],
) -> [AdditiveShare<F>; 5] {
    let a3 = ex2_rep3(&[a[0], a[1]], &a[2]);
    let a4 = ex2_rep3(&[a[1], a3], &a[2]);
    let b3 = ex2_rep3(&[b[0], b[1]], &b[2]);
    let b4 = ex2_rep3(&[b[1], b3], &b[2]);
    [
        &a[0] * &b[0], // at 1
        &a[1] * &b[1], // at 2
        a3 * b3,       // at 3
        a4 * b4,       // at 4
        &a[2] * &b[2], // at ∞
    ]
}

/// Level 3: given two level-2 products as Rep3 quintuples at `{1, 2, 3, 4, ∞}`,
/// extrapolate each to `{1, ..., 8, ∞}` and multiply pointwise.
///
/// Returns 9 additive shares at `{1, ..., 8, ∞}`.
pub fn eval_inter8_rep3<F: JoltField>(
    a: &[Rep3PrimeFieldShare<F>; 5],
    b: &[Rep3PrimeFieldShare<F>; 5],
) -> [AdditiveShare<F>; 9] {
    let a_inf6 = a[4] * F::from(6u64);
    let (a5, a6) = ex4_2_rep3(&[a[0], a[1], a[2], a[3]], &a_inf6);
    let (a7, a8) = ex4_2_rep3(&[a[2], a[3], a5, a6], &a_inf6);
    let b_inf6 = b[4] * F::from(6u64);
    let (b5, b6) = ex4_2_rep3(&[b[0], b[1], b[2], b[3]], &b_inf6);
    let (b7, b8) = ex4_2_rep3(&[b[2], b[3], b5, b6], &b_inf6);
    [
        &a[0] * &b[0], // at 1
        &a[1] * &b[1], // at 2
        &a[2] * &b[2], // at 3
        &a[3] * &b[3], // at 4
        a5 * b5,       // at 5
        a6 * b6,       // at 6
        a7 * b7,       // at 7
        a8 * b8,       // at 8
        &a[4] * &b[4], // at ∞
    ]
}

/// Level 4 (final accumulate): given two level-3 products as Rep3 nine-tuples
/// at `{1, ..., 8, ∞}`, extrapolate each to `{1, ..., 15, ∞}`, multiply
/// pointwise, and accumulate (weighted by `eq_wr_eval`) into `sum_evals`.
pub fn eval_inter16_final_accumulate_rep3<F: JoltField>(
    a: &[Rep3PrimeFieldShare<F>; 9],
    b: &[Rep3PrimeFieldShare<F>; 9],
    eq_wr_eval: F,
    sum_evals: &mut [AdditiveShare<F>],
) {
    let a_full = extrapolate_9_to_16_rep3(a);
    let b_full = extrapolate_9_to_16_rep3(b);
    for k in 0..D {
        let prod: AdditiveShare<F> = a_full[k] * b_full[k];
        sum_evals[k] += prod * eq_wr_eval;
    }
}

// ---------------------------------------------------------------------------
// Orchestrator: drives all 4 levels + 3 reshares
// ---------------------------------------------------------------------------

/// Compute the product of D=16 Rep3-shared degree-1 polynomials, summed over
/// all `j` weighted by split-eq factors `eq_wl_evals` and `eq_wr_evals`.
///
/// Performs 3 network round trips (reshares between levels 1→2, 2→3, 3→4).
///
/// Returns evaluations of the product polynomial (without the eq(X,r[round])
/// factor) at `{1, 2, ..., 15, ∞}` as `D` additive shares.
pub fn compute_mles_product_16_rep3<F: JoltField, N: Rep3NetworkWorker>(
    eq_wl_evals: &[F],
    eq_wr_evals: &[F],
    ra_i_polys: &[Rep3RaPolynomial<u8, F>],
    half: usize,
    wl_len: usize,
    io_ctx: &mut IoContextPool<N>,
) -> eyre::Result<Vec<AdditiveShare<F>>> {
    let n_wl = eq_wl_evals.len();
    let n_wr = eq_wr_evals.len();

    // ---- Level 1: local rep3 × rep3 → additive, per (j_wr, j_wl) ----
    // Layout: blocks of 8 triples per (j_wr, j_wl), each triple stored as 3 additive shares.
    let level1_block_len = 8 * 3;
    let level1_len = n_wr * n_wl * level1_block_len;
    let mut level1_additive: Vec<AdditiveShare<F>> = vec![AdditiveShare::zero(); level1_len];

    level1_additive
        .par_chunks_mut(n_wl * level1_block_len)
        .enumerate()
        .for_each(|(j_wr, out_wr)| {
            for (j_wl, &eq_wl_eval) in eq_wl_evals.iter().enumerate() {
                let j = j_wl + (j_wr << wl_len);
                let out = &mut out_wr[j_wl * level1_block_len..(j_wl + 1) * level1_block_len];

                let mut pairs: [(Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>); D] =
                    std::array::from_fn(|i| {
                        let e0 = ra_i_polys[i].get_bound_coeff(j);
                        let e1 = ra_i_polys[i].get_bound_coeff(j + half);
                        (e0, e1)
                    });

                // Fold eq_wl into the first pair (public × rep3).
                pairs[0].0 *= eq_wl_eval;
                pairs[0].1 *= eq_wl_eval;

                for pair_idx in 0..8 {
                    let triple = eval_inter2_rep3(pairs[2 * pair_idx], pairs[2 * pair_idx + 1]);
                    let base = pair_idx * 3;
                    out[base] = triple[0];
                    out[base + 1] = triple[1];
                    out[base + 2] = triple[2];
                }
            }
        });

    // ---- Reshare level 1 → rep3 ----
    let level1_rep3 = rep3::arithmetic::reshare_additive_many(&level1_additive, io_ctx.main())
        .context("reshare level 1")?;
    drop(level1_additive);

    // ---- Level 2: eval_inter4 on rep3 shares → additive ----
    let level2_block_len = 4 * 5;
    let level2_len = n_wr * n_wl * level2_block_len;
    let mut level2_additive: Vec<AdditiveShare<F>> = vec![AdditiveShare::zero(); level2_len];

    level2_additive
        .par_chunks_mut(level2_block_len)
        .enumerate()
        .for_each(|(idx, out)| {
            let start = idx * level1_block_len;
            let in_block = &level1_rep3[start..start + level1_block_len];

            let triples: [[Rep3PrimeFieldShare<F>; 3]; 8] =
                std::array::from_fn(|t| std::array::from_fn(|k| in_block[t * 3 + k]));

            for group in 0..4 {
                let result = eval_inter4_rep3(&triples[2 * group], &triples[2 * group + 1]);
                let base = group * 5;
                out[base..base + 5].copy_from_slice(&result);
            }
        });

    // ---- Reshare level 2 → rep3 ----
    let level2_rep3 = rep3::arithmetic::reshare_additive_many(&level2_additive, io_ctx.main())
        .context("reshare level 2")?;
    drop(level2_additive);

    // ---- Level 3: eval_inter8 on rep3 shares → additive ----
    let level3_block_len = 2 * 9;
    let level3_len = n_wr * n_wl * level3_block_len;
    let mut level3_additive: Vec<AdditiveShare<F>> = vec![AdditiveShare::zero(); level3_len];

    level3_additive
        .par_chunks_mut(level3_block_len)
        .enumerate()
        .for_each(|(idx, out)| {
            let start = idx * level2_block_len;
            let in_block = &level2_rep3[start..start + level2_block_len];

            let quints: [[Rep3PrimeFieldShare<F>; 5]; 4] =
                std::array::from_fn(|q| std::array::from_fn(|k| in_block[q * 5 + k]));

            for group in 0..2 {
                let result = eval_inter8_rep3(&quints[2 * group], &quints[2 * group + 1]);
                let base = group * 9;
                out[base..base + 9].copy_from_slice(&result);
            }
        });

    // ---- Reshare level 3 → rep3 ----
    let level3_rep3 = rep3::arithmetic::reshare_additive_many(&level3_additive, io_ctx.main())
        .context("reshare level 3")?;
    drop(level3_additive);

    // ---- Level 4: eval_inter16 final accumulate → additive, sum over j ----
    let sum = (0..n_wr)
        .into_par_iter()
        .map(|j_wr| {
            let mut local = [AdditiveShare::<F>::zero(); D];
            let eq_wr_eval = eq_wr_evals[j_wr];

            let base = j_wr * n_wl * level3_block_len;
            for j_wl in 0..n_wl {
                let start = base + j_wl * level3_block_len;
                let in_block = &level3_rep3[start..start + level3_block_len];

                let a: [Rep3PrimeFieldShare<F>; 9] = std::array::from_fn(|k| in_block[k]);
                let b: [Rep3PrimeFieldShare<F>; 9] = std::array::from_fn(|k| in_block[9 + k]);

                eval_inter16_final_accumulate_rep3(&a, &b, eq_wr_eval, &mut local);
            }

            local
        })
        .reduce(
            || [AdditiveShare::<F>::zero(); D],
            |mut running, new| {
                for i in 0..D {
                    running[i] += new[i];
                }
                running
            },
        );

    Ok(sum.to_vec())
}

// ---------------------------------------------------------------------------
// Extrapolation helpers (linear ops on Rep3 shares, no network)
// ---------------------------------------------------------------------------

/// Extrapolate from `{f(1), f(2), f(∞)}` to `f(3)`.
///
/// Formula: `f(3) = 2*(f(2) + f(∞)) - f(1)`.
#[inline]
fn ex2_rep3<F: JoltField>(
    f: &[Rep3PrimeFieldShare<F>; 2],
    f_inf: &Rep3PrimeFieldShare<F>,
) -> Rep3PrimeFieldShare<F> {
    let sum = f[1] + *f_inf;
    sum + sum - f[0]
}

/// Extrapolate from `{f(1), f(2), f(3), f(4), 6*f(∞)}` to `(f(5), f(6))`.
#[inline]
fn ex4_2_rep3<F: JoltField>(
    f: &[Rep3PrimeFieldShare<F>; 4],
    f_inf6: &Rep3PrimeFieldShare<F>,
) -> (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>) {
    let f3m2 = f[3] - f[2];
    let mut f4 = *f_inf6 + f3m2 + f[1];
    f4 = f4 + f4; // 2x
    f4 = f4 - f[2];
    f4 = f4 + f4; // 2x
    f4 = f4 - f[0];

    let mut f5 = f4 - f3m2 + *f_inf6;
    f5 = f5 + f5; // 2x
    f5 = f5 - f[3];
    f5 = f5 + f5; // 2x
    f5 = f5 - f[1];

    (f4, f5)
}

/// Extrapolate from `{f(1), ..., f(8), 40320*f(∞)}` to `f(9)`.
#[inline]
fn ex8_rep3<F: JoltField>(
    f: &[Rep3PrimeFieldShare<F>; 8],
    f_inf40320: Rep3PrimeFieldShare<F>,
) -> Rep3PrimeFieldShare<F> {
    let pos = (f[1] + f[7]) * F::from(8u64) + (f[3] + f[5]) * F::from(56u64) + f_inf40320;

    let neg = (f[2] + f[6]) * F::from(28u64) + f[4] * F::from(70u64) + f[0];

    pos - neg
}

/// Extrapolate 9 evaluations at `{1, ..., 8, ∞}` to 16 evaluations at
/// `{1, ..., 15, ∞}`.
fn extrapolate_9_to_16_rep3<F: JoltField>(
    vals: &[Rep3PrimeFieldShare<F>; 9],
) -> [Rep3PrimeFieldShare<F>; D] {
    let mut f = [Rep3PrimeFieldShare::<F>::zero_share(); D];
    f[..8].copy_from_slice(&vals[..8]);
    f[15] = vals[8]; // f(∞)

    let f_inf40320 = vals[8] * F::from(40320u64);
    for i in 0..7 {
        let slice: [Rep3PrimeFieldShare<F>; 8] = f[i..i + 8].try_into().unwrap();
        f[8 + i] = ex8_rep3(&slice, f_inf40320);
    }

    f
}
