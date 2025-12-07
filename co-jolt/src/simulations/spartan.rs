use std::env;

use ark_ff::AdditiveGroup;
use itertools::{izip, Itertools};
use jolt_core::{
    field::OptimizedMul,
    poly::{
        dense_interleaved_poly::DenseInterleavedPolynomial,
        eq_poly,
        multilinear_polynomial::MultilinearPolynomial,
        sparse_interleaved_poly::SparseCoefficient,
        spartan_interleaved_poly::SpartanInterleavedPolynomial,
        split_eq_poly::{GruenSplitEqPolynomial, SplitEqPolynomial},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    r1cs::{
        builder::CombinedUniformBuilder,
        constraints::{JoltRV32IMConstraints, R1CSConstraints as _},
        inputs::JoltR1CSInputs,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::transcript::{AppendToTranscript, KeccakTranscript, Transcript},
};
use rayon::prelude::*;
use snarks_core::math::Math;

use crate::{field::JoltField, poly::split_eq_poly::DistributedSplitEqPolynomial};

pub fn simulate_sumcheck_distributed_spartan<F: JoltField, ProofTranscript: Transcript>(
    claim: &F,
    num_rounds: usize,
    eq_polys: &mut [GruenSplitEqPolynomial<F>],
    eq_poly: &mut GruenSplitEqPolynomial<F>,
    // r_grand_product: &[F],
    workers_polys: &mut [SpartanInterleavedPolynomial<F>],
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, (F, F)) {
    let mut previous_claim = *claim;
    let mut r: Vec<F> = Vec::new();
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

    let num_workers = workers_polys.len();
    let worker_rounds = eq_polys[0].get_num_vars() - 2;

    let remaining_rounds = num_rounds - worker_rounds;
    println!(
        "num_rounds: {} worker_rounds {} remaining_rounds {}",
        num_rounds, worker_rounds, remaining_rounds
    );

    for round in 0..worker_rounds {
        let mut eval_points = workers_polys
            .iter()
            .enumerate()
            .map(|(worker, poly)| {
                let evals = if round == 0 {
                    first_sumcheck_round(poly, &eq_polys[worker])
                } else {
                    subsequent_sumcheck_round(poly, &eq_polys[worker])
                };
                println!("----------------");
                vec![evals.0, evals.1]
            })
            .reduce(|mut eval_points, eval_points_next| {
                izip!(eval_points.iter_mut(), eval_points_next).for_each(|(a, b)| *a += b);
                eval_points
            })
            .unwrap();

        println!("-------");
        println!("eval points: {:?}", eval_points);
        println!("--------------");

        let cubic_poly = {
            let scalar_times_w_i = eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];

            let cubic_poly = UniPoly::from_linear_times_quadratic_with_hint(
                // The coefficients of `eq(w[(n - i)..], r[..i]) * eq(w[n - i - 1], X)`
                [
                    eq_poly.current_scalar - scalar_times_w_i,
                    scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
                ],
                eval_points[0],
                eval_points[1],
                previous_claim,
            );

            // Compress and add to transcript
            cubic_poly
        };
        let compressed_poly = cubic_poly.compress();
        compressed_poly.append_to_transcript(transcript);

        // Derive challenge
        let r_i: F = transcript.challenge_scalar();
        r.push(r_i);

        // Evaluate for next round's claim
        previous_claim = cubic_poly.evaluate(&r_i);
        eq_poly.bind(r_i);

        // bound all tables to the verifier's challenge
        workers_polys.iter_mut().for_each(|poly| {
            if round == 0 {
                first_sumcheck_round_bind(poly, r_i);
            } else {
                subsequent_sumcheck_round_bind(poly, r_i);
            }
        });
        eq_polys.par_iter_mut().for_each(|poly| poly.bind(r_i));
        compressed_polys.push(compressed_poly);
    }

    (
        SumcheckInstanceProof::new(compressed_polys),
        r,
        (F::ZERO, F::ZERO),
    )
}

/// The first round of the first Spartan sumcheck. Since the polynomials
/// are still unbound at the beginning of this round, we can replace some
/// of the field arithmetic with `i128` arithmetic.
///
/// Note that we implement the extra optimization of only computing the quadratic
/// evaluation at infinity, since the eval at zero is always zero.
#[tracing::instrument(skip_all, name = "SpartanInterleavedPolynomial::first_sumcheck_round")]
pub fn first_sumcheck_round<F: JoltField>(
    poly: &SpartanInterleavedPolynomial<F>,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> (F, F) {
    let num_x_in_bits = eq_poly.E_in_current_len().log_2();
    let x_in_bitmask = (1 << num_x_in_bits) - 1;

    // In the first round, we only need to compute the quadratic evaluation at infinity,
    // since the eval at zero is always zero.
    let quadratic_eval_at_infty = poly
        .unbound_coeffs_shards
        .par_iter()
        .map(|shard_coeffs| {
            let mut shard_eval_point_infty = F::zero();

            // let mut current_shard_inner_sums = F::zero();
            let mut current_shard_prev_x_out = 0;

            for sparse_block in shard_coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                let block_index = sparse_block[0].index / 6;
                let x_in = block_index & x_in_bitmask;
                println!(
                    "x_in: {} x_out: {} E_in_len: {} E_out_len: {}",
                    x_in,
                    block_index >> num_x_in_bits,
                    eq_poly.E_in_current_len(),
                    eq_poly.E_out_current_len()
                );

                let x_out = block_index >> num_x_in_bits;
                let E_in_evals = eq_poly.E_in_current()[x_in] * eq_poly.E_out_current()[x_out];

                // if x_out != current_shard_prev_x_out {
                // //     shard_eval_point_infty += eq_poly.E_out_current()[current_shard_prev_x_out]
                // //         * current_shard_inner_sums;
                // //     current_shard_inner_sums = F::zero();
                //     current_shard_prev_x_out = x_out;
                // }

                // This holds the az0, az1, bz0, bz1 evals. No need for cz0, cz1 since we only need
                // the eval at infinity.
                let mut az0 = 0i128;
                let mut az1 = 0i128;
                let mut bz0 = 0i128;
                let mut bz1 = 0i128;
                for coeff in sparse_block {
                    let local_idx = coeff.index % 6;
                    if local_idx == 0 {
                        az0 = coeff.value;
                    } else if local_idx == 1 {
                        bz0 = coeff.value;
                    } else if local_idx == 3 {
                        az1 = coeff.value;
                    } else if local_idx == 4 {
                        bz1 = coeff.value;
                    }
                }
                let az_infty = az1 - az0;
                let bz_infty = bz1 - bz0;

                println!("nonzero {} {:?}", E_in_evals, [az1, az0, bz1, bz0]);

                if az_infty != 0 && bz_infty != 0 {
                    shard_eval_point_infty += E_in_evals.mul_i128(bz_infty * az_infty);
                } else {
                    // println!(
                    //     "zero az {:?} bz {:?}, E_eval: {:?}",
                    //     [az0, az1],
                    //     [bz0, bz1],
                    //     E_in_evals
                    // );
                }

                // shard_eval_point_infty += E_in_evals.mul_i128(az_infty);
                // shard_eval_point_infty += E_in_evals * az_infty;
            }
            // shard_eval_point_infty +=
            //     eq_poly.E_out_current()[current_shard_prev_x_out] * current_shard_inner_sums;
            shard_eval_point_infty
        })
        .sum();

    (F::zero(), quadratic_eval_at_infty)
}

pub fn first_sumcheck_round_bind<F: JoltField>(poly: &mut SpartanInterleavedPolynomial<F>, r_i: F) {
    // Compute the number of non-zero bound coefficients that will be produced
    // per chunk.
    let output_sizes: Vec<_> = poly
        .unbound_coeffs_shards
        .par_iter()
        .map(|shard| SpartanInterleavedPolynomial::<F>::binding_output_length(shard))
        .collect();

    let total_output_len = output_sizes.iter().sum();
    poly.bound_coeffs = Vec::with_capacity(total_output_len);
    #[allow(clippy::uninit_vec)]
    unsafe {
        poly.bound_coeffs.set_len(total_output_len);
    }
    let mut output_slices: Vec<&mut [SparseCoefficient<F>]> =
        Vec::with_capacity(poly.unbound_coeffs_shards.len());
    let mut remainder = poly.bound_coeffs.as_mut_slice();
    for slice_len in output_sizes {
        let (first, second) = remainder.split_at_mut(slice_len);
        output_slices.push(first);
        remainder = second;
    }
    debug_assert_eq!(remainder.len(), 0);

    poly.unbound_coeffs_shards
        .par_iter()
        .zip_eq(output_slices.into_par_iter())
        .for_each(|(unbound_coeffs_in_shard, output_slice_for_shard)| {
            let mut output_index = 0;
            for block in unbound_coeffs_in_shard.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                let block_index = block[0].index / 6;

                let mut az_coeff: (Option<i128>, Option<i128>) = (None, None);
                let mut bz_coeff: (Option<i128>, Option<i128>) = (None, None);
                let mut cz_coeff: (Option<i128>, Option<i128>) = (None, None);

                for coeff in block {
                    match coeff.index % 6 {
                        0 => az_coeff.0 = Some(coeff.value),
                        1 => bz_coeff.0 = Some(coeff.value),
                        2 => cz_coeff.0 = Some(coeff.value),
                        3 => az_coeff.1 = Some(coeff.value),
                        4 => bz_coeff.1 = Some(coeff.value),
                        5 => cz_coeff.1 = Some(coeff.value),
                        _ => unreachable!(),
                    }
                }
                if az_coeff != (None, None) {
                    let (low, high) = (az_coeff.0.unwrap_or(0), az_coeff.1.unwrap_or(0));
                    output_slice_for_shard[output_index] = (
                        3 * block_index,
                        F::from_i128(low) + r_i.mul_i128(high - low),
                    )
                        .into();
                    output_index += 1;
                }
                if bz_coeff != (None, None) {
                    let (low, high) = (bz_coeff.0.unwrap_or(0), bz_coeff.1.unwrap_or(0));
                    output_slice_for_shard[output_index] = (
                        3 * block_index + 1,
                        F::from_i128(low) + r_i.mul_i128(high - low),
                    )
                        .into();
                    output_index += 1;
                }
                if cz_coeff != (None, None) {
                    let (low, high) = (cz_coeff.0.unwrap_or(0), cz_coeff.1.unwrap_or(0));
                    output_slice_for_shard[output_index] = (
                        3 * block_index + 2,
                        F::from_i128(low) + r_i.mul_i128(high - low),
                    )
                        .into();
                    output_index += 1;
                }
            }
            debug_assert_eq!(output_index, output_slice_for_shard.len())
        });

    // Drop the unbound coeffs shards now that we've bound them
    poly.unbound_coeffs_shards.clear();
    poly.unbound_coeffs_shards.shrink_to_fit();

    poly.dense_len /= 2;
}

pub fn subsequent_sumcheck_round<F: JoltField>(
    poly: &SpartanInterleavedPolynomial<F>,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> (F, F) {
    // In order to parallelize, we do a first pass over the coefficients to
    // determine how to divide it into chunks that can be processed independently.
    // In particular, coefficients whose indices are the same modulo 6 cannot
    // be processed independently.
    let block_size = poly
        .bound_coeffs
        .len()
        .div_ceil(rayon::current_num_threads())
        .next_multiple_of(6);
    let chunks: Vec<_> = poly
        .bound_coeffs
        .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
        .collect();

    if eq_poly.E_in_current_len() == 1 {
        let evals = chunks
            .par_iter()
            .flat_map_iter(|chunk| {
                chunk
                    .chunk_by(|x, y| x.index / 6 == y.index / 6)
                    .map(|sparse_block| {
                        let block_index = sparse_block[0].index / 6;
                        let mut block = [F::zero(); 6];
                        for coeff in sparse_block {
                            block[coeff.index % 6] = coeff.value;
                        }

                        let az = (block[0], block[3]);
                        let bz = (block[1], block[4]);
                        let cz0 = block[2];

                        let az_eval_infty = az.1 - az.0;
                        let bz_eval_infty = bz.1 - bz.0;

                        let eq_evals = eq_poly.E_out_current()[block_index];

                        (
                            eq_evals.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0),
                            eq_evals.mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty)),
                        )
                    })
            })
            .reduce(
                || (F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
            );
        evals
    } else {
        let num_x_in_bits = eq_poly.E_in_current_len().log_2();
        let x_bitmask = (1 << num_x_in_bits) - 1;

        let evals = chunks
            .par_iter()
            .map(|chunk| {
                let mut eval_point_0 = F::zero();
                let mut eval_point_infty = F::zero();

                let mut inner_sums = (F::zero(), F::zero());
                let mut prev_x_out = 0;

                for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = sparse_block[0].index / 6;
                    let x_in = block_index & x_bitmask;
                    let E_in_eval = eq_poly.E_in_current()[x_in];
                    let x_out = block_index >> num_x_in_bits;
                    // println!(
                    //     "x_in: {} x_out: {} E_in_len: {} E_out_len: {}",
                    //     x_in,
                    //     x_out,
                    //     eq_poly.E_in_current_len(),
                    //     eq_poly.E_out_current_len()
                    // );

                    if x_out != prev_x_out {
                        let E_out_eval = eq_poly.E_out_current()[prev_x_out];
                        eval_point_0 += E_out_eval * inner_sums.0;
                        eval_point_infty += E_out_eval * inner_sums.1;

                        inner_sums = (F::zero(), F::zero());
                        prev_x_out = x_out;
                    }

                    let mut block = [F::zero(); 6];
                    for coeff in sparse_block {
                        block[coeff.index % 6] = coeff.value;
                    }

                    let az = (block[0], block[3]);
                    let bz = (block[1], block[4]);
                    let cz0 = block[2];

                    let az_eval_infty = az.1 - az.0;
                    let bz_eval_infty = bz.1 - bz.0;

                    // println!(
                    //     "{} block {:?}",
                    //     E_in_eval * eq_poly.E_out_current()[prev_x_out],
                    //     block
                    // );

                    inner_sums.0 += E_in_eval.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0);
                    inner_sums.1 +=
                        E_in_eval.mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty));
                    // println!("------");
                }

                eval_point_0 += eq_poly.E_out_current()[prev_x_out] * inner_sums.0;
                eval_point_infty += eq_poly.E_out_current()[prev_x_out] * inner_sums.1;

                (eval_point_0, eval_point_infty)
            })
            .reduce(
                || (F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
            );
        evals
    }
}

pub fn subsequent_sumcheck_round_bind<F: JoltField>(
    poly: &mut SpartanInterleavedPolynomial<F>,
    r_i: F,
) {
    let block_size = poly
        .bound_coeffs
        .len()
        .div_ceil(rayon::current_num_threads())
        .next_multiple_of(6);

    let chunks: Vec<_> = poly
        .bound_coeffs
        .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
        .collect();

    let output_sizes: Vec<_> = chunks
        .par_iter()
        .map(|chunk| SpartanInterleavedPolynomial::<F>::binding_output_length(chunk))
        .collect();

    let total_output_len = output_sizes.iter().sum();
    if poly.binding_scratch_space.is_empty() {
        poly.binding_scratch_space = Vec::with_capacity(total_output_len);
    }
    unsafe {
        poly.binding_scratch_space.set_len(total_output_len);
    }

    let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(chunks.len());
    let mut remainder = poly.binding_scratch_space.as_mut_slice();
    for slice_len in output_sizes {
        let (first, second) = remainder.split_at_mut(slice_len);
        output_slices.push(first);
        remainder = second;
    }
    debug_assert_eq!(remainder.len(), 0);

    chunks
        .par_iter()
        .zip_eq(output_slices.into_par_iter())
        .for_each(|(coeffs, output_slice)| {
            let mut output_index = 0;
            for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                let block_index = block[0].index / 6;

                let mut az_coeff: (Option<F>, Option<F>) = (None, None);
                let mut bz_coeff: (Option<F>, Option<F>) = (None, None);
                let mut cz_coeff: (Option<F>, Option<F>) = (None, None);

                for coeff in block {
                    match coeff.index % 6 {
                        0 => az_coeff.0 = Some(coeff.value),
                        1 => bz_coeff.0 = Some(coeff.value),
                        2 => cz_coeff.0 = Some(coeff.value),
                        3 => az_coeff.1 = Some(coeff.value),
                        4 => bz_coeff.1 = Some(coeff.value),
                        5 => cz_coeff.1 = Some(coeff.value),
                        _ => unreachable!(),
                    }
                }
                if az_coeff != (None, None) {
                    let (low, high) = (
                        az_coeff.0.unwrap_or(F::zero()),
                        az_coeff.1.unwrap_or(F::zero()),
                    );
                    output_slice[output_index] = (3 * block_index, low + r_i * (high - low)).into();
                    output_index += 1;
                }
                if bz_coeff != (None, None) {
                    let (low, high) = (
                        bz_coeff.0.unwrap_or(F::zero()),
                        bz_coeff.1.unwrap_or(F::zero()),
                    );
                    output_slice[output_index] =
                        (3 * block_index + 1, low + r_i * (high - low)).into();
                    output_index += 1;
                }
                if cz_coeff != (None, None) {
                    let (low, high) = (
                        cz_coeff.0.unwrap_or(F::zero()),
                        cz_coeff.1.unwrap_or(F::zero()),
                    );
                    output_slice[output_index] =
                        (3 * block_index + 2, low + r_i * (high - low)).into();
                    output_index += 1;
                }
            }
            debug_assert_eq!(output_index, output_slice.len())
        });

    std::mem::swap(&mut poly.bound_coeffs, &mut poly.binding_scratch_space);
    poly.dense_len /= 2;
}

#[test]
fn test_distributed_spartan_simulation() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    type F = ark_bn254::Fr;

    let P: usize = env::var("P")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let W = 2;
    let P_w = P / W;

    let constraint_builder: Vec<CombinedUniformBuilder<4, F, JoltR1CSInputs>> = (0..W)
        .map(|_| JoltRV32IMConstraints::construct_constraints(P_w, (P / 2) as u64))
        .collect_vec();

    let flattened_polys = (0..W)
        .map(|w| {
            (0..78)
                .map(|i| {
                    MultilinearPolynomial::from(
                        (P_w * w..P_w * (w + 1))
                            .map(|e| (e + i) as i64)
                            .collect_vec(),
                        // vec![i as i64; P_w],
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    let num_rounds_x = 20;
    let mut transcript = KeccakTranscript::new(&[]);
    let tau = transcript.challenge_vector(num_rounds_x);
    println!("tau: {:?}", tau);
    let mut eq_tau = GruenSplitEqPolynomial::new(&tau);
    let mut eq_polys = (0..W)
        .map(|w| GruenSplitEqPolynomial::new_worker(&tau, 1, w))
        .collect_vec();

    for eq in &eq_polys {
        println!(
            "eq: E_in {:?}, E_out {:?}",
            eq.E_in_vec.iter().map(|e| e.len()).collect_vec(),
            eq.E_out_vec.iter().map(|e| e.len()).collect_vec()
        );
    }

    // let weq_check = eq_polys.iter().flat_map(|e| e.merge().Z).collect_vec();
    // assert_eq!(eq_tau.merge().Z, weq_check);

    let mut az_bz_cz_poly = flattened_polys
        .iter()
        .enumerate()
        .map(|(i, fp)| constraint_builder[i].compute_spartan_Az_Bz_Cz(&fp.iter().collect_vec()))
        .collect_vec();
    let mut claim = F::ZERO;
    let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
        simulate_sumcheck_distributed_spartan(
            &mut claim,
            num_rounds_x,
            &mut eq_polys,
            &mut eq_tau,
            &mut az_bz_cz_poly,
            &mut transcript,
        );

    let mut transcript = KeccakTranscript::new(&[]);

    outer_sumcheck_proof
        .verify(F::ZERO, num_rounds_x, 3, &mut transcript)
        .unwrap();
}

#[test]
fn test_local_spartan_simulation() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    type F = ark_bn254::Fr;

    let P: usize = env::var("P")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let N: usize = env::var("BATCH_SIZE")
        .unwrap_or_else(|_| "4".to_string())
        .parse()
        .unwrap();

    let constraint_builder: CombinedUniformBuilder<4, F, JoltR1CSInputs> =
        JoltRV32IMConstraints::construct_constraints(P, (P / 2) as u64);

    let flattened_polys = (0..78)
        .map(|i| {
            MultilinearPolynomial::from({
                (0..P).map(|e| (e + i) as i64).collect_vec()
                // vec![i as i64; P]
            })
        })
        .collect_vec();

    let num_rounds_x = 20;
    let mut transcript = KeccakTranscript::new(&[]);
    let tau = transcript.challenge_vector(num_rounds_x);
    println!("tau: {:?}", tau);
    let mut eq_tau = GruenSplitEqPolynomial::new(&tau);
    println!(
        "eq_tau: E_in {:?}, E_out {:?}",
        eq_tau.E_in_vec.iter().map(|e| e.len()).collect_vec(),
        eq_tau.E_out_vec.iter().map(|e| e.len()).collect_vec()
    );

    let mut az_bz_cz_poly =
        constraint_builder.compute_spartan_Az_Bz_Cz(&flattened_polys.iter().collect_vec());
    let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
        SumcheckInstanceProof::prove_spartan_cubic(
            num_rounds_x,
            &mut eq_tau,
            &mut az_bz_cz_poly,
            &mut transcript,
        );

    let mut transcript = KeccakTranscript::new(&[]);

    outer_sumcheck_proof
        .verify(F::ZERO, num_rounds_x, 3, &mut transcript)
        .unwrap();
}
