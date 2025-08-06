use core::num;

use ark_ff::Zero;
use itertools::Itertools;
use jolt_core::poly::{
    eq_poly::EqPolynomial, multilinear_polynomial::MultilinearPolynomial,
    sparse_interleaved_poly::SparseCoefficient, split_eq_poly::GruenSplitEqPolynomial,
    unipoly::CompressedUniPoly,
};
use jolt_core::r1cs::builder::OffsetLC;
use jolt_core::subprotocols::sumcheck::process_eq_sumcheck_round;
use jolt_core::{
    field::{JoltField, OptimizedMul, OptimizedMulI128},
    r1cs::builder::{eval_offset_lc, Constraint, OffsetEqConstraint},
    utils::{
        math::Math,
        small_value::{svo_helpers, NUM_SVO_ROUNDS},
        transcript::Transcript,
    },
};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::{IoContext, Rep3NetworkWorker};

use super::multilinear_polynomial::Rep3MultilinearPolynomial;
use crate::r1cs::ops::LinearCombinationExt;
use crate::subprotocols::sumcheck_spartan::process_eq_sumcheck_round_worker;
use crate::utils::element::{SharedOrPublic, SharedOrPublicIter as _, SharedOrPublicParIter};
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Rep3SpartanInterleavedPolynomial<F: JoltField> {
    /// Shards of sparse vectors representing the (interleaved) coefficients in the Az, Bz, Cz
    /// polynomials used in the first Spartan sumcheck. Before the polynomial is bound
    /// TODO: make SharedOrPublic<F> support i128
    /// ~~the first time, all the coefficients can be represented by `i128`s.~~
    pub(crate) unbound_coeffs_shards: Vec<Vec<SparseCoefficient<SharedOrPublic<F>>>>,

    /// The bound coefficients for the Az, Bz, Cz polynomials. Will be populated in the streaming round
    pub(crate) bound_coeffs: Vec<SparseCoefficient<SharedOrPublic<F>>>,

    binding_scratch_space: Vec<SparseCoefficient<SharedOrPublic<F>>>,
    /// The length of one of the Az, Bz, or Cz polynomials if it were represented by
    /// a single dense vector.
    dense_len: usize,
}

impl<F: JoltField> Rep3SpartanInterleavedPolynomial<F> {
    /// Computes the matrix-vector products Az, Bz, and Cz as a single interleaved sparse vector
    pub fn new(
        uniform_constraints: &[Constraint],
        cross_step_constraints: &[OffsetEqConstraint],
        flattened_polynomials: &[&Rep3MultilinearPolynomial<F>], // N variables of (S steps)
        padded_num_constraints: usize,
        party_id: PartyID,
    ) -> Self {
        let num_steps = flattened_polynomials[0].len();

        let num_chunks = std::cmp::min(
            rayon::current_num_threads().next_power_of_two() * 16,
            num_steps / 2,
        );
        let chunk_size = num_steps.div_ceil(num_chunks);

        // let unbound_coeffs_shards_iter = (0..num_chunks).into_par_iter().map(|chunk_index| {
        let unbound_coeffs_shards_iter = (0..num_chunks).into_par_iter().map(|chunk_index| {
            let mut coeffs: Vec<SparseCoefficient<SharedOrPublic<F>>> =
                Vec::with_capacity(chunk_size * padded_num_constraints * 3);
            for step_index in chunk_size * chunk_index..chunk_size * (chunk_index + 1) {
                // Uniform constraints
                for (constraint_index, constraint) in uniform_constraints.iter().enumerate() {
                    let global_index = 3 * (step_index * padded_num_constraints + constraint_index);

                    // Az
                    let mut az_coeff = SharedOrPublic::zero_public();
                    if !constraint.a.terms().is_empty() {
                        az_coeff = constraint.a.evaluate_row_rep3_mixed(
                            flattened_polynomials,
                            step_index,
                            party_id,
                        );
                        if az_coeff.shared_or_not_zero() {
                            coeffs.push((global_index, az_coeff).into());
                        }
                    }
                    // Bz
                    let mut bz_coeff = SharedOrPublic::zero_public();
                    if !constraint.b.terms().is_empty() {
                        bz_coeff = constraint.b.evaluate_row_rep3_mixed(
                            flattened_polynomials,
                            step_index,
                            party_id,
                        );
                        if bz_coeff.shared_or_not_zero() {
                            coeffs.push((global_index + 1, bz_coeff).into());
                        }
                    }

                    // Cz = Az ⊙ Bz

                    match (az_coeff, bz_coeff) {
                        (SharedOrPublic::Public(x), SharedOrPublic::Public(y))
                            if x.is_zero() && y.is_zero() =>
                        {
                            continue;
                        }
                        (SharedOrPublic::Shared(_), SharedOrPublic::Shared(_)) => {
                            // If both Az and Bz are shared, then Cz is also shared, to avoid communication we can compute Cz via evaluation
                            let cz_coeff = constraint.c.evaluate_row_rep3_mixed(
                                flattened_polynomials,
                                step_index,
                                party_id,
                            );
                            coeffs.push((global_index + 2, cz_coeff.into()).into());
                        }
                        // otherwise we can compute Cz via multiplication Az ⊙ Bz
                        (_, _) => {
                            let cz_coeff = az_coeff.mul(&bz_coeff);
                            coeffs.push((global_index + 2, cz_coeff).into());
                        }
                    }
                }

                // For the final step we will not compute the offset terms, and will assume the condition to be set to 0
                let next_step_index = if step_index + 1 < num_steps {
                    Some(step_index + 1)
                } else {
                    None
                };

                // Cross-step constraints
                for (constraint_index, constraint) in cross_step_constraints.iter().enumerate() {
                    let global_index = 3
                        * (step_index * padded_num_constraints
                            + uniform_constraints.len()
                            + constraint_index);

                    // Az
                    let eq_a_eval = eval_offset_lc_rep3_mixed(
                        &constraint.a,
                        flattened_polynomials,
                        step_index,
                        next_step_index,
                        party_id,
                    );
                    let eq_b_eval = eval_offset_lc_rep3_mixed(
                        &constraint.b,
                        flattened_polynomials,
                        step_index,
                        next_step_index,
                        party_id,
                    );
                    let az_coeff = eq_a_eval.sub(&eq_b_eval, party_id);
                    coeffs.push((global_index, az_coeff).into());
                    // If Az != 0 and not shared, then the condition must be false (i.e. Bz = 0)
                    if matches!(az_coeff, SharedOrPublic::Public(f) if !f.is_zero()) {
                        continue;
                    }
                    // Otherwise, Bz could be != 0, so we need to compute it
                    let bz_coeff = eval_offset_lc_rep3_mixed(
                        &constraint.cond,
                        flattened_polynomials,
                        step_index,
                        next_step_index,
                        party_id,
                    );
                    coeffs.push((global_index + 1, bz_coeff).into());
                    // Cz is always 0 for cross-step constraints
                }
            }

            coeffs
        });
        let unbound_coeffs_shards: Vec<_> = unbound_coeffs_shards_iter.collect();

        Self {
            unbound_coeffs_shards,
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
            dense_len: num_steps * padded_num_constraints,
        }
    }

    pub fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    /// The first round of the first Spartan sumcheck. Since the polynomials
    /// are still unbound at the beginning of this round, we can replace some
    /// of the field arithmetic with `i128` arithmetic.
    ///
    /// Note that we implement the extra optimization of only computing the quadratic
    /// evaluation at infinity, since the eval at zero is always zero.
    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomial::first_sumcheck_round",
        level = "trace"
    )]
    pub fn first_sumcheck_round<Network: Rep3NetworkWorker>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        r: &mut Vec<F>,
        claim: &mut F,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.id;
        assert!(!self.is_bound());

        let num_x_in_bits = eq_poly.E_in_current_len().log_2();
        let x_in_bitmask = (1 << num_x_in_bits) - 1;

        // In the first round, we only need to compute the quadratic evaluation at infinity,
        // since the eval at zero is always zero.
        let span = tracing::trace_span!("quadratic_eval_at_infty");
        let _span_enter = span.enter();
        let quadratic_eval_at_infty = self
            .unbound_coeffs_shards
            .par_iter()
            .map(|shard_coeffs| {
                let mut shard_eval_point_infty = SharedOrPublic::zero_additive();

                // let mut current_shard_inner_sums = SharedOrPublic::zero_additive();
                // let mut current_shard_prev_x_out = 0;

                for sparse_block in shard_coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = sparse_block[0].index / 6;
                    let x_in = block_index & x_in_bitmask;
                    let x_out = block_index >> num_x_in_bits;

                    let E_in_evals = eq_poly.E_in_current()[x_in] * eq_poly.E_out_current()[x_out];

                    // if x_out != current_shard_prev_x_out {
                    //     shard_eval_point_infty += eq_poly.E_out_current()[current_shard_prev_x_out]
                    //         * current_shard_inner_sums;
                    //     current_shard_inner_sums = SharedOrPublic::zero_additive();
                    //     current_shard_prev_x_out = x_out;
                    // }

                    // This holds the az0, az1, bz0, bz1 evals. No need for cz0, cz1 since we only need
                    // the eval at infinity.
                    let mut az0 = SharedOrPublic::zero_public();
                    let mut az1 = SharedOrPublic::zero_public();
                    let mut bz0 = SharedOrPublic::zero_public();
                    let mut bz1 = SharedOrPublic::zero_public();
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
                    let az_infty = az1.sub(&az0, party_id);
                    let bz_infty = bz1.sub(&bz0, party_id);

                    if matches!((az_infty, bz_infty), (SharedOrPublic::Public(x), SharedOrPublic::Public(y)) if x.is_zero() && y.is_zero()) {
                        continue;
                    }

                    shard_eval_point_infty.add_assign(
                        &az_infty.mul_mul_public(&bz_infty, E_in_evals),
                        party_id,
                    );
                }
                shard_eval_point_infty
            })
            .sum_for(party_id);
        drop(_span_enter);

        let r_i = process_eq_sumcheck_round_worker(
            (F::zero(), quadratic_eval_at_infty.as_additive()),
            eq_poly,
            r,
            claim,
            io_ctx,
        )?;

        // Compute the number of non-zero bound coefficients that will be produced
        // per chunk.
        let output_sizes: Vec<_> = self
            .unbound_coeffs_shards
            .par_iter()
            .map(|shard| Self::binding_output_length(shard))
            .collect();

        let total_output_len = output_sizes.iter().sum();
        self.bound_coeffs = Vec::with_capacity(total_output_len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.bound_coeffs.set_len(total_output_len);
        }
        let mut output_slices: Vec<&mut [SparseCoefficient<SharedOrPublic<F>>]> =
            Vec::with_capacity(self.unbound_coeffs_shards.len());
        let mut remainder = self.bound_coeffs.as_mut_slice();
        for slice_len in output_sizes {
            let (first, second) = remainder.split_at_mut(slice_len);
            output_slices.push(first);
            remainder = second;
        }
        debug_assert_eq!(remainder.len(), 0);

        let span = tracing::trace_span!("unbound_coeffs_shards");
        let _span_enter = span.enter();
        self.unbound_coeffs_shards
            .par_iter()
            .zip_eq(output_slices.into_par_iter())
            .for_each(|(unbound_coeffs_in_shard, output_slice_for_shard)| {
                let mut output_index = 0;
                for block in unbound_coeffs_in_shard.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = block[0].index / 6;

                    let mut az_coeff: (Option<SharedOrPublic<F>>, Option<SharedOrPublic<F>>) =
                        (None, None);
                    let mut bz_coeff: (Option<SharedOrPublic<F>>, Option<SharedOrPublic<F>>) =
                        (None, None);
                    let mut cz_coeff: (Option<SharedOrPublic<F>>, Option<SharedOrPublic<F>>) =
                        (None, None);

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
                            az_coeff.0.unwrap_or(SharedOrPublic::zero_public()),
                            az_coeff.1.unwrap_or(SharedOrPublic::zero_public()),
                        );
                        output_slice_for_shard[output_index] = (
                            3 * block_index,
                            low.add(&high.sub(&low, party_id).mul_public(r_i), party_id),
                        )
                            .into();
                        output_index += 1;
                    }
                    if bz_coeff != (None, None) {
                        let (low, high) = (
                            bz_coeff.0.unwrap_or(SharedOrPublic::zero_public()),
                            bz_coeff.1.unwrap_or(SharedOrPublic::zero_public()),
                        );
                        output_slice_for_shard[output_index] = (
                            3 * block_index + 1,
                            low.add(&high.sub(&low, party_id).mul_public(r_i), party_id),
                        )
                            .into();
                        output_index += 1;
                    }
                    if cz_coeff != (None, None) {
                        let (low, high) = (
                            cz_coeff.0.unwrap_or(SharedOrPublic::zero_public()),
                            cz_coeff.1.unwrap_or(SharedOrPublic::zero_public()),
                        );
                        output_slice_for_shard[output_index] = (
                            3 * block_index + 2,
                            low.add(&high.sub(&low, party_id).mul_public(r_i), party_id),
                        )
                            .into();
                        output_index += 1;
                    }
                }
                debug_assert_eq!(output_index, output_slice_for_shard.len())
            });
        drop(_span_enter);

        // Drop the unbound coeffs shards now that we've bound them
        self.unbound_coeffs_shards.clear();
        self.unbound_coeffs_shards.shrink_to_fit();

        self.dense_len /= 2;

        Ok(())
    }

    /// All subsequent rounds of the first Spartan sumcheck.
    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomial::subsequent_sumcheck_round",
        level = "trace"
    )]
    pub fn subsequent_sumcheck_round<Network: Rep3NetworkWorker>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        r: &mut Vec<F>,
        claim: &mut F,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.id;
        assert!(self.is_bound());

        // In order to parallelize, we do a first pass over the coefficients to
        // determine how to divide it into chunks that can be processed independently.
        // In particular, coefficients whose indices are the same modulo 6 cannot
        // be processed independently.
        let block_size = self
            .bound_coeffs
            .len()
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(6);
        let chunks: Vec<_> = self
            .bound_coeffs
            .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
            .collect();

        let quadratic_evals = if eq_poly.E_in_current_len() == 1 {
            let span = tracing::trace_span!("quadratic_evals_in_current_len_1");
            let _span_enter = span.enter();
            let evals = chunks
                .par_iter()
                .flat_map_iter(|chunk| {
                    chunk
                        .chunk_by(|x, y| x.index / 6 == y.index / 6)
                        .map(|sparse_block| {
                            let block_index = sparse_block[0].index / 6;
                            let mut block = [SharedOrPublic::zero_additive(); 6];
                            for coeff in sparse_block {
                                block[coeff.index % 6] = coeff.value;
                            }

                            let az = (block[0], block[3]);
                            let bz = (block[1], block[4]);
                            let cz0 = block[2];

                            let az_eval_infty = az.1.sub(&az.0, party_id);
                            let bz_eval_infty = bz.1.sub(&bz.0, party_id);

                            let eq_evals = eq_poly.E_out_current()[block_index];

                            (
                                az.0.mul_mul_public(&bz.0, eq_evals).into_additive(party_id)
                                    - cz0.mul_public(eq_evals).into_additive(party_id),
                                az_eval_infty
                                    .mul_mul_public(&bz_eval_infty, eq_evals)
                                    .into_additive(party_id),
                            )
                        })
                })
                .reduce(
                    || (AdditiveShare::zero(), AdditiveShare::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                );
            drop(_span_enter);
            evals
        } else {
            let span = tracing::trace_span!("quadratic_evals_in_current_len_gt_1");
            let _span_enter = span.enter();
            let num_x_in_bits = eq_poly.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            let evals = chunks
                .par_iter()
                .map(|chunk| {
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_infty = F::zero();

                    // let mut inner_sums = (F::zero(), F::zero());
                    // let mut prev_x_out = 0;

                    for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let block_index = sparse_block[0].index / 6;
                        let x_in = block_index & x_bitmask;
                        let x_out = block_index >> num_x_in_bits;

                        let E_in_eval =
                            eq_poly.E_in_current()[x_in] * eq_poly.E_out_current()[x_out];

                        // if x_out != prev_x_out {
                        //     let E_out_eval = eq_poly.E_out_current()[prev_x_out];
                        //     eval_point_0 += E_out_eval * inner_sums.0;
                        //     eval_point_infty += E_out_eval * inner_sums.1;

                        //     inner_sums = (F::zero(), F::zero());
                        //     prev_x_out = x_out;
                        // }

                        let mut block = [SharedOrPublic::zero_public(); 6];
                        for coeff in sparse_block {
                            block[coeff.index % 6] = coeff.value;
                        }

                        let az = (block[0], block[3]);
                        let bz = (block[1], block[4]);
                        let cz0 = block[2];

                        let az_eval_infty = az.1.sub(&az.0, party_id);
                        let bz_eval_infty = bz.1.sub(&bz.0, party_id);

                        eval_point_0 +=
                            az.0.mul_mul_public(&bz.0, E_in_eval)
                                .into_additive(party_id)
                                - cz0.mul_public(E_in_eval).into_additive(party_id);
                        eval_point_infty += az_eval_infty
                            .mul_mul_public(&bz_eval_infty, E_in_eval)
                            .into_additive(party_id);
                    }

                    (eval_point_0, eval_point_infty)
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                );
            drop(_span_enter);
            evals
        };

        let r_i = process_eq_sumcheck_round_worker(quadratic_evals, eq_poly, r, claim, io_ctx)?;

        let output_sizes: Vec<_> = chunks
            .par_iter()
            .map(|chunk| Self::binding_output_length(chunk))
            .collect();

        let total_output_len = output_sizes.iter().sum();
        if self.binding_scratch_space.is_empty() {
            self.binding_scratch_space = Vec::with_capacity(total_output_len);
        }
        unsafe {
            self.binding_scratch_space.set_len(total_output_len);
        }

        let mut output_slices: Vec<&mut [SparseCoefficient<SharedOrPublic<F>>]> =
            Vec::with_capacity(chunks.len());
        let mut remainder = self.binding_scratch_space.as_mut_slice();
        for slice_len in output_sizes {
            let (first, second) = remainder.split_at_mut(slice_len);
            output_slices.push(first);
            remainder = second;
        }
        debug_assert_eq!(remainder.len(), 0);

        let span = tracing::trace_span!("bind_chunks");
        let _span_enter = span.enter();
        chunks
            .par_iter()
            .zip_eq(output_slices.into_par_iter())
            .for_each(|(coeffs, output_slice)| {
                let mut output_index = 0;
                for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = block[0].index / 6;

                    let mut az_coeff: (Option<SharedOrPublic<F>>, Option<SharedOrPublic<F>>) =
                        (None, None);
                    let mut bz_coeff: (Option<SharedOrPublic<F>>, Option<SharedOrPublic<F>>) =
                        (None, None);
                    let mut cz_coeff: (Option<SharedOrPublic<F>>, Option<SharedOrPublic<F>>) =
                        (None, None);

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
                            az_coeff.0.unwrap_or(SharedOrPublic::zero_public()),
                            az_coeff.1.unwrap_or(SharedOrPublic::zero_public()),
                        );
                        //  low + r_i * (high - low)
                        output_slice[output_index] = (
                            3 * block_index,
                            low.add(&high.sub(&low, party_id).mul_public(r_i), party_id),
                        )
                            .into();
                        output_index += 1;
                    }
                    if bz_coeff != (None, None) {
                        let (low, high) = (
                            bz_coeff.0.unwrap_or(SharedOrPublic::zero_public()),
                            bz_coeff.1.unwrap_or(SharedOrPublic::zero_public()),
                        );
                        output_slice[output_index] = (
                            3 * block_index + 1,
                            low.add(&high.sub(&low, party_id).mul_public(r_i), party_id),
                        )
                            .into();
                        output_index += 1;
                    }
                    if cz_coeff != (None, None) {
                        let (low, high) = (
                            cz_coeff.0.unwrap_or(SharedOrPublic::zero_public()),
                            cz_coeff.1.unwrap_or(SharedOrPublic::zero_public()),
                        );
                        output_slice[output_index] = (
                            3 * block_index + 2,
                            low.add(&high.sub(&low, party_id).mul_public(r_i), party_id),
                        )
                            .into();
                        output_index += 1;
                    }
                }
                debug_assert_eq!(output_index, output_slice.len())
            });
        drop(_span_enter);

        std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
        self.dense_len /= 2;

        Ok(())
    }

    /// Computes the number of non-zero coefficients that would result from
    /// binding the given slice of coefficients.
    fn binding_output_length<T>(coeffs: &[SparseCoefficient<T>]) -> usize {
        let mut output_size = 0;
        for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
            let mut Az_coeff_found = false;
            let mut Bz_coeff_found = false;
            let mut Cz_coeff_found = false;
            for coeff in block {
                match coeff.index % 3 {
                    0 => {
                        if !Az_coeff_found {
                            Az_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    1 => {
                        if !Bz_coeff_found {
                            Bz_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    2 => {
                        if !Cz_coeff_found {
                            Cz_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        output_size
    }

    pub fn final_sumcheck_evals(&self, party_id: PartyID) -> [AdditiveShare<F>; 3] {
        let mut final_az_eval = AdditiveShare::zero();
        let mut final_bz_eval = AdditiveShare::zero();
        let mut final_cz_eval = AdditiveShare::zero();
        for i in 0..3 {
            if let Some(coeff) = self.bound_coeffs.get(i) {
                match coeff.index {
                    0 => final_az_eval = coeff.value.into_additive(party_id),
                    1 => final_bz_eval = coeff.value.into_additive(party_id),
                    2 => final_cz_eval = coeff.value.into_additive(party_id),
                    _ => {}
                }
            }
        }

        [final_az_eval, final_bz_eval, final_cz_eval]
    }
}

pub fn eval_offset_lc_rep3_mixed<F: JoltField>(
    offset: &OffsetLC,
    flattened_polynomials: &[&Rep3MultilinearPolynomial<F>],
    step: usize,
    next_step_m: Option<usize>,
    party_id: PartyID,
) -> SharedOrPublic<F> {
    if !offset.0 {
        offset
            .1
            .evaluate_row_rep3_mixed(flattened_polynomials, step, party_id)
    } else if let Some(next_step) = next_step_m {
        offset
            .1
            .evaluate_row_rep3_mixed(flattened_polynomials, next_step, party_id)
    } else {
        SharedOrPublic::Public(F::from_i128(offset.1.constant_term_field()))
    }
}
