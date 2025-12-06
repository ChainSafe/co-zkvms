use crate::field::JoltField;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::sparse_interleaved_poly::Rep3SparseInterleavedPolynomial;
use crate::poly::split_eq_poly::DistributedSplitEqPolynomial;
use crate::subprotocols::grand_product::{
    Rep3BatchedGrandProduct, Rep3BatchedGrandProductLayer, Rep3BatchedGrandProductLayerWorker,
    Rep3BatchedGrandProductWorker,
};
use crate::subprotocols::sumcheck::{
    Rep3BatchedCubicSumcheck, Rep3BatchedCubicSumcheckWorker, Rep3Bindable,
};
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use itertools::chain;
use jolt_core::poly::sparse_interleaved_poly::SparseCoefficient;
use jolt_core::poly::split_eq_poly::SplitEqPolynomial;
use jolt_core::subprotocols::grand_product::BatchedGrandProductLayerProof;
use jolt_core::subprotocols::sparse_grand_product::BatchedGrandProductToggleLayer;
use jolt_core::subprotocols::sumcheck::{BatchedCubicSumcheck, SumcheckInstanceProof};
use jolt_core::utils::transcript::Transcript;
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};
use rayon::prelude::*;

#[derive(Debug, Default)]
struct Rep3BatchedGrandProductToggleLayer<F: JoltField> {
    /// The list of non-zero flag indices for each circuit in the batch.
    flag_indices: Vec<Vec<usize>>,
    /// The list of non-zero flag values for each circuit in the batch.
    /// Before the first binding iteration of sumcheck, this will be empty
    /// (we know that all non-zero, unbound flag values are 1).
    flag_values: Vec<Vec<F>>,
    /// The Reed-Solomon fingerprints for each circuit in the batch.
    fingerprints: Vec<Vec<Rep3PrimeFieldShare<F>>>,
    /// Once the sparse flag/fingerprint vectors cannot be bound further
    /// (i.e. binding would require processing values in different vectors),
    /// we switch to using `coalesced_flags` to represent the flag values.
    coalesced_flags: Option<Vec<F>>,
    /// Once the sparse flag/fingerprint vectors cannot be bound further
    /// (i.e. binding would require processing values in different vectors),
    /// we switch to using `coalesced_fingerprints` to represent the fingerprint values.
    coalesced_fingerprints: Option<Vec<Rep3PrimeFieldShare<F>>>,
    /// The length of a layer in one of the circuits in the batch.
    layer_len: usize,

    batched_layer_len: usize,
}

impl<F: JoltField> Rep3BatchedGrandProductToggleLayer<F> {
    fn new(flag_indices: Vec<Vec<usize>>, fingerprints: Vec<Vec<Rep3PrimeFieldShare<F>>>) -> Self {
        let layer_len = 2 * fingerprints[0].len();
        let batched_layer_len = fingerprints.len() * layer_len;
        Self {
            flag_indices,
            // While flags remain unbound, all values are boolean, so we can assume any flag that appears in `flag_indices` has value 1.
            flag_values: vec![],
            fingerprints,
            layer_len,
            batched_layer_len,
            coalesced_flags: None,
            coalesced_fingerprints: None,
        }
    }

    /// Computes the grand product layer output by this one.
    #[tracing::instrument(
        skip_all,
        name = "BatchedGrandProductToggleLayer::layer_output",
        level = "trace"
    )]
    fn layer_output(&self, party_id: PartyID) -> Rep3SparseInterleavedPolynomial<F> {
        let values: Vec<Vec<SparseCoefficient<_>>> = self
            .fingerprints
            .iter()
            .enumerate()
            .map(|(batch_index, fingerprints)| {
                let flag_indices = &self.flag_indices[batch_index / 2];
                let mut sparse_coeffs = Vec::with_capacity(self.layer_len);
                for i in flag_indices {
                    sparse_coeffs
                        .push((batch_index * self.layer_len / 2 + i, fingerprints[*i]).into());
                }
                sparse_coeffs
            })
            .collect();

        Rep3SparseInterleavedPolynomial::new(values, self.batched_layer_len / 2, party_id)
    }

    /// Coalesces flags and fingerprints into one (dense) vector each.
    /// After a certain number of bindings, we can no longer process the k
    /// circuits in the batch in independently, at which point we coalesce.
    #[tracing::instrument(
        skip_all,
        name = "BatchedGrandProductToggleLayer::coalesce",
        level = "trace"
    )]
    fn coalesce(&mut self) {
        let mut coalesced_fingerprints: Vec<_> =
            self.fingerprints.iter().map(|f| f[0]).collect::<Vec<_>>();
        coalesced_fingerprints.resize(
            coalesced_fingerprints.len().next_power_of_two(),
            Rep3PrimeFieldShare::zero_share(),
        );

        let mut coalesced_flags: Vec<_> = self
            .flag_indices
            .iter()
            .zip(self.flag_values.iter())
            .flat_map(|(indices, values)| {
                debug_assert!(indices.len() <= 1);
                let mut coalesced = [F::zero(), F::zero()];
                for (index, value) in indices.iter().zip(values.iter()) {
                    assert_eq!(*index, 0);
                    coalesced[0] = *value;
                    coalesced[1] = *value;
                }
                coalesced
            })
            .collect();
        // Fingerprints are padded with 0s, flags are padded with 1s
        coalesced_flags.resize(coalesced_flags.len().next_power_of_two(), F::one());

        self.coalesced_fingerprints = Some(coalesced_fingerprints);
        self.coalesced_flags = Some(coalesced_flags);
    }
}

impl<F: JoltField> Rep3Bindable<F> for Rep3BatchedGrandProductToggleLayer<F> {
    /// Incrementally binds a variable of the flag and fingerprint polynomials.
    /// Similar to `SparseInterleavedPolynomial::bind`, in that flags use
    /// a sparse representation, but different in a couple of key ways:
    /// - flags use two separate vectors (for indices and values) rather than
    ///   a single vector of (index, value) pairs
    /// - The left and right nodes in this layer are flags and fingerprints, respectively.
    ///   They are represented by *separate* vectors, so they are *not* interleaved. This
    ///   means we process 2 flag values at a time, rather than 4.
    /// - In `BatchedSparseGrandProductLayer`, the absence of a node implies that it has
    ///   value 1. For our sparse representation of flags, the absence of a node implies
    ///   that it has value 0. In other words, a flag with value 1 will be present in both
    ///   `self.flag_indices` and `self.flag_values`.
    #[tracing::instrument(
        skip_all,
        name = "BatchedGrandProductToggleLayer::bind",
        level = "trace"
    )]
    fn bind(&mut self, r: F, _party_id: PartyID) {
        if let Some(coalesced_flags) = &mut self.coalesced_flags {
            // Polynomials have already been coalesced, so bind the coalesced vectors.
            let mut bound_flags = vec![F::one(); coalesced_flags.len() / 2];
            for i in 0..bound_flags.len() {
                bound_flags[i] = coalesced_flags[2 * i]
                    + r * (coalesced_flags[2 * i + 1] - coalesced_flags[2 * i]);
            }
            self.coalesced_flags = Some(bound_flags);

            let coalesced_fingerprints = self.coalesced_fingerprints.as_mut().unwrap();
            let mut bound_fingerprints =
                vec![Rep3PrimeFieldShare::zero_share(); coalesced_fingerprints.len() / 2];
            for i in 0..bound_fingerprints.len() {
                bound_fingerprints[i] = rep3::arithmetic::add_mul_public(
                    coalesced_fingerprints[2 * i],
                    coalesced_fingerprints[2 * i + 1] - coalesced_fingerprints[2 * i],
                    r,
                );
            }
            self.coalesced_fingerprints = Some(bound_fingerprints);
            self.batched_layer_len /= 2;

            return;
        }

        debug_assert!(self.layer_len % 4 == 0);

        // Bind the fingerprints
        self.fingerprints
            .par_iter_mut()
            .for_each(|layer: &mut Vec<_>| {
                let n = self.layer_len / 4;
                for i in 0..n {
                    layer[i] = rep3::arithmetic::add_mul_public(
                        layer[2 * i],
                        layer[2 * i + 1] - layer[2 * i],
                        r,
                    );
                }
            });

        let is_first_bind = self.flag_values.is_empty();
        if is_first_bind {
            self.flag_values = vec![vec![]; self.flag_indices.len()];
        }

        // Bind the flags
        self.flag_indices
            .par_iter_mut()
            .zip(self.flag_values.par_iter_mut())
            .for_each(|(flag_indices, flag_values)| {
                let mut next_index_to_process = 0usize;

                let mut bound_index = 0usize;
                for j in 0..flag_indices.len() {
                    let index = flag_indices[j];
                    if index < next_index_to_process {
                        // This flag was already bound with its sibling in the previous iteration.
                        continue;
                    }

                    // Bind indices in place
                    flag_indices[bound_index] = index / 2;

                    if index % 2 == 0 {
                        let neighbor = flag_indices.get(j + 1).cloned().unwrap_or(0);
                        if neighbor == index + 1 {
                            // Neighbor is flag's sibling

                            if is_first_bind {
                                // For first bind, all non-zero flag values are 1.
                                // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                //                = 1 - r * (1 - 1)
                                //                = 1
                                flag_values.push(F::one());
                            } else {
                                // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                flag_values[bound_index] =
                                    flag_values[j] + r * (flag_values[j + 1] - flag_values[j]);
                            };
                        } else {
                            // This flag's sibling wasn't found, so it must have value 0.

                            if is_first_bind {
                                // For first bind, all non-zero flag values are 1.
                                // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                //                = flags[2 * i] - r * flags[2 * i]
                                //                = 1 - r
                                flag_values.push(F::one() - r);
                            } else {
                                // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                //                = flags[2 * i] - r * flags[2 * i]
                                flag_values[bound_index] = flag_values[j] - r * flag_values[j];
                            };
                        }
                        next_index_to_process = index + 2;
                    } else {
                        // This flag's sibling wasn't encountered in a previous iteration,
                        // so it must have had value 0.

                        if is_first_bind {
                            // For first bind, all non-zero flag values are 1.
                            // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                            //                = r * flags[2 * i + 1]
                            //                = r
                            flag_values.push(r);
                        } else {
                            // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                            //                = r * flags[2 * i + 1]
                            flag_values[bound_index] = r * flag_values[j];
                        };
                        next_index_to_process = index + 1;
                    }

                    bound_index += 1;
                }

                flag_indices.truncate(bound_index);
                // We only ever use `flag_indices.len()`, so no need to truncate `flag_values`
                // flag_values.truncate(bound_index);
            });
        self.layer_len /= 2;
        self.batched_layer_len /= 2;

        if self.layer_len == 2 {
            // Time to coalesce
            assert!(self.coalesced_fingerprints.is_none());
            assert!(self.coalesced_flags.is_none());
            self.coalesce();
        }
    }
}

impl<F: JoltField, Network: Rep3NetworkWorker> Rep3BatchedCubicSumcheckWorker<F, Network>
    for Rep3BatchedGrandProductToggleLayer<F>
{
    /// Similar to `SparseInterleavedPolynomial::compute_cubic`, but with changes to
    /// accommodate the differences between `SparseInterleavedPolynomial` and
    /// `BatchedGrandProductToggleLayer`. These differences are described in the doc comments
    /// for `BatchedGrandProductToggleLayer::bind`.
    ///
    /// Since we are using the Dao-Thaler EQ optimization, there are four cases to handle:
    /// 1. Flags/fingerprints are coalesced, and E1 is fully bound
    /// 2. Flags/fingerprints are coalesced, and E1 isn't fully bound
    /// 3. Flags/fingerprints aren't coalesced, and E1 is fully bound
    /// 4. Flags/fingerprints aren't coalesced, and E1 isn't fully bound
    #[tracing::instrument(
        skip_all,
        name = "BatchedGrandProductToggleLayer::compute_cubic",
        level = "trace"
    )]
    fn compute_cubic(
        &self,
        eq_poly: &DistributedSplitEqPolynomial<F>,
        // previous_round_claim: AdditiveShare<F>,
        party_id: PartyID,
    ) -> [AdditiveShare<F>; 3] {
        let E1_len = eq_poly.E1_len;

        if let Some(coalesced_flags) = &self.coalesced_flags {
            let coalesced_fingerprints = self.coalesced_fingerprints.as_ref().unwrap();

            let cubic_evals = if eq_poly.E1_len == 1 {
                // 1. Flags/fingerprints are coalesced, and E1 is fully bound
                // This is similar to the if case of `DenseInterleavedPolynomial::compute_cubic`
                coalesced_flags
                    .par_chunks(2)
                    .zip(coalesced_fingerprints.par_chunks(2))
                    .zip(eq_poly.E2.par_chunks(2))
                    .map(|((flags, fingerprints), eq_chunk)| {
                        let eq_evals = {
                            let eval_point_0 = eq_chunk[0];
                            let m_eq = eq_chunk[1] - eq_chunk[0];
                            let eval_point_2 = eq_chunk[1] + m_eq;
                            let eval_point_3 = eval_point_2 + m_eq;
                            (eval_point_0, eval_point_2, eval_point_3)
                        };
                        let m_flag = flags[1] - flags[0];
                        let m_fingerprint = fingerprints[1] - fingerprints[0];

                        let flag_eval_2 = flags[1] + m_flag;
                        let flag_eval_3 = flag_eval_2 + m_flag;

                        let fingerprint_eval_2 = fingerprints[1] + m_fingerprint;
                        let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                        [
                            additive::add_public(
                                fingerprints[0].into_additive() * flags[0],
                                F::one() - flags[0],
                                party_id,
                            ) * eq_evals.0,
                            additive::add_public(
                                fingerprint_eval_2.into_additive() * flag_eval_2,
                                F::one() - flag_eval_2,
                                party_id,
                            ) * eq_evals.1,
                            additive::add_public(
                                fingerprint_eval_3.into_additive() * flag_eval_3,
                                F::one() - flag_eval_3,
                                party_id,
                            ) * eq_evals.2,
                        ]
                    })
                    .reduce(
                        || [AdditiveShare::<F>::zero(); 3],
                        |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                    )
            } else {
                // 2. Flags/fingerprints are coalesced, and E1 isn't fully bound
                // This is similar to the else case of `DenseInterleavedPolynomial::compute_cubic`
                let E1_evals: Vec<_> = eq_poly.E1[..eq_poly.E1_len]
                    .par_chunks(2)
                    .map(|E1_chunk| {
                        let eval_point_0 = E1_chunk[0];
                        let m_eq = E1_chunk[1] - E1_chunk[0];
                        let eval_point_2 = E1_chunk[1] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        (eval_point_0, eval_point_2, eval_point_3)
                    })
                    .collect();

                let eq_slice_end =
                    eq_poly.global_start + core::cmp::min(eq_poly.len, self.batched_layer_len / 2);
                let E2_local_bound = eq_slice_end
                    .div_ceil(E1_len) // first row index strictly after slice_end
                    .saturating_sub(eq_poly.row_start)
                    .min(eq_poly.E2_len);

                eq_poly.E2[..E2_local_bound]
                    .par_iter()
                    .enumerate()
                    .map(|(x2, E2_eval)| {
                        let row_global = eq_poly.row_start + x2;
                        let row_first = row_global * E1_len;
                        let row_last = row_first + E1_len;
                        let eq_first = eq_poly.global_start.max(row_first);
                        let eq_last = (eq_poly.global_start + eq_poly.len).min(row_last);
                        debug_assert!(eq_last > eq_first);
                        let col_from = eq_first - row_first;
                        let col_to = eq_last - row_first;
                        debug_assert!(
                            col_from % 2 == 0 && col_to % 2 == 0,
                            "misaligned Eq slice within row"
                        );

                        let poly_from = eq_first - eq_poly.global_start;
                        debug_assert!(
                            poly_from < self.batched_layer_len,
                            "coeff_start out of bounds"
                        );

                        let mut inner_sum = [AdditiveShare::<F>::zero(); 3];
                        for ((E1_evals, flag_chunk), fingerprint_chunk) in E1_evals
                            .iter()
                            .zip(coalesced_flags[poly_from..].chunks(2))
                            .zip(coalesced_fingerprints[poly_from..].chunks(2))
                        {
                            let m_flag = flag_chunk[1] - flag_chunk[0];
                            let m_fingerprint = fingerprint_chunk[1] - fingerprint_chunk[0];

                            let flag_eval_2 = flag_chunk[1] + m_flag;
                            let flag_eval_3 = flag_eval_2 + m_flag;

                            let fingerprint_eval_2 = fingerprint_chunk[1] + m_fingerprint;
                            let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                            inner_sum[0] += additive::add_public(
                                fingerprint_chunk[0].into_additive() * flag_chunk[0],
                                F::one() - flag_chunk[0],
                                party_id,
                            ) * E1_evals.0;
                            inner_sum[1] += additive::add_public(
                                fingerprint_eval_2.into_additive() * flag_eval_2,
                                F::one() - flag_eval_2,
                                party_id,
                            ) * E1_evals.1;
                            inner_sum[2] += additive::add_public(
                                fingerprint_eval_3.into_additive() * flag_eval_3,
                                F::one() - flag_eval_3,
                                party_id,
                            ) * E1_evals.2;
                        }

                        inner_sum.map(|inner_sum| inner_sum * *E2_eval)
                    })
                    .reduce(
                        || [AdditiveShare::<F>::zero(); 3],
                        |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                    )
            };

            return cubic_evals;
        }

        if eq_poly.E1_len == 1 {
            // 3. Flags/fingerprints aren't coalesced, and E1 is fully bound
            // This is similar to the if case of `SparseInterleavedPolynomial::compute_cubic`
            let eq_evals: Vec<(F, F, F)> = eq_poly.E2[..eq_poly.E2_len]
                .par_chunks(2)
                .take(self.batched_layer_len / 4)
                .map(|eq_chunk| {
                    let eval_point_0 = eq_chunk[0];
                    let m_eq = eq_chunk[1] - eq_chunk[0];
                    let eval_point_2 = eq_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
            let eq_eval_sums: (F, F, F) = eq_evals
                .par_iter()
                .fold(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                )
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            let deltas: [AdditiveShare<F>; 3] = (0..self.fingerprints.len())
                .into_par_iter()
                .map(|batch_index| {
                    // Computes:
                    //     ∆ := Σ eq_evals[j] * (flag[j] * fingerprint[j] - flag[j])    ∀j where flag[j] ≠ 0
                    // for the evaluation points {0, 2, 3}

                    let fingerprints = &self.fingerprints[batch_index];
                    let flag_indices = &self.flag_indices[batch_index / 2];

                    let unbound = self.flag_values.is_empty();
                    let mut delta = [AdditiveShare::<F>::zero(); 3];

                    let mut next_index_to_process = 0usize;
                    for (j, index) in flag_indices.iter().enumerate() {
                        if *index < next_index_to_process {
                            // This node was already processed in a previous iteration
                            continue;
                        }

                        let (flags, fingerprints) = if index % 2 == 0 {
                            let neighbor = flag_indices.get(j + 1).cloned().unwrap_or(0);
                            let flags = if neighbor == index + 1 {
                                // Neighbor is flag's sibling
                                if unbound {
                                    (F::one(), F::one())
                                } else {
                                    (
                                        self.flag_values[batch_index / 2][j],
                                        self.flag_values[batch_index / 2][j + 1],
                                    )
                                }
                            } else {
                                // This flag's sibling wasn't found, so it must have value 0.
                                if unbound {
                                    (F::one(), F::zero())
                                } else {
                                    (self.flag_values[batch_index / 2][j], F::zero())
                                }
                            };
                            let fingerprints = (fingerprints[*index], fingerprints[index + 1]);

                            next_index_to_process = index + 2;
                            (flags, fingerprints)
                        } else {
                            // This flag's sibling wasn't encountered in a previous iteration,
                            // so it must have had value 0.
                            let flags = if unbound {
                                (F::zero(), F::one())
                            } else {
                                (F::zero(), self.flag_values[batch_index / 2][j])
                            };
                            let fingerprints = (fingerprints[index - 1], fingerprints[*index]);

                            next_index_to_process = index + 1;
                            (flags, fingerprints)
                        };

                        let m_flag = flags.1 - flags.0;
                        let m_fingerprint = fingerprints.1 - fingerprints.0;

                        // If flags are still unbound, flag evals will mostly be 0s and 1s
                        // Bound flags are still mostly 0s, so flag evals will mostly be 0s.
                        let flag_eval_2 = flags.1 + m_flag;
                        let flag_eval_3 = flag_eval_2 + m_flag;

                        let fingerprint_eval_2 = fingerprints.1 + m_fingerprint;
                        let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                        let block_index = (self.layer_len * batch_index) / 4 + index / 2;
                        let eq_evals = eq_evals[block_index];

                        delta[0] += additive::sub_shared_by_public(
                            fingerprints
                                .0
                                .into_additive()
                                .mul_public_01_optimized(flags.0),
                            flags.0,
                            party_id,
                        ) * eq_evals.0;
                        delta[1] += additive::sub_shared_by_public(
                            fingerprint_eval_2
                                .into_additive()
                                .mul_public_01_optimized(flag_eval_2),
                            flag_eval_2,
                            party_id,
                        ) * eq_evals.1;
                        delta[2] += additive::sub_shared_by_public(
                            fingerprint_eval_3
                                .into_additive()
                                .mul_public_01_optimized(flag_eval_3),
                            flag_eval_3,
                            party_id,
                        ) * eq_evals.2;
                    }

                    delta
                })
                .reduce(
                    || [AdditiveShare::zero(); 3],
                    |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                );
            // eq_eval_sum + ∆ = Σ eq_evals[i] + Σ eq_evals[i] * (flag[i] * fingerprint[i] - flag[i]))
            //                 = Σ eq_evals[j] * (flag[i] * fingerprint[i] + 1 - flag[i])
            [
                additive::add_public(deltas[0], eq_eval_sums.0, party_id),
                additive::add_public(deltas[1], eq_eval_sums.1, party_id),
                additive::add_public(deltas[2], eq_eval_sums.2, party_id),
            ]
        } else {
            // 4. Flags/fingerprints aren't coalesced, and E1 isn't fully bound
            // This is similar to the else case of `SparseInterleavedPolynomial::compute_cubic`
            let E1_evals: Vec<_> = eq_poly.E1[..eq_poly.E1_len]
                .par_chunks(2)
                .map(|E1_chunk| {
                    let eval_point_0 = E1_chunk[0];
                    let m_eq = E1_chunk[1] - E1_chunk[0];
                    let eval_point_2 = E1_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            let mut prefix_sums = vec![[F::zero(); 3]; E1_len + 1];
            for (i, e) in E1_evals.iter().enumerate() {
                prefix_sums[i + 1][0] = prefix_sums[i][0] + e.0;
                prefix_sums[i + 1][1] = prefix_sums[i][1] + e.1;
                prefix_sums[i + 1][2] = prefix_sums[i][2] + e.2;
            }

            let eq_slice_start = eq_poly.global_start;
            let eq_slice_end =
                eq_slice_start + core::cmp::min(eq_poly.len, self.batched_layer_len / 2);

            let E2_local_bound = eq_slice_end
                .div_ceil(E1_len)
                .saturating_sub(eq_poly.row_start)
                .min(eq_poly.E2_len);

            let num_x1_bits = eq_poly.E1_len.log_2() - 1;
            let x1_bitmask = (1 << num_x1_bits) - 1;

            let deltas = (0..self.fingerprints.len())
                .into_par_iter()
                .map(|batch_index| {
                    // Computes:
                    //     ∆ := Σ eq_evals[j] * (flag[j] * fingerprint[j] - flag[j])    ∀j where flag[j] ≠ 0
                    // for the evaluation points {0, 2, 3}

                    let fingerprints = &self.fingerprints[batch_index];
                    let flag_indices = &self.flag_indices[batch_index / 2];

                    let unbound = self.flag_values.is_empty();
                    let mut delta = [AdditiveShare::<F>::zero(); 3];
                    let mut inner_sum = [AdditiveShare::<F>::zero(); 3];

                    let mut prev_x2: usize = 0;

                    let mut next_index_to_process = 0usize;
                    for (j, index) in flag_indices.iter().enumerate() {
                        if *index < next_index_to_process {
                            // This node was already processed in a previous iteration
                            continue;
                        }

                        let (flags, fingerprints) = if index % 2 == 0 {
                            let neighbor = flag_indices.get(j + 1).cloned().unwrap_or(0);
                            let flags = if neighbor == index + 1 {
                                // Neighbor is flag's sibling
                                if unbound {
                                    (F::one(), F::one())
                                } else {
                                    (
                                        self.flag_values[batch_index / 2][j],
                                        self.flag_values[batch_index / 2][j + 1],
                                    )
                                }
                            } else {
                                // This flag's sibling wasn't found, so it must have value 0.
                                if unbound {
                                    (F::one(), F::zero())
                                } else {
                                    (self.flag_values[batch_index / 2][j], F::zero())
                                }
                            };
                            let fingerprints = (fingerprints[*index], fingerprints[index + 1]);

                            next_index_to_process = index + 2;
                            (flags, fingerprints)
                        } else {
                            // This flag's sibling wasn't encountered in a previous iteration,
                            // so it must have had value 0.
                            let flags = if unbound {
                                (F::zero(), F::one())
                            } else {
                                (F::zero(), self.flag_values[batch_index / 2][j])
                            };
                            let fingerprints = (fingerprints[index - 1], fingerprints[*index]);

                            next_index_to_process = index + 1;
                            (flags, fingerprints)
                        };

                        let m_flag = flags.1 - flags.0;
                        let m_fingerprint = fingerprints.1 - fingerprints.0;

                        // If flags are still unbound, flag evals will mostly be 0s and 1s
                        // Bound flags are still mostly 0s, so flag evals will mostly be 0s.
                        let flag_eval_2 = flags.1 + m_flag;
                        let flag_eval_3 = flag_eval_2 + m_flag;

                        let fingerprint_eval_2 = fingerprints.1 + m_fingerprint;
                        let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                        let block_index = (self.layer_len * batch_index) / 4 + index / 2;
                        let x2 = block_index >> num_x1_bits;
                        if x2 != prev_x2 {
                            delta[0] += inner_sum[0] * eq_poly.E2[prev_x2];
                            delta[1] += inner_sum[1] * eq_poly.E2[prev_x2];
                            delta[2] += inner_sum[2] * eq_poly.E2[prev_x2];
                            inner_sum = [AdditiveShare::<F>::zero(); 3];
                            prev_x2 = x2;
                        }

                        let x1 = block_index & x1_bitmask;

                        inner_sum[0] += additive::sub_shared_by_public(
                            fingerprints
                                .0
                                .into_additive()
                                .mul_public_01_optimized(flags.0),
                            flags.0,
                            party_id,
                        ) * E1_evals[x1].0;
                        inner_sum[1] += additive::sub_shared_by_public(
                            fingerprint_eval_2
                                .into_additive()
                                .mul_public_01_optimized(flag_eval_2),
                            flag_eval_2,
                            party_id,
                        ) * E1_evals[x1].1;
                        inner_sum[2] += additive::sub_shared_by_public(
                            fingerprint_eval_3
                                .into_additive()
                                .mul_public_01_optimized(flag_eval_3),
                            flag_eval_3,
                            party_id,
                        ) * E1_evals[x1].2;
                    }

                    delta[0] += inner_sum[0] * eq_poly.E2[prev_x2];
                    delta[1] += inner_sum[1] * eq_poly.E2[prev_x2];
                    delta[2] += inner_sum[2] * eq_poly.E2[prev_x2];

                    delta
                })
                .reduce(
                    || [AdditiveShare::<F>::zero(); 3],
                    |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                );

            // The cubic evals assuming all the coefficients are ones is affected by the
            // `batched_layer_len`, since we implicitly pad the `batched_layer_len` to a power of 2.
            // By pad here we mean that flags are padded with 1s, and fingerprints are
            // padded with 0s.
            // Optimized baseline assuming all P == 1 on the active part of this worker's slice.
            let evals_assuming_all_ones: [F; 3] = eq_poly.E2[..E2_local_bound]
                .par_iter()
                .enumerate()
                .map(|(E2_i, E2_eval)| {
                    let row_global = eq_poly.row_start + E2_i;
                    let row_first = row_global * E1_len;
                    let row_last = row_first + E1_len;

                    // Intersection with this worker’s slice [slice_start, slice_end).
                    let eq_first = eq_slice_start.max(row_first);
                    let eq_last = eq_slice_end.min(row_last);
                    assert!(eq_first < eq_last);

                    // Column offsets inside the row (in Eq points).
                    let col_from = eq_first - row_first;
                    let col_to = eq_last - row_first;

                    // Each Dao–Thaler E1 entry spans 2 Eq points; enforce alignment.
                    debug_assert!(
                        col_from % 2 == 0 && col_to % 2 == 0,
                        "misaligned Eq slice within row"
                    );

                    // Local offset in the dense polynomial (each Eq point → 2 coeffs).
                    let poly_from = (eq_first - eq_poly.global_start) * 2;
                    debug_assert!(poly_from < self.batched_layer_len);
                    let poly_bound = (self.batched_layer_len - poly_from) / 4;

                    // Range of C-indices (pairs) in this row that belong to this worker.
                    let E1_from = col_from / 2;
                    let E1_to = (col_to / 2).min(poly_bound);
                    debug_assert!(E1_from < E1_to);

                    let s0 = prefix_sums[E1_to][0] - prefix_sums[E1_from][0];
                    let s1 = prefix_sums[E1_to][1] - prefix_sums[E1_from][1];
                    let s2 = prefix_sums[E1_to][2] - prefix_sums[E1_from][2];

                    [*E2_eval * s0, *E2_eval * s1, *E2_eval * s2]
                })
                .reduce(
                    || [F::zero(); 3],
                    |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                );

            [
                additive::add_public(deltas[0], evals_assuming_all_ones[0], party_id),
                additive::add_public(deltas[1], evals_assuming_all_ones[1], party_id),
                additive::add_public(deltas[2], evals_assuming_all_ones[2], party_id),
            ]
        }
    }

    fn final_evals(&self, worker_len: usize, party_id: PartyID) -> Vec<AdditiveShare<F>> {
        let flags = self.coalesced_flags.as_ref().unwrap();
        let fingerprints = self.coalesced_fingerprints.as_ref().unwrap();
        if flags.len() == 1 {
            vec![
                additive::promote_to_trivial_share(flags[0], party_id),
                fingerprints[0].into_additive(),
            ]
        } else {
            chain!(
                self.coalesced_flags
                    .as_ref()
                    .unwrap()
                    .iter()
                    .take(worker_len)
                    .map(|e| additive::promote_to_trivial_share(*e, party_id)),
                self.coalesced_fingerprints
                    .as_ref()
                    .unwrap()
                    .iter()
                    .take(worker_len)
                    .map(|e| e.into_additive())
            )
            .collect()
        }
    }
}

impl<F: JoltField, ProofTranscript, Network> Rep3BatchedCubicSumcheck<F, ProofTranscript, Network>
    for Rep3BatchedGrandProductToggleLayer<F>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn prove_remaining_rounds(
        &self,
        r_grand_product: &[F],
        r: &mut Vec<F>,
        claim: F,
        proof: &mut SumcheckInstanceProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(F, F)> {
        let mut eq_poly = SplitEqPolynomial::new_bind(r_grand_product, r);

        let (mut coalesced_flags, mut coalesced_fingerprints) = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
            .into_iter()
            .map(|shares| {
                let mut flags = additive::combine_additive_vec(shares);
                let fingerprints = flags.split_off(flags.len() / 2);

                (flags, fingerprints)
            })
            .fold(
                (vec![], vec![]),
                |(mut flags, mut fingerprints), (final_l, final_r)| {
                    flags.extend(final_l);
                    fingerprints.extend(final_r);
                    (flags, fingerprints)
                },
            );

        tracing::info!(
            "eq_poly.len: {} coalesced_flags: {} coalesced_fingerprints: {}",
            eq_poly.len(),
            coalesced_flags.len(),
            coalesced_fingerprints.len()
        );

        coalesced_flags.resize(eq_poly.len(), F::one());
        coalesced_fingerprints.resize(eq_poly.len(), F::zero());

        let mut layer = BatchedGrandProductToggleLayer {
            coalesced_flags: Some(coalesced_flags),
            coalesced_fingerprints: Some(coalesced_fingerprints),
            layer_len: 2,
            ..Default::default()
        };

        let (remaining_proof, remaining_r, final_claims) =
            layer.prove_sumcheck(&claim, &mut eq_poly, transcript);

        network.broadcast_request(remaining_r.clone())?;
        proof
            .compressed_polys
            .extend(remaining_proof.compressed_polys);
        r.extend(remaining_r);

        Ok(final_claims)
    }
}

impl<F: JoltField, Network: Rep3NetworkWorker> Rep3BatchedGrandProductLayerWorker<F, Network>
    for Rep3BatchedGrandProductToggleLayer<F>
{
    #[tracing::instrument(
        skip_all,
        name = "Rep3BatchedGrandProductToggleLayer::prove_layer",
        level = "trace"
    )]
    fn prove_layer(
        &mut self,
        r_grand_product: &mut Vec<F>,
        eq_chunk_size: usize,
        worker_symmetric: bool,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let mut eq_poly = DistributedSplitEqPolynomial::new(
            r_grand_product,
            io_ctx.log_num_workers(),
            io_ctx.worker_idx(),
            eq_chunk_size,
        );

        let r_sumcheck = self.prove_sumcheck(&mut eq_poly, worker_symmetric, io_ctx)?;

        drop_in_background_thread(eq_poly);

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_grand_product);

        Ok(())
    }
}

impl<F: JoltField, ProofTranscript, Network>
    Rep3BatchedGrandProductLayer<F, ProofTranscript, Network>
    for Rep3BatchedGrandProductToggleLayer<F>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn coordinate_prove_layer(
        &self,
        previous_claim: &mut F,
        r_grand_product: &mut Vec<F>,
        worker_symmetric: bool,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<BatchedGrandProductLayerProof<F, ProofTranscript>> {
        let num_rounds = r_grand_product.len();

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) = self.coordinate_prove_sumcheck(
            previous_claim,
            r_grand_product,
            num_rounds,
            worker_symmetric,
            transcript,
            network,
        )?;

        let (left_claim, right_claim) = sumcheck_claims;
        transcript.append_scalar(&left_claim);
        transcript.append_scalar(&right_claim);

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_grand_product);

        Ok(BatchedGrandProductLayerProof {
            proof: sumcheck_proof,
            left_claim,
            right_claim,
        })
    }
}

pub struct Rep3ToggledBatchedGrandProduct<F: JoltField> {
    batch_size_minus_delta: usize,
    toggle_layer: Rep3BatchedGrandProductToggleLayer<F>,
    sparse_layers: Vec<Rep3SparseInterleavedPolynomial<F>>,
    is_worker_symmetric: bool,
    // quark_poly: Option<Vec<F>>,
}

impl<F, PCS, ProofTranscript, Network>
    Rep3BatchedGrandProductWorker<F, PCS, ProofTranscript, Network>
    for Rep3ToggledBatchedGrandProduct<F>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    type Leaves = (Vec<Vec<usize>>, Vec<Vec<Rep3PrimeFieldShare<F>>>, usize); // (flags, fingerprints)

    #[tracing::instrument(skip_all, name = "ToggledBatchedGrandProduct::construct")]
    fn construct(leaves: Self::Leaves, io_ctx: &mut IoContextPool<Network>) -> eyre::Result<Self> {
        let (flags, fingerprints, batch_size_full) = leaves;
        let batch_size = fingerprints.len();
        let tree_depth = fingerprints[0].len().log_2();

        let batch_size_minus_delta =
            if io_ctx.log_num_workers() > 0 && io_ctx.worker_idx() == io_ctx.num_workers() - 1 {
                (batch_size_full - batch_size) / (io_ctx.num_workers() - 1)
            } else {
                batch_size
            };

        let num_sparse_layers = tree_depth - 1;

        let toggle_layer = Rep3BatchedGrandProductToggleLayer::new(flags, fingerprints);
        let mut sparse_layers: Vec<_> = Vec::with_capacity(1 + num_sparse_layers);
        sparse_layers.push(toggle_layer.layer_output(io_ctx.party_id()));

        // let mut dense_len = toggle_layer.dense_len;
        for i in 0..num_sparse_layers {
            let previous_layer = &sparse_layers[i];
            sparse_layers.push(previous_layer.layer_output(io_ctx)?);
        }

        Ok(Self {
            batch_size_minus_delta,
            toggle_layer,
            sparse_layers,
            is_worker_symmetric: batch_size_full.is_power_of_two(),
        })
    }

    fn num_layers(&self) -> usize {
        self.sparse_layers.len() + 1
    }

    fn claimed_outputs(&self) -> Vec<AdditiveShare<F>> {
        // If there's a quark poly, then that's the claimed output
        let last_layer = self.sparse_layers.last().unwrap();
        let (left, right) = last_layer.uninterleave();
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| *l * *r)
            .collect()
    }

    fn layers(
        &'_ mut self,
    ) -> impl Iterator<Item = &'_ mut dyn Rep3BatchedGrandProductLayerWorker<F, Network>> {
        [&mut self.toggle_layer as &mut dyn Rep3BatchedGrandProductLayerWorker<F, Network>]
            .into_iter()
            .chain(
                self.sparse_layers
                    .iter_mut()
                    .map(|layer| layer as &mut dyn Rep3BatchedGrandProductLayerWorker<F, Network>),
            )
            .rev()
    }

    fn batch_size_minus_delta(&self) -> usize {
        self.batch_size_minus_delta
    }

    fn is_worker_symmetric(&self) -> bool {
        self.is_worker_symmetric
    }
}

impl<F: JoltField, PCS, ProofTranscript, Network>
    Rep3BatchedGrandProduct<F, PCS, ProofTranscript, Network> for Rep3ToggledBatchedGrandProduct<F>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn construct(num_layers: usize, batch_size: usize) -> Self {
        let sparse_layers = num_layers - 1;
        Self {
            batch_size_minus_delta: 0,
            toggle_layer: Rep3BatchedGrandProductToggleLayer::default(),
            sparse_layers: vec![Rep3SparseInterleavedPolynomial::default(); sparse_layers],
            is_worker_symmetric: batch_size.is_power_of_two(),
        }
    }

    fn num_layers(&self) -> usize {
        self.sparse_layers.len() + 1
    }

    fn is_worker_symmetric(&self) -> bool {
        self.is_worker_symmetric
    }

    fn layers(
        &'_ self,
    ) -> impl Iterator<Item = &'_ dyn Rep3BatchedGrandProductLayer<F, ProofTranscript, Network>>
    {
        [&self.toggle_layer as &dyn Rep3BatchedGrandProductLayer<F, ProofTranscript, Network>]
            .into_iter()
            .chain(self.sparse_layers.iter().map(|layer| {
                layer as &dyn Rep3BatchedGrandProductLayer<F, ProofTranscript, Network>
            }))
            .rev()
    }
}
