use crate::field::{JoltField, OptimizedMul};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
#[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::opening_proof::{
    ProverOpeningAccumulator, Rep3ProverOpeningAccumulator, VerifierOpeningAccumulator,
};
use crate::poly::sparse_interleaved_poly::Rep3SparseInterleavedPolynomial;
use crate::poly::Rep3DensePolynomial;
use crate::subprotocols::grand_product::{
    Rep3BatchedGrandProduct, Rep3BatchedGrandProductLayer, Rep3BatchedGrandProductLayerWorker,
    Rep3BatchedGrandProductWorker,
};
use crate::subprotocols::sumcheck::{
    Rep3BatchedCubicSumcheck, Rep3BatchedCubicSumcheckWorker, Rep3Bindable,
};
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::Transcript;
use itertools::Itertools;
use jolt_core::poly::sparse_interleaved_poly::SparseInterleavedPolynomial;
use jolt_core::poly::split_eq_poly::SplitEqPolynomial;
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::grand_product::{
    BatchedGrandProduct, BatchedGrandProductLayer, BatchedGrandProductLayerProof,
    BatchedGrandProductProof,
};
use jolt_core::subprotocols::grand_product_quarks::QuarkGrandProductBase;
use jolt_core::subprotocols::sumcheck::{BatchedCubicSumcheck, Bindable};
use jolt_core::subprotocols::QuarkHybridLayerDepth;
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::{IoContext, Rep3NetworkCoordinator, Rep3NetworkWorker};
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
        let values: Vec<_> = self
            .fingerprints
            .par_iter()
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
    fn bind(&mut self, r: F, party_id: PartyID) {
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
        eq_poly: &SplitEqPolynomial<F>,
        previous_round_claim: F,
        party_id: PartyID,
    ) -> UniPoly<F> {
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

                        let t0 = flags[0].mul_0_optimized(eq_evals.0);
                        let t1 = flag_eval_2.mul_0_optimized(eq_evals.1);
                        let t2 = flag_eval_3.mul_0_optimized(eq_evals.2);

                        let e0 = additive::add_public(
                            rep3::arithmetic::mul_public(fingerprints[0], t0).into_additive(),
                            eq_evals.0 - t0,
                            party_id,
                        );

                        let e1 = additive::add_public(
                            rep3::arithmetic::mul_public(fingerprint_eval_2, t1).into_additive(),
                            eq_evals.1 - t1,
                            party_id,
                        );

                        let e2 = additive::add_public(
                            rep3::arithmetic::mul_public(fingerprint_eval_3, t2).into_additive(),
                            eq_evals.2 - t2,
                            party_id,
                        );

                        // let e0 = rep3::arithmetic::mul_public(
                        //     rep3::arithmetic::add_public(
                        //         rep3::arithmetic::mul_public(fingerprints[0], flags[0]),
                        //         F::one() - flags[0],
                        //         party_id,
                        //     ),
                        //     eq_evals.0,
                        // )
                        // .into_additive();

                        // let e0 = rep3::arithmetic::mul_public(
                        //     rep3::arithmetic::add_public(
                        //         rep3::arithmetic::mul_public(fingerprints[0], flags[0]),
                        //         F::one() - flags[0],
                        //         party_id,
                        //     ),
                        //     eq_evals.0,
                        // )
                        // .into_additive();

                        // let e1 = rep3::arithmetic::mul_public(
                        //     rep3::arithmetic::add_public(
                        //         rep3::arithmetic::mul_public(fingerprint_eval_2, flag_eval_2),
                        //         F::one() - flag_eval_2,
                        //         party_id,
                        //     ),
                        //     eq_evals.1,
                        // )
                        // .into_additive();

                        // let e2 = rep3::arithmetic::mul_public(
                        //     rep3::arithmetic::add_public(
                        //         rep3::arithmetic::mul_public(fingerprint_eval_3, flag_eval_3),
                        //         F::one() - flag_eval_3,
                        //         party_id,
                        //     ),
                        //     eq_evals.2,
                        // )
                        // .into_additive();

                        (e0, e1, e2)
                    })
                    .reduce(
                        || (F::zero(), F::zero(), F::zero()),
                        |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
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

                let flag_chunk_size = coalesced_flags.len().next_power_of_two() / eq_poly.E2_len;
                let fingerprint_chunk_size =
                    coalesced_fingerprints.len().next_power_of_two() / eq_poly.E2_len;

                eq_poly.E2[..eq_poly.E2_len]
                    .par_iter()
                    .zip(coalesced_flags.par_chunks(flag_chunk_size))
                    .zip(coalesced_fingerprints.par_chunks(fingerprint_chunk_size))
                    .map(|((E2_eval, flag_x2), fingerprint_x2)| {
                        let mut inner_sum = (F::zero(), F::zero(), F::zero());
                        for ((E1_evals, flag_chunk), fingerprint_chunk) in E1_evals
                            .iter()
                            .zip(flag_x2.chunks(2))
                            .zip(fingerprint_x2.chunks(2))
                        {
                            let m_flag = flag_chunk[1] - flag_chunk[0];
                            let m_fingerprint = fingerprint_chunk[1] - fingerprint_chunk[0];

                            let flag_eval_2 = flag_chunk[1] + m_flag;
                            let flag_eval_3 = flag_eval_2 + m_flag;

                            let fingerprint_eval_2 = fingerprint_chunk[1] + m_fingerprint;
                            let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                            let t0_0 = E1_evals.0 * *E2_eval;
                            let t0_1 = E1_evals.1 * *E2_eval;
                            let t0_2 = E1_evals.2 * *E2_eval;

                            let t0 = flag_chunk[0].mul_0_optimized(t0_0);
                            let t1 = flag_eval_2.mul_0_optimized(t0_1);
                            let t2 = flag_eval_3.mul_0_optimized(t0_2);

                            inner_sum.0 += additive::add_public(
                                rep3::arithmetic::mul_public(fingerprint_chunk[0], t0)
                                    .into_additive(),
                                t0_0 - t0,
                                party_id,
                            );

                            inner_sum.1 += additive::add_public(
                                rep3::arithmetic::mul_public(fingerprint_eval_2, t1)
                                    .into_additive(),
                                t0_1 - t1,
                                party_id,
                            );

                            inner_sum.2 += additive::add_public(
                                rep3::arithmetic::mul_public(fingerprint_eval_3, t2)
                                    .into_additive(),
                                t0_2 - t2,
                                party_id,
                            );
                        }

                        (inner_sum.0, inner_sum.1, inner_sum.2)
                    })
                    .reduce(
                        || (F::zero(), F::zero(), F::zero()),
                        |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                    )
            };

            let cubic_evals = [
                cubic_evals.0,
                previous_round_claim - cubic_evals.0,
                cubic_evals.1,
                cubic_evals.2,
            ];
            return UniPoly::from_evals(&cubic_evals);
        }

        let cubic_evals = if eq_poly.E1_len == 1 {
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

            let deltas: (F, F, F) = (0..self.fingerprints.len())
                .into_par_iter()
                .map(|batch_index| {
                    // Computes:
                    //     ∆ := Σ eq_evals[j] * (flag[j] * fingerprint[j] - flag[j])    ∀j where flag[j] ≠ 0
                    // for the evaluation points {0, 2, 3}

                    let fingerprints = &self.fingerprints[batch_index];
                    let flag_indices = &self.flag_indices[batch_index / 2];

                    let unbound = self.flag_values.is_empty();
                    let mut delta = (F::zero(), F::zero(), F::zero());

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

                        let t0 = flags.0.mul_0_optimized(eq_evals.0);
                        let t1 = flag_eval_2.mul_01_optimized(eq_evals.1);
                        let t2 = flag_eval_3.mul_01_optimized(eq_evals.2);

                        delta.0 += additive::sub_shared_by_public(
                            rep3::arithmetic::mul_public(fingerprints.0, t0).into_additive(),
                            t0,
                            party_id,
                        );

                        // delta.1 += eq_evals.1.mul_0_optimized(
                        //     flag_eval_2.mul_01_optimized(fingerprint_eval_2) - flag_eval_2,
                        // );

                        delta.1 += additive::sub_shared_by_public(
                            rep3::arithmetic::mul_public(fingerprint_eval_2, t1).into_additive(),
                            t1,
                            party_id,
                        );

                        // delta.2 += eq_evals.2.mul_0_optimized(
                        //     flag_eval_3.mul_01_optimized(fingerprint_eval_3) - flag_eval_3,
                        // );
                        delta.2 += additive::sub_shared_by_public(
                            rep3::arithmetic::mul_public(fingerprint_eval_3, t2).into_additive(),
                            t2,
                            party_id,
                        );
                    }

                    (delta.0, delta.1, delta.2)
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );
            // eq_eval_sum + ∆ = Σ eq_evals[i] + Σ eq_evals[i] * (flag[i] * fingerprint[i] - flag[i]))
            //                 = Σ eq_evals[j] * (flag[i] * fingerprint[i] + 1 - flag[i])
            (
                additive::add_public(deltas.0, eq_eval_sums.0, party_id),
                additive::add_public(deltas.1, eq_eval_sums.1, party_id),
                additive::add_public(deltas.2, eq_eval_sums.2, party_id),
            )
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
            let E1_eval_sums: (F, F, F) = E1_evals
                .par_iter()
                .fold(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                )
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

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
                    // let mut delta = (F::zero(), F::zero(), F::zero());
                    let mut delta = (F::zero(), F::zero(), F::zero());

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

                        let x1 = block_index & x1_bitmask;

                        let t0 = flags.0.mul_0_optimized(E1_evals[x1].0 * eq_poly.E2[x2]);
                        let t1 = flag_eval_2.mul_0_optimized(E1_evals[x1].1 * eq_poly.E2[x2]);
                        let t2 = flag_eval_3.mul_0_optimized(E1_evals[x1].2 * eq_poly.E2[x2]);

                        delta.0 += additive::sub_shared_by_public(
                            rep3::arithmetic::mul_public(fingerprints.0, t0).into_additive(),
                            t0,
                            party_id,
                        );

                        delta.1 += additive::sub_shared_by_public(
                            rep3::arithmetic::mul_public(fingerprint_eval_2, t1).into_additive(),
                            t1,
                            party_id,
                        );

                        delta.2 += additive::sub_shared_by_public(
                            rep3::arithmetic::mul_public(fingerprint_eval_3, t2).into_additive(),
                            t2,
                            party_id,
                        );
                    }

                    delta
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            // The cubic evals assuming all the coefficients are ones is affected by the
            // `batched_layer_len`, since we implicitly pad the `batched_layer_len` to a power of 2.
            // By pad here we mean that flags are padded with 1s, and fingerprints are
            // padded with 0s.
            //
            // As a refresher, the cubic evals we're computing are:
            //
            // \sum_x2 E2[x2] * (\sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)))
            let evals_assuming_all_ones = if self.batched_layer_len.is_power_of_two() {
                // If `batched_layer_len` is a power of 2, there is no 0-padding.
                //
                // So we have:
                // \sum_x2 (E2[x2] * (\sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * 1))
                //   = \sum_x2 (E2[x2] * \sum_x1 E1_evals[x1])
                //   = (\sum_x2 E2[x2]) * (\sum_x1 E1_evals[x1])
                //   = 1 * E1_eval_sums
                E1_eval_sums
            } else {
                let chunk_size = self.batched_layer_len.next_power_of_two() / eq_poly.E2_len;
                let num_all_one_chunks = self.batched_layer_len / chunk_size;
                let E2_sum: F = eq_poly.E2[..num_all_one_chunks].iter().sum();
                if self.batched_layer_len % chunk_size == 0 {
                    // If `batched_layer_len` isn't a power of 2 but evenly divides `chunk_size`,
                    // that means that for the last values of x2, we have:
                    //   (1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)) = 0
                    // due to the 0-padding.
                    //
                    // This makes the entire inner sum 0 for those values of x2.
                    // So we can simply sum over E2 for the _other_ values of x2, and
                    // multiply by `E1_eval_sums`.
                    (
                        E2_sum * E1_eval_sums.0,
                        E2_sum * E1_eval_sums.1,
                        E2_sum * E1_eval_sums.2,
                    )
                } else {
                    // If `batched_layer_len` isn't a power of 2 and doesn't divide `chunk_size`,
                    // the last nonzero "chunk" will have (self.dense_len % chunk_size) ones,
                    // followed by (chunk_size - self.dense_len % chunk_size) zeros,
                    // e.g. 1 1 1 1 1 1 1 1 0 0 0 0
                    //
                    // This handles this last chunk:
                    let last_chunk_evals = E1_evals[..(self.batched_layer_len % chunk_size) / 4]
                        .par_iter()
                        .fold(
                            || (F::zero(), F::zero(), F::zero()),
                            |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                        )
                        .reduce(
                            || (F::zero(), F::zero(), F::zero()),
                            |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                        );
                    (
                        E2_sum * E1_eval_sums.0
                            + eq_poly.E2[num_all_one_chunks] * last_chunk_evals.0,
                        E2_sum * E1_eval_sums.1
                            + eq_poly.E2[num_all_one_chunks] * last_chunk_evals.1,
                        E2_sum * E1_eval_sums.2
                            + eq_poly.E2[num_all_one_chunks] * last_chunk_evals.2,
                    )
                }
            };

            (
                additive::add_public(deltas.0, evals_assuming_all_ones.0, party_id),
                additive::add_public(deltas.1, evals_assuming_all_ones.1, party_id),
                additive::add_public(deltas.2, evals_assuming_all_ones.2, party_id),
            )
        };

        let cubic_evals = [
            cubic_evals.0,
            previous_round_claim - cubic_evals.0,
            cubic_evals.1,
            cubic_evals.2,
        ];

        UniPoly::from_evals(&cubic_evals)
    }

    fn final_claims(&self, party_id: PartyID) -> (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>) {
        assert_eq!(self.layer_len, 2);
        let flags = self.coalesced_flags.as_ref().unwrap();
        let fingerprints = self.coalesced_fingerprints.as_ref().unwrap();

        (
            rep3::arithmetic::promote_to_trivial_share(party_id, flags[0]),
            fingerprints[0],
        )
    }
}

impl<F: JoltField, ProofTranscript, Network> Rep3BatchedCubicSumcheck<F, ProofTranscript, Network>
    for Rep3BatchedGrandProductToggleLayer<F>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
}

impl<F: JoltField, Network: Rep3NetworkWorker> Rep3BatchedGrandProductLayerWorker<F, Network>
    for Rep3BatchedGrandProductToggleLayer<F>
{
    fn prove_layer(
        &mut self,
        claim: &mut AdditiveShare<F>,
        r_grand_product: &mut Vec<F>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        let mut eq_poly = SplitEqPolynomial::new(r_grand_product);

        if io_ctx.network.get_id() == rep3::PartyID::ID0 {
            io_ctx.network.send_response(eq_poly.get_num_vars())?;
        }

        let (r_sumcheck, _) = self.prove_sumcheck(claim, &mut eq_poly, io_ctx)?;

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
        _claim: &mut F,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<BatchedGrandProductLayerProof<F, ProofTranscript>> {
        let num_rounds = network.receive_response::<usize>(rep3::PartyID::ID0, 0, 0)?;

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.coordinate_prove_sumcheck(num_rounds, transcript, network)?;

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
    batch_size: usize,
    toggle_layer: Rep3BatchedGrandProductToggleLayer<F>,
    sparse_layers: Vec<Rep3SparseInterleavedPolynomial<F>>,
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
    type Leaves = (Vec<Vec<usize>>, Vec<Vec<Rep3PrimeFieldShare<F>>>); // (flags, fingerprints)

    #[tracing::instrument(skip_all, name = "ToggledBatchedGrandProduct::construct")]
    fn construct(leaves: Self::Leaves, io_ctx: &mut IoContext<Network>) -> eyre::Result<Self> {
        let (flags, fingerprints) = leaves;
        let batch_size = fingerprints.len();
        let tree_depth = fingerprints[0].len().log_2();

        let num_sparse_layers = tree_depth - 1;

        let toggle_layer = Rep3BatchedGrandProductToggleLayer::new(flags, fingerprints);
        let mut sparse_layers: Vec<_> = Vec::with_capacity(1 + num_sparse_layers);
        sparse_layers.push(toggle_layer.layer_output(io_ctx.id));

        for i in 0..num_sparse_layers {
            let previous_layer = &sparse_layers[i];
            sparse_layers.push(previous_layer.layer_output(io_ctx)?);
        }

        Ok(Self {
            batch_size,
            toggle_layer,
            sparse_layers,
        })
    }

    fn num_layers(&self) -> usize {
        self.sparse_layers.len() + 1
    }

    fn claimed_outputs(&self) -> Vec<F> {
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
}

impl<F: JoltField, PCS, ProofTranscript, Network>
    Rep3BatchedGrandProduct<F, PCS, ProofTranscript, Network> for Rep3ToggledBatchedGrandProduct<F>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn construct(num_layers: usize) -> Self {
        let sparse_layers = num_layers - 1;
        Self {
            batch_size: 1,
            toggle_layer: Rep3BatchedGrandProductToggleLayer::default(),
            sparse_layers: vec![Rep3SparseInterleavedPolynomial::default(); sparse_layers],
        }
    }

    fn num_layers(&self) -> usize {
        self.sparse_layers.len() + 1
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
