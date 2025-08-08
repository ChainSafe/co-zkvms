use super::dense_interleaved_poly::Rep3DenseInterleavedPolynomial;
use crate::poly::Rep3DensePolynomial;
use crate::subprotocols::grand_product::Rep3BatchedGrandProductLayerWorker;
use crate::subprotocols::sumcheck::{Rep3BatchedCubicSumcheckWorker, Rep3Bindable};
use crate::subprotocols::{
    grand_product::Rep3BatchedGrandProductLayer, sumcheck::Rep3BatchedCubicSumcheck,
};
use crate::utils::future::{FutureExt, FutureVal};

use eyre::Context;
use jolt_core::poly::{
    sparse_interleaved_poly::SparseCoefficient, split_eq_poly::SplitEqPolynomial, unipoly::UniPoly,
};
use jolt_core::{
    field::JoltField,
    subprotocols::sumcheck::Bindable,
    utils::{math::Math, transcript::Transcript},
};
use mpc_core::protocols::additive;
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};
use rayon::prelude::*;

/// Represents a single layer of a sparse grand product circuit.
#[derive(Default, Debug, Clone)]
pub struct Rep3SparseInterleavedPolynomial<F: JoltField> {
    /// A vector of sparse vectors representing the coefficients in a batched grand product
    /// layer, where batch size = coeffs.len().
    pub(crate) coeffs: Vec<Vec<SparseCoefficient<Rep3PrimeFieldShare<F>>>>,
    /// Once `coeffs` cannot be bound further (i.e. binding would require processing values
    /// in different vectors), we switch to using `coalesced` to represent the grand product
    /// layer. See `SparseInterleavedPolynomial::coalesce()`.
    pub(crate) coalesced: Option<Rep3DenseInterleavedPolynomial<F>>,
    /// The length of the layer if it were represented by a single dense vector.
    pub(crate) dense_len: usize,

    pub(crate) one: Rep3PrimeFieldShare<F>,
}

impl<F: JoltField> Rep3SparseInterleavedPolynomial<F> {
    pub fn new(
        coeffs: Vec<Vec<SparseCoefficient<Rep3PrimeFieldShare<F>>>>,
        dense_len: usize,
        party_id: PartyID,
    ) -> Self {
        let batch_size = coeffs.len();
        assert!((dense_len / batch_size).is_power_of_two());
        let one: Rep3PrimeFieldShare<F> =
            rep3::arithmetic::promote_to_trivial_share(party_id, F::one());
        if (dense_len / batch_size) <= 2 {
            // Coalesce
            let mut coalesced = vec![one; dense_len];
            coeffs
                .iter()
                .flatten()
                .for_each(|sparse_coeff| coalesced[sparse_coeff.index] = sparse_coeff.value);
            Self {
                dense_len,
                coeffs: vec![vec![]; batch_size],
                coalesced: Some(Rep3DenseInterleavedPolynomial::new(coalesced)),
                one,
            }
        } else {
            Self {
                dense_len,
                coeffs,
                coalesced: None,
                one,
            }
        }
    }

    pub fn batch_size(&self) -> usize {
        self.coeffs.len()
    }

    /// Converts a `SparseInterleavedPolynomial` into the equivalent `DensePolynomial`.
    pub fn to_dense(&self) -> Rep3DensePolynomial<F> {
        Rep3DensePolynomial::new_padded(self.coalesce())
    }

    #[tracing::instrument(
        skip_all,
        name = "SparseInterleavedPolynomial::coalesce",
        level = "trace"
    )]
    /// Coalesces a `SparseInterleavedPolynomial` into a `DenseInterleavedPolynomial`.
    pub fn coalesce(&self) -> Vec<Rep3PrimeFieldShare<F>> {
        if let Some(coalesced) = &self.coalesced {
            coalesced.coeffs[..coalesced.len()].to_vec()
        } else {
            let mut coalesced = vec![self.one; self.dense_len];
            self.coeffs
                .iter()
                .flatten()
                .for_each(|sparse_coeff| coalesced[sparse_coeff.index] = sparse_coeff.value);
            coalesced
        }
    }

    /// Uninterleaves a `SparseInterleavedPolynomial` into two vectors
    /// containing the left and right coefficients.
    pub fn uninterleave(&self) -> (Vec<Rep3PrimeFieldShare<F>>, Vec<Rep3PrimeFieldShare<F>>) {
        if let Some(coalesced) = &self.coalesced {
            coalesced.uninterleave()
        } else {
            let mut left = vec![self.one; self.dense_len / 2];
            let mut right = vec![self.one; self.dense_len / 2];

            self.coeffs.iter().flatten().for_each(|coeff| {
                if coeff.index % 2 == 0 {
                    left[coeff.index / 2] = coeff.value;
                } else {
                    right[coeff.index / 2] = coeff.value;
                }
            });
            (left, right)
        }
    }

    /// Computes the grand product layer output by this one.
    ///      L0'       R0'       L1'       R1'     <- Output layer
    ///      /\        /\        /\        /\
    ///     /  \      /  \      /  \      /  \
    ///    L0  R0    L1  R1    L2  R2    L3  R3   <- This layer
    #[tracing::instrument(
        skip_all,
        name = "SparseInterleavedPolynomial::layer_output",
        level = "trace"
    )]
    pub fn layer_output<N: Rep3NetworkWorker>(
        &self,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<Self> {
        if let Some(coalesced) = &self.coalesced {
            Ok(Self {
                dense_len: self.dense_len / 2,
                coeffs: vec![vec![]; self.batch_size()],
                coalesced: Some(coalesced.layer_output(io_ctx)?),
                one: self.one,
            })
        } else {
            let one_share = rep3::arithmetic::promote_to_trivial_share(io_ctx.id, F::one());
            let coeffs = io_ctx
                .worker(0)
                .par_iter(&self.coeffs, None, |segment, io_ctx| {
                    let mut output_segment: Vec<
                        FutureVal<F, SparseCoefficient<Rep3PrimeFieldShare<F>>, usize>,
                    > = Vec::with_capacity(segment.len());
                    let mut next_index_to_process = 0usize;
                    for (j, coeff) in segment.iter().enumerate() {
                        if coeff.index < next_index_to_process {
                            // Node was already multiplied with its sibling in a previous iteration
                            continue;
                        }
                        if coeff.index % 2 == 0 {
                            // Left node; try to find corresponding right node
                            let right = segment
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((coeff.index + 1, one_share).into());
                            if right.index == coeff.index + 1 {
                                // Corresponding right node was found; multiply them together
                                output_segment.push(FutureVal::pending_mul_args(
                                    right.value,
                                    coeff.value,
                                    coeff.index / 2,
                                ));
                            } else {
                                // Corresponding right node not found, so it must be 1
                                output_segment
                                    .push(FutureVal::Ready((coeff.index / 2, coeff.value).into()));
                            }
                            next_index_to_process = coeff.index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration, so it must have value 1
                            output_segment
                                .push(FutureVal::Ready((coeff.index / 2, coeff.value).into()));
                            next_index_to_process = coeff.index + 1;
                        }
                    }
                    output_segment.fufill_batched(io_ctx, |c, index| (index, c).into())
                })
                .context("while computing layer output")?;

            Ok(Self::new(coeffs, self.dense_len / 2, io_ctx.id))
        }
    }
}

impl<F: JoltField> Rep3Bindable<F> for Rep3SparseInterleavedPolynomial<F> {
    /// Incrementally binds a variable of the interleaved left and right polynomials.
    /// If `self` is coalesced, we invoke `DenseInterleavedPolynomial::bind`,
    /// processing nodes 4 at a time to preserve the interleaved order:
    ///   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    ///
    /// If `self` is not coalesced, we basically do the same thing but with the
    /// sparse vectors in `self.coeffs`, and many more cases to check 😬
    #[tracing::instrument(skip_all, name = "SparseInterleavedPolynomial::bind", level = "trace")]
    fn bind(&mut self, r: F, party_id: PartyID) {
        #[cfg(test)]
        let (mut left_before_binding, mut right_before_binding) = self.uninterleave();

        if let Some(coalesced) = &mut self.coalesced {
            let padded_len = self.dense_len.next_multiple_of(4);
            coalesced.bind(r, party_id);
            self.dense_len = padded_len / 2;
        } else {
            self.coeffs
                .par_iter_mut()
                .for_each(|segment: &mut Vec<SparseCoefficient<_>>| {
                    let mut next_left_node_to_process = 0;
                    let mut next_right_node_to_process = 0;
                    let mut bound_index = 0;

                    for j in 0..segment.len() {
                        let current = segment[j];
                        if current.index % 2 == 0 && current.index < next_left_node_to_process {
                            // This left node was already bound with its sibling in a previous iteration
                            continue;
                        }
                        if current.index % 2 == 1 && current.index < next_right_node_to_process {
                            // This right node was already bound with its sibling in a previous iteration
                            continue;
                        }

                        let neighbors = [
                            segment
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((current.index + 1, self.one).into()),
                            segment
                                .get(j + 2)
                                .cloned()
                                .unwrap_or((current.index + 2, self.one).into()),
                        ];
                        let find_neighbor = |query_index: usize| {
                            neighbors
                                .iter()
                                .find_map(|neighbor| {
                                    if neighbor.index == query_index {
                                        Some(neighbor.value)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(self.one)
                        };

                        match current.index % 4 {
                            0 => {
                                // Find sibling left node
                                let sibling_value = find_neighbor(current.index + 2);
                                segment[bound_index] = (
                                    current.index / 2,
                                    rep3::arithmetic::add_mul_public(
                                        current.value,
                                        sibling_value - current.value,
                                        r,
                                    ),
                                )
                                    .into();
                                next_left_node_to_process = current.index + 4;
                            }
                            1 => {
                                // Edge case: If this right node's neighbor is not 1 and has _not_
                                // been bound yet, we need to bind the neighbor first to preserve
                                // the monotonic ordering of the bound layer.
                                if next_left_node_to_process <= current.index + 1 {
                                    let left_neighbour_if_not_bound =
                                        segment.get(j + 1).map_or(None, |n| {
                                            if n.index == current.index + 1 {
                                                Some(n.value)
                                            } else {
                                                None
                                            }
                                        });
                                    if let Some(left_neighbor) = left_neighbour_if_not_bound {
                                        segment[bound_index] = (
                                            current.index / 2,
                                            rep3::arithmetic::add_public(
                                                rep3::arithmetic::mul_public(
                                                    rep3::arithmetic::sub_shared_by_public(
                                                        left_neighbor,
                                                        F::one(),
                                                        party_id,
                                                    ),
                                                    r,
                                                ),
                                                F::one(),
                                                party_id,
                                            ),
                                        )
                                            .into();
                                        bound_index += 1;
                                    }
                                    next_left_node_to_process = current.index + 3;
                                }

                                // Find sibling right node
                                let sibling_value = find_neighbor(current.index + 2);
                                segment[bound_index] = (
                                    current.index / 2 + 1,
                                    rep3::arithmetic::add_mul_public(
                                        current.value,
                                        sibling_value - current.value,
                                        r,
                                    ),
                                )
                                    .into();
                                next_right_node_to_process = current.index + 4;
                            }
                            2 => {
                                // Sibling left node wasn't encountered in previous iteration,
                                // so sibling must have value 1.
                                segment[bound_index] = (
                                    current.index / 2 - 1,
                                    // F::one() + r * (current.value - F::one()),
                                    rep3::arithmetic::add_public(
                                        rep3::arithmetic::sub_shared_by_public(
                                            current.value,
                                            F::one(),
                                            party_id,
                                        ) * r,
                                        F::one(),
                                        party_id,
                                    ),
                                )
                                    .into();
                                next_left_node_to_process = current.index + 2;
                            }
                            3 => {
                                // Sibling right node wasn't encountered in previous iteration,
                                // so sibling must have value 1.
                                segment[bound_index] = (
                                    current.index / 2,
                                    // F::one() + r * (current.value - F::one())
                                    rep3::arithmetic::add_public(
                                        rep3::arithmetic::sub_shared_by_public(
                                            current.value,
                                            F::one(),
                                            party_id,
                                        ) * r,
                                        F::one(),
                                        party_id,
                                    ),
                                )
                                    .into();
                                next_right_node_to_process = current.index + 2;
                            }
                            _ => unreachable!("?_?"),
                        }
                        bound_index += 1;
                    }
                    segment.truncate(bound_index);
                });

            self.dense_len /= 2;
            if (self.dense_len / self.batch_size()) == 2 {
                // Coalesce
                self.coalesced = Some(Rep3DenseInterleavedPolynomial::new(self.coalesce()));
            }
        }
    }
}

impl<F: JoltField, Network: Rep3NetworkWorker> Rep3BatchedGrandProductLayerWorker<F, Network>
    for Rep3SparseInterleavedPolynomial<F>
{
}

impl<F: JoltField, ProofTranscript, Network>
    Rep3BatchedGrandProductLayer<F, ProofTranscript, Network> for Rep3SparseInterleavedPolynomial<F>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
}

impl<F: JoltField, Network: Rep3NetworkWorker> Rep3BatchedCubicSumcheckWorker<F, Network>
    for Rep3SparseInterleavedPolynomial<F>
{
    /// We want to compute the evaluations of the following univariate cubic polynomial at
    /// points {0, 1, 2, 3}:
    ///     \sum_{x} eq(r, x) * left(x) * right(x)
    /// where the inner summation is over all but the "least significant bit" of the multilinear
    /// polynomials `eq`, `left`, and `right`. We denote this "least significant" variable x_b.
    ///
    /// Computing these evaluations requires processing pairs of adjacent coefficients of
    /// `eq`, `left`, and `right`.
    /// If `self` is coalesced, we invoke `DenseInterleavedPolynomial::compute_cubic`, processing
    /// 4 values at a time:
    ///                 coeffs = [L, R, L, R, L, R, ...]
    ///                           |  |  |  |
    ///    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
    ///     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
    ///
    /// If `self` is not coalesced, we basically do the same thing but with with the
    /// sparse vectors in `self.coeffs`, some fancy optimizations, and many more cases to check 😬
    #[tracing::instrument(
        skip_all,
        name = "SparseInterleavedPolynomial::compute_cubic",
        level = "trace"
    )]
    fn compute_cubic(
        &self,
        eq_poly: &SplitEqPolynomial<F>,
        previous_round_claim: F,
        party_id: PartyID,
        // io_ctx: &mut IoContext<Network>,
    ) -> UniPoly<F> {
        if let Some(coalesced) = &self.coalesced {
            let span = tracing::trace_span!("sparse_interleaved_poly::compute_cubic::coalesced");
            let _enter = span.enter();
            return Rep3BatchedCubicSumcheckWorker::<F, Network>::compute_cubic(
                coalesced,
                eq_poly,
                previous_round_claim,
                party_id,
            );
        }

        let one_share = rep3::arithmetic::promote_to_trivial_share(party_id, F::one());

        // We use the Dao-Thaler optimization for the EQ polynomial, so there are two cases we
        // must handle. For details, refer to Section 2.2 of https://eprint.iacr.org/2024/1210.pdf
        let cubic_evals = if eq_poly.E1_len == 1 {
            let span = tracing::trace_span!("sparse_interleaved_poly::compute_cubic::E1_len=1");
            let _enter = span.enter();
            // If `eq_poly.E1` has been fully bound, we compute the cubic polynomial as we
            // would without the Dao-Thaler optimization, using the standard linear-time
            // sumcheck algorithm with optimizations for sparsity.

            let eq_evals: Vec<(F, F, F)> = eq_poly
                .E2
                .par_chunks(2)
                .take(self.dense_len / 4)
                .map(|eq_chunk| {
                    let eval_point_0 = eq_chunk[0];
                    let m_eq = eq_chunk[1] - eq_chunk[0];
                    let eval_point_2 = eq_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
            // This is what \sum_{x} eq(r, x) * left(x) * right(x) would be if
            // `left` and `right` were both all ones.
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
            // Now we compute the deltas, correcting `eq_eval_sums` for the
            // elements of `left` and `right` that aren't ones.
            let deltas: (F, F, F) = self
                .coeffs
                .par_iter()
                .flat_map(|segment| {
                    segment
                        .par_chunk_by(|x, y| x.index / 4 == y.index / 4)
                        .map(|sparse_block| {
                            let block_index = sparse_block[0].index / 4;
                            let mut block = [one_share; 4];
                            for coeff in sparse_block {
                                block[coeff.index % 4] = coeff.value;
                            }

                            let left = (block[0], block[2]);
                            let right = (block[1], block[3]);

                            let m_left = left.1 - left.0;
                            let m_right = right.1 - right.0;

                            let left_eval_2 = left.1 + m_left;
                            let left_eval_3 = left_eval_2 + m_left;

                            let right_eval_2 = right.1 + m_right;
                            let right_eval_3 = right_eval_2 + m_right;

                            let eq_evals = eq_evals[block_index];
                            let e0 = additive::sub_shared_by_public(
                                rep3::arithmetic::mul_mul_public(left.0, right.0, eq_evals.0),
                                eq_evals.0,
                                party_id,
                            );
                            let e1 = additive::sub_shared_by_public(
                                rep3::arithmetic::mul_mul_public(
                                    left_eval_2,
                                    right_eval_2,
                                    eq_evals.1,
                                ),
                                eq_evals.1,
                                party_id,
                            );
                            let e2 = additive::sub_shared_by_public(
                                rep3::arithmetic::mul_mul_public(
                                    left_eval_3,
                                    right_eval_3,
                                    eq_evals.2,
                                ),
                                eq_evals.2,
                                party_id,
                            );

                            (e0, e1, e2)
                        })
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            (
                additive::add_public(deltas.0, eq_eval_sums.0, party_id),
                additive::add_public(deltas.1, eq_eval_sums.1, party_id),
                additive::add_public(deltas.2, eq_eval_sums.2, party_id),
            )
        } else {
            let span = tracing::trace_span!("sparse_interleaved_poly::compute_cubic::E1_len_not_1");
            let _enter = span.enter();
            // This is a more complicated version of the `else` case in
            // `DenseInterleavedPolynomial::compute_cubic`. Read that one first.

            // We start by computing the E1 evals:
            // (1 - j) * E1[0, x1] + j * E1[1, x1]
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
            // Now compute \sum_{x1} ((1 - j) * E1[0, x1] + j * E1[1, x1])
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

            // Iterate over the non-one coefficients and compute the deltas (relative to
            // what the cubic would be if all the coefficients were ones).
            let deltas = self
                .coeffs
                .par_iter()
                .flat_map(|segment| {
                    segment
                        .par_chunk_by(|a, b| {
                            // Group by x2
                            let a_x2 = (a.index / 4) >> num_x1_bits;
                            let b_x2 = (b.index / 4) >> num_x1_bits;
                            a_x2 == b_x2
                        })
                        .map(|chunk| {
                            let mut inner_sum = (F::zero(), F::zero(), F::zero());
                            let x2 = (chunk[0].index / 4) >> num_x1_bits;

                            for sparse_block in chunk.chunk_by(|x, y| x.index / 4 == y.index / 4) {
                                let block_index = sparse_block[0].index / 4;
                                let mut block = [one_share; 4];
                                for coeff in sparse_block {
                                    block[coeff.index % 4] = coeff.value;
                                }

                                let left = (block[0], block[2]);
                                let right = (block[1], block[3]);

                                let m_left = left.1 - left.0;
                                let m_right = right.1 - right.0;

                                let left_eval_2 = left.1 + m_left;
                                let left_eval_3 = left_eval_2 + m_left;

                                let right_eval_2 = right.1 + m_right;
                                let right_eval_3 = right_eval_2 + m_right;

                                let x1 = block_index & x1_bitmask;
                                let delta = (
                                    // E1_evals[x1].0.mul_0_optimized(
                                    //     left.0.mul_1_optimized(right.0) - F::one(),
                                    // ),
                                    // E1_evals[x1].1 * (left_eval_2 * right_eval_2 - F::one()),
                                    // E1_evals[x1].2 * (left_eval_3 * right_eval_3 - F::one()),
                                    additive::sub_shared_by_public(
                                        rep3::arithmetic::mul_mul_public(
                                            left.0,
                                            right.0,
                                            E1_evals[x1].0 * eq_poly.E2[x2],
                                        ),
                                        E1_evals[x1].0 * eq_poly.E2[x2],
                                        party_id,
                                    ),
                                    additive::sub_shared_by_public(
                                        rep3::arithmetic::mul_mul_public(
                                            left_eval_2,
                                            right_eval_2,
                                            E1_evals[x1].1 * eq_poly.E2[x2],
                                        ),
                                        E1_evals[x1].1 * eq_poly.E2[x2],
                                        party_id,
                                    ),
                                    additive::sub_shared_by_public(
                                        rep3::arithmetic::mul_mul_public(
                                            left_eval_3,
                                            right_eval_3,
                                            E1_evals[x1].2 * eq_poly.E2[x2],
                                        ),
                                        E1_evals[x1].2 * eq_poly.E2[x2],
                                        party_id,
                                    ),
                                );
                                inner_sum.0 += delta.0;
                                inner_sum.1 += delta.1;
                                inner_sum.2 += delta.2;
                            }

                            (inner_sum.0, inner_sum.1, inner_sum.2)
                        })
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            // The cubic evals assuming all the coefficients are ones is affected by the
            // `dense_len`, since we implicitly 0-pad the `dense_len` to a power of 2.
            //
            // As a refresher, the cubic evals we're computing are:
            //
            // \sum_{x2} E2[x2] * (\sum_{x1} ((1 - j) * E1[0, x1] + j * E1[1, x1]) * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)))
            let evals_assuming_all_ones = if self.dense_len.is_power_of_two() {
                // If `dense_len` is a power of 2, there is no 0-padding.
                //
                // So we have:
                // \sum_{x2} (E2[x2] * (\sum_{x1} ((1 - j) * E1[0, x1] + j * E1[1, x1]) * 1))
                //   = \sum_{x2} (E2[x2] * \sum_{x1} E1_evals[x1])
                //   = (\sum_{x2} E2[x2]) * (\sum_{x1} E1_evals[x1])
                //   = 1 * E1_eval_sums
                E1_eval_sums
            } else {
                let chunk_size = self.dense_len.next_power_of_two() / eq_poly.E2_len;
                let num_all_one_chunks = self.dense_len / chunk_size;
                let E2_sum: F = eq_poly.E2[..num_all_one_chunks].iter().sum();
                if self.dense_len % chunk_size == 0 {
                    // If `dense_len` isn't a power of 2 but evenly divides `chunk_size`,
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
                    let span = tracing::trace_span!("sparse_interleaved_poly::compute_cubic::E1_len_not_1_dense_len_not_power_of_two");
                    // If `dense_len` isn't a power of 2 and doesn't divide `chunk_size`,
                    // the last nonzero "chunk" will have (self.dense_len % chunk_size) ones,
                    // followed by (chunk_size - self.dense_len % chunk_size) zeros,
                    // e.g. 1 1 1 1 1 1 1 1 0 0 0 0
                    //
                    // This handles this last chunk:
                    let last_chunk_evals = E1_evals[..(self.dense_len % chunk_size) / 4]
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

        let cubic = UniPoly::from_evals(&cubic_evals);

        // #[cfg(test)]
        // {
        //     let dense = DenseInterleavedPolynomial::new(self.coalesce());
        //     let dense_cubic = BatchedCubicSumcheck::<F, ProofTranscript>::compute_cubic(
        //         &dense,
        //         eq_poly,
        //         previous_round_claim,
        //     );
        //     assert_eq!(cubic, dense_cubic);
        // }

        cubic
    }

    fn final_claims(&self, _: PartyID) -> (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>) {
        assert_eq!(self.dense_len, 2);
        let dense = self.to_dense();
        (dense[0], dense[1])
    }
}

impl<F: JoltField, ProofTranscript, Network> Rep3BatchedCubicSumcheck<F, ProofTranscript, Network>
    for Rep3SparseInterleavedPolynomial<F>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
}
