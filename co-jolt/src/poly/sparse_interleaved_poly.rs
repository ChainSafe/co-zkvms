use super::dense_interleaved_poly::Rep3DenseInterleavedPolynomial;
use crate::field::JoltField;
use crate::poly::split_eq_poly::DistributedSplitEqPolynomial;
use crate::poly::Rep3DensePolynomial;
use crate::subprotocols::grand_product::Rep3BatchedGrandProductLayerWorker;
use crate::subprotocols::sumcheck::{Rep3BatchedCubicSumcheckWorker, Rep3Bindable};
use crate::subprotocols::{
    grand_product::Rep3BatchedGrandProductLayer, sumcheck::Rep3BatchedCubicSumcheck,
};
use crate::utils::future::{FutureExt, FutureRep3};

use eyre::Context;
use jolt_core::poly::{
    sparse_interleaved_poly::SparseCoefficient, split_eq_poly::SplitEqPolynomial,
};
use jolt_core::utils::{math::Math, transcript::Transcript};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};
use rayon::{prelude::*, vec};

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

        let mut coalesced = vec![one; dense_len];
        coeffs
            .iter()
            .flatten()
            .for_each(|sparse_coeff| coalesced[sparse_coeff.index] = sparse_coeff.value);

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
            let one_share = rep3::arithmetic::promote_to_trivial_share(io_ctx.party_id(), F::one());
            let coeffs = io_ctx
                .par_iter(&self.coeffs, None, |segment, io_ctx| {
                    let mut output_segment: Vec<
                        FutureRep3<F, SparseCoefficient<Rep3PrimeFieldShare<F>>, usize>,
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
                                output_segment.push(FutureRep3::mul_args(
                                    right.value,
                                    coeff.value,
                                    coeff.index / 2,
                                ));
                            } else {
                                // Corresponding right node not found, so it must be 1
                                output_segment
                                    .push(FutureRep3::Ready((coeff.index / 2, coeff.value).into()));
                            }
                            next_index_to_process = coeff.index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration, so it must have value 1
                            output_segment
                                .push(FutureRep3::Ready((coeff.index / 2, coeff.value).into()));
                            next_index_to_process = coeff.index + 1;
                        }
                    }
                    output_segment.fulfill_batched(io_ctx, |c, index| (index, c).into())
                })
                .context("while computing layer output")?;

            Ok(Self::new(coeffs, self.dense_len / 2, io_ctx.party_id()))
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
        eq_poly: &DistributedSplitEqPolynomial<F>,
        // previous_round_claim: AdditiveShare<F>,
        party_id: PartyID,
    ) -> [AdditiveShare<F>; 3] {
        if let Some(coalesced) = &self.coalesced {
            let span = tracing::trace_span!("sparse_interleaved_poly::compute_cubic::coalesced");
            let _enter = span.enter();
            return Rep3BatchedCubicSumcheckWorker::<F, Network>::compute_cubic(
                coalesced, eq_poly, // previous_round_claim,
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

            let eq_evals: Vec<[F; 3]> = eq_poly
                .E2
                .par_chunks(2)
                .take(self.dense_len / 4)
                .map(|eq_chunk| {
                    let eval_point_0 = eq_chunk[0];
                    let m_eq = eq_chunk[1] - eq_chunk[0];
                    let eval_point_2 = eq_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    [eval_point_0, eval_point_2, eval_point_3]
                })
                .collect();
            // This is what \sum_{x} eq(r, x) * left(x) * right(x) would be if
            // `left` and `right` were both all ones.
            let eq_eval_sums: [F; 3] = eq_evals
                .par_iter()
                .fold(
                    || [F::zero(); 3],
                    |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                )
                .reduce(
                    || [F::zero(); 3],
                    |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                );
            // Now we compute the deltas, correcting `eq_eval_sums` for the
            // elements of `left` and `right` that aren't ones.
            let deltas: [AdditiveShare<F>; 3] = self
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
                                left.0 * right.0 * eq_evals[0],
                                eq_evals[0],
                                party_id,
                            );
                            let e1 = additive::sub_shared_by_public(
                                left_eval_2 * right_eval_2 * eq_evals[1],
                                eq_evals[1],
                                party_id,
                            );
                            let e2 = additive::sub_shared_by_public(
                                left_eval_3 * right_eval_3 * eq_evals[2],
                                eq_evals[2],
                                party_id,
                            );

                            [e0, e1, e2]
                        })
                })
                .reduce(
                    || [AdditiveShare::<F>::zero(); 3],
                    |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                );

            [
                additive::add_public(deltas[0], eq_eval_sums[0], party_id),
                additive::add_public(deltas[1], eq_eval_sums[1], party_id),
                additive::add_public(deltas[2], eq_eval_sums[2], party_id),
            ]
        } else {
            let span = tracing::trace_span!("sparse_interleaved_poly::compute_cubic::E1_len_not_1");
            let _enter = span.enter();
            // This is a more complicated version of the `else` case in
            // `DenseInterleavedPolynomial::compute_cubic`. Read that one first.
            let E1_len = eq_poly.E1_len;

            // We start by computing the E1 evals:
            // (1 - j) * E1[0, x1] + j * E1[1, x1]
            let E1_evals: Vec<_> = eq_poly.E1[..E1_len]
                .par_chunks(2)
                .map(|E1_chunk| {
                    let eval_point_0 = E1_chunk[0];
                    let m_eq = E1_chunk[1] - E1_chunk[0];
                    let eval_point_2 = E1_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    [eval_point_0, eval_point_2, eval_point_3]
                })
                .collect();

            // Prefix sums over E1_evals.
            // prefix[j][i] = sum_{k < i} E1_evals[k][j]
            let mut prefix_sums = vec![[F::zero(); 3]; E1_len + 1];

            for (i, e) in E1_evals.iter().enumerate() {
                prefix_sums[i + 1][0] = prefix_sums[i][0] + e[0];
                prefix_sums[i + 1][1] = prefix_sums[i][1] + e[1];
                prefix_sums[i + 1][2] = prefix_sums[i][2] + e[2];
            }

            let eq_slice_start = eq_poly.global_start;
            let eq_slice_end = eq_slice_start + core::cmp::min(eq_poly.len, self.dense_len / 2);

            let E2_local_bound = eq_slice_end
                .div_ceil(E1_len)
                .saturating_sub(eq_poly.row_start)
                .min(eq_poly.E2_len);

            // Iterate over the non-one coefficients and compute the deltas (relative to
            // what the cubic would be if all the coefficients were ones).
            let deltas = self
                .coeffs
                .par_iter()
                .flat_map(|segment| {
                    segment
                        .par_chunk_by(|a, b| {
                            // Group by *global* row index (after accounting for global_start
                            // and the fact that each 4-coeff block corresponds to 2 Eq points).
                            let a_block = a.index / 4;
                            let b_block = b.index / 4;
                            let a_eq = eq_slice_start + 2 * a_block;
                            let b_eq = eq_slice_start + 2 * b_block;
                            let a_row = a_eq / E1_len;
                            let b_row = b_eq / E1_len;

                            a_row == b_row
                        })
                        .map(|chunk| {
                            let mut inner_sum = [AdditiveShare::<F>::zero(); 3];

                            // Global row index for this chunk.
                            // let E2_i = (chunk[0].index / 4) >> num_x1_bits;
                            let first_block = chunk[0].index / 4;
                            let eq0 = eq_slice_start + 2 * first_block;
                            let r = eq0 / E1_len; // global row index

                            // Map to local E2 index.
                            debug_assert!(r >= eq_poly.row_start);
                            let x2 = r - eq_poly.row_start;
                            debug_assert!(x2 <= E2_local_bound);

                            let row_global = eq_poly.row_start + x2;
                            let row_first = row_global * E1_len;
                            let row_last = row_first + E1_len;

                            let eq_first = eq_slice_start.max(row_first);
                            let eq_last = (eq_slice_start + eq_poly.len).min(row_last);
                            debug_assert!(eq_last > eq_first);

                            let col_from = eq_first - row_first;
                            let col_to = eq_last - row_first;
                            debug_assert!(
                                col_from % 2 == 0 && col_to % 2 == 0,
                                "misaligned Eq slice within row"
                            );

                            for sparse_block in chunk.chunk_by(|x, y| x.index / 4 == y.index / 4) {
                                let block_index = sparse_block[0].index / 4;
                                let eq_global = eq_slice_start + 2 * block_index;
                                debug_assert!(
                                    eq_global >= eq_first && eq_global < eq_last,
                                    "block out of bounds"
                                );

                                // Column inside the row.
                                let col = eq_global - row_first;
                                debug_assert!(col < E1_len);
                                debug_assert!(col % 2 == 0, "block not aligned to E1 pair");

                                // Pair index for this (i_C) inside the row.
                                let x1 = (col / 2) as usize;
                                debug_assert!(x1 < E1_evals.len());

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

                                let delta = (
                                    additive::sub_shared_by_public(
                                        left.0 * right.0,
                                        F::one(),
                                        party_id,
                                    ) * E1_evals[x1][0],
                                    additive::sub_shared_by_public(
                                        left_eval_2 * right_eval_2,
                                        F::one(),
                                        party_id,
                                    ) * E1_evals[x1][1],
                                    additive::sub_shared_by_public(
                                        left_eval_3 * right_eval_3,
                                        F::one(),
                                        party_id,
                                    ) * E1_evals[x1][2],
                                );
                                inner_sum[0] += delta.0;
                                inner_sum[1] += delta.1;
                                inner_sum[2] += delta.2;
                            }

                            inner_sum.map(|x| x * eq_poly.E2[x2])
                        })
                })
                .reduce(
                    || [AdditiveShare::<F>::zero(); 3],
                    |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                );

            // The cubic evals assuming all the coefficients are ones is affected by the
            // `dense_len`, since we implicitly 0-pad the `dense_len` to a power of 2.
            //
            // \sum_{x2} E2[x2] * (\sum_{x1} ((1 - j) * E1[0, x1] + j * E1[1, x1]) *
            // * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)))
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
                    debug_assert!(poly_from < self.dense_len);
                    let poly_bound = (self.dense_len - poly_from) / 4;

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
        };

        cubic_evals
    }

    fn final_evals(&self, _: usize, _: PartyID) -> Vec<AdditiveShare<F>> {
        // assert_eq!(self.dense_len, 2);
        let dense = self.to_dense();
        dense
            .bound_coeffs()
            .par_iter()
            .map(|c| c.into_additive())
            .collect()
    }
}

impl<F: JoltField, ProofTranscript, Network> Rep3BatchedCubicSumcheck<F, ProofTranscript, Network>
    for Rep3SparseInterleavedPolynomial<F>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
}
