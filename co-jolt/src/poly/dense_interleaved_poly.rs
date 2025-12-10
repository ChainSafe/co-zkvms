use crate::field::JoltField;
use crate::poly::split_eq_poly::DistributedSplitEqPolynomial;
use crate::{
    subprotocols::{
        grand_product::{Rep3BatchedGrandProductLayer, Rep3BatchedGrandProductLayerWorker},
        sumcheck::{Rep3BatchedCubicSumcheck, Rep3BatchedCubicSumcheckWorker, Rep3Bindable},
    },
    utils::transcript::Transcript,
};
use eyre::Context;
use std::slice::Chunks;

use mpc_core::protocols::{
    additive::AdditiveShare,
    rep3::{
        self,
        network::{IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker},
        PartyID, Rep3PrimeFieldShare,
    },
};
use rayon::{prelude::*, slice::Chunks as RayonChunks};

/// Represents a single layer of a grand product circuit.
///
/// A layer is assumed to be arranged in "interleaved" order, i.e. the natural
/// order in the visual representation of the circuit:
///      /\        /\        /\        /\
///     /  \      /  \      /  \      /  \
///    L0  R0    L1  R1    L2  R2    L3  R3   <- This layer would be represented as [L0, R0, L1, R1, L2, R2, L3, R3]
///                                           (as opposed to e.g. [L0, L1, L2, L3, R0, R1, R2, R3])
#[derive(Default, Debug, Clone)]
pub struct Rep3DenseInterleavedPolynomial<F: JoltField> {
    /// The coefficients for the "left" and "right" polynomials comprising a
    /// dense grand product layer.
    /// The coefficients are in interleaved order:
    /// [L0, R0, L1, R1, L2, R2, L3, R3, ...]
    pub(crate) coeffs: Vec<Rep3PrimeFieldShare<F>>,
    /// The effective length of `coeffs`. When binding, we update this length
    /// instead of truncating `coeffs`, which incurs the cost of dropping the
    /// truncated values.
    len: usize,
    /// A reused buffer where bound values are written to during `bind`.
    /// With every bind, `coeffs` and `binding_scratch_space` are swapped.
    binding_scratch_space: Vec<Rep3PrimeFieldShare<F>>,
}

// impl<F: JoltField> PartialEq for DenseInterleavedPolynomial<F> {
//     fn eq(&self, other: &Self) -> bool {
//         if self.len != other.len {
//             false
//         } else {
//             self.coeffs[..self.len] == other.coeffs[..other.len]
//         }
//     }
// }

impl<F: JoltField> Rep3DenseInterleavedPolynomial<F> {
    pub fn new(coeffs: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        assert!(coeffs.len() % 2 == 0);
        let len = coeffs.len();
        Self {
            coeffs,
            len,
            binding_scratch_space: vec![
                Rep3PrimeFieldShare::zero_share();
                len.next_multiple_of(4) / 2
            ],
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> impl Iterator<Item = &Rep3PrimeFieldShare<F>> {
        self.coeffs[..self.len].iter()
    }

    pub fn chunks(&self, chunk_size: usize) -> Chunks<'_, Rep3PrimeFieldShare<F>> {
        self.coeffs[..self.len].chunks(chunk_size)
    }
    pub fn par_chunks(&self, chunk_size: usize) -> RayonChunks<'_, Rep3PrimeFieldShare<F>> {
        self.coeffs[..self.len].par_chunks(chunk_size)
    }

    pub fn interleave(left: &[Rep3PrimeFieldShare<F>], right: &[Rep3PrimeFieldShare<F>]) -> Self {
        assert_eq!(left.len(), right.len());
        let mut interleaved = vec![];
        for i in 0..left.len() {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }
        Self::new(interleaved)
    }

    #[tracing::instrument(
        skip_all,
        name = "DenseInterleavedPolynomial::uninterleave",
        level = "trace"
    )]
    pub fn uninterleave(&self) -> (Vec<Rep3PrimeFieldShare<F>>, Vec<Rep3PrimeFieldShare<F>>) {
        let left: Vec<_> = self.coeffs[..self.len]
            .par_iter()
            .copied()
            .step_by(2)
            .collect();
        let mut right: Vec<_> = self.coeffs[..self.len]
            .par_iter()
            .copied()
            .skip(1)
            .step_by(2)
            .collect();
        if right.len() < left.len() {
            right.resize(left.len(), Rep3PrimeFieldShare::zero_share());
        }
        (left, right)
    }

    #[tracing::instrument(
        skip_all,
        name = "DenseInterleavedPolynomial::layer_output",
        level = "trace"
    )]
    pub fn layer_output<N: Rep3NetworkWorker>(
        &self,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<Self> {
        let (left, right) = self.uninterleave();
        let prod = io_ctx.par_chunks(
            left.into_par_iter().zip(right.into_par_iter()),
            None,
            |chunk, io_ctx| {
                let (left, right): (Vec<_>, Vec<_>) = chunk.into_iter().unzip();
                rep3::arithmetic::mul_vec(&left, &right, io_ctx).context("while multiplying left")
            },
        )?;
        Ok(Self::new(prod))
    }
}

impl<F: JoltField> Rep3Bindable<F> for Rep3DenseInterleavedPolynomial<F> {
    /// Incrementally binds a variable of the interleaved left and right polynomials.
    /// To preserve the interleaved order of coefficients, we bind values like this:
    ///   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    #[tracing::instrument(skip_all, name = "DenseInterleavedPolynomial::bind", level = "trace")]
    fn bind(&mut self, r: F, _: PartyID) {
        let padded_len = self.len.next_multiple_of(4);
        // In order to parallelize binding while obeying Rust ownership rules, we
        // must write to a different vector than we are reading from. `binding_scratch_space`
        // serves this purpose.
        self.binding_scratch_space
            .par_chunks_mut(2)
            .zip(self.coeffs[..self.len].par_chunks(4))
            .for_each(|(bound_chunk, unbound_chunk)| {
                let unbound_chunk = [
                    *unbound_chunk
                        .first()
                        .unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                    *unbound_chunk
                        .get(1)
                        .unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                    *unbound_chunk
                        .get(2)
                        .unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                    *unbound_chunk
                        .get(3)
                        .unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                ];

                bound_chunk[0] = rep3::arithmetic::add_mul_public(
                    unbound_chunk[0],
                    unbound_chunk[2] - unbound_chunk[0],
                    r,
                );
                bound_chunk[1] = rep3::arithmetic::add_mul_public(
                    unbound_chunk[1],
                    unbound_chunk[3] - unbound_chunk[1],
                    r,
                );
            });

        self.len = padded_len / 2;
        // Point `self.coeffs` to the bound coefficients, and `self.coeffs` will serve as the
        // binding scratch space in the next invocation of `bind`.
        std::mem::swap(&mut self.coeffs, &mut self.binding_scratch_space);
    }
}

// impl<F: JoltField, ProofTranscript: Transcript> BatchedGrandProductLayer<F, ProofTranscript>
//     for Rep3DenseInterleavedPolynomial<F>
// {
// }
impl<F: JoltField, Network: Rep3NetworkWorker> Rep3BatchedCubicSumcheckWorker<F, Network>
    for Rep3DenseInterleavedPolynomial<F>
{
    #[tracing::instrument(
        skip_all,
        name = "Rep3DenseInterleavedPolynomial::compute_cubic",
        level = "trace"
    )]
    fn compute_cubic(
        &self,
        eq_poly: &DistributedSplitEqPolynomial<F>,
        // previous_round_claim: AdditiveShare<F>,
        _: PartyID,
    ) -> [AdditiveShare<F>; 3] {
        // We use the Dao–Thaler optimization for the EQ polynomial, so there are two cases:
        //   1) E1_len == 1: fully bound inner dimension → standard linear-time sumcheck.
        //   2) E1_len > 1:  factored Eq = E2(i_A,i_B) * E1(i_C) → nested summation.
        // For details, refer to Section 2.2 of https://eprint.iacr.org/2024/1210.pdf
        let cubic_evals = if eq_poly.E1_len == 1 {
            // ---------------- linear-time mode: no Dao–Thaler factorization left ----------------
            //
            // At this point, E2 already contains the full Eq evaluations over the remaining
            // variables, aligned 1:1 with the points that `poly` represents.
            self.par_chunks(4)
                .zip(eq_poly.E2.par_chunks(2))
                .map(|(layer_chunk, eq_chunk)| {
                    let eq_evals = {
                        let eval_point_0 = eq_chunk[0];
                        let m_eq = eq_chunk[1] - eq_chunk[0];
                        let eval_point_2 = eq_chunk[1] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        (eval_point_0, eval_point_2, eval_point_3)
                    };

                    // Interleaved [L0, R0, L1, R1] chunk for this point.
                    let left = (
                        *layer_chunk
                            .first()
                            .unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                        *layer_chunk
                            .get(2)
                            .unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                    );
                    let right = (
                        *layer_chunk
                            .get(1)
                            .unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                        *layer_chunk
                            .get(3)
                            .unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                    );

                    // Evaluate left(r) and right(r) at j = 2, 3 using affine interpolation.
                    let m_left = left.1 - left.0;
                    let m_right = right.1 - right.0;

                    let left_eval_2 = left.1 + m_left;
                    let left_eval_3 = left_eval_2 + m_left;

                    let right_eval_2 = right.1 + m_right;
                    let right_eval_3 = right_eval_2 + m_right;

                    [
                        left.0 * right.0 * eq_evals.0,
                        left_eval_2 * right_eval_2 * eq_evals.1,
                        left_eval_3 * right_eval_3 * eq_evals.2,
                    ]
                })
                .reduce(
                    || [AdditiveShare::<F>::zero(); 3],
                    |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                )
        } else {
            // ---------------- Dao–Thaler mode: Eq(i_A,i_C,i_B) = E2(i_A,i_B) * E1(i_C) ----------------
            //
            // Here we treat:
            //   - E1: inner dimension over C (columns),
            //   - E2: outer dimension over A|B (rows).
            //
            // For each row (fixed i_A,i_B), we:
            //   1. combine the relevant E1 entries with P-chunks along the C direction,
            //   2. multiply the resulting inner sum by the corresponding E2(row) value.
            let E1_len = eq_poly.E1_len;

            // Precompute Dao–Thaler E1 evaluations at the three needed points j ∈ {0,2,3}
            // for each C-position (i_C). E1_evals[c] = (E1(c, j=0), E1(c, j=2), E1(c, j=3)).
            let E1_evals: Vec<_> = eq_poly.E1[..E1_len]
                .par_chunks(2)
                .map(|E1_chunk| {
                    let eval_point_0 = E1_chunk[0];
                    let m_eq = E1_chunk[1] - E1_chunk[0];
                    let eval_point_2 = E1_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            // The poly currently represents `poly.len() / 2` Eq points (each point
            // corresponds to 2 coefficients in an interleaved L/R representation).
            //
            // This worker is logically responsible for `worker_len` Eq points starting
            // at `global_start`, but we must not read beyond what `poly` actually has.
            let eq_slice_end = eq_poly.global_start + core::cmp::min(eq_poly.len, self.len() / 2);

            // Upper bound (exclusive) on E2 indices this worker can actually use:
            //
            //   - A row with global index r covers global Eq indices [r * E1_len, (r+1)*E1_len),
            //   - we only care about rows that intersect [global_start, slice_end),
            //   - convert that intersection into a local row-offset range for this worker.
            let E2_local_bound = eq_slice_end
                .div_ceil(E1_len) // first row index strictly after slice_end
                .saturating_sub(eq_poly.row_start)
                .min(eq_poly.E2_len);

            // Dao–Thaler outer loop: iterate over each relevant row of E2 and perform
            // the inner sum over the C dimension, restricted to this worker’s slice.
            eq_poly.E2[..E2_local_bound]
                .par_iter()
                .enumerate()
                .map(|(E2_i, E2_eval)| {
                    // Global row index in the full Eq table.
                    let r = eq_poly.row_start + E2_i;

                    // Global Eq index range covered by this row: [row_first, row_last).
                    let row_first = r * E1_len;
                    let row_last = row_first + E1_len;

                    // Intersection with the worker’s assigned slice [global_start, worker_end),
                    // expressed in global Eq indices.
                    let eq_first = eq_poly.global_start.max(row_first);
                    let eq_last = (eq_poly.global_start + eq_poly.len).min(row_last);

                    // We expect this row to intersect the slice if it is within E2_local_bound.
                    debug_assert!(eq_last > eq_first);

                    // Column offsets inside the row (in Eq points).
                    let col_from = eq_first - row_first;
                    let col_to = eq_last - row_first;

                    // Each Dao–Thaler E1 entry spans 2 Eq points; enforce alignment.
                    debug_assert!(
                        col_from % 2 == 0 && col_to % 2 == 0,
                        "misaligned Eq slice within row"
                    );

                    // Range of C-indices (pairs) in this row that belong to this worker.
                    let E1_from = col_from / 2;
                    let E1_to = col_to / 2;

                    // Local Eq point index inside this worker’s slice:
                    //
                    //   local_point_idx = eq_first - global_start
                    //
                    // Each point corresponds to 2 coefficients in the interleaved polynomial.
                    let poly_from = (eq_first - eq_poly.global_start) * 2;
                    assert!(poly_from < self.len(), "coeff_start out of bounds");

                    let mut inner_sum = [AdditiveShare::<F>::zero(); 3];

                    // Inner Dao–Thaler sum along C:
                    //
                    //   sum_{c in [pair_from,pair_to)} E1_evals[c] * P_chunk(c)
                    //
                    // where P_chunk(c) is the 4-coefficient interleaved block for that
                    // position in the grand product GKR wiring.
                    for (E1_evals, P_chunk) in E1_evals[E1_from..E1_to]
                        .iter()
                        .zip(self.coeffs[poly_from..self.len()].chunks(4))
                    {
                        let left = (
                            *P_chunk
                                .first()
                                .unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                            *P_chunk.get(2).unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                        );
                        let right = (
                            *P_chunk.get(1).unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                            *P_chunk.get(3).unwrap_or(&Rep3PrimeFieldShare::zero_share()),
                        );
                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let left_eval_2 = left.1 + m_left;
                        let left_eval_3 = left_eval_2 + m_left;

                        let right_eval_2 = right.1 + m_right;
                        let right_eval_3 = right_eval_2 + m_right;

                        inner_sum[0] += left.0 * right.0 * E1_evals.0;
                        inner_sum[1] += left_eval_2 * right_eval_2 * E1_evals.1;
                        inner_sum[2] += left_eval_3 * right_eval_3 * E1_evals.2;
                    }

                    // Multiply the inner sum by E2[x2]
                    inner_sum.map(|x| x * *E2_eval)
                })
                .reduce(
                    || [AdditiveShare::<F>::zero(); 3],
                    |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                )
        };

        cubic_evals
    }

    fn final_evals(&self, _: usize, _: PartyID) -> Vec<AdditiveShare<F>> {
        self.coeffs[..self.len()]
            .par_iter()
            .map(|c| c.into_additive())
            .collect()
    }
}

impl<F: JoltField, ProofTranscript, Network> Rep3BatchedCubicSumcheck<F, ProofTranscript, Network>
    for Rep3DenseInterleavedPolynomial<F>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
}

impl<F: JoltField, Network: Rep3NetworkWorker> Rep3BatchedGrandProductLayerWorker<F, Network>
    for Rep3DenseInterleavedPolynomial<F>
{
}

impl<F: JoltField, ProofTranscript, Network>
    Rep3BatchedGrandProductLayer<F, ProofTranscript, Network> for Rep3DenseInterleavedPolynomial<F>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
}
