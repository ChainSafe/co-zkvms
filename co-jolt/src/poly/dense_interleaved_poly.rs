use std::slice::Chunks;

use crate::{
    poly::Rep3DensePolynomial,
    subprotocols::{
        grand_product::{Rep3BatchedGrandProductLayer, Rep3BatchedGrandProductLayerWorker},
        sumcheck::{Rep3BatchedCubicSumcheck, Rep3BatchedCubicSumcheckWorker, Rep3Bindable},
    },
    utils::{thread::unsafe_allocate_zero_vec, transcript::Transcript},
};
use eyre::Context;
use itertools::izip;
use jolt_core::field::JoltField;
use jolt_core::subprotocols::{
    grand_product::BatchedGrandProductLayer,
    sumcheck::{BatchedCubicSumcheck, Bindable},
};
use mpc_core::protocols::{
    additive::AdditiveShare,
    rep3::{
        self,
        network::{
            IoContext, IoContextPool, Rep3Network, Rep3NetworkCoordinator, Rep3NetworkWorker,
        },
        PartyID, Rep3PrimeFieldShare,
    },
};
use rayon::{prelude::*, slice::Chunks as RayonChunks};

use jolt_core::poly::{split_eq_poly::SplitEqPolynomial, unipoly::UniPoly};

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
        let prod = io_ctx.worker(0).par_chunks(
            left.into_par_iter().zip(right.into_par_iter()),
            None,
            |chunk, io_ctx| {
                let (left, right): (Vec<_>, Vec<_>) =
                    tracing::trace_span!("unzip").in_scope(|| chunk.into_iter().unzip());
                let span = tracing::trace_span!("mul_vec");
                let _span_enter = span.enter();
                rep3::arithmetic::mul_vec(&left, &right, io_ctx).context("while multiplying left")
            },
        )?;
        // let span = tracing::trace_span!("mul_vec");
        // let _span_enter = span.enter();
        // let prod = rep3::arithmetic::mul_vec_par(&left, &right, io_ctx)
        //     .context("while multiplying left")?;
        // drop(_span_enter);
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
    fn bind(&mut self, r: F, party_id: PartyID) {
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
        eq_poly: &SplitEqPolynomial<F>,
        previous_round_claim: AdditiveShare<F>,
        _: PartyID,
    ) -> UniPoly<F> {
        // We use the Dao-Thaler optimization for the EQ polynomial, so there are two cases we
        // must handle. For details, refer to Section 2.2 of https://eprint.iacr.org/2024/1210.pdf
        let cubic_evals = if eq_poly.E1_len == 1 {
            // If `eq_poly.E1` has been fully bound, we compute the cubic polynomial as we
            // would without the Dao-Thaler optimization, using the standard linear-time
            // sumcheck algorithm.
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

                    let m_left = left.1 - left.0;
                    let m_right = right.1 - right.0;

                    let left_eval_2 = left.1 + m_left;
                    let left_eval_3 = left_eval_2 + m_left;

                    let right_eval_2 = right.1 + m_right;
                    let right_eval_3 = right_eval_2 + m_right;

                    (
                        rep3::arithmetic::mul_mul_public(left.0, right.0, eq_evals.0),
                        rep3::arithmetic::mul_mul_public(left_eval_2, right_eval_2, eq_evals.1),
                        rep3::arithmetic::mul_mul_public(left_eval_3, right_eval_3, eq_evals.2),
                    )
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                )
        } else {
            // If `eq_poly.E1` has NOT been fully bound, we compute the cubic polynomial
            // using the nested summation approach described in Section 2.2 of https://eprint.iacr.org/2024/1210.pdf
            //
            // Note, however, that we reverse the inner/outer summation compared to the
            // description in the paper. I.e. instead of:
            //
            // \sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * (\sum_x2 E2[x2] * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)))
            //
            // we do:
            //
            // \sum_x2 E2[x2] * (\sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)))
            //
            // because it has better memory locality.

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

            let chunk_size = (self.len.next_power_of_two() / eq_poly.E2_len).max(1);
            eq_poly.E2[..eq_poly.E2_len]
                .par_iter()
                .zip(self.par_chunks(chunk_size))
                .map(|(E2_eval, P_x2)| {
                    // The for-loop below corresponds to the inner sum:
                    // \sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2))
                    let mut inner_sum = (F::zero(), F::zero(), F::zero());
                    for (E1_evals, P_chunk) in E1_evals.iter().zip(P_x2.chunks(4)) {
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

                        inner_sum.0 +=
                            rep3::arithmetic::mul_mul_public(left.0, right.0, E1_evals.0);
                        inner_sum.1 +=
                            rep3::arithmetic::mul_mul_public(left_eval_2, right_eval_2, E1_evals.1);
                        inner_sum.2 +=
                            rep3::arithmetic::mul_mul_public(left_eval_3, right_eval_3, E1_evals.2);
                    }

                    // Multiply the inner sum by E2[x2]
                    (
                        *E2_eval * inner_sum.0,
                        *E2_eval * inner_sum.1,
                        *E2_eval * inner_sum.2,
                    )
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
        UniPoly::from_evals(&cubic_evals)
    }

    fn final_claims(&self, _: PartyID) -> (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>) {
        assert_eq!(self.len(), 2);
        let left_claim = self.coeffs[0];
        let right_claim = self.coeffs[1];
        (left_claim, right_claim)
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
