use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::Polynomial;
use crate::utils::types::Rep3Value;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use jolt_core::utils::compute_dotproduct;
use jolt_core::{
    field::OptimizedMul,
    poly::compact_polynomial::{CompactPolynomial, SmallScalar},
};
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};

use rayon::prelude::*;
use snarks_core::math::Math;

#[derive(Debug, Clone, PartialEq)]
pub enum Rep3MultilinearPolynomial<F: JoltField> {
    Public(MultilinearPolynomial<F>),
    Shared(Rep3DensePolynomial<F>),
}

impl<F: JoltField> Default for Rep3MultilinearPolynomial<F> {
    fn default() -> Self {
        Self::Public(MultilinearPolynomial::default())
    }
}

impl<F: JoltField> Rep3MultilinearPolynomial<F> {
    pub fn public(poly: MultilinearPolynomial<F>) -> Self {
        Self::Public(poly)
    }

    pub fn shared(poly: Rep3DensePolynomial<F>) -> Self {
        Self::Shared(poly)
    }

    pub fn from_shared_coeffs(coeffs: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        Self::shared(Rep3DensePolynomial::new(coeffs))
    }

    pub fn new_shard_shared(
        coeffs: Vec<Rep3PrimeFieldShare<F>>,
        full_len: usize,
        log_num_workers: usize,
        worker_idx: usize,
    ) -> Self {
        Self::shared(Rep3DensePolynomial::new_shard(
            coeffs,
            full_len,
            log_num_workers,
            worker_idx,
        ))
    }

    pub fn new_shard_public_u8(
        coeffs: Vec<u8>,
        full_len: usize,
        log_num_workers: usize,
        worker_idx: usize,
    ) -> Self {
        Self::public(MultilinearPolynomial::U8Scalars(
            CompactPolynomial::new_shard(coeffs, full_len, log_num_workers, worker_idx),
        ))
    }

    pub fn new_shard_public_u32(
        coeffs: Vec<u32>,
        full_len: usize,
        log_num_workers: usize,
        worker_idx: usize,
    ) -> Self {
        Self::public(MultilinearPolynomial::U32Scalars(
            CompactPolynomial::new_shard(coeffs, full_len, log_num_workers, worker_idx),
        ))
    }

    pub fn new_shard_public_u64(
        coeffs: Vec<u64>,
        full_len: usize,
        log_num_workers: usize,
        worker_idx: usize,
    ) -> Self {
        Self::public(MultilinearPolynomial::U64Scalars(
            CompactPolynomial::new_shard(coeffs, full_len, log_num_workers, worker_idx),
        ))
    }

    pub fn as_shared(&self) -> &Rep3DensePolynomial<F> {
        match self {
            Rep3MultilinearPolynomial::Shared(poly) => poly,
            Rep3MultilinearPolynomial::Public { .. } => {
                panic!("Not a shared polynomial")
            }
        }
    }

    pub fn as_shared_mut(&mut self) -> &mut Rep3DensePolynomial<F> {
        match self {
            Rep3MultilinearPolynomial::Shared(poly) => poly,
            Rep3MultilinearPolynomial::Public { .. } => {
                panic!("Not a shared polynomial")
            }
        }
    }

    pub fn as_public(&self) -> &MultilinearPolynomial<F> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly,
            Rep3MultilinearPolynomial::Shared(_) => {
                panic!("Not a public polynomial")
            }
        }
    }

    pub fn as_public_mut(&mut self) -> &mut MultilinearPolynomial<F> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly,
            Rep3MultilinearPolynomial::Shared(_) => {
                panic!("Not a public polynomial")
            }
        }
    }

    pub fn to_full_poly(self) -> Self {
        match self {
            Self::Public { .. } => unreachable!(),
            Self::Shared(poly) => Self::shared(poly.as_full_poly()),
        }
    }

    pub fn combine_shares(polys: Vec<Self>) -> MultilinearPolynomial<F> {
        let [s0, s1, s2] = polys.try_into().unwrap();
        let a = rep3::combine_field_elements::<F>(
            s0.as_shared().coeffs_ref(),
            s1.as_shared().coeffs_ref(),
            s2.as_shared().coeffs_ref(),
        );
        MultilinearPolynomial::from(a)
    }

    pub fn dot_product_with_public(&self, other: &[F]) -> Rep3Value<F> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.dot_product(other).into(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.dot_product_with_public(other).into(),
        }
    }

    pub fn get_coeff(&self, index: usize) -> Rep3Value<F> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.get_coeff(index).into(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.get_coeff(index).into(),
        }
    }

    pub fn get_bound_coeff(&self, index: usize) -> Rep3Value<F> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.get_bound_coeff(index).into(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.get_bound_coeff(index).into(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.len(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.len(),
        }
    }

    pub fn original_len(&self) -> usize {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.original_len(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.coeffs_ref().len(),
        }
    }

    pub fn full_len(&self) -> usize {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.full_len(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.full_len(),
        }
    }

    pub fn get_num_vars(&self) -> usize {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.get_num_vars(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.get_num_vars(),
        }
    }

    /// Returns the shard range (start, end) for this polynomial in global coordinates.
    /// For unsharded polynomials, returns (0, full_len).
    pub fn shard_range(&self) -> (usize, usize) {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => match poly {
                MultilinearPolynomial::LargeScalars(_) => (0, poly.full_len()),
                MultilinearPolynomial::U8Scalars(p) => p.chunk_global_range(),
                MultilinearPolynomial::U16Scalars(p) => p.chunk_global_range(),
                MultilinearPolynomial::U32Scalars(p) => p.chunk_global_range(),
                MultilinearPolynomial::U64Scalars(p) => p.chunk_global_range(),
                MultilinearPolynomial::I64Scalars(p) => p.chunk_global_range(),
            },
            Rep3MultilinearPolynomial::Shared(poly) => {
                poly.global_chunk_range.unwrap_or((0, poly.full_len()))
            }
        }
    }

    /// Multiplies the polynomial's coefficient at `index` by a field element.
    pub fn scale_coeff(
        &self,
        index: usize,
        scaling_factor: F,
        scaling_factor_r2_adjusted: F,
    ) -> Rep3Value<F> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => Rep3Value::Public(poly.scale_coeff(
                index,
                scaling_factor,
                scaling_factor_r2_adjusted,
            )),
            Rep3MultilinearPolynomial::Shared(poly) => Rep3Value::Shared(
                rep3::arithmetic::mul_public(poly.get_coeff(index), scaling_factor),
            ),
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "Rep3MultilinearPolynomial::linear_combination",
        level = "trace"
    )]
    pub fn linear_combination(
        polynomials: &[&Self],
        coefficients: &[F],
        party_id: PartyID,
    ) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());

        let max_length = polynomials
            .iter()
            .map(|poly| poly.full_len())
            .max()
            .unwrap();
        tracing::trace!("Max length: {}", max_length);

        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(max_length);
        let chunk_size = (max_length / num_chunks).max(1);

        // If any of the polynomials is shared, the resulting polynomial will be shared
        let result_is_shared = polynomials
            .iter()
            .any(|poly| matches!(poly, Rep3MultilinearPolynomial::Shared(_)));

        let lc_coeffs: Vec<Rep3Value<F>> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let global_index = chunk_index * chunk_size;
                let mut chunk = vec![Rep3Value::Public(F::zero()); chunk_size];

                for (coeff, poly) in coefficients.iter().zip(polynomials.iter()) {
                    let poly_len = poly.full_len();
                    if global_index >= poly_len {
                        continue;
                    }

                    // Get the shard range for this polynomial
                    let (shard_start, shard_end) = poly.shard_range();

                    // Calculate the overlap between the current chunk and the polynomial's shard
                    let chunk_end = (global_index + chunk_size).min(poly_len);
                    let overlap_start = global_index.max(shard_start);
                    let overlap_end = chunk_end.min(shard_end);

                    if overlap_start >= overlap_end {
                        continue;
                    }

                    // Calculate offsets
                    let chunk_offset = overlap_start - global_index;
                    let local_index = overlap_start - shard_start;
                    // let overlap_len = overlap_end - overlap_start;

                    match poly {
                        Rep3MultilinearPolynomial::Public(poly) => match poly {
                            MultilinearPolynomial::LargeScalars(poly) => {
                                debug_assert!(!poly.is_bound());
                                let poly_evals = &poly.evals_ref()[global_index..];
                                for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                    rlc.add_public_assign(
                                        poly_eval.mul_01_optimized(*coeff),
                                        party_id,
                                    );
                                }
                            }
                            MultilinearPolynomial::U8Scalars(poly) => {
                                for (rlc, poly_eval) in
                                    chunk[chunk_offset..] // ..chunk_offset + overlap_len
                                        .iter_mut()
                                        .zip(&poly.coeffs_ref()[local_index..])
                                {
                                    rlc.add_public_assign(poly_eval.field_mul(*coeff), party_id);
                                }
                            }
                            MultilinearPolynomial::U16Scalars(poly) => {
                                for (rlc, poly_eval) in chunk[chunk_offset..]
                                    .iter_mut()
                                    .zip(&poly.coeffs_ref()[local_index..])
                                {
                                    rlc.add_public_assign(poly_eval.field_mul(*coeff), party_id);
                                }
                            }
                            MultilinearPolynomial::U32Scalars(poly) => {
                                for (rlc, poly_eval) in chunk[chunk_offset..]
                                    .iter_mut()
                                    .zip(&poly.coeffs_ref()[local_index..])
                                {
                                    rlc.add_public_assign(poly_eval.field_mul(*coeff), party_id);
                                }
                            }
                            MultilinearPolynomial::U64Scalars(poly) => {
                                for (rlc, poly_eval) in chunk[chunk_offset..]
                                    .iter_mut()
                                    .zip(&poly.coeffs_ref()[local_index..])
                                {
                                    rlc.add_public_assign(poly_eval.field_mul(*coeff), party_id);
                                }
                            }
                            _ => unreachable!(),
                        },
                        Rep3MultilinearPolynomial::Shared(poly) => {
                            for (rlc, poly_eval) in chunk[chunk_offset..]
                                .iter_mut()
                                .zip(&poly.coeffs_ref()[local_index..])
                            {
                                rlc.add_shared_assign(
                                    rep3::arithmetic::mul_public(*poly_eval, *coeff),
                                    party_id,
                                );
                            }
                        }
                    }
                }
                chunk
            })
            .collect();

        if result_is_shared {
            Rep3MultilinearPolynomial::from_shared_coeffs(
                lc_coeffs
                    .into_par_iter()
                    .map(|x| x.into_shared_rep3(party_id))
                    .collect(),
            )
        } else {
            Rep3MultilinearPolynomial::public(MultilinearPolynomial::from(
                lc_coeffs
                    .into_par_iter()
                    .map(|x| x.as_public())
                    .collect::<Vec<F>>(),
            ))
        }
    }

    #[inline]
    pub fn sumcheck_evals_into_share(
        &self,
        index: usize,
        degree: usize,
        order: BindingOrder,
        party_id: PartyID,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => rep3::arithmetic::promote_to_trivial_shares(
                poly.sumcheck_evals(index, degree, order),
                party_id,
            ),
            Rep3MultilinearPolynomial::Shared(poly) => poly.sumcheck_evals(index, degree, order),
        }
    }

    pub fn batch_evaluate_worker(
        polys: &[&Self],
        r: &[F],
        log_num_workers: usize,
        worker_idx: usize,
    ) -> (Vec<Rep3Value<F>>, Vec<F>) {
        let eq = EqPolynomial::evals(r);
        let evals = Rep3MultilinearPolynomial::batch_evaluate_at_chi(
            polys,
            &eq[EqPolynomial::evals_range_worker(r, log_num_workers, worker_idx)],
        );

        (evals, eq)
    }

    #[tracing::instrument(skip_all, name = "Rep3MultilinearPolynomial::batch_evaluate_at_chi")]
    pub fn batch_evaluate_at_chi(polys: &[&Self], chi: &[F]) -> Vec<Rep3Value<F>> {
        let evals: Vec<_> = polys
            .into_par_iter()
            .map(|&poly| match poly {
                Rep3MultilinearPolynomial::Public(MultilinearPolynomial::LargeScalars(poly)) => {
                    Rep3Value::Public(poly.evaluate_at_chi_low_optimized(&chi))
                }
                Rep3MultilinearPolynomial::Public(poly) => {
                    Rep3Value::Public(poly.dot_product(&chi))
                }
                Rep3MultilinearPolynomial::Shared(poly) => {
                    Rep3Value::Additive(poly.evaluate_at_chi_optimized(&chi))
                }
            })
            .collect();
        evals
    }

    pub fn batch_evaluate_full(
        polys: &[&Rep3MultilinearPolynomial<F>],
        r: &[F],
    ) -> (Vec<Rep3Value<F>>, Vec<F>) {
        let eq = EqPolynomial::evals(r);

        let evals: Vec<_> = polys
            .into_par_iter()
            .map(|&poly| match poly {
                Rep3MultilinearPolynomial::Public(MultilinearPolynomial::LargeScalars(poly)) => {
                    Rep3Value::Public(poly.evaluate_at_chi_low_optimized(&eq))
                }
                Rep3MultilinearPolynomial::Public(poly) => match poly {
                    MultilinearPolynomial::LargeScalars(poly) => compute_dotproduct(&poly.Z, &eq),
                    MultilinearPolynomial::U8Scalars(poly) => poly
                        .coeffs
                        .par_iter()
                        .zip_eq(eq.par_iter())
                        .map(|(a, b)| a.field_mul(*b))
                        .sum(),
                    MultilinearPolynomial::U16Scalars(poly) => poly
                        .coeffs
                        .par_iter()
                        .zip_eq(eq.par_iter())
                        .map(|(a, b)| a.field_mul(*b))
                        .sum(),
                    MultilinearPolynomial::U32Scalars(poly) => poly
                        .coeffs
                        .par_iter()
                        .zip_eq(eq.par_iter())
                        .map(|(a, b)| a.field_mul(*b))
                        .sum(),
                    MultilinearPolynomial::U64Scalars(poly) => poly
                        .coeffs
                        .par_iter()
                        .zip_eq(eq.par_iter())
                        .map(|(a, b)| a.field_mul(*b))
                        .sum(),
                    MultilinearPolynomial::I64Scalars(poly) => poly
                        .coeffs
                        .par_iter()
                        .zip_eq(eq.par_iter())
                        .map(|(a, b)| a.field_mul(*b))
                        .sum(),
                }
                .into(),
                Rep3MultilinearPolynomial::Shared(poly) => {
                    Rep3Value::Additive(poly.evaluate_at_chi_optimized_full(&eq))
                }
            })
            .collect();
        (evals, eq)
    }
}

impl<F: JoltField> PolynomialBinding<F, Rep3Value<F>> for Rep3MultilinearPolynomial<F> {
    fn is_bound(&self) -> bool {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.is_bound(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.is_bound(),
        }
    }

    fn bind(&mut self, r: F, order: BindingOrder) {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.bind(r, order),
            Rep3MultilinearPolynomial::Shared(poly) => poly.bind(r, order),
        }
    }

    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.bind_parallel(r, order),
            Rep3MultilinearPolynomial::Shared(poly) => poly.bind_parallel(r, order),
        }
    }

    /// Warning: when poly is shared, returns the additive share.
    /// Use `final_sumcheck_claim_additive` instead.
    fn final_sumcheck_claim(&self) -> Rep3Value<F> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.final_sumcheck_claim().into(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.final_sumcheck_claim().into(),
        }
    }
}

impl<F: JoltField> PolynomialEvaluation<F, Rep3Value<F>> for Rep3MultilinearPolynomial<F> {
    fn evaluate(&self, r: &[F]) -> Rep3Value<F> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.evaluate(r).into(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.evaluate(r).into(),
        }
    }

    #[tracing::instrument(skip_all, name = "Rep3MultilinearPolynomial::batch_evaluate")]
    fn batch_evaluate(polys: &[&Self], r: &[F]) -> (Vec<Rep3Value<F>>, Vec<F>) {
        let eq = EqPolynomial::evals(r);

        let evals: Vec<_> = polys
            .into_par_iter()
            .map(|&poly| match poly {
                Rep3MultilinearPolynomial::Public(MultilinearPolynomial::LargeScalars(poly)) => {
                    Rep3Value::Public(poly.evaluate_at_chi_low_optimized(&eq))
                }
                Rep3MultilinearPolynomial::Public(poly) => Rep3Value::Public(poly.dot_product(&eq)),
                Rep3MultilinearPolynomial::Shared(poly) => {
                    Rep3Value::Additive(poly.evaluate_at_chi_optimized(&eq))
                }
            })
            .collect();
        (evals, eq)
    }

    #[inline]
    fn sumcheck_evals(
        &self,
        index: usize,
        degree: usize,
        order: BindingOrder,
    ) -> Vec<Rep3Value<F>> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly
                .sumcheck_evals(index, degree, order)
                .into_iter()
                .map(|x| x.into())
                .collect(),
            Rep3MultilinearPolynomial::Shared(poly) => poly
                .sumcheck_evals(index, degree, order)
                .into_iter()
                .map(|x| x.into())
                .collect(),
        }
    }
}

impl<F: JoltField> Polynomial<F> for Rep3MultilinearPolynomial<F> {
    fn len(&self) -> usize {
        self.len()
    }

    fn get_num_vars(&self) -> usize {
        self.get_num_vars()
    }

    fn get_bound_coeffs(&self) -> Vec<Rep3Value<F>> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => (0..self.len())
                .map(|i| Rep3Value::Public(poly.get_bound_coeff(i)))
                .collect(),
            Rep3MultilinearPolynomial::Shared(poly) => poly
                .bound_coeffs()
                .iter()
                .copied()
                .map(|x| x.into())
                .collect(),
        }
    }
}

//---------------- Conversion ----------------//

impl<'a, F: JoltField> TryInto<&'a Rep3DensePolynomial<F>> for &'a Rep3MultilinearPolynomial<F> {
    type Error = eyre::Error;

    fn try_into(self) -> Result<&'a Rep3DensePolynomial<F>, Self::Error> {
        match self {
            Rep3MultilinearPolynomial::Public(_) => Err(eyre::eyre!("Public polynomial")),
            Rep3MultilinearPolynomial::Shared(poly) => Ok(poly),
        }
    }
}

impl<'a, F: JoltField> TryInto<&'a mut Rep3DensePolynomial<F>>
    for &'a mut Rep3MultilinearPolynomial<F>
{
    type Error = eyre::Error;

    fn try_into(self) -> Result<&'a mut Rep3DensePolynomial<F>, Self::Error> {
        match self {
            Rep3MultilinearPolynomial::Public(_) => Err(eyre::eyre!("Public polynomial")),
            Rep3MultilinearPolynomial::Shared(poly) => Ok(poly),
        }
    }
}

impl<F: JoltField> TryInto<Rep3DensePolynomial<F>> for Rep3MultilinearPolynomial<F> {
    type Error = eyre::Error;

    fn try_into(self) -> Result<Rep3DensePolynomial<F>, Self::Error> {
        match self {
            Rep3MultilinearPolynomial::Public(_) => Err(eyre::eyre!("Public polynomial")),
            Rep3MultilinearPolynomial::Shared(poly) => Ok(poly),
        }
    }
}

impl<'a, F: JoltField> TryInto<&'a MultilinearPolynomial<F>> for &'a Rep3MultilinearPolynomial<F> {
    type Error = eyre::Error;

    fn try_into(self) -> Result<&'a MultilinearPolynomial<F>, Self::Error> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => Ok(poly),
            Rep3MultilinearPolynomial::Shared(_) => Err(eyre::eyre!("No public polynomial")),
        }
    }
}

impl<'a, F: JoltField> TryInto<&'a mut MultilinearPolynomial<F>>
    for &'a mut Rep3MultilinearPolynomial<F>
{
    type Error = eyre::Error;

    fn try_into(self) -> Result<&'a mut MultilinearPolynomial<F>, Self::Error> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => Ok(poly),
            Rep3MultilinearPolynomial::Shared(_) => Err(eyre::eyre!("No public polynomial")),
        }
    }
}

impl<F: JoltField> TryInto<MultilinearPolynomial<F>> for Rep3MultilinearPolynomial<F> {
    type Error = eyre::Error;

    fn try_into(self) -> Result<MultilinearPolynomial<F>, Self::Error> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => Ok(poly),
            Rep3MultilinearPolynomial::Shared(_) => Err(eyre::eyre!("No public polynomial")),
        }
    }
}

impl<F: JoltField> From<MultilinearPolynomial<F>> for Rep3MultilinearPolynomial<F> {
    fn from(poly: MultilinearPolynomial<F>) -> Self {
        Rep3MultilinearPolynomial::Public(poly)
    }
}

impl<F: JoltField> From<Rep3DensePolynomial<F>> for Rep3MultilinearPolynomial<F> {
    fn from(poly: Rep3DensePolynomial<F>) -> Self {
        Rep3MultilinearPolynomial::Shared(poly)
    }
}

impl<F: JoltField> From<Vec<Rep3PrimeFieldShare<F>>> for Rep3MultilinearPolynomial<F> {
    fn from(evals: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        Rep3MultilinearPolynomial::Shared(Rep3DensePolynomial::new(evals))
    }
}

impl<F: JoltField> From<Vec<u8>> for Rep3MultilinearPolynomial<F> {
    fn from(coeffs: Vec<u8>) -> Self {
        let poly = MultilinearPolynomial::U8Scalars(CompactPolynomial::from_coeffs(coeffs));
        Self::Public(poly)
    }
}

impl<F: JoltField> From<Vec<u32>> for Rep3MultilinearPolynomial<F> {
    fn from(coeffs: Vec<u32>) -> Self {
        let poly = MultilinearPolynomial::U32Scalars(CompactPolynomial::from_coeffs(coeffs));
        Self::Public(poly)
    }
}

impl<F: JoltField> From<Vec<u64>> for Rep3MultilinearPolynomial<F> {
    fn from(coeffs: Vec<u64>) -> Self {
        let poly = MultilinearPolynomial::U64Scalars(CompactPolynomial::from_coeffs(coeffs));
        Self::Public(poly)
    }
}

impl<'a, F: JoltField> TryFrom<&'a Rep3MultilinearPolynomial<F>> for &'a DensePolynomial<F> {
    type Error = eyre::Error;

    fn try_from(poly: &'a Rep3MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            Rep3MultilinearPolynomial::Public(poly) => poly
                .try_into()
                .map_err(|_| eyre::eyre!("Not a dense polynomial")),
            _ => Err(eyre::eyre!("Not a public polynomial")),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a Rep3MultilinearPolynomial<F>> for &'a CompactPolynomial<u8, F> {
    type Error = eyre::Error;

    fn try_from(poly: &'a Rep3MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            Rep3MultilinearPolynomial::Public(poly) => poly
                .try_into()
                .map_err(|_| eyre::eyre!("Not a u8 polynomial")),
            _ => Err(eyre::eyre!("Not a public polynomial")),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a Rep3MultilinearPolynomial<F>> for &'a CompactPolynomial<u16, F> {
    type Error = eyre::Error;

    fn try_from(poly: &'a Rep3MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            Rep3MultilinearPolynomial::Public(poly) => poly
                .try_into()
                .map_err(|_| eyre::eyre!("Not a u16 polynomial")),
            _ => Err(eyre::eyre!("Not a public polynomial")),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a Rep3MultilinearPolynomial<F>> for &'a CompactPolynomial<u32, F> {
    type Error = eyre::Error;

    fn try_from(poly: &'a Rep3MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            Rep3MultilinearPolynomial::Public(poly) => poly
                .try_into()
                .map_err(|_| eyre::eyre!("Not a u32 polynomial")),
            _ => Err(eyre::eyre!("Not a public polynomial")),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a Rep3MultilinearPolynomial<F>> for &'a CompactPolynomial<u64, F> {
    type Error = eyre::Error;

    fn try_from(poly: &'a Rep3MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            Rep3MultilinearPolynomial::Public(poly) => poly
                .try_into()
                .map_err(|_| eyre::eyre!("Not a u64 polynomial")),
            _ => Err(eyre::eyre!("Not a public polynomial")),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a Rep3MultilinearPolynomial<F>> for &'a CompactPolynomial<i64, F> {
    type Error = eyre::Error;

    fn try_from(poly: &'a Rep3MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            Rep3MultilinearPolynomial::Public(poly) => poly
                .try_into()
                .map_err(|_| eyre::eyre!("Not a i64 polynomial")),
            _ => Err(eyre::eyre!("Not a public polynomial")),
        }
    }
}

pub trait Rep3PolysConversion<'a, F: JoltField> {
    fn try_into_shared(self) -> Vec<&'a Rep3DensePolynomial<F>>;

    fn try_into_public(self) -> Vec<&'a MultilinearPolynomial<F>>;
}

pub trait Rep3PolysConversionMut<'a, F: JoltField> {
    fn try_into_shared_mut(self) -> Vec<&'a mut Rep3DensePolynomial<F>>;

    fn try_into_public_mut(self) -> Vec<&'a mut MultilinearPolynomial<F>>;
}

impl<'a, F: JoltField, I> Rep3PolysConversion<'a, F> for I
where
    I: IntoIterator<Item = &'a Rep3MultilinearPolynomial<F>>,
{
    fn try_into_shared(self) -> Vec<&'a Rep3DensePolynomial<F>> {
        self.into_iter()
            .map(|p| p.try_into())
            .collect::<Result<Vec<_>, eyre::Error>>()
            .unwrap()
    }

    fn try_into_public(self) -> Vec<&'a MultilinearPolynomial<F>> {
        self.into_iter()
            .map(|p| p.try_into())
            .collect::<Result<Vec<_>, eyre::Error>>()
            .unwrap()
    }
}

impl<'a, F: JoltField, I> Rep3PolysConversionMut<'a, F> for I
where
    I: IntoIterator<Item = &'a mut Rep3MultilinearPolynomial<F>>,
{
    fn try_into_shared_mut(self) -> Vec<&'a mut Rep3DensePolynomial<F>> {
        self.into_iter()
            .map(|p| p.try_into())
            .collect::<Result<Vec<_>, eyre::Error>>()
            .unwrap()
    }

    fn try_into_public_mut(self) -> Vec<&'a mut MultilinearPolynomial<F>> {
        self.into_iter()
            .map(|p| p.try_into())
            .collect::<Result<Vec<_>, eyre::Error>>()
            .unwrap()
    }
}

//---------------- Serialization ----------------//

impl<F: JoltField> CanonicalSerialize for Rep3MultilinearPolynomial<F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => {
                (0_u8).serialize_with_mode(&mut writer, compress)?;
                poly.serialize_with_mode(&mut writer, compress)?;
            }
            Rep3MultilinearPolynomial::Shared(poly) => {
                (1_u8).serialize_with_mode(&mut writer, compress)?;
                poly.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => {
                (0_u8).serialized_size(compress) + poly.serialized_size(compress)
            }
            Rep3MultilinearPolynomial::Shared(poly) => {
                (1_u8).serialized_size(compress) + poly.serialized_size(compress)
            }
        }
    }
}

impl<F: JoltField> CanonicalDeserialize for Rep3MultilinearPolynomial<F> {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        // TODO(protoben) Can we use strum for this?
        let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let res =
            match discriminant {
                0 => Rep3MultilinearPolynomial::Public(
                    MultilinearPolynomial::deserialize_with_mode(&mut reader, compress, validate)?,
                ),
                1 => Rep3MultilinearPolynomial::Shared(Rep3DensePolynomial::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?),
                _ => Err(SerializationError::InvalidData)?,
            };
        Ok(res)
    }
}

impl<F: JoltField> Valid for Rep3MultilinearPolynomial<F> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.check(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.check(),
        }
    }
}

#[cfg(test)]
mod test {
    use itertools::{izip, Itertools};
    use snarks_core::math::Math;

    use super::*;

    #[test]
    fn test_rls() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build_global()
            .expect("set global Rayon pool");
        let env_filter = tracing_subscriber::EnvFilter::builder()
            .with_default_directive(tracing::Level::INFO.into())
            .from_env_lossy();
        let subscriber = tracing_subscriber::layer::SubscriberExt::with(
            tracing_subscriber::Registry::default(),
            env_filter,
        );
        let _ = tracing::subscriber::set_global_default(
            tracing_subscriber::layer::SubscriberExt::with(
                subscriber,
                tracing_forest::ForestLayer::default(),
            ),
        );
        type F = ark_bn254::Fr;

        const N: usize = 1 << 10;

        let worker1_polys = vec![
            Rep3MultilinearPolynomial::<F>::new_shard_shared(
                rep3::arithmetic::promote_to_trivial_shares(
                    (0u64..N as u64).map(F::from).collect(),
                    // vec![F::from(1); N],
                    PartyID::ID0,
                ),
                N,
                1,
                0,
            ),
            Rep3MultilinearPolynomial::<F>::new_shard_shared(
                rep3::arithmetic::promote_to_trivial_shares(
                    (1u64..N as u64 + 1).map(F::from).collect(),
                    // vec![F::from(2); N],
                    PartyID::ID0,
                ),
                N,
                1,
                0,
            ),
            Rep3MultilinearPolynomial::<F>::from(rep3::arithmetic::promote_to_trivial_shares(
                (2u64..N as u64 + 2).map(F::from).collect(),
                // vec![F::from(2); N],
                PartyID::ID0,
            )),
        ];

        let worker2_polys = vec![
            Rep3MultilinearPolynomial::<F>::new_shard_shared(
                rep3::arithmetic::promote_to_trivial_shares(
                    (0u64..N as u64).map(F::from).collect(),
                    // vec![F::from(1); N],
                    PartyID::ID0,
                ),
                N,
                1,
                1,
            ),
            Rep3MultilinearPolynomial::<F>::new_shard_shared(
                rep3::arithmetic::promote_to_trivial_shares(
                    (1u64..N as u64 + 1).map(F::from).collect(),
                    // vec![F::from(2); N],
                    PartyID::ID0,
                ),
                N,
                1,
                1,
            ),
            Rep3MultilinearPolynomial::<F>::from(rep3::arithmetic::promote_to_trivial_shares(
                (3u64..N as u64 + 3).map(F::from).collect(),
                // vec![F::from(2); N],
                PartyID::ID0,
            )),
        ];
        tracing::info!("-------------worker 1-------------");
        let rls1 = Rep3MultilinearPolynomial::<F>::linear_combination(
            &worker1_polys.iter().collect_vec(),
            &vec![F::from(2), F::from(3), F::from(4)],
            PartyID::ID0,
        )
        .as_shared()
        .coeffs
        .iter()
        .map(|coeff| coeff.a)
        .collect_vec();
        tracing::info!("-------------worker 2-------------");
        let rls2 = Rep3MultilinearPolynomial::<F>::linear_combination(
            &worker2_polys.iter().collect_vec(),
            &vec![F::from(2), F::from(3), F::from(5)],
            PartyID::ID0,
        )
        .as_shared()
        .coeffs
        .iter()
        .map(|coeff| coeff.a)
        .collect_vec();
        tracing::info!("----------------------------");

        let polys = vec![
            Rep3MultilinearPolynomial::<F>::from(rep3::arithmetic::promote_to_trivial_shares(
                (0u64..N as u64).map(F::from).collect(),
                // vec![F::from(1); N],
                PartyID::ID0,
            )),
            Rep3MultilinearPolynomial::<F>::from(rep3::arithmetic::promote_to_trivial_shares(
                (1u64..N as u64 + 1).map(F::from).collect(),
                // vec![F::from(2); N],
                PartyID::ID0,
            )),
            Rep3MultilinearPolynomial::<F>::from(rep3::arithmetic::promote_to_trivial_shares(
                (2u64..N as u64 + 2).map(F::from).collect(),
                // vec![F::from(2); N],
                PartyID::ID0,
            )),
            Rep3MultilinearPolynomial::<F>::from(rep3::arithmetic::promote_to_trivial_shares(
                (3u64..N as u64 + 3).map(F::from).collect(),
                // vec![F::from(2); N],
                PartyID::ID0,
            )),
        ];

        let rls = Rep3MultilinearPolynomial::<F>::linear_combination(
            &polys.iter().collect_vec(),
            &vec![F::from(2), F::from(3), F::from(4), F::from(5)],
            PartyID::ID0,
        )
        .as_shared()
        .coeffs
        .iter()
        .map(|coeff| coeff.a)
        .collect_vec();

        let rls_check = izip!(rls1, rls2).map(|(a, b)| a + b).collect_vec();

        assert_eq!(rls_check, rls);
    }
}
