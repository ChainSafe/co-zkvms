use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::PolyDegree;
use crate::utils::element::SharedOrPublic;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use jolt_core::{
    field::{JoltField, OptimizedMul},
    poly::compact_polynomial::{CompactPolynomial, SmallScalar},
};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};

use rayon::prelude::*;

// pub type MultilinearPolynomial<F> = DensePolynomial<F>;

#[derive(Debug, Clone, PartialEq)]
pub enum Rep3MultilinearPolynomial<F: JoltField> {
    Public {
        poly: MultilinearPolynomial<F>,
        trivial_share: Option<Rep3DensePolynomial<F>>,
    },
    Shared(Rep3DensePolynomial<F>),
}

impl<F: JoltField> Default for Rep3MultilinearPolynomial<F> {
    fn default() -> Self {
        Self::Public {
            poly: MultilinearPolynomial::default(),
            trivial_share: None,
        }
    }
}

impl<F: JoltField> Rep3MultilinearPolynomial<F> {
    pub fn public(poly: MultilinearPolynomial<F>) -> Self {
        Self::Public {
            poly,
            trivial_share: None,
        }
    }

    pub fn shared(poly: Rep3DensePolynomial<F>) -> Self {
        Self::Shared(poly)
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

    pub fn from_shared_evals(evals: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        Self::shared(Rep3DensePolynomial::new(evals))
    }

    pub fn public_zero(num_evals: usize) -> Self {
        Self::public(MultilinearPolynomial::from(vec![F::zero(); num_evals]))
    }

    pub fn as_public(&self) -> &MultilinearPolynomial<F> {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly,
            Rep3MultilinearPolynomial::Shared(_) => {
                panic!("Not a public polynomial")
            }
        }
    }

    pub fn as_public_mut(&mut self) -> &mut MultilinearPolynomial<F> {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly,
            Rep3MultilinearPolynomial::Shared(_) => {
                panic!("Not a public polynomial")
            }
        }
    }

    pub fn public_with_trivial_share(poly: MultilinearPolynomial<F>, party_id: PartyID) -> Self {
        let trivial_share = Rep3DensePolynomial::new(rep3::arithmetic::promote_to_trivial_shares(
            poly.coeffs_as_field_elements(),
            party_id,
        ));
        Self::Public {
            poly,
            trivial_share: Some(trivial_share),
        }
    }

    pub fn try_combine_shares(polys: Vec<Self>) -> Result<MultilinearPolynomial<F>, eyre::Error> {
        let [s0, s1, s2] = polys.try_into().unwrap();
        let a = rep3::combine_field_elements::<F>(
            s0.as_shared().evals_ref(),
            s1.as_shared().evals_ref(),
            s2.as_shared().evals_ref(),
        );
        Ok(MultilinearPolynomial::from(a))
    }

    pub fn dot_product_with_public(&self, other: &[F]) -> SharedOrPublic<F> {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly.dot_product(other).into(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.dot_product_with_public(other).into(),
        }
    }

    pub fn get_coeff(&self, index: usize) -> SharedOrPublic<F> {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly.get_coeff(index).into(),
            Rep3MultilinearPolynomial::Shared(poly) => poly[index].into(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly.len(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.len(),
        }
    }

    pub fn get_num_vars(&self) -> usize {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly.get_num_vars(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.get_num_vars(),
        }
    }

    pub fn public_with_trivial_share_vec(
        polys: Vec<MultilinearPolynomial<F>>,
        party_id: PartyID,
    ) -> Vec<Self> {
        polys
            .iter()
            .map(|poly| Self::public_with_trivial_share(poly.clone(), party_id))
            .collect()
    }

    pub fn public_vec(polys: Vec<MultilinearPolynomial<F>>) -> Vec<Self> {
        polys
            .iter()
            .map(|poly| Self::public(poly.clone()))
            .collect()
    }

    /// Multiplies the polynomial's coefficient at `index` by a field element.
    pub fn scale_coeff(
        &self,
        index: usize,
        scaling_factor: F,
        scaling_factor_r2_adjusted: F,
    ) -> SharedOrPublic<F> {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => SharedOrPublic::Public(
                poly.scale_coeff(index, scaling_factor, scaling_factor_r2_adjusted),
            ),
            Rep3MultilinearPolynomial::Shared(poly) => {
                SharedOrPublic::Shared(rep3::arithmetic::mul_public(poly[index], scaling_factor))
            }
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

        let max_length = polynomials.iter().map(|poly| poly.len()).max().unwrap();
        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(max_length);
        let chunk_size = (max_length / num_chunks).max(1);

        // If any of the polynomials is shared, the resulting polynomial will be shared
        let result_is_shared = polynomials
            .iter()
            .any(|poly| matches!(poly, Rep3MultilinearPolynomial::Shared(_)));

        let lc_coeffs: Vec<SharedOrPublic<F>> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let index = chunk_index * chunk_size;
                let mut chunk = vec![SharedOrPublic::Public(F::zero()); chunk_size];

                for (coeff, poly) in coefficients.iter().zip(polynomials.iter()) {
                    let poly_len = poly.len();
                    if index >= poly_len {
                        continue;
                    }

                    match poly {
                        Rep3MultilinearPolynomial::Public { poly, .. } => match poly {
                            MultilinearPolynomial::LargeScalars(poly) => {
                                debug_assert!(!poly.is_bound());
                                let poly_evals = &poly.evals_ref()[index..];
                                for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                    rlc.add_public_assign(
                                        poly_eval.mul_01_optimized(*coeff),
                                        party_id,
                                    );
                                }
                            }
                            MultilinearPolynomial::U8Scalars(poly) => {
                                let poly_evals = &poly.coeffs[index..];
                                for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                    rlc.add_public_assign(poly_eval.field_mul(*coeff), party_id);
                                }
                            }
                            MultilinearPolynomial::U16Scalars(poly) => {
                                let poly_evals = &poly.coeffs[index..];
                                for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                    rlc.add_public_assign(poly_eval.field_mul(*coeff), party_id);
                                }
                            }
                            MultilinearPolynomial::U32Scalars(poly) => {
                                let poly_evals = &poly.coeffs[index..];
                                for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                    rlc.add_public_assign(poly_eval.field_mul(*coeff), party_id);
                                }
                            }
                            MultilinearPolynomial::U64Scalars(poly) => {
                                let poly_evals = &poly.coeffs[index..];
                                for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                    rlc.add_public_assign(poly_eval.field_mul(*coeff), party_id);
                                }
                            }
                            MultilinearPolynomial::I64Scalars(poly) => {
                                let poly_evals = &poly.coeffs[index..];
                                for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                    rlc.add_public_assign(poly_eval.field_mul(*coeff), party_id);
                                }
                            }
                        },
                        Rep3MultilinearPolynomial::Shared(poly) => {
                            let poly_evals = &poly.evals_ref()[index..];
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
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
            Rep3MultilinearPolynomial::from_shared_evals(
                lc_coeffs.into_par_iter().map(|x| x.as_shared()).collect(),
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

    pub fn split_poly(
        polys: Rep3MultilinearPolynomial<F>,
        log_workers: usize,
    ) -> Vec<Rep3MultilinearPolynomial<F>> {
        match polys {
            Rep3MultilinearPolynomial::Shared(poly) => {
                Rep3DensePolynomial::split_poly(poly, log_workers)
            }
            Rep3MultilinearPolynomial::Public { poly, .. } => split_public_poly(poly, log_workers)
                .into_iter()
                .map(|poly| Rep3MultilinearPolynomial::public(poly))
                .collect(),
        }
    }

    pub fn split_poly_vec(
        polys: Vec<Rep3MultilinearPolynomial<F>>,
        log_workers: usize,
    ) -> Vec<Vec<Rep3MultilinearPolynomial<F>>> {
        let mut chunks = vec![vec![]; 1 << log_workers];
        for poly in polys {
            let poly_chunks = Self::split_poly(poly, log_workers);
            for (chunk, poly_chunk) in chunks.iter_mut().zip(poly_chunks) {
                chunk.push(poly_chunk);
            }
        }
        chunks
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
            Rep3MultilinearPolynomial::Public { poly, .. } => {
                rep3::arithmetic::promote_to_trivial_shares(
                    poly.sumcheck_evals(index, degree, order),
                    party_id,
                )
            }
            Rep3MultilinearPolynomial::Shared(poly) => poly.sumcheck_evals(index, degree, order),
        }
    }

    pub fn final_sumcheck_claim_safe(&self) -> SharedOrPublic<F> {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly.final_sumcheck_claim().into(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.evals[0].into(),
        }
    }

    pub fn set_bound_eval(&mut self, index: usize, eval: SharedOrPublic<F>) {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => match poly {
                MultilinearPolynomial::LargeScalars(poly) => poly.Z[index] = eval.as_public(),
                MultilinearPolynomial::U8Scalars(poly) => {
                    poly.bound_coeffs[index] = eval.as_public()
                }
                MultilinearPolynomial::U16Scalars(poly) => {
                    poly.bound_coeffs[index] = eval.as_public()
                }
                MultilinearPolynomial::U32Scalars(poly) => {
                    poly.bound_coeffs[index] = eval.as_public()
                }
                MultilinearPolynomial::U64Scalars(poly) => {
                    poly.bound_coeffs[index] = eval.as_public()
                }
                MultilinearPolynomial::I64Scalars(poly) => {
                    poly.bound_coeffs[index] = eval.as_public()
                }
            },
            Rep3MultilinearPolynomial::Shared(poly) => poly.evals[index] = eval.as_shared(),
        }
    }
}

pub fn split_public_poly<F: JoltField>(
    mut poly: MultilinearPolynomial<F>,
    log_workers: usize,
) -> Vec<MultilinearPolynomial<F>> {
    let nv = poly.get_num_vars() - log_workers;
    let chunk_size = 1 << nv;
    let mut res = Vec::new();

    for _ in 0..1 << log_workers {
        match &mut poly {
            MultilinearPolynomial::LargeScalars(poly) => res.push(MultilinearPolynomial::from(
                poly.Z.drain(..chunk_size).collect::<Vec<_>>(),
            )),
            MultilinearPolynomial::U8Scalars(poly) => res.push(MultilinearPolynomial::from(
                poly.coeffs.drain(..chunk_size).collect::<Vec<_>>(),
            )),
            MultilinearPolynomial::U16Scalars(poly) => res.push(MultilinearPolynomial::from(
                poly.coeffs.drain(..chunk_size).collect::<Vec<_>>(),
            )),
            MultilinearPolynomial::U32Scalars(poly) => res.push(MultilinearPolynomial::from(
                poly.coeffs.drain(..chunk_size).collect::<Vec<_>>(),
            )),
            MultilinearPolynomial::U64Scalars(poly) => res.push(MultilinearPolynomial::from(
                poly.coeffs.drain(..chunk_size).collect::<Vec<_>>(),
            )),
            MultilinearPolynomial::I64Scalars(poly) => res.push(MultilinearPolynomial::from(
                poly.coeffs.drain(..chunk_size).collect::<Vec<_>>(),
            )),
        }
    }

    res
}

impl<F: JoltField> PolynomialBinding<F> for Rep3MultilinearPolynomial<F> {
    fn is_bound(&self) -> bool {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly.is_bound(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.is_bound(),
        }
    }

    fn bind(&mut self, r: F, order: BindingOrder) {
        match self {
            Rep3MultilinearPolynomial::Public {
                poly,
                trivial_share,
            } => {
                poly.bind(r, order);
                if let Some(trivial_share) = trivial_share {
                    trivial_share.bind(r, order);
                }
            }
            Rep3MultilinearPolynomial::Shared(poly) => poly.bind(r, order),
        }
    }

    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        match self {
            Rep3MultilinearPolynomial::Public {
                poly,
                trivial_share,
            } => {
                poly.bind_parallel(r, order);
                if let Some(trivial_share) = trivial_share {
                    trivial_share.bind_parallel(r, order);
                }
            }
            Rep3MultilinearPolynomial::Shared(poly) => poly.bind_parallel(r, order),
        }
    }

    /// Warning: when poly is shared, returns the additive share.
    /// Use `final_sumcheck_claim_additive` instead.
    fn final_sumcheck_claim(&self) -> F {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly.final_sumcheck_claim(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.final_sumcheck_claim(),
        }
    }
}

impl<F: JoltField> PolynomialEvaluation<F, SharedOrPublic<F>> for Rep3MultilinearPolynomial<F> {
    fn evaluate(&self, r: &[F]) -> SharedOrPublic<F> {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly.evaluate(r).into(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.evaluate(r).into(),
        }
    }

    #[tracing::instrument(skip_all, name = "Rep3MultilinearPolynomial::batch_evaluate")]
    fn batch_evaluate(polys: &[&Self], r: &[F]) -> (Vec<SharedOrPublic<F>>, Vec<F>) {
        let eq = EqPolynomial::evals(r);

        let evals: Vec<_> = polys
            .into_par_iter()
            .map(|&poly| match poly {
                Rep3MultilinearPolynomial::Public {
                    poly: MultilinearPolynomial::LargeScalars(poly),
                    ..
                } => SharedOrPublic::Public(poly.evaluate_at_chi_low_optimized(&eq)),
                Rep3MultilinearPolynomial::Public { poly, .. } => {
                    SharedOrPublic::Public(poly.dot_product(&eq))
                }
                Rep3MultilinearPolynomial::Shared(poly) => {
                    SharedOrPublic::Additive(poly.evaluate_at_chi(&eq))
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
    ) -> Vec<SharedOrPublic<F>> {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly
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

impl<F: JoltField> PolyDegree for Rep3MultilinearPolynomial<F> {
    fn len(&self) -> usize {
        self.len()
    }

    fn get_num_vars(&self) -> usize {
        self.get_num_vars()
    }
}

impl<'a, F: JoltField> TryInto<&'a Rep3DensePolynomial<F>> for &'a Rep3MultilinearPolynomial<F> {
    type Error = eyre::Error;

    fn try_into(self) -> Result<&'a Rep3DensePolynomial<F>, Self::Error> {
        match self {
            Rep3MultilinearPolynomial::Public { trivial_share, .. } => trivial_share
                .as_ref()
                .ok_or(eyre::eyre!("No trivial share")),
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
            Rep3MultilinearPolynomial::Public { trivial_share, .. } => trivial_share
                .as_mut()
                .ok_or(eyre::eyre!("No trivial share")),
            Rep3MultilinearPolynomial::Shared(poly) => Ok(poly),
        }
    }
}

impl<F: JoltField> TryInto<Rep3DensePolynomial<F>> for Rep3MultilinearPolynomial<F> {
    type Error = eyre::Error;

    fn try_into(self) -> Result<Rep3DensePolynomial<F>, Self::Error> {
        match self {
            Rep3MultilinearPolynomial::Public { trivial_share, .. } => {
                trivial_share.ok_or(eyre::eyre!("No trivial share"))
            }
            Rep3MultilinearPolynomial::Shared(poly) => Ok(poly),
        }
    }
}

impl<'a, F: JoltField> TryInto<&'a MultilinearPolynomial<F>> for &'a Rep3MultilinearPolynomial<F> {
    type Error = eyre::Error;

    fn try_into(self) -> Result<&'a MultilinearPolynomial<F>, Self::Error> {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => Ok(poly),
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
            Rep3MultilinearPolynomial::Public { poly, .. } => Ok(poly),
            Rep3MultilinearPolynomial::Shared(_) => Err(eyre::eyre!("No public polynomial")),
        }
    }
}

impl<F: JoltField> TryInto<MultilinearPolynomial<F>> for Rep3MultilinearPolynomial<F> {
    type Error = eyre::Error;

    fn try_into(self) -> Result<MultilinearPolynomial<F>, Self::Error> {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => Ok(poly),
            Rep3MultilinearPolynomial::Shared(_) => Err(eyre::eyre!("No public polynomial")),
        }
    }
}

impl<F: JoltField> From<MultilinearPolynomial<F>> for Rep3MultilinearPolynomial<F> {
    fn from(poly: MultilinearPolynomial<F>) -> Self {
        Rep3MultilinearPolynomial::Public {
            poly,
            trivial_share: None,
        }
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

impl<'a, F: JoltField> TryFrom<&'a Rep3MultilinearPolynomial<F>> for &'a DensePolynomial<F> {
    type Error = eyre::Error;

    fn try_from(poly: &'a Rep3MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly
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
            Rep3MultilinearPolynomial::Public { poly, .. } => poly
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
            Rep3MultilinearPolynomial::Public { poly, .. } => poly
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
            Rep3MultilinearPolynomial::Public { poly, .. } => poly
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
            Rep3MultilinearPolynomial::Public { poly, .. } => poly
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
            Rep3MultilinearPolynomial::Public { poly, .. } => poly
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

impl<F: JoltField> CanonicalSerialize for Rep3MultilinearPolynomial<F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Rep3MultilinearPolynomial::Public {
                poly,
                trivial_share,
            } => {
                (0_u8).serialize_with_mode(&mut writer, compress)?;
                poly.serialize_with_mode(&mut writer, compress)?;
                trivial_share.serialize_with_mode(&mut writer, compress)?;
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
            Rep3MultilinearPolynomial::Public {
                poly,
                trivial_share,
            } => {
                (0_u8).serialized_size(compress)
                    + poly.serialized_size(compress)
                    + trivial_share.serialized_size(compress)
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
        let res = match discriminant {
            0 => Rep3MultilinearPolynomial::Public {
                poly: MultilinearPolynomial::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?,
                trivial_share: Option::<Rep3DensePolynomial<F>>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?,
            },
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
            Rep3MultilinearPolynomial::Public {
                poly,
                trivial_share,
            } => {
                poly.check()?;
                trivial_share.check()
            }
            Rep3MultilinearPolynomial::Shared(poly) => poly.check(),
        }
    }
}
