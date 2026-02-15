use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::utils::types::Rep3Value;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial};
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};

use rayon::prelude::*;

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
            Rep3MultilinearPolynomial::Public(poly) => poly.len(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.full_len(),
        }
    }

    pub fn get_num_vars(&self) -> usize {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => poly.get_num_vars(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.get_num_vars(),
        }
    }

    pub fn is_bound(&self) -> bool {
        match self {
            Rep3MultilinearPolynomial::Public(_poly) => {
                // TODO: delegate via PolynomialBinding<F> trait
                false
            }
            Rep3MultilinearPolynomial::Shared(poly) => poly.is_bound(),
        }
    }

    pub fn bind(&mut self, r: F, order: BindingOrder) {
        match self {
            Rep3MultilinearPolynomial::Public(_poly) => {
                // TODO: delegate via PolynomialBinding<F> trait (takes F::Challenge)
                todo!("bind for public polynomial requires Challenge type")
            }
            Rep3MultilinearPolynomial::Shared(poly) => poly.bind(r, order),
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
            Rep3MultilinearPolynomial::Public(_poly) => {
                // TODO: delegate via PolynomialEvaluation<F> trait
                todo!("sumcheck_evals for public polynomial requires Challenge type")
            }
            Rep3MultilinearPolynomial::Shared(poly) => poly.sumcheck_evals(index, degree, order),
        }
    }

    #[tracing::instrument(skip_all, name = "Rep3MultilinearPolynomial::batch_evaluate_at_chi")]
    pub fn batch_evaluate_at_chi(polys: &[&Self], chi: &[F]) -> Vec<Rep3Value<F>> {
        let evals: Vec<_> = polys
            .into_par_iter()
            .map(|&poly| match poly {
                Rep3MultilinearPolynomial::Public(poly) => Rep3Value::Public(poly.dot_product(chi)),
                Rep3MultilinearPolynomial::Shared(poly) => {
                    Rep3Value::Additive(poly.evaluate_at_chi_optimized(chi))
                }
            })
            .collect();
        evals
    }

    pub fn batch_evaluate(
        polys: &[&Rep3MultilinearPolynomial<F>],
        r: &[F],
    ) -> (Vec<Rep3Value<F>>, Vec<F>) {
        let eq = EqPolynomial::evals(r);

        let evals: Vec<_> = polys
            .into_par_iter()
            .map(|&poly| match poly {
                Rep3MultilinearPolynomial::Public(poly) => Rep3Value::Public(poly.dot_product(&eq)),
                Rep3MultilinearPolynomial::Shared(poly) => {
                    Rep3Value::Additive(poly.evaluate_at_chi_optimized(&eq))
                }
            })
            .collect();
        (evals, eq)
    }

    pub fn evaluate(&self, r: &[F]) -> Rep3Value<F> {
        match self {
            Rep3MultilinearPolynomial::Public(poly) => {
                // Use dot_product with EqPolynomial evals as a workaround
                let eq = EqPolynomial::evals(r);
                Rep3Value::Public(poly.dot_product(&eq))
            }
            Rep3MultilinearPolynomial::Shared(poly) => poly.evaluate(r).into(),
        }
    }

    pub fn final_sumcheck_claim(&self) -> Rep3Value<F> {
        match self {
            Rep3MultilinearPolynomial::Public(_poly) => {
                // TODO: delegate via PolynomialBinding<F> trait
                todo!("final_sumcheck_claim for public polynomial requires Challenge type")
            }
            Rep3MultilinearPolynomial::Shared(poly) => poly.final_sumcheck_claim().into(),
        }
    }

    pub fn sumcheck_evals(
        &self,
        index: usize,
        degree: usize,
        order: BindingOrder,
    ) -> Vec<Rep3Value<F>> {
        match self {
            Rep3MultilinearPolynomial::Public(_poly) => {
                // TODO: delegate via PolynomialEvaluation<F> trait
                todo!("sumcheck_evals for public polynomial requires Challenge type")
            }
            Rep3MultilinearPolynomial::Shared(poly) => poly
                .sumcheck_evals(index, degree, order)
                .into_iter()
                .map(|x| x.into())
                .collect(),
        }
    }
}

// Note: Polynomial<F> trait not implemented here due to Mul<&Self> ambiguity
// between jolt_core::field::JoltField and PrimeField. Use inherent methods instead.
impl<F: JoltField> Rep3MultilinearPolynomial<F> {
    pub fn get_bound_coeffs(&self) -> Vec<Rep3Value<F>> {
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

// //---------------- Serialization ----------------//

// impl<F: JoltField> CanonicalSerialize for Rep3MultilinearPolynomial<F> {
//     fn serialize_with_mode<W: std::io::Write>(
//         &self,
//         mut writer: W,
//         compress: Compress,
//     ) -> Result<(), SerializationError> {
//         match self {
//             Rep3MultilinearPolynomial::Public(poly) => {
//                 (0_u8).serialize_with_mode(&mut writer, compress)?;
//                 poly.serialize_with_mode(&mut writer, compress)?;
//             }
//             Rep3MultilinearPolynomial::Shared(poly) => {
//                 (1_u8).serialize_with_mode(&mut writer, compress)?;
//                 poly.serialize_with_mode(&mut writer, compress)?;
//             }
//         }
//         Ok(())
//     }

//     fn serialized_size(&self, compress: Compress) -> usize {
//         match self {
//             Rep3MultilinearPolynomial::Public(poly) => {
//                 (0_u8).serialized_size(compress) + poly.serialized_size(compress)
//             }
//             Rep3MultilinearPolynomial::Shared(poly) => {
//                 (1_u8).serialized_size(compress) + poly.serialized_size(compress)
//             }
//         }
//     }
// }

// impl<F: jolt_core::field::JoltField> CanonicalDeserialize for Rep3MultilinearPolynomial<F> {
//     fn deserialize_with_mode<R: std::io::Read>(
//         mut reader: R,
//         compress: Compress,
//         validate: Validate,
//     ) -> Result<Self, SerializationError> {
//         let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
//         let res =
//             match discriminant {
//                 0 => Rep3MultilinearPolynomial::Public(
//                     MultilinearPolynomial::deserialize_with_mode(&mut reader, compress, validate)?,
//                 ),
//                 1 => Rep3MultilinearPolynomial::Shared(Rep3DensePolynomial::deserialize_with_mode(
//                     &mut reader,
//                     compress,
//                     validate,
//                 )?),
//                 _ => Err(SerializationError::InvalidData)?,
//             };
//         Ok(res)
//     }
// }

// impl<F: jolt_core::field::JoltField> Valid for Rep3MultilinearPolynomial<F> {
//     fn check(&self) -> Result<(), SerializationError> {
//         Ok(())
//     }
// }
