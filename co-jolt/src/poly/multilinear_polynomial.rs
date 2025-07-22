use std::ops::Index;

use crate::poly::rep3_poly::Rep3DensePolynomial;
use crate::utils::element::SharedOrPublic;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use jolt_core::{field::JoltField, poly::compact_polynomial::CompactPolynomial};
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};
use rayon::iter::IntoParallelIterator;

// pub type MultilinearPolynomial<F> = DensePolynomial<F>;

#[derive(Debug, Clone)]
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

    pub fn as_public(&self) -> &MultilinearPolynomial<F> {
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

    #[inline]
    pub fn sumcheck_evals(
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
    fn final_sumcheck_claim(&self) -> F {
        match self {
            Rep3MultilinearPolynomial::Public { poly, .. } => poly.final_sumcheck_claim(),
            Rep3MultilinearPolynomial::Shared(poly) => poly.final_sumcheck_claim(),
        }
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
