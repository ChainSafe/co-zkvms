use crate::field::JoltField;
use ark_ff::Zero;
use eyre::Context;
use mpc_core::protocols::{
    additive::{self, AdditiveShare},
    rep3::{
        self,
        network::{IoContext, Rep3Network},
        PartyID, Rep3PrimeFieldShare,
    },
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Either<Pub, Share> {
    Public(Pub),
    Shared(Share),
}

impl<Pub, Share> From<Pub> for Either<Pub, Share> {
    fn from(public: Pub) -> Self {
        Either::Public(public)
    }
}

/// Stores and implements interation between different value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rep3Value<F: JoltField> {
    Public(F),
    Shared(Rep3PrimeFieldShare<F>),
    Additive(AdditiveShare<F>),
}

impl<F: JoltField> Rep3Value<F> {
    pub fn zero_public() -> Self {
        Rep3Value::Public(F::ZERO)
    }

    pub fn zero_share() -> Self {
        Rep3Value::Shared(Rep3PrimeFieldShare::<F>::zero_share())
    }

    pub fn zero_additive() -> Self {
        Rep3Value::Additive(AdditiveShare::<F>::zero())
    }

    pub fn try_into_public(self) -> eyre::Result<F> {
        match self {
            Rep3Value::Public(x) => Ok(x),
            Rep3Value::Shared(_) => Err(eyre::eyre!("Not a public field element")),
            Rep3Value::Additive(_) => Err(eyre::eyre!("Not a public field element")),
        }
    }

    pub fn as_additive(self) -> AdditiveShare<F> {
        match self {
            Rep3Value::Public(_) => panic!("Not an additive share"),
            Rep3Value::Shared(_) => panic!("Not an additive share"),
            Rep3Value::Additive(x) => x,
        }
    }

    pub fn as_shared(self) -> Rep3PrimeFieldShare<F> {
        match self {
            Rep3Value::Public(_) => panic!("Not an arithmetic share"),
            Rep3Value::Shared(x) => x,
            Rep3Value::Additive(_) => panic!("Not an rep3 share"),
        }
    }

    pub fn as_shared_ref(&self) -> &Rep3PrimeFieldShare<F> {
        match self {
            Rep3Value::Public(_) => panic!("Not an arithmetic share"),
            Rep3Value::Shared(x) => x,
            Rep3Value::Additive(_) => panic!("Not an rep3 share"),
        }
    }

    pub fn as_shared_mut(&mut self) -> &mut Rep3PrimeFieldShare<F> {
        match self {
            Rep3Value::Public(_) => panic!("Not an arithmetic share"),
            Rep3Value::Shared(x) => x,
            Rep3Value::Additive(_) => panic!("Not an rep3 share"),
        }
    }

    pub fn as_public(self) -> F {
        match self {
            Rep3Value::Public(x) => x,
            _ => panic!("Not a public field element"),
        }
    }

    pub fn as_public_ref(&self) -> &F {
        match self {
            Rep3Value::Public(x) => x,
            _ => panic!("Not a public field element"),
        }
    }

    pub fn as_public_mut(&mut self) -> &mut F {
        match self {
            Rep3Value::Public(x) => x,
            _ => panic!("Not a public field element"),
        }
    }

    pub fn into_additive(self, party_id: PartyID) -> AdditiveShare<F> {
        match self {
            Rep3Value::Public(x) => additive::promote_to_trivial_share(x, party_id),
            Rep3Value::Shared(x) => x.into_additive(),
            Rep3Value::Additive(x) => x,
        }
    }

    pub fn add(&self, other: &Self, party_id: PartyID) -> Self {
        match (self, other) {
            (Rep3Value::Shared(x), Rep3Value::Shared(y)) => Rep3Value::Shared(x + y),
            (Rep3Value::Shared(x), Rep3Value::Public(y)) => {
                Rep3Value::Shared(rep3::arithmetic::add_public(*x, *y, party_id))
            }
            (Rep3Value::Public(x), Rep3Value::Shared(y)) => {
                Rep3Value::Shared(rep3::arithmetic::add_public(*y, *x, party_id))
            }
            (Rep3Value::Public(x), Rep3Value::Public(y)) => Rep3Value::Public(*x + *y),
            (Rep3Value::Additive(x), Rep3Value::Additive(y)) => Rep3Value::Additive(*x + *y),
            (Rep3Value::Additive(x), Rep3Value::Public(y)) => {
                Rep3Value::Additive(additive::add_public(*x, *y, party_id))
            }
            (Rep3Value::Public(x), Rep3Value::Additive(y)) => {
                Rep3Value::Additive(additive::add_public(*y, *x, party_id))
            }
            (Rep3Value::Additive(x), Rep3Value::Shared(y)) => {
                Rep3Value::Additive(*x + y.into_additive())
            }
            (Rep3Value::Shared(x), Rep3Value::Additive(y)) => {
                Rep3Value::Additive(x.into_additive() + *y)
            }
        }
    }

    pub fn add_assign(&mut self, other: &Self, party_id: PartyID) {
        *self = self.add(other, party_id);
    }

    pub fn add_public(&self, other: F, party_id: PartyID) -> Self {
        match self {
            Rep3Value::Shared(x) => {
                Rep3Value::Shared(rep3::arithmetic::add_public(*x, other, party_id))
            }
            Rep3Value::Public(x) => Rep3Value::Public(*x + other),
            Rep3Value::Additive(x) => {
                Rep3Value::Additive(additive::add_public(*x, other, party_id))
            }
        }
    }

    pub fn add_public_assign(&mut self, other: F, party_id: PartyID) {
        *self = self.add_public(other, party_id);
    }

    pub fn add_shared(&self, other: Rep3PrimeFieldShare<F>, party_id: PartyID) -> Self {
        match self {
            Rep3Value::Shared(x) => Rep3Value::Shared(*x + other),
            Rep3Value::Public(x) => {
                Rep3Value::Shared(rep3::arithmetic::add_public(other, *x, party_id))
            }
            Rep3Value::Additive(_) => {
                panic!("Addition of rep3 and additive shares are not allowed")
            }
        }
    }

    pub fn add_shared_assign(&mut self, other: Rep3PrimeFieldShare<F>, party_id: PartyID) {
        *self = self.add_shared(other, party_id);
    }

    pub fn sub(&self, other: &Self, party_id: PartyID) -> Self {
        match (self, other) {
            (Rep3Value::Shared(x), Rep3Value::Shared(y)) => Rep3Value::Shared(x - y),
            (Rep3Value::Shared(x), Rep3Value::Public(y)) => {
                Rep3Value::Shared(rep3::arithmetic::sub_shared_by_public(*x, *y, party_id))
            }
            (Rep3Value::Public(x), Rep3Value::Shared(y)) => {
                Rep3Value::Shared(rep3::arithmetic::sub_public_by_shared(*x, *y, party_id))
            }
            (Rep3Value::Public(x), Rep3Value::Public(y)) => Rep3Value::Public(*x - *y),
            (Rep3Value::Additive(x), Rep3Value::Additive(y)) => Rep3Value::Additive(*x - *y),
            (Rep3Value::Additive(x), Rep3Value::Public(y)) => {
                Rep3Value::Additive(additive::sub_shared_by_public(*x, *y, party_id))
            }
            (Rep3Value::Public(x), Rep3Value::Additive(y)) => {
                Rep3Value::Additive(additive::sub_public_by_shared(*x, *y, party_id))
            }
            (Rep3Value::Additive(x), Rep3Value::Shared(y)) => {
                Rep3Value::Additive(*x - y.into_additive())
            }
            (Rep3Value::Shared(x), Rep3Value::Additive(y)) => {
                Rep3Value::Additive(x.into_additive() - *y)
            }
        }
    }

    pub fn sub_public(&self, other: &F, party_id: PartyID) -> Self {
        match self {
            Rep3Value::Shared(x) => {
                Rep3Value::Shared(rep3::arithmetic::sub_shared_by_public(*x, *other, party_id))
            }
            Rep3Value::Public(x) => Rep3Value::Public(*x - *other),
            Rep3Value::Additive(x) => {
                Rep3Value::Additive(additive::sub_public_by_shared(*other, *x, party_id))
            }
        }
    }

    pub fn mul_reshare<Network>(
        &self,
        other: &Self,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Network: Rep3Network,
    {
        Ok(match (self, other) {
            (Rep3Value::Shared(x), Rep3Value::Shared(y)) => Rep3Value::Shared(
                rep3::arithmetic::mul(*x, *y, io_ctx)
                    .context("Shared and shared multiplication failed")?,
            ),
            (_, _) => self.mul(other),
        })
    }

    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Rep3Value::Public(x), Rep3Value::Public(y)) => Rep3Value::Public(*x * *y),
            (Rep3Value::Shared(x), Rep3Value::Public(y)) => {
                Rep3Value::Shared(rep3::arithmetic::mul_public(*x, *y))
            }
            (Rep3Value::Public(x), Rep3Value::Shared(y)) => {
                Rep3Value::Shared(rep3::arithmetic::mul_public(*y, *x))
            }
            (Rep3Value::Shared(x), Rep3Value::Shared(y)) => Rep3Value::Additive(*x * *y),
            (Rep3Value::Additive(x), Rep3Value::Public(y)) => Rep3Value::Additive(*x * *y),
            (Rep3Value::Public(x), Rep3Value::Additive(y)) => Rep3Value::Additive(*y * *x),
            _ => panic!("Multiplication of additive shares are not allowed"),
        }
    }

    pub fn mul_public(&self, other: F) -> Self {
        self.mul(&other.into())
    }

    pub fn mul_mul_public(&self, other: &Self, public: F) -> Self {
        match (self, other) {
            (Rep3Value::Shared(x), Rep3Value::Shared(y)) => Rep3Value::Additive(*x * *y * public),
            (Rep3Value::Additive(x), Rep3Value::Public(y)) => {
                Rep3Value::Additive(*x * (*y * public))
            }
            (Rep3Value::Public(x), Rep3Value::Additive(y)) => {
                Rep3Value::Additive(*y * (*x * public))
            }
            (Rep3Value::Shared(x), Rep3Value::Public(y)) => {
                Rep3Value::Additive(x.into_additive() * (*y * public))
            }
            (Rep3Value::Public(x), Rep3Value::Shared(y)) => {
                Rep3Value::Additive(y.into_additive() * (*x * public))
            }
            _ => self.mul(&other.mul(&public.into())),
        }
    }

    pub fn shared_or_not_zero(&self) -> bool {
        match self {
            Rep3Value::Public(x) => !x.is_zero(),
            Rep3Value::Shared(_) => true,
            Rep3Value::Additive(_) => true,
        }
    }
}

pub trait SharedOrPublicIter<F: JoltField> {
    fn sum_for(self, party_id: PartyID) -> Rep3Value<F>;
}

impl<F: JoltField, I> SharedOrPublicIter<F> for I
where
    I: IntoIterator<Item = Rep3Value<F>>,
{
    fn sum_for(self, party_id: PartyID) -> Rep3Value<F> {
        self.into_iter()
            .fold(Rep3Value::Public(F::ZERO), |acc, x| acc.add(&x, party_id))
    }
}

pub trait SharedOrPublicParIter<F: JoltField> {
    fn sum_for(self, party_id: PartyID) -> Rep3Value<F>;
}

impl<F: JoltField, I> SharedOrPublicParIter<F> for I
where
    I: IntoParallelIterator<Item = Rep3Value<F>>,
{
    fn sum_for(self, party_id: PartyID) -> Rep3Value<F> {
        self.into_par_iter().reduce(
            || Rep3Value::Public(F::ZERO),
            |acc, x| acc.add(&x, party_id),
        )
    }
}

impl<F: JoltField> From<F> for Rep3Value<F> {
    fn from(value: F) -> Self {
        Rep3Value::Public(value)
    }
}

impl<F: JoltField> From<AdditiveShare<F>> for Rep3Value<F> {
    fn from(value: AdditiveShare<F>) -> Self {
        Rep3Value::Additive(value)
    }
}

impl<F: JoltField> From<Rep3PrimeFieldShare<F>> for Rep3Value<F> {
    fn from(value: Rep3PrimeFieldShare<F>) -> Self {
        Rep3Value::Shared(value)
    }
}

impl<F: JoltField> TryInto<Rep3PrimeFieldShare<F>> for Rep3Value<F> {
    type Error = eyre::Error;

    fn try_into(self) -> Result<Rep3PrimeFieldShare<F>, Self::Error> {
        match self {
            Rep3Value::Public(_) => Err(eyre::eyre!("Not an arithmetic share")),
            Rep3Value::Shared(x) => Ok(x),
            Rep3Value::Additive(_) => Err(eyre::eyre!("Not a rep3 share")),
        }
    }
}

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaybeShared<U> {
    Public(Option<U>),
    Shared(U),
}

impl<U> MaybeShared<U> {
    pub fn try_into_public_mut(&mut self) -> Option<&mut U> {
        match self {
            MaybeShared::Public(Some(inner)) => Some(inner),
            _ => None,
        }
    }
}

impl<U> Default for MaybeShared<U>
where
    U: CanonicalSerialize + CanonicalDeserialize + Default + Sync,
{
    fn default() -> Self {
        MaybeShared::Public(None)
    }
}

impl<U> CanonicalSerialize for MaybeShared<U>
where
    U: CanonicalSerialize + CanonicalDeserialize + Default + Sync,
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            MaybeShared::Public(inner) => {
                (0_u8).serialize_with_mode(&mut writer, compress)?;
                inner.serialize_with_mode(&mut writer, compress)?;
            }
            MaybeShared::Shared(inner) => {
                (1_u8).serialize_with_mode(&mut writer, compress)?;
                inner.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            MaybeShared::Public(inner) => {
                (0_u8).serialized_size(compress) + inner.serialized_size(compress)
            }
            MaybeShared::Shared(inner) => {
                (1_u8).serialized_size(compress) + inner.serialized_size(compress)
            }
        }
    }
}

impl<U> CanonicalDeserialize for MaybeShared<U>
where
    U: CanonicalSerialize + CanonicalDeserialize + Default + Sync,
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        // TODO(protoben) Can we use strum for this?
        let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let res = match discriminant {
            0 => MaybeShared::Public(Option::<U>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?),
            1 => MaybeShared::Shared(U::deserialize_with_mode(&mut reader, compress, validate)?),
            _ => Err(SerializationError::InvalidData)?,
        };
        Ok(res)
    }
}

impl<U> Valid for MaybeShared<U>
where
    U: CanonicalSerialize + CanonicalDeserialize + Default + Sync,
{
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            MaybeShared::Public(inner) => inner.check(),
            MaybeShared::Shared(inner) => inner.check(),
        }
    }
}
