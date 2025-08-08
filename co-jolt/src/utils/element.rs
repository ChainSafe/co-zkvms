use eyre::Context;
use jolt_core::field::JoltField;
use mpc_core::protocols::{
    additive::{self, AdditiveShare},
    rep3::{
        self,
        network::{IoContext, Rep3Network},
        PartyID, Rep3PrimeFieldShare,
    },
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedOrPublic<F: JoltField> {
    Public(F),
    Shared(Rep3PrimeFieldShare<F>),
    Additive(AdditiveShare<F>),
}

impl<F: JoltField> SharedOrPublic<F> {
    pub fn zero_public() -> Self {
        SharedOrPublic::Public(F::ZERO)
    }

    pub fn zero_share() -> Self {
        SharedOrPublic::Shared(Rep3PrimeFieldShare::<F>::zero_share())
    }

    pub fn zero_additive() -> Self {
        SharedOrPublic::Additive(F::ZERO)
    }

    pub fn try_into_public(self) -> eyre::Result<F> {
        match self {
            SharedOrPublic::Public(x) => Ok(x),
            SharedOrPublic::Shared(_) => Err(eyre::eyre!("Not a public field element")),
            SharedOrPublic::Additive(_) => Err(eyre::eyre!("Not a public field element")),
        }
    }

    pub fn as_additive(self) -> AdditiveShare<F> {
        match self {
            SharedOrPublic::Public(_) => panic!("Not an additive share"),
            SharedOrPublic::Shared(_) => panic!("Not an additive share"),
            SharedOrPublic::Additive(x) => x,
        }
    }

    pub fn as_shared(self) -> Rep3PrimeFieldShare<F> {
        match self {
            SharedOrPublic::Public(_) => panic!("Not an arithmetic share"),
            SharedOrPublic::Shared(x) => x,
            SharedOrPublic::Additive(_) => panic!("Not an rep3 share"),
        }
    }

    pub fn as_shared_ref(&self) -> &Rep3PrimeFieldShare<F> {
        match self {
            SharedOrPublic::Public(_) => panic!("Not an arithmetic share"),
            SharedOrPublic::Shared(x) => x,
            SharedOrPublic::Additive(_) => panic!("Not an rep3 share"),
        }
    }

    pub fn as_shared_mut(&mut self) -> &mut Rep3PrimeFieldShare<F> {
        match self {
            SharedOrPublic::Public(_) => panic!("Not an arithmetic share"),
            SharedOrPublic::Shared(x) => x,
            SharedOrPublic::Additive(_) => panic!("Not an rep3 share"),
        }
    }

    pub fn as_public(self) -> F {
        match self {
            SharedOrPublic::Public(x) => x,
            _ => panic!("Not a public field element"),
        }
    }

    pub fn as_public_ref(&self) -> &F {
        match self {
            SharedOrPublic::Public(x) => x,
            _ => panic!("Not a public field element"),
        }
    }

    pub fn as_public_mut(&mut self) -> &mut F {
        match self {
            SharedOrPublic::Public(x) => x,
            _ => panic!("Not a public field element"),
        }
    }

    pub fn into_additive(self, party_id: PartyID) -> AdditiveShare<F> {
        match self {
            SharedOrPublic::Public(x) => additive::promote_to_trivial_share(x, party_id),
            SharedOrPublic::Shared(x) => x.into_additive(),
            SharedOrPublic::Additive(x) => x,
        }
    }

    pub fn add(&self, other: &Self, party_id: PartyID) -> Self {
        match (self, other) {
            (SharedOrPublic::Shared(x), SharedOrPublic::Shared(y)) => SharedOrPublic::Shared(x + y),
            (SharedOrPublic::Shared(x), SharedOrPublic::Public(y)) => {
                SharedOrPublic::Shared(rep3::arithmetic::add_public(*x, *y, party_id))
            }
            (SharedOrPublic::Public(x), SharedOrPublic::Shared(y)) => {
                SharedOrPublic::Shared(rep3::arithmetic::add_public(*y, *x, party_id))
            }
            (SharedOrPublic::Public(x), SharedOrPublic::Public(y)) => {
                SharedOrPublic::Public(*x + *y)
            }
            (SharedOrPublic::Additive(x), SharedOrPublic::Additive(y)) => {
                SharedOrPublic::Additive(*x + *y)
            }
            (SharedOrPublic::Additive(x), SharedOrPublic::Public(y)) => {
                SharedOrPublic::Additive(additive::add_public(*x, *y, party_id))
            }
            (SharedOrPublic::Public(x), SharedOrPublic::Additive(y)) => {
                SharedOrPublic::Additive(additive::add_public(*y, *x, party_id))
            }
            (SharedOrPublic::Additive(x), SharedOrPublic::Shared(y)) => {
                SharedOrPublic::Additive(*x + y.into_additive())
            }
            (SharedOrPublic::Shared(x), SharedOrPublic::Additive(y)) => {
                SharedOrPublic::Additive(x.into_additive() + *y)
            }
        }
    }

    pub fn add_assign(&mut self, other: &Self, party_id: PartyID) {
        *self = self.add(other, party_id);
    }

    pub fn add_public(&self, other: F, party_id: PartyID) -> Self {
        match self {
            SharedOrPublic::Shared(x) => {
                SharedOrPublic::Shared(rep3::arithmetic::add_public(*x, other, party_id))
            }
            SharedOrPublic::Public(x) => SharedOrPublic::Public(*x + other),
            SharedOrPublic::Additive(x) => {
                SharedOrPublic::Additive(additive::add_public(*x, other, party_id))
            }
        }
    }

    pub fn add_public_assign(&mut self, other: F, party_id: PartyID) {
        *self = self.add_public(other, party_id);
    }

    pub fn add_shared(&self, other: Rep3PrimeFieldShare<F>, party_id: PartyID) -> Self {
        match self {
            SharedOrPublic::Shared(x) => SharedOrPublic::Shared(*x + other),
            SharedOrPublic::Public(x) => {
                SharedOrPublic::Shared(rep3::arithmetic::add_public(other, *x, party_id))
            }
            SharedOrPublic::Additive(_) => {
                panic!("Addition of rep3 and additive shares are not allowed")
            }
        }
    }

    pub fn add_shared_assign(&mut self, other: Rep3PrimeFieldShare<F>, party_id: PartyID) {
        *self = self.add_shared(other, party_id);
    }

    pub fn sub(&self, other: &Self, party_id: PartyID) -> Self {
        match (self, other) {
            (SharedOrPublic::Shared(x), SharedOrPublic::Shared(y)) => SharedOrPublic::Shared(x - y),
            (SharedOrPublic::Shared(x), SharedOrPublic::Public(y)) => {
                SharedOrPublic::Shared(rep3::arithmetic::sub_shared_by_public(*x, *y, party_id))
            }
            (SharedOrPublic::Public(x), SharedOrPublic::Shared(y)) => {
                SharedOrPublic::Shared(rep3::arithmetic::sub_public_by_shared(*x, *y, party_id))
            }
            (SharedOrPublic::Public(x), SharedOrPublic::Public(y)) => {
                SharedOrPublic::Public(*x - *y)
            }
            (SharedOrPublic::Additive(x), SharedOrPublic::Additive(y)) => {
                SharedOrPublic::Additive(*x - *y)
            }
            (SharedOrPublic::Additive(x), SharedOrPublic::Public(y)) => {
                SharedOrPublic::Additive(additive::sub_shared_by_public(*x, *y, party_id))
            }
            (SharedOrPublic::Public(x), SharedOrPublic::Additive(y)) => {
                SharedOrPublic::Additive(additive::sub_public_by_shared(*x, *y, party_id))
            }
            (SharedOrPublic::Additive(x), SharedOrPublic::Shared(y)) => {
                SharedOrPublic::Additive(*x - y.into_additive())
            }
            (SharedOrPublic::Shared(x), SharedOrPublic::Additive(y)) => {
                SharedOrPublic::Additive(x.into_additive() - *y)
            }
        }
    }

    pub fn sub_public(&self, other: &F, party_id: PartyID) -> Self {
        match self {
            SharedOrPublic::Shared(x) => {
                SharedOrPublic::Shared(rep3::arithmetic::sub_shared_by_public(*x, *other, party_id))
            }
            SharedOrPublic::Public(x) => SharedOrPublic::Public(*x - *other),
            SharedOrPublic::Additive(x) => {
                SharedOrPublic::Additive(additive::sub_public_by_shared(*x, *other, party_id))
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
            (SharedOrPublic::Shared(x), SharedOrPublic::Shared(y)) => SharedOrPublic::Shared(
                rep3::arithmetic::mul(*x, *y, io_ctx)
                    .context("Shared and shared multiplication failed")?,
            ),
            (_, _) => self.mul(other),
        })
    }

    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (SharedOrPublic::Public(x), SharedOrPublic::Public(y)) => {
                SharedOrPublic::Public(*x * *y)
            }
            (SharedOrPublic::Shared(x), SharedOrPublic::Public(y)) => {
                SharedOrPublic::Shared(rep3::arithmetic::mul_public(*x, *y))
            }
            (SharedOrPublic::Public(x), SharedOrPublic::Shared(y)) => {
                SharedOrPublic::Shared(rep3::arithmetic::mul_public(*y, *x))
            }
            (SharedOrPublic::Shared(x), SharedOrPublic::Shared(y)) => {
                tracing::warn!("mul_shared");
                SharedOrPublic::Additive(x * y)
            }
            _ => panic!("Multiplication of additive shares are not allowed"),
        }
    }

    pub fn mul_public(&self, other: F) -> Self {
        self.mul(&other.into())
    }

    pub fn mul_mul_public(&self, other: &Self, public: F) -> Self {
        match (self, other) {
            (SharedOrPublic::Shared(x), SharedOrPublic::Shared(y)) => {
                SharedOrPublic::Additive(rep3::arithmetic::mul_mul_public(*x, *y, public))
            }
            _ => self.mul(&other.mul(&public.into())),
        }
    }

    pub fn shared_or_not_zero(&self) -> bool {
        match self {
            SharedOrPublic::Public(x) => !x.is_zero(),
            SharedOrPublic::Shared(_) => true,
            SharedOrPublic::Additive(_) => true,
        }
    }
}

pub trait SharedOrPublicIter<F: JoltField> {
    fn sum_for(self, party_id: PartyID) -> SharedOrPublic<F>;
}

impl<F: JoltField, I> SharedOrPublicIter<F> for I
where
    I: IntoIterator<Item = SharedOrPublic<F>>,
{
    fn sum_for(self, party_id: PartyID) -> SharedOrPublic<F> {
        self.into_iter()
            .fold(SharedOrPublic::Public(F::ZERO), |acc, x| {
                acc.add(&x, party_id)
            })
    }
}

pub trait SharedOrPublicParIter<F: JoltField> {
    fn sum_for(self, party_id: PartyID) -> SharedOrPublic<F>;
}

impl<F: JoltField, I> SharedOrPublicParIter<F> for I
where
    I: IntoParallelIterator<Item = SharedOrPublic<F>>,
{
    fn sum_for(self, party_id: PartyID) -> SharedOrPublic<F> {
        self.into_par_iter().reduce(
            || SharedOrPublic::Public(F::ZERO),
            |acc, x| acc.add(&x, party_id),
        )
    }
}

impl<F: JoltField> From<F> for SharedOrPublic<F> {
    fn from(value: F) -> Self {
        SharedOrPublic::Public(value)
    }
}

impl<F: JoltField> From<Rep3PrimeFieldShare<F>> for SharedOrPublic<F> {
    fn from(value: Rep3PrimeFieldShare<F>) -> Self {
        SharedOrPublic::Shared(value)
    }
}

impl<F: JoltField> TryInto<Rep3PrimeFieldShare<F>> for SharedOrPublic<F> {
    type Error = eyre::Error;

    fn try_into(self) -> Result<Rep3PrimeFieldShare<F>, Self::Error> {
        match self {
            SharedOrPublic::Public(_) => Err(eyre::eyre!("Not an arithmetic share")),
            SharedOrPublic::Shared(x) => Ok(x),
            SharedOrPublic::Additive(_) => Err(eyre::eyre!("Not a rep3 share")),
        }
    }
}

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};

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
