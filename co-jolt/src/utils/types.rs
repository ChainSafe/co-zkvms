mod either;
mod rep3_value;

pub use either::Either;
pub use rep3_value::{Rep3Value, SharedOrPublicIter, SharedOrPublicParIter};

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
