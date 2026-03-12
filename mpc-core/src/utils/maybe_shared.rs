use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate};

#[derive(Clone, Debug)]
pub enum MaybeShared<T> {
    /// Public data is known to all parties. `None` is used when the caller explicitly
    /// requests not to commit to public polynomials (to save work).
    Public(Option<T>),
    /// Shared data is secret-shared across parties.
    Shared(T),
}

impl<T> Default for MaybeShared<T> {
    fn default() -> Self {
        Self::Public(None)
    }
}

impl<T> MaybeShared<T> {
    pub fn try_into_public_mut(&mut self) -> Option<&mut T> {
        match self {
            MaybeShared::Public(Some(v)) => Some(v),
            _ => None,
        }
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
            MaybeShared::Public(inner) => (0_u8).serialized_size(compress) + inner.serialized_size(compress),
            MaybeShared::Shared(inner) => (1_u8).serialized_size(compress) + inner.serialized_size(compress),
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
        let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let res = match discriminant {
            0 => MaybeShared::Public(Option::<U>::deserialize_with_mode(&mut reader, compress, validate)?),
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
