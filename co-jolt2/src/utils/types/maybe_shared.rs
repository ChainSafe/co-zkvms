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

