use jolt_core::field::JoltField;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedOrPublic<F: JoltField> {
    Public(F),
    Shared(Rep3PrimeFieldShare<F>),
}

impl<F: JoltField> SharedOrPublic<F> {
    pub fn try_into_public(self) -> eyre::Result<F> {
        match self {
            SharedOrPublic::Public(x) => Ok(x),
            SharedOrPublic::Shared(x) => Err(eyre::eyre!("Not a public field element")),
        }
    }

    pub fn as_shared(&self) -> &Rep3PrimeFieldShare<F> {
        match self {
            SharedOrPublic::Public(_) => panic!("Not an arithmetic share"),
            SharedOrPublic::Shared(x) => x,
        }
    }

    pub fn as_shared_mut(&mut self) -> &mut Rep3PrimeFieldShare<F> {
        match self {
            SharedOrPublic::Public(_) => panic!("Not an arithmetic share"),
            SharedOrPublic::Shared(x) => x,
        }
    }

    pub fn as_public(&self) -> &F {
        match self {
            SharedOrPublic::Public(x) => x,
            SharedOrPublic::Shared(_) => panic!("Not a public field element"),
        }
    }

    pub fn as_public_mut(&mut self) -> &mut F {
        match self {
            SharedOrPublic::Public(x) => x,
            SharedOrPublic::Shared(_) => panic!("Not a public field element"),
        }
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
        }
    }
}
