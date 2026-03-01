use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Either<Pub, Share> {
    Public(Pub),
    Shared(Share),
}

impl<Pub, Share> Either<Pub, Share> {
    pub fn as_public(&self) -> &Pub {
        match self {
            Either::Public(p) => p,
            Either::Shared(_) => panic!("Expected public"),
        }
    }

    pub fn as_shared(&self) -> &Share {
        match self {
            Either::Public(_) => panic!("Expected shared"),
            Either::Shared(s) => s,
        }
    }
}
