use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::casts::downcast;
use mpc_core::protocols::rep3_ring::ring::int_ring::IntRing2k;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Rep3Operand {
    Shared {
        binary: Rep3RingShare<u64>,
        arithmetic: Option<Rep3RingShare<u128>>,
        public: Option<u64>, // Some for trivial shares
    },
    Public(u64),
}

impl Rep3Operand {
    pub fn from_binary(share: Rep3RingShare<u64>) -> Self {
        Rep3Operand::Shared {
            binary: share,
            arithmetic: None,
            public: None,
        }
    }

    pub fn from_arithmetic(binary: Rep3RingShare<u64>, arithmetic: Rep3RingShare<u128>) -> Self {
        Rep3Operand::Shared {
            binary,
            arithmetic: Some(arithmetic),
            public: None,
        }
    }

    pub fn as_public(&self) -> u64 {
        match self {
            Rep3Operand::Public(x)
            | Rep3Operand::Shared {
                public: Some(x), ..
            } => *x,
            _ => panic!("Not a public operand"),
        }
    }

    pub fn as_arithmetic<T: IntRing2k>(&self) -> Rep3RingShare<T>
    where
        u128: AsPrimitive<T>,
    {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_arithmetic_u32(&self) -> Rep3RingShare<u32> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_arithmetic_u64(&self) -> Rep3RingShare<u64> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_arithmetic_u128(&self) -> Rep3RingShare<u128> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_binary(&self) -> Rep3RingShare<u64> {
        match self {
            Rep3Operand::Shared { binary, .. } => binary.clone(),
            _ => panic!("Not a binary operand"),
        }
    }

    pub fn as_binary_or_trivial(&self, id: PartyID) -> Rep3RingShare<u64> {
        match *self {
            Rep3Operand::Shared { binary, .. } => binary,
            Rep3Operand::Public(value) => {
                rep3_ring::binary::promote_to_trivial_share(id, &value.into())
            }
        }
    }

    pub fn as_arithmetic_or_trivial_u128(&self, id: PartyID) -> Rep3RingShare<u128> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => arithmetic.unwrap(),
            Rep3Operand::Public(v) => {
                rep3_ring::arithmetic::promote_to_trivial_share(id, RingElement(*v as u128))
            }
        }
    }

    pub fn as_arithmetic_or_trivial<T: IntRing2k>(&self, id: PartyID) -> Rep3RingShare<T>
    where
        u128: AsPrimitive<T>,
    {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            Rep3Operand::Public(v) => {
                rep3_ring::arithmetic::promote_to_trivial_share(id, RingElement((*v as u128).as_()))
            }
        }
    }
}

/// Static zero operand for returning references from formats without certain registers.
pub static PUBLIC_ZERO: Rep3Operand = Rep3Operand::Public(0);

impl Default for Rep3Operand {
    fn default() -> Self {
        Rep3Operand::Public(0)
    }
}

impl From<u64> for Rep3Operand {
    fn from(value: u64) -> Self {
        Rep3Operand::Public(value)
    }
}

impl From<u32> for Rep3Operand {
    fn from(value: u32) -> Self {
        Rep3Operand::Public(value as u64)
    }
}

impl From<Rep3Operand> for u64 {
    fn from(value: Rep3Operand) -> u64 {
        match value {
            Rep3Operand::Public(x) => x,
            _ => panic!("Cannot convert Rep3Operand to u64"),
        }
    }
}

impl From<Rep3Operand> for u32 {
    fn from(value: Rep3Operand) -> u32 {
        match value {
            Rep3Operand::Public(x) => x as u32,
            _ => panic!("Cannot convert Rep3Operand to u32"),
        }
    }
}

/// Convert a public Rep3Operand to a shared operand with trivial shares.
pub fn promote_operand_to_share(operand: &Rep3Operand, party_id: PartyID) -> Rep3Operand {
    match operand {
        Rep3Operand::Public(x) => Rep3Operand::Shared {
            binary: rep3_ring::binary::promote_to_trivial_share(party_id, &RingElement(*x)),
            arithmetic: Some(rep3_ring::arithmetic::promote_to_trivial_share(
                party_id,
                RingElement(*x as u128),
            )),
            public: Some(*x),
        },
        already_shared @ Rep3Operand::Shared { .. } => already_shared.clone(),
    }
}
