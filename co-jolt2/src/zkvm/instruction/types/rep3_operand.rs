use jolt2_common::constants::{ArithmeticWideInt, XlenInt};
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
        binary: Rep3RingShare<XlenInt>,
        arithmetic: Option<Rep3RingShare<ArithmeticWideInt>>,
        public: Option<u64>, // Some for trivial shares
    },
    /// Public (plaintext) operand, stored as signed i128.
    ///
    /// Stores both unsigned register values (0..2^64, via `u64 as i128`) and
    /// signed immediates (-2^63..2^63, via `i64 as i128`). The `as u64` casts
    /// in ring-domain helpers give the correct two's complement bit pattern
    /// for both cases. Values outside [-2^63, 2^64) would silently truncate.
    Public(i128),
}

impl Rep3Operand {
    pub fn from_binary(share: Rep3RingShare<XlenInt>) -> Self {
        Rep3Operand::Shared {
            binary: share,
            arithmetic: None,
            public: None,
        }
    }

    pub fn from_arithmetic(
        binary: Rep3RingShare<XlenInt>,
        arithmetic: Rep3RingShare<ArithmeticWideInt>,
    ) -> Self {
        Rep3Operand::Shared {
            binary,
            arithmetic: Some(arithmetic),
            public: None,
        }
    }

    pub fn as_public(&self) -> u64 {
        match self {
            Rep3Operand::Public(x) => *x as u64,
            Rep3Operand::Shared {
                public: Some(x), ..
            } => *x,
            _ => panic!("Not a public operand"),
        }
    }

    pub fn as_arithmetic<T: IntRing2k>(&self) -> Rep3RingShare<T>
    where
        ArithmeticWideInt: AsPrimitive<T>,
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

    pub fn as_arithmetic_wide(&self) -> Rep3RingShare<ArithmeticWideInt> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => arithmetic.unwrap(),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_binary(&self) -> Rep3RingShare<XlenInt> {
        match self {
            Rep3Operand::Shared { binary, .. } => binary.clone(),
            _ => panic!("Not a binary operand"),
        }
    }

    pub fn as_binary_or_trivial(&self, id: PartyID) -> Rep3RingShare<XlenInt> {
        match *self {
            Rep3Operand::Shared { binary, .. } => binary,
            Rep3Operand::Public(value) => {
                rep3_ring::binary::promote_to_trivial_share(id, &RingElement(value as XlenInt))
            }
        }
    }

    pub fn as_arithmetic_or_trivial_wide(&self, id: PartyID) -> Rep3RingShare<ArithmeticWideInt> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => arithmetic.unwrap(),
            // Truncate to XlenInt first to match vanilla Jolt's `val as u32/u64`
            // truncation. For rv32, this drops the upper 32 bits of sign-extended imms.
            Rep3Operand::Public(v) => rep3_ring::arithmetic::promote_to_trivial_share(
                id,
                RingElement(*v as XlenInt as ArithmeticWideInt),
            ),
        }
    }

    pub fn as_arithmetic_or_trivial<T: IntRing2k>(&self, id: PartyID) -> Rep3RingShare<T>
    where
        ArithmeticWideInt: AsPrimitive<T>,
    {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            Rep3Operand::Public(v) => rep3_ring::arithmetic::promote_to_trivial_share(
                id,
                RingElement((*v as XlenInt as ArithmeticWideInt).as_()),
            ),
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
        Rep3Operand::Public(value as i128)
    }
}

impl From<u32> for Rep3Operand {
    fn from(value: u32) -> Self {
        Rep3Operand::Public(value as i128)
    }
}

impl From<Rep3Operand> for u64 {
    fn from(value: Rep3Operand) -> u64 {
        match value {
            Rep3Operand::Public(x) => x as u64,
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
