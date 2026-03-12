use crate::protocols::{
    rep3::PartyID,
    rep3_ring::ring::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement},
};
use num_traits::{AsPrimitive, Zero};
use serde::{Deserialize, Serialize};

/// This type represents a replicated shared value. Since a replicated share of a ring element contains additive shares of two parties, this type contains two ring elements.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Rep3RingShare<T: IntRing2k> {
    /// Share of this party
    pub a: RingElement<T>,
    /// Share of the prev party
    pub b: RingElement<T>,
}

impl<T: IntRing2k> Default for Rep3RingShare<T> {
    fn default() -> Self {
        Self::zero_share()
    }
}

impl<T: IntRing2k> Rep3RingShare<T> {
    /// Constructs the type from two additive shares.
    pub fn new(a: T, b: T) -> Self {
        Self {
            a: RingElement(a),
            b: RingElement(b),
        }
    }

    /// Constructs a new share from two ring elements
    pub fn new_ring(a: RingElement<T>, b: RingElement<T>) -> Self {
        Self { a, b }
    }

    /// Constructs a zero share.
    pub fn zero_share() -> Self {
        Self {
            a: RingElement::zero(),
            b: RingElement::zero(),
        }
    }

    /// Unwraps the type into two additive shares.
    pub fn ab(self) -> (RingElement<T>, RingElement<T>) {
        (self.a, self.b)
    }

    /// Double the share in place
    pub fn double(&mut self) {
        self.a <<= 1;
        self.b <<= 1;
    }

    /// Promotes a public ring element to a replicated share by setting the additive share of the party with id=0 and leaving all other shares to be 0. Thus, the replicated shares of party 0 and party 1 are set.
    pub fn promote_from_trivial(val: &RingElement<T>, id: PartyID) -> Self {
        match id {
            PartyID::ID0 => Self::new_ring(*val, RingElement::zero()),
            PartyID::ID1 => Self::new_ring(RingElement::zero(), *val),
            PartyID::ID2 => Self::zero_share(),
        }
    }

    /// Return the bit at position `index`.
    pub fn get_bit(&self, index: usize) -> Rep3RingShare<Bit> {
        Rep3RingShare {
            a: RingElement(Bit::new(self.a.get_bit(index).0 == T::one())),
            b: RingElement(Bit::new(self.b.get_bit(index).0 == T::one())),
        }
    }

    pub fn downcast<U: IntRing2k>(&self) -> Rep3RingShare<U>
    where
        T: AsPrimitive<U>,
        U: IntRing2k,
    {
        assert!(T::K >= U::K);

        Rep3RingShare {
            a: RingElement(self.a.0.as_()),
            b: RingElement(self.b.0.as_()),
        }
    }

    pub fn is_even(&self) -> Rep3RingShare<Bit> {
        !self.get_bit(0)
    }

    /// Converts the share to a vector of bytes in little-endian order, padding with zeros if necessary.
    pub fn to_le_bytes(&self) -> Vec<Rep3RingShare<u8>> {
        todo!()
    }

    /// Converts a vector of bits in little-endian order to a share.
    pub fn from_le_bytes(bytes: &[Rep3RingShare<u8>]) -> Self {
        assert!(bytes.len() <= T::K / 8);

        // let mut acc = Self::zero_share();
        // let mut i = 0;
        // for byte in bytes {
        //     acc |= *byte << (8 * i);
        //     i += 1;
        // }

        let mut a_bytes = vec![0u8; T::K / 8];
        let mut b_bytes = vec![0u8; T::K / 8];

        for (i, byte_share) in bytes.iter().enumerate() {
            a_bytes[i] = byte_share.a.0;
            b_bytes[i] = byte_share.b.0;
        }

        Self {
            a: RingElement(T::from_le_bytes(&a_bytes)),
            b: RingElement(T::from_le_bytes(&b_bytes)),
        }
    }
}

impl<T: IntRing2k> AsRef<Rep3RingShare<T>> for Rep3RingShare<T> {
    fn as_ref(&self) -> &Self {
        self
    }
}

/// This type represents a replicated shared value. Since a replicated share of a ring element contains additive shares of two parties, this type contains two ring elements.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Rep3RingSignedShare<T: IntRing2k> {
    pub abs: Rep3RingShare<T>,
    pub sign: Rep3RingShare<Bit>,
}

impl<T: IntRing2k> Rep3RingSignedShare<T> {
    pub fn new(abs: Rep3RingShare<T>, sign: Rep3RingShare<Bit>) -> Self {
        Self { abs, sign }
    }
}

impl<T: IntRing2k> std::ops::Add for Rep3RingShare<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Rep3RingShare {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}
impl<T: IntRing2k> std::ops::Add<&Rep3RingShare<T>> for &'_ Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn add(self, rhs: &Rep3RingShare<T>) -> Self::Output {
        Rep3RingShare {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl<T: IntRing2k> std::ops::AddAssign<Rep3RingShare<T>> for Rep3RingShare<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl<T: IntRing2k> std::ops::AddAssign<&Rep3RingShare<T>> for Rep3RingShare<T> {
    fn add_assign(&mut self, rhs: &Rep3RingShare<T>) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl<T: IntRing2k> std::ops::Sub for Rep3RingShare<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Rep3RingShare {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<T: IntRing2k> std::ops::Sub<&Rep3RingShare<T>> for &'_ Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn sub(self, rhs: &Rep3RingShare<T>) -> Self::Output {
        Rep3RingShare {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<T: IntRing2k> std::ops::SubAssign<Rep3RingShare<T>> for Rep3RingShare<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.a -= rhs.a;
        self.b -= rhs.b;
    }
}

impl<T: IntRing2k> std::ops::Mul for Rep3RingShare<T> {
    type Output = RingElement<T>;

    // Local part of mul only
    fn mul(self, rhs: Rep3RingShare<T>) -> Self::Output {
        self.a * rhs.a + self.a * rhs.b + self.b * rhs.a
    }
}

impl<T: IntRing2k> std::ops::Mul<RingElement<T>> for Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn mul(self, rhs: RingElement<T>) -> Self::Output {
        Self::Output {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}

impl<T: IntRing2k> std::ops::Mul<RingElement<T>> for &Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn mul(self, rhs: RingElement<T>) -> Self::Output {
        Self::Output {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}

impl<T: IntRing2k> std::ops::Mul<&Rep3RingShare<T>> for &'_ Rep3RingShare<T> {
    type Output = RingElement<T>;

    // Local part of mul only
    fn mul(self, rhs: &Rep3RingShare<T>) -> Self::Output {
        self.a * rhs.a + self.a * rhs.b + self.b * rhs.a
    }
}

impl<T: IntRing2k> std::ops::MulAssign<RingElement<T>> for Rep3RingShare<T> {
    fn mul_assign(&mut self, rhs: RingElement<T>) {
        self.a *= rhs;
        self.b *= rhs;
    }
}

impl<T: IntRing2k> std::ops::Neg for Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn neg(self) -> Self::Output {
        Rep3RingShare {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl<T: IntRing2k> ark_ff::Zero for Rep3RingShare<T> {
    fn zero() -> Self {
        Self {
            a: RingElement::zero(),
            b: RingElement::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        panic!(
            "is_zero is not a meaningful operation for Rep3PrimeFieldShare, use interative zero check instead"
        );
    }
}

// Binary (bitwise) operations on Rep3RingShare

impl<T: IntRing2k> std::ops::BitXor for Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::Output {
            a: self.a ^ rhs.a,
            b: self.b ^ rhs.b,
        }
    }
}

impl<T: IntRing2k> std::ops::BitXor<&Rep3RingShare<T>> for &'_ Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn bitxor(self, rhs: &Rep3RingShare<T>) -> Self::Output {
        Self::Output {
            a: self.a ^ rhs.a,
            b: self.b ^ rhs.b,
        }
    }
}

impl<T: IntRing2k> std::ops::BitXor<RingElement<T>> for Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn bitxor(self, rhs: RingElement<T>) -> Self::Output {
        Self::Output {
            a: self.a ^ rhs,
            b: self.b ^ rhs,
        }
    }
}

impl<T: IntRing2k> std::ops::BitXor<&RingElement<T>> for &Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn bitxor(self, rhs: &RingElement<T>) -> Self::Output {
        Self::Output {
            a: self.a ^ rhs,
            b: self.b ^ rhs,
        }
    }
}

impl<T: IntRing2k> std::ops::BitXorAssign<Self> for Rep3RingShare<T> {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.a ^= rhs.a;
        self.b ^= rhs.b;
    }
}

impl<T: IntRing2k> std::ops::BitXorAssign<&Self> for Rep3RingShare<T> {
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.a ^= &rhs.a;
        self.b ^= &rhs.b;
    }
}

impl<T: IntRing2k> std::ops::BitXorAssign<RingElement<T>> for Rep3RingShare<T> {
    fn bitxor_assign(&mut self, rhs: RingElement<T>) {
        self.a ^= rhs;
        self.b ^= rhs;
    }
}

impl<T: IntRing2k> std::ops::BitXorAssign<&RingElement<T>> for Rep3RingShare<T> {
    fn bitxor_assign(&mut self, rhs: &RingElement<T>) {
        self.a ^= rhs;
        self.b ^= rhs;
    }
}

impl<T: IntRing2k> std::ops::BitAnd<RingElement<T>> for Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn bitand(self, rhs: RingElement<T>) -> Self::Output {
        Rep3RingShare {
            a: self.a & rhs,
            b: self.b & rhs,
        }
    }
}

impl<T: IntRing2k> std::ops::BitAnd<&RingElement<T>> for &Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn bitand(self, rhs: &RingElement<T>) -> Self::Output {
        Rep3RingShare {
            a: self.a & rhs,
            b: self.b & rhs,
        }
    }
}

impl<T: IntRing2k> std::ops::BitAnd for Rep3RingShare<T> {
    type Output = RingElement<T>;

    fn bitand(self, rhs: Self) -> Self::Output {
        (self.a & rhs.a) ^ (self.a & rhs.b) ^ (self.b & rhs.a)
    }
}

impl<T: IntRing2k> std::ops::BitAnd<&Rep3RingShare<T>> for &'_ Rep3RingShare<T> {
    type Output = RingElement<T>;

    fn bitand(self, rhs: &Rep3RingShare<T>) -> Self::Output {
        (self.a & rhs.a) ^ (self.a & rhs.b) ^ (self.b & rhs.a)
    }
}

impl<T: IntRing2k> std::ops::BitAndAssign<&RingElement<T>> for Rep3RingShare<T> {
    fn bitand_assign(&mut self, rhs: &RingElement<T>) {
        self.a &= rhs;
        self.b &= rhs;
    }
}

impl<T: IntRing2k> std::ops::BitAndAssign<RingElement<T>> for Rep3RingShare<T> {
    fn bitand_assign(&mut self, rhs: RingElement<T>) {
        self.a &= &rhs;
        self.b &= rhs;
    }
}

impl<T: IntRing2k> std::ops::ShlAssign<usize> for Rep3RingShare<T> {
    fn shl_assign(&mut self, rhs: usize) {
        self.a <<= rhs;
        self.b <<= rhs;
    }
}

impl<T: IntRing2k> std::ops::Shl<usize> for Rep3RingShare<T> {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        Rep3RingShare {
            a: self.a << rhs,
            b: self.b << rhs,
        }
    }
}

impl<T: IntRing2k> std::ops::Shl<usize> for &Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn shl(self, rhs: usize) -> Self::Output {
        Rep3RingShare {
            a: self.a << rhs,
            b: self.b << rhs,
        }
    }
}

impl<T: IntRing2k> std::ops::Shr<usize> for Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn shr(self, rhs: usize) -> Self::Output {
        Rep3RingShare {
            a: self.a >> rhs,
            b: self.b >> rhs,
        }
    }
}

impl<T: IntRing2k> std::ops::Shr<usize> for &Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn shr(self, rhs: usize) -> Self::Output {
        Rep3RingShare {
            a: self.a >> rhs,
            b: self.b >> rhs,
        }
    }
}

impl<T: IntRing2k> std::ops::Not for Rep3RingShare<T> {
    type Output = Self;

    fn not(self) -> Self::Output {
        Rep3RingShare {
            a: !self.a,
            b: !self.b,
        }
    }
}

impl<T: IntRing2k> std::ops::Not for &Rep3RingShare<T> {
    type Output = Rep3RingShare<T>;

    fn not(self) -> Self::Output {
        Rep3RingShare {
            a: !self.a,
            b: !self.b,
        }
    }
}

