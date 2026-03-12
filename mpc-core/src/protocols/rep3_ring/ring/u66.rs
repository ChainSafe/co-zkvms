//! U66 — a 66-bit unsigned integer for Z_{2^66}.
//!
//! Implemented as a newtype over `u128` with a 66-bit mask applied after
//! every arithmetic/bitwise operation.

use num_bigint::BigUint;
use num_traits::{One, WrappingAdd, WrappingMul, WrappingNeg, WrappingSub, Zero};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

use super::int_ring::IntRing2k;
use crate::protocols::rep3::IoResult;

const MASK66: u128 = (1u128 << 66) - 1;

/// 66-bit unsigned integer (Z_{2^66}).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct U66(u128);

impl U66 {
    #[inline(always)]
    pub const fn new(v: u128) -> Self {
        Self(v & MASK66)
    }

    #[inline(always)]
    pub const fn inner(self) -> u128 {
        self.0
    }
}

// --- Display ---
impl fmt::Display for U66 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// --- From<bool> ---
impl From<bool> for U66 {
    #[inline(always)]
    fn from(v: bool) -> Self {
        Self(v as u128)
    }
}

// --- Into<u128> ---
impl From<U66> for u128 {
    #[inline(always)]
    fn from(v: U66) -> u128 {
        v.0
    }
}

// --- TryFrom<u128> ---
impl TryFrom<u128> for U66 {
    type Error = &'static str;
    #[inline(always)]
    fn try_from(v: u128) -> Result<Self, Self::Error> {
        Ok(Self(v & MASK66))
    }
}

// --- TryFrom<u64> ---
impl TryFrom<u64> for U66 {
    type Error = &'static str;
    #[inline(always)]
    fn try_from(v: u64) -> Result<Self, Self::Error> {
        Ok(Self(v as u128))
    }
}

// --- TryInto<usize> ---
impl TryFrom<U66> for usize {
    type Error = &'static str;
    fn try_from(v: U66) -> Result<Self, Self::Error> {
        usize::try_from(v.0).map_err(|_| "U66 value too large for usize")
    }
}

// --- Zero / One ---
impl Zero for U66 {
    #[inline(always)]
    fn zero() -> Self {
        Self(0)
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for U66 {
    #[inline(always)]
    fn one() -> Self {
        Self(1)
    }
}

// --- Wrapping arithmetic ---
impl WrappingAdd for U66 {
    #[inline(always)]
    fn wrapping_add(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_add(rhs.0) & MASK66)
    }
}

impl WrappingSub for U66 {
    #[inline(always)]
    fn wrapping_sub(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0) & MASK66)
    }
}

impl WrappingMul for U66 {
    #[inline(always)]
    fn wrapping_mul(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0) & MASK66)
    }
}

impl WrappingNeg for U66 {
    #[inline(always)]
    fn wrapping_neg(&self) -> Self {
        Self(self.0.wrapping_neg() & MASK66)
    }
}

// --- Bitwise ops ---
impl BitXor for U66 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self((self.0 ^ rhs.0) & MASK66)
    }
}

impl BitAnd for U66 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0) // AND can't overflow
    }
}

impl BitOr for U66 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self((self.0 | rhs.0) & MASK66)
    }
}

impl BitXorAssign for U66 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 = (self.0 ^ rhs.0) & MASK66;
    }
}

impl BitAndAssign for U66 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOrAssign for U66 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = (self.0 | rhs.0) & MASK66;
    }
}

impl Not for U66 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0 & MASK66)
    }
}

impl Shl<usize> for U66 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: usize) -> Self {
        if rhs >= 66 { Self(0) } else { Self((self.0 << rhs) & MASK66) }
    }
}

impl Shr<usize> for U66 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: usize) -> Self {
        if rhs >= 66 {
            Self(0)
        } else {
            Self(self.0 >> rhs) // right shift can't overflow
        }
    }
}

// --- std::ops::Add/Sub/Mul for num-traits compatibility ---
impl std::ops::Add for U66 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(&rhs)
    }
}

impl std::ops::Sub for U66 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(&rhs)
    }
}

impl std::ops::Mul for U66 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(&rhs)
    }
}

// --- IntRing2k ---
impl IntRing2k for U66 {
    type Signed = U66; // No separate signed type; wrap correction doesn't need sign.

    const K: usize = 66;
    const BYTES: usize = 9; // ceil(66/8) = 9

    fn from_reader<R: std::io::Read>(mut reader: R) -> IoResult<Self> {
        let mut bytes = [0u8; 16];
        reader.read_exact(&mut bytes[..Self::BYTES])?;
        Ok(Self(u128::from_le_bytes(bytes) & MASK66))
    }

    fn write<W: std::io::Write>(&self, mut writer: W) -> IoResult<()> {
        let bytes = self.0.to_le_bytes();
        writer.write_all(&bytes[..Self::BYTES])
    }

    fn bits(&self) -> usize {
        if self.0 == 0 {
            return 0;
        }
        self.0.ilog2() as usize
    }

    fn cast_to_biguint(&self) -> BigUint {
        BigUint::from(self.0)
    }

    fn cast_from_biguint(biguint: &BigUint) -> Self {
        let mut iter = biguint.iter_u64_digits();
        let x0 = iter.next().unwrap_or_default();
        let x1 = iter.next().unwrap_or_default();
        Self(((x1 as u128) << 64 | x0 as u128) & MASK66)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 16];
        let n = bytes.len().min(Self::BYTES);
        buf[..n].copy_from_slice(&bytes[..n]);
        Self(u128::from_le_bytes(buf) & MASK66)
    }
}

// Neg for Signed (= U66 itself): required by IntRing2k::Signed bound
impl std::ops::Neg for U66 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}

// AsPrimitive<U66> for U66: required by IntRing2k::Signed bound
impl num_traits::AsPrimitive<U66> for U66 {
    #[inline(always)]
    fn as_(self) -> U66 {
        self
    }
}

// Random generation support for MPC share generation.
impl rand::distributions::Distribution<U66> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> U66 {
        U66::new(rng.r#gen::<u128>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u66_wrapping_arithmetic() {
        let max = U66::new(MASK66);
        assert_eq!(max.wrapping_add(&U66::new(1)), U66::new(0));
        assert_eq!(U66::new(0).wrapping_sub(&U66::new(1)), max);
        assert_eq!(U66::new(0).wrapping_neg(), U66::new(0));
        assert_eq!(U66::new(1).wrapping_neg(), max);
    }

    #[test]
    fn u66_shift() {
        let v = U66::new(1);
        assert_eq!(v << 65, U66::new(1u128 << 65));
        assert_eq!(v << 66, U66::new(0));
        let v2 = U66::new(1u128 << 65);
        assert_eq!(v2 >> 65, U66::new(1));
        assert_eq!(v2 >> 66, U66::new(0));
    }

    #[test]
    fn u66_bitwise() {
        let a = U66::new(0xFF);
        let b = U66::new(0x0F);
        assert_eq!(a & b, U66::new(0x0F));
        assert_eq!(a | b, U66::new(0xFF));
        assert_eq!(a ^ b, U66::new(0xF0));
        assert_eq!(!U66::new(0), U66::new(MASK66));
    }

    #[test]
    fn u66_serialization_roundtrip() {
        let v = U66::new((1u128 << 65) | 0xDEAD);
        let mut buf = Vec::new();
        IntRing2k::write(&v, &mut buf).unwrap();
        assert_eq!(buf.len(), 9);
        let v2 = U66::from_reader(&buf[..]).unwrap();
        assert_eq!(v, v2);
    }
}
