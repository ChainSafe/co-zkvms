//! U34 — a 34-bit unsigned integer for Z_{2^34}.
//!
//! Implemented as a newtype over `u64` with a 34-bit mask applied after
//! every arithmetic/bitwise operation. Analogous to U66 for rv32 mode
//! where XlenInt = u32 (32 + 2 wrap bits = 34).

use num_bigint::BigUint;
use num_traits::{One, WrappingAdd, WrappingMul, WrappingNeg, WrappingSub, Zero};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

use super::int_ring::IntRing2k;
use crate::protocols::rep3::IoResult;

const MASK34: u64 = (1u64 << 34) - 1;

/// 34-bit unsigned integer (Z_{2^34}).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct U34(u64);

impl U34 {
    #[inline(always)]
    pub const fn new(v: u64) -> Self {
        Self(v & MASK34)
    }

    #[inline(always)]
    pub const fn inner(self) -> u64 {
        self.0
    }
}

// --- Display ---
impl fmt::Display for U34 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// --- From<bool> ---
impl From<bool> for U34 {
    #[inline(always)]
    fn from(v: bool) -> Self {
        Self(v as u64)
    }
}

// --- Into<u64> ---
impl From<U34> for u64 {
    #[inline(always)]
    fn from(v: U34) -> u64 {
        v.0
    }
}

// --- Into<u128> (required by IntRing2k) ---
impl From<U34> for u128 {
    #[inline(always)]
    fn from(v: U34) -> u128 {
        v.0 as u128
    }
}

// --- TryFrom<u128> ---
impl TryFrom<u128> for U34 {
    type Error = &'static str;
    #[inline(always)]
    fn try_from(v: u128) -> Result<Self, Self::Error> {
        Ok(Self((v as u64) & MASK34))
    }
}

// --- TryFrom<u64> ---
impl TryFrom<u64> for U34 {
    type Error = &'static str;
    #[inline(always)]
    fn try_from(v: u64) -> Result<Self, Self::Error> {
        Ok(Self(v & MASK34))
    }
}

// --- TryInto<usize> ---
impl TryFrom<U34> for usize {
    type Error = &'static str;
    fn try_from(v: U34) -> Result<Self, Self::Error> {
        usize::try_from(v.0).map_err(|_| "U34 value too large for usize")
    }
}

// --- Zero / One ---
impl Zero for U34 {
    #[inline(always)]
    fn zero() -> Self {
        Self(0)
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for U34 {
    #[inline(always)]
    fn one() -> Self {
        Self(1)
    }
}

// --- Wrapping arithmetic ---
impl WrappingAdd for U34 {
    #[inline(always)]
    fn wrapping_add(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_add(rhs.0) & MASK34)
    }
}

impl WrappingSub for U34 {
    #[inline(always)]
    fn wrapping_sub(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0) & MASK34)
    }
}

impl WrappingMul for U34 {
    #[inline(always)]
    fn wrapping_mul(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0) & MASK34)
    }
}

impl WrappingNeg for U34 {
    #[inline(always)]
    fn wrapping_neg(&self) -> Self {
        Self(self.0.wrapping_neg() & MASK34)
    }
}

// --- Bitwise ops ---
impl BitXor for U34 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self((self.0 ^ rhs.0) & MASK34)
    }
}

impl BitAnd for U34 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0) // AND can't overflow
    }
}

impl BitOr for U34 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self((self.0 | rhs.0) & MASK34)
    }
}

impl BitXorAssign for U34 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 = (self.0 ^ rhs.0) & MASK34;
    }
}

impl BitAndAssign for U34 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOrAssign for U34 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = (self.0 | rhs.0) & MASK34;
    }
}

impl Not for U34 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0 & MASK34)
    }
}

impl Shl<usize> for U34 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: usize) -> Self {
        if rhs >= 34 {
            Self(0)
        } else {
            Self((self.0 << rhs) & MASK34)
        }
    }
}

impl Shr<usize> for U34 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: usize) -> Self {
        if rhs >= 34 {
            Self(0)
        } else {
            Self(self.0 >> rhs) // right shift can't overflow
        }
    }
}

// --- std::ops::Add/Sub/Mul for num-traits compatibility ---
impl std::ops::Add for U34 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(&rhs)
    }
}

impl std::ops::Sub for U34 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(&rhs)
    }
}

impl std::ops::Mul for U34 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(&rhs)
    }
}

// --- IntRing2k ---
impl IntRing2k for U34 {
    type Signed = U34; // No separate signed type; wrap correction doesn't need sign.

    const K: usize = 34;
    const BYTES: usize = 5; // ceil(34/8) = 5

    fn from_reader<R: std::io::Read>(mut reader: R) -> IoResult<Self> {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes[..Self::BYTES])?;
        Ok(Self(u64::from_le_bytes(bytes) & MASK34))
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
        Self(x0 & MASK34)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 8];
        let n = bytes.len().min(Self::BYTES);
        buf[..n].copy_from_slice(&bytes[..n]);
        Self(u64::from_le_bytes(buf) & MASK34)
    }
}

// Neg for Signed (= U34 itself): required by IntRing2k::Signed bound
impl std::ops::Neg for U34 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}

// AsPrimitive<U34> for U34: required by IntRing2k::Signed bound
impl num_traits::AsPrimitive<U34> for U34 {
    #[inline(always)]
    fn as_(self) -> U34 {
        self
    }
}

// Random generation support for MPC share generation.
impl rand::distributions::Distribution<U34> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> U34 {
        U34::new(rng.r#gen::<u64>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u34_wrapping_arithmetic() {
        let max = U34::new(MASK34);
        assert_eq!(max.wrapping_add(&U34::new(1)), U34::new(0));
        assert_eq!(U34::new(0).wrapping_sub(&U34::new(1)), max);
        assert_eq!(U34::new(0).wrapping_neg(), U34::new(0));
        assert_eq!(U34::new(1).wrapping_neg(), max);
    }

    #[test]
    fn u34_shift() {
        let v = U34::new(1);
        assert_eq!(v << 33, U34::new(1u64 << 33));
        assert_eq!(v << 34, U34::new(0));
        let v2 = U34::new(1u64 << 33);
        assert_eq!(v2 >> 33, U34::new(1));
        assert_eq!(v2 >> 34, U34::new(0));
    }

    #[test]
    fn u34_bitwise() {
        let a = U34::new(0xFF);
        let b = U34::new(0x0F);
        assert_eq!(a & b, U34::new(0x0F));
        assert_eq!(a | b, U34::new(0xFF));
        assert_eq!(a ^ b, U34::new(0xF0));
        assert_eq!(!U34::new(0), U34::new(MASK34));
    }

    #[test]
    fn u34_serialization_roundtrip() {
        let v = U34::new((1u64 << 33) | 0xDEAD);
        let mut buf = Vec::new();
        IntRing2k::write(&v, &mut buf).unwrap();
        assert_eq!(buf.len(), 5);
        let v2 = U34::from_reader(&buf[..]).unwrap();
        assert_eq!(v, v2);
    }
}
