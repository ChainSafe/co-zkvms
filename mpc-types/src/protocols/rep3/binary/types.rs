use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use std::marker::PhantomData;

use ark_ff::{PrimeField, Zero};
use num_bigint::BigUint;

use crate::protocols::rep3;

/// This type represents a packed vector of replicated shared bits. Each additively shared vector is represented as [BigUint]. Thus, this type contains two [BigUint]s.
#[derive(Debug, Clone, PartialEq, Eq, Hash, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3BigUintShare<F: PrimeField> {
    /// Share of this party
    pub a: BigUint,
    /// Share of the prev party
    pub b: BigUint,
    pub(crate) phantom: PhantomData<F>,
}

impl<F: PrimeField> Default for Rep3BigUintShare<F> {
    fn default() -> Self {
        Self::zero_share()
    }
}

impl<F: PrimeField> Rep3BigUintShare<F> {
    /// Constructs the type from two additive shares.
    pub fn new(a: BigUint, b: BigUint) -> Self {
        Self {
            a,
            b,
            phantom: PhantomData,
        }
    }

    /// Constructs a zero share.
    pub fn zero_share() -> Self {
        Self {
            a: BigUint::ZERO,
            b: BigUint::ZERO,
            phantom: PhantomData,
        }
    }

    /// Unwraps the type into two additive shares.
    pub fn ab(self) -> (BigUint, BigUint) {
        (self.a, self.b)
    }

    /// Converts the share to a vector of bits in little-endian order.
    pub fn to_le_bits(&self) -> Vec<Rep3BigUintShare<F>> {
        let bits_a = biguint_to_bits_le(&self.a, F::MODULUS_BIT_SIZE as usize);
        let bits_b = biguint_to_bits_le(&self.b, F::MODULUS_BIT_SIZE as usize);

        bits_a
            .into_iter()
            .zip(bits_b.into_iter())
            .take(F::MODULUS_BIT_SIZE as usize)
            .map(|(a, b)| Rep3BigUintShare::new(BigUint::from(a as u64), BigUint::from(b as u64)))
            .collect()
    }

    /// Converts the share to a vector of bits in little-endian order, padding with zeros if necessary.
    pub fn to_le_bits_padded(&self, num_bits: usize) -> Vec<Rep3BigUintShare<F>> {
        let bits = biguint_to_bits_le(&self.a, F::MODULUS_BIT_SIZE as usize);
        bits.into_iter()
            .take(F::MODULUS_BIT_SIZE as usize)
            .pad_using(num_bits, |_| false)
            .map(|b| Rep3BigUintShare::new(BigUint::from(b as u64), BigUint::from(0 as u64)))
            .collect()
    }

    /// Converts a vector of bits in little-endian order to a share.
    pub fn from_le_bits(bits: &[Self]) -> Self {
        bits.iter()
            .rev()
            .fold(Self::zero_share(), |int, bit| int << 1 ^ bit.clone())
    }
}

/// Convert BigUint to little-endian bits
fn biguint_to_bits_le(val: &BigUint, num_bits: usize) -> Vec<bool> {
    let mut bits = Vec::with_capacity(num_bits);

    let bits_per_digit = 64u64;
    for digit in val.iter_u64_digits() {
        for bit_idx in 0..64 {
            let bit_mask = (1 as u64) << (bit_idx % bits_per_digit);
            bits.push((digit & bit_mask) != 0);
        }
    }

    bits
}
