use ark_ff::{One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use num_bigint::BigUint;
use rand::{CryptoRng, Rng};
use std::marker::PhantomData;

use crate::mpc::SSOpen;

pub type BinaryShare<F> = Rep3BigUintShare<F>;

/// This type represents a packed vector of replicated shared bits. Each additively shared vector is represented as [BigUint]. Thus, this type contains two [BigUint]s.
#[derive(Debug, Clone, PartialEq, Eq, Hash, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3BigUintShare<F: PrimeField> {
    /// Share of this party
    pub a: BigUint,
    /// Share of the prev party
    pub b: BigUint,
    pub(crate) phantom: PhantomData<F>,
}

impl<F: PrimeField> Rep3BigUintShare<F> {
    pub fn new(a: BigUint, b: BigUint) -> Self {
        Self {
            a,
            b,
            phantom: PhantomData,
        }
    }

    pub fn to_le_bits(&self) -> Vec<BinaryShare<F>> {
        let bits = biguint_to_bits_le(&self.a, F::MODULUS_BIT_SIZE as usize);
        bits.into_iter()
            .take(F::MODULUS_BIT_SIZE as usize)
            .map(|b| Rep3BigUintShare::new(BigUint::from(b as u64), BigUint::from(0 as u64)))
            .collect()
    }

    pub fn to_le_bits_padded(&self, num_bits: usize) -> Vec<BinaryShare<F>> {
        let bits = biguint_to_bits_le(&self.a, F::MODULUS_BIT_SIZE as usize);
        bits.into_iter()
            .take(F::MODULUS_BIT_SIZE as usize)
            .pad_using(num_bits, |_| false)
            .map(|b| Rep3BigUintShare::new(BigUint::from(b as u64), BigUint::from(0 as u64)))
            .collect()
    }

    pub fn from_le_bits(bits: &[Self]) -> Self {
        const WORD: usize = 64;

        let mut a = BigUint::zero();
        let mut b = BigUint::zero();

        for (block_idx, chunk) in bits.chunks(WORD).enumerate() {
            let mut acc_a: u64 = 0;
            let mut acc_b: u64 = 0;

            // gather bits of this 64-bit block into two u64 words
            for (i, bit) in chunk.iter().enumerate() {
                if !bit.a.is_zero() {
                    acc_a |= 1u64 << i;
                }
                if !bit.b.is_zero() {
                    acc_b |= 1u64 << i;
                }
            }

            if acc_a != 0 {
                a |= BigUint::from(acc_a) << (block_idx * WORD);
            }
            if acc_b != 0 {
                b |= BigUint::from(acc_b) << (block_idx * WORD);
            }
        }

        Rep3BigUintShare {
            a,
            b,
            phantom: PhantomData,
        }
    }

    pub fn zero() -> Self {
        Rep3BigUintShare {
            a: BigUint::zero(),
            b: BigUint::zero(),
            phantom: PhantomData,
        }
    }
}

/// Secret shares a field element using replicated secret sharing and the provided random number generator. The field element is split into three binary shares, where each party holds two. The outputs are of type [Rep3BigUintShare].
pub fn share_biguint<F: PrimeField, R: Rng + CryptoRng>(
    val: F,
    rng: &mut R,
) -> [Rep3BigUintShare<F>; 3] {
    let val: BigUint = val.into();
    let limbsize = F::MODULUS_BIT_SIZE.div_ceil(32);
    let mask = (BigUint::from(1u32) << F::MODULUS_BIT_SIZE) - BigUint::one();
    let a = BigUint::new((0..limbsize).map(|_| rng.r#gen()).collect()) & &mask;
    let b = BigUint::new((0..limbsize).map(|_| rng.r#gen()).collect()) & mask;

    let c = val ^ &a ^ &b;
    let share1 = Rep3BigUintShare::new(a.to_owned(), c.to_owned());
    let share2 = Rep3BigUintShare::new(b.to_owned(), a);
    let share3 = Rep3BigUintShare::new(c, b);
    [share1, share2, share3]
}

impl<F: PrimeField> SSOpen<F> for Rep3BigUintShare<F> {
    fn open(shares: &[Self]) -> F {
        assert_eq!(shares.len(), 3);
        let val = combine_binary_element(shares[0].clone(), shares[1].clone(), shares[2].clone());
        F::from(val)
    }
}

/// Reconstructs a value (represented as [BigUint]) from its binary replicated shares. Since binary operations can lead to results >= p, the result is not guaranteed to be a valid field element.
pub fn combine_binary_element<F: PrimeField>(
    share1: Rep3BigUintShare<F>,
    share2: Rep3BigUintShare<F>,
    share3: Rep3BigUintShare<F>,
) -> BigUint {
    share1.a ^ share2.a ^ share3.a
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

impl<F: PrimeField> std::ops::BitXor for Rep3BigUintShare<F> {
    type Output = Rep3BigUintShare<F>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::Output {
            a: self.a ^ rhs.a,
            b: self.b ^ rhs.b,
            phantom: PhantomData,
        }
    }
}

impl<F: PrimeField> std::ops::BitXor<&Rep3BigUintShare<F>> for &'_ Rep3BigUintShare<F> {
    type Output = Rep3BigUintShare<F>;

    fn bitxor(self, rhs: &Rep3BigUintShare<F>) -> Self::Output {
        Self::Output {
            a: &self.a ^ &rhs.a,
            b: &self.b ^ &rhs.b,
            phantom: PhantomData,
        }
    }
}

impl<F: PrimeField> std::ops::BitXor<BigUint> for Rep3BigUintShare<F> {
    type Output = Rep3BigUintShare<F>;

    fn bitxor(self, rhs: BigUint) -> Self::Output {
        Self::Output {
            a: &self.a ^ &rhs,
            b: &self.b ^ &rhs,
            phantom: PhantomData,
        }
    }
}

impl<F: PrimeField> std::ops::BitXor<&BigUint> for &Rep3BigUintShare<F> {
    type Output = Rep3BigUintShare<F>;

    fn bitxor(self, rhs: &BigUint) -> Self::Output {
        Self::Output {
            a: &self.a ^ rhs,
            b: &self.b ^ rhs,
            phantom: PhantomData,
        }
    }
}

impl<F: PrimeField> std::ops::BitXorAssign<Self> for Rep3BigUintShare<F> {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.a ^= &rhs.a;
        self.b ^= &rhs.b;
    }
}

impl<F: PrimeField> std::ops::BitXorAssign<&Self> for Rep3BigUintShare<F> {
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.a ^= &rhs.a;
        self.b ^= &rhs.b;
    }
}

impl<F: PrimeField> std::ops::BitXorAssign<BigUint> for Rep3BigUintShare<F> {
    fn bitxor_assign(&mut self, rhs: BigUint) {
        self.a ^= &rhs;
        self.b ^= &rhs;
    }
}

impl<F: PrimeField> std::ops::BitXorAssign<&BigUint> for Rep3BigUintShare<F> {
    fn bitxor_assign(&mut self, rhs: &BigUint) {
        self.a ^= rhs;
        self.b ^= rhs;
    }
}

impl<F: PrimeField> std::ops::BitAnd<BigUint> for Rep3BigUintShare<F> {
    type Output = Rep3BigUintShare<F>;

    fn bitand(self, rhs: BigUint) -> Self::Output {
        Rep3BigUintShare {
            a: &self.a & &rhs,
            b: &self.b & &rhs,
            phantom: PhantomData,
        }
    }
}

impl<F: PrimeField> std::ops::BitAnd<&BigUint> for &Rep3BigUintShare<F> {
    type Output = Rep3BigUintShare<F>;

    fn bitand(self, rhs: &BigUint) -> Self::Output {
        Rep3BigUintShare {
            a: &self.a & rhs,
            b: &self.b & rhs,
            phantom: PhantomData,
        }
    }
}

impl<F: PrimeField> std::ops::BitAnd for Rep3BigUintShare<F> {
    type Output = BigUint;

    fn bitand(self, rhs: Self) -> Self::Output {
        (&self.a & &rhs.a) ^ (&self.a & &rhs.b) ^ (&self.b & &rhs.a)
    }
}

impl<F: PrimeField> std::ops::BitAnd<&Rep3BigUintShare<F>> for &'_ Rep3BigUintShare<F> {
    type Output = BigUint;

    fn bitand(self, rhs: &Rep3BigUintShare<F>) -> Self::Output {
        (&self.a & &rhs.a) ^ (&self.a & &rhs.b) ^ (&self.b & &rhs.a)
    }
}

impl<F: PrimeField> std::ops::BitAndAssign<&BigUint> for Rep3BigUintShare<F> {
    fn bitand_assign(&mut self, rhs: &BigUint) {
        self.a &= rhs;
        self.b &= rhs;
    }
}

impl<F: PrimeField> std::ops::BitAndAssign<BigUint> for Rep3BigUintShare<F> {
    fn bitand_assign(&mut self, rhs: BigUint) {
        self.a &= &rhs;
        self.b &= &rhs;
    }
}

impl<F: PrimeField> std::ops::ShlAssign<usize> for Rep3BigUintShare<F> {
    fn shl_assign(&mut self, rhs: usize) {
        self.a <<= rhs;
        self.b <<= rhs;
    }
}

impl<F: PrimeField> std::ops::Shl<usize> for Rep3BigUintShare<F> {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        Rep3BigUintShare {
            a: &self.a << rhs,
            b: &self.b << rhs,
            phantom: PhantomData,
        }
    }
}

impl<F: PrimeField> std::ops::Shl<usize> for &Rep3BigUintShare<F> {
    type Output = Rep3BigUintShare<F>;

    fn shl(self, rhs: usize) -> Self::Output {
        Rep3BigUintShare {
            a: &self.a << rhs,
            b: &self.b << rhs,
            phantom: PhantomData,
        }
    }
}

impl<F: PrimeField> std::ops::Shr<usize> for Rep3BigUintShare<F> {
    type Output = Rep3BigUintShare<F>;

    fn shr(self, rhs: usize) -> Self::Output {
        Rep3BigUintShare {
            a: &self.a >> rhs,
            b: &self.b >> rhs,
            phantom: PhantomData,
        }
    }
}

impl<F: PrimeField> std::ops::Shr<usize> for &Rep3BigUintShare<F> {
    type Output = Rep3BigUintShare<F>;

    fn shr(self, rhs: usize) -> Self::Output {
        Rep3BigUintShare {
            a: &self.a >> rhs,
            b: &self.b >> rhs,
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpc::{rep3::share_field_element, SSOpen};
    use ark_ff::BigInteger;
    use ark_ff::One;
    use ark_linear_sumcheck::rng::Blake2s512Rng;
    use rand::SeedableRng;

    #[test]
    fn test_to_le_bits_rep3() {
        let mut rng = rand::rngs::StdRng::from_seed(Default::default());
        let fe = ark_bn254::Fr::from(8u64);
        let s = share_biguint(fe, &mut rng);

        let max_bits = <ark_bn254::Fr as PrimeField>::BigInt::NUM_LIMBS * 64;
        let bits1 = s[0].to_le_bits_padded(max_bits);
        let bits2 = s[1].to_le_bits_padded(max_bits);
        let bits3 = s[2].to_le_bits_padded(max_bits);

        let mut bits = bits1
            .iter()
            .zip(bits2.iter())
            .zip(bits3.iter())
            .map(|((b1, b2), b3)| Rep3BigUintShare::open(&[b1.clone(), b2.clone(), b3.clone()]).is_one())
            .collect::<Vec<_>>();

        let bits_check = fe.into_bigint().to_bits_le().to_vec();
        assert_eq!(bits, bits_check);

        let fe1 = Rep3BigUintShare::from_le_bits(&bits1);
        let fe2 = Rep3BigUintShare::from_le_bits(&bits2);
        let fe3 = Rep3BigUintShare::from_le_bits(&bits3);
        let fe_reconstructed = Rep3BigUintShare::open(&[fe1, fe2, fe3]);
        assert_eq!(fe, fe_reconstructed);
    }
}
