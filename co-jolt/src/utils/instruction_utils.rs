use ark_ff::One;
use itertools::izip;
pub use jolt_core::utils::instruction_utils::*;
use num_traits::AsPrimitive;

use crate::field::JoltField;
use mpc_core::protocols::{
    rep3::{self, Rep3PrimeFieldShare},
    rep3_ring::{
        casts::downcast,
        ring::{int_ring::IntRing2k, ring_impl::RingElement},
        Rep3RingShare,
    },
};
use std::ops::Shr;

pub fn concatenate_lookups_rep3<F: JoltField>(
    vals: &[Rep3PrimeFieldShare<F>],
    C: usize,
    operand_bits: usize,
) -> Rep3PrimeFieldShare<F> {
    assert_eq!(vals.len(), C);

    let mut sum = Rep3PrimeFieldShare::zero_share();
    let mut weight = F::one();
    let shift = F::from_u64(1u64 << operand_bits).unwrap();
    for i in 0..C {
        sum += rep3::arithmetic::mul_public(vals[C - i - 1], weight);
        weight *= shift;
    }
    sum
}

pub fn concatenate_lookups_rep3_batched<F: JoltField>(
    vals: impl IntoIterator<
        Item = Vec<Rep3PrimeFieldShare<F>>,
        IntoIter: DoubleEndedIterator + ExactSizeIterator,
    >,
    C: usize,
    operand_bits: usize,
) -> Vec<Rep3PrimeFieldShare<F>> {
    let mut vals_rev = vals.into_iter().rev();
    assert_eq!(vals_rev.len(), C);
    let mut sums = vals_rev.next().unwrap();
    let shift = F::from_u64(1u64 << operand_bits).unwrap();
    let mut weight = shift;
    for val in vals_rev {
        // sum += rep3::arithmetic::mul_public(vals[C - i - 1], weight);
        izip!(sums.iter_mut(), val).for_each(|(sum, val)| {
            *sum += rep3::arithmetic::mul_public(val, weight);
        });
        weight *= shift;
    }
    sums
}

pub fn rep3_chunk_and_concatenate_operands(
    x: Rep3RingShare<u32>,
    y: Rep3RingShare<u32>,
    C: usize,
    log_M: usize,
) -> Vec<Rep3RingShare<u32>> {
    let operand_bits: usize = log_M / 2;

    let operand_bit_mask = RingElement(((1 << operand_bits) - 1) as u32);
    (0..C)
        .map(|i| {
            let shift = (C - i - 1) * operand_bits;
            let left = x.clone().shr(shift) & operand_bit_mask.clone();
            let right = y.clone().shr(shift) & operand_bit_mask.clone();
            // (left << operand_bits) | right
            // since we performed left shift the right part are all zero bits so we can do XOR instead of OR
            (left << operand_bits) ^ right
        })
        .collect()
}

/// z = x + y (mod 2^{C*log_M}), then split z into C chunks of `log_M` bits (MSB-first).
pub fn rep3_add_and_chunk_operands(
    z: &Rep3RingShare<u128>,
    C: usize,
    log_M: usize,
) -> Vec<Rep3RingShare<u32>> {
    let sum_chunk_bits: usize = log_M;
    let sum_chunk_bit_mask = RingElement(((1 << sum_chunk_bits) - 1) as u64);

    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * sum_chunk_bits) as u32 as usize;

            ((z >> shift).downcast() & sum_chunk_bit_mask.clone()).downcast()
        })
        .collect()
}

/// Chunks `z` into `C` chunks bitwise where `z = x * y`.
/// `log_M` is the number of bits for each of the `C` chunks of `z`.
pub fn rep3_multiply_and_chunk_operands(
    z: &Rep3RingShare<u128>,
    C: usize,
    log_M: usize,
) -> Vec<Rep3RingShare<u32>> {
    let product_chunk_bits: usize = log_M;
    let product_chunk_bit_mask = RingElement(((1 << product_chunk_bits) - 1) as u128);
    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * product_chunk_bits) as u32 as usize;
            downcast((z >> shift) & product_chunk_bit_mask.clone())
        })
        .collect()
}

/// Chunks and concatenates two 64-bit unsigned integers `x` and `y` into a vector of concatenated chunks,
/// where the second half of each concatenated chunk is always `y_0`, the last chunk of `y` (from left to right).
pub fn rep3_chunk_and_concatenate_for_shift(
    x: Rep3RingShare<u32>,
    y: Rep3RingShare<u32>,
    C: usize,
    log_M: usize,
) -> Vec<Rep3RingShare<u32>> {
    let operand_bits: usize = log_M / 2;
    let operand_bit_mask = RingElement(((1 << operand_bits) - 1) as u32);

    let y_lowest_chunk = y & operand_bit_mask.clone();

    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * operand_bits) as u32 as usize;
            let left = x.clone().shr(shift) & operand_bit_mask.clone();
            // (left << operand_bits) | y_lowest_chunk
            // since we performed left shift the right part are all zero bits so we can do XOR instead of OR
            (left << operand_bits) ^ y_lowest_chunk.clone()
        })
        .collect()
}

/// Splits a 64-bit unsigned integer `x` into a `C`-length vector of `usize`, each representing a
/// `chunk_len`-bit chunk.
pub fn rep3_chunk_operand<U: IntRing2k>(
    x: Rep3RingShare<u32>,
    C: usize,
    chunk_len: usize,
) -> Vec<Rep3RingShare<U>>
where
    u32: AsPrimitive<U>,
{
    let bit_mask = RingElement(((1 << chunk_len) - 1) as u32);
    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * chunk_len) as u32 as usize;
            downcast(x.clone().shr(shift) & bit_mask.clone())
        })
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    use ark_std::test_rng;

    type F = ark_bn254::Fr;

    #[test]
    fn test_chunk_and_concatenate_operands() {
        let x = 0b10101010101010;
        let y = 0b11001100110011;
        let C = 4;
        let log_M = 8;
        let indices = chunk_and_concatenate_operands(x, y, C, log_M);
        let indices_alt = chunk_and_concatenate_operands_alt(x, y, C, log_M);
        assert_eq!(indices, indices_alt);
        println!("{:?}", indices);
    }

    fn chunk_and_concatenate_operands_alt(x: u64, y: u64, C: usize, log_M: usize) -> Vec<usize> {
        let operand_bits: usize = log_M / 2;

        #[cfg(test)]
        {
            let max_operand_bits = C * log_M / 2;
            println!("max_operand_bits: {}", max_operand_bits);
            if max_operand_bits != 64 {
                // if 64, handled by normal overflow checking
                let max_operand: u64 = (1 << max_operand_bits) - 1;
                assert!(x <= max_operand);
                assert!(y <= max_operand);
            }
        }

        let operand_bit_mask: u64 = (1 << operand_bits) - 1;
        (0..C)
            .map(|i| {
                let shift = ((C - i - 1) * operand_bits) as u32;
                let left = x.shr(shift) & operand_bit_mask;
                let right = y.shr(shift) & operand_bit_mask;
                ((left << operand_bits) ^ right) as usize
            })
            .collect()
    }
}
