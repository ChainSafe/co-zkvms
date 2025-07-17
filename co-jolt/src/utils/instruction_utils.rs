pub use jolt_core::utils::instruction_utils::*;

use jolt_core::field::JoltField;
use mpc_core::protocols::rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare};
use num_bigint::BigUint;
use std::ops::Shr;

use crate::jolt::instruction::JoltInstruction;

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

pub fn rep3_chunk_and_concatenate_operands<F: JoltField>(
    x: Rep3BigUintShare<F>,
    y: Rep3BigUintShare<F>,
    C: usize,
    log_M: usize,
) -> Vec<Rep3BigUintShare<F>> {
    let operand_bits: usize = log_M / 2;

    let operand_bit_mask = BigUint::from(((1 << operand_bits) - 1) as u64);
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

pub fn slice_values_rep3<'a, F: JoltField, I: JoltInstruction<F>>(op: &I, vals: &'a [Rep3PrimeFieldShare<F>], C: usize, M: usize) -> Vec<&'a [Rep3PrimeFieldShare<F>]> {
    let mut offset = 0;
    let mut slices = vec![];
    for (_, indices) in op.subtables(C, M) {
        slices.push(&vals[offset..offset + indices.len()]);
        offset += indices.len();
    }
    assert_eq!(offset, vals.len());
    slices
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
