use ark_std::log2;
use jolt_core::jolt::instruction::SubtableIndices;
use jolt_core::poly::field::JoltField;
use jolt_core::utils::instruction_utils::{chunk_and_concatenate_operands, concatenate_lookups};
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare};
use num_bigint::BigUint;
use std::ops::{BitAnd, BitOr, Shr};

use super::utils::concatenate_lookups_rep3;
use super::{JoltInstruction, JoltInstructionSet, Rep3JoltInstruction, Rep3JoltInstructionSet};
use crate::jolt::subtable::{LassoSubtable, XorSubtable};

#[derive(Clone, Debug, PartialEq)]
pub enum XORInstruction<F: JoltField> {
    Shared(Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>),
    SharedBinary(Rep3BigUintShare<F>, Rep3BigUintShare<F>),
    Public(u64, u64),
}

impl<F: JoltField> XORInstruction<F> {
    pub fn public(x: u64, y: u64) -> Self {
        Self::Public(x, y)
    }

    pub fn shared_binary(x: Rep3BigUintShare<F>, y: Rep3BigUintShare<F>) -> Self {
        Self::SharedBinary(x, y)
    }

    pub fn shared(x: Rep3PrimeFieldShare<F>, y: Rep3PrimeFieldShare<F>) -> Self {
        Self::Shared(x, y)
    }
}

impl<F: JoltField> JoltInstruction<F> for XORInstruction<F> {
    // fn operands(&self) -> (u64, u64) {
    //     (self.0, self.1)
    // }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize / 2)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(Box::new(XorSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match self {
            XORInstruction::Public(x, y) => chunk_and_concatenate_operands(*x, *y, C, log_M),
            _ => unreachable!(),
        }
    }

    fn lookup_entry(&self) -> F {
        match self {
            XORInstruction::Public(x, y) => F::from(*x ^ *y),
            _ => unreachable!(),
        }
    }

    // fn random(&self, rng: &mut StdRng) -> Self {
    //     Self(rng.next_u32() as u64, rng.next_u32() as u64)
    // }
}

impl<F: JoltField> Rep3JoltInstruction<F> for XORInstruction<F> {
    fn operands(&self) -> Vec<Rep3PrimeFieldShare<F>> {
        match self {
            XORInstruction::SharedBinary(..) => vec![],
            XORInstruction::Shared(x, y) => vec![x.clone(), y.clone()],
            _ => unreachable!(),
        }
    }

    fn insert_binary_operands(&mut self, mut operands: Vec<Rep3BigUintShare<F>>) {
        match self {
            XORInstruction::Shared(..) => {
                operands.reverse();
                assert_eq!(operands.len(), 2);
                *self =
                    XORInstruction::SharedBinary(operands.pop().unwrap(), operands.pop().unwrap());
            }
            XORInstruction::SharedBinary(..) => {
                assert_eq!(operands.len(), 0);
            }
            _ => unreachable!(),
        }
    }

    fn combine_lookups(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
    ) -> Rep3PrimeFieldShare<F> {
        concatenate_lookups_rep3(vals, C, log2(M) as usize / 2)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn to_indices(
        &self,
        C: usize,
        log_M: usize,
    ) -> Vec<mpc_core::protocols::rep3::Rep3BigUintShare<F>> {
        match self {
            XORInstruction::SharedBinary(x, y) => {
                rep3_chunk_and_concatenate_operands(x.clone(), y.clone(), C, log_M)
            }
            _ => todo!(),
        }
    }

    fn output<N: Rep3Network>(&self, io_ctx: &mut IoContext<N>) -> Rep3PrimeFieldShare<F> {
        match self {
            XORInstruction::SharedBinary(x, y) => {
                rep3::conversion::b2a_selector(&(x.clone() ^ y.clone()), io_ctx).unwrap()
            }
            _ => unreachable!(),
        }
    }
}

impl<F: JoltField> Default for XORInstruction<F> {
    fn default() -> Self {
        Self::Public(0, 0)
    }
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

pub fn chunk_and_concatenate_operands_alt(x: u64, y: u64, C: usize, log_M: usize) -> Vec<usize> {
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
