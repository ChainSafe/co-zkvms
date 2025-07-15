use ark_ff::PrimeField;
use itertools::Itertools;
use jolt_core::poly::field::JoltField;
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3BigUintShare,
    Rep3PrimeFieldShare,
};
use std::iter::Sum;
use std::{borrow::Borrow, iter};

use super::{JoltInstruction, Rep3JoltInstruction};
use crate::jolt::subtable::{
    range_check::{BoundSubtable, FullLimbSubtable},
    LassoSubtable, SubtableIndices,
};

#[derive(Clone, Debug, PartialEq)]
pub enum RangeLookup<const BOUND: u64, F: JoltField> {
    Public(F),
    Shared {
        value: Rep3PrimeFieldShare<F>,
        binary: Option<Rep3BigUintShare<F>>,
    },
}

impl<const BOUND: u64, F: JoltField> RangeLookup<BOUND, F> {
    pub fn public(value: F) -> Self {
        Self::Public(value)
    }

    pub fn shared(value: Rep3PrimeFieldShare<F>) -> Self {
        Self::Shared { value, binary: None }
    }
}

impl<const BOUND: u64, F: JoltField> JoltInstruction<F> for RangeLookup<BOUND, F> {
    fn combine_lookups(&self, operands: &[F], _: usize, M: usize) -> F {
        let weight = F::from(M as u64);
        inner_product(
            operands,
            iter::successors(Some(F::ONE), |power_of_weight| {
                Some(*power_of_weight * weight)
            })
            .take(operands.len())
            .collect_vec()
            .iter(),
        )
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    // SubtableIndices map subtable to memories
    fn subtables(&self, _C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let full = Box::new(FullLimbSubtable::<F>::new());
        let log_M = M.ilog2() as usize;
        let bound_bits = BOUND.ilog2() as usize;
        let num_chunks = bound_bits / log_M;
        let rem = Box::new(BoundSubtable::<BOUND, F>::new());

        if BOUND % M as u64 == 0 {
            vec![(full, SubtableIndices::from(0..num_chunks))]
        } else if BOUND < M as u64 {
            vec![(rem, SubtableIndices::from(0))]
        } else {
            vec![
                (full, SubtableIndices::from(0..num_chunks)),
                (rem, SubtableIndices::from(num_chunks)),
            ]
        }
    }

    fn lookup_entry(&self) -> F {
        match self {
            Self::Public(value) => *value,
            _ => unreachable!(),
        }
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match self {
            Self::Public(value) => {
                let mut index_bits = fe_to_bits_le(*value);
                index_bits.truncate(chunk_bits(log_M, BOUND).iter().sum());
                let mut chunked_index = iter::repeat(0).take(C).collect_vec();
                let chunked_index_bits = index_bits.chunks(log_M).map(Vec::from).collect_vec();
                chunked_index
                    .iter_mut()
                    .zip(chunked_index_bits)
                    .map(|(chunked_index, index_bits)| {
                        *chunked_index = usize_from_bits_le(&index_bits);
                    })
                    .collect_vec();
                chunked_index
            }
            _ => unreachable!(),
        }
    }
}

impl<const BOUND: u64, F: JoltField> Rep3JoltInstruction<F> for RangeLookup<BOUND, F> {
    fn operands(&self) -> Vec<Rep3PrimeFieldShare<F>> {
        match self {
            Self::Shared { value, .. } => vec![value.clone()],
            _ => unreachable!(),
        }
    }

      
    fn insert_binary_operands(&mut self, mut operands: Vec<Rep3BigUintShare<F>>) {
        assert_eq!(operands.len(), 1);
        match self {
            Self::Shared { binary, .. } => {
                *binary = Some(operands.pop().unwrap());
            },
            _ => unreachable!(),
        }
    }

    fn combine_lookups(
        &self,
        operands: &[Rep3PrimeFieldShare<F>],
        _: usize,
        M: usize,
    ) -> Rep3PrimeFieldShare<F> {
        let weight = F::from(M as u64);
        inner_product_rep3(
            operands,
            iter::successors(Some(F::ONE), |power_of_weight| {
                Some(*power_of_weight * weight)
            })
            .take(operands.len())
            .collect_vec()
            .iter(),
        )
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn output<N: Rep3Network>(&self, _: &mut IoContext<N>) -> Rep3PrimeFieldShare<F> {
        match self {
            Self::Shared { value, .. } => value.clone(),
            _ => unreachable!(),
        }
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<Rep3BigUintShare<F>> {
        let input = match self {
            Self::Shared { binary, .. } => binary.as_ref().expect("binary share must be set"),
            _ => panic!("RangeLookup::to_indices must be called on binary shared value"),
        };

        let mut index_bits = input.to_le_bits();
        index_bits.truncate(chunk_bits(log_M, BOUND).iter().sum());

        let mut chunked_index = iter::repeat(Rep3BigUintShare::zero_share())
            .take(C)
            .collect_vec();
        let chunked_index_bits = index_bits.chunks(log_M).map(Vec::from).collect_vec();
        chunked_index
            .iter_mut()
            .zip(chunked_index_bits)
            .map(|(chunked_index, index_bits)| {
                *chunked_index = Rep3BigUintShare::from_le_bits(&index_bits);
            })
            .collect_vec();
        chunked_index
    }
}

impl<const BOUND: u64, F: JoltField> Default for RangeLookup<BOUND, F> {
    fn default() -> Self {
        Self::Public(F::ZERO)
    }
}

fn inner_product<F: JoltField>(
    lhs: impl IntoIterator<Item = impl Borrow<F>>,
    rhs: impl IntoIterator<Item = impl Borrow<F>>,
) -> F {
    F::sum(
        lhs.into_iter()
            .zip(rhs.into_iter())
            .map(|(lhs, rhs)| *rhs.borrow() * *lhs.borrow()),
    )
}

fn inner_product_rep3<F: JoltField>(
    lhs: impl IntoIterator<Item = impl Borrow<Rep3PrimeFieldShare<F>>>,
    rhs: impl IntoIterator<Item = impl Borrow<F>>,
) -> Rep3PrimeFieldShare<F> {
    Rep3PrimeFieldShare::sum(
        lhs.into_iter()
            .zip(rhs.into_iter())
            .map(|(lhs, rhs)| rep3::arithmetic::mul_public(*lhs.borrow(), *rhs.borrow())),
    )
}

fn chunk_bits(log2_M: usize, bound: u64) -> Vec<usize> {
    let M = 1 << log2_M;
    let bound_bits = bound.ilog2() as usize;

    let remainder_bits = if bound % M as u64 != 0 {
        let reminder = 1 << (bound_bits % log2_M);
        let cutoff = reminder + bound % M as u64;
        let cutoff_log2 = cutoff.ilog2() as usize;
        vec![cutoff_log2]
    } else {
        vec![]
    };
    iter::repeat(log2_M)
        .take(bound_bits / log2_M)
        .chain(remainder_bits)
        .collect_vec()
}

pub fn fe_to_bits_le<F: PrimeField>(fe: F) -> Vec<bool> {
    ark_ff::BigInteger::to_bits_le(&fe.into_bigint())
}

pub fn usize_from_bits_le(bits: &[bool]) -> usize {
    bits.iter()
        .rev()
        .fold(0, |int, bit| (int << 1) + (*bit as usize))
}
