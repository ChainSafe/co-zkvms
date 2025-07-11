use std::{borrow::Borrow, iter, marker::PhantomData};

use ark_ff::PrimeField;
use ark_std::log2;
use co_spartan::mpc::rep3::Rep3PrimeFieldShare;
use itertools::Itertools;
use jolt_core::poly::{dense_mlpoly::DensePolynomial, field::JoltField};
use mpc_core::protocols::rep3::Rep3BigUintShare;

use crate::subtables::{LassoSubtable, LookupType, Rep3LookupType, SubtableIndices};

#[derive(Clone, Debug, Default)]
pub struct FullLimbSubtable<F: JoltField>(PhantomData<F>);

impl<F: JoltField> LassoSubtable<F> for FullLimbSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        (0..M).map(|x| F::from(x as u64)).collect_vec()
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        let b = point.len();
        let mut result = F::ZERO;
        for i in 0..b {
            result += point[i] * F::from(1u64 << (i));
        }
        result
    }
}

impl<F: JoltField> FullLimbSubtable<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

#[derive(Clone, Debug, Default)]
pub struct BoundSubtable<const BOUND: u64, F> {
    _marker: PhantomData<F>,
}

impl<const BOUND: u64, F: JoltField> LassoSubtable<F> for BoundSubtable<BOUND, F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        let bound_bits = BOUND.ilog2() as usize;
        let reminder = 1 << (bound_bits % M.ilog2() as usize);
        let cutoff = (reminder + BOUND % M as u64) as usize;

        (0..M)
            .map(|i| {
                if i < cutoff {
                    F::from(i as u64)
                } else {
                    F::ZERO
                }
            })
            .collect()
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        let log2_M = point.len();
        let b = point.len();

        let bound_bits = BOUND.ilog2() as usize;
        let reminder = 1 << (bound_bits % log2_M);
        let cutoff = reminder + BOUND % (1 << log2_M);
        let cutoff_log2 = cutoff.ilog2() as usize;

        let g_base = 1 << cutoff_log2;
        let num_extra = cutoff - g_base;

        let mut result = F::ZERO;
        for i in 0..b {
            if i < cutoff_log2 {
                result += point[i] * F::from(1u64 << (i));
            } else {
                let mut g_value = F::ZERO;

                if i == cutoff_log2 {
                    for k in 0..num_extra {
                        let mut term: F = F::from(g_base + k).into();
                        for j in 0..cutoff_log2 {
                            if (k & (1 << j)) != 0 {
                                term *= point[j];
                            } else {
                                term *= F::ONE - point[j];
                            }
                        }
                        g_value += term;
                    }
                }

                result = (F::ONE - point[i]) * result + point[i] * g_value
            }
        }

        result
    }
}

impl<const BOUND: u64, F: JoltField> BoundSubtable<BOUND, F> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RangeLookup<const BOUND: u64, F: JoltField> {
    Public(F),
    Shared {
        value: Rep3PrimeFieldShare<F>,
        binary: Rep3BigUintShare<F>,
    },
}

impl<const BOUND: u64, F: JoltField> RangeLookup<BOUND, F> {
    pub fn public(value: F) -> Self {
        Self::Public(value)
    }

    pub fn shared(value: Rep3PrimeFieldShare<F>, binary: Rep3BigUintShare<F>) -> Self {
        Self::Shared {
            value,
            binary,
        }
    }
}

impl<const BOUND: u64, F: JoltField> LookupType<F> for RangeLookup<BOUND, F> {
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

    fn output(&self) -> F {
        match self {
            Self::Public(value) => *value,
            _ => unreachable!(),
        }
    }

    fn subtable_indices(&self, index_bits: Vec<bool>, log_M: usize) -> Vec<Vec<bool>> {
        index_bits.chunks(log_M).map(Vec::from).collect_vec()
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match self {
            Self::Public(value) => {
                let mut index_bits = fe_to_bits_le(*value);
                index_bits.truncate(chunk_bits(log_M, BOUND).iter().sum());
                let mut chunked_index = iter::repeat(0).take(C).collect_vec();
                let chunked_index_bits = self.subtable_indices(index_bits, log_M);
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

impl<const BOUND: u64, F: JoltField> Rep3LookupType<F> for RangeLookup<BOUND, F> {
    fn combine_lookups(
        &self,
        operands: &[Rep3PrimeFieldShare<F>],
        _: usize,
        M: usize,
    ) -> Rep3PrimeFieldShare<F> {
        todo!()
    }

    fn output(&self) -> Rep3PrimeFieldShare<F> {
        match self {
            Self::Shared { value, .. } => value.clone(),
            _ => unreachable!(),
        }
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<Rep3BigUintShare<F>> {
        let input = match self {
            Self::Shared { binary, .. } => binary,
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
