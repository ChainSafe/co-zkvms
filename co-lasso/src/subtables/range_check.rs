use std::marker::PhantomData;
use itertools::Itertools;
use jolt_core::poly::field::JoltField;

use crate::subtables::LassoSubtable;

#[derive(Clone, Debug, Default)]
pub struct FullLimbSubtable<F: JoltField>(PhantomData<F>);

impl<F: JoltField> LassoSubtable<F> for FullLimbSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        (0..M).map(|x| F::from(x as u64)).collect_vec()
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        let point_rev = point.to_vec().into_iter().rev().collect::<Vec<_>>();

        let b = point.len();
        let mut result = F::ZERO;
        for i in 0..b {
            result += point_rev[i] * F::from(1u64 << (i));
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
        let point_rev = point.to_vec().into_iter().rev().collect::<Vec<_>>();

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
                result += point_rev[i] * F::from(1u64 << (i));
            } else {
                let mut g_value = F::ZERO;

                if i == cutoff_log2 {
                    for k in 0..num_extra {
                        let mut term: F = F::from(g_base + k).into();
                        for j in 0..cutoff_log2 {
                            if (k & (1 << j)) != 0 {
                                term *= point_rev[j];
                            } else {
                                term *= F::ONE - point_rev[j];
                            }
                        }
                        g_value += term;
                    }
                }

                result = (F::ONE - point_rev[i]) * result + point_rev[i] * g_value
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
