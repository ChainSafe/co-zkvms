use std::{borrow::Borrow, iter, marker::PhantomData};

use ark_ff::PrimeField;
use ark_std::log2;
use co_spartan::mpc::rep3::Rep3PrimeFieldShare;
use itertools::Itertools;
use jolt_core::poly::{dense_mlpoly::DensePolynomial, field::JoltField};
use mpc_core::protocols::rep3::Rep3BigUintShare;

use crate::subtables::{LassoSubtable, LookupType, SubtableIndices};

#[derive(Clone, Debug, Default)]
pub struct FullLimbSubtable<F: JoltField>(PhantomData<F>);

impl<F: JoltField> LassoSubtable<F> for FullLimbSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        (0..M).map(|x| F::from(x as u64)).collect_vec()
    }

    fn evaluate_mle(&self, point: &[F], _: usize) -> F {
        let b = point.len();
        let mut result = F::ZERO;
        for i in 0..b {
            result += point[i] * F::from(1u64 << (i));
        }
        result
    }

    // fn evaluate_mle_expr(&self, log2_M: usize) -> MultilinearPolyTerms<F> {
    //     let limb_init = PolyExpr::Var(0);
    //     let mut limb_terms = vec![limb_init];
    //     (1..log2_M).for_each(|i| {
    //         let coeff = PolyExpr::Pow(Box::new(PolyExpr::Const(F::from(2))), i as u32);
    //         let x = PolyExpr::Var(i);
    //         let term = PolyExpr::Prod(vec![coeff, x]);
    //         limb_terms.push(term);
    //     });
    //     MultilinearPolyTerms::new(log2_M, PolyExpr::Sum(limb_terms))
    // }

    fn subtable_id(&self) -> super::SubtableId {
        "full".to_string()
    }
}

impl<F: JoltField> FullLimbSubtable<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}


#[derive(Clone, Debug, Default, Copy)]
pub struct RangeLookup<F: JoltField> {
    bound: u64,
    _marker: PhantomData<F>,
}

impl<F: JoltField> LookupType<F> for RangeLookup<F> {
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

    // fn combine_lookup_expressions(
    //     &self,
    //     expressions: Vec<Expression<E, usize>>,
    //     _C: usize,
    //     M: usize,
    // ) -> Expression<E, usize> {
    //     Expression::distribute_powers(expressions, F::from(M as u64).into())
    // }

    // SubtableIndices map subtable to memories
    fn subtables(
        &self,
        _C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let full = Box::new(FullLimbSubtable::<F>::new());
        let log_M = M.ilog2() as usize;
        let bound_bits = self.bound.ilog2() as usize;
        let num_chunks = bound_bits / log_M;

        if self.bound % M as u64 == 0 {
            vec![(full, SubtableIndices::from(0..num_chunks))]
        } else {
            unimplemented!()
        }
        // else if self.bound < M as u64 {
        //     vec![(rem, SubtableIndices::from(0))]
        // } else {
        //     vec![
        //         (full, SubtableIndices::from(0..num_chunks)),
        //         (rem, SubtableIndices::from(num_chunks)),
        //     ]
        // }
    }

    fn output(&self, index: &F) -> F {
        *index
    }

    fn output_rep3(&self, index: &Rep3PrimeFieldShare<F>) -> Rep3PrimeFieldShare<F> {
        index.clone()
    }

    fn chunk_bits(&self, M: usize) -> Vec<usize> {
        let log2_M = M.ilog2() as usize;
        let bound_bits = self.bound.ilog2() as usize;

        let remainder_bits = if self.bound % M as u64 != 0 {
            let reminder = 1 << (bound_bits % log2_M);
            let cutoff = reminder + self.bound % M as u64;
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

    fn subtable_indices_rep3(&self, index_bits: Vec<Rep3BigUintShare<F>>, log_M: usize) -> Vec<Vec<Rep3BigUintShare<F>>> {
        index_bits.chunks(log_M).map(Vec::from).collect_vec()
    }

    fn subtable_indices(&self, index_bits: Vec<bool>, log_M: usize) -> Vec<Vec<bool>> {
        index_bits.chunks(log_M).map(Vec::from).collect_vec()
    }

    fn lookup_id(&self) -> super::LookupId {
        format!("range_{}", self.bound)
    }
}

impl<F: JoltField> RangeLookup<F> {
    pub fn new_boxed(bound: u64) -> Box<dyn LookupType<F>> {
        Box::new(Self {
            bound,
            _marker: PhantomData,
        })
    }
}

impl<F: JoltField> RangeLookup<F> {
    pub fn id_for(bound: u64) -> super::LookupId {
        format!("range_{}", bound)
    }
}


fn inner_product<F: JoltField>(
    lhs: impl IntoIterator<Item = impl Borrow<F>>,
    rhs: impl IntoIterator<Item = impl Borrow<F>>,
) -> F {
    F::sum(lhs.into_iter().zip(rhs.into_iter()).map(|(lhs, rhs)| *rhs.borrow() * *lhs.borrow()))
}
