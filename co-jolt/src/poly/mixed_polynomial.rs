use std::ops::Index;

use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::PolyDegree;
use crate::utils::element::SharedOrPublic;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use jolt_core::{field::JoltField, poly::compact_polynomial::CompactPolynomial};
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};
use rayon::iter::IntoParallelIterator;
use snarks_core::math::Math;

pub struct MixedPolynomial<F: JoltField> {
    pub evals: Vec<SharedOrPublic<F>>,
    num_vars: usize,
    len: usize,
    party_id: PartyID,
}

impl<F: JoltField> MixedPolynomial<F> {
    pub fn new(evals: Vec<SharedOrPublic<F>>, party_id: PartyID) -> Self {
        Self {
            num_vars: evals.len().log_2(),
            len: evals.len(),
            evals,
            party_id,
        }
    }

    pub fn from_public_evals(evals: Vec<F>, party_id: PartyID) -> Self {
        Self {
            num_vars: evals.len().log_2(),
            len: evals.len(),
            evals: evals.into_iter().map(|e| e.into()).collect(),
            party_id,
        }
    }

    #[inline]
    pub fn sumcheck_evals(
        &self,
        index: usize,
        degree: usize,
        order: BindingOrder,
        party_id: PartyID,
    ) -> Vec<SharedOrPublic<F>> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![SharedOrPublic::zero_public(); degree];
        match order {
            BindingOrder::HighToLow => {
                evals[0] = self.evals[index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.evals[index + self.len() / 2];
                let m = eval.sub(&evals[0], party_id);
                for i in 1..degree {
                    eval.add_assign(&m, party_id);
                    evals[i] = eval;
                }
            }
            BindingOrder::LowToHigh => {
                evals[0] = self.evals[2 * index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.evals[2 * index + 1];
                let m = eval.sub(&evals[0], party_id);
                for i in 1..degree {
                    eval.add_assign(&m, party_id);
                    evals[i] = eval;
                }
            }
        };
        evals
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.evals.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            a.add_assign(&b.sub(&a, self.party_id).mul_public(*r), self.party_id);
        });

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_bot(&mut self, r: &F) {
        let n = self.len() / 2;
        for i in 0..n {
            self.evals[i] = self.evals[2 * i].add(
                &self.evals[2 * i + 1]
                    .sub(&self.evals[2 * i], self.party_id)
                    .mul_public(*r),
                self.party_id,
            );
        }

        self.num_vars -= 1;
        self.len = n;
    }
}

impl<F: JoltField> PolynomialBinding<F> for MixedPolynomial<F> {
    fn is_bound(&self) -> bool {
        unimplemented!()
    }

    fn bind(&mut self, r: F, order: BindingOrder) {
        match order {
            BindingOrder::HighToLow => self.bound_poly_var_top(&r),
            BindingOrder::LowToHigh => self.bound_poly_var_bot(&r),
        }
    }

    fn bind_parallel(&mut self, _r: F, _order: BindingOrder) {
        todo!()
    }

    fn final_sumcheck_claim(&self) -> F {
        self.evals[0].into_additive(self.party_id)
    }
}

impl<F: JoltField> PolynomialEvaluation<F, SharedOrPublic<F>> for MixedPolynomial<F> {
    fn evaluate(&self, r: &[F]) -> SharedOrPublic<F> {
        todo!()
    }

    fn batch_evaluate(polys: &[&Self], r: &[F]) -> (Vec<SharedOrPublic<F>>, Vec<F>) {
        todo!()
    }

    fn sumcheck_evals(
        &self,
        index: usize,
        degree: usize,
        order: BindingOrder,
    ) -> Vec<SharedOrPublic<F>> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![SharedOrPublic::zero_public(); degree];
        match order {
            BindingOrder::HighToLow => {
                evals[0] = self.evals[index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.evals[index + self.len() / 2];
                let m = eval.sub(&evals[0], self.party_id);
                for i in 1..degree {
                    eval.add_assign(&m, self.party_id);
                    evals[i] = eval;
                }
            }
            BindingOrder::LowToHigh => {
                evals[0] = self.evals[2 * index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.evals[2 * index + 1];
                let m = eval.sub(&evals[0], self.party_id);
                for i in 1..degree {
                    eval.add_assign(&m, self.party_id);
                    evals[i] = eval;
                }
            }
        };
        evals
    }
}

impl<F: JoltField> PolyDegree for MixedPolynomial<F> {
    fn len(&self) -> usize {
        self.len
    }

    fn get_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: JoltField> Index<usize> for MixedPolynomial<F> {
    type Output = SharedOrPublic<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evals[index]
    }
}
