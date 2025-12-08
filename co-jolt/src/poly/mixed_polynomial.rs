use std::ops::Index;

use crate::field::JoltField;
use crate::poly::Polynomial;
use crate::utils::types::Rep3Value;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, PolynomialBinding, PolynomialEvaluation,
};
use mpc_core::protocols::rep3::PartyID;
use snarks_core::math::Math;

#[derive(Debug, Clone)]
pub struct MixedPolynomial<F: JoltField> {
    pub coeffs: Vec<Rep3Value<F>>,
    num_vars: usize,
    len: usize,
    party_id: PartyID,
}

impl<F: JoltField> MixedPolynomial<F> {
    pub fn new(evals: Vec<Rep3Value<F>>, party_id: PartyID) -> Self {
        Self {
            num_vars: evals.len().log_2(),
            len: evals.len(),
            coeffs: evals,
            party_id,
        }
    }

    pub fn from_public_evals(evals: Vec<F>, party_id: PartyID) -> Self {
        Self {
            num_vars: evals.len().log_2(),
            len: evals.len(),
            coeffs: evals.into_iter().map(|e| e.into()).collect(),
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
    ) -> Vec<Rep3Value<F>> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![Rep3Value::zero_public(); degree];
        match order {
            BindingOrder::HighToLow => {
                evals[0] = self.coeffs[index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.coeffs[index + self.len() / 2];
                let m = eval.sub(&evals[0], party_id);
                for i in 1..degree {
                    eval.add_assign(&m, party_id);
                    evals[i] = eval;
                }
            }
            BindingOrder::LowToHigh => {
                evals[0] = self.coeffs[2 * index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.coeffs[2 * index + 1];
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
        let (left, right) = self.coeffs.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            a.add_assign(&b.sub(&a, self.party_id).mul_public(*r), self.party_id);
        });

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_bot(&mut self, r: &F) {
        let n = self.len() / 2;
        for i in 0..n {
            self.coeffs[i] = self.coeffs[2 * i].add(
                &self.coeffs[2 * i + 1]
                    .sub(&self.coeffs[2 * i], self.party_id)
                    .mul_public(*r),
                self.party_id,
            );
        }

        self.num_vars -= 1;
        self.len = n;
    }
}

impl<F: JoltField> PolynomialBinding<F, Rep3Value<F>> for MixedPolynomial<F> {
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

    fn final_sumcheck_claim(&self) -> Rep3Value<F> {
        self.coeffs[0]
    }
}

impl<F: JoltField> PolynomialEvaluation<F, Rep3Value<F>> for MixedPolynomial<F> {
    fn evaluate(&self, _r: &[F]) -> Rep3Value<F> {
        todo!()
    }

    fn batch_evaluate(_polys: &[&Self], _r: &[F]) -> (Vec<Rep3Value<F>>, Vec<F>) {
        todo!()
    }

    fn sumcheck_evals(
        &self,
        index: usize,
        degree: usize,
        order: BindingOrder,
    ) -> Vec<Rep3Value<F>> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![Rep3Value::zero_public(); degree];
        match order {
            BindingOrder::HighToLow => {
                evals[0] = self.coeffs[index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.coeffs[index + self.len() / 2];
                let m = eval.sub(&evals[0], self.party_id);
                for i in 1..degree {
                    eval.add_assign(&m, self.party_id);
                    evals[i] = eval;
                }
            }
            BindingOrder::LowToHigh => {
                evals[0] = self.coeffs[2 * index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.coeffs[2 * index + 1];
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

impl<F: JoltField> Polynomial<F> for MixedPolynomial<F> {
    fn len(&self) -> usize {
        self.len
    }

    fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    fn get_bound_coeffs(&self) -> Vec<Rep3Value<F>> {
        self.coeffs[..self.len].to_vec()
    }
}

impl<F: JoltField> Index<usize> for MixedPolynomial<F> {
    type Output = Rep3Value<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coeffs[index]
    }
}
