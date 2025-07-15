use ark_ff::{Field, PrimeField, Zero};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::Rng;
use std::ops::Index;

use super::Rep3PrimeFieldShare;

#[derive(Debug, CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct Rep3DensePolynomial<F: PrimeField> {
    // pub party_id: usize,
    pub share_0: DenseMultilinearExtension<F>,
    pub share_1: DenseMultilinearExtension<F>,
}

impl<F: PrimeField> Rep3DensePolynomial<F> {
    pub fn new(evals: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        let num_vars = evals.len().ilog2() as usize;
        Self::new_with_vars(evals, num_vars)
    }

    pub fn new_with_vars(evals_rep3: Vec<Rep3PrimeFieldShare<F>>, num_vars: usize) -> Self {
        let mut share_0 = Vec::with_capacity(1 << num_vars);
        let mut share_1 = Vec::with_capacity(1 << num_vars);
        // let party = evals_rep3[0].party;
        for share in evals_rep3 {
            share_0.push(share.a);
            share_1.push(share.b);
        }
        Rep3DensePolynomial {
            // party_id: party,
            share_0: DenseMultilinearExtension::<F>::from_evaluations_vec(num_vars, share_0),
            share_1: DenseMultilinearExtension::<F>::from_evaluations_vec(num_vars, share_1),
        }
    }

    pub fn from_poly_shares(
        share_0: DenseMultilinearExtension<F>,
        share_1: DenseMultilinearExtension<F>,
    ) -> Self {
        Rep3DensePolynomial {
            // party_id: party,
            share_0,
            share_1,
        }
    }

    pub fn get_share_by_idx(&self, i: usize) -> Rep3PrimeFieldShare<F> {
        Rep3PrimeFieldShare {
            // party: self.party_id,
            a: self.share_0.index(i).clone(),
            b: self.share_1.index(i).clone(),
        }
    }

    pub fn fix_variables(&self, partial_point: &[F]) -> Self {
        Rep3DensePolynomial {
            // party_id: self.party_id,
            share_0: self.share_0.fix_variables(partial_point),
            share_1: self.share_1.fix_variables(partial_point),
        }
    }

    pub fn num_vars(&self) -> usize {
        assert_eq!(self.share_0.num_vars(), self.share_1.num_vars());
        self.share_0.num_vars()
    }

    pub fn len(&self) -> usize {
        1 << self.num_vars()
    }

    pub fn evals(&self) -> Vec<Rep3PrimeFieldShare<F>> {
        let mut evals = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            evals.push(self.get_share_by_idx(i));
        }
        evals
    }
}

pub fn generate_poly_shares_rss<F: Field, R: Rng>(
    poly: &DenseMultilinearExtension<F>,
    rng: &mut R,
) -> [DenseMultilinearExtension<F>; 3] {
    if poly.num_vars == 0 {
        return [
            DenseMultilinearExtension::<F>::zero(),
            DenseMultilinearExtension::<F>::zero(),
            DenseMultilinearExtension::<F>::zero(),
        ];
    }
    let num_vars = poly.num_vars;
    let p_share_0 = DenseMultilinearExtension::<F>::rand(num_vars, rng);
    let p_share_1 = DenseMultilinearExtension::<F>::rand(num_vars, rng);
    let p_share_2 = (poly - &p_share_0) - p_share_1.clone();

    [p_share_0, p_share_1, p_share_2]
}
