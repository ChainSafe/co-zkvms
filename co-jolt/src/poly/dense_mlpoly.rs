use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::cfg_iter;
use jolt_core::{
    poly::multilinear_polynomial::{
        BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
    },
    utils,
};
use mpc_core::protocols::rep3;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use rand::Rng;
use std::ops::Index;

use crate::poly::Rep3MultilinearPolynomial;
use jolt_core::{
    field::JoltField,
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    utils::math::Math,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug, Clone, Default, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub struct Rep3DensePolynomial<F: JoltField> {
    // pub party_id: usize,
    pub num_vars: usize,
    pub evals: Vec<Rep3PrimeFieldShare<F>>,
    // pub a: DensePolynomial<F>,
    // pub b: DensePolynomial<F>,
}

impl<F: JoltField> Rep3DensePolynomial<F> {
    pub fn new(evals: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        let num_vars = evals.len().log_2();

        Rep3DensePolynomial { num_vars, evals }
    }
    pub fn new_padded(evals: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_evals = evals;
        while !(utils::is_power_of_two(poly_evals.len())) {
            poly_evals.push(Rep3PrimeFieldShare::zero_share());
        }
        let num_vars = poly_evals.len().log_2();
        Rep3DensePolynomial {
            num_vars,
            evals: poly_evals,
        }
    }

    pub fn from_vec_shares(a: Vec<F>, b: Vec<F>) -> Self {
        let evals = a
            .into_iter()
            .zip(b.into_iter())
            .map(|(a, b)| Rep3PrimeFieldShare::new(a, b))
            .collect();
        Rep3DensePolynomial::new(evals)
    }

    pub fn from_poly_shares(a: DensePolynomial<F>, b: DensePolynomial<F>) -> Self {
        assert_eq!(a.evals_ref().len(), b.evals_ref().len());
        Rep3DensePolynomial::from_vec_shares(
            a.evals()[..1 << a.get_num_vars()].to_vec(),
            b.evals()[..1 << b.get_num_vars()].to_vec(),
        )
    }

    pub fn into_poly_shares(self) -> (DensePolynomial<F>, DensePolynomial<F>) {
        let mut a = Vec::with_capacity(1 << self.num_vars);
        let mut b = Vec::with_capacity(1 << self.num_vars);
        for share in self.evals {
            a.push(share.a);
            b.push(share.b);
        }
        (DensePolynomial::new(a), DensePolynomial::new(b))
    }

    pub fn copy_poly_shares(&self) -> (DensePolynomial<F>, DensePolynomial<F>) {
        let mut a = Vec::with_capacity(1 << self.num_vars);
        let mut b = Vec::with_capacity(1 << self.num_vars);
        for share in &self.evals {
            a.push(share.a);
            b.push(share.b);
        }
        (DensePolynomial::new(a), DensePolynomial::new(b))
    }

    pub fn copy_share_a(&self) -> DensePolynomial<F> {
        let mut a = Vec::with_capacity(1 << self.num_vars);
        for share in &self.evals {
            a.push(share.a);
        }
        DensePolynomial::new(a)
    }

    pub fn split(&self, idx: usize) -> (Self, Self) {
        let (a, b) = self.copy_poly_shares();
        let (left_a, right_a) = a.split(idx);
        let (left_b, right_b) = b.split(idx);
        assert!(idx < self.len());
        (
            Self::from_poly_shares(left_a, left_b),
            Self::from_poly_shares(right_a, right_b),
        )
    }

    #[inline]
    pub fn sumcheck_evals(
        &self,
        index: usize,
        degree: usize,
        order: BindingOrder,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let (a, b) = self.copy_poly_shares();

        let evals_a = MultilinearPolynomial::LargeScalars(a).sumcheck_evals(index, degree, order);
        let evals_b = MultilinearPolynomial::LargeScalars(b).sumcheck_evals(index, degree, order);

        assert_eq!(evals_a.len(), evals_b.len());
        evals_a
            .into_iter()
            .zip(evals_b.into_iter())
            .map(|(a, b)| Rep3PrimeFieldShare::new(a, b))
            .collect()
    }

    pub fn split_evals(
        &self,
        idx: usize,
    ) -> (&[Rep3PrimeFieldShare<F>], &[Rep3PrimeFieldShare<F>]) {
        (&self.evals[..idx], &self.evals[idx..])
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let (mut a, mut b) = self.copy_poly_shares();
        rayon::join(|| a.bound_poly_var_top(r), || b.bound_poly_var_top(r));

        *self = Self::from_poly_shares(a, b);
    }

    pub fn bound_poly_var_bot(&mut self, r: &F) {
        let (mut a, mut b) = self.copy_poly_shares();
        rayon::join(|| a.bound_poly_var_bot(r), || b.bound_poly_var_bot(r));

        *self = Self::from_poly_shares(a, b);
    }

    pub fn bound_poly_var_bot_01_optimized(&mut self, r: &F) {
        let (mut a, mut b) = self.copy_poly_shares();
        rayon::join(
            || a.bound_poly_var_bot_01_optimized(r),
            || b.bound_poly_var_bot_01_optimized(r),
        );

        *self = Self::from_poly_shares(a, b);
    }

    pub fn bound_poly_var_top_zero_optimized(&mut self, r: &F) {
        let (mut a, mut b) = self.copy_poly_shares();
        rayon::join(
            || a.bound_poly_var_top_zero_optimized(r),
            || b.bound_poly_var_top_zero_optimized(r),
        );

        *self = Self::from_poly_shares(a, b);
    }

    pub fn fix_var_top_many_ones(&mut self, r: &F) {
        let (mut a, mut b) = self.copy_poly_shares();
        a.bound_poly_var_top_many_ones(r);
        b.bound_poly_var_top_many_ones(r);

        *self = Self::from_poly_shares(a, b);
    }

    pub fn new_poly_from_fix_var_top(&self, r: &F) -> Self {
        let (a, b) = self.copy_poly_shares();
        let a = a.new_poly_from_bound_poly_var_top(r);
        let b = b.new_poly_from_bound_poly_var_top(r); // TODO: check if this is correct with rep3 shares

        Self::from_poly_shares(a, b)
    }

    pub fn evaluate(&self, r: &[F]) -> F {
        let chis = EqPolynomial::evals(r);
        assert_eq!(chis.len(), self.evals_ref().len());
        self.evaluate_at_chi(&chis)
    }

    pub fn evaluate_at_chi(&self, chis: &[F]) -> F {
        self.evals
            .par_iter()
            .zip_eq(chis.par_iter())
            .map(|(eval, chi)| rep3::arithmetic::mul_public(*eval, *chi).into_additive())
            .sum()
    }

    #[tracing::instrument(skip_all, name = "Rep3DensePolynomial::batch_evaluate")]
    pub fn batch_evaluate(polys: &[&Self], r: &[F]) -> (Vec<F>, Vec<F>) {
        let eq = EqPolynomial::evals(r);

        let evals: Vec<F> = polys
            .into_par_iter()
            .map(|&poly| poly.evaluate_at_chi(&eq))
            .collect();
        (evals, eq)
    }

    #[tracing::instrument(skip_all, name = "Rep3DensePolynomial::linear_combination")]
    pub fn linear_combination(polynomials: &[&Self], coefficients: &[F]) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());

        let max_length = polynomials.iter().map(|poly| poly.len()).max().unwrap();
        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(max_length);
        let chunk_size = (max_length / num_chunks).max(1);

        let lc_coeffs: Vec<_> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let index = chunk_index * chunk_size;
                let mut chunk = vec![Rep3PrimeFieldShare::zero_share(); chunk_size];

                for (coeff, poly) in coefficients.iter().zip(polynomials.iter()) {
                    let poly_len = poly.len();
                    if index >= poly_len {
                        continue;
                    }

                    let poly_evals = &poly.evals_ref()[index..];
                    for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                        *rlc += rep3::arithmetic::mul_public(*poly_eval, *coeff);
                    }
                }
                chunk
            })
            .collect();

        Rep3DensePolynomial::new(lc_coeffs)
    }

    pub fn dot_product_with_public(&self, other: &[F]) -> Rep3PrimeFieldShare<F> {
        self.evals
            .par_iter()
            .zip_eq(other.par_iter())
            .map(|(a_i, b_i)| rep3::arithmetic::mul_public(*a_i, *b_i))
            .sum::<Rep3PrimeFieldShare<F>>()
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        1 << self.get_num_vars()
    }

    pub fn is_bound(&self) -> bool {
        unimplemented!()
    }

    pub fn evals_ref(&self) -> &[Rep3PrimeFieldShare<F>] {
        &self.evals
    }

    pub fn zero() -> Self {
        Rep3DensePolynomial {
            num_vars: 0,
            evals: vec![Rep3PrimeFieldShare::zero()],
        }
    }

    pub fn split_poly(
        polys: Rep3DensePolynomial<F>,
        log_workers: usize,
    ) -> Vec<Rep3MultilinearPolynomial<F>> {
        let nv = polys.num_vars - log_workers;
        let chunk_size = 1 << nv;
        let mut res = Vec::new();

        let (a, b) = polys.copy_poly_shares();
        let mut a_evals = a.Z;
        let mut b_evals = b.Z;

        for _ in 0..1 << log_workers {
            res.push(Rep3MultilinearPolynomial::shared(
                Rep3DensePolynomial::<F>::from_poly_shares(
                    DensePolynomial::<F>::new(a_evals.drain(..chunk_size).collect()),
                    DensePolynomial::<F>::new(b_evals.drain(..chunk_size).collect()),
                ),
            ))
        }

        res
    }
}

impl<F: JoltField> PolynomialBinding<F> for Rep3DensePolynomial<F> {
    fn is_bound(&self) -> bool {
        unimplemented!()
    }

    #[inline]
    fn bind(&mut self, r: F, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh => self.bound_poly_var_bot(&r),
            BindingOrder::HighToLow => self.bound_poly_var_top(&r),
        }
    }

    #[inline]
    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh => self.bound_poly_var_bot_01_optimized(&r),
            BindingOrder::HighToLow => self.bound_poly_var_top_zero_optimized(&r),
        }
    }

    /// Warning: returns the additive share.
    fn final_sumcheck_claim(&self) -> F {
        self.evals[0].into_additive()
    }
}

impl<F: JoltField> Index<usize> for Rep3DensePolynomial<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evals[index]
    }
}

// Implement Index for Range<usize> to support [1..3]
impl<F: JoltField> Index<std::ops::Range<usize>> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        &self.evals[index]
    }
}

// Implement Index for RangeFrom<usize> to support [1..]
impl<F: JoltField> Index<std::ops::RangeFrom<usize>> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.evals[index]
    }
}

// Implement Index for RangeTo<usize> to support [..3]
impl<F: JoltField> Index<std::ops::RangeTo<usize>> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, index: std::ops::RangeTo<usize>) -> &Self::Output {
        &self.evals[index]
    }
}

// Implement Index for RangeFull to support [..]
impl<F: JoltField> Index<std::ops::RangeFull> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, _index: std::ops::RangeFull) -> &Self::Output {
        &self.evals
    }
}

// Implement Index for RangeInclusive<usize> to support [1..=3]
impl<F: JoltField> Index<std::ops::RangeInclusive<usize>> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, index: std::ops::RangeInclusive<usize>) -> &Self::Output {
        &self.evals[index]
    }
}

// Implement Index for RangeToInclusive<usize> to support [..=3]
impl<F: JoltField> Index<std::ops::RangeToInclusive<usize>> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, index: std::ops::RangeToInclusive<usize>) -> &Self::Output {
        &self.evals[index]
    }
}

pub fn combine_poly_shares_rep3<F: JoltField>(
    poly_shares: Vec<Rep3DensePolynomial<F>>,
) -> DensePolynomial<F> {
    assert_eq!(poly_shares.len(), 3);
    let [s0, s1, s2] = poly_shares.try_into().unwrap();
    let a = rep3::combine_field_elements(&s0.evals, &s1.evals, &s2.evals);
    DensePolynomial::new(a)
}

pub fn combine_polys_shares_rep3<F: JoltField>(
    poly_shares: Vec<Vec<Rep3DensePolynomial<F>>>,
) -> Vec<DensePolynomial<F>> {
    assert_eq!(poly_shares.len(), 3);
    let [s0, s1, s2] = poly_shares.try_into().unwrap(); // TODO: check if this is correct
    itertools::multizip((s0, s1, s2))
        .map(|(a, b, c)| combine_poly_shares_rep3(vec![a, b, c]))
        .collect()
}

pub fn generate_poly_shares_rep3<F: JoltField, R: Rng>(
    poly: &MultilinearPolynomial<F>,
    rng: &mut R,
) -> Vec<Rep3MultilinearPolynomial<F>> {
    let dense_poly = DensePolynomial::new_padded(poly.coeffs_as_field_elements());

    let num_vars = poly.get_num_vars();
    if num_vars == 0 {
        return vec![
            Rep3DensePolynomial::<F>::zero().into(),
            Rep3DensePolynomial::<F>::zero().into(),
            Rep3DensePolynomial::<F>::zero().into(),
        ];
    }
    let t0 =
        DensePolynomial::<F>::new(itertools::repeat_n(F::random(rng), 1 << num_vars).collect());
    let t1 =
        DensePolynomial::<F>::new(itertools::repeat_n(F::random(rng), 1 << num_vars).collect());
    let t2 = (dense_poly - &t0) - &t1;

    let p_share_0 = Rep3DensePolynomial::<F>::from_poly_shares(t0.clone(), t2.clone());
    let p_share_1 = Rep3DensePolynomial::<F>::from_poly_shares(t1.clone(), t0);
    let p_share_2 = Rep3DensePolynomial::<F>::from_poly_shares(t2, t1);

    vec![p_share_0.into(), p_share_1.into(), p_share_2.into()]
}

pub fn generate_poly_shares_rep3_vec<F: JoltField, R: Rng>(
    polys: &[MultilinearPolynomial<F>],
    rng: &mut R,
) -> Vec<Vec<Rep3MultilinearPolynomial<F>>> {
    if polys.is_empty() {
        return Vec::new();
    }

    let poly_shares: Vec<Vec<_>> = polys
        .iter()
        .map(|poly| generate_poly_shares_rep3(poly, rng))
        .collect();

    let num_shares = poly_shares[0].len();
    let num_polys = polys.len();

    let mut shares_of_polys: Vec<Vec<Rep3MultilinearPolynomial<F>>> = (0..num_shares)
        .map(|_| Vec::with_capacity(num_polys))
        .collect();

    for poly_shares in poly_shares {
        for (i, share) in poly_shares.into_iter().enumerate() {
            shares_of_polys[i].push(share);
        }
    }

    shares_of_polys
}

#[cfg(test)]
mod tests {
    use ark_ff::{Field, One};
    use ark_std::test_rng;

    use super::*;

    type F = ark_bn254::Fr;

    // #[test]
    // fn test_share_and_combine_poly_rep3() {
    //     let mut rng = test_rng();
    //     let poly = DensePolynomial::<F>::rand(10, &mut rng);
    //     let shares = generate_poly_shares_rep3(&poly, &mut rng);
    //     let combined = combine_poly_shares_rep3(vec![shares.0, shares.1, shares.2]);
    //     assert_eq!(poly, combined);
    // }
}
