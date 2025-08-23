use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::{
    poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    utils,
};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::{additive::AdditiveShare, rep3};
use rand::{Rng, SeedableRng};
use std::ops::Index;
use std::sync::Arc;

use crate::poly::Rep3MultilinearPolynomial;
use crate::field::JoltField;
use jolt_core::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    utils::math::Math,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug, Clone, Default, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub struct Rep3DensePolynomial<F: JoltField> {
    // pub party_id: usize,
    num_vars: usize,
    coeffs: Arc<Vec<Rep3PrimeFieldShare<F>>>,
    bound_coeffs: Vec<Rep3PrimeFieldShare<F>>,
    binding_scratch_space: Option<Vec<Rep3PrimeFieldShare<F>>>,
    len: usize,
    chunk_range: (usize, usize),
}

impl<F: JoltField> Rep3DensePolynomial<F> {
    pub fn new(coeffs: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        let num_vars = coeffs.len().log_2();

        Rep3DensePolynomial {
            num_vars,
            len: coeffs.len(),
            chunk_range: (0, coeffs.len()),
            coeffs: Arc::new(coeffs),
            bound_coeffs: vec![],
            binding_scratch_space: None,
        }
    }

    pub fn from_bound_coeffs(bound_coeffs: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        Rep3DensePolynomial {
            num_vars: bound_coeffs.len().log_2(),
            len: bound_coeffs.len(),
            chunk_range: (0, bound_coeffs.len()),
            bound_coeffs: bound_coeffs.clone(),
            coeffs: Arc::new(bound_coeffs),
            binding_scratch_space: None,
        }
    }

    pub fn new_padded(evals: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_coeffs = evals;
        while !(utils::is_power_of_two(poly_coeffs.len())) {
            poly_coeffs.push(Rep3PrimeFieldShare::zero_share());
        }
        let num_vars = poly_coeffs.len().log_2();
        Rep3DensePolynomial {
            num_vars,
            len: 1 << num_vars,
            chunk_range: (0, poly_coeffs.len()),
            coeffs: Arc::new(poly_coeffs),
            bound_coeffs: vec![],
            binding_scratch_space: None,
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
        let (a, b) = Arc::try_unwrap(self.coeffs)
            .unwrap()
            .into_iter()
            .map(|share| (share.a, share.b))
            .unzip();
        (DensePolynomial::new(a), DensePolynomial::new(b))
    }

    #[inline]
    pub fn copy_share_a(&self) -> DensePolynomial<F> {
        DensePolynomial::new(
            self.coeffs[self.chunk_range.0..self.chunk_range.1]
                .par_iter()
                .map(|share| share.a)
                .collect(),
        )
    }

    #[inline]
    pub fn sumcheck_evals(
        &self,
        index: usize,
        degree: usize,
        order: BindingOrder,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let mut evals = vec![Rep3PrimeFieldShare::zero_share(); degree];
        match order {
            BindingOrder::LowToHigh => {
                evals[0] = self.get_bound_coeff(2 * index);
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(2 * index + 1);
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
            BindingOrder::HighToLow => {
                evals[0] = self.get_bound_coeff(index);
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(index + self.len() / 2);
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
        }
        evals
    }

    pub fn evaluate(&self, r: &[F]) -> AdditiveShare<F> {
        let chis = EqPolynomial::evals(r);
        assert_eq!(chis.len(), self.coeffs_ref().len());
        self.evaluate_at_chi_optimized(&chis)
    }

    #[tracing::instrument(
        skip_all,
        name = "Rep3DensePolynomial::evaluate_at_chi",
        level = "trace"
    )]
    pub fn evaluate_at_chi(&self, chis: &[F]) -> AdditiveShare<F> {
        self.coeffs_ref()
            .par_iter()
            .zip_eq(chis.par_iter())
            .map(|(&eval, &chi)| eval.into_additive() * chi)
            .sum()
    }

    #[tracing::instrument(
        skip_all,
        name = "Rep3DensePolynomial::evaluate_at_chi",
        level = "trace"
    )]
    pub fn evaluate_at_chi_optimized(&self, chis: &[F]) -> AdditiveShare<F> {
        self.coeffs_ref()
            .par_iter()
            .zip_eq(chis.par_iter())
            .map(|(&eval, &chi)| {
                eval.into_additive().mul_public_01_optimized(chi)
            })
            .sum()
    }

    #[tracing::instrument(skip_all, name = "Rep3DensePolynomial::batch_evaluate")]
    pub fn batch_evaluate(polys: &[&Self], r: &[F]) -> (Vec<AdditiveShare<F>>, Vec<F>) {
        let eq = EqPolynomial::evals(r);

        let evals: Vec<_> = polys
            .into_par_iter()
            .map(|&poly| poly.evaluate_at_chi_optimized(&eq))
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

                    let poly_evals = &poly.coeffs_ref()[index..];
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
        self.coeffs_ref()
            .par_iter()
            .zip_eq(other.par_iter())
            .map(|(&a_i, &b_i)| rep3::arithmetic::mul_public(a_i, b_i))
            .sum::<Rep3PrimeFieldShare<F>>()
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    pub fn get_bound_coeff(&self, index: usize) -> Rep3PrimeFieldShare<F> {
        if self.is_bound() {
            self.bound_coeffs[index]
        } else {
            self.coeffs[self.chunk_range.0 + index]
        }
    }

    pub fn set_bound_coeff(&mut self, index: usize, eval: Rep3PrimeFieldShare<F>) {
        self.bound_coeffs[index] = eval;
    }

    pub fn coeffs_ref(&self) -> &[Rep3PrimeFieldShare<F>] {
        &self.coeffs[self.chunk_range.0..self.chunk_range.1]
    }

    pub fn zero() -> Self {
        Rep3DensePolynomial {
            num_vars: 0,
            len: 1,
            chunk_range: (0, 1),
            coeffs: Arc::new(vec![Rep3PrimeFieldShare::zero()]),
            bound_coeffs: vec![],
            binding_scratch_space: None,
        }
    }

    pub fn split_poly(
        poly: Rep3DensePolynomial<F>,
        log_workers: usize,
    ) -> Vec<Rep3MultilinearPolynomial<F>> {
        if log_workers == 0 {
            return vec![poly.into()];
        }

        assert!(!poly.is_bound());
        let nv = poly.num_vars - log_workers;
        let chunk_size = 1 << nv;
        let mut res = Vec::new();

        let mut offset = 0;

        for _ in 0..1 << log_workers {
            let mut poly = poly.clone();
            poly.chunk_range = (offset, offset + chunk_size);
            poly.len = chunk_size;
            poly.num_vars = nv;
            offset += chunk_size;

            res.push(Rep3MultilinearPolynomial::shared(poly))
        }

        res
    }
}

impl<F: JoltField> PolynomialBinding<F, Rep3PrimeFieldShare<F>> for Rep3DensePolynomial<F> {
    fn is_bound(&self) -> bool {
        unimplemented!()
    }

    #[inline]
    fn bind(&mut self, r: F, order: BindingOrder) {
        let n = self.len() / 2;
        let offset = self.chunk_range.0;
        let cutoff = self.chunk_range.1;

        if self.is_bound() {
            match order {
                BindingOrder::LowToHigh => {
                    for i in 0..n {
                        self.bound_coeffs[i] = self.bound_coeffs[2 * i]
                            + rep3::arithmetic::mul_public(
                                self.bound_coeffs[2 * i + 1] - self.bound_coeffs[2 * i],
                                r,
                            );
                    }
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.bound_coeffs.split_at_mut(n);

                    left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
                        *a += rep3::arithmetic::mul_public(*b - *a, r);
                    });
                }
            }
        } else {
            match order {
                BindingOrder::LowToHigh => {
                    if self.binding_scratch_space.is_none() {
                        self.binding_scratch_space = Some(unsafe_allocate_zero_share_vec(n));
                    }

                    let scratch_space = self.binding_scratch_space.as_mut().unwrap();

                    scratch_space
                        .par_iter_mut()
                        .take(n)
                        .enumerate()
                        .for_each(|(i, z)| {
                            let m = self.coeffs[offset + 2 * i + 1] - self.coeffs[offset + 2 * i];
                            *z = self.coeffs[offset + 2 * i] + rep3::arithmetic::mul_public(m, r)
                        });

                    std::mem::swap(&mut self.bound_coeffs, scratch_space);
                }
                BindingOrder::HighToLow => {
                    if self.binding_scratch_space.is_none() {
                        self.binding_scratch_space = Some(unsafe_allocate_zero_share_vec(n));
                    }

                    let scratch_space = self.binding_scratch_space.as_mut().unwrap();

                    let (left, right) = self.coeffs[offset..cutoff].split_at(n);

                    scratch_space
                        .par_iter_mut()
                        .take(n)
                        .enumerate()
                        .for_each(|(i, z)| {
                            let m = right[i] - left[i];
                            *z = left[i] + rep3::arithmetic::mul_public(m, r)
                        });

                    std::mem::swap(&mut self.bound_coeffs, scratch_space);
                }
            }
        }
        self.num_vars -= 1;
        self.len = n;
    }

    #[inline]
    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        let n = self.len() / 2;
        let offset = self.chunk_range.0;
        let cutoff = self.chunk_range.1;
        if self.is_bound() {
            match order {
                BindingOrder::LowToHigh => {
                    if self.binding_scratch_space.is_none() {
                        self.binding_scratch_space = Some(unsafe_allocate_zero_share_vec(n));
                    }
                    let binding_scratch_space = self.binding_scratch_space.as_mut().unwrap();

                    binding_scratch_space
                        .par_iter_mut() // TODO: iter_mut
                        .take(n)
                        .enumerate()
                        .for_each(|(i, new_coeff)| {
                            *new_coeff = self.bound_coeffs[2 * i]
                                + rep3::arithmetic::mul_public(
                                    self.bound_coeffs[2 * i + 1] - self.bound_coeffs[2 * i],
                                    r,
                                );
                        });
                    std::mem::swap(&mut self.bound_coeffs, binding_scratch_space);
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.bound_coeffs.split_at_mut(n);

                    left.par_iter_mut()
                        .zip(right.par_iter())
                        .for_each(|(a, b)| *a += rep3::arithmetic::mul_public(*b - *a, r));
                }
            }
        } else {
            match order {
                BindingOrder::LowToHigh => {
                    if self.binding_scratch_space.is_none() {
                        self.binding_scratch_space = Some(unsafe_allocate_zero_share_vec(n));
                    }

                    let scratch_space = self.binding_scratch_space.as_mut().unwrap();

                    scratch_space
                        .par_iter_mut()
                        .take(n)
                        .enumerate()
                        .for_each(|(i, z)| {
                            let m = self.coeffs[offset + 2 * i + 1] - self.coeffs[offset + 2 * i];
                            *z = self.coeffs[offset + 2 * i] + rep3::arithmetic::mul_public(m, r)
                        });

                    std::mem::swap(&mut self.bound_coeffs, scratch_space);
                }
                BindingOrder::HighToLow => {
                    if self.binding_scratch_space.is_none() {
                        self.binding_scratch_space = Some(unsafe_allocate_zero_share_vec(n));
                    }

                    let scratch_space = self.binding_scratch_space.as_mut().unwrap();

                    let (left, right) = self.coeffs[offset..cutoff].split_at(n);

                    scratch_space
                        .par_iter_mut()
                        .take(n)
                        .enumerate()
                        .for_each(|(i, z)| {
                            let m = right[i] - left[i];
                            *z = rep3::arithmetic::mul_public(m, r)
                        });

                    std::mem::swap(&mut self.bound_coeffs, scratch_space);
                }
            }
        }

        self.num_vars -= 1;
        self.len = n;
    }

    /// Warning: returns the additive share.
    fn final_sumcheck_claim(&self) -> Rep3PrimeFieldShare<F> {
        assert_eq!(self.len, 1);
        self.bound_coeffs[0]
    }
}

impl<F: JoltField> Index<usize> for Rep3DensePolynomial<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coeffs[self.chunk_range.0 + index]
    }
}

// Implement Index for Range<usize> to support [1..3]
impl<F: JoltField> Index<std::ops::Range<usize>> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        assert!(index.end <= self.chunk_range.1);
        &self.coeffs[self.chunk_range.0 + index.start..self.chunk_range.0 + index.end]
    }
}

// Implement Index for RangeFrom<usize> to support [1..]
impl<F: JoltField> Index<std::ops::RangeFrom<usize>> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.coeffs[self.chunk_range.0 + index.start..self.chunk_range.1]
    }
}

// Implement Index for RangeTo<usize> to support [..3]
impl<F: JoltField> Index<std::ops::RangeTo<usize>> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, index: std::ops::RangeTo<usize>) -> &Self::Output {
        assert!(index.end <= self.chunk_range.1);
        &self.coeffs[self.chunk_range.0..self.chunk_range.0 + index.end]
    }
}

// Implement Index for RangeFull to support [..]
impl<F: JoltField> Index<std::ops::RangeFull> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, _index: std::ops::RangeFull) -> &Self::Output {
        &self.coeffs[self.chunk_range.0..self.chunk_range.1]
    }
}

// Implement Index for RangeInclusive<usize> to support [1..=3]
impl<F: JoltField> Index<std::ops::RangeInclusive<usize>> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, index: std::ops::RangeInclusive<usize>) -> &Self::Output {
        assert!(*index.end() <= self.chunk_range.1);
        &self.coeffs[self.chunk_range.0 + *index.start()..=self.chunk_range.0 + *index.end()]
    }
}

// Implement Index for RangeToInclusive<usize> to support [..=3]
impl<F: JoltField> Index<std::ops::RangeToInclusive<usize>> for Rep3DensePolynomial<F> {
    type Output = [Rep3PrimeFieldShare<F>];

    fn index(&self, index: std::ops::RangeToInclusive<usize>) -> &Self::Output {
        assert!(index.end <= self.chunk_range.1);
        &self.coeffs[self.chunk_range.0..self.chunk_range.0 + index.end]
    }
}

pub fn combine_poly_shares_rep3<F: JoltField>(
    poly_shares: Vec<Rep3DensePolynomial<F>>,
) -> DensePolynomial<F> {
    assert_eq!(poly_shares.len(), 3);
    let [s0, s1, s2] = poly_shares.try_into().unwrap();
    let a = rep3::combine_field_elements(s0.coeffs_ref(), s1.coeffs_ref(), s2.coeffs_ref());
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
    let mut rng1 = rand_chacha::ChaCha12Rng::from_seed(rng.r#gen());
    let mut rng2 = rand_chacha::ChaCha12Rng::from_seed(rng.r#gen());
    let (t0, t1) = rayon::join(
        || {
            DensePolynomial::<F>::new(
                itertools::repeat_n(F::random(&mut rng1), 1 << num_vars).collect(),
            )
        },
        || {
            DensePolynomial::<F>::new(
                itertools::repeat_n(F::random(&mut rng2), 1 << num_vars).collect(),
            )
        },
    );
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

    let rngs: Vec<_> = (0..polys.len())
        .map(|_| rand_chacha::ChaCha12Rng::from_seed(rng.r#gen()))
        .collect();

    let polys_shares: Vec<Vec<_>> = polys
        .into_par_iter()
        .zip(rngs.into_par_iter())
        .map(|(poly, mut rng)| generate_poly_shares_rep3(poly, &mut rng))
        .collect();

    let num_shares = polys_shares[0].len();
    let num_polys = polys.len();

    let shares_of_polys = polys_shares
        .into_par_iter()
        .fold(
            || vec![Vec::with_capacity(num_polys); num_shares],
            |mut acc, poly_shares| {
                for (i, share) in poly_shares.into_iter().enumerate() {
                    acc[i].push(share);
                }
                acc
            },
        )
        .reduce(
            || vec![Vec::with_capacity(num_polys); num_shares],
            |mut acc1, mut acc2| {
                for i in 0..num_shares {
                    acc1[i].extend(std::mem::take(&mut acc2[i]));
                }
                acc1
            },
        );

    shares_of_polys
}

pub fn unsafe_allocate_zero_share_vec<F: JoltField + Sized>(
    size: usize,
) -> Vec<Rep3PrimeFieldShare<F>> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    // Check for safety of 0 allocation
    unsafe {
        let value = &Rep3PrimeFieldShare::<F>::zero_share();
        let ptr = value as *const Rep3PrimeFieldShare<F> as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<Rep3PrimeFieldShare<F>>;
    unsafe {
        let layout = std::alloc::Layout::array::<Rep3PrimeFieldShare<F>>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut Rep3PrimeFieldShare<F>;

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
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
