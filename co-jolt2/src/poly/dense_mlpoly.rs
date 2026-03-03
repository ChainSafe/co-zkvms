use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::{additive::AdditiveShare, rep3};
use std::ops::{Index, Range};
use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::Rep3MultilinearPolynomial;
use jolt_core::{poly::dense_mlpoly::DensePolynomial, utils::math::Math};

use rayon::prelude::*;

#[derive(Debug, Clone, Default, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub struct Rep3DensePolynomial<F: JoltField> {
    num_vars: usize,
    pub(crate) coeffs: Arc<Vec<Rep3PrimeFieldShare<F>>>,
    bound_coeffs: Vec<Rep3PrimeFieldShare<F>>,
    binding_scratch_space: Option<Vec<Rep3PrimeFieldShare<F>>>,
    len: usize,
    chunk_range: (usize, usize),
    pub(super) global_chunk_range: Option<(usize, usize)>,
    full_len: usize,
}

impl<F: JoltField> Rep3DensePolynomial<F> {
    pub fn new(coeffs: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        let num_vars = coeffs.len().log_2();

        Rep3DensePolynomial {
            num_vars,
            len: coeffs.len(),
            full_len: coeffs.len(),
            chunk_range: (0, coeffs.len()),
            coeffs: Arc::new(coeffs),
            bound_coeffs: vec![],
            binding_scratch_space: None,
            global_chunk_range: None,
        }
    }

    pub fn from_coeffs_arc(coeffs: Arc<Vec<Rep3PrimeFieldShare<F>>>) -> Self {
        let len = coeffs.len();
        let num_vars = len.log_2();

        Rep3DensePolynomial {
            num_vars,
            len,
            full_len: len,
            chunk_range: (0, len),
            coeffs,
            bound_coeffs: vec![],
            binding_scratch_space: None,
            global_chunk_range: None,
        }
    }

    pub(super) fn new_shard(
        coeffs: Vec<Rep3PrimeFieldShare<F>>,
        full_len: usize,
        log_num_workers: usize,
        worker_idx: usize,
    ) -> Self {
        let shard_nv = full_len.log_2() - log_num_workers;
        let num_vars = coeffs.len().log_2();
        let chunk_size = 1 << shard_nv;

        let chunk_range = if shard_nv == num_vars {
            (0, coeffs.len())
        } else {
            (worker_idx * chunk_size, (worker_idx + 1) * chunk_size)
        };

        let global_chunk_range = Some((worker_idx * chunk_size, (worker_idx + 1) * chunk_size));

        Rep3DensePolynomial {
            num_vars: shard_nv,
            len: chunk_size,
            chunk_range,
            coeffs: Arc::new(coeffs),
            bound_coeffs: vec![],
            binding_scratch_space: None,
            global_chunk_range,
            full_len,
        }
    }

    pub fn new_padded(evals: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        let mut poly_coeffs = evals;
        while !poly_coeffs.len().is_power_of_two() {
            poly_coeffs.push(Rep3PrimeFieldShare::zero_share());
        }
        let num_vars = poly_coeffs.len().log_2();
        Rep3DensePolynomial {
            num_vars,
            len: 1 << num_vars,
            full_len: 1 << num_vars,
            chunk_range: (0, poly_coeffs.len()),
            coeffs: Arc::new(poly_coeffs),
            bound_coeffs: vec![],
            binding_scratch_space: None,
            global_chunk_range: None,
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

    pub fn into_distributed_commit_form(&self) -> DensePolynomial<F> {
        let mut coeffs = vec![ark_ff::Zero::zero(); self.full_len];
        coeffs.splice(
            self.global_chunk_range
                .map(|(start, end)| start..end)
                .unwrap_or(self.chunk_range.0..self.chunk_range.1),
            self.coeffs[self.chunk_range.0..self.chunk_range.1]
                .iter()
                .map(|share| share.a),
        );
        DensePolynomial::new(coeffs)
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
        let chis = jolt_core::poly::eq_poly::EqPolynomial::evals(r);
        assert_eq!(chis.len(), self.coeffs_ref().len());
        self.evaluate_at_chi_optimized(&chis)
    }

    pub fn evaluate_at_chi(&self, chis: &[F]) -> AdditiveShare<F> {
        self.coeffs_ref()
            .par_iter()
            .zip_eq(chis.par_iter())
            .map(|(&eval, &chi)| eval.into_additive() * chi)
            .sum()
    }

    pub fn evaluate_at_chi_optimized(&self, chis: &[F]) -> AdditiveShare<F> {
        self.coeffs_ref()
            .par_iter()
            .zip_eq(chis.par_iter())
            .map(|(&eval, &chi)| eval.into_additive().mul_public_01_optimized(chi))
            .sum()
    }

    pub fn evaluate_at_chi_optimized_full(&self, chis: &[F]) -> AdditiveShare<F> {
        self.coeffs
            .par_iter()
            .zip_eq(chis.par_iter())
            .map(|(&eval, &chi)| eval.into_additive().mul_public_01_optimized(chi))
            .sum()
    }

    pub fn batch_evaluate(polys: &[&Self], r: &[F]) -> (Vec<AdditiveShare<F>>, Vec<F>) {
        let eq = jolt_core::poly::eq_poly::EqPolynomial::evals(r);

        let evals: Vec<_> = polys
            .into_par_iter()
            .map(|&poly| poly.evaluate_at_chi_optimized(&eq))
            .collect();
        (evals, eq)
    }

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

    pub fn full_len(&self) -> usize {
        self.full_len
    }

    pub fn as_full_poly(mut self) -> Self {
        self.chunk_range = (0, self.full_len);
        self.global_chunk_range = Some((0, self.full_len));
        self
    }

    pub fn shard_global_range(&self) -> Range<usize> {
        if let Some((start, end)) = self.global_chunk_range {
            start..end
        } else {
            0..self.len
        }
    }

    pub fn shard_local_range(&self) -> Range<usize> {
        self.chunk_range.0..self.chunk_range.1
    }

    pub fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    pub fn get_coeff(&self, index: usize) -> Rep3PrimeFieldShare<F> {
        self.coeffs[self.chunk_range.0 + index]
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

    pub fn bound_coeffs(&self) -> &[Rep3PrimeFieldShare<F>] {
        if self.is_bound() {
            &self.bound_coeffs
        } else {
            &self.coeffs[self.chunk_range.0..self.chunk_range.1]
        }
    }

    pub fn zero() -> Self {
        Rep3DensePolynomial {
            num_vars: 0,
            len: 1,
            full_len: 1,
            chunk_range: (0, 1),
            coeffs: Arc::new(vec![Rep3PrimeFieldShare::zero()]),
            bound_coeffs: vec![],
            binding_scratch_space: None,
            global_chunk_range: None,
        }
    }

    pub fn poly_shard_for_worker(
        poly: &Rep3DensePolynomial<F>,
        shard_nv: usize,
        worker_idx: usize,
    ) -> Rep3MultilinearPolynomial<F> {
        assert!(shard_nv <= poly.get_num_vars());
        if poly.get_num_vars() == shard_nv {
            return poly.clone().into();
        }

        assert!(!poly.is_bound());
        let chunk_size = 1 << shard_nv;

        let mut poly = poly.clone();
        let offset = worker_idx * chunk_size;
        poly.chunk_range = (offset, offset + chunk_size);
        poly.len = chunk_size;
        poly.num_vars = chunk_size.log_2();

        Rep3MultilinearPolynomial::shared(poly)
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

    pub fn bind(&mut self, r: F, order: BindingOrder) {
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
            if self.binding_scratch_space.is_none() {
                self.binding_scratch_space = Some(unsafe_allocate_zero_share_vec(n));
            }
            let scratch_space = self.binding_scratch_space.as_mut().unwrap();

            match order {
                BindingOrder::LowToHigh => {
                    scratch_space
                        .par_iter_mut()
                        .take(n)
                        .enumerate()
                        .for_each(|(i, z)| {
                            let m = self.coeffs[offset + 2 * i + 1] - self.coeffs[offset + 2 * i];
                            *z = self.coeffs[offset + 2 * i] + rep3::arithmetic::mul_public(m, r)
                        });
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.coeffs[offset..cutoff].split_at(n);
                    scratch_space
                        .par_iter_mut()
                        .take(n)
                        .enumerate()
                        .for_each(|(i, z)| {
                            let m = right[i] - left[i];
                            *z = left[i] + rep3::arithmetic::mul_public(m, r)
                        });
                }
            }
            std::mem::swap(&mut self.bound_coeffs, scratch_space);
        }
        self.num_vars -= 1;
        self.len = n;
    }

    /// Warning: returns the additive share.
    pub fn final_sumcheck_claim(&self) -> Rep3PrimeFieldShare<F> {
        assert_eq!(self.len, 1);
        // When the polynomial was created at length 1 (e.g. from RaPolynomialRound3::bind),
        // bound_coeffs is empty and the value lives in coeffs[0].
        if self.bound_coeffs.is_empty() {
            self.coeffs[0]
        } else {
            self.bound_coeffs[0]
        }
    }
}

impl<F: JoltField> Index<usize> for Rep3DensePolynomial<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coeffs[index]
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
    let [s0, s1, s2] = poly_shares.try_into().unwrap();
    itertools::multizip((s0, s1, s2))
        .map(|(a, b, c)| combine_poly_shares_rep3(vec![a, b, c]))
        .collect()
}

pub fn unsafe_allocate_zero_share_vec<F: JoltField + Sized>(
    size: usize,
) -> Vec<Rep3PrimeFieldShare<F>> {
    // Check for safety of 0 allocation
    unsafe {
        let value = &Rep3PrimeFieldShare::<F>::zero_share();
        let ptr = value as *const Rep3PrimeFieldShare<F> as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

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
