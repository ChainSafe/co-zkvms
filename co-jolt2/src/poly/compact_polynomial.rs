use std::ops::Index;

use jolt_core::field::JoltField;
use crate::poly::Polynomial;
use crate::utils::future_ring::{FutureOp, FutureRep3Ring, Rep3RingFutureExt};
use crate::utils::types::Rep3Value;
use eyre::Context;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, PolynomialBinding, PolynomialEvaluation,
};
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{arithmetic, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::ring::int_ring::IntRing2k;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rand::{distributions::Standard, prelude::Distribution};
use rayon::prelude::*;
use jolt_core::utils::math::Math;

/// Compact polynomials are used to store coefficients of small scalars (in a 2^k ring).
/// They have two representations:
/// 1. `coeffs` is a vector of ring shares (small scalars in a 2^k ring)
/// 2. `bound_coeffs` is a vector of field shares
///
/// They are often initialized with `coeffs` and then converted to `bound_coeffs`
/// when binding the polynomial for sumcheck.
#[derive(Default, Debug, PartialEq)]
pub struct CompactPolynomial<T: IntRing2k, F: JoltField> {
    num_vars: usize,
    len: usize,
    pub coeffs: Vec<Rep3RingShare<T>>,
    pub bound_coeffs: Vec<Rep3PrimeFieldShare<F>>,
    // Pending ring->field casts for the first bind when `coeffs` are still ring shares.
    // We attach the public scalar multiplier as Args so `finilize_bound` can apply it after casting.
    bound_coeffs_fut: Vec<FutureRep3Ring<T, Rep3PrimeFieldShare<F>, F>>,
    binding_scratch_space: Option<Vec<Rep3PrimeFieldShare<F>>>,
}

impl<T: IntRing2k, F: JoltField> CompactPolynomial<T, F> {
    pub fn from_coeffs(coeffs: Vec<Rep3RingShare<T>>) -> Self {
        assert!(
            coeffs.len().is_power_of_two(),
            "Multilinear polynomials must be made from a power of 2 (not {})",
            coeffs.len()
        );

        Self {
            num_vars: coeffs.len().log_2(),
            len: coeffs.len(),
            coeffs,
            bound_coeffs: vec![],
            bound_coeffs_fut: vec![],
            binding_scratch_space: None,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &Rep3RingShare<T>> {
        self.coeffs.iter()
    }

    fn ensure_no_pending_casts(&self) {
        assert!(
            self.bound_coeffs_fut.is_empty(),
            "CompactPolynomial has pending ring->field casts; call `finilize_bound` first"
        );
    }

    fn ensure_bound_ready(&self) {
        self.ensure_no_pending_casts();
        assert!(
            !self.bound_coeffs.is_empty(),
            "CompactPolynomial is not bound yet; call `bind` then `finilize_bound`"
        );
    }

    #[inline]
    fn dot_product_with_public(&self, other: &[F]) -> Rep3PrimeFieldShare<F> {
        self.bound_coeffs[..self.len]
            .par_iter()
            .zip_eq(other.par_iter())
            .map(|(&a_i, &b_i)| arithmetic::mul_public(a_i, b_i))
            .sum::<Rep3PrimeFieldShare<F>>()
    }

    pub fn get_bound_coeff(&self, index: usize) -> Rep3PrimeFieldShare<F> {
        self.ensure_bound_ready();
        self.bound_coeffs[index]
    }

    pub fn coeffs_as_field_elements(&self) -> Vec<Rep3PrimeFieldShare<F>> {
        self.ensure_bound_ready();
        self.bound_coeffs[..self.len].to_vec()
    }

    pub fn split_eq_evaluate(
        &self,
        r_len: usize,
        eq_one: &[F],
        eq_two: &[F],
    ) -> Rep3PrimeFieldShare<F> {
        self.ensure_bound_ready();
        const PARALLEL_THRESHOLD: usize = 16;
        if r_len < PARALLEL_THRESHOLD {
            self.evaluate_split_eq_serial(eq_one, eq_two)
        } else {
            self.evaluate_split_eq_parallel(eq_one, eq_two)
        }
    }

    fn evaluate_split_eq_parallel(&self, eq_one: &[F], eq_two: &[F]) -> Rep3PrimeFieldShare<F> {
        (0..eq_one.len())
            .into_par_iter()
            .map(|x1| {
                let partial_sum = (0..eq_two.len())
                    .into_par_iter()
                    .map(|x2| {
                        let idx = x1 * eq_two.len() + x2;
                        arithmetic::mul_public(self.bound_coeffs[idx], eq_two[x2])
                    })
                    .reduce(Rep3PrimeFieldShare::zero_share, |acc, val| acc + val);
                arithmetic::mul_public(partial_sum, eq_one[x1])
            })
            .reduce(Rep3PrimeFieldShare::zero_share, |acc, val| acc + val)
    }

    fn evaluate_split_eq_serial(&self, eq_one: &[F], eq_two: &[F]) -> Rep3PrimeFieldShare<F> {
        (0..eq_one.len())
            .map(|x1| {
                let partial_sum = (0..eq_two.len())
                    .map(|x2| {
                        let idx = x1 * eq_two.len() + x2;
                        arithmetic::mul_public(self.bound_coeffs[idx], eq_two[x2])
                    })
                    .fold(Rep3PrimeFieldShare::zero_share(), |acc, val| acc + val);
                arithmetic::mul_public(partial_sum, eq_one[x1])
            })
            .fold(Rep3PrimeFieldShare::zero_share(), |acc, val| acc + val)
    }

    // Faster evaluation based on
    // https://randomwalks.xyz/publish/fast_polynomial_evaluation.html
    pub fn inside_out_evaluate(&self, r: &[F]) -> Rep3PrimeFieldShare<F> {
        self.ensure_bound_ready();
        const PARALLEL_THRESHOLD: usize = 16;
        assert_eq!(r.len(), self.get_num_vars());
        let m = r.len();
        if m < PARALLEL_THRESHOLD {
            self.inside_out_serial(r)
        } else {
            self.inside_out_parallel(r)
        }
    }

    fn inside_out_serial(&self, r: &[F]) -> Rep3PrimeFieldShare<F> {
        let mut current = self.bound_coeffs[..self.len].to_vec();
        let m = r.len();
        for i in (0..m).rev() {
            let stride = 1 << i;
            let r_val = r[m - 1 - i];
            for j in 0..stride {
                let f0 = current[j];
                let f1 = current[j + stride];
                let slope = f1 - f0;
                current[j] = f0 + arithmetic::mul_public(slope, r_val);
            }
        }
        current[0]
    }

    fn inside_out_parallel(&self, r: &[F]) -> Rep3PrimeFieldShare<F> {
        let mut current = self.bound_coeffs[..self.len].to_vec();
        let m = r.len();
        for i in (0..m).rev() {
            let stride = 1 << i;
            let r_val = r[m - 1 - i];
            let (evals_left, evals_right) = current.split_at_mut(stride);
            let (evals_right, _) = evals_right.split_at_mut(stride);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter())
                .for_each(|(x, y)| {
                    let slope = *y - *x;
                    *x += arithmetic::mul_public(slope, r_val);
                });
        }
        current[0]
    }

    /// Completes the first bind by fulfilling the batched ring->field casts accumulated in
    /// `self.bound_coeffs_fut` and writing the results into `self.bound_coeffs`.
    ///
    /// Must be called after `bind`/`bind_parallel` when `self.is_bound()` was previously `false`.
    pub fn finilize_bound<N: Rep3NetworkWorker>(
        &mut self,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<()>
    where
        Standard: Distribution<T>,
    {
        if self.bound_coeffs_fut.is_empty() {
            return Ok(());
        }

        let contributions: Vec<Rep3PrimeFieldShare<F>> = std::mem::take(&mut self.bound_coeffs_fut)
            .fulfill_batched(io_ctx, |share, scalar| {
                arithmetic::mul_public(share, scalar)
            })
            .context("compact polynomial ring->field cast failed")?;

        eyre::ensure!(
            contributions.len() == 2 * self.len,
            "invalid compact polynomial finalize: expected {} contributions, got {}",
            2 * self.len,
            contributions.len()
        );

        self.bound_coeffs = (0..self.len)
            .map(|i| contributions[2 * i] + contributions[2 * i + 1])
            .collect();
        Ok(())
    }
}

impl<T: IntRing2k, F: JoltField> PolynomialBinding<F, Rep3Value<F>> for CompactPolynomial<T, F> {
    fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    #[tracing::instrument(skip_all, name = "CompactPoly::bind", level = "trace")]
    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        self.ensure_no_pending_casts();

        let n = self.len() / 2;

        if self.is_bound() {
            let r_f: F = r.into();
            match order {
                BindingOrder::LowToHigh => {
                    for i in 0..n {
                        let a = self.bound_coeffs[2 * i];
                        let b = self.bound_coeffs[2 * i + 1];
                        self.bound_coeffs[i] = a + arithmetic::mul_public(b - a, r_f);
                    }
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.bound_coeffs.split_at_mut(n);
                    left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
                        *a += arithmetic::mul_public(*b - *a, r_f);
                    });
                }
            }
            self.bound_coeffs.truncate(n);
        } else {
            // `a.cmp(&b)` is not possible on shared values. Use:
            //   a * (1 - r) + b * r
            // by casting ring shares into field shares first (batched, deferred).
            let r_f: F = r.into();
            let one_minus_r = F::one() - r;
            self.bound_coeffs_fut = Vec::with_capacity(2 * n);

            match order {
                BindingOrder::LowToHigh => {
                    for i in 0..n {
                        let a = self.coeffs[2 * i];
                        let b = self.coeffs[2 * i + 1];
                        self.bound_coeffs_fut.push(FutureRep3Ring::Pending(
                            FutureOp::CastToField(a),
                            one_minus_r,
                        ));
                        self.bound_coeffs_fut
                            .push(FutureRep3Ring::Pending(FutureOp::CastToField(b), r_f));
                    }
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.coeffs.split_at(n);
                    for (&a, &b) in left.iter().zip(right.iter()) {
                        self.bound_coeffs_fut.push(FutureRep3Ring::Pending(
                            FutureOp::CastToField(a),
                            one_minus_r,
                        ));
                        self.bound_coeffs_fut
                            .push(FutureRep3Ring::Pending(FutureOp::CastToField(b), r_f));
                    }
                }
            }
        }

        self.num_vars -= 1;
        self.len = n;
    }

    #[tracing::instrument(skip_all, name = "CompactPoly::bind_parallel", level = "trace")]
    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        self.ensure_no_pending_casts();

        let n = self.len() / 2;

        if self.is_bound() {
            let r_f: F = r.into();
            match order {
                BindingOrder::LowToHigh => {
                    let needs_alloc = self
                        .binding_scratch_space
                        .as_ref()
                        .map(|v| v.len() < n)
                        .unwrap_or(true);
                    if needs_alloc {
                        self.binding_scratch_space =
                            Some(vec![Rep3PrimeFieldShare::zero_share(); n]);
                    }
                    let scratch = self.binding_scratch_space.as_mut().unwrap();

                    scratch
                        .par_iter_mut()
                        .take(n)
                        .enumerate()
                        .for_each(|(i, new_coeff)| {
                            let a = self.bound_coeffs[2 * i];
                            let b = self.bound_coeffs[2 * i + 1];
                            *new_coeff = a + arithmetic::mul_public(b - a, r_f);
                        });
                    self.bound_coeffs[..n].copy_from_slice(&scratch[..n]);
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.bound_coeffs.split_at_mut(n);
                    left.par_iter_mut()
                        .zip(right.par_iter())
                        .for_each(|(a, b)| {
                            *a += arithmetic::mul_public(*b - *a, r_f);
                        });
                }
            }
            self.bound_coeffs.truncate(n);
        } else {
            let r_f: F = r.into();
            let one_minus_r = F::one() - r;
            // Keep bind_parallel non-interactive: just enqueue futures.
            self.bound_coeffs_fut = Vec::with_capacity(2 * n);

            match order {
                BindingOrder::LowToHigh => {
                    self.bound_coeffs_fut = (0..n)
                        .into_par_iter()
                        .flat_map_iter(|i| {
                            let a = self.coeffs[2 * i];
                            let b = self.coeffs[2 * i + 1];
                            [
                                FutureRep3Ring::Pending(FutureOp::CastToField(a), one_minus_r),
                                FutureRep3Ring::Pending(FutureOp::CastToField(b), r_f),
                            ]
                        })
                        .collect();
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.coeffs.split_at(n);
                    self.bound_coeffs_fut = left
                        .par_iter()
                        .zip(right.par_iter())
                        .flat_map_iter(|(&a, &b)| {
                            [
                                FutureRep3Ring::Pending(FutureOp::CastToField(a), one_minus_r),
                                FutureRep3Ring::Pending(FutureOp::CastToField(b), r_f),
                            ]
                        })
                        .collect();
                }
            }
        }

        self.num_vars -= 1;
        self.len = n;
    }

    fn final_sumcheck_claim(&self) -> Rep3Value<F> {
        self.ensure_bound_ready();
        assert_eq!(self.len, 1);
        Rep3Value::Shared(self.bound_coeffs[0])
    }
}

impl<T: IntRing2k, F: JoltField> PolynomialEvaluation<F, Rep3Value<F>> for CompactPolynomial<T, F> {
    fn evaluate<C>(&self, r: &[C]) -> Rep3Value<F>
    where
        C: Copy + Send + Sync + Into<F> + jolt_core::field::ChallengeFieldOps<F>,
        F: jolt_core::field::FieldChallengeOps<C>,
    {
        self.ensure_bound_ready();
        let chis = EqPolynomial::evals(r);
        Rep3Value::Shared(self.dot_product_with_public(&chis))
    }

    #[tracing::instrument(skip_all, name = "CompactPoly::batch_evaluate", level = "trace")]
    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        Self: Sized,
        C: Copy + Send + Sync + Into<F> + jolt_core::field::ChallengeFieldOps<F>,
        F: jolt_core::field::FieldChallengeOps<C>,
    {
        unimplemented!("Currently unused for shared compact polynomials")
    }

    #[inline]
    fn sumcheck_evals(
        &self,
        index: usize,
        degree: usize,
        order: BindingOrder,
    ) -> Vec<Rep3Value<F>> {
        self.ensure_bound_ready();
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![Rep3PrimeFieldShare::zero_share(); degree];
        match order {
            BindingOrder::LowToHigh => {
                evals[0] = self.get_bound_coeff(2 * index);
                if degree == 1 {
                    return evals.into_iter().map(Rep3Value::Shared).collect();
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
                    return evals.into_iter().map(Rep3Value::Shared).collect();
                }
                let mut eval = self.get_bound_coeff(index + self.len() / 2);
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
        }
        evals.into_iter().map(Rep3Value::Shared).collect()
    }
}

impl<T: IntRing2k, F: JoltField> Polynomial<F> for CompactPolynomial<T, F> {
    fn len(&self) -> usize {
        self.len()
    }

    fn get_num_vars(&self) -> usize {
        self.get_num_vars()
    }

    fn get_bound_coeffs(&self) -> Vec<Rep3Value<F>> {
        self.ensure_bound_ready();
        self.bound_coeffs
            .iter()
            .copied()
            .map(Rep3Value::Shared)
            .collect()
    }
}

impl<T: IntRing2k, F: JoltField> Clone for CompactPolynomial<T, F> {
    fn clone(&self) -> Self {
        Self::from_coeffs(self.coeffs.to_vec())
    }
}

impl<T: IntRing2k, F: JoltField> Index<usize> for CompactPolynomial<T, F> {
    type Output = Rep3RingShare<T>;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.coeffs[index]
    }
}
