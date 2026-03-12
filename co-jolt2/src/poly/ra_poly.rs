use core::mem;
use std::sync::Arc;

use jolt_core::poly::multilinear_polynomial::BindingOrder;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3::{self, arithmetic as rep3_arith};
use rayon::prelude::*;

use crate::poly::dense_mlpoly::{unsafe_allocate_zero_share_vec, Rep3DensePolynomial};
use crate::utils::fwht::fwht_in_place;
use jolt_core::field::JoltField;

/// Rep3 version of vanilla Jolt's `RaPolynomial`.
///
/// This represents a multilinear polynomial over the cycle variables where coefficients are
/// computed via lookup into a preweighted EQ table, indexed by a (public) per-cycle lookup index.
#[derive(Clone, Debug, PartialEq)]
pub enum Rep3RaPolynomial<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> {
    None,
    Round1(Rep3RaPolynomialRound1<I, F>),
    Round2(Rep3RaPolynomialRound2<I, F>),
    Round3(Rep3RaPolynomialRound3<I, F>),
    RoundN(Rep3DensePolynomial<F>),
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> Rep3RaPolynomial<I, F> {
    pub fn new(lookup_indices: Arc<Vec<Option<I>>>, table: Vec<Rep3PrimeFieldShare<F>>) -> Self {
        Self::Round1(Rep3RaPolynomialRound1 { shifted_table: table, masked_lookup_indices: lookup_indices })
    }

    #[inline]
    pub fn get_bound_coeff(&self, j: usize) -> Rep3PrimeFieldShare<F> {
        match self {
            Self::None => panic!("Rep3RaPolynomial::get_bound_coeff called on None"),
            Self::Round1(mle) => mle.get_bound_coeff(j),
            Self::Round2(mle) => mle.get_bound_coeff(j),
            Self::Round3(mle) => mle.get_bound_coeff(j),
            Self::RoundN(mle) => mle.get_bound_coeff(j),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::None => panic!("Rep3RaPolynomial::len called on None"),
            Self::Round1(mle) => mle.len(),
            Self::Round2(mle) => mle.len(),
            Self::Round3(mle) => mle.len(),
            Self::RoundN(mle) => mle.len(),
        }
    }

    #[inline]
    pub fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<Rep3PrimeFieldShare<F>> {
        assert!(degree > 0);
        assert!(index < self.len() / 2);

        let mut evals = vec![Rep3PrimeFieldShare::zero_share(); degree];
        match order {
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
        }
        evals
    }

    pub fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        match self {
            Self::None => panic!("Rep3RaPolynomial::bind called on None"),
            Self::Round1(mle) => *self = Self::Round2(mem::take(mle).bind(r, order)),
            Self::Round2(mle) => *self = Self::Round3(mem::take(mle).bind(r, order)),
            Self::Round3(mle) => *self = Self::RoundN(mem::take(mle).bind(r, order)),
            Self::RoundN(mle) => mle.bind(r.into(), order),
        };
    }

    pub fn final_sumcheck_claim(&self) -> Rep3PrimeFieldShare<F> {
        match self {
            Self::RoundN(mle) => mle.final_sumcheck_claim(),
            _ => panic!("Rep3RaPolynomial::final_sumcheck_claim only valid in RoundN"),
        }
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct Rep3RaPolynomialRound1<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> {
    shifted_table: Vec<Rep3PrimeFieldShare<F>>,
    masked_lookup_indices: Arc<Vec<Option<I>>>,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> Rep3RaPolynomialRound1<I, F> {
    fn len(&self) -> usize {
        self.masked_lookup_indices.len()
    }

    fn bind(self, r0: F::Challenge, binding_order: BindingOrder) -> Rep3RaPolynomialRound2<I, F> {
        let r0_f: F = r0.into();
        let eq_0_r0 = F::one() - r0_f;
        let eq_1_r0 = r0_f;

        let shifted = self.shifted_table;
        let k = shifted.len();
        let mut f_0 = unsafe_allocate_zero_share_vec::<F>(k);
        let mut f_1 = unsafe_allocate_zero_share_vec::<F>(k);
        f_0.par_iter_mut().zip_eq(f_1.par_iter_mut()).zip_eq(shifted.par_iter()).for_each(|((o0, o1), &v)| {
            *o0 = v * eq_0_r0;
            *o1 = v * eq_1_r0;
        });

        Rep3RaPolynomialRound2 { f_0, f_1, masked_lookup_indices: self.masked_lookup_indices, binding_order }
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> Rep3PrimeFieldShare<F> {
        self.masked_lookup_indices
            .get(j)
            .expect("j out of bounds")
            .map_or(Rep3PrimeFieldShare::zero_share(), |i| self.shifted_table[i.into()])
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct Rep3RaPolynomialRound2<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> {
    f_0: Vec<Rep3PrimeFieldShare<F>>,
    f_1: Vec<Rep3PrimeFieldShare<F>>,
    masked_lookup_indices: Arc<Vec<Option<I>>>,
    binding_order: BindingOrder,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> Rep3RaPolynomialRound2<I, F> {
    fn len(&self) -> usize {
        self.masked_lookup_indices.len() / 2
    }

    fn bind(self, r1: F::Challenge, binding_order: BindingOrder) -> Rep3RaPolynomialRound3<I, F> {
        assert_eq!(binding_order, self.binding_order);
        let r1_f: F = r1.into();
        let eq_0_r1 = F::one() - r1_f;
        let eq_1_r1 = r1_f;

        let f0 = self.f_0;
        let f1 = self.f_1;
        let k = f0.len();
        debug_assert_eq!(f1.len(), k);
        let mut f_00 = unsafe_allocate_zero_share_vec::<F>(k);
        let mut f_01 = unsafe_allocate_zero_share_vec::<F>(k);
        let mut f_10 = unsafe_allocate_zero_share_vec::<F>(k);
        let mut f_11 = unsafe_allocate_zero_share_vec::<F>(k);

        f_00.par_iter_mut().zip_eq(f0.par_iter()).for_each(|(dst, &src)| *dst = src * eq_0_r1);
        f_01.par_iter_mut().zip_eq(f0.par_iter()).for_each(|(dst, &src)| *dst = src * eq_1_r1);
        f_10.par_iter_mut().zip_eq(f1.par_iter()).for_each(|(dst, &src)| *dst = src * eq_0_r1);
        f_11.par_iter_mut().zip_eq(f1.par_iter()).for_each(|(dst, &src)| *dst = src * eq_1_r1);

        Rep3RaPolynomialRound3 {
            f_00,
            f_01,
            f_10,
            f_11,
            masked_lookup_indices: self.masked_lookup_indices,
            binding_order: self.binding_order,
        }
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> Rep3PrimeFieldShare<F> {
        let mid = self.masked_lookup_indices.len() / 2;
        match self.binding_order {
            BindingOrder::HighToLow => {
                let h_0 =
                    self.masked_lookup_indices[j].map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_0[i.into()]);
                let h_1 = self.masked_lookup_indices[mid + j]
                    .map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_1[i.into()]);
                h_0 + h_1
            }
            BindingOrder::LowToHigh => {
                let h_0 =
                    self.masked_lookup_indices[2 * j].map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_0[i.into()]);
                let h_1 = self.masked_lookup_indices[2 * j + 1]
                    .map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_1[i.into()]);
                h_0 + h_1
            }
        }
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct Rep3RaPolynomialRound3<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> {
    f_00: Vec<Rep3PrimeFieldShare<F>>,
    f_01: Vec<Rep3PrimeFieldShare<F>>,
    f_10: Vec<Rep3PrimeFieldShare<F>>,
    f_11: Vec<Rep3PrimeFieldShare<F>>,
    masked_lookup_indices: Arc<Vec<Option<I>>>,
    binding_order: BindingOrder,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> Rep3RaPolynomialRound3<I, F> {
    fn len(&self) -> usize {
        self.masked_lookup_indices.len() / 4
    }

    fn bind(self, r2: F::Challenge, _binding_order: BindingOrder) -> Rep3DensePolynomial<F> {
        let r2_f: F = r2.into();
        let eq_0_r2 = F::one() - r2_f;
        let eq_1_r2 = r2_f;

        let mut f_000 = self.f_00.clone();
        let mut f_001 = self.f_00;
        let mut f_010 = self.f_01.clone();
        let mut f_011 = self.f_01;
        let mut f_100 = self.f_10.clone();
        let mut f_101 = self.f_10;
        let mut f_110 = self.f_11.clone();
        let mut f_111 = self.f_11;

        f_000.par_iter_mut().for_each(|v| *v *= eq_0_r2);
        f_010.par_iter_mut().for_each(|v| *v *= eq_0_r2);
        f_100.par_iter_mut().for_each(|v| *v *= eq_0_r2);
        f_110.par_iter_mut().for_each(|v| *v *= eq_0_r2);
        f_001.par_iter_mut().for_each(|v| *v *= eq_1_r2);
        f_011.par_iter_mut().for_each(|v| *v *= eq_1_r2);
        f_101.par_iter_mut().for_each(|v| *v *= eq_1_r2);
        f_111.par_iter_mut().for_each(|v| *v *= eq_1_r2);

        let lookup_indices = &self.masked_lookup_indices;
        let n = lookup_indices.len() / 8;
        let mut res = unsafe_allocate_zero_share_vec::<F>(n);
        let chunk_size = 1 << 16;

        match self.binding_order {
            BindingOrder::HighToLow => {
                res.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_index, evals_chunk)| {
                    for (j, eval) in (chunk_index * chunk_size..).zip(evals_chunk.iter_mut()) {
                        let h_000 = lookup_indices[j].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_000[i.into()]);
                        let h_001 =
                            lookup_indices[j + n].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_001[i.into()]);
                        let h_010 =
                            lookup_indices[j + n * 2].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_010[i.into()]);
                        let h_011 =
                            lookup_indices[j + n * 3].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_011[i.into()]);
                        let h_100 =
                            lookup_indices[j + n * 4].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_100[i.into()]);
                        let h_101 =
                            lookup_indices[j + n * 5].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_101[i.into()]);
                        let h_110 =
                            lookup_indices[j + n * 6].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_110[i.into()]);
                        let h_111 =
                            lookup_indices[j + n * 7].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_111[i.into()]);
                        *eval = h_000 + h_010 + h_100 + h_110 + h_001 + h_011 + h_101 + h_111;
                    }
                });
            }
            BindingOrder::LowToHigh => {
                res.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_index, evals_chunk)| {
                    for (j, eval) in (chunk_index * chunk_size..).zip(evals_chunk.iter_mut()) {
                        let h_000 =
                            lookup_indices[8 * j].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_000[i.into()]);
                        let h_100 =
                            lookup_indices[8 * j + 1].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_100[i.into()]);
                        let h_010 =
                            lookup_indices[8 * j + 2].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_010[i.into()]);
                        let h_110 =
                            lookup_indices[8 * j + 3].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_110[i.into()]);
                        let h_001 =
                            lookup_indices[8 * j + 4].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_001[i.into()]);
                        let h_101 =
                            lookup_indices[8 * j + 5].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_101[i.into()]);
                        let h_011 =
                            lookup_indices[8 * j + 6].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_011[i.into()]);
                        let h_111 =
                            lookup_indices[8 * j + 7].map_or(Rep3PrimeFieldShare::zero_share(), |i| f_111[i.into()]);
                        *eval = h_000 + h_010 + h_100 + h_110 + h_001 + h_011 + h_101 + h_111;
                    }
                });
            }
        }

        Rep3DensePolynomial::new(res)
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> Rep3PrimeFieldShare<F> {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let n = self.masked_lookup_indices.len() / 4;
                let h_00 =
                    self.masked_lookup_indices[j].map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_00[i.into()]);
                let h_01 = self.masked_lookup_indices[j + n]
                    .map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_01[i.into()]);
                let h_10 = self.masked_lookup_indices[j + n * 2]
                    .map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_10[i.into()]);
                let h_11 = self.masked_lookup_indices[j + n * 3]
                    .map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_11[i.into()]);
                h_00 + h_10 + h_01 + h_11
            }
            BindingOrder::LowToHigh => {
                let h_00 = self.masked_lookup_indices[4 * j]
                    .map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_00[i.into()]);
                let h_10 = self.masked_lookup_indices[4 * j + 1]
                    .map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_10[i.into()]);
                let h_01 = self.masked_lookup_indices[4 * j + 2]
                    .map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_01[i.into()]);
                let h_11 = self.masked_lookup_indices[4 * j + 3]
                    .map_or(Rep3PrimeFieldShare::zero_share(), |i| self.f_11[i.into()]);
                h_00 + h_10 + h_01 + h_11
            }
        }
    }
}

/// Computes a secret-shared table `F_shifted[c] = eq_u[r XOR c]` from a public `eq_u` table
/// and secret-shared one-hot vector `E_field = e(r)`.
pub fn shifted_table_from_rand_ohv<F: JoltField>(
    eq_u: &[F],
    e_field: &[Rep3PrimeFieldShare<F>],
) -> Vec<Rep3PrimeFieldShare<F>> {
    assert_eq!(eq_u.len(), e_field.len());
    let k = eq_u.len();
    debug_assert!(k.is_power_of_two(), "K must be power-of-two for FWHT");
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![rep3_arith::mul_public(e_field[0], eq_u[0])];
    }

    // We want: shifted[c] = Σ_i e_field[i] * eq_u[i XOR c].
    //
    // This is an XOR-correlation and can be computed via FWHT:
    //   FWHT(shifted) = FWHT(e_field) ⊙ FWHT(eq_u)
    // followed by inverse FWHT and scaling by 1/k.
    let mut e_hat: Vec<Rep3PrimeFieldShare<F>> = e_field.to_vec();
    fwht_in_place(&mut e_hat);

    let mut u_hat: Vec<F> = eq_u.to_vec();
    fwht_in_place(&mut u_hat);

    e_hat.par_iter_mut().zip_eq(u_hat.par_iter()).for_each(|(e, &u)| {
        *e = rep3::arithmetic::mul_public(*e, u);
    });

    fwht_in_place(&mut e_hat);

    let inv_k = F::from(k as u64).inverse().expect("K invertible in field");
    e_hat.par_iter_mut().for_each(|e| {
        *e = rep3::arithmetic::mul_public(*e, inv_k);
    });

    e_hat
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;
    use jolt_core::ark_bn254::Fr;
    use mpc_core::protocols::rep3::combine_field_element;
    use num_traits::{One, Zero};
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    fn share_field_element_rep3<F: JoltField, R: rand::Rng + rand::CryptoRng>(val: F, rng: &mut R) -> [Rep3PrimeFieldShare<F>; 3] {
        mpc_core::protocols::rep3::share_field_element(val, rng)
    }

    fn bind_high_to_low_plain<F: JoltField>(vals: &mut Vec<F>, r: F) {
        assert!(vals.len().is_power_of_two());
        let n = vals.len() / 2;
        let (left, right) = vals.split_at_mut(n);
        for i in 0..n {
            let m = right[i] - left[i];
            left[i] += m * r;
        }
        vals.truncate(n);
    }

    #[test]
    fn rep3_ra_shift_correct() {
        type F = Fr;
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        let k = 256usize;
        let t = 1usize << 5;

        let eq_u: Vec<F> = (0..k).map(|_| F::rand(&mut rng)).collect();

        let r_mask: u8 = (rng.next_u32() as u8) & 0xff;
        let mut k_plain: Vec<Option<u8>> = (0..t)
            .map(|_| if (rng.next_u32() & 3) == 0 { None } else { Some((rng.next_u32() as u8) & 0xff) })
            .collect();
        if k_plain.iter().all(|x| x.is_none()) {
            k_plain[0] = Some(7);
        }

        let masked_indices_c: Arc<Vec<Option<u8>>> =
            Arc::new(k_plain.iter().map(|opt| opt.map(|kj| kj ^ r_mask)).collect());

        let mut e_field_party: [Vec<Rep3PrimeFieldShare<F>>; 3] = std::array::from_fn(|_| Vec::with_capacity(k));
        for i in 0..k {
            let bit = if i as u8 == r_mask { F::one() } else { F::zero() };
            let shares = share_field_element_rep3(bit, &mut rng);
            for pid in 0..3 {
                e_field_party[pid].push(shares[pid]);
            }
        }

        let ra_party: [Rep3RaPolynomial<u8, F>; 3] = std::array::from_fn(|pid| {
            let table_shifted = shifted_table_from_rand_ohv(&eq_u, &e_field_party[pid]);
            Rep3RaPolynomial::new(masked_indices_c.clone(), table_shifted)
        });

        for (j, opt_k) in k_plain.iter().enumerate() {
            let shares: [Rep3PrimeFieldShare<F>; 3] = std::array::from_fn(|pid| ra_party[pid].get_bound_coeff(j));
            let got = combine_field_element(shares[0], shares[1], shares[2]);
            let want = opt_k.map(|kk| eq_u[kk as usize]).unwrap_or(F::zero());
            assert_eq!(got, want, "j {j}");
        }
    }

    #[test]
    fn rep3_ra_bind_correct() {
        type F = Fr;
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        let k = 256usize;
        let t = 1usize << 5;

        let eq_u: Vec<F> = (0..k).map(|_| F::rand(&mut rng)).collect();
        let r_mask: u8 = (rng.next_u32() as u8) & 0xff;
        let mut k_plain: Vec<Option<u8>> = (0..t)
            .map(|_| if (rng.next_u32() & 3) == 0 { None } else { Some((rng.next_u32() as u8) & 0xff) })
            .collect();
        if k_plain.iter().all(|x| x.is_none()) {
            k_plain[0] = Some(7);
        }

        let masked_indices_c: Arc<Vec<Option<u8>>> =
            Arc::new(k_plain.iter().map(|opt| opt.map(|kj| kj ^ r_mask)).collect());

        let mut e_field_party: [Vec<Rep3PrimeFieldShare<F>>; 3] = std::array::from_fn(|_| Vec::with_capacity(k));
        for i in 0..k {
            let bit = if i as u8 == r_mask { F::one() } else { F::zero() };
            let shares = share_field_element_rep3(bit, &mut rng);
            for pid in 0..3 {
                e_field_party[pid].push(shares[pid]);
            }
        }

        let mut ra_party: [Rep3RaPolynomial<u8, F>; 3] = std::array::from_fn(|pid| {
            let table_shifted = shifted_table_from_rand_ohv(&eq_u, &e_field_party[pid]);
            Rep3RaPolynomial::new(masked_indices_c.clone(), table_shifted)
        });

        let mut plain: Vec<F> =
            k_plain.iter().map(|opt| opt.map(|kk| eq_u[kk as usize]).unwrap_or(F::zero())).collect();

        for round in 0..3 {
            let r = <F as jolt_core::field::JoltField>::Challenge::random(&mut rng);
            let r_f: F = r.into();
            bind_high_to_low_plain(&mut plain, r_f);
            for pid in 0..3 {
                ra_party[pid].bind_parallel(r, BindingOrder::HighToLow);
            }

            for j in 0..plain.len() {
                let shares: [Rep3PrimeFieldShare<F>; 3] = std::array::from_fn(|pid| ra_party[pid].get_bound_coeff(j));
                let got = combine_field_element(shares[0], shares[1], shares[2]);
                assert_eq!(got, plain[j], "round {round} j {j}");
            }
        }
    }

    #[test]
    fn rep3_ra_evals_correct() {
        type F = Fr;
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        let k = 256usize;
        let t = 1usize << 5;

        let eq_u: Vec<F> = (0..k).map(|_| F::rand(&mut rng)).collect();
        let r_mask: u8 = (rng.next_u32() as u8) & 0xff;
        let mut k_plain: Vec<Option<u8>> = (0..t)
            .map(|_| if (rng.next_u32() & 3) == 0 { None } else { Some((rng.next_u32() as u8) & 0xff) })
            .collect();
        if k_plain.iter().all(|x| x.is_none()) {
            k_plain[0] = Some(7);
        }

        let masked_indices_c: Arc<Vec<Option<u8>>> =
            Arc::new(k_plain.iter().map(|opt| opt.map(|kj| kj ^ r_mask)).collect());

        let mut e_field_party: [Vec<Rep3PrimeFieldShare<F>>; 3] = std::array::from_fn(|_| Vec::with_capacity(k));
        for i in 0..k {
            let bit = if i as u8 == r_mask { F::one() } else { F::zero() };
            let shares = share_field_element_rep3(bit, &mut rng);
            for pid in 0..3 {
                e_field_party[pid].push(shares[pid]);
            }
        }

        let ra_party: [Rep3RaPolynomial<u8, F>; 3] = std::array::from_fn(|pid| {
            let table_shifted = shifted_table_from_rand_ohv(&eq_u, &e_field_party[pid]);
            Rep3RaPolynomial::new(masked_indices_c.clone(), table_shifted)
        });

        let plain: Vec<F> = k_plain.iter().map(|opt| opt.map(|kk| eq_u[kk as usize]).unwrap_or(F::zero())).collect();

        let degree = 3usize;
        let order = BindingOrder::HighToLow;
        let index = 0usize;

        // expected evals computed like vanilla sumcheck_evals: [f(0), f(2), f(3)].
        let f0 = plain[index];
        let f1 = plain[index + plain.len() / 2];
        let m = f1 - f0;
        let mut eval = f1;
        let mut expected = vec![F::zero(); degree];
        expected[0] = f0;
        for i in 1..degree {
            eval += m;
            expected[i] = eval;
        }

        let evals_party: [Vec<Rep3PrimeFieldShare<F>>; 3] =
            std::array::from_fn(|pid| ra_party[pid].sumcheck_evals(index, degree, order));
        for i in 0..degree {
            let got = combine_field_element(evals_party[0][i], evals_party[1][i], evals_party[2][i]);
            assert_eq!(got, expected[i], "eval {i}");
        }
    }
}
