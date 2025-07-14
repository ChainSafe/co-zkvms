use ark_ff::{Field, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::cfg_iter;
use itertools::Itertools;
use jolt_core::{
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        field::JoltField,
        structured_poly::{StructuredCommitment, StructuredOpeningProof},
    },
    utils::transcript::ProofTranscript,
};
use mpc_core::protocols::rep3;
use mpc_net::mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker};
use rand::Rng;
use spartan::math::Math;
use std::{ops::Index, slice::SliceIndex};

use crate::subprotocols::commitment::DistributedCommitmentScheme;

use super::Rep3PrimeFieldShare;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug, Clone, Default, CanonicalDeserialize, CanonicalSerialize)]
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
        // let mut share_0 = Vec::with_capacity(1 << num_vars);
        // let mut share_1 = Vec::with_capacity(1 << num_vars);
        // // let party = evals_rep3[0].party;
        // for share in evals {
        //     share_0.push(share.a);
        //     share_1.push(share.b);
        // }
        Rep3DensePolynomial { num_vars, evals }
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
        assert_eq!(a.evals_ref().len(), 1 << a.get_num_vars());
        assert_eq!(b.evals_ref().len(), 1 << b.get_num_vars());
        assert_eq!(a.evals_ref().len(), b.evals_ref().len());
        Rep3DensePolynomial::from_vec_shares(a.into_evals(), b.into_evals())
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

    // pub fn split_evals(&self, idx: usize) -> (Self, Self) {
    //     let (a, b) = self.copy_poly_shares();
    //     let (left_a, right_a) = a.split_evals(idx);
    //     let (left_b, right_b) = b.split_evals(idx);
    //     assert!(idx < self.len());
    //     (
    //         Self::from_poly_shares(left_a, left_b),
    //         Self::from_poly_shares(right_a, right_b),
    //     )
    // }

    pub fn split_evals(
        &self,
        idx: usize,
    ) -> (&[Rep3PrimeFieldShare<F>], &[Rep3PrimeFieldShare<F>]) {
        (&self.evals[..idx], &self.evals[idx..])
    }

    pub fn fix_var_top(&mut self, r: &F) {
        let (mut a, mut b) = self.copy_poly_shares();
        a.bound_poly_var_top(r);
        b.bound_poly_var_top(r);

        *self = Self::from_poly_shares(a, b);
    }

    pub fn fix_var_top_many_ones(&mut self, r: &F) {
        let (mut a, mut b) = self.copy_poly_shares();
        a.bound_poly_var_top_many_ones(r);
        b.bound_poly_var_top_many_ones(r);

        *self = Self::from_poly_shares(a, b);
    }

    pub fn new_poly_from_fix_var_top(&self, r: &F) -> Self {
        let (mut a, mut b) = self.copy_poly_shares();
        let a = a.new_poly_from_bound_poly_var_top(r);
        let b = b.new_poly_from_bound_poly_var_top(r); // TODO: check if this is correct with rep3 shares

        Self::from_poly_shares(a, b)
    }

    pub fn evaluate(&self, r: &[F]) -> F {
        let chis = EqPolynomial::new(r.to_vec()).evals();
        assert_eq!(chis.len(), self.evals_ref().len());
        self.evaluate_at_chi(&chis)
    }

    pub fn evaluate_at_chi(&self, chis: &[F]) -> F {
        cfg_iter!(self.evals)
            .zip_eq(cfg_iter!(chis))
            .map(|(eval, chi)| rep3::arithmetic::mul_public(*eval, *chi).into_additive())
            .sum()
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        1 << self.num_vars()
    }

    pub fn evals_ref(&self) -> &[Rep3PrimeFieldShare<F>] {
        &self.evals
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

pub fn combine_poly_shares<F: JoltField>(
    poly_shares: Vec<Rep3DensePolynomial<F>>,
) -> DensePolynomial<F> {
    assert_eq!(poly_shares.len(), 3);
    let [s0, s1, s2] = poly_shares.try_into().unwrap();
    let a = rep3::combine_field_elements(&s0.evals, &s1.evals, &s2.evals);
    DensePolynomial::new(a)
}

/// Encapsulates the pattern of opening a batched polynomial commitment at a single point.
/// Note that there may be a one-to-many mapping from `StructuredCommitment` to `StructuredOpeningProof`:
/// different subset of the same polynomials may be opened at different points, resulting in
/// different opening proofs.
pub trait Rep3StructuredOpeningProof<F, C, Polynomials>:
    StructuredOpeningProof<F, C, Polynomials>
where
    F: JoltField,
    C: DistributedCommitmentScheme<F>,
    Polynomials: StructuredCommitment<C> + ?Sized,
{
    type Rep3Polynomials: ?Sized;

    /// Evaluates each of the given `polynomials` at the given `opening_point`.
    fn open_rep3<Network: MpcStarNetCoordinator>(
        opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<Self>;

    /// Evaluates each of the given `polynomials` at the given `opening_point`.
    fn open_rep3_worker<Network: MpcStarNetWorker>(
        polynomials: &Self::Rep3Polynomials,
        opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<()>;

    /// Proves that the `polynomials`, evaluated at `opening_point`, output the values given
    /// by `openings`. The polynomials should already be committed by the prover.
    fn prove_openings_rep3_worker<Network: MpcStarNetWorker>(
        polynomials: &Self::Rep3Polynomials,
        opening_point: &[F],
        setup: &C::Setup,
        network: &mut Network,
    ) -> eyre::Result<()>;

    /// Proves that the `polynomials`, evaluated at `opening_point`, output the values given
    /// by `openings`. The polynomials should already be committed by the prover.
    fn prove_openings_rep3<Network: MpcStarNetCoordinator>(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Self::Proof>;
}
