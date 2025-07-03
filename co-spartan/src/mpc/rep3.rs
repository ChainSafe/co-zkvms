use std::ops::{Add, AddAssign, Index, Mul, Sub};

use ark_ec::pairing::Pairing;
use ark_ff::{Field, UniformRand, Zero};
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{Rng, RngCore};

use crate::mpc::{additive::AdditiveShare, SSOpen, SSRandom};

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone, Copy)]
pub struct Rep3Share<E: Pairing> {
    pub party: usize,
    pub share_0: E::ScalarField,
    pub share_1: E::ScalarField,
}

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone, Copy)]
pub struct Rep3GroupShare<E: Pairing> {
    pub share_0: E::G1,
    pub share_1: E::G1,
}

#[derive(Debug, CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct RssPoly<E: Pairing> {
    pub party: usize,
    pub share_0: DenseMultilinearExtension<E::ScalarField>,
    pub share_1: DenseMultilinearExtension<E::ScalarField>,
}

impl<E: Pairing> Add<Self> for Rep3Share<E> {
    type Output = Self;
    fn add(self, rhs: Rep3Share<E>) -> <Self as Add<Rep3Share<E>>>::Output {
        assert_eq!(self.party, rhs.party);
        Rep3Share {
            party: self.party,
            share_0: self.share_0 + rhs.share_0,
            share_1: self.share_1 + rhs.share_1,
        }
    }
}

impl<E: Pairing> Add<Rep3Share<E>> for AdditiveShare<E> {
    type Output = Self;
    fn add(self, rhs: Rep3Share<E>) -> Self::Output {
        assert_eq!(self.party, rhs.party);
        AdditiveShare {
            party: self.party,
            share_0: self.share_0 + rhs.to_additive().share_0,
        }
    }
}

impl<'a, E: Pairing> Add<&'a Self> for Rep3Share<E> {
    type Output = Self;
    fn add(self, rhs: &Rep3Share<E>) -> <Self as Add<&Rep3Share<E>>>::Output {
        assert_eq!(self.party, rhs.party);
        Rep3Share {
            party: self.party,
            share_0: self.share_0 + rhs.share_0,
            share_1: self.share_1 + rhs.share_1,
        }
    }
}

impl<'a, E: Pairing> Add<&'a Rep3Share<E>> for AdditiveShare<E> {
    type Output = Self;
    fn add(self, rhs: &Rep3Share<E>) -> Self::Output {
        assert_eq!(self.party, rhs.party);
        AdditiveShare {
            party: self.party,
            share_0: self.share_0 + rhs.to_additive().share_0,
        }
    }
}

impl<E: Pairing> Sub<Self> for Rep3Share<E> {
    type Output = Self;
    fn sub(self, rhs: Rep3Share<E>) -> Self::Output {
        assert_eq!(self.party, rhs.party);
        Rep3Share {
            party: self.party,
            share_0: self.share_0 - rhs.share_0,
            share_1: self.share_1 - rhs.share_1,
        }
    }
}

impl<E: Pairing> Mul<E::ScalarField> for Rep3Share<E> {
    type Output = Self;
    fn mul(self, rhs: E::ScalarField) -> Self::Output {
        Rep3Share {
            party: self.party,
            share_0: self.share_0 * rhs,
            share_1: self.share_1 * rhs,
        }
    }
}

impl<E: Pairing> AddAssign for Rep3Share<E> {
    fn add_assign(&mut self, rhs: Rep3Share<E>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 += rhs.share_0;
        self.share_1 += rhs.share_1;
    }
}

impl<'a, E: Pairing> AddAssign<&'a Rep3Share<E>> for Rep3Share<E> {
    fn add_assign(&mut self, rhs: &Rep3Share<E>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 += rhs.share_0;
        self.share_1 += rhs.share_1;
    }
}

impl<E: Pairing> Zero for Rep3Share<E> {
    fn zero() -> Self {
        Rep3Share {
            party: 0,
            share_0: E::ScalarField::zero(),
            share_1: E::ScalarField::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.share_0.is_zero() && self.share_1.is_zero()
    }
}

impl<E: Pairing> SSOpen<E::ScalarField> for Rep3Share<E> {
    fn open(shares: &[Rep3Share<E>]) -> <E as Pairing>::ScalarField {
        assert!(shares.len() == 3);
        let mut sum = E::ScalarField::zero();
        for rss in shares.iter() {
            sum += rss.share_0;
        }
        sum
    }
}

impl<E: Pairing> Rep3Share<E> {
    pub fn with_party(mut self, party: usize) -> Self {
        self.party = party;
        self
    }

    pub fn inner_prod<R: RngCore + FeedableRNG>(
        lhs: &[Rep3Share<E>],
        rhs: &[Rep3Share<E>],
        rng: &mut SSRandom<R>,
    ) -> E::ScalarField {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(l, r)| Self::mul_wo_zero(l, r))
            .sum::<E::ScalarField>()
            + AdditiveShare::<E>::get_mask_scalar(rng)
    }

    pub fn mul_wo_zero(lhs: &Rep3Share<E>, rhs: &Rep3Share<E>) -> E::ScalarField {
        lhs.share_0 * (rhs.share_0 + rhs.share_1) + lhs.share_1 * rhs.share_0
    }

    pub fn get_mask_scalar<R: RngCore + FeedableRNG>(
        rng: &mut SSRandom<R>,
    ) -> (E::ScalarField, E::ScalarField) {
        let mask_share = (
            E::ScalarField::rand(&mut rng.seed_1) - E::ScalarField::rand(&mut rng.seed_0),
            E::ScalarField::rand(&mut rng.seed_1) - E::ScalarField::rand(&mut rng.seed_0),
        );
        rng.update();
        mask_share
    }

    /// Only for addition of Additive and Rep3
    pub(crate) fn to_additive(&self) -> AdditiveShare<E> {
        AdditiveShare {
            party: self.party,
            share_0: (self.share_0 + self.share_1) / E::ScalarField::from(2u8),
        }
    }
}

impl<E: Pairing> RssPoly<E> {
    pub fn new(
        party: usize,
        share_0: DenseMultilinearExtension<E::ScalarField>,
        share_1: DenseMultilinearExtension<E::ScalarField>,
    ) -> Self {
        RssPoly {
            party,
            share_0,
            share_1,
        }
    }
    pub fn get_share_by_idx(&self, i: usize) -> Rep3Share<E> {
        Rep3Share {
            party: self.party,
            share_0: self.share_0.index(i).clone(),
            share_1: self.share_1.index(i).clone(),
        }
    }
    pub fn fix_variables(&self, partial_point: &[E::ScalarField]) -> Self {
        RssPoly {
            party: self.party,
            share_0: self.share_0.fix_variables(partial_point),
            share_1: self.share_1.fix_variables(partial_point),
        }
    }
    pub fn from_rep3_evals(RssVec: &Vec<Rep3Share<E>>, num_vars: usize) -> Self {
        let mut share_0 = Vec::with_capacity(1 << num_vars);
        let mut share_1 = Vec::with_capacity(1 << num_vars);
        let party = RssVec[0].party;
        for share in RssVec {
            share_0.push(share.share_0);
            share_1.push(share.share_1);
        }
        RssPoly {
            party,
            share_0: DenseMultilinearExtension::<E::ScalarField>::from_evaluations_vec(
                num_vars, share_0,
            ),
            share_1: DenseMultilinearExtension::<E::ScalarField>::from_evaluations_vec(
                num_vars, share_1,
            ),
        }
    }
}

pub fn generate_poly_shares_rss<F: Field, R: Rng>(
    poly: &DenseMultilinearExtension<F>,
    rng: &mut R,
) -> [DenseMultilinearExtension<F>; 3] {
    let num_vars = poly.num_vars;
    let p_share_0 = DenseMultilinearExtension::<F>::rand(num_vars, rng);
    let p_share_1 = DenseMultilinearExtension::<F>::rand(num_vars, rng);
    let p_share_2 = (poly - &p_share_0) - p_share_1.clone();

    [p_share_0, p_share_1, p_share_2]
}

pub fn generate_rss_share_randomness<R: RngCore + FeedableRNG>() -> Vec<SSRandom<R>> {
    let mut seed_0 = R::setup();
    seed_0.feed(&"seed 0".as_bytes());
    let mut seed_1 = R::setup();
    seed_1.feed(&"seed 1".as_bytes());
    let mut random_0 = SSRandom::<R>::new(seed_0, seed_1);

    let mut seed_1 = R::setup();
    seed_1.feed(&"seed 1".as_bytes());
    let mut seed_2 = R::setup();
    seed_2.feed(&"seed 2".as_bytes());
    let mut random_1 = SSRandom::<R>::new(seed_1, seed_2);

    let mut seed_2 = R::setup();
    seed_2.feed(&"seed 2".as_bytes());
    let mut seed_0 = R::setup();
    seed_0.feed(&"seed 0".as_bytes());
    let mut random_2 = SSRandom::<R>::new(seed_2, seed_0);

    let mut vec_random = vec![random_0, random_1, random_2];
    vec_random
}
