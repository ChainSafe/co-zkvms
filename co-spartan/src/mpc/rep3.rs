use std::ops::{Add, AddAssign, Index, Mul, Sub};

use ark_ff::{Field, PrimeField, Zero};
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{Rng, RngCore};

use crate::mpc::{additive::AdditiveShare, SSOpen, SSRandom};

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone, Copy)]
pub struct Rep3Share<F: PrimeField> {
    pub party: usize,
    pub share_0: F,
    pub share_1: F,
}


impl<F: PrimeField> Add<Self> for Rep3Share<F> {
    type Output = Self;
    fn add(self, rhs: Rep3Share<F>) -> <Self as Add<Rep3Share<F>>>::Output {
        assert_eq!(self.party, rhs.party);
        Rep3Share {
            party: self.party,
            share_0: self.share_0 + rhs.share_0,
            share_1: self.share_1 + rhs.share_1,
        }
    }
}

impl<F: PrimeField> Add<Rep3Share<F>> for AdditiveShare<F> {
    type Output = Self;
    fn add(self, rhs: Rep3Share<F>) -> Self::Output {
        assert_eq!(self.party, rhs.party);
        AdditiveShare {
            party: self.party,
            share_0: self.share_0 + rhs.into_additive().share_0,
        }
    }
}

impl<'a, F: PrimeField> Add<&'a Self> for Rep3Share<F> {
    type Output = Self;
    fn add(self, rhs: &Rep3Share<F>) -> <Self as Add<&Rep3Share<F>>>::Output {
        assert_eq!(self.party, rhs.party);
        Rep3Share {
            party: self.party,
            share_0: self.share_0 + rhs.share_0,
            share_1: self.share_1 + rhs.share_1,
        }
    }
}

impl<'a, F: PrimeField> Add<&'a Rep3Share<F>> for AdditiveShare<F> {
    type Output = Self;
    fn add(self, rhs: &Rep3Share<F>) -> Self::Output {
        assert_eq!(self.party, rhs.party);
        AdditiveShare {
            party: self.party,
            share_0: self.share_0 + rhs.into_additive().share_0,
        }
    }
}

impl<F: PrimeField> Sub<Self> for Rep3Share<F> {
    type Output = Self;
    fn sub(self, rhs: Rep3Share<F>) -> Self::Output {
        assert_eq!(self.party, rhs.party);
        Rep3Share {
            party: self.party,
            share_0: self.share_0 - rhs.share_0,
            share_1: self.share_1 - rhs.share_1,
        }
    }
}

impl<F: PrimeField> Mul<F> for Rep3Share<F> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self::Output {
        Rep3Share {
            party: self.party,
            share_0: self.share_0 * rhs,
            share_1: self.share_1 * rhs,
        }
    }
}

impl<F: PrimeField> AddAssign for Rep3Share<F> {
    fn add_assign(&mut self, rhs: Rep3Share<F>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 += rhs.share_0;
        self.share_1 += rhs.share_1;
    }
}

impl<'a, F: PrimeField> AddAssign<&'a Rep3Share<F>> for Rep3Share<F> {
    fn add_assign(&mut self, rhs: &Rep3Share<F>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 += rhs.share_0;
        self.share_1 += rhs.share_1;
    }
}

impl<F: PrimeField> Zero for Rep3Share<F> {
    fn zero() -> Self {
        Rep3Share {
            party: 0,
            share_0: F::zero(),
            share_1: F::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.share_0.is_zero() && self.share_1.is_zero()
    }
}

impl<F: PrimeField> SSOpen<F> for Rep3Share<F> {
    fn open(shares: &[Rep3Share<F>]) -> F {
        assert!(shares.len() == 3);
        let mut sum = F::zero();
        for rss in shares.iter() {
            sum += rss.share_0;
        }
        sum
    }
}

impl<F: PrimeField> Rep3Share<F> {
    pub fn with_party(mut self, party: usize) -> Self {
        self.party = party;
        self
    }

    pub fn inner_prod<R: RngCore + FeedableRNG>(
        lhs: &[Rep3Share<F>],
        rhs: &[Rep3Share<F>],
        rng: &mut SSRandom<R>,
    ) -> F {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(l, r)| Self::mul_wo_zero(l, r))
            .sum::<F>()
            + AdditiveShare::<F>::get_mask_scalar(rng)
    }

    pub fn mul_wo_zero(lhs: &Rep3Share<F>, rhs: &Rep3Share<F>) -> F {
        lhs.share_0 * (rhs.share_0 + rhs.share_1) + lhs.share_1 * rhs.share_0
    }

    pub fn get_mask_scalar<R: RngCore + FeedableRNG>(
        rng: &mut SSRandom<R>,
    ) -> (F, F) {
        let mask_share = (
            F::rand(&mut rng.seed_1) - F::rand(&mut rng.seed_0),
            F::rand(&mut rng.seed_1) - F::rand(&mut rng.seed_0),
        );
        rng.update();
        mask_share
    }

    /// Only for addition of Additive and Rep3
    pub(crate) fn into_additive(&self) -> AdditiveShare<F> {
        AdditiveShare {
            party: self.party,
            share_0: (self.share_0 + self.share_1) / F::from(2u8),
        }
    }
}

#[derive(Debug, CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct Rep3Poly<F: PrimeField> {
    pub party_id: usize,
    pub share_0: DenseMultilinearExtension<F>,
    pub share_1: DenseMultilinearExtension<F>,
}

impl<F: PrimeField> Rep3Poly<F> {
    pub fn new(
        party: usize,
        share_0: DenseMultilinearExtension<F>,
        share_1: DenseMultilinearExtension<F>,
    ) -> Self {
        Rep3Poly {
            party_id: party,
            share_0,
            share_1,
        }
    }
    pub fn get_share_by_idx(&self, i: usize) -> Rep3Share<F> {
        Rep3Share {
            party: self.party_id,
            share_0: self.share_0.index(i).clone(),
            share_1: self.share_1.index(i).clone(),
        }
    }
    pub fn fix_variables(&self, partial_point: &[F]) -> Self {
        Rep3Poly {
            party_id: self.party_id,
            share_0: self.share_0.fix_variables(partial_point),
            share_1: self.share_1.fix_variables(partial_point),
        }
    }
    pub fn from_rep3_evals(evals_rep3: &Vec<Rep3Share<F>>, num_vars: usize) -> Self {
        let mut share_0 = Vec::with_capacity(1 << num_vars);
        let mut share_1 = Vec::with_capacity(1 << num_vars);
        let party = evals_rep3[0].party;
        for share in evals_rep3 {
            share_0.push(share.share_0);
            share_1.push(share.share_1);
        }
        Rep3Poly {
            party_id: party,
            share_0: DenseMultilinearExtension::<F>::from_evaluations_vec(
                num_vars, share_0,
            ),
            share_1: DenseMultilinearExtension::<F>::from_evaluations_vec(
                num_vars, share_1,
            ),
        }
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
