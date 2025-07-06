use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use ark_ff::{PrimeField, Zero};
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{Rng, RngCore};

use crate::mpc::{SSOpen, SSRandom};

pub type ArithmeticShare<F> = AdditiveShare<F>;

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone, Copy)]
pub struct AdditiveShare<F: PrimeField> {
    pub party: usize,
    pub share_0: F,
}

impl<F: PrimeField> Add<Self> for AdditiveShare<F> {
    type Output = Self;
    fn add(self, rhs: AdditiveShare<F>) -> <Self as Add<AdditiveShare<F>>>::Output {
        assert_eq!(self.party, rhs.party);
        AdditiveShare {
            party: self.party,
            share_0: self.share_0 + rhs.share_0,
        }
    }
}

impl<F: PrimeField> Sub<Self> for AdditiveShare<F> {
    type Output = Self;
    fn sub(self, rhs: AdditiveShare<F>) -> Self::Output {
        assert_eq!(self.party, rhs.party);
        AdditiveShare {
            party: self.party,
            share_0: self.share_0 - rhs.share_0,
        }
    }
}

impl<F: PrimeField> Mul<F> for AdditiveShare<F> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self::Output {
        AdditiveShare {
            party: self.party,
            share_0: self.share_0 * rhs,
        }
    }
}

impl<F: PrimeField> AddAssign for AdditiveShare<F> {
    fn add_assign(&mut self, rhs: AdditiveShare<F>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 += rhs.share_0;
    }
}

impl<'a, F: PrimeField> AddAssign<&'a AdditiveShare<F>> for AdditiveShare<F> {
    fn add_assign(&mut self, rhs: &AdditiveShare<F>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 += rhs.share_0;
    }
}

impl<F: PrimeField> SubAssign for AdditiveShare<F> {
    fn sub_assign(&mut self, rhs: AdditiveShare<F>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 -= rhs.share_0;
    }
}

impl<F: PrimeField> Zero for AdditiveShare<F> {
    fn zero() -> Self {
        AdditiveShare {
            party: 0,
            share_0: F::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.share_0.is_zero()
    }
}

impl<F: PrimeField> SSOpen<F> for AdditiveShare<F> {
    fn open(shares: &[AdditiveShare<F>]) -> F {
        assert!(shares.len() == 3);
        let mut sum = F::zero();
        for ass in shares.iter() {
            sum += ass.share_0;
        }
        sum
    }
}

impl<F: PrimeField> AdditiveShare<F> {
    pub fn with_party(mut self, party: usize) -> Self {
        self.party = party;
        self
    }

    pub fn get_mask_scalar<R: RngCore + FeedableRNG>(rng: &mut SSRandom<R>) -> F {
        let zero_share = F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0);
        rng.update();
        zero_share
    }
}

pub fn share_field_element<F: PrimeField, R: Rng>(
    val: F,
    rng: &mut R,
) -> [AdditiveShare<F>; 3] {
    let a = F::rand(rng);
    let b = F::rand(rng);
    let c = val - a - b;
    [
        AdditiveShare { party: 0, share_0: a },
        AdditiveShare { party: 1, share_0: b },
        AdditiveShare { party: 2, share_0: c },
    ]
}
