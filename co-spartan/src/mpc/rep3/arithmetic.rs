use ark_ff::{PrimeField, Zero};
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{CryptoRng, Rng, RngCore};
use std::ops::{Add, AddAssign, Mul, Sub};

use crate::mpc::{additive::AdditiveShare, SSOpen, SSRandom};

pub type ArithmeticShare<F> = Rep3Share<F>;

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone, Copy)]
pub struct Rep3Share<F: PrimeField> {
    pub party: usize,
    pub share_0: F,
    pub share_1: F,
}

impl<F: PrimeField> Rep3Share<F> {
    pub fn new(share_0: F, share_1: F) -> Self {
        Rep3Share {
            party: 0,
            share_0,
            share_1,
        }
    }

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

    pub fn get_mask_scalar<R: RngCore + FeedableRNG>(rng: &mut SSRandom<R>) -> (F, F) {
        let mask_share = (
            F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0),
            F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0),
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

/// Secret shares a field element using replicated secret sharing and the provided random number generator. The field element is split into three additive shares, where each party holds two. The outputs are of type [Rep3PrimeFieldShare].
pub fn share_field_element<F: PrimeField, R: Rng + CryptoRng>(
    val: F,
    rng: &mut R,
) -> [Rep3Share<F>; 3] {
    let a = F::rand(rng);
    let b = F::rand(rng);
    let c = val - a - b;
    let share1 = Rep3Share::new(a, c).with_party(0);
    let share2 = Rep3Share::new(b, a).with_party(1);
    let share3 = Rep3Share::new(c, b).with_party(2);
    [share1, share2, share3]
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
