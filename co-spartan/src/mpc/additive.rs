use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Index, Mul, Neg, Sub, SubAssign},
    rc::Rc,
};

use ark_ec::{bls12::Bls12, pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{Field, One, PrimeField, UniformRand, Zero};
use ark_linear_sumcheck::{
    ml_sumcheck::{
        protocol::{
            prover, prover::ProverMsg, verifier::VerifierMsg, IPForMLSumcheck,
            ListOfProductsOfPolynomials,
        },
        Proof,
    },
    rng::{Blake2s512Rng, FeedableRNG},
    Error,
};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter_mut, test_rng};
use rand::RngCore;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::mpc::{SSOpen, SSRandom};

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone, Copy)]
pub struct AdditiveShare<E: Pairing> {
    pub party: usize,
    pub share_0: E::ScalarField,
}

impl<E: Pairing> Add<Self> for AdditiveShare<E> {
    type Output = Self;
    fn add(self, rhs: AdditiveShare<E>) -> <Self as Add<AdditiveShare<E>>>::Output {
        assert_eq!(self.party, rhs.party);
        AdditiveShare {
            party: self.party,
            share_0: self.share_0 + rhs.share_0,
        }
    }
}

impl<E: Pairing> Sub<Self> for AdditiveShare<E> {
    type Output = Self;
    fn sub(self, rhs: AdditiveShare<E>) -> Self::Output {
        assert_eq!(self.party, rhs.party);
        AdditiveShare {
            party: self.party,
            share_0: self.share_0 - rhs.share_0,
        }
    }
}

impl<E: Pairing> Mul<E::ScalarField> for AdditiveShare<E> {
    type Output = Self;
    fn mul(self, rhs: E::ScalarField) -> Self::Output {
        AdditiveShare {
            party: self.party,
            share_0: self.share_0 * rhs,
        }
    }
}

impl<E: Pairing> AddAssign for AdditiveShare<E> {
    fn add_assign(&mut self, rhs: AdditiveShare<E>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 += rhs.share_0;
    }
}

impl<'a, E: Pairing> AddAssign<&'a AdditiveShare<E>> for AdditiveShare<E> {
    fn add_assign(&mut self, rhs: &AdditiveShare<E>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 += rhs.share_0;
    }
}

impl<E: Pairing> SubAssign for AdditiveShare<E> {
    fn sub_assign(&mut self, rhs: AdditiveShare<E>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 -= rhs.share_0;
    }
}

impl<E: Pairing> Zero for AdditiveShare<E> {
    fn zero() -> Self {
        AdditiveShare {
            party: 0,
            share_0: E::ScalarField::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.share_0.is_zero()
    }
}

impl<E: Pairing> SSOpen<E::ScalarField> for AdditiveShare<E> {
    fn open(shares: &[AdditiveShare<E>]) -> <E as Pairing>::ScalarField {
        assert!(shares.len() == 2);
        let mut sum = E::ScalarField::zero();
        for ass in shares.iter() {
            sum += ass.share_0;
        }
        sum
    }
}

impl<E: Pairing> AdditiveShare<E> {
    pub fn with_party(mut self, party: usize) -> Self {
        self.party = party;
        self
    }

    pub fn get_mask_scalar<R: RngCore + FeedableRNG>(rng: &mut SSRandom<R>) -> E::ScalarField {
        let zero_share =
            E::ScalarField::rand(&mut rng.seed_1) - E::ScalarField::rand(&mut rng.seed_0);
        rng.update();
        zero_share
    }
}
