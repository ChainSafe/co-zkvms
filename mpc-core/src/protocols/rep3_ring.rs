//! # REP3 Ring
//!
//! This module implements the rep3 share and combine operations for rings

pub mod arithmetic;
pub mod binary;
pub mod casts;
pub mod conversion;
use std::marker::PhantomData;

/// Re-export preprocessing module for backwards compatibility.
pub use crate::preprocessing;
pub use crate::preprocessing::{daPoint, dabits, edabits, wrap_mask};
mod detail;
pub mod gadgets;
pub mod ring;
pub mod types;

use rand::SeedableRng;
use rand::{CryptoRng, Rng, distributions::Standard, prelude::Distribution};
use ring::{int_ring::IntRing2k, ring_impl::RingElement};

use crate::serde_compat::{ark_de, ark_se};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

/// Shorthand type for a secret shared bit.
pub type Rep3BitShare = Rep3RingShare<ring::bit::Bit>;
pub use arithmetic::Rep3RingShare;
pub use arithmetic::Rep3RingSignedShare;
/// The Rng used for expanding compressed Shares
pub type SeedRng = rand_chacha::ChaCha12Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum Rep3RingShareVec<T: IntRing2k + CanonicalSerialize + CanonicalDeserialize> {
    /// A fully expanded replicated share.
    Replicated(Vec<Rep3RingShare<T>>),
    /// A compressed replicated share.
    Seeded(ReplicatedSeedType<Vec<T>, SeedRng>),
}

impl<T: IntRing2k + CanonicalSerialize + CanonicalDeserialize> Rep3RingShareVec<T>
where
    Standard: Distribution<T>,
{
    pub fn length(&self) -> usize {
        match self {
            Rep3RingShareVec::Replicated(v) => v.len(),
            Rep3RingShareVec::Seeded(s) => match &s.a {
                SeededType::Shares(v) => v.len(),
                SeededType::Seed(_, len, _) => *len,
            },
        }
    }

    pub fn is_empty(&self) -> bool {
        self.length() == 0
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = Rep3RingShare<T>> + '_> {
        match self {
            Rep3RingShareVec::Replicated(v) => Box::new(v.iter().cloned()),
            Rep3RingShareVec::Seeded(s) => {
                let it_a: Box<dyn Iterator<Item = RingElement<T>>> = match &s.a {
                    SeededType::Shares(v) => Box::new(v.iter().cloned().map(RingElement)),
                    SeededType::Seed(seed, len, _) => {
                        let mut rng = SeedRng::from_seed(seed.clone());
                        Box::new((0..*len).map(move |_| RingElement(rng.r#gen::<T>())))
                    }
                };
                let it_b: Box<dyn Iterator<Item = RingElement<T>>> = match &s.b {
                    SeededType::Shares(v) => Box::new(v.iter().cloned().map(RingElement)),
                    SeededType::Seed(seed, len, _) => {
                        let mut rng = SeedRng::from_seed(seed.clone());
                        Box::new((0..*len).map(move |_| RingElement(rng.r#gen::<T>())))
                    }
                };
                Box::new(it_a.zip(it_b).map(|(a, b)| Rep3RingShare { a, b }))
            }
        }
    }

    pub fn into_expand_vec(self) -> Vec<Rep3RingShare<T>> {
        match self {
            Rep3RingShareVec::Replicated(v) => v,
            Rep3RingShareVec::Seeded(s) => {
                let a = match s.a {
                    SeededType::Shares(v) => v,
                    SeededType::Seed(seed, len, _) => {
                        let mut rng = SeedRng::from_seed(seed);
                        (0..len).map(|_| rng.r#gen::<T>()).collect()
                    }
                };
                let b = match s.b {
                    SeededType::Shares(v) => v,
                    SeededType::Seed(seed, len, _) => {
                        let mut rng = SeedRng::from_seed(seed);
                        (0..len).map(|_| rng.r#gen::<T>()).collect()
                    }
                };
                a.into_iter().zip(b).map(|(a, b)| Rep3RingShare { a: RingElement(a), b: RingElement(b) }).collect()
            }
        }
    }

    pub fn to_vec(&self) -> Vec<Rep3RingShare<T>> {
        self.iter().collect()
    }
}

/// A type that represents a compressed additive share. It can either be a seed (with length) or the actual share.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum SeededType<T: Clone + CanonicalSerialize + CanonicalDeserialize, U: Rng + SeedableRng + CryptoRng>
where
    U::Seed: std::fmt::Debug + Clone + Serialize + for<'a> Deserialize<'a>,
{
    /// The actual additive share
    Shares(#[serde(serialize_with = "ark_se", deserialize_with = "ark_de")] T),
    /// A compressed additive share
    Seed(U::Seed, usize, PhantomData<U>),
}

impl<T: Clone + CanonicalSerialize + CanonicalDeserialize, U: Rng + SeedableRng + CryptoRng> Clone for SeededType<T, U>
where
    U::Seed: std::fmt::Debug + Clone + Serialize + for<'a> Deserialize<'a>,
{
    fn clone(&self) -> Self {
        match self {
            SeededType::Shares(val) => SeededType::Shares(val.clone()),
            SeededType::Seed(seed, len, _) => SeededType::Seed(seed.clone(), *len, PhantomData),
        }
    }
}

/// A type that represents a compressed replicated share. It consists of two compressed additive shares.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ReplicatedSeedType<T: Clone + CanonicalSerialize + CanonicalDeserialize, U: Rng + SeedableRng + CryptoRng>
where
    U::Seed: std::fmt::Debug + Clone + Serialize + for<'a> Deserialize<'a>,
{
    /// The first compressed additive share
    pub a: SeededType<T, U>,
    /// The second compressed additive share
    pub b: SeededType<T, U>,
}

/// Secret shares a ring element using replicated secret sharing and the provided random number generator. The ring element is split into three additive shares, where each party holds two. The outputs are of type [`Rep3RingShare`].
pub fn share_ring_element<T: IntRing2k, R: Rng + CryptoRng>(val: RingElement<T>, rng: &mut R) -> [Rep3RingShare<T>; 3]
where
    Standard: Distribution<T>,
{
    let a = rng.r#gen::<RingElement<T>>();
    let b = rng.r#gen::<RingElement<T>>();

    let c = val - a - b;
    let share1 = Rep3RingShare::new_ring(a, c);
    let share2 = Rep3RingShare::new_ring(b, a);
    let share3 = Rep3RingShare::new_ring(c, b);
    [share1, share2, share3]
}

/// Secret shares a vector of ring elements using replicated secret sharing and the provided random number generator. The ring elements are split into three additive shares each, where each party holds two. The outputs are of type [`Rep3RingShare`].
pub fn share_ring_elements<T: IntRing2k, R: Rng + CryptoRng>(
    vals: &[RingElement<T>],
    rng: &mut R,
) -> [Vec<Rep3RingShare<T>>; 3]
where
    Standard: Distribution<T>,
{
    let mut shares1 = Vec::with_capacity(vals.len());
    let mut shares2 = Vec::with_capacity(vals.len());
    let mut shares3 = Vec::with_capacity(vals.len());
    for val in vals {
        let [share1, share2, share3] = share_ring_element(val.to_owned(), rng);
        shares1.push(share1);
        shares2.push(share2);
        shares3.push(share3);
    }
    [shares1, shares2, shares3]
}

/// Secret shares a ring element using replicated secret sharing and the provided random number generator. The ring element is split into three binary shares, where each party holds two. The outputs are of type [`Rep3RingShare`].
pub fn share_ring_element_binary<T: IntRing2k, R: Rng + CryptoRng>(
    val: RingElement<T>,
    rng: &mut R,
) -> [Rep3RingShare<T>; 3]
where
    Standard: Distribution<T>,
{
    let a = rng.r#gen::<RingElement<T>>();
    let b = rng.r#gen::<RingElement<T>>();
    let c = val ^ a ^ b;
    let share1 = Rep3RingShare::new_ring(a, c);
    let share2 = Rep3RingShare::new_ring(b, a);
    let share3 = Rep3RingShare::new_ring(c, b);
    [share1, share2, share3]
}

/// Reconstructs a ring element from its arithmetic replicated shares.
pub fn combine_ring_element<T: IntRing2k>(
    share1: Rep3RingShare<T>,
    share2: Rep3RingShare<T>,
    share3: Rep3RingShare<T>,
) -> RingElement<T> {
    share1.a + share2.a + share3.a
}

/// Reconstructs a vector of ring elements from its arithmetic replicated shares.
/// # Panics
/// Panics if the provided `Vec` sizes do not match.
pub fn combine_ring_elements<T: IntRing2k>(
    share1: &[Rep3RingShare<T>],
    share2: &[Rep3RingShare<T>],
    share3: &[Rep3RingShare<T>],
) -> Vec<RingElement<T>> {
    assert_eq!(share1.len(), share2.len());
    assert_eq!(share2.len(), share3.len());

    itertools::multizip((share1, share2, share3)).map(|(x1, x2, x3)| x1.a + x2.a + x3.a).collect::<Vec<_>>()
}

/// Reconstructs a ring element from its binary replicated shares.
pub fn combine_ring_element_binary<T: IntRing2k>(
    share1: Rep3RingShare<T>,
    share2: Rep3RingShare<T>,
    share3: Rep3RingShare<T>,
) -> RingElement<T> {
    share1.a ^ share2.a ^ share3.a
}

pub fn share_ring_elements_seeded<T: IntRing2k + CanonicalSerialize + CanonicalDeserialize, R: Rng + CryptoRng>(
    vals: &[RingElement<T>],
    rng: &mut R,
) -> [Rep3RingShareVec<T>; 3]
where
    Standard: Distribution<T>,
{
    let len = vals.len();
    let seed_b = rng.r#gen::<<SeedRng as SeedableRng>::Seed>();
    let seed_c = rng.r#gen::<<SeedRng as SeedableRng>::Seed>();

    let mut rng_b = SeedRng::from_seed(seed_b.clone());
    let mut rng_c = SeedRng::from_seed(seed_c.clone());

    let b_seeded = SeededType::Seed(seed_b, len, PhantomData);
    let c_seeded = SeededType::Seed(seed_c, len, PhantomData);

    let mut a = Vec::with_capacity(len);
    for val in vals {
        let b_rand = rng_b.r#gen::<T>();
        let c_rand = rng_c.r#gen::<T>();
        let a_val = val.0.wrapping_sub(&b_rand).wrapping_sub(&c_rand);
        a.push(a_val);
    }

    let a_seeded = SeededType::Shares(a);

    let share1 = ReplicatedSeedType { a: a_seeded.clone(), b: c_seeded.clone() };
    let share2 = ReplicatedSeedType { a: b_seeded.clone(), b: a_seeded };
    let share3 = ReplicatedSeedType { a: c_seeded, b: b_seeded };

    [Rep3RingShareVec::Seeded(share1), Rep3RingShareVec::Seeded(share2), Rep3RingShareVec::Seeded(share3)]
}

impl<T: IntRing2k + CanonicalSerialize + CanonicalDeserialize> PartialEq for Rep3RingShareVec<T>
where
    Standard: Distribution<T>,
{
    fn eq(&self, other: &Self) -> bool {
        if self.length() != other.length() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: IntRing2k + CanonicalSerialize + CanonicalDeserialize> Eq for Rep3RingShareVec<T> where Standard: Distribution<T>
{}

impl<T: IntRing2k + CanonicalSerialize + CanonicalDeserialize> Default for Rep3RingShareVec<T> {
    fn default() -> Self {
        Rep3RingShareVec::Replicated(Vec::new())
    }
}
