//! # REP3
//!
//! This module implements the rep3 share and combine operations

pub mod arithmetic;
pub mod binary;
pub mod conversion;
pub mod detail;
pub use crate::network;
pub mod pointshare;
pub mod rngs;
#[cfg(feature = "test-utils")]
pub mod test_utils;
pub mod types;

use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_bigint::BigUint;

use crate::field::PrimeField;
use ark_ff::One;
use rand::{CryptoRng, Rng, SeedableRng, distributions::Standard, prelude::Distribution};

pub use crate::network::PartyID;
pub use arithmetic::Rep3PrimeFieldShare;
pub use binary::Rep3BigUintShare;

pub(crate) type IoResult<T> = std::io::Result<T>;

/// Secret shares a field element using replicated secret sharing and the provided random number generator. The field element is split into three additive shares, where each party holds two. The outputs are of type [Rep3PrimeFieldShare].
pub fn share_field_element<F: PrimeField, R: Rng + CryptoRng>(val: F, rng: &mut R) -> [Rep3PrimeFieldShare<F>; 3] {
    let a = F::random(rng);
    let b = F::random(rng);
    let c = val - a - b;
    let share1 = Rep3PrimeFieldShare::new(a, c);
    let share2 = Rep3PrimeFieldShare::new(b, a);
    let share3 = Rep3PrimeFieldShare::new(c, b);
    [share1, share2, share3]
}

/// Secret shares a field element using additive secret sharing and the provided random number generator. The field element is split into three additive shares. The outputs are three [PrimeField].
pub fn share_field_element_additive<F: PrimeField, R: Rng + CryptoRng>(val: F, rng: &mut R) -> [F; 3] {
    let a = F::random(rng);
    let b = F::random(rng);
    let c = val - a - b;
    [a, b, c]
}

/// Secret shares a vector of field elements using replicated secret sharing and the provided random number generator. The field elements are split into three additive shares each, where each party holds two. The outputs are of type [Rep3PrimeFieldShare].
pub fn share_field_elements<F: PrimeField, R: Rng + CryptoRng>(
    vals: &[F],
    rng: &mut R,
) -> [Vec<Rep3PrimeFieldShare<F>>; 3] {
    let mut shares1 = Vec::with_capacity(vals.len());
    let mut shares2 = Vec::with_capacity(vals.len());
    let mut shares3 = Vec::with_capacity(vals.len());
    for val in vals {
        let [share1, share2, share3] = share_field_element(*val, rng);
        shares1.push(share1);
        shares2.push(share2);
        shares3.push(share3);
    }
    [shares1, shares2, shares3]
}

/// Secret shares a vector of field element using replicated secret sharing and the provided random number generator. The field elements are split into three additive shares each, where each party holds two. The outputs are of type [Rep3PrimeFieldShare].
pub fn share_maybe_field_elements<F: PrimeField, R: Rng + CryptoRng>(
    vals: &[Option<F>],
    rng: &mut R,
) -> [Vec<Option<Rep3PrimeFieldShare<F>>>; 3] {
    let mut shares1 = Vec::with_capacity(vals.len());
    let mut shares2 = Vec::with_capacity(vals.len());
    let mut shares3 = Vec::with_capacity(vals.len());
    for val in vals {
        if let Some(val) = val {
            let [share1, share2, share3] = share_field_element(*val, rng);
            shares1.push(Some(share1));
            shares2.push(Some(share2));
            shares3.push(Some(share3));
        } else {
            shares1.push(None);
            shares2.push(None);
            shares3.push(None);
        }
    }
    [shares1, shares2, shares3]
}

/// Secret shares a vector of field element using additive secret sharing and the provided random number generator. The field elements are split into three additive shares each. The outputs are `Vecs` of type [`PrimeField`].
pub fn share_field_elements_additive<F: PrimeField, R: Rng + CryptoRng>(vals: &[F], rng: &mut R) -> [Vec<F>; 3] {
    let mut shares1 = Vec::with_capacity(vals.len());
    let mut shares2 = Vec::with_capacity(vals.len());
    let mut shares3 = Vec::with_capacity(vals.len());
    for val in vals {
        let [share1, share2, share3] = share_field_element_additive(*val, rng);
        shares1.push(share1);
        shares2.push(share2);
        shares3.push(share3);
    }
    [shares1, shares2, shares3]
}

/// Secret shares a field element using replicated secret sharing and the provided random number generator. The field element is split into three binary shares, where each party holds two. The outputs are of type [Rep3BigUintShare].
pub fn share_biguint<F: PrimeField, R: Rng + CryptoRng>(val: F, rng: &mut R) -> [Rep3BigUintShare<F>; 3] {
    let val: BigUint = val.into_biguint();
    let limbsize = F::MODULUS_BIT_SIZE.div_ceil(32);
    let mask = (BigUint::from(1u32) << F::MODULUS_BIT_SIZE) - BigUint::one();
    let a = BigUint::new((0..limbsize).map(|_| rng.r#gen()).collect()) & &mask;
    let b = BigUint::new((0..limbsize).map(|_| rng.r#gen()).collect()) & mask;

    let c = val ^ &a ^ &b;
    let share1 = Rep3BigUintShare::new(a.to_owned(), c.to_owned());
    let share2 = Rep3BigUintShare::new(b.to_owned(), a);
    let share3 = Rep3BigUintShare::new(c, b);
    [share1, share2, share3]
}

/// Reconstructs a field element from its arithmetic replicated shares.
pub fn combine_field_element<F: PrimeField>(
    share1: Rep3PrimeFieldShare<F>,
    share2: Rep3PrimeFieldShare<F>,
    share3: Rep3PrimeFieldShare<F>,
) -> F {
    share1.a + share2.a + share3.a
}

/// Reconstructs a vector of field elements from its arithmetic replicated shares.
/// # Panics
/// Panics if the provided `Vec` sizes do not match.
pub fn combine_field_elements<F: PrimeField>(
    share1: &[Rep3PrimeFieldShare<F>],
    share2: &[Rep3PrimeFieldShare<F>],
    share3: &[Rep3PrimeFieldShare<F>],
) -> Vec<F> {
    assert_eq!(share1.len(), share2.len());
    assert_eq!(share2.len(), share3.len());

    itertools::multizip((share1, share2, share3)).map(|(x1, x2, x3)| x1.a + x2.a + x3.a).collect::<Vec<_>>()
}

/// Reconstructs a value (represented as [BigUint]) from its binary replicated shares. Since binary operations can lead to results >= p, the result is not guaranteed to be a valid field element.
pub fn combine_binary_element<F: PrimeField>(
    share1: Rep3BigUintShare<F>,
    share2: Rep3BigUintShare<F>,
    share3: Rep3BigUintShare<F>,
) -> BigUint {
    share1.a ^ share2.a ^ share3.a
}
