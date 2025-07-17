use ark_ff::PrimeField;
use ark_linear_sumcheck::rng::FeedableRNG;
use mpc_core::protocols::rep3::PartyID;
use rand::RngCore;

use crate::protocols::rep3::rngs::SSRandom;

pub type AdditiveShare<F> = F;

pub fn add_public<F: PrimeField>(shared: AdditiveShare<F>, public: F, id: PartyID) -> AdditiveShare<F> {
    let mut res = shared;
    match id {
        PartyID::ID0 => res += public,
        PartyID::ID1 => {},
        PartyID::ID2 => {}
    }
    res
}

pub fn sub_public<F: PrimeField>(shared: AdditiveShare<F>, public: F, id: PartyID) -> AdditiveShare<F> {
    add_public(shared, -public, id)
}

pub fn get_mask_scalar_additive<F: PrimeField, R: RngCore + FeedableRNG>(
    rng: &mut SSRandom<R>,
) -> F {
    let zero_share = F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0);
    rng.update();
    zero_share
}

pub fn promote_to_trivial_shares<F: PrimeField>(public_values: Vec<F>, id: PartyID) -> Vec<AdditiveShare<F>> {
    public_values
        .into_iter()
        .map(|value| promote_to_trivial_share(value, id))
        .collect()
}

pub fn promote_to_trivial_share<F: PrimeField>(public_value: F, id: PartyID) -> AdditiveShare<F> {
    match id {
        PartyID::ID0 => public_value,
        PartyID::ID1 => F::zero(),
        PartyID::ID2 => F::zero(),
    }
}

pub fn combine_field_elements<F: PrimeField>(
    share1: &[F],
    share2: &[F],
    share3: &[F],
) -> Vec<F> {
    assert_eq!(share1.len(), share2.len());
    assert_eq!(share2.len(), share3.len());

    itertools::multizip((share1, share2, share3))
        .map(|(&x1, &x2, &x3)| combine_field_element(&x1, &x2, &x3))
        .collect::<Vec<_>>()
}

/// Reconstructs a vector of field elements from its arithmetic replicated shares.
/// # Panics
/// Panics if the provided `Vec` sizes do not match.
pub fn combine_field_element_vec<F: PrimeField>(
    shares: Vec<Vec<F>>,
) -> Vec<F> {
   let [s0, s1, s2]: [Vec<F>; 3] = shares.try_into().unwrap();
   combine_field_elements(&s0, &s1, &s2)
}


pub fn combine_field_element<F: PrimeField>(
    share1: &F,
    share2: &F,
    share3: &F,
) -> F {
    *share1 + *share2 + *share3
}
