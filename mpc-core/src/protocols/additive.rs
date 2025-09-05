use ark_ff::PrimeField;
use ark_linear_sumcheck::rng::FeedableRNG;
use eyre::Context;
use crate::protocols::rep3::{
    PartyID,
    network::{IoContext, Rep3Network},
};
use mpc_types::protocols::additive::AdditivePrimeFieldShare;
use rand::RngCore;

use crate::protocols::rep3::rngs::SSRandom;

pub type AdditiveShare<F> = AdditivePrimeFieldShare<F>;

pub fn add_public<F: PrimeField>(
    shared: AdditiveShare<F>,
    public: F,
    id: PartyID,
) -> AdditiveShare<F> {
    match id {
        PartyID::ID0 => AdditiveShare::from_fe(shared.into_fe() + public),
        PartyID::ID1 => shared,
        PartyID::ID2 => shared,
    }
}

pub fn sub_shared_by_public<F: PrimeField>(
    shared: AdditiveShare<F>,
    public: F,
    id: PartyID,
) -> AdditiveShare<F> {
    add_public(shared, -public, id)
}

/// Performs subtraction between a shared value and a public value, returning public - shared.
pub fn sub_public_by_shared<F: PrimeField>(
    public: F,
    shared: AdditiveShare<F>,
    id: PartyID,
) -> AdditiveShare<F> {
    add_public(-shared, public, id)
}

pub fn get_mask_scalar_additive<F: PrimeField, R: RngCore + FeedableRNG>(
    rng: &mut SSRandom<R>,
) -> F {
    let zero_share = F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0);
    rng.update();
    zero_share
}

pub fn promote_to_trivial_shares<F: PrimeField>(
    public_values: Vec<F>,
    id: PartyID,
) -> Vec<AdditiveShare<F>> {
    public_values
        .into_iter()
        .map(|value| promote_to_trivial_share(value, id))
        .collect()
}

pub fn promote_to_trivial_share<F: PrimeField>(public_value: F, id: PartyID) -> AdditiveShare<F> {
    AdditiveShare::promote_from_trivial(public_value, id)
}

fn combine_field_elements<F: PrimeField>(
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

pub fn combine_additive_shares<F: PrimeField>(
    share1: &[AdditiveShare<F>],
    share2: &[AdditiveShare<F>],
    share3: &[AdditiveShare<F>],
) -> Vec<F> {
    assert_eq!(share1.len(), share2.len());
    assert_eq!(share2.len(), share3.len());

    itertools::multizip((share1, share2, share3))
        .map(|(&x1, &x2, &x3)| combine_field_element(&x1.into_fe(), &x2.into_fe(), &x3.into_fe()))
        .collect::<Vec<_>>()
}

pub fn combine_additive_share<F: PrimeField>(
    shares: Vec<AdditiveShare<F>>,
) -> F {
    assert_eq!(shares.len(), 3);

    AdditiveShare::into_fe_vec(shares).iter().sum::<F>()
}

/// Reconstructs a vector of field elements from its arithmetic replicated shares.
/// # Panics
/// Panics if the provided `Vec` sizes do not match.
pub fn combine_additive_vec<F: PrimeField>(shares: Vec<Vec<AdditiveShare<F>>>) -> Vec<F> {
    let [s0, s1, s2]: [Vec<AdditiveShare<F>>; 3] = shares.try_into().unwrap();
    combine_additive_shares(
        &s0,
        &s1,
        &s2,
    )
}

fn combine_field_element<F: PrimeField>(share1: &F, share2: &F, share3: &F) -> F {
    *share1 + *share2 + *share3
}

pub fn open<F: PrimeField, Network: Rep3Network>(
    a: F,
    io_ctx: &mut IoContext<Network>,
) -> eyre::Result<F> {
    Ok(open_vec(vec![a], io_ctx)?[0])
}

pub fn open_vec<F: PrimeField, Network: Rep3Network>(
    a: Vec<F>,
    io_ctx: &mut IoContext<Network>,
) -> eyre::Result<Vec<F>> {
    io_ctx.network.send_many(io_ctx.id.prev_id(), &a)?;
    io_ctx.network.send_many(io_ctx.id.next_id(), &a)?;
    let prev = io_ctx
        .network
        .recv_many(io_ctx.id.prev_id())
        .context("while receiving previous shares")?;
    let next = io_ctx
        .network
        .recv_many(io_ctx.id.next_id())
        .context("while sending shares")?;

    let res = combine_field_elements(&a, &prev, &next);

    Ok(res)
}
