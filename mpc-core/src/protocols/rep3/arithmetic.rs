use ark_ff::PrimeField;
use ark_linear_sumcheck::rng::FeedableRNG;
use rand::Rng;
use rand::RngCore;

pub use mpc_core::protocols::rep3::arithmetic::*;
pub use mpc_core::protocols::rep3::{
    Rep3PrimeFieldShare,
    network::{IoContext, Rep3Network},
};

use crate::protocols::additive::AdditiveShare;
use crate::protocols::rep3::rngs::SSRandom;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn generate_shares_rep3<F: PrimeField, R: Rng>(
    val: F,
    rng: &mut R,
) -> (
    Rep3PrimeFieldShare<F>,
    Rep3PrimeFieldShare<F>,
    Rep3PrimeFieldShare<F>,
) {
    let t0 = F::rand(rng);
    let t1 = F::rand(rng);
    let t2 = val - t0 - t1;

    let p_share_0 = Rep3PrimeFieldShare::new(t0, t2); // Party 0 gets (t_0, t_2)
    let p_share_1 = Rep3PrimeFieldShare::new(t1, t0); // Party 1 gets (t_1, t_0)
    let p_share_2 = Rep3PrimeFieldShare::new(t2, t1); // Party 2 gets (t_2, t_1)
    (p_share_0, p_share_1, p_share_2)
}

pub fn open<F: PrimeField>(shares: [Rep3PrimeFieldShare<F>; 3], id: usize) -> F {
    shares[id].a + shares[id].b + shares[(id + 2) % 3].b
}

pub fn get_mask_scalar_rep3<F: PrimeField, R: RngCore + FeedableRNG>(
    rng: &mut SSRandom<R>,
) -> (F, F) {
    let mask_share = (
        F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0),
        F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0),
    );
    rng.update();
    mask_share
}

pub fn reshare_additive<F: PrimeField, N: Rep3Network>(
    additive: AdditiveShare<F>,
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Rep3PrimeFieldShare<F>> {
    let prev_share: F = io_ctx.network.reshare(additive)?;
    Ok(Rep3PrimeFieldShare::new(additive, prev_share))
}

pub fn reshare_additive_many<F: PrimeField, N: Rep3Network>(
    additive_shares: &[AdditiveShare<F>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let b_shares: Vec<F> = io_ctx.network.reshare_many(additive_shares)?;
    Ok(additive_shares
        .into_par_iter()
        .zip(b_shares.into_par_iter())
        .map(|(a, b)| Rep3PrimeFieldShare::new(*a, b))
        .collect())
}

/// Convenience method for \[a\] * (\[b\] * c)
pub fn mul_mul_public<F: PrimeField>(a: FieldShare<F>, b: FieldShare<F>, c: F) -> F {
    a * mul_public(b, c)
}

/// Reconstructs a vector of field elements from its arithmetic replicated shares.
/// # Panics
/// Panics if the provided `Vec` sizes do not match.
pub fn combine_field_elements_vec<F: PrimeField>(
    shares: Vec<Vec<Rep3PrimeFieldShare<F>>>,
) -> Vec<F> {
    let [s0, s1, s2]: [Vec<_>; 3] = shares.try_into().unwrap();
    mpc_core::protocols::rep3::combine_field_elements(&s0, &s1, &s2)
}
