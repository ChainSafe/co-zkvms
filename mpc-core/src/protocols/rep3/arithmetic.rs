use ark_ff::PrimeField;
use ark_linear_sumcheck::rng::FeedableRNG;
use rand::RngCore;

pub use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
pub use mpc_core::protocols::rep3::arithmetic::*;

use crate::protocols::rep3::rngs::SSRandom;

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
