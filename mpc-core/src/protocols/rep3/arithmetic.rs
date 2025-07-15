use ark_ff::PrimeField;
use ark_linear_sumcheck::rng::FeedableRNG;
use rand::Rng;
use rand::RngCore;

pub use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
pub use mpc_core::protocols::rep3::arithmetic::*;

use crate::protocols::rep3::rngs::SSRandom;

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
