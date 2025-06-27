use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_linear_sumcheck::rng::Blake2s512Rng;
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use rand::Rng;
use rand::RngCore;

use super::rss::SSRandom;
use ark_ff::Zero;
// use ark_ff::UniformRand;

#[derive(PartialEq, Eq)]
pub enum ShareType {
    Ass,
    Rss,
}

pub fn generate_poly_shares_rss<F: Field, R: Rng>(
    poly: &DenseMultilinearExtension<F>,
    rng: &mut R,
) -> (
    DenseMultilinearExtension<F>,
    DenseMultilinearExtension<F>,
    DenseMultilinearExtension<F>,
) {
    let num_vars = poly.num_vars;
    let p_share_0 = DenseMultilinearExtension::<F>::rand(num_vars, rng);
    let p_share_1 = DenseMultilinearExtension::<F>::rand(num_vars, rng);
    let p_share_2 = (poly - &p_share_0) - p_share_1.clone();

    (p_share_0, p_share_1, p_share_2)
}

pub fn generate_rss_share_randomness<R: RngCore + FeedableRNG>() -> Vec<SSRandom<R>> {
    let mut seed_0 = R::setup();
    seed_0.feed(&"seed 0".as_bytes());
    let mut seed_1 = R::setup();
    seed_1.feed(&"seed 1".as_bytes());
    let mut random_0 = SSRandom::<R>::new(seed_0, seed_1);

    let mut seed_1 = R::setup();
    seed_1.feed(&"seed 1".as_bytes());
    let mut seed_2 = R::setup();
    seed_2.feed(&"seed 2".as_bytes());
    let mut random_1 = SSRandom::<R>::new(seed_1, seed_2);

    let mut seed_2 = R::setup();
    seed_2.feed(&"seed 2".as_bytes());
    let mut seed_0 = R::setup();
    seed_0.feed(&"seed 0".as_bytes());
    let mut random_2 = SSRandom::<R>::new(seed_2, seed_0);

    let mut vec_random = vec![random_0, random_1, random_2];
    vec_random
}

pub fn generate_ass_share_randomness<R: RngCore + FeedableRNG>() -> Vec<SSRandom<R>> {
    let mut seed_0 = R::setup();
    seed_0.feed(&"seed 0".as_bytes());
    let mut seed_1 = R::setup();
    seed_1.feed(&"seed 1".as_bytes());
    let mut random_0 = SSRandom::<R>::new(seed_0, seed_1);

    let mut seed_1 = R::setup();
    seed_1.feed(&"seed 1".as_bytes());
    let mut seed_0 = R::setup();
    seed_0.feed(&"seed 0".as_bytes());
    let mut random_1 = SSRandom::<R>::new(seed_1, seed_0);

    let mut vec_random = vec![random_0, random_1];
    vec_random
}
