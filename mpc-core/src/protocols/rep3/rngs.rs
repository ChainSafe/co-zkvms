use std::ops::Add;

use ark_linear_sumcheck::rng::FeedableRNG;
pub use mpc_core::protocols::rep3::rngs::{Rep3CorrelatedRng, Rep3Rand, Rep3RandBitComp};
use rand::RngCore;

pub trait SSOpen<F: Add>: Sized {
    fn open(ss: &[Self]) -> F;
}

pub struct SSRandom<R: RngCore> {
    pub rng_0: R,
    pub rng_1: R,
    pub counter: usize,
}

impl<R: RngCore + FeedableRNG> SSRandom<R> {
    pub fn update(&mut self) {
        self.counter += 1;
        let _ = self.rng_0.feed(&self.counter);
        let _ = self.rng_1.feed(&self.counter);
    }
    pub fn new(rng_0: R, rng_1: R) -> Self {
        SSRandom {
            rng_0,
            rng_1,
            counter: 0,
        }
    }

    pub fn new_from_str(seed_0: &str, seed_1: &str) -> Self {
        let mut rng_0 = R::setup();
        let _ = rng_0.feed(&seed_0.as_bytes());
        let mut rng_1 = R::setup();
        let _ = rng_1.feed(&seed_1.as_bytes());
        SSRandom::new(rng_0, rng_1)
    }
}
