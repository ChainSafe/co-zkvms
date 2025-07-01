use ark_linear_sumcheck::rng::FeedableRNG;
use ark_std::ops::Add;
use rand::RngCore;

pub mod additive;
pub mod rep3;
pub mod sumcheck;

pub trait SSOpen<F: Add>: Sized {
    fn open(ss: &[Self]) -> F;
}

pub struct SSRandom<R: RngCore> {
    pub seed_0: R,
    pub seed_1: R,
    pub counter: usize,
}

impl<R: RngCore + FeedableRNG> SSRandom<R> {
    pub fn update(&mut self) {
        self.counter += 1;
        let _ = self.seed_0.feed(&self.counter);
        let _ = self.seed_1.feed(&self.counter);
    }
    pub fn new(seed_0: R, seed_1: R) -> Self {
        SSRandom {
            seed_0,
            seed_1,
            counter: 0,
        }
    }
}

