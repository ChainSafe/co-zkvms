use ark_std::ops::Add;

pub mod ass;
pub mod rss;
pub mod utils;

pub trait SSOpen<F: Add>: Sized {
    fn open(ss: &[Self]) -> F;
}
