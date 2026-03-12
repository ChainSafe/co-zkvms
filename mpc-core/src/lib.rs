pub mod field;
pub mod preprocessing;
pub mod protocols;
pub mod serde_compat;
pub mod utils;

pub use serde_compat::{ark_de, ark_se};
pub use utils::MaybeShared;

pub(crate) type RngType = rand_chacha::ChaCha12Rng;
pub(crate) type IoResult<T> = std::io::Result<T>;
pub(crate) const SEED_SIZE: usize = std::mem::size_of::<<RngType as rand::SeedableRng>::Seed>();

fn downcast<A: 'static, B: 'static>(a: &A) -> Option<&B> {
    (a as &dyn std::any::Any).downcast_ref::<B>()
}
