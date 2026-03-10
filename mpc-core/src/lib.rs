pub mod lut;
pub mod protocols;

pub use mpc_types::serde_compat::{ark_de, ark_se};
pub use mpc_types::utils::MaybeShared;

pub(crate) type RngType = rand_chacha::ChaCha12Rng;
pub(crate) type IoResult<T> = std::io::Result<T>;
pub(crate) const SEED_SIZE: usize = std::mem::size_of::<<RngType as rand::SeedableRng>::Seed>();

fn downcast<A: 'static, B: 'static>(a: &A) -> Option<&B> {
    (a as &dyn std::any::Any).downcast_ref::<B>()
}
