pub mod protocols;

pub use mpc_core::lut;

pub(crate) type RngType = rand_chacha::ChaCha12Rng;
pub(crate) type IoResult<T> = std::io::Result<T>;
pub(crate) const SEED_SIZE: usize = std::mem::size_of::<<RngType as rand::SeedableRng>::Seed>();
