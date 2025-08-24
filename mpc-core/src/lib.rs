pub mod lut;
pub mod protocols;

pub(crate) type RngType = rand_chacha::ChaCha12Rng;
pub(crate) type IoResult<T> = std::io::Result<T>;
pub(crate) const SEED_SIZE: usize = std::mem::size_of::<<RngType as rand::SeedableRng>::Seed>();

fn downcast<A: 'static, B: 'static>(a: &A) -> Option<&B> {
    (a as &dyn std::any::Any).downcast_ref::<B>()
}
