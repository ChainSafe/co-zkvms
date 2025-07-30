pub mod commitment;
pub mod dense_interleaved_poly;
pub mod dense_mlpoly;
pub mod mixed_polynomial;
pub mod multilinear_polynomial;
pub mod opening_proof;
pub mod sparse_interleaved_poly;
pub mod spartan_interleaved_poly;

pub use dense_mlpoly::*;
pub use jolt_core::poly::{eq_poly, identity_poly, unipoly};
pub use multilinear_polynomial::*;

pub trait PolyDegree {
    fn len(&self) -> usize;

    fn get_num_vars(&self) -> usize;
}
