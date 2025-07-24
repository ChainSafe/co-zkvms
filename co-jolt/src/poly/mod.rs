pub mod multilinear_polynomial;
pub mod opening_proof;
pub mod dense_mlpoly;
pub mod dense_interleaved_poly;
pub mod sparse_interleaved_poly;
pub mod spartan_interleaved_poly;
pub mod mixed_polynomial;

pub use jolt_core::poly::{commitment, eq_poly, identity_poly, unipoly};
pub use multilinear_polynomial::*;
pub use dense_mlpoly::*;

pub trait PolyDegree {
    fn len(&self) -> usize;

    fn get_num_vars(&self) -> usize;
}
