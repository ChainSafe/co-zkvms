pub mod multilinear_polynomial;
pub mod opening_proof;
pub mod rep3_poly;
pub mod dense_interleaved_poly;

pub use jolt_core::poly::{commitment, dense_mlpoly, eq_poly, identity_poly, unipoly};
pub use multilinear_polynomial::*;
pub use rep3_poly::*;
