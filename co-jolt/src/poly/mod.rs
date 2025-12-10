pub mod commitment;
pub mod dense_interleaved_poly;
pub mod dense_mlpoly;
pub mod mixed_polynomial;
pub mod multilinear_polynomial;
pub mod opening_proof;
pub mod sparse_interleaved_poly;
pub mod spartan_interleaved_poly;
pub mod split_eq_poly;
pub mod unipoly;

pub use dense_mlpoly::*;
pub use jolt_core::poly::{eq_poly, identity_poly};
pub use multilinear_polynomial::*;

use crate::{field::JoltField, utils::types::Rep3Value};

pub trait Polynomial<F: JoltField> {
    fn len(&self) -> usize;

    fn get_num_vars(&self) -> usize;

    fn get_bound_coeffs(&self) -> Vec<Rep3Value<F>>;
}
