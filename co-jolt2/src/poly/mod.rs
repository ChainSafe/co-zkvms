pub mod dense_mlpoly;
pub mod mixed_polynomial;
pub mod multilinear_polynomial;

pub use dense_mlpoly::*;
pub use multilinear_polynomial::*;

use crate::{field::JoltField, utils::types::Rep3Value};

pub trait Polynomial<F: JoltField> {
    fn len(&self) -> usize;

    fn get_num_vars(&self) -> usize;

    fn get_bound_coeffs(&self) -> Vec<Rep3Value<F>>;
}
