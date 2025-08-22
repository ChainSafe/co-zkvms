use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use ark_poly::multivariate::{SparsePolynomial, SparseTerm};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[allow(type_alias_bounds)]
pub type MaskPolynomial<E: Pairing> = SparsePolynomial<E::ScalarField, SparseTerm>;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct SparseMatEntry<F: Field> {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) val: F,
}

impl<F: PrimeField> SparseMatEntry<F> {
    pub fn new(row: usize, col: usize, val: F) -> Self {
        SparseMatEntry { row, col, val }
    }
}
