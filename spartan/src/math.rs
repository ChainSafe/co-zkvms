use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use ark_poly::multivariate::{SparsePolynomial, SparseTerm};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub type MaskPolynomial<E: Pairing> = SparsePolynomial<E::ScalarField, SparseTerm>;
pub trait Math {
    fn square_root(self) -> usize;
    fn pow2(self) -> usize;
    fn get_bits(self, num_bits: usize) -> Vec<bool>;
    fn log_2(self) -> usize;
}

impl Math for usize {
    #[inline]
    fn square_root(self) -> usize {
        (self as f64).sqrt() as usize
    }

    #[inline]
    fn pow2(self) -> usize {
        let base: usize = 2;
        base.pow(self as u32)
    }

    /// Returns the num_bits from n in a canonical order
    fn get_bits(self, num_bits: usize) -> Vec<bool> {
        (0..num_bits)
            .map(|shift_amount| ((self & (1 << (num_bits - shift_amount - 1))) > 0))
            .collect::<Vec<bool>>()
    }

    fn log_2(self) -> usize {
        if self.is_power_of_two() {
            (1usize.leading_zeros() - self.leading_zeros()) as usize
        } else {
            (0usize.leading_zeros() - self.leading_zeros()) as usize
        }
    }
}

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
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct SparseMatPolynomial<F: Field> {
    pub(crate) num_vars_x: usize,
    pub(crate) num_vars_y: usize,
    pub(crate) M: Vec<SparseMatEntry<F>>,
}

impl<F: Field> SparseMatPolynomial<F> {
    pub fn new(num_vars_x: usize, num_vars_y: usize, M: Vec<SparseMatEntry<F>>) -> Self {
        SparseMatPolynomial {
            num_vars_x,
            num_vars_y,
            M,
        }
    }

    pub fn num_vars_x(&self) -> usize {
        self.num_vars_x
    }
    pub fn num_vars_y(&self) -> usize {
        self.num_vars_y
    }

    pub fn num_entries(&self) -> usize {
        self.M.len()
    }

    pub fn entries(&self) -> &Vec<SparseMatEntry<F>> {
        &self.M
    }

    pub fn multiply_vec(&self, num_rows: usize, num_cols: usize, z: &[F]) -> Vec<F> {
        assert_eq!(z.len(), num_cols);

        (0..self.M.len())
            .map(|i| {
                let row = self.M[i].row;
                let col = self.M[i].col;
                let val = self.M[i].val;
                (row, val * z[col])
            })
            .fold(vec![F::zero(); num_rows], |mut Mz, (r, v)| {
                Mz[r] += v;
                Mz
            })
    }
}
