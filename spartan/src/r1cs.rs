use crate::math::{SparseMatEntry, SparseMatPolynomial};
use ark_ff::{Field, PrimeField};
use ark_poly::{
    multivariate::{SparsePolynomial, SparseTerm},
    DenseMultilinearExtension,
};
use ark_serialize::SerializationError;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::cfg_iter;
use rand::Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct R1CSInstance<F: PrimeField> {
    pub num_cons: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub A: SparseMatPolynomial<F>,
    pub B: SparseMatPolynomial<F>,
    pub C: SparseMatPolynomial<F>,
}
impl<F: PrimeField> R1CSInstance<F> {
    /// `A`, `B`, and `C` are the three matrices defining the R1CS instance.
    /// They are assumed to be in row-major order.
    pub fn new(
        num_cons: usize,
        num_vars: usize,
        num_inputs: usize,
        A: &[(usize, usize, F)],
        B: &[(usize, usize, F)],
        C: &[(usize, usize, F)],
    ) -> R1CSInstance<F> {
        // Timer::print(&format!("number_of_constraints {}", num_cons));
        // Timer::print(&format!("number_of_variables {}", num_vars));
        // Timer::print(&format!("number_of_inputs {}", num_inputs));
        // Timer::print(&format!("number_non-zero_entries_A {}", A.len()));
        // Timer::print(&format!("number_non-zero_entries_B {}", B.len()));
        // Timer::print(&format!("number_non-zero_entries_C {}", C.len()));

        // check that num_cons is a power of 2
        assert_eq!(num_cons.next_power_of_two(), num_cons);

        // check that num_vars is a power of 2
        // assert_eq!(num_vars.next_power_of_two(), num_vars);

        // check that number_inputs + 1 <= num_vars
        assert!(num_inputs < num_vars);

        // no errors, so create polynomials
        let num_poly_vars_x = num_cons as usize;
        let num_poly_vars_y = num_vars as usize;

        let mat_A = cfg_iter!(&A)
            .map(|(row, col, val)| SparseMatEntry::new(*row, *col, *val))
            .collect();
        let mat_B = cfg_iter!(&B)
            .map(|(row, col, val)| SparseMatEntry::new(*row, *col, *val))
            .collect();
        let mat_C = cfg_iter!(&C)
            .map(|(row, col, val)| SparseMatEntry::new(*row, *col, *val))
            .collect();

        let poly_A = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, mat_A);
        let poly_B = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, mat_B);
        let poly_C = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, mat_C);

        R1CSInstance {
            num_cons,
            num_vars,
            num_inputs,
            A: poly_A,
            B: poly_B,
            C: poly_C,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn get_num_cons(&self) -> usize {
        self.num_cons
    }

    pub fn get_num_inputs(&self) -> usize {
        self.num_inputs
    }
}

/// `Instance` holds the description of R1CS matrices
pub struct Instance<F: PrimeField> {
    inst: R1CSInstance<F>,
}

impl<F: PrimeField> Instance<F> {
    /// Constructs a new `Instance` and an associated satisfying assignment
    pub fn new(
        num_cons: usize,
        num_vars: usize,
        num_inputs: usize,
        A: &[(usize, usize, F)],
        B: &[(usize, usize, F)],
        C: &[(usize, usize, F)],
    ) -> Result<Self, R1CSError> {
        let (num_vars_padded, num_cons_padded) = {
            let num_vars_padded = {
                let mut num_vars_padded = num_vars;

                // ensure that num_inputs + 1 <= num_vars
                num_vars_padded = std::cmp::max(num_vars_padded, num_inputs + 1);

                // ensure that num_vars_padded a power of two
                if num_vars_padded.next_power_of_two() != num_vars_padded {
                    num_vars_padded = num_vars_padded.next_power_of_two();
                }
                num_vars_padded
            };

            let num_cons_padded = {
                let mut num_cons_padded = num_cons;

                // ensure that num_cons_padded is at least 2
                if num_cons_padded == 0 || num_cons_padded == 1 {
                    num_cons_padded = 2;
                }

                // ensure that num_cons_padded is power of 2
                if num_cons.next_power_of_two() != num_cons {
                    num_cons_padded = num_cons.next_power_of_two();
                }
                num_cons_padded
            };

            (num_vars_padded, num_cons_padded)
        };

        let bytes_to_scalar =
            |tups: &[(usize, usize, F)]| -> Result<Vec<(usize, usize, F)>, R1CSError> {
                let mut mat: Vec<(usize, usize, F)> = Vec::new();
                for &(row, col, val) in tups {
                    // row must be smaller than num_cons
                    if row >= num_cons {
                        return Err(R1CSError::InvalidIndices);
                    }

                    // col must be smaller than num_vars + 1 + num_inputs
                    if col >= num_vars + 1 + num_inputs {
                        return Err(R1CSError::InvalidIndices);
                    }

                    if col >= num_vars {
                        mat.push((row, col + num_vars_padded - num_vars, val));
                    } else {
                        mat.push((row, col, val));
                    }
                }

                // pad with additional constraints up until num_cons_padded if the original constraints were 0 or 1
                // we do not need to pad otherwise because the dummy constraints are implicit in the sum-check protocol
                if num_cons == 0 || num_cons == 1 {
                    for i in tups.len()..num_cons_padded {
                        mat.push((i, num_vars, F::zero()));
                    }
                }

                Ok(mat)
            };

        let A_scalar = bytes_to_scalar(A);
        if A_scalar.is_err() {
            return Err(A_scalar.err().unwrap());
        }

        let B_scalar = bytes_to_scalar(B);
        if B_scalar.is_err() {
            return Err(B_scalar.err().unwrap());
        }

        let C_scalar = bytes_to_scalar(C);
        if C_scalar.is_err() {
            return Err(C_scalar.err().unwrap());
        }

        let inst = R1CSInstance::<F>::new(
            num_cons_padded,
            num_vars_padded,
            num_inputs,
            &A_scalar.unwrap(),
            &B_scalar.unwrap(),
            &C_scalar.unwrap(),
        );

        Ok(Instance { inst })
    }
}

#[derive(Debug)]
pub enum R1CSError {
    /// returned if the number of constraints is not a power of 2
    NonPowerOfTwoCons,
    /// returned if the number of variables is not a power of 2
    NonPowerOfTwoVars,
    /// returned if a wrong number of inputs in an assignment are supplied
    InvalidNumberOfInputs,
    /// returned if a wrong number of variables in an assignment are supplied
    InvalidNumberOfVars,
    /// returned if a [u8;32] does not parse into a valid Scalar in the field of ristretto255
    InvalidScalar,
    /// returned if the supplied row or col in (row,col,val) tuple is out of range
    InvalidIndices,
    /// Ark serialization error
    ArkSerializationError(SerializationError),
}

impl From<SerializationError> for R1CSError {
    fn from(e: SerializationError) -> Self {
        Self::ArkSerializationError(e)
    }
}
