use crate::{errors::R1CSError, math::Math};
/// Largely taken from https://github.com/arkworks-rs/spartan
///
///
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, cfg_iter};
use rand::Rng;
use std::fmt;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

mod errors;
mod math;
pub mod mpi_snark;
pub mod mpi_utils;
pub mod snark;
pub mod logup;
pub mod transcript;
#[macro_use]
pub mod utils;
pub mod mpc;
pub mod mpc_snark;
pub mod co_spartan;
pub mod network;

/// The domain separator, used when proving statements on dfs.
pub(crate) const PROTOCOL_NAME: &[u8] = b"DFS-v0";

/// Error identifying a failure in the proof verification.
#[derive(Debug, Clone)]
pub struct VerificationError;

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Verification Error.")
    }
}

/// Verification result.
pub type VerificationResult = std::result::Result<(), VerificationError>;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct SparseMatEntry<F: Field> {
    row: usize,
    col: usize,
    val: F,
}

impl<F: PrimeField> SparseMatEntry<F> {
    pub fn new(row: usize, col: usize, val: F) -> Self {
        SparseMatEntry { row, col, val }
    }
}
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct SparseMatPolynomial<F: Field> {
    num_vars_x: usize,
    num_vars_y: usize,
    M: Vec<SparseMatEntry<F>>,
}

impl<F: Field> SparseMatPolynomial<F> {
    pub fn new(num_vars_x: usize, num_vars_y: usize, M: Vec<SparseMatEntry<F>>) -> Self {
        SparseMatPolynomial {
            num_vars_x,
            num_vars_y,
            M,
        }
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

    #[allow(non_snake_case)]
    pub fn produce_synthetic_r1cs(
        num_cons: usize,
        size_z: usize,
        num_witness: usize,
        mut prng: &mut impl Rng,
    ) -> (R1CSInstance<F>, Vec<F>, Vec<F>) {
        // z is organized as [io,1,witness]
        //let size_z = num_witness + num_inputs + 1;
        //num_inputs contains (1,io)
        let num_inputs = size_z - num_witness;

        // produce a random satisfying assignment
        let Z = {
            let mut Z: Vec<F> = (0..size_z).map(|_i| F::rand(&mut prng)).collect::<Vec<F>>();
            Z[num_inputs - 1] = F::one(); // set the constant term to 1
            Z
        };

        // three sparse matrices
        let mut A: Vec<SparseMatEntry<F>> = Vec::new();
        let mut B: Vec<SparseMatEntry<F>> = Vec::new();
        let mut C: Vec<SparseMatEntry<F>> = Vec::new();
        let one = F::one();
        for i in 0..num_cons {
            let A_idx = i % size_z;
            let B_idx = (i + 2) % size_z;
            A.push(SparseMatEntry::new(i, A_idx, one));
            B.push(SparseMatEntry::new(i, B_idx, one));
            let AB_val = Z[A_idx] * Z[B_idx];

            let C_idx = (i + 3) % size_z;
            let C_val = Z[C_idx];

            if C_val == F::zero() {
                C.push(SparseMatEntry::new(i, num_witness, AB_val));
            } else {
                C.push(SparseMatEntry::new(
                    i,
                    C_idx,
                    AB_val * C_val.inverse().unwrap(),
                ));
            }
        }

        let num_poly_vars_x = num_cons;
        let num_poly_vars_y = size_z;
        let poly_A = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, A);
        let poly_B = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, B);
        let poly_C = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, C);

        let inst = R1CSInstance {
            num_cons,
            num_vars: size_z,
            num_inputs,
            A: poly_A,
            B: poly_B,
            C: poly_C,
        };

        //assert!(inst.is_sat(&Z[..num_vars], &Z[num_vars + 1..]));

        //(inst, Z[..num_vars].to_vec(), Z[num_vars + 1..].to_vec())
        (inst, Z[..num_inputs].to_vec(), Z[num_inputs..].to_vec())
    }

    /*pub fn is_sat(&self, vars: &[F], input: &[F]) -> bool {
        assert_eq!(vars.len(), self.num_vars);
        assert_eq!(input.len(), self.num_inputs);

        let z = {
            let mut z = vars.to_vec();
            z.extend(&vec![F::one()]);
            z.extend(input);
            z
        };

        // verify if Az * Bz - Cz = [0...]
        let Az = self
            .A
            .multiply_vec(self.num_cons, self.num_vars + self.num_inputs + 1, &z);
        let Bz = self
            .B
            .multiply_vec(self.num_cons, self.num_vars + self.num_inputs + 1, &z);
        let Cz = self
            .C
            .multiply_vec(self.num_cons, self.num_vars + self.num_inputs + 1, &z);

        assert_eq!(Az.len(), self.num_cons);
        assert_eq!(Bz.len(), self.num_cons);
        assert_eq!(Cz.len(), self.num_cons);
        let res: usize = (0..self.num_cons)
            .map(|i| if Az[i] * Bz[i] == Cz[i] { 0 } else { 1 })
            .sum();

        res == 0
    }

    pub fn multiply_vec(
        &self,
        num_rows: usize,
        num_cols: usize,
        z: &[F],
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        assert_eq!(num_rows, self.num_cons);
        assert_eq!(z.len(), num_cols);
        assert!(num_cols > self.num_vars);
        (
            self.A.multiply_vec(num_rows, num_cols, z),
            self.B.multiply_vec(num_rows, num_cols, z),
            self.C.multiply_vec(num_rows, num_cols, z),
        )
    }*/
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
