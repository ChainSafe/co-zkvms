use std::cmp::max;

use ark_ff::{Field, PrimeField};
use ark_poly::{
    multivariate::{SparsePolynomial, SparseTerm},
    DenseMultilinearExtension,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use ark_std::cfg_iter;
use noir_r1cs::{serde_ark, HydratedSparseMatrix, Interner, SparseMatrix};
use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::math::{SparseMatEntry, SparseMatPolynomial};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R1CS<F: PrimeField> {
    pub public_inputs: usize,
    pub witnesses: usize,
    pub constraints: usize,
    #[serde(with = "serde_ark")]
    pub interner: Interner<F>,
    pub a: SparseMatrix,
    pub b: SparseMatrix,
    pub c: SparseMatrix,
}

impl<F: PrimeField> R1CS<F> {
    pub fn a(&self) -> HydratedSparseMatrix<'_, F> {
        self.a.hydrate(&self.interner)
    }

    pub fn b(&self) -> HydratedSparseMatrix<'_, F> {
        self.b.hydrate(&self.interner)
    }

    pub fn c(&self) -> HydratedSparseMatrix<'_, F> {
        self.c.hydrate(&self.interner)
    }

    /// Returns ⌈log₂(instance_size)⌉ where:
    /// instance_size = max(#constraints M, #vars m, #nonzeros N)
    pub fn log2_instance_size(&self) -> usize {
        // Count non-zero entries across A, B, C
        let nonzeros = max(
            self.a.num_entries(),
            max(self.b.num_entries(), self.c.num_entries()),
        );
        // Determine maximal component
        let max_size = *[self.constraints, self.witnesses, nonzeros]
            .iter()
            .max()
            .unwrap();
        // Compute ceil(log2(max_size))
        (64 - (max_size - 1).leading_zeros()) as usize
    }
}

impl<F: PrimeField<BigInt = <noir_r1cs::FieldElement as PrimeField>::BigInt>> From<noir_r1cs::R1CS>
    for R1CS<F>
{
    fn from(r1cs: noir_r1cs::R1CS) -> Self {
        let interner = Interner::new(
            r1cs.interner
                .iter()
                .map(|v| F::from_bigint(v.into_bigint()).unwrap())
                .collect(),
        );
        Self {
            public_inputs: r1cs.public_inputs,
            witnesses: r1cs.witnesses,
            constraints: r1cs.constraints,
            interner,
            a: r1cs.a,
            b: r1cs.b,
            c: r1cs.c,
        }
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
