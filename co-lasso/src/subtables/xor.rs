use ark_ff::PrimeField;
use ark_std::log2;
use jolt_core::poly::{dense_mlpoly::DensePolynomial, field::JoltField};

pub struct InstructionPreprocessing<F: JoltField> {
  subtable_to_memory_indices: Vec<Vec<usize>>,
  instruction_to_memory_indices: Vec<Vec<usize>>,
  memory_to_subtable_index: Vec<usize>,
  memory_to_dimension_index: Vec<usize>,
  materialized_subtables: Vec<Vec<F>>,
  num_memories: usize,
}

pub struct InstructionPolynomials<F: JoltField> {
   /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
    /// indices at which the memories will be evaluated. Each `DensePolynomial` has size
    /// `m` (# lookups).
    pub dim: Vec<DensePolynomial<F>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// read access counts to the memory. Each `DensePolynomial` has size `m` (# lookups).
    pub read_cts: Vec<DensePolynomial<F>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// final access counts to the memory. Each `DensePolynomial` has size M, AKA subtable size.
    pub final_cts: Vec<DensePolynomial<F>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// the evaluation of memory accessed at each step of the CPU. Each `DensePolynomial` has
    /// size `m` (# lookups).
    pub E_polys: Vec<DensePolynomial<F>>,

    /// The lookup output for each instruction of the execution trace.
    pub lookup_outputs: DensePolynomial<F>,
}

pub fn polynomialize(preprocessing: &InstructionPreprocessing<F>) -> InstructionPolynomials<F> {

}
