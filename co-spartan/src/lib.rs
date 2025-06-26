mod utils;
mod setup;
mod work;

use dfs::{R1CSInstance, SparseMatEntry, SparseMatPolynomial};
use noir_r1cs::FieldElement;

pub use setup::{setup, DistributedRootKey};
pub use work::work;
pub use utils::current_num_threads;

/// Convert from a noir-r1cs R1CS struct to this R1CSInstance struct.
/// This method converts the sparse matrix format used by noir-r1cs to the
/// SparseMatPolynomial format used by this implementation.
pub fn from_noir_r1cs(noir_r1cs: &noir_r1cs::R1CS) -> R1CSInstance<FieldElement> {
    let log_instance_size = noir_r1cs.log2_instance_size();

    // Convert sparse matrix entries from noir-r1cs format to our format
    let convert_matrix = |matrix: &noir_r1cs::SparseMatrix| -> Vec<SparseMatEntry<FieldElement>> {
        let hydrated = matrix.hydrate(&noir_r1cs.interner);
        let mut entries = Vec::new();

        for ((row, col), value) in hydrated.iter() {
            entries.push(SparseMatEntry::new(row, col, value));
        }

        entries
    };

    // Convert the three matrices
    let a_entries = convert_matrix(&noir_r1cs.a);
    let b_entries = convert_matrix(&noir_r1cs.b);
    let c_entries = convert_matrix(&noir_r1cs.c);

    // Create SparseMatPolynomials
    let poly_a = SparseMatPolynomial::new(noir_r1cs.constraints, log_instance_size, a_entries);
    let poly_b = SparseMatPolynomial::new(noir_r1cs.constraints, log_instance_size, b_entries);
    let poly_c = SparseMatPolynomial::new(noir_r1cs.constraints, log_instance_size, c_entries);

    R1CSInstance {
        num_cons: 1 << log_instance_size,
        num_vars: log_instance_size,
        num_inputs: noir_r1cs.public_inputs,
        A: poly_a,
        B: poly_b,
        C: poly_c,
    }
}
