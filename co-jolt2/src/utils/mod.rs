pub mod future;
pub mod future_ring;
pub mod instruction_utils;
pub mod tracing;
pub mod types;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::JoltSharedPreprocessing;
use rayon::prelude::*;
use tracer::instruction::Cycle;

/// Compute ram_K from a vanilla trace and shared preprocessing.
pub fn compute_ram_k(trace: &[Cycle], preprocessing: &JoltSharedPreprocessing) -> usize {
    let max_from_trace = trace
        .par_iter()
        .filter_map(|cycle| {
            remap_address(
                cycle.ram_access().address() as u64,
                &preprocessing.memory_layout,
            )
        })
        .max()
        .unwrap_or(0);

    let max_from_bytecode = remap_address(
        preprocessing.ram.min_bytecode_address,
        &preprocessing.memory_layout,
    )
    .unwrap_or(0)
        + preprocessing.ram.bytecode_words.len() as u64
        + 1;

    max_from_trace.max(max_from_bytecode).next_power_of_two() as usize
}

/// Transpose a matrix represented as `Vec<Vec<T>>` (rows of columns)
/// into columns of rows.
pub fn transpose<T>(matrix: Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut it = matrix.into_iter();
    let first_row = match it.next() {
        Some(r) => r,
        None => return Vec::new(),
    };
    let cols = first_row.len();
    let (low, _) = it.size_hint();
    let mut out: Vec<Vec<T>> = (0..cols).map(|_| Vec::with_capacity(low + 1)).collect();

    for (c, v) in first_row.into_iter().enumerate() {
        out[c].push(v);
    }
    for row in it {
        assert_eq!(row.len(), cols, "ragged matrix");
        for (c, v) in row.into_iter().enumerate() {
            out[c].push(v);
        }
    }
    out
}
