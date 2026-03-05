pub mod future;
pub mod future_ring;
pub mod fwht;
pub mod instruction_utils;
pub mod memory;
pub(crate) mod send_ptr;
pub mod tracing;
pub mod types;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

use crate::field::JoltField;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::JoltSharedPreprocessing;
use mpc_core::protocols::additive::AdditiveShare;
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

/// Lagrange interpolation through 4 points (0, y0), (1, y1), (2, y2), (3, y3) at x.
///
/// Denominators are constants: -6, 2, -2, 6. Their inverses are precomputed.
pub(crate) fn lagrange_interp_4<F: JoltField>(
    y0: AdditiveShare<F>,
    y1: AdditiveShare<F>,
    y2: AdditiveShare<F>,
    y3: AdditiveShare<F>,
    x: F,
) -> AdditiveShare<F> {
    let inv6 = F::from(6u64).inverse().unwrap();
    let inv2 = F::TWO_INV;
    let inv_neg6 = -inv6;
    let inv_neg2 = -inv2;

    let xm1 = x - F::one();
    let xm2 = x - F::from(2u64);
    let xm3 = x - F::from(3u64);

    let l0 = xm1 * xm2 * xm3 * inv_neg6;
    let l1 = x * xm2 * xm3 * inv2;
    let l2 = x * xm1 * xm3 * inv_neg2;
    let l3 = x * xm1 * xm2 * inv6;

    y0 * l0 + y1 * l1 + y2 * l2 + y3 * l3
}
