pub mod future;
pub mod future_ring;
pub mod instruction_utils;
pub mod types;
pub mod transcript;

use std::collections::HashMap;

pub use jolt_core::{
    poly::dense_mlpoly::DensePolynomial,
    utils::{
        compute_dotproduct, errors, gaussian_elimination, gen_random_point,
        index_to_field_bitvector, is_power_of_two, math, mul_0_1_optimized, mul_0_optimized,
        split_bits, thread,
    },
};

use rayon::prelude::*;

pub fn transpose<I, T>(matrix: I) -> Vec<Vec<T>>
where
    I: IntoIterator<Item = Vec<T>>,
{
    let mut it = matrix.into_iter();
    let first_row = match it.next() {
        Some(r) => r,
        None => return Vec::new(),
    };
    let cols = first_row.len();
    let (low, _) = it.size_hint();
    let mut out: Vec<Vec<T>> = (0..cols).map(|_| Vec::with_capacity(low + 1)).collect();

    // push first row
    for (c, v) in first_row.into_iter().enumerate() {
        out[c].push(v);
    }
    // push remaining rows
    for row in it {
        assert_eq!(row.len(), cols, "ragged matrix");
        for (c, v) in row.into_iter().enumerate() {
            out[c].push(v);
        }
    }
    out
}

/// Parallel transpose of a single matrix.
///
/// Accepts an owned matrix `Vec<Vec<T>>` and returns its transpose.
/// Uses Rayon when the `parallel` feature is enabled; falls back to the
/// sequential `transpose` otherwise.
pub fn transpose_par_from_flat<T>(matrix_flat: Vec<T>, rows: usize, cols: usize) -> Vec<Vec<T>>
where
    T: Send + Sync,
{
    use rayon::prelude::*;
    use std::mem::ManuallyDrop;

    assert_eq!(matrix_flat.len(), rows * cols, "matrix dimensions mismatch");

    // Flatten input row-major so we can relocate elements into transposed order.
    let mut flat = ManuallyDrop::new(matrix_flat);
    // Capture address as usize to avoid sharing raw pointer across threads (not Sync)
    let src_addr = flat.as_mut_ptr() as usize;

    // Prepare output columns; each column is length `rows` and will be filled in parallel.
    let mut out: Vec<Vec<T>> = (0..cols).map(|_| Vec::with_capacity(rows)).collect();

    out.par_iter_mut().enumerate().for_each(|(c, col)| {
        unsafe {
            col.set_len(rows);
            let dst_ptr = col.as_mut_ptr();
            for r in 0..rows {
                let idx = r * cols + c; // index into row-major source
                let src_ptr = src_addr as *const T;
                let val = std::ptr::read(src_ptr.add(idx));
                std::ptr::write(dst_ptr.add(r), val);
            }
        }
    });

    // Deallocate the flattened buffer without dropping moved elements.
    unsafe {
        let cap = flat.capacity();
        let ptr = flat.as_mut_ptr();
        let _ = Vec::from_raw_parts(ptr, 0, cap);
    }

    out
}

pub fn transpose_flatten<I, T>(matrix: I) -> Vec<Vec<T>>
where
    I: IntoIterator<Item = Vec<Vec<T>>>, // [R][C][D] with D possibly var-length
{
    let mut rows = matrix.into_iter();
    let first = match rows.next() {
        Some(r) => r,
        None => return Vec::new(),
    };
    let cols = first.len();
    let (low, _) = rows.size_hint();
    // estimate avg depth from first row
    let avg_depth = if cols > 0 {
        first.iter().map(Vec::len).sum::<usize>() / cols
    } else {
        0
    };
    // pre-allocate each column to (rows_est × avg_depth)
    let mut out: Vec<Vec<T>> = (0..cols)
        .map(|_| Vec::with_capacity((low + 1) * avg_depth))
        .collect();

    // flatten first row
    for (c, dv) in first.into_iter().enumerate() {
        out[c].extend(dv);
    }
    // flatten remaining rows
    for row in rows {
        assert_eq!(row.len(), cols, "ragged cols");
        for (c, dv) in row.into_iter().enumerate() {
            out[c].extend(dv);
        }
    }
    out
}

/// Serial, cache-friendly tiled transpose of a single matrix.
///
/// `tile` controls the square tile size used for blocking. For best results,
/// choose a power-of-two that fits your cache (e.g., 32 or 64 for small T).
pub fn transpose_tiled<T>(matrix: Vec<Vec<T>>, tile: usize) -> Vec<Vec<T>> {
    use std::mem::ManuallyDrop;

    let rows = matrix.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = matrix[0].len();
    for row in &matrix {
        assert_eq!(row.len(), cols, "ragged matrix");
    }

    // Flatten input row-major so we can relocate elements into transposed order.
    let flat: Vec<T> = matrix.into_iter().flat_map(|r| r.into_iter()).collect();
    let mut flat = ManuallyDrop::new(flat);
    let src_ptr = flat.as_mut_ptr();

    // Prepare output columns; each column is length `rows` and will be filled.
    let mut out: Vec<Vec<T>> = (0..cols).map(|_| Vec::with_capacity(rows)).collect();

    let row_tiles = (rows + tile.saturating_sub(1)) / tile.max(1);
    let col_tiles = (cols + tile.saturating_sub(1)) / tile.max(1);

    for ct in 0..col_tiles {
        let c_start = ct * tile;
        let c_end = (c_start + tile).min(cols);
        for rt in 0..row_tiles {
            let r_start = rt * tile;
            let r_end = (r_start + tile).min(rows);
            for c in c_start..c_end {
                let col = &mut out[c];
                unsafe {
                    if col.len() < r_end {
                        col.set_len(r_end);
                    }
                    let dst_ptr = col.as_mut_ptr();
                    for r in r_start..r_end {
                        let idx = r * cols + c; // index into row-major source
                        let val = std::ptr::read(src_ptr.add(idx));
                        std::ptr::write(dst_ptr.add(r), val);
                    }
                }
            }
        }
    }

    // Deallocate the flattened buffer without dropping moved elements.
    unsafe {
        let cap = flat.capacity();
        let ptr = flat.as_mut_ptr();
        let _ = Vec::from_raw_parts(ptr, 0, cap);
    }

    out
}

pub fn transpose_hashmap<T>(rows: Vec<HashMap<usize, T>>) -> HashMap<usize, Vec<T>> {
    let mut out: HashMap<usize, Vec<T>> = HashMap::new();
    for (_, row) in rows.into_iter().enumerate() {
        for (k, v) in row {
            out.entry(k).or_default().push(v);
        }
    }
    out
}

pub fn chunks_take_nth<'a, T>(
    data: &'a [T],
    chunk_len: usize,
    step: usize,
) -> impl Iterator<Item = impl Iterator<Item = &'a T>> {
    // for each offset 0‥step-1 build a strided view
    (0..step).map(move |off| data.iter().skip(off).step_by(step).take(chunk_len))
}
