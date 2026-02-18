pub mod future;
pub mod future_ring;
pub mod instruction_utils;
pub mod tracing;
pub mod types;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

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
