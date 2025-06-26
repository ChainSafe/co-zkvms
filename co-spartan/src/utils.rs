/// Pads the vector with 0 so that the number of elements in the vector is a
/// power of 2
pub fn pad_to_power_of_two<T: Default>(witness: &mut Vec<T>, log2n: usize) {
    let target_len = 1 << log2n;
    witness.reserve_exact(target_len - witness.len());
    while witness.len() < target_len {
        witness.push(T::default());
    }
}

/// Pads the vector with 0 so that the number of elements in the vector is a
/// power of 2
pub fn pad_to_next_power_of_two<T: Default>(witness: &mut Vec<T>) {
    let target_len = 1 << next_power_of_two(witness.len());
    witness.reserve_exact(target_len - witness.len());
    while witness.len() < target_len {
        witness.push(T::default());
    }
}

/// Calculates the degree of the next smallest power of two
pub fn next_power_of_two(n: usize) -> usize {
    let mut power = 1;
    let mut ans = 0;
    while power < n {
        power <<= 1;
        ans += 1;
    }
    ans
}

#[cfg(feature = "parallel")]
pub use rayon::current_num_threads;

#[cfg(not(feature = "parallel"))]
pub fn current_num_threads() -> usize {
    1
}

