use ark_ff::PrimeField;
use itertools;
use mpc_core::protocols::rep3::Rep3BigUintShare;

pub use mpc_core::protocols::rep3::binary::*;

/// Reconstructs a vector of field elements from its binary replicated shares.
/// # Panics
/// Panics if the provided `Vec` sizes do not match.
pub fn combine_binary_elements<F: PrimeField>(
    share1: &[Rep3BigUintShare<F>],
    share2: &[Rep3BigUintShare<F>],
    share3: &[Rep3BigUintShare<F>],
) -> Vec<F> {
    assert_eq!(share1.len(), share2.len());
    assert_eq!(share2.len(), share3.len());

    itertools::multizip((share1, share2, share3))
        .map(|(x1, x2, x3)| (x1.a.clone() ^ x2.a.clone() ^ x3.a.clone()).into())
        .collect::<Vec<_>>()
}
