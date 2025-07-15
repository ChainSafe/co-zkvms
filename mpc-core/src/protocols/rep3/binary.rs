use ark_ff::{BigInteger, PrimeField};
use itertools;
use mpc_core::protocols::rep3::Rep3BigUintShare;
use num_bigint::BigUint;
use rand::Rng;

pub use mpc_core::protocols::rep3::binary::*;

pub fn generate_shares_rep3<F: PrimeField, R: Rng>(
    val: F,
    rng: &mut R,
) -> (
    Rep3BigUintShare<F>,
    Rep3BigUintShare<F>,
    Rep3BigUintShare<F>,
) {
    let val = BigUint::from_bytes_le(&val.into_bigint().to_bytes_le());
    let t0 = BigUint::from(rng.r#gen::<u64>());
    let t1 = BigUint::from(rng.r#gen::<u64>());
    let t2 = (val ^ t0.clone()) ^ t1.clone();

    let p_share_0 = Rep3BigUintShare::new(t0.clone(), t2.clone());
    let p_share_1 = Rep3BigUintShare::new(t1.clone(), t0);
    let p_share_2 = Rep3BigUintShare::new(t2, t1);
    (p_share_0, p_share_1, p_share_2)
}

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

/// Reconstructs a value (represented as [BigUint]) from its binary replicated shares. Since binary operations can lead to results >= p, the result is not guaranteed to be a valid field element.
pub fn combine_binary_element<F: PrimeField>(
    share1: Rep3BigUintShare<F>,
    share2: Rep3BigUintShare<F>,
    share3: Rep3BigUintShare<F>,
) -> F {
    (share1.a ^ share2.a ^ share3.a).into()
}

#[cfg(test)]
mod tests {
    use crate::protocols::rep3;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use super::*;

    type F = ark_bn254::Fr;
    #[test]
    fn test_share_rep3_binary() {
        let mut rng = test_rng();
        let secret = F::rand(&mut rng);
        let shares = generate_shares_rep3(secret.clone(), &mut rng);

        let combined = combine_binary_element(shares.0, shares.1, shares.2);
        assert_eq!(combined, secret);
    }
}
