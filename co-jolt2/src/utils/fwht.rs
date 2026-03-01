use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use rayon::prelude::*;

use crate::field::JoltField;

/// Fast Walsh-Hadamard Transform in-place on a slice of Rep3 field shares.
///
/// This is purely local (no MPC communication) — it only performs additions
/// and subtractions on the share components.
///
/// Precondition: `a.len()` must be a power of two.
#[inline]
pub fn fwht_rep3_in_place<F: JoltField>(a: &mut [Rep3PrimeFieldShare<F>]) {
    debug_assert!(
        a.len().is_power_of_two(),
        "FWHT input length must be power-of-two, got {}",
        a.len()
    );
    let n = a.len();
    let mut len = 1usize;
    while len < n {
        let step = len * 2;
        for i in (0..n).step_by(step) {
            for j in 0..len {
                let u = a[i + j];
                let v = a[i + j + len];
                a[i + j] = u + v;
                a[i + j + len] = u - v;
            }
        }
        len = step;
    }
}

/// Fast Walsh-Hadamard Transform in-place on additive shares.
///
/// Purely local — only additions and subtractions.
#[inline]
pub fn fwht_additive_in_place<F: JoltField>(a: &mut [AdditiveShare<F>]) {
    debug_assert!(
        a.len().is_power_of_two(),
        "FWHT input length must be power-of-two, got {}",
        a.len()
    );
    let n = a.len();
    let mut len = 1usize;
    while len < n {
        let step = len * 2;
        for i in (0..n).step_by(step) {
            for j in 0..len {
                let u = a[i + j];
                let v = a[i + j + len];
                a[i + j] = u + v;
                a[i + j + len] = u - v;
            }
        }
        len = step;
    }
}

/// FWHT on a slice of plain field elements (public values).
#[inline]
pub fn fwht_field_in_place<F: JoltField>(a: &mut [F]) {
    debug_assert!(
        a.len().is_power_of_two(),
        "FWHT input length must be power-of-two, got {}",
        a.len()
    );
    let n = a.len();
    let mut len = 1usize;
    while len < n {
        let step = len * 2;
        for i in (0..n).step_by(step) {
            for j in 0..len {
                let u = a[i + j];
                let v = a[i + j + len];
                a[i + j] = u + v;
                a[i + j + len] = u - v;
            }
        }
        len = step;
    }
}

/// Unmask a public (plain F) histogram using FWHT and ehat.
///
/// Equivalent to `unmask_histogram(promote(h_f), ehat)` but ~2x cheaper:
/// - First FWHT operates on F (1 field elem) instead of Rep3 (2 field elems)
/// - Pointwise F×Rep3→Additive: ~1 op/elem avg vs Rep3×Rep3→Additive: 5 ops/elem
pub fn unmask_histogram_public<F: JoltField>(
    h_f: &mut [F],
    ehat: &[Rep3PrimeFieldShare<F>],
    party_id: mpc_core::protocols::rep3::PartyID,
) -> Vec<AdditiveShare<F>> {
    use mpc_core::protocols::rep3::PartyID;
    let m = h_f.len();
    debug_assert_eq!(m, ehat.len());
    debug_assert!(m.is_power_of_two());

    // FWHT on plain F — half the cost of fwht_rep3
    fwht_field_in_place(h_f);

    // Pointwise F × Rep3 → Additive (no communication, parallelized)
    // promote(f, id) * rep3 = trivial_rep3 * rep3
    // ID0: (f,0)*(a,b) → f*a + f*b = f*(a+b)
    // ID1: (0,f)*(a,b) → f*a
    // ID2: (0,0)*(a,b) → 0
    let mut result: Vec<AdditiveShare<F>> = match party_id {
        PartyID::ID0 => h_f
            .par_iter()
            .zip(ehat.par_iter())
            .map(|(&f, e)| AdditiveShare::from_fe(f * (e.a + e.b)))
            .collect(),
        PartyID::ID1 => h_f
            .par_iter()
            .zip(ehat.par_iter())
            .map(|(&f, e)| AdditiveShare::from_fe(f * e.a))
            .collect(),
        PartyID::ID2 => vec![AdditiveShare::zero(); m],
    };

    fwht_additive_in_place(&mut result);

    let inv_m = F::from(m as u64)
        .inverse()
        .expect("M must be invertible in field");
    result.par_iter_mut().for_each(|r| *r = *r * inv_m);

    result
}

/// XOR-convolution of two length-N vectors of Rep3 shares:
///   result[k] = (1/N) * Σ_c a[c] * b[c XOR k]
///
/// Computed via FWHT:
///   result = invFWHT(FWHT(a) ⊙ FWHT(b)) / N
///
/// where invFWHT = FWHT (self-inverse up to the 1/N factor).
///
/// `a` and `b` are modified in-place as scratch space.
/// Returns the result vector.
///
/// **Note**: The pointwise multiply `⊙` is shared×shared, so
/// the caller must provide a function to perform it (via `mul_fn`).
/// The `mul_fn` takes the full slices and returns the pointwise product.
///
/// For local (no-network) usage, the caller can provide a function that
/// does element-wise `Rep3PrimeFieldShare::mul_lazy()` or similar.
pub fn xor_convolve_rep3_with_mul<F: JoltField>(
    a: &mut [Rep3PrimeFieldShare<F>],
    b: &mut [Rep3PrimeFieldShare<F>],
    mul_fn: impl FnOnce(
        &[Rep3PrimeFieldShare<F>],
        &[Rep3PrimeFieldShare<F>],
    ) -> Vec<Rep3PrimeFieldShare<F>>,
) -> Vec<Rep3PrimeFieldShare<F>> {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert!(n.is_power_of_two());

    fwht_rep3_in_place(a);
    fwht_rep3_in_place(b);

    let mut result = mul_fn(a, b);

    fwht_rep3_in_place(&mut result);

    let inv_n = F::from(n as u64)
        .inverse()
        .expect("N must be invertible in field");
    for r in result.iter_mut() {
        *r = *r * inv_n;
    }

    result
}

/// Unmask a histogram from masked domain to true domain using FWHT.
///
/// Given:
///   - `h_c`: histogram in masked domain (length M), secret-shared
///   - `ehat`: FWHT of the one-hot mask vector E (length M), secret-shared
///   - `mul_fn`: batched shared×shared multiply
///
/// Computes: `h_k = invFWHT(FWHT(h_c) ⊙ ehat) / M`
///
/// This is the core unmasking step from section C/G3 of the PLAN.
pub fn unmask_histogram<F: JoltField>(
    h_c: &mut [Rep3PrimeFieldShare<F>],
    ehat: &[Rep3PrimeFieldShare<F>],
    mul_fn: impl FnOnce(
        &[Rep3PrimeFieldShare<F>],
        &[Rep3PrimeFieldShare<F>],
    ) -> Vec<Rep3PrimeFieldShare<F>>,
) -> Vec<Rep3PrimeFieldShare<F>> {
    let m = h_c.len();
    debug_assert_eq!(m, ehat.len());
    debug_assert!(m.is_power_of_two());

    fwht_rep3_in_place(h_c);

    let mut result = mul_fn(h_c, ehat);

    fwht_rep3_in_place(&mut result);

    let inv_m = F::from(m as u64)
        .inverse()
        .expect("M must be invertible in field");
    for r in result.iter_mut() {
        *r = *r * inv_m;
    }

    result
}

/// Compute the Ehat16 tensor product from two 8-bit Ehat vectors.
///
/// Given:
///   - `ehat8_hi`: FWHT(E8_hi) of length 256, secret-shared
///   - `ehat8_lo`: FWHT(E8_lo) of length 256, secret-shared
///   - `mul_fn`: batched shared×shared multiply (length 65536)
///
/// Computes: `Ehat16[(a<<8)|b] = Ehat8_hi[a] * Ehat8_lo[b]`
///
/// This is section F of the PLAN.
pub fn compute_ehat16_tensor<F: JoltField>(
    ehat8_hi: &[Rep3PrimeFieldShare<F>],
    ehat8_lo: &[Rep3PrimeFieldShare<F>],
    mul_fn: impl FnOnce(
        &[Rep3PrimeFieldShare<F>],
        &[Rep3PrimeFieldShare<F>],
    ) -> Vec<Rep3PrimeFieldShare<F>>,
) -> Vec<Rep3PrimeFieldShare<F>> {
    debug_assert_eq!(ehat8_hi.len(), 256);
    debug_assert_eq!(ehat8_lo.len(), 256);

    let m = 65536usize; // 256 * 256
    let mut a_expanded = Vec::with_capacity(m);
    let mut b_expanded = Vec::with_capacity(m);

    for a_idx in 0..256 {
        for _b_idx in 0..256 {
            a_expanded.push(ehat8_hi[a_idx]);
        }
    }
    for _a_idx in 0..256 {
        for b_idx in 0..256 {
            b_expanded.push(ehat8_lo[b_idx]);
        }
    }

    mul_fn(&a_expanded, &b_expanded)
}

/// Shift a public EQ table into masked domain using a secret mask one-hot.
///
/// Given:
///   - `eq_table`: public EQ table (length M)
///   - `ehat`: FWHT of the one-hot mask vector E (length M), secret-shared
///   - `party_id`: for promoting public values to trivial shares
///
/// Computes `eq_shifted[c] = eq_table[c XOR r]` as secret shares,
/// using the identity: eq_shifted = invFWHT(FWHT(eq_table) ⊙ Ehat) / M.
///
/// This uses only public×share multiplication (no network).
///
/// This is section G4 of the PLAN.
pub fn shift_eq_table_with_mask<F: JoltField>(
    eq_table: &[F],
    ehat: &[Rep3PrimeFieldShare<F>],
) -> Vec<Rep3PrimeFieldShare<F>> {
    let m = eq_table.len();
    debug_assert_eq!(m, ehat.len());
    debug_assert!(m.is_power_of_two());

    let mut eq_hat = eq_table.to_vec();
    fwht_field_in_place(&mut eq_hat);

    // Pointwise multiply: public × share (no communication, parallelized)
    let mut result: Vec<Rep3PrimeFieldShare<F>> = ehat
        .par_iter()
        .zip(eq_hat.par_iter())
        .map(|(e, &eq)| *e * eq)
        .collect();

    fwht_rep3_in_place(&mut result);

    let inv_m = F::from(m as u64)
        .inverse()
        .expect("M must be invertible in field");
    result.par_iter_mut().for_each(|r| *r = *r * inv_m);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use mpc_core::protocols::rep3::combine_field_element;

    fn share_field<R: rand::Rng>(val: Fr, rng: &mut R) -> [Rep3PrimeFieldShare<Fr>; 3] {
        let shares =
            mpc_core::protocols::rep3::arithmetic::generate_shares_rep3::<Fr, _>(val, rng);
        shares.try_into().unwrap()
    }

    #[test]
    fn fwht_rep3_roundtrip() {
        let mut rng = ark_std::test_rng();
        let n = 16usize;

        // Create shared values
        let plain: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64 + 1)).collect();
        let mut shares: [Vec<Rep3PrimeFieldShare<Fr>>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(n));
        for &v in &plain {
            let s = share_field(v, &mut rng);
            for pid in 0..3 {
                shares[pid].push(s[pid]);
            }
        }

        // Forward FWHT on each party's shares
        for pid in 0..3 {
            fwht_rep3_in_place(&mut shares[pid]);
        }

        // Also do forward on plaintext
        let mut plain_fwht = plain.clone();
        fwht_field_in_place(&mut plain_fwht);

        // Reconstruct and check forward transform matches
        for i in 0..n {
            let got = combine_field_element(shares[0][i], shares[1][i], shares[2][i]);
            assert_eq!(got, plain_fwht[i], "forward FWHT mismatch at index {i}");
        }

        // Inverse (apply FWHT again + divide by N)
        for pid in 0..3 {
            fwht_rep3_in_place(&mut shares[pid]);
            let inv_n = Fr::from(n as u64).inverse().unwrap();
            for s in shares[pid].iter_mut() {
                *s = *s * inv_n;
            }
        }

        // Should recover original values
        for i in 0..n {
            let got = combine_field_element(shares[0][i], shares[1][i], shares[2][i]);
            assert_eq!(got, plain[i], "roundtrip mismatch at index {i}");
        }
    }

    #[test]
    fn shift_eq_table_correct() {
        let mut rng = ark_std::test_rng();
        let n = 256usize;

        // Random mask r
        let r_mask: u8 = rand::Rng::gen(&mut rng);

        // Build E = one-hot(r_mask) and share it
        let mut e_shares: [Vec<Rep3PrimeFieldShare<Fr>>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(n));
        for i in 0..n {
            let bit = if i as u8 == r_mask {
                Fr::from(1u64)
            } else {
                Fr::from(0u64)
            };
            let s = share_field(bit, &mut rng);
            for pid in 0..3 {
                e_shares[pid].push(s[pid]);
            }
        }

        // Compute Ehat for each party
        let mut ehat: [Vec<Rep3PrimeFieldShare<Fr>>; 3] = e_shares.clone();
        for pid in 0..3 {
            fwht_rep3_in_place(&mut ehat[pid]);
        }

        // Random public EQ table
        let eq_table: Vec<Fr> = (0..n)
            .map(|_| <Fr as ark_ff::UniformRand>::rand(&mut rng))
            .collect();

        // Shift using each party's Ehat
        let shifted: [Vec<Rep3PrimeFieldShare<Fr>>; 3] =
            std::array::from_fn(|pid| shift_eq_table_with_mask(&eq_table, &ehat[pid]));

        // Reconstruct and verify: shifted[c] == eq_table[c ^ r_mask]
        for c in 0..n {
            let got = combine_field_element(shifted[0][c], shifted[1][c], shifted[2][c]);
            let want = eq_table[c ^ r_mask as usize];
            assert_eq!(got, want, "shift mismatch at c={c}, r_mask={r_mask}");
        }
    }
}
