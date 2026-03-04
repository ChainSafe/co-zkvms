use std::ops::{Add, Sub};

use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use rayon::prelude::*;

use crate::field::JoltField;

// ─── Butterfly primitives ─────────────────────────────────────────────────────

/// Radix-4 butterfly: fuses two consecutive radix-2 stages.
/// Loads 4 values, performs 8 add/sub, writes 4 back — halving memory traffic
/// vs two separate radix-2 passes over the same data.
#[inline(always)]
fn butterfly4<T: Copy + Add<Output = T> + Sub<Output = T>>(
    a: &mut T,
    b: &mut T,
    c: &mut T,
    d: &mut T,
) {
    // Stage 1: butterflies at stride 1×len
    let t0 = *a + *b;
    let t1 = *a - *b;
    let t2 = *c + *d;
    let t3 = *c - *d;
    // Stage 2: butterflies at stride 2×len
    *a = t0 + t2;
    *b = t1 + t3;
    *c = t0 - t2;
    *d = t1 - t3;
}

/// Radix-8 butterfly: fuses three consecutive radix-2 stages.
/// Loads 8 values, performs 24 add/sub, writes 8 back — ⅓ memory traffic.
/// Higher register pressure (8 live values × element size); beneficial on
/// architectures with deep out-of-order buffers and fast store-forwarding.
#[cfg(any(target_arch = "x86_64", test))]
#[inline(always)]
fn butterfly8<T: Copy + Add<Output = T> + Sub<Output = T>>(
    a: &mut T,
    b: &mut T,
    c: &mut T,
    d: &mut T,
    e: &mut T,
    f: &mut T,
    g: &mut T,
    h: &mut T,
) {
    // Stage 1: pairs at stride 1×len
    let t0 = *a + *b;
    let t1 = *a - *b;
    let t2 = *c + *d;
    let t3 = *c - *d;
    let t4 = *e + *f;
    let t5 = *e - *f;
    let t6 = *g + *h;
    let t7 = *g - *h;
    // Stage 2: pairs at stride 2×len
    let u0 = t0 + t2;
    let u1 = t1 + t3;
    let u2 = t0 - t2;
    let u3 = t1 - t3;
    let u4 = t4 + t6;
    let u5 = t5 + t7;
    let u6 = t4 - t6;
    let u7 = t5 - t7;
    // Stage 3: pairs at stride 4×len
    *a = u0 + u4;
    *b = u1 + u5;
    *c = u2 + u6;
    *d = u3 + u7;
    *e = u0 - u4;
    *f = u1 - u5;
    *g = u2 - u6;
    *h = u3 - u7;
}

// ─── Radix-4 FWHT (default) ──────────────────────────────────────────────────

/// Radix-4 FWHT: processes two butterfly stages per pass over the array.
/// Falls back to one radix-2 stage when log₂(n) is odd.
/// Uses `split_at_mut` + `zip` to eliminate all bounds checks.
#[inline]
fn fwht_radix4<T: Copy + Add<Output = T> + Sub<Output = T>>(a: &mut [T]) {
    let n = a.len();
    let mut len = 1usize;

    // Radix-4: two stages per iteration (len advances ×4)
    while len * 4 <= n {
        for block in a.chunks_exact_mut(len * 4) {
            let (ab, cd) = block.split_at_mut(len * 2);
            let (sa, sb) = ab.split_at_mut(len);
            let (sc, sd) = cd.split_at_mut(len);
            for ((a, b), (c, d)) in sa
                .iter_mut()
                .zip(sb.iter_mut())
                .zip(sc.iter_mut().zip(sd.iter_mut()))
            {
                butterfly4(a, b, c, d);
            }
        }
        len *= 4;
    }

    // Radix-2 tail: one remaining stage when log₂(n) is odd
    if len < n {
        for block in a.chunks_exact_mut(len * 2) {
            let (lo, hi) = block.split_at_mut(len);
            for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
                let sum = *u + *v;
                let diff = *u - *v;
                *u = sum;
                *v = diff;
            }
        }
    }
}

// ─── Radix-8 FWHT (x86_64 — deep OOO hides register spills) ─────────────────

/// Radix-8 FWHT: processes three butterfly stages per pass.
/// Falls back to radix-4 and radix-2 for remaining stages.
/// Uses 8-way `split_at_mut` + nested `zip` for bounds-check-free iteration.
#[cfg(any(target_arch = "x86_64", test))]
#[inline]
fn fwht_radix8<T: Copy + Add<Output = T> + Sub<Output = T>>(a: &mut [T]) {
    let n = a.len();
    let mut len = 1usize;

    // Radix-8: three stages per iteration (len advances ×8)
    while len * 8 <= n {
        for block in a.chunks_exact_mut(len * 8) {
            let (half0, half1) = block.split_at_mut(len * 4);
            let (q0, q1) = half0.split_at_mut(len * 2);
            let (q2, q3) = half1.split_at_mut(len * 2);
            let (s0, s1) = q0.split_at_mut(len);
            let (s2, s3) = q1.split_at_mut(len);
            let (s4, s5) = q2.split_at_mut(len);
            let (s6, s7) = q3.split_at_mut(len);
            for (((a, b), (c, d)), ((e, f), (g, h))) in s0
                .iter_mut()
                .zip(s1.iter_mut())
                .zip(s2.iter_mut().zip(s3.iter_mut()))
                .zip(
                    s4.iter_mut()
                        .zip(s5.iter_mut())
                        .zip(s6.iter_mut().zip(s7.iter_mut())),
                )
            {
                butterfly8(a, b, c, d, e, f, g, h);
            }
        }
        len *= 8;
    }

    // Radix-4 tail: up to two remaining stages
    if len * 4 <= n {
        for block in a.chunks_exact_mut(len * 4) {
            let (ab, cd) = block.split_at_mut(len * 2);
            let (sa, sb) = ab.split_at_mut(len);
            let (sc, sd) = cd.split_at_mut(len);
            for ((a, b), (c, d)) in sa
                .iter_mut()
                .zip(sb.iter_mut())
                .zip(sc.iter_mut().zip(sd.iter_mut()))
            {
                butterfly4(a, b, c, d);
            }
        }
        len *= 4;
    }

    // Radix-2 tail: one remaining stage
    if len < n {
        for block in a.chunks_exact_mut(len * 2) {
            let (lo, hi) = block.split_at_mut(len);
            for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
                let sum = *u + *v;
                let diff = *u - *v;
                *u = sum;
                *v = diff;
            }
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Generic Fast Walsh-Hadamard Transform in-place.
///
/// Purely local — only additions and subtractions.
/// Uses radix-8 on x86_64 (deep OOO buffers hide register spills),
/// radix-4 elsewhere. Both fall back to radix-2 for remaining stages.
/// All inner loops use `split_at_mut` + `zip` to eliminate bounds checks.
///
/// Precondition: `a.len()` must be a power of two.
#[inline(always)]
pub fn fwht_in_place<T>(a: &mut [T])
where
    T: Copy + Send + Sync + Add<Output = T> + Sub<Output = T>,
{
    let n = a.len();
    debug_assert!(
        n.is_power_of_two(),
        "FWHT input length must be power-of-two, got {}",
        n
    );
    if n <= 1 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        fwht_radix8(a);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        fwht_radix4(a);
    }
}

/// Fast Walsh-Hadamard Transform in-place on a slice of Rep3 field shares.
///
/// Decomposes into two independent field FWHTs on the `a` and `b` components.
/// This halves the per-FWHT working set (64B→32B elements), improving cache utilization.
///
/// Precondition: `shares.len()` must be a power of two.
#[inline]
#[tracing::instrument(skip_all, name = "fwht_rep3", level = "trace")]
pub fn fwht_rep3_in_place<F: JoltField>(shares: &mut [Rep3PrimeFieldShare<F>]) {
    let n = shares.len();
    let mut a_parts: Vec<F> = Vec::with_capacity(n);
    let mut b_parts: Vec<F> = Vec::with_capacity(n);
    for s in shares.iter() {
        a_parts.push(s.a);
        b_parts.push(s.b);
    }
    fwht_in_place(&mut a_parts);
    fwht_in_place(&mut b_parts);
    for (s, (a, b)) in shares.iter_mut().zip(a_parts.into_iter().zip(b_parts)) {
        s.a = a;
        s.b = b;
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
    fwht_in_place(h_f);

    let inv_m = F::from(m as u64)
        .inverse()
        .expect("M must be invertible in field");

    // Pointwise F × Rep3 → Additive (no communication, parallelized)
    // Fuse inv_m scaling into pointwise multiply to eliminate a full memory pass.
    // promote(f, id) * rep3 = trivial_rep3 * rep3
    // ID0: (f,0)*(a,b) → f*a + f*b = f*(a+b)
    // ID1: (0,f)*(a,b) → f*a
    // ID2: (0,0)*(a,b) → 0
    let mut result: Vec<AdditiveShare<F>> = match party_id {
        PartyID::ID0 => h_f
            .par_iter()
            .zip(ehat.par_iter())
            .map(|(&f, e)| {
                let fi = f * inv_m;
                AdditiveShare::from_fe(fi * (e.a + e.b))
            })
            .collect(),
        PartyID::ID1 => h_f
            .par_iter()
            .zip(ehat.par_iter())
            .map(|(&f, e)| {
                let fi = f * inv_m;
                AdditiveShare::from_fe(fi * e.a)
            })
            .collect(),
        PartyID::ID2 => vec![AdditiveShare::zero(); m],
    };

    fwht_in_place(&mut result);

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
    fwht_in_place(&mut eq_hat);

    let inv_m = F::from(m as u64)
        .inverse()
        .expect("M must be invertible in field");

    // Pointwise multiply: public × share (no communication, parallelized)
    // Fuse inv_m scaling into the multiply to eliminate a full memory pass.
    let mut result: Vec<Rep3PrimeFieldShare<F>> = ehat
        .par_iter()
        .zip(eq_hat.par_iter())
        .map(|(e, &eq)| *e * (eq * inv_m))
        .collect();

    fwht_rep3_in_place(&mut result);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use mpc_core::protocols::rep3::combine_field_element;

    fn share_field<R: rand::Rng>(val: Fr, rng: &mut R) -> [Rep3PrimeFieldShare<Fr>; 3] {
        let shares = mpc_core::protocols::rep3::arithmetic::generate_shares_rep3::<Fr, _>(val, rng);
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
        fwht_in_place(&mut plain_fwht);

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

    /// Reference radix-2 FWHT for correctness testing.
    fn fwht_reference<T: Copy + Add<Output = T> + Sub<Output = T>>(a: &mut [T]) {
        let n = a.len();
        let mut len = 1usize;
        while len < n {
            for block in a.chunks_exact_mut(len * 2) {
                let (lo, hi) = block.split_at_mut(len);
                for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
                    let sum = *u + *v;
                    let diff = *u - *v;
                    *u = sum;
                    *v = diff;
                }
            }
            len *= 2;
        }
    }

    #[test]
    fn radix4_matches_reference() {
        for k in 1..=14 {
            let n = 1usize << k;
            let vals: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64 + 1)).collect();
            let mut ref_out = vals.clone();
            let mut r4_out = vals.clone();
            fwht_reference(&mut ref_out);
            super::fwht_radix4(&mut r4_out);
            assert_eq!(ref_out, r4_out, "radix-4 mismatch at n={n}");
        }
    }

    #[test]
    fn radix8_matches_reference() {
        for k in 1..=14 {
            let n = 1usize << k;
            let vals: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64 + 1)).collect();
            let mut ref_out = vals.clone();
            let mut r8_out = vals.clone();
            fwht_reference(&mut ref_out);
            super::fwht_radix8(&mut r8_out);
            assert_eq!(ref_out, r8_out, "radix-8 mismatch at n={n}");
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
