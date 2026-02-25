//! edaBits helpers for Rep3 over rings.
//!
//! This module provides an opt-in conversion primitive to translate an
//! arithmetic Rep3 sharing over `Z_{2^K}` into an arithmetic Rep3 sharing over a
//! prime field `Fp`, using an edaBits mask that links the same random `r` across
//! both domains.

use crate::IoResult;
use crate::protocols::rep3::{
    PartyID, Rep3PrimeFieldShare, arithmetic as rep3_arith,
    network::{IoContext, Rep3Network},
};
use crate::protocols::rep3_ring::arithmetic as rep3_ring_arith;

use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3_ring::{
    Rep3RingShare,
    ring::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement},
};
use ark_ff::One as _;
use num_bigint::BigUint;
use rand::distributions::Standard;
use rand::RngCore;
use rand::prelude::Distribution;

/// An edaBits mask linking the same random `r` across:
/// - `r_ring`: arithmetic sharing over `Z_{2^K}`
/// - `r_fp`: arithmetic sharing over `Fp` (embedding of the same integer `r`)
///
/// `r_bits` is optional and unused by [`b2a_ring_to_field_masked`]; it is kept
/// for completeness and future mixed-circuit protocols.
#[derive(Debug, Clone)]
pub struct EdaBits<T: IntRing2k, F: PrimeField> {
    pub r_ring: Rep3RingShare<T>,
    pub r_fp: Rep3PrimeFieldShare<F>,
    pub r_bits: Option<Vec<Rep3RingShare<Bit>>>,
}

/// A doubly-authenticated bit (daBit) where the binary share is represented
/// directly as a Rep3 share over `Z2`.
///
/// This is more communication/storage-efficient than representing a single bit
/// as a [`num_bigint::BigUint`] inside [`mpc_types::protocols::rep3::Rep3BigUintShare`].
#[derive(Debug, Clone, Copy)]
pub struct DaBit<F: PrimeField> {
    pub bit: Rep3RingShare<Bit>,
    pub value: Rep3PrimeFieldShare<F>,
}

/// Generate `num` *trivially shared* daBits for tests, using `Rep3RingShare<Bit>`
/// as the binary representation.
///
/// **Important:** to obtain *consistent* daBits across parties, each party must
/// call this function with the same RNG seed and the same `num`.
pub fn trivial_dabits<F: PrimeField>(
    num: usize,
    party_id: PartyID,
    rng: &mut impl RngCore,
) -> Vec<DaBit<F>> {
    (0..num)
        .map(|_| {
            let r = (rng.next_u32() & 1) != 0;
            let bit = rep3_ring_arith::promote_to_trivial_share(
                party_id,
                RingElement(Bit::new(r)),
            );
            let value = rep3_arith::promote_to_trivial_share(party_id, F::from(r as u64));
            DaBit { bit, value }
        })
        .collect()
}

/// Generate `num` *trivially shared* edaBits masks for tests.
///
/// Each edaBits represents a random public integer `r ∈ [0,2^K)`:
/// - `r_ring` is a trivial Rep3 arithmetic share of `r mod 2^K`
/// - `r_fp` is a trivial Rep3 arithmetic share of the same integer embedded in `Fp`
///
/// `r_bits` is set to `None`.
///
/// **Important:** to obtain *consistent* edaBits across parties, each party must
/// call this function with the same RNG seed and the same `num`.
pub fn trivial_edabits<T: IntRing2k, F: PrimeField>(
    num: usize,
    party_id: PartyID,
    rng: &mut impl RngCore,
) -> Vec<EdaBits<T, F>> {
    // Mask for keeping exactly K bits.
    let mask = if T::K == 0 {
        BigUint::ZERO
    } else {
        (BigUint::one() << T::K) - BigUint::one()
    };

    (0..num)
        .map(|_| {
            let mut bytes = vec![0u8; T::BYTES.max(1)];
            rng.fill_bytes(&mut bytes);
            let mut r_big = BigUint::from_bytes_le(&bytes);
            r_big &= &mask;

            let r_ring_val = T::cast_from_biguint(&r_big);
            let r_ring = rep3_ring_arith::promote_to_trivial_share(
                party_id,
                RingElement(r_ring_val),
            );

            let r_f = F::from_be_bytes_mod_order(&r_big.to_bytes_be());
            let r_fp = rep3_arith::promote_to_trivial_share(party_id, r_f);

            EdaBits {
                r_ring,
                r_fp,
                r_bits: None,
            }
        })
        .collect()
}

/// Generate `num` random daBits using Rep3 preprocessing.
///
/// Produces random bits `r ∈ {0,1}` shared as:
/// - `bit`: Rep3 sharing over `Z2` (`Rep3RingShare<Bit>`)
/// - `value`: arithmetic Rep3 sharing of the same bit in `Fp`
///
/// `rng` is not used for secrecy; secrecy comes from correlated RNG inside `io`.
/// Callers must invoke this function in the same order on all parties to keep
/// `io` RNG streams aligned.
pub fn random_dabits<F: PrimeField, N: Rep3Network>(
    num: usize,
    _rng: &mut impl RngCore,
    io: &mut IoContext<N>,
) -> IoResult<Vec<DaBit<F>>> {
    let bits = (0..num)
        .map(|_| rep3_ring_arith::rand::<Bit, _>(io))
        .collect::<Vec<_>>();

    let values =
        crate::protocols::rep3_ring::conversion::bit_inject_from_bits_to_field_many::<F, _>(
            &bits, io,
        )?;

    Ok(bits
        .into_iter()
        .zip(values)
        .map(|(bit, value)| DaBit { bit, value })
        .collect())
}

/// Generate `num` random edaBits using Rep3 preprocessing.
///
/// Produces random `r ∈ [0,2^K)` (where `K = T::K`) linked across:
/// - `r_bits`: per-bit Rep3 sharing (little-endian)
/// - `r_ring`: arithmetic Rep3 sharing over `Z_{2^K}`
/// - `r_fp`: arithmetic Rep3 sharing in `Fp` (embedding of the same integer)
///
/// `rng` is not used for secrecy; secrecy comes from correlated RNG inside `io`.
/// Callers must invoke this function in the same order on all parties to keep
/// `io` RNG streams aligned.
pub fn random_edabits<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    num: usize,
    _rng: &mut impl RngCore,
    io: &mut IoContext<N>,
) -> IoResult<Vec<EdaBits<T, F>>>
where
    Standard: Distribution<T>,
{
    let r_bits = (0..num)
        .map(|_| {
            (0..T::K)
                .map(|_| rep3_ring_arith::rand::<Bit, _>(io))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let r_bin = r_bits
        .iter()
        .map(|bits| crate::protocols::rep3_ring::binary::pack_bits::<T>(bits))
        .collect::<Vec<_>>();

    let r_ring = crate::protocols::rep3_ring::conversion::b2a_many::<T, _>(&r_bin, io)?;
    let r_fp =
        crate::protocols::rep3_ring::casts::binary_ring_to_field_many::<T, F, _>(&r_bin, io)?;

    Ok((0..num)
        .map(|i| EdaBits {
            r_ring: r_ring[i],
            r_fp: r_fp[i],
            r_bits: Some(r_bits[i].clone()),
        })
        .collect())
}

/// Convert an arithmetic ring share `[x]` over `Z_{2^K}` into an arithmetic
/// field share `[x]` over `Fp`, using a masked opening.
///
/// Protocol:
/// 1) Compute `[c] = [x] - [r]` in `Z_{2^K}`.
/// 2) Open `c` to a public ring element (interpreted as an integer in `[0,2^K)`).
/// 3) Output `[x]_{Fp} = [r]_{Fp} + c`.
///
/// # Assumptions
/// Intended when `x` represents a non-negative integer smaller than `2^K`, `Fp`
/// is large enough for the application (typical SNARK fields), and the opened
/// value `c` corresponds to the *integer* difference `x - r` (i.e., no wrap
/// around modulo `2^K`).
pub fn binary_ring_to_field<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x: Rep3RingShare<T>,
    eda: EdaBits<T, F>,
    io: &mut IoContext<N>,
) -> IoResult<Rep3PrimeFieldShare<F>> {
    let c_share = x - eda.r_ring;
    let c_open: RingElement<T> = rep3_ring_arith::open(c_share, io)?;

    let c_big = c_open.0.cast_to_biguint();
    let c_f = F::from_be_bytes_mod_order(&c_big.to_bytes_be());
    let c_fp_share = rep3_arith::promote_to_trivial_share(io.id, c_f);

    Ok(eda.r_fp + c_fp_share)
}

pub fn binary_ring_to_field_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x: &[Rep3RingShare<T>],
    eda: &[EdaBits<T, F>],
    io: &mut IoContext<N>,
) -> IoResult<Vec<Rep3PrimeFieldShare<F>>> {
    debug_assert_eq!(x.len(), eda.len());

    let masked = x
        .iter()
        .zip(eda)
        .map(|(x, eda)| *x - eda.r_ring)
        .collect::<Vec<_>>();

    let opened = rep3_ring_arith::open_vec(&masked, io)?;

    Ok(opened
        .into_iter()
        .zip(eda)
        .map(|(c_open, eda)| {
            let c_big = c_open.0.cast_to_biguint();
            let c_f = F::from_be_bytes_mod_order(&c_big.to_bytes_be());
            let c_fp_share = rep3_arith::promote_to_trivial_share(io.id, c_f);
            eda.r_fp + c_fp_share
        })
        .collect())
}

pub fn bit_inject_field<F: PrimeField, N: Rep3Network>(
    x: Rep3RingShare<Bit>,
    da: DaBit<F>,
    io: &mut IoContext<N>,
) -> IoResult<Rep3PrimeFieldShare<F>> {
    let c_share = x ^ da.bit;
    let c_open = rep3_ring_arith::open_bit(c_share, io)?;
    let c = c_open.0.convert();

    if !c {
        Ok(da.value)
    } else {
        Ok(rep3_arith::sub_public_by_shared(F::one(), da.value, io.id))
    }
}

pub fn bit_inject_field_many<F: PrimeField, N: Rep3Network>(
    x: &[Rep3RingShare<Bit>],
    da: &[DaBit<F>],
    io: &mut IoContext<N>,
) -> IoResult<Vec<Rep3PrimeFieldShare<F>>> {
    debug_assert_eq!(x.len(), da.len());

    let c_shares = x
        .iter()
        .zip(da)
        .map(|(x, da)| *x ^ da.bit)
        .collect::<Vec<_>>();

    let opened =
        crate::protocols::rep3_ring::binary::open_vec::<Bit, _>(c_shares, io)?;

    opened
        .into_iter()
        .zip(da)
        .map(|(c, da)| {
            debug_assert!(c <= 1);
            if c == 0 {
                Ok(da.value)
            } else {
                Ok(rep3_arith::sub_public_by_shared(F::one(), da.value, io.id))
            }
        })
        .collect()
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;

    use ark_bn254::Fr;
    use mpc_types::protocols::rep3::{
        combine_field_element, combine_field_elements, share_field_element,
    };
    use mpc_types::protocols::rep3_ring::{
        combine_ring_element, combine_ring_element_binary, share_ring_element, ring::bit::Bit as RingBit,
    };
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use crate::protocols::rep3::test_utils::run_rep3_local_test_with_coordinator;

    #[test]
    fn b2a_ring_to_field_masked_recovers_x() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xBADA55);
        let x_u32 = rng.next_u32();
        let r_u32 = rng.next_u32();
        let x_u64 = x_u32 as u64;
        let r_u64 = r_u32 as u64;

        let x_ring_shares = share_ring_element::<u64, _>(RingElement(x_u64), &mut rng);
        let r_ring_shares = share_ring_element::<u64, _>(RingElement(r_u64), &mut rng);

        let r_fp = Fr::from(r_u64);
        let r_fp_shares = share_field_element::<Fr, _>(r_fp, &mut rng);

        let edas: [EdaBits<u64, Fr>; 3] = std::array::from_fn(|i| EdaBits {
            r_ring: r_ring_shares[i],
            r_fp: r_fp_shares[i],
            r_bits: None,
        });

        let outputs: [Rep3PrimeFieldShare<Fr>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| (x_ring_shares[i], edas[i].clone()),
            || (),
            |(x_share, eda), mut io_ctx| {
                let io = io_ctx.main();
                binary_ring_to_field::<u64, Fr, _>(x_share, eda, io).map_err(Into::into)
            },
            |(), _net| Ok(()),
        )
        .0;

        let opened = combine_field_element(outputs[0], outputs[1], outputs[2]);
        let expected = Fr::from(x_u64);
        assert_eq!(opened, expected);
    }

    #[test]
    fn b2a_ring_to_field_masked_many_matches_single() {
        const NVALS: usize = 8;
        let mut rng = ChaCha20Rng::seed_from_u64(0x1234_5678);

        // Pick masks r <= x (as integers) so that `c = x - r` does not wrap mod 2^64.
        let xs_u64 = (0..NVALS).map(|_| rng.next_u64()).collect::<Vec<_>>();
        let rs_u64 = xs_u64
            .iter()
            .map(|&x| x >> 1)
            .collect::<Vec<_>>();

        let x_shares_per_val = xs_u64
            .iter()
            .map(|&x| share_ring_element::<u64, _>(RingElement(x), &mut rng))
            .collect::<Vec<_>>();
        let r_ring_shares_per_val = rs_u64
            .iter()
            .map(|&r| share_ring_element::<u64, _>(RingElement(r), &mut rng))
            .collect::<Vec<_>>();
        let r_fp_shares_per_val = rs_u64
            .iter()
            .map(|&r| share_field_element::<Fr, _>(Fr::from(r), &mut rng))
            .collect::<Vec<_>>();

        let x_ring_shares: [Vec<Rep3RingShare<u64>>; 3] =
            std::array::from_fn(|pid| x_shares_per_val.iter().map(|s| s[pid]).collect());
        let eda_shares: [Vec<EdaBits<u64, Fr>>; 3] = std::array::from_fn(|pid| {
            (0..NVALS)
                .map(|i| EdaBits {
                    r_ring: r_ring_shares_per_val[i][pid],
                    r_fp: r_fp_shares_per_val[i][pid],
                    r_bits: None,
                })
                .collect()
        });

        let outputs: [Vec<Rep3PrimeFieldShare<Fr>>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| (x_ring_shares[i].clone(), eda_shares[i].clone()),
            || (),
            |(x_shares, edas), mut io_ctx| {
                let io = io_ctx.main();
                binary_ring_to_field_many::<u64, Fr, _>(&x_shares, &edas, io).map_err(Into::into)
            },
            |(), _net| Ok(()),
        )
        .0;

        let combined = combine_field_elements(&outputs[0], &outputs[1], &outputs[2]);
        let expected = xs_u64.into_iter().map(Fr::from).collect::<Vec<_>>();
        assert_eq!(combined, expected);
    }

    #[test]
    fn bit_inject_field_many_roundtrip() {
        const NBITS: usize = 16;
        let mut rng = ChaCha20Rng::seed_from_u64(0xDAB1_0001);
        let bits = (0..NBITS)
            .map(|_| (rng.next_u32() & 1) == 1)
            .collect::<Vec<_>>();

        let per_bit_shares = bits
            .iter()
            .map(|&b| share_ring_element::<RingBit, _>(RingElement(RingBit::new(b)), &mut rng))
            .collect::<Vec<_>>();
        let x_bit_shares: [Vec<Rep3RingShare<RingBit>>; 3] =
            std::array::from_fn(|pid| per_bit_shares.iter().map(|s| s[pid]).collect());

        let outputs: [Vec<Rep3PrimeFieldShare<Fr>>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| x_bit_shares[i].clone(),
            || (),
            |x_shares, mut io_ctx| {
                let io = io_ctx.main();
                let mut local_rng = ChaCha20Rng::seed_from_u64(0xDAB1_0002);
                let dabits = trivial_dabits::<Fr>(x_shares.len(), io.id, &mut local_rng);
                bit_inject_field_many::<Fr, _>(&x_shares, &dabits, io).map_err(Into::into)
            },
            |(), _net| Ok(()),
        )
        .0;

        let combined = combine_field_elements(&outputs[0], &outputs[1], &outputs[2]);
        let expected = bits.into_iter().map(|b| Fr::from(b as u64)).collect::<Vec<_>>();
        assert_eq!(combined, expected);
    }

    #[test]
    fn trivial_dabits_are_consistent() {
        let outputs: [(Rep3RingShare<RingBit>, Rep3PrimeFieldShare<Fr>); 3] =
            run_rep3_local_test_with_coordinator(
                1,
                |i| i,
                || (),
                |party_idx, mut io_ctx| {
                    let io = io_ctx.main();
                    assert_eq!(usize::from(io.id), party_idx);
                    let mut rng = ChaCha20Rng::seed_from_u64(0xDAB1_1001);
                    let da = trivial_dabits::<Fr>(1, io.id, &mut rng)
                        .into_iter()
                        .next()
                        .unwrap();
                    Ok((da.bit, da.value))
                },
                |(), _net| Ok(()),
            )
            .0;

        let r_bit = combine_ring_element(outputs[0].0, outputs[1].0, outputs[2].0);
        let r_fp = combine_field_element(outputs[0].1, outputs[1].1, outputs[2].1);
        assert_eq!(r_fp, Fr::from(r_bit.0.convert() as u64));
    }

    #[test]
    fn trivial_edabits_are_consistent() {
        let outputs: [(Rep3RingShare<u64>, Rep3PrimeFieldShare<Fr>); 3] =
            run_rep3_local_test_with_coordinator(
                1,
                |i| i,
                || (),
                |party_idx, mut io_ctx| {
                    let io = io_ctx.main();
                    assert_eq!(usize::from(io.id), party_idx);
                    let mut rng = ChaCha20Rng::seed_from_u64(0xEDA_0001);
                    let eda = trivial_edabits::<u64, Fr>(1, io.id, &mut rng)
                        .into_iter()
                        .next()
                        .unwrap();
                    Ok((eda.r_ring, eda.r_fp))
                },
                |(), _net| Ok(()),
            )
            .0;

        let r_ring = combine_ring_element(outputs[0].0, outputs[1].0, outputs[2].0);
        let r_fp = combine_field_element(outputs[0].1, outputs[1].1, outputs[2].1);
        assert_eq!(r_fp, Fr::from(r_ring.0));
    }

    #[test]
    fn random_dabits_consistent() {
        const NUM: usize = 32;
        let outputs: [Vec<DaBit<Fr>>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| i,
            || (),
            |party_idx, mut io_ctx| {
                let io = io_ctx.main();
                assert_eq!(usize::from(io.id), party_idx);
                let mut rng = ChaCha20Rng::seed_from_u64(0xDAB1_2001);
                random_dabits::<Fr, _>(NUM, &mut rng, io).map_err(Into::into)
            },
            |(), _net| Ok(()),
        )
        .0;

        for i in 0..NUM {
            let r_bit = combine_ring_element_binary(
                outputs[0][i].bit,
                outputs[1][i].bit,
                outputs[2][i].bit,
            );
            let r_fp = combine_field_element(
                outputs[0][i].value,
                outputs[1][i].value,
                outputs[2][i].value,
            );
            assert_eq!(r_fp, Fr::from(r_bit.0.convert() as u64));
        }
    }

    #[test]
    fn random_edabits_consistent() {
        const NUM: usize = 8;
        let outputs: [Vec<EdaBits<u64, Fr>>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| i,
            || (),
            |party_idx, mut io_ctx| {
                let io = io_ctx.main();
                assert_eq!(usize::from(io.id), party_idx);
                let mut rng = ChaCha20Rng::seed_from_u64(0xEDA_2001);
                random_edabits::<u64, Fr, _>(NUM, &mut rng, io).map_err(Into::into)
            },
            |(), _net| Ok(()),
        )
        .0;

        for i in 0..NUM {
            let r_ring = combine_ring_element(
                outputs[0][i].r_ring,
                outputs[1][i].r_ring,
                outputs[2][i].r_ring,
            );
            let r_fp = combine_field_element(
                outputs[0][i].r_fp,
                outputs[1][i].r_fp,
                outputs[2][i].r_fp,
            );

            let bits0 = outputs[0][i].r_bits.as_ref().unwrap();
            let bits1 = outputs[1][i].r_bits.as_ref().unwrap();
            let bits2 = outputs[2][i].r_bits.as_ref().unwrap();
            assert_eq!(bits0.len(), u64::K);

            let mut reconstructed = 0u64;
            for b in 0..u64::K {
                let bit = combine_ring_element_binary(bits0[b], bits1[b], bits2[b]);
                if bit.0.convert() {
                    reconstructed |= 1u64 << b;
                }
            }

            assert_eq!(r_ring.0, reconstructed);
            assert_eq!(r_fp, Fr::from(r_ring.0));
        }
    }
}
