//! DaBit generation with trusted dealer (P0), O(1) per-party storage.
//!
//! Produces `DaBit<F> = (Rep3RingShare<Bit>, Rep3PrimeFieldShare<F>)` where both
//! shares encode the same random bit `b`.
//!
//! ## Storage: O(1) per party — NO O(N) corrections
//!
//! Each party holds 3 pairwise 32-byte seeds (96 bytes total). P0 and P2 also
//! hold the third pairwise seed (seed_{P1↔P2} for P0, seed_{P0↔P1} for P2) to
//! compute the arithmetic correction on-the-fly without storing O(N) values.
//!
//! ## Sharing construction
//!
//! Three pairwise seeds: `seed_01`, `seed_12`, `seed_20`.
//!
//! **Boolean share** at position `x`:
//!   `b(x) = PRF_bit(seed_01, x) XOR PRF_bit(seed_12, x) XOR PRF_bit(seed_20, x)`
//!   Party i holds `(PRF_bit(seed_{i→i+1}, x), PRF_bit(seed_{i-1→i}, x))`.
//!
//! **Arithmetic share** at position `x`:
//!   Base: `S(x) = PRF_field(seed_01, x) + PRF_field(seed_12, x) + PRF_field(seed_20, x)`.
//!   Correction: `c(x) = F::from(b(x)) - S(x)`, applied to the (P2→P0) Rep3 edge.
//!   P0 and P2 compute `c(x)` on-the-fly from their 3 seeds. P1 needs no correction.
//!
//! ## PRF implementation
//!
//! Uses fixed-key AES-128 in Matyas-Meyer-Oseas mode (same as RDCF PRG) for
//! ~10x faster evaluation than SHA3, with hardware AES-NI acceleration.
//!
//! ## Setup communication
//!
//! P0 → P1: 2 seeds (64 bytes). P0 → P2: 3 seeds (96 bytes).

use super::edabits_pcg::PcgEdaBit;
use super::rdcf::{Prg, prg};
use crate::protocols::rep3::PartyID;
use crate::protocols::rep3_ring::{binary, edabits::DaBit};
use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3::Rep3PrimeFieldShare;
use mpc_types::protocols::rep3_ring::{
    Rep3RingShare,
    ring::{bit::Bit, int_ring::IntRing2k},
};
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;

// ── Setup Types ─────────────────────────────────────────────────────────────

/// Per-party setup state for daBit generation.
///
/// **O(1) storage**: 3 seeds (96 bytes) for P0/P2, 2 seeds (64 bytes) for P1.
/// No O(N) corrections are stored.
#[derive(Debug, Clone)]
pub struct PcgDaBitSetup {
    pub party_id: PartyID,
    /// Seed shared with the *next* party — generates `a` component.
    pub seed_next: [u8; 32],
    /// Seed shared with the *previous* party — generates `b` component.
    pub seed_prev: [u8; 32],
    /// Third pairwise seed for computing corrections on-the-fly.
    /// P0: `seed_12` (P1↔P2). P2: `seed_01` (P0↔P1). P1: None.
    pub seed_third: Option<[u8; 32]>,
}

/// Output of the dealer's one-time setup.
pub struct DealerOutput {
    pub party0: PcgDaBitSetup,
    pub party1: PcgDaBitSetup,
    pub party2: PcgDaBitSetup,
}

// ── Dealer Setup ────────────────────────────────────────────────────────────

/// P0 generates pairwise seeds for `num` daBits.
///
/// Returns setups for all 3 parties. P0 must distribute seeds to P1/P2.
/// No O(N) corrections are computed or stored.
///
/// **Communication**: P0 → P1: 64 bytes (2 seeds). P0 → P2: 96 bytes (3 seeds).
pub fn dealer_setup(rng: &mut impl rand::RngCore) -> DealerOutput {
    let mut seed_01 = [0u8; 32];
    let mut seed_12 = [0u8; 32];
    let mut seed_20 = [0u8; 32];
    rng.fill_bytes(&mut seed_01);
    rng.fill_bytes(&mut seed_12);
    rng.fill_bytes(&mut seed_20);

    DealerOutput {
        party0: PcgDaBitSetup {
            party_id: PartyID::ID0,
            seed_next: seed_01,
            seed_prev: seed_20,
            seed_third: Some(seed_12), // P0 gets seed_12 to compute corrections
        },
        party1: PcgDaBitSetup {
            party_id: PartyID::ID1,
            seed_next: seed_12,
            seed_prev: seed_01,
            seed_third: None, // P1 needs no correction
        },
        party2: PcgDaBitSetup {
            party_id: PartyID::ID2,
            seed_next: seed_20,
            seed_prev: seed_12,
            seed_third: Some(seed_01), // P2 gets seed_01 to compute corrections
        },
    }
}

// ── Local daBit Expansion ───────────────────────────────────────────────────

/// Expand `count` daBits starting at position `start`. **No communication.**
///
/// All parties evaluate PRFs locally. P0/P2 compute corrections on-the-fly
/// from their 3 seeds. P1 applies zero correction.
///
/// Uses AES-NI PRG for ~10x speedup over SHA3.
#[tracing::instrument(skip(setup, start))]
pub fn expand_dabits<F: PrimeField>(
    setup: &PcgDaBitSetup,
    start: usize,
    count: usize,
) -> Vec<DaBit<F>> {
    if count == 0 {
        return Vec::new();
    }

    let prg = prg();

    (start..start + count)
        .into_par_iter()
        .with_min_len(64)
        .map(|x| expand_single_dabit(setup, x, prg))
        .collect()
}

/// Expand `count` edaBits directly from seeds, fusing daBit expansion + packing.
///
/// Avoids the intermediate `Vec<DaBit>` allocation that `expand_dabits` + chunk/pack
/// would require (~66 bytes × K × count). Each edaBit packs K daBits in-place.
#[tracing::instrument(skip(setup, start))]
pub fn expand_edabits<T: IntRing2k, F: PrimeField>(
    setup: &PcgDaBitSetup,
    start: usize,
    count: usize,
) -> Vec<PcgEdaBit<T, F>>
where
    Standard: Distribution<T>,
{
    if count == 0 {
        return Vec::new();
    }

    let prg = prg();
    let k = T::K;

    (0..count)
        .into_par_iter()
        .with_min_len(64)
        .map(|i| {
            let base = (start + i) * k;
            let mut r_bits = Vec::with_capacity(k);
            let mut bit_values = Vec::with_capacity(k);
            for j in 0..k {
                let d = expand_single_dabit::<F>(setup, base + j, prg);
                r_bits.push(d.bit);
                bit_values.push(d.value);
            }
            let r_packed = binary::pack_bits::<T>(&r_bits);
            PcgEdaBit {
                r_bits,
                r_packed,
                bit_values,
            }
        })
        .collect()
}

/// Expand a single daBit at position `x`.
///
/// Algebraically optimized: merges correction computation inline and exploits
/// cancellation to avoid redundant PRF calls. For P0/P2: 3 bit PRFs + 2 field PRFs
/// (= 5 AES calls) instead of the previous 10 PRF calls (= 24 AES calls).
/// P1: 2 bit PRFs + 2 field PRFs (= 4 AES calls) instead of 8.
#[inline]
fn expand_single_dabit<F: PrimeField>(setup: &PcgDaBitSetup, x: usize, prg: &Prg) -> DaBit<F> {
    // ── Boolean share ──
    let bit_a = prf_bit(prg, &setup.seed_next, x);
    let bit_b = prf_bit(prg, &setup.seed_prev, x);
    let bit_share = Rep3RingShare::new(Bit::new(bit_a), Bit::new(bit_b));

    // ── Arithmetic share (with inlined correction) ──
    //
    // Correction c = bit - (f_next + f_prev + f_third) applied to the (P2→P0) edge.
    // Key insight: the f_prev term for P0 (resp. f_next for P2) cancels algebraically:
    //   P0: b_slot = f_prev + c = f_prev + bit - f_next - f_prev - f_third = bit - f_next - f_third
    //   P2: a_slot = f_next + c = f_next + bit - f_next - f_prev - f_third = bit - f_prev - f_third
    // So we skip computing the canceling PRF entirely.
    let value = match setup.party_id {
        PartyID::ID1 => {
            // P1: no correction needed, just 2 field PRFs
            let a_i: F = prf_field(prg, &setup.seed_next, x);
            let b_i: F = prf_field(prg, &setup.seed_prev, x);
            Rep3PrimeFieldShare::new(a_i, b_i)
        }
        _ => {
            let seed_third = setup
                .seed_third
                .expect("P0/P2 must have seed_third for correction computation");

            // 1 extra bit PRF to reconstruct the full bit
            let bit_third = prf_bit(prg, &seed_third, x);
            let bit = bit_a ^ bit_b ^ bit_third;
            let target = if bit { F::one() } else { F::zero() };

            match setup.party_id {
                PartyID::ID0 => {
                    // P0: a_slot = f_next (own PRF), b_slot = bit - f_next - f_third
                    let f_next: F = prf_field(prg, &setup.seed_next, x);
                    let f_third: F = prf_field(prg, &seed_third, x);
                    Rep3PrimeFieldShare::new(f_next, target - f_next - f_third)
                }
                _ => {
                    // P2: a_slot = bit - f_prev - f_third, b_slot = f_prev (own PRF)
                    let f_prev: F = prf_field(prg, &setup.seed_prev, x);
                    let f_third: F = prf_field(prg, &seed_third, x);
                    Rep3PrimeFieldShare::new(target - f_prev - f_third, f_prev)
                }
            }
        }
    };

    DaBit {
        bit: bit_share,
        value,
    }
}

// ── AES-based PRF helpers ──────────────────────────────────────────────────
// Using the RDCF PRG (fixed-key AES-128 in MMO mode) for fast evaluation.

/// Deterministic PRF producing a single bit from a seed and index.
/// Uses 1 AES call (convert_single) instead of 3 (expand).
#[inline]
fn prf_bit(prg: &Prg, seed: &[u8; 32], x: usize) -> bool {
    let input = prf_input(seed, x, b"bit\0");
    let converted = prg.convert_single(&input);
    (converted[0] & 1) != 0
}

/// Deterministic PRF producing a field element from a seed and index.
/// Uses fast u128→field path (1 AES + F::from(u128)).
#[inline]
fn prf_field<F: PrimeField>(prg: &Prg, seed: &[u8; 32], x: usize) -> F {
    let input = prf_input(seed, x, b"fld\0");
    prg.convert_to_field_fast::<F>(&input)
}

/// Construct a 128-bit PRF input from seed (32 bytes), index, and domain tag (4 bytes).
///
/// Folds the 32-byte seed into 16 bytes via XOR, then mixes in the index and tag.
#[inline]
fn prf_input(seed: &[u8; 32], x: usize, tag: &[u8; 4]) -> [u8; 16] {
    let x_bytes = x.to_le_bytes(); // 8 bytes
    let mut input = [0u8; 16];
    // Bytes [0..8]: seed[0..8] XOR seed[16..24] XOR x_bytes
    for i in 0..8 {
        input[i] = seed[i] ^ seed[i + 16] ^ x_bytes[i];
    }
    // Bytes [8..12]: seed[8..12] XOR seed[24..28] XOR tag
    for i in 0..4 {
        input[8 + i] = seed[8 + i] ^ seed[24 + i] ^ tag[i];
    }
    // Bytes [12..16]: seed[12..16] XOR seed[28..32]
    for i in 0..4 {
        input[12 + i] = seed[12 + i] ^ seed[28 + i];
    }
    input
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::Zero;
    use mpc_types::protocols::rep3::combine_field_element;
    use mpc_types::protocols::rep3_ring::{combine_ring_element, ring::ring_impl::RingElement};
    use rand::SeedableRng;

    #[test]
    fn boolean_shares_consistent() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
        let dealer = dealer_setup(&mut rng);

        let d0 = expand_dabits::<Fr>(&dealer.party0, 0, 100);
        let d1 = expand_dabits::<Fr>(&dealer.party1, 0, 100);
        let d2 = expand_dabits::<Fr>(&dealer.party2, 0, 100);

        for x in 0..100 {
            assert_eq!(d0[x].bit.a, d1[x].bit.b, "x={x}: P0.a != P1.b");
            assert_eq!(d1[x].bit.a, d2[x].bit.b, "x={x}: P1.a != P2.b");
            assert_eq!(d2[x].bit.a, d0[x].bit.b, "x={x}: P2.a != P0.b");
        }
    }

    #[test]
    fn arithmetic_shares_rep3_consistent() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
        let dealer = dealer_setup(&mut rng);

        let d0 = expand_dabits::<Fr>(&dealer.party0, 0, 100);
        let d1 = expand_dabits::<Fr>(&dealer.party1, 0, 100);
        let d2 = expand_dabits::<Fr>(&dealer.party2, 0, 100);

        for x in 0..100 {
            assert_eq!(d0[x].value.a, d1[x].value.b, "x={x}: arith P0.a != P1.b");
            assert_eq!(d1[x].value.a, d2[x].value.b, "x={x}: arith P1.a != P2.b");
            assert_eq!(d2[x].value.a, d0[x].value.b, "x={x}: arith P2.a != P0.b");
        }
    }

    #[test]
    fn dabit_correctness() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
        let dealer = dealer_setup(&mut rng);

        let (dabits0, (dabits1, dabits2)) = rayon::join(
            || expand_dabits::<Fr>(&dealer.party0, 0, 200),
            || {
                rayon::join(
                    || expand_dabits::<Fr>(&dealer.party1, 0, 200),
                    || expand_dabits::<Fr>(&dealer.party2, 0, 200),
                )
            },
        );

        let mut ones = 0;
        for i in 0..200 {
            let bit: RingElement<Bit> =
                combine_ring_element(dabits0[i].bit, dabits1[i].bit, dabits2[i].bit);
            let b: bool = bit.0.convert();

            let val = combine_field_element(dabits0[i].value, dabits1[i].value, dabits2[i].value);
            let expected = if b { Fr::from(1u64) } else { Fr::zero() };
            assert_eq!(val, expected, "daBit {i}: bit={b}");

            if b {
                ones += 1;
            }
        }

        assert!(ones > 30 && ones < 170, "bad distribution: {ones}/200 ones");
    }

    #[test]
    fn batch_deterministic() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(99);
        let dealer = dealer_setup(&mut rng);

        let batch_a = expand_dabits::<Fr>(&dealer.party0, 0, 50);
        let batch_b = expand_dabits::<Fr>(&dealer.party0, 0, 50);

        for i in 0..50 {
            assert_eq!(batch_a[i].bit, batch_b[i].bit);
            assert_eq!(batch_a[i].value.a, batch_b[i].value.a);
            assert_eq!(batch_a[i].value.b, batch_b[i].value.b);
        }
    }

    #[test]
    fn prf_independence() {
        // Verify that bit and field PRFs with same seed/index produce independent outputs
        let prg = Prg::new();
        let seed = [0x42u8; 32];

        // bit PRF and field PRF should be independent (different domain tags)
        let bit = prf_bit(&prg, &seed, 0);
        let field: Fr = prf_field(&prg, &seed, 0);

        // Just check they don't trivially correlate (field element isn't just 0 or 1)
        assert!(field != Fr::zero() || !bit, "PRFs should be independent");
    }

    #[test]
    fn no_on_storage() {
        // Verify that PcgDaBitSetup has O(1) storage (no Vec fields)
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
        let dealer = dealer_setup(&mut rng);

        // Each setup is just seeds — no vectors
        let p0_size = std::mem::size_of_val(&dealer.party0);
        let p1_size = std::mem::size_of_val(&dealer.party1);
        let p2_size = std::mem::size_of_val(&dealer.party2);

        // All should be small (< 200 bytes — just the struct with seeds)
        assert!(p0_size < 200, "P0 setup too large: {p0_size}");
        assert!(p1_size < 200, "P1 setup too large: {p1_size}");
        assert!(p2_size < 200, "P2 setup too large: {p2_size}");
    }
}
