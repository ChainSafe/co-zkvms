//! Standard daBit generation with trusted dealer (P0).
//!
//! Produces `DaBit<F> = (Rep3RingShare<Bit>, Rep3PrimeFieldShare<F>)` where both
//! shares encode the same random bit `b`. Boolean shares are Rep3-consistent by
//! construction (pairwise PRFs). Arithmetic shares are Rep3-consistent because
//! P0 distributes corrections to P2 during a one-time preprocessing step.
//!
//! **Setup communication**: P0 → P2: one field element per daBit.
//! **Online expansion**: zero communication — each party evaluates PRFs locally.
//!
//! ## Sharing construction
//!
//! Each party holds pairwise seeds `(seed_next, seed_prev)`. For position `x`:
//!
//! - **Boolean**: `a = PRF_bool(seed_next, x)`, `b = PRF_bool(seed_prev, x)`.
//!   Secret bit `b(x) = XOR of all 3 pairwise PRF bits`.
//!
//! - **Arithmetic**: `a = PRF_arith(seed_next, x)`, `b = PRF_arith(seed_prev, x)`.
//!   Gives Rep3-consistent shares with secret `S(x) = sum of 3 PRFs`.
//!   P0 computes correction `c(x) = F::from(b(x)) - S(x)` and sends it to P2.
//!   P2 adds `c(x)` to its `a` component; P0 adds `c(x)` to its `b` component.
//!   (Both point to the same Rep3 slot `s_2`, maintaining consistency.)

use crate::protocols::rep3::PartyID;
use crate::protocols::rep3_ring::edabits::DaBit;
use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3::Rep3PrimeFieldShare;
use mpc_types::protocols::rep3_ring::{Rep3RingShare, ring::bit::Bit};
use rayon::prelude::*;
use sha3::Digest;

// ── Setup Types ─────────────────────────────────────────────────────────────

/// Per-party setup state for daBit generation.
#[derive(Debug, Clone)]
pub struct PcgDaBitSetup<F: PrimeField> {
    pub party_id: PartyID,
    /// Seed shared with the *next* party — generates `a` component.
    pub seed_next: [u8; 32],
    /// Seed shared with the *previous* party — generates `b` component.
    pub seed_prev: [u8; 32],
    /// P0: the third pairwise seed (P1↔P2) for computing corrections.
    pub seed_third: Option<[u8; 32]>,
    /// P2: pre-computed corrections (O(N) field elements). P0/P1: empty.
    /// P0 recomputes corrections on-the-fly from its 3 pairwise seeds.
    pub corrections: Vec<F>,
}

/// Output of the dealer's one-time setup.
pub struct DealerOutput<F: PrimeField> {
    pub party0: PcgDaBitSetup<F>,
    pub party1: PcgDaBitSetup<F>,
    pub party2: PcgDaBitSetup<F>,
}

// ── Dealer Setup ────────────────────────────────────────────────────────────

/// P0 generates pairwise seeds and computes corrections for `num` daBits.
///
/// Returns setups for all 3 parties. P0 must send `party2.corrections` to P2.
pub fn dealer_setup<F: PrimeField>(num: usize, rng: &mut impl rand::RngCore) -> DealerOutput<F> {
    let mut seed_01 = [0u8; 32];
    let mut seed_12 = [0u8; 32];
    let mut seed_20 = [0u8; 32];
    rng.fill_bytes(&mut seed_01);
    rng.fill_bytes(&mut seed_12);
    rng.fill_bytes(&mut seed_20);

    // P0 computes corrections in parallel: c(x) = F::from(b(x)) - S(x).
    let corrections: Vec<F> = (0..num)
        .into_par_iter()
        .with_min_len(256)
        .map(|x| {
            let bit_01 = prf_bit(&seed_01, x);
            let bit_12 = prf_bit(&seed_12, x);
            let bit_20 = prf_bit(&seed_20, x);
            let b = bit_01 ^ bit_12 ^ bit_20;

            let s: F = prf_field::<F>(&seed_01, x)
                + prf_field::<F>(&seed_12, x)
                + prf_field::<F>(&seed_20, x);

            let target = if b { F::one() } else { F::zero() };
            target - s
        })
        .collect();

    DealerOutput {
        party0: PcgDaBitSetup {
            party_id: PartyID::ID0,
            seed_next: seed_01,
            seed_prev: seed_20,
            seed_third: Some(seed_12),
            corrections: Vec::new(), // P0 recomputes on-the-fly from 3 seeds
        },
        party1: PcgDaBitSetup {
            party_id: PartyID::ID1,
            seed_next: seed_12,
            seed_prev: seed_01,
            seed_third: None,
            corrections: Vec::new(), // P1 needs no corrections
        },
        party2: PcgDaBitSetup {
            party_id: PartyID::ID2,
            seed_next: seed_20,
            seed_prev: seed_12,
            seed_third: None,
            corrections,             // P2 must store corrections (no seed_third)
        },
    }
}

// ── Local daBit Expansion ───────────────────────────────────────────────────

/// Expand `count` daBits starting at position `start`. No communication.
///
/// P1 uses zero corrections (its share needs none by construction).
/// P2 looks up pre-stored corrections.
/// P0 recomputes corrections on-the-fly from its three pairwise seeds,
/// so P0's `PcgDaBitSetup` doesn't need to store them.
pub fn expand_dabits<F: PrimeField>(
    setup: &PcgDaBitSetup<F>,
    start: usize,
    count: usize,
) -> Vec<DaBit<F>> {
    (start..start + count)
        .into_par_iter()
        .with_min_len(64)
        .map(|x| {
            let correction = match setup.party_id {
                PartyID::ID1 => F::zero(),
                PartyID::ID2 => setup.corrections[x],
                PartyID::ID0 => {
                    // P0 has all 3 seeds → recompute c(x) = F::from(b(x)) - S(x)
                    let seed_third = setup.seed_third
                        .expect("P0 must have seed_third");
                    let bit_01 = prf_bit(&setup.seed_next, x);
                    let bit_12 = prf_bit(&seed_third, x);
                    let bit_20 = prf_bit(&setup.seed_prev, x);
                    let b = bit_01 ^ bit_12 ^ bit_20;

                    let s: F = prf_field::<F>(&setup.seed_next, x)
                        + prf_field::<F>(&seed_third, x)
                        + prf_field::<F>(&setup.seed_prev, x);

                    let target = if b { F::one() } else { F::zero() };
                    target - s
                }
            };
            expand_single_dabit(setup, x, correction)
        })
        .collect()
}

/// Expand a single daBit at position `x` with the given correction.
fn expand_single_dabit<F: PrimeField>(
    setup: &PcgDaBitSetup<F>,
    x: usize,
    correction: F,
) -> DaBit<F> {
    // ── Boolean share ──
    let bit_a = prf_bit(&setup.seed_next, x);
    let bit_b = prf_bit(&setup.seed_prev, x);
    let bit_share = Rep3RingShare::new(Bit::new(bit_a), Bit::new(bit_b));

    // ── Arithmetic share ──
    // Base: a_i = PRF(seed_next, x), b_i = PRF(seed_prev, x).
    // Correction goes to the (P2→P0) edge of the Rep3 ring:
    //   P2 adds c to a_2 (= s_2), P0 adds c to b_0 (= s_2).
    //   Both adjust the same Rep3 slot, keeping consistency.
    let a_i: F = prf_field(&setup.seed_next, x);
    let b_i: F = prf_field(&setup.seed_prev, x);

    let value = match setup.party_id {
        PartyID::ID0 => Rep3PrimeFieldShare::new(a_i, b_i + correction),
        PartyID::ID2 => Rep3PrimeFieldShare::new(a_i + correction, b_i),
        PartyID::ID1 => Rep3PrimeFieldShare::new(a_i, b_i),
    };

    DaBit {
        bit: bit_share,
        value,
    }
}

// ── PRF helpers ─────────────────────────────────────────────────────────────

/// Deterministic PRF producing a single bit.
fn prf_bit(seed: &[u8; 32], x: usize) -> bool {
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(seed);
    hasher.update(b"bool");
    hasher.update(x.to_le_bytes());
    let hash: [u8; 32] = hasher.finalize().into();
    (hash[0] & 1) != 0
}

/// Deterministic PRF producing a field element.
fn prf_field<F: PrimeField>(seed: &[u8; 32], x: usize) -> F {
    let mut hasher = sha3::Sha3_256::new(); // TODO: faster hash func?
    hasher.update(seed);
    hasher.update(b"arith");
    hasher.update(x.to_le_bytes());
    let hash: [u8; 32] = hasher.finalize().into();
    F::from_be_bytes_mod_order(&hash) // TODO: from_be_bytes_mod_order slow
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
        let dealer = dealer_setup::<Fr>(100, &mut rng);

        let d0 = expand_dabits::<Fr>(&dealer.party0, 0, 100);
        let d1 = expand_dabits::<Fr>(&dealer.party1, 0, 100);
        let d2 = expand_dabits::<Fr>(&dealer.party2, 0, 100);

        for x in 0..100 {
            // Rep3 consistency
            assert_eq!(d0[x].bit.a, d1[x].bit.b, "x={x}: P0.a != P1.b");
            assert_eq!(d1[x].bit.a, d2[x].bit.b, "x={x}: P1.a != P2.b");
            assert_eq!(d2[x].bit.a, d0[x].bit.b, "x={x}: P2.a != P0.b");
        }
    }

    #[test]
    fn arithmetic_shares_rep3_consistent() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
        let dealer = dealer_setup::<Fr>(100, &mut rng);

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
        let dealer = dealer_setup::<Fr>(200, &mut rng);

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
        let dealer = dealer_setup::<Fr>(50, &mut rng);

        let batch_a = expand_dabits::<Fr>(&dealer.party0, 0, 50);
        let batch_b = expand_dabits::<Fr>(&dealer.party0, 0, 50);

        for i in 0..50 {
            assert_eq!(batch_a[i].bit, batch_b[i].bit);
            assert_eq!(batch_a[i].value.a, batch_b[i].value.a);
            assert_eq!(batch_a[i].value.b, batch_b[i].value.b);
        }
    }
}
