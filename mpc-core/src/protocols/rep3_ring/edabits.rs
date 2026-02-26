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
use crate::protocols::rep3_ring::{arithmetic as rep3_ring_arith, binary};

use ark_ff::One as _;
use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3_ring::{
    Rep3RingShare,
    ring::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement},
};
use num_bigint::BigUint;
use rand::RngCore;
use rand::distributions::Standard;
use rand::prelude::Distribution;

/// An edaBits *mask-only* value linking the same random `r` across:
/// - `r_ring`: arithmetic sharing over `Z_{2^K}`
/// - `r_fp`: arithmetic sharing over `Fp` (embedding of the same integer `r`)
///
/// This is sufficient for mask-open-add conversions (see [`ring_to_field`]).
#[derive(Debug, Clone)]
pub struct DaRing<T: IntRing2k, F: PrimeField> {
    pub r_ring: Rep3RingShare<T>,
    pub r_fp: Rep3PrimeFieldShare<F>,
}

/// Correlated random tuple for Protocol Π₂ B2A conversion.
///
/// For each K-bit binary→field conversion, stores:
/// - `gamma`: packed random bits known only to P0 (P1/P2 store zero)
/// - `alphas`: per-bit 2-of-2 arithmetic share components in `Fp`
///   (P0 and P1 store α₁, P2 stores α₂, where α₁ + α₂ = γᵢ embedded in Fp)
#[derive(Debug, Clone)]
pub struct EdaBits<T: IntRing2k, F: PrimeField> {
    pub gamma: RingElement<T>,
    pub alphas: Vec<F>,
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
            let bit = rep3_ring_arith::promote_to_trivial_share(party_id, RingElement(Bit::new(r)));
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
/// **Important:** to obtain *consistent* edaBits across parties, each party must
/// call this function with the same RNG seed and the same `num`.
pub fn trivial_edabits_mask<T: IntRing2k, F: PrimeField>(
    num: usize,
    party_id: PartyID,
    rng: &mut impl RngCore,
) -> Vec<DaRing<T, F>> {
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
            let r_ring =
                rep3_ring_arith::promote_to_trivial_share(party_id, RingElement(r_ring_val));

            let r_f = F::from_be_bytes_mod_order(&r_big.to_bytes_be());
            let r_fp = rep3_arith::promote_to_trivial_share(party_id, r_f);

            DaRing { r_ring, r_fp }
        })
        .collect()
}

/// Generate `num` *trivially shared* Protocol Π₂ B2A tuples for tests.
///
/// Each tuple contains:
/// - `gamma`: K random bits packed into `RingElement<T>` (known to P0)
/// - `alphas`: per-bit 2-of-2 arithmetic shares (P0/P1 get α₁, P2 gets α₂)
///
/// **Important:** to obtain *consistent* tuples across parties, each party must
/// call this function with the same RNG seed and the same `num`.
pub fn trivial_edabits<T: IntRing2k, F: PrimeField>(
    num: usize,
    party_id: PartyID,
    rng: &mut impl RngCore,
) -> Vec<EdaBits<T, F>>
where
    Standard: Distribution<T>,
{
    (0..num)
        .map(|_| {
            // Generate K random gamma bits, packed.
            let gamma_val: T = Standard.sample(rng);
            let gamma = RingElement(gamma_val);

            // For each bit, generate the 2-of-2 alpha shares.
            let alphas = (0..T::K)
                .map(|i| {
                    let gamma_bit = ((gamma_val >> i) & T::one()) == T::one();
                    let alpha_1 = F::rand(rng);
                    let alpha_2 = F::from(gamma_bit as u64) - alpha_1;
                    match party_id {
                        PartyID::ID0 | PartyID::ID1 => alpha_1,
                        PartyID::ID2 => alpha_2,
                    }
                })
                .collect();

            EdaBits { gamma, alphas }
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
#[tracing::instrument(skip_all, name = "dabits_preprocess")]
pub fn random_dabits<F: PrimeField, N: Rep3Network>(
    num: usize,
    _rng: &mut impl RngCore,
    io: &mut IoContext<N>,
) -> IoResult<Vec<DaBit<F>>> {
    let bits = (0..num)
        .map(|_| rep3_ring_arith::rand::<Bit, _>(io))
        .collect::<Vec<_>>();

    let values = crate::protocols::rep3_ring::conversion::bit_inject_from_bits_to_field_many::<F, _>(
        &bits, io,
    )?;

    Ok(bits
        .into_iter()
        .zip(values)
        .map(|(bit, value)| DaBit { bit, value })
        .collect())
}

/// Generate `num` random Protocol Π₂ B2A tuples using Rep3 preprocessing.
///
/// For each tuple, generates K random bits γ known only to P0, plus a 2-of-2
/// arithmetic sharing of each γ bit in `Fp` held by P1 and P2.
///
/// **Communication:** P0 → P2: `num * K` field elements (one preprocessing round).
///
/// `rng` is not used for secrecy; secrecy comes from correlated RNG inside `io`.
/// Callers must invoke this function in the same order on all parties to keep
/// `io` RNG streams aligned.
#[tracing::instrument(skip_all, name = "edabits_preprocess")]
pub fn random_edabits<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    num: usize,
    _rng: &mut impl RngCore,
    io: &mut IoContext<N>,
) -> IoResult<Vec<EdaBits<T, F>>>
where
    Standard: Distribution<T>,
{
    let mut gammas = Vec::with_capacity(num);
    let mut all_alphas = Vec::with_capacity(num);

    for _ in 0..num {
        // Generate gamma: XOR of both correlated RNG outputs → private to P0.
        let (g1, g2): (T, T) = io.random_elements();
        let gamma = if io.id == PartyID::ID0 {
            RingElement(g1 ^ g2)
        } else {
            RingElement(T::zero())
        };

        // Generate per-bit alpha_1 from the P0-P1 shared RNG stream.
        // Convention: rng1 shared with next, rng2 shared with prev.
        // P0.rng1 = P1.rng2 → random_fes().0 for P0 = random_fes().1 for P1.
        let mut alphas = Vec::with_capacity(T::K);
        for _ in 0..T::K {
            let (from_rng1, from_rng2) = io.random_fes::<F>();
            let alpha = match io.id {
                PartyID::ID0 => from_rng1,
                PartyID::ID1 => from_rng2,
                PartyID::ID2 => F::zero(), // placeholder, overwritten below
            };
            alphas.push(alpha);
        }

        gammas.push(gamma);
        all_alphas.push(alphas);
    }

    // P0 → P2: send alpha_2 = F::from(gamma_bit) - alpha_1 for each bit.
    if io.id == PartyID::ID0 {
        let alpha_2_all: Vec<F> = gammas
            .iter()
            .zip(&all_alphas)
            .flat_map(|(gamma, alphas)| {
                (0..T::K).map(move |i| {
                    let gamma_bit = ((gamma.0 >> i) & T::one()) == T::one();
                    F::from(gamma_bit as u64) - alphas[i]
                })
            })
            .collect();
        io.network.send_many(PartyID::ID2, &alpha_2_all)?;
    }
    if io.id == PartyID::ID2 {
        let alpha_2_all: Vec<F> = io.network.recv_many(PartyID::ID0)?;
        debug_assert_eq!(alpha_2_all.len(), num * T::K);
        for (j, alphas) in all_alphas.iter_mut().enumerate() {
            for i in 0..T::K {
                alphas[i] = alpha_2_all[j * T::K + i];
            }
        }
    }

    Ok(gammas
        .into_iter()
        .zip(all_alphas)
        .map(|(gamma, alphas)| EdaBits { gamma, alphas })
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
pub fn ring_to_field<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x: Rep3RingShare<T>,
    eda: DaRing<T, F>,
    io: &mut IoContext<N>,
) -> IoResult<Rep3PrimeFieldShare<F>> {
    let c_share = x - eda.r_ring;
    let c_open: RingElement<T> = rep3_ring_arith::open(c_share, io)?;

    let c_big = c_open.0.cast_to_biguint();
    let c_f = F::from_be_bytes_mod_order(&c_big.to_bytes_be());
    let c_fp_share = rep3_arith::promote_to_trivial_share(io.id, c_f);

    Ok(eda.r_fp + c_fp_share)
}

pub fn ring_to_field_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x: &[Rep3RingShare<T>],
    eda: &[DaRing<T, F>],
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

/// Convert a binary (XOR-shared) ring share `[x]` over `Z_{2^K}` into an
/// arithmetic field share `[x]` over `Fp`, using Protocol Π₂.
///
/// Online communication: K bits (P0 → P1, P2) + 1 field element reshare.
pub fn ring_to_field_b2a<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x_binary: Rep3RingShare<T>,
    eda: EdaBits<T, F>,
    io: &mut IoContext<N>,
) -> IoResult<Rep3PrimeFieldShare<F>>
where
    Standard: Distribution<T>,
{
    let mut out = ring_to_field_b2a_many::<T, F, N>(&[x_binary], vec![eda], io)?;
    Ok(out.remove(0))
}

/// Batched Protocol Π₂ B2A conversion.
///
/// For each binary Rep3 share `x`, converts to an arithmetic Rep3 field share
/// using the correlated random tuple in `eda`.
///
/// **Online communication:**
/// - Round 1: P0 broadcasts N packed ring elements (K bits each) to P1 and P2.
/// - Round 2: ShareConvert via `reshare_many` (one field element per conversion).
pub fn ring_to_field_b2a_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x_binary: &[Rep3RingShare<T>],
    eda: Vec<EdaBits<T, F>>,
    io: &mut IoContext<N>,
) -> IoResult<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    if x_binary.len() != eda.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "ring_to_field_b2a_many: length mismatch",
        ));
    }

    let n = x_binary.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Precompute powers of 2 in Fp.
    let pow2 = {
        let mut pow2 = Vec::with_capacity(T::K);
        let mut cur = F::one();
        for _ in 0..T::K {
            pow2.push(cur);
            cur = cur + cur;
        }
        pow2
    };

    // --- Round 1: P0 broadcasts masked values ---
    // P0 computes m = x.a ^ x.b ^ gamma for each value.
    // P1/P2 receive m from P0.
    let ms: Vec<RingElement<T>> = if io.id == PartyID::ID0 {
        let ms: Vec<_> = x_binary
            .iter()
            .zip(&eda)
            .map(|(x, e)| x.a ^ x.b ^ e.gamma)
            .collect();
        io.network.send_many(PartyID::ID1, &ms)?;
        io.network.send_many(PartyID::ID2, &ms)?;
        ms
    } else {
        io.network.recv_many(PartyID::ID0)?
    };

    // --- Local computation: 2-of-2 share components ---
    // β = m ⊕ r_1 = x ⊕ γ  (public to P1, P2; P0 stores zero)
    // v = Σ 2^i · (β_i + (-1)^{β_i} · α_i) = F::from(β) + Σ 2^i · (-1)^{β_i} · α_i
    // P1 adds the public F::from(β); P2 does not (2-of-2 convention).
    let v_components: Vec<F> = ms
        .iter()
        .zip(x_binary)
        .zip(&eda)
        .map(|((m, x), e)| {
            // P0 has no role in the 2-of-2 share; only contributes zero to ShareConvert.
            if io.id == PartyID::ID0 {
                return F::zero();
            }

            let beta = match io.id {
                PartyID::ID0 => unreachable!(),
                PartyID::ID1 => *m ^ x.a, // P1.a = r_1
                PartyID::ID2 => *m ^ x.b, // P2.b = r_1
            };

            let mut v = F::zero();
            for i in 0..T::K {
                let beta_bit = ((beta.0 >> i) & T::one()) == T::one();
                let signed_alpha = if beta_bit { -e.alphas[i] } else { e.alphas[i] };
                v += pow2[i] * signed_alpha;
            }

            // P1 adds the public β value (one party must add it for 2-of-2).
            if io.id == PartyID::ID1 {
                let beta_big = beta.0.cast_to_biguint();
                let beta_f = F::from_be_bytes_mod_order(&beta_big.to_bytes_be());
                v += beta_f;
            }

            v
        })
        .collect();

    // --- Round 2: ShareConvert (2-of-2 → Rep3) via reshare ---
    // z_i = masking_field_element() is a zero-sharing component (Σ z_i = 0).
    // s_self = v_component + z_self.  reshare gives us s_prev from the prev party.
    // Result: Rep3PrimeFieldShare { a: s_self, b: s_prev }.
    let s_selfs: Vec<F> = v_components
        .iter()
        .map(|v| *v + io.masking_field_element::<F>())
        .collect();
    let s_prevs = io.network.reshare_many(&s_selfs)?;

    Ok(s_selfs
        .into_iter()
        .zip(s_prevs)
        .map(|(s_self, s_prev)| Rep3PrimeFieldShare::new(s_self, s_prev))
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

    let opened = binary::open_vec::<Bit, _>(&c_shares, io)?;

    opened
        .into_iter()
        .zip(da)
        .map(|(c, da)| {
            if !c.convert() {
                Ok(da.value)
            } else {
                Ok(rep3_arith::sub_public_by_shared(F::one(), da.value, io.id))
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// EdaBitsPool: pre-generated edaBits/daBits for batched conversions
// ---------------------------------------------------------------------------

/// A pool of pre-generated edaBits and daBits for batched binary→field conversions.
///
/// Elements are consumed (drained from the front) as they are used, ensuring
/// each random mask is used exactly once.
pub struct EdaBitsPool<F: PrimeField> {
    edabits_u64: Vec<EdaBits<u64, F>>,
    edabits_u128: Vec<EdaBits<u128, F>>,
    dabits: Vec<DaBit<F>>,
}

impl<F: PrimeField> EdaBitsPool<F> {
    /// Create an empty pool.
    pub fn empty() -> Self {
        Self {
            edabits_u64: Vec::new(),
            edabits_u128: Vec::new(),
            dabits: Vec::new(),
        }
    }

    /// Create a pool from pre-generated vectors.
    pub fn new(
        edabits_u64: Vec<EdaBits<u64, F>>,
        edabits_u128: Vec<EdaBits<u128, F>>,
        dabits: Vec<DaBit<F>>,
    ) -> Self {
        Self {
            edabits_u64,
            edabits_u128,
            dabits,
        }
    }

    /// Create a trivial pool for testing (all parties must use same RNG seed).
    pub fn trivial(
        num_u64: usize,
        num_u128: usize,
        num_dabits: usize,
        party_id: PartyID,
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            edabits_u64: trivial_edabits(num_u64, party_id, rng),
            edabits_u128: trivial_edabits(num_u128, party_id, rng),
            dabits: trivial_dabits(num_dabits, party_id, rng),
        }
    }

    /// Drain `n` u64-edabits from the pool. Panics if insufficient.
    pub fn take_edabits_u64(&mut self, n: usize) -> Vec<EdaBits<u64, F>> {
        assert!(
            self.edabits_u64.len() >= n,
            "EdaBitsPool: need {n} u64-edabits, have {}",
            self.edabits_u64.len()
        );
        self.edabits_u64.drain(..n).collect()
    }

    /// Drain `n` u128-edabits from the pool. Panics if insufficient.
    pub fn take_edabits_u128(&mut self, n: usize) -> Vec<EdaBits<u128, F>> {
        assert!(
            self.edabits_u128.len() >= n,
            "EdaBitsPool: need {n} u128-edabits, have {}",
            self.edabits_u128.len()
        );
        self.edabits_u128.drain(..n).collect()
    }

    /// Drain `n` dabits from the pool. Panics if insufficient.
    pub fn take_dabits(&mut self, n: usize) -> Vec<DaBit<F>> {
        assert!(
            self.dabits.len() >= n,
            "EdaBitsPool: need {n} dabits, have {}",
            self.dabits.len()
        );
        self.dabits.drain(..n).collect()
    }

    pub fn remaining_u64(&self) -> usize {
        self.edabits_u64.len()
    }

    pub fn remaining_u128(&self) -> usize {
        self.edabits_u128.len()
    }

    pub fn remaining_dabits(&self) -> usize {
        self.dabits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edabits_u64.is_empty() && self.edabits_u128.is_empty() && self.dabits.is_empty()
    }
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;

    use ark_bn254::Fr;
    use mpc_types::protocols::rep3::{
        combine_field_element, combine_field_elements, share_field_element,
    };
    use mpc_types::protocols::rep3_ring::{
        combine_ring_element, combine_ring_element_binary, ring::bit::Bit as RingBit,
        share_ring_element, share_ring_element_binary,
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

        let edas: [DaRing<u64, Fr>; 3] = std::array::from_fn(|i| DaRing {
            r_ring: r_ring_shares[i],
            r_fp: r_fp_shares[i],
        });

        let outputs: [Rep3PrimeFieldShare<Fr>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| (x_ring_shares[i], edas[i].clone()),
            || (),
            |(x_share, eda), mut io_ctx| {
                let io = io_ctx.main();
                ring_to_field::<u64, Fr, _>(x_share, eda, io).map_err(Into::into)
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
        let rs_u64 = xs_u64.iter().map(|&x| x >> 1).collect::<Vec<_>>();

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
        let eda_shares: [Vec<DaRing<u64, Fr>>; 3] = std::array::from_fn(|pid| {
            (0..NVALS)
                .map(|i| DaRing {
                    r_ring: r_ring_shares_per_val[i][pid],
                    r_fp: r_fp_shares_per_val[i][pid],
                })
                .collect()
        });

        let outputs: [Vec<Rep3PrimeFieldShare<Fr>>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| (x_ring_shares[i].clone(), eda_shares[i].clone()),
            || (),
            |(x_shares, edas), mut io_ctx| {
                let io = io_ctx.main();
                ring_to_field_many::<u64, Fr, _>(&x_shares, &edas, io).map_err(Into::into)
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
        let expected = bits
            .into_iter()
            .map(|b| Fr::from(b as u64))
            .collect::<Vec<_>>();
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
        // Verify that the 2-of-2 alpha shares sum to the gamma bits.
        let outputs: [EdaBits<u64, Fr>; 3] = run_rep3_local_test_with_coordinator(
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
                Ok(eda)
            },
            |(), _net| Ok(()),
        )
        .0;

        // P0 and P1 hold alpha_1, P2 holds alpha_2. Check alpha_1 + alpha_2 = gamma_bit.
        let gamma = outputs[0].gamma.0; // P0 knows gamma
        for i in 0..u64::K {
            let gamma_bit = ((gamma >> i) & 1u64) == 1u64;
            let alpha_1 = outputs[0].alphas[i]; // P0's alpha = alpha_1
            let alpha_2 = outputs[2].alphas[i]; // P2's alpha = alpha_2
            assert_eq!(alpha_1 + alpha_2, Fr::from(gamma_bit as u64));
            // P1 also holds alpha_1
            assert_eq!(outputs[1].alphas[i], alpha_1);
        }
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

        // P0 knows gamma. P0 and P1 hold alpha_1, P2 holds alpha_2.
        // Check: alpha_1 + alpha_2 = F::from(gamma_bit) for each bit.
        for i in 0..NUM {
            let gamma = outputs[0][i].gamma.0; // P0 knows gamma
            for b in 0..u64::K {
                let gamma_bit = ((gamma >> b) & 1u64) == 1u64;
                let alpha_1 = outputs[0][i].alphas[b];
                let alpha_2 = outputs[2][i].alphas[b];
                assert_eq!(
                    alpha_1 + alpha_2,
                    Fr::from(gamma_bit as u64),
                    "alpha mismatch at value {i}, bit {b}"
                );
                // P1 also holds alpha_1
                assert_eq!(outputs[1][i].alphas[b], alpha_1);
            }
        }
    }

    #[test]
    fn ring_to_field_b2a_recovers_x_u64() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xB2A_EDA_0001);
        let x = rng.next_u64();
        let x_bin_shares = share_ring_element_binary::<u64, _>(RingElement(x), &mut rng);

        let outputs: [Rep3PrimeFieldShare<Fr>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| x_bin_shares[i],
            || (),
            |x_share, mut io_ctx| {
                let io = io_ctx.main();
                let mut local_rng = ChaCha20Rng::seed_from_u64(0xB2A_EDA_0002);
                let eda = random_edabits::<u64, Fr, _>(1, &mut local_rng, io)?
                    .into_iter()
                    .next()
                    .unwrap();
                ring_to_field_b2a::<u64, Fr, _>(x_share, eda, io).map_err(Into::into)
            },
            |(), _net| Ok(()),
        )
        .0;

        let combined = combine_field_element(outputs[0], outputs[1], outputs[2]);
        assert_eq!(combined, Fr::from(x));
    }

    #[test]
    fn ring_to_field_b2a_many_recovers_xs_u64() {
        const NUM: usize = 16;
        let mut rng = ChaCha20Rng::seed_from_u64(0xB2A_EDA_1001);
        let xs = (0..NUM).map(|_| rng.next_u64()).collect::<Vec<_>>();
        let per_val_shares = xs
            .iter()
            .map(|&x| share_ring_element_binary::<u64, _>(RingElement(x), &mut rng))
            .collect::<Vec<_>>();
        let x_bin_shares: [Vec<Rep3RingShare<u64>>; 3] =
            std::array::from_fn(|pid| per_val_shares.iter().map(|s| s[pid]).collect());

        let outputs: [Vec<Rep3PrimeFieldShare<Fr>>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| x_bin_shares[i].clone(),
            || (),
            |x_shares, mut io_ctx| {
                let io = io_ctx.main();
                let mut local_rng = ChaCha20Rng::seed_from_u64(0xB2A_EDA_1002);
                let edas = random_edabits::<u64, Fr, _>(NUM, &mut local_rng, io)?;
                ring_to_field_b2a_many::<u64, Fr, _>(&x_shares, edas, io).map_err(Into::into)
            },
            |(), _net| Ok(()),
        )
        .0;

        let combined = combine_field_elements(&outputs[0], &outputs[1], &outputs[2]);
        let expected = xs.into_iter().map(Fr::from).collect::<Vec<_>>();
        assert_eq!(combined, expected);
    }
}
