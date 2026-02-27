//! edaBits helpers for Rep3 over rings.
//!
//! This module provides an opt-in conversion primitive to translate an
//! arithmetic Rep3 sharing over `Z_{2^K}` into an arithmetic Rep3 sharing over a
//! prime field `Fp`, using an edaBits mask that links the same random `r` across
//! both domains.

use crate::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use crate::protocols::rep3::{
    PartyID, Rep3PrimeFieldShare, arithmetic as rep3_arith,
    network::{IoContext, Rep3Network},
};
use crate::protocols::rep3_ring::pcg::dabit_gen::{self, PcgDaBitSetup};
use crate::protocols::rep3_ring::{arithmetic as rep3_ring_arith, binary};

use ark_ff::One as _;
use eyre::Ok;
use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3_ring::{
    Rep3RingShare,
    ring::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement},
};
use num_bigint::BigUint;
use rand::RngCore;
use rand::SeedableRng;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rayon::prelude::*;
use std::marker::PhantomData;
use tracing::info_span;

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

            let r_f = F::from(Into::<u128>::into(r_ring_val));
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
) -> eyre::Result<Vec<DaBit<F>>> {
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
pub fn random_edabits<T: IntRing2k, F: PrimeField, N: Rep3NetworkWorker>(
    num: usize,
    _rng: &mut impl RngCore,
    io: &mut IoContextPool<N>,
) -> eyre::Result<Vec<EdaBits<T, F>>>
where
    Standard: Distribution<T>,
{
    let mut gammas = Vec::with_capacity(num);
    let mut all_alphas = Vec::with_capacity(num);

    let _span = info_span!("gen_gamma_alphas").entered();
    for _ in 0..num {
        // Generate gamma: XOR of both correlated RNG outputs → private to P0.
        let (g1, g2): (T, T) = io.main().random_elements();
        let gamma = if io.party_id() == PartyID::ID0 {
            RingElement(g1 ^ g2)
        } else {
            RingElement(T::zero())
        };

        // Generate per-bit alpha_1 from the P0-P1 shared RNG stream.
        // Convention: rng1 shared with next, rng2 shared with prev.
        // P0.rng1 = P1.rng2 → random_fes().0 for P0 = random_fes().1 for P1.
        let mut alphas = Vec::with_capacity(T::K);
        for _ in 0..T::K {
            let (from_rng1, from_rng2) = io.main().random_fes::<F>();
            let alpha = match io.party_id() {
                PartyID::ID0 => from_rng1,
                PartyID::ID1 => from_rng2,
                PartyID::ID2 => F::zero(), // placeholder, overwritten below
            };
            alphas.push(alpha);
        }

        gammas.push(gamma);
        all_alphas.push(alphas);
    }
    drop(_span);

    let _span = info_span!("reshare").entered();

    // P0 → P2: send alpha_2 = F::from(gamma_bit) - alpha_1 for each bit.
    if io.party_id() == PartyID::ID0 {
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
        io.par_chunks(alpha_2_all, None, |chunk, io| {
            io.network.send_many(PartyID::ID2, &chunk)?;
            eyre::Ok(vec![()])
        })?;
    }
    if io.party_id() == PartyID::ID2 {
        let alpha_2_all: Vec<F> =
            io.par_chunks(rayon::iter::repeat_n((), num * T::K), None, |_, io| {
                io.network.recv_many(PartyID::ID0)
            })?;
        debug_assert_eq!(alpha_2_all.len(), num * T::K);
        for (j, alphas) in all_alphas.iter_mut().enumerate() {
            for i in 0..T::K {
                alphas[i] = alpha_2_all[j * T::K + i];
            }
        }
    }
    drop(_span);

    Ok(gammas
        .into_iter()
        .zip(all_alphas)
        .map(|(gamma, alphas)| EdaBits { gamma, alphas })
        .collect())
}

// ---------------------------------------------------------------------------
// Lazy EdaBits: O(1) storage for P0/P1 via deterministic RNG regeneration
// ---------------------------------------------------------------------------

/// Lazy edaBits source that stores RNG seeds instead of materialized gamma/alphas.
///
/// P0/P1 regenerate gamma and alpha values on demand from ChaCha12Rng seeds;
/// P2 stores the received `alpha_2` values (cannot regenerate locally).
///
/// Generation order (deterministic): ALL gammas first (total × size_of::<T>() bytes
/// from each RNG), then ALL alphas (total × T::K × field_bytes from each RNG).
pub struct LazyEdaBits<T: IntRing2k, F: PrimeField> {
    /// RNG state snapshot (from dedicated forked Rep3Rand).
    seed1: [u8; crate::SEED_SIZE],
    pos1: u128,
    seed2: [u8; crate::SEED_SIZE],
    pos2: u128,
    /// Total number of edaBits generated.
    total: usize,
    /// Bytes per field element: ceil(MODULUS_BIT_SIZE / 8).
    field_bytes: usize,
    /// Consumption cursor: next edaBit index to produce.
    cursor: usize,
    /// P2-only storage: flat vec of alpha_2 values (length = total * T::K).
    /// Empty for P0/P1.
    alpha2_flat: Vec<F>,
    /// This party's ID.
    party_id: PartyID,
    _phantom: PhantomData<(T, F)>,
    /// Optional eager storage for backward compatibility (trivial tests).
    eager: Option<Vec<EdaBits<T, F>>>,
}

impl<T: IntRing2k, F: PrimeField> LazyEdaBits<T, F>
where
    Standard: Distribution<T>,
{
    /// Create an empty lazy source (for unused ring types, e.g. u128 when only u64 is needed).
    pub fn empty(party_id: PartyID) -> Self {
        Self {
            seed1: [0u8; crate::SEED_SIZE],
            pos1: 0,
            seed2: [0u8; crate::SEED_SIZE],
            pos2: 0,
            total: 0,
            field_bytes: usize::try_from(F::MODULUS_BIT_SIZE)
                .expect("u32 fits into usize")
                .div_ceil(8),
            cursor: 0,
            alpha2_flat: Vec::new(),
            party_id,
            _phantom: PhantomData,
            eager: None,
        }
    }

    /// Construct a truly lazy source from RNG seeds + P2's alpha_2.
    ///
    /// P0/P1 regenerate gamma/alphas on demand via `take()`.
    /// P2 slices from `alpha2_flat` (received from P0 during preprocessing).
    pub fn new(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        total: usize,
        alpha2_flat: Vec<F>,
        party_id: PartyID,
    ) -> Self {
        let field_bytes = usize::try_from(F::MODULUS_BIT_SIZE)
            .expect("u32 fits into usize")
            .div_ceil(8);
        Self {
            seed1,
            pos1,
            seed2,
            pos2,
            total,
            field_bytes,
            cursor: 0,
            alpha2_flat,
            party_id,
            _phantom: PhantomData,
            eager: None,
        }
    }

    /// Wrap an eagerly-generated `Vec<EdaBits>` as a lazy source.
    ///
    /// Used for backward compatibility with `trivial_edabits` in tests.
    /// The `take()` method drains from the internal vec instead of regenerating.
    pub fn from_eager(edabits: Vec<EdaBits<T, F>>, party_id: PartyID) -> Self {
        let total = edabits.len();
        let field_bytes = usize::try_from(F::MODULUS_BIT_SIZE)
            .expect("u32 fits into usize")
            .div_ceil(8);
        Self {
            seed1: [0u8; crate::SEED_SIZE],
            pos1: 0,
            seed2: [0u8; crate::SEED_SIZE],
            pos2: 0,
            total,
            field_bytes,
            cursor: 0,
            alpha2_flat: Vec::new(),
            party_id,
            _phantom: PhantomData,
            eager: Some(edabits),
        }
    }

    /// Number of remaining edaBits that can be produced.
    pub fn remaining(&self) -> usize {
        self.total - self.cursor
    }

    /// Regenerate `n` edaBits starting from the current cursor position.
    ///
    /// Advances the cursor by `n`. Panics if `n > remaining()`.
    #[tracing::instrument(skip_all, name = "EdaBits::take", fields(n))]
    pub fn take(&mut self, n: usize) -> Vec<EdaBits<T, F>> {
        assert!(
            self.cursor + n <= self.total,
            "LazyEdaBits<u{}>: need {n}, have {} (cursor={}, total={})",
            T::K,
            self.remaining(),
            self.cursor,
            self.total
        );

        if n == 0 {
            return Vec::new();
        }

        // Fast path: eager storage (trivial tests).
        if let Some(eager) = &mut self.eager {
            let result = eager.drain(..n).collect();
            self.cursor += n;
            return result;
        }

        let t_bytes = std::mem::size_of::<T>();
        let k = T::K;
        let party_id = self.party_id;
        let cursor_base = self.cursor;
        let fb = self.field_bytes;

        // P2 only needs its stored alpha_2 values — skip RNG regeneration entirely.
        if party_id == PartyID::ID2 {
            let result: Vec<EdaBits<T, F>> = (0..n)
                .into_par_iter()
                .with_min_len(256)
                .map(|i| {
                    let flat_base = (cursor_base + i) * k;
                    let alphas = self.alpha2_flat[flat_base..flat_base + k].to_vec();
                    EdaBits {
                        gamma: RingElement(T::zero()),
                        alphas,
                    }
                })
                .collect();
            self.cursor += n;
            return result;
        }

        // P0/P1: regenerate gamma and alpha bytes from RNG seeds.
        // Generation order: gammas occupy [0 .. total * t_bytes) bytes,
        // alphas occupy [total * t_bytes .. total * t_bytes + total * k * field_bytes).
        // ChaCha12Rng word_pos is in units of u32 words (4 bytes).
        //
        // IMPORTANT: For ring types with t_bytes < 4 (u8, u16), the byte offset
        // may not be word-aligned. We must seek to the containing word, then skip
        // the leading bytes within that word.
        let gamma_byte_offset = cursor_base * t_bytes;
        let alpha_byte_offset = self.total * t_bytes + cursor_base * k * fb;

        // Helper: seek RNG to the word containing `byte_offset`, generate
        // `needed + skip` bytes, then strip the leading `skip` bytes.
        fn seek_and_generate(
            seed: [u8; crate::SEED_SIZE],
            base_pos: u128,
            byte_offset: usize,
            needed: usize,
        ) -> Vec<u8> {
            let word_offset = (byte_offset as u128) / 4;
            let skip = byte_offset % 4; // bytes to discard within the first word
            let mut rng = crate::RngType::from_seed(seed);
            rng.set_word_pos(base_pos + word_offset);
            let mut buf = vec![0u8; needed + skip];
            rng.fill_bytes(&mut buf);
            if skip > 0 {
                buf.drain(..skip);
            }
            buf
        }

        let gamma_total_bytes = n * t_bytes;
        let alpha_total_bytes = n * k * fb;

        // P0 needs: gamma (both seeds for XOR) + alpha from seed1 only.
        // P1 needs: alpha from seed2 only (no gamma).
        let (g1_bytes, g2_bytes, alpha_bytes);
        if party_id == PartyID::ID0 {
            let _span = info_span!("gen_gamma_alpha").entered();
            let (g1, (g2, a1)) = rayon::join(
                || seek_and_generate(self.seed1, self.pos1, gamma_byte_offset, gamma_total_bytes),
                || {
                    rayon::join(
                        || {
                            seek_and_generate(
                                self.seed2,
                                self.pos2,
                                gamma_byte_offset,
                                gamma_total_bytes,
                            )
                        },
                        || {
                            seek_and_generate(
                                self.seed1,
                                self.pos1,
                                alpha_byte_offset,
                                alpha_total_bytes,
                            )
                        },
                    )
                },
            );
            g1_bytes = g1;
            g2_bytes = g2;
            alpha_bytes = a1;
        } else {
            // P1: only needs alpha from seed2 (= P0's seed1).
            let _span = info_span!("gen_alpha").entered();
            g1_bytes = Vec::new();
            g2_bytes = Vec::new();
            alpha_bytes =
                seek_and_generate(self.seed2, self.pos2, alpha_byte_offset, alpha_total_bytes);
        }

        let _span = info_span!("parse_gamma_alpha").entered();
        let result: Vec<EdaBits<T, F>> = (0..n)
            .into_par_iter()
            .with_min_len(256)
            .map(|i| {
                // Parse gamma: XOR of the two RNG outputs (only P0).
                let gamma = if party_id == PartyID::ID0 {
                    let g_start = i * t_bytes;
                    let g1_val = T::from_le_bytes(&g1_bytes[g_start..g_start + t_bytes]);
                    let g2_val = T::from_le_bytes(&g2_bytes[g_start..g_start + t_bytes]);
                    RingElement(g1_val ^ g2_val)
                } else {
                    RingElement(T::zero())
                };

                // Parse alphas: take first 16 bytes as u128 → F::from(u128).
                // This loses ~half the entropy vs from_be_bytes_mod_order but
                // is fine for semi-honest pseudorandom alphas (128 bits of PRG
                // output maps to a negligibly-biased field element).
                let alphas: Vec<F> = (0..k)
                    .map(|j| {
                        let a_start = (i * k + j) * fb;
                        let lo = u64::from_le_bytes(
                            alpha_bytes[a_start..a_start + 8].try_into().unwrap(),
                        );
                        let hi = u64::from_le_bytes(
                            alpha_bytes[a_start + 8..a_start + 16].try_into().unwrap(),
                        );
                        F::from((hi as u128) << 64 | lo as u128)
                    })
                    .collect();

                EdaBits { gamma, alphas }
            })
            .collect();

        self.cursor += n;
        result
    }
}

/// Generate edaBits: P0→P2 communication only, truly lazy storage.
///
/// P0/P1 store only RNG seeds (~192 bytes). P2 stores the received alpha_2
/// flat vec. The online `take()` method regenerates EdaBits on demand.
///
/// **Communication:** P0 → P2: `num * K` field elements (one preprocessing round).
#[tracing::instrument(skip_all, name = "edabits_preprocess_lazy")]
pub fn random_edabits_lazy<T: IntRing2k, F: PrimeField, N: Rep3NetworkWorker>(
    num: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<LazyEdaBits<T, F>>
where
    Standard: Distribution<T>,
{
    let party_id = io.party_id();
    if num == 0 {
        return Ok(LazyEdaBits::empty(party_id));
    }

    let t_bytes = std::mem::size_of::<T>();
    let k = T::K;
    let field_bytes = usize::try_from(F::MODULUS_BIT_SIZE)
        .expect("u32 fits into usize")
        .div_ceil(8);

    // Fork a dedicated Rep3Rand and snapshot its state BEFORE generating bytes.
    let mut eda_rand = io.main().rngs.rand.fork();
    let (seed1, pos1, seed2, pos2) = eda_rand.snapshot();

    let gamma_total_bytes = num * t_bytes;
    let alpha_total_bytes = num * k * field_bytes;

    // P0 → P2: send alpha_2 = F::from(gamma_bit) - alpha_1.
    // Only P0 needs RNG bytes; P1/P2 skip generation entirely.
    if party_id == PartyID::ID0 {
        let _span = info_span!("gen_rng_bytes").entered();
        // P0 needs: gamma from both seeds (for XOR) + alpha from seed1 only.
        // rng1: generate gamma + alpha contiguously. rng2: gamma only.
        let (all_bytes1, g2_bytes) = {
            let mut a = vec![0u8; gamma_total_bytes + alpha_total_bytes];
            let mut b = vec![0u8; gamma_total_bytes];
            rayon::join(
                || eda_rand.rng1.fill_bytes(&mut a),
                || eda_rand.rng2.fill_bytes(&mut b),
            );
            (a, b)
        };
        let g1_bytes = &all_bytes1[..gamma_total_bytes];
        let a1_bytes = &all_bytes1[gamma_total_bytes..];
        drop(_span);

        let _span = info_span!("compute_send_alpha2").entered();
        let alpha_2_all: Vec<F> = (0..num)
            .into_par_iter()
            .with_min_len(256)
            .flat_map(|i| {
                let g_start = i * t_bytes;
                let g1_val = T::from_le_bytes(&g1_bytes[g_start..g_start + t_bytes]);
                let g2_val = T::from_le_bytes(&g2_bytes[g_start..g_start + t_bytes]);
                let gamma = g1_val ^ g2_val;
                (0..k)
                    .map(|j| {
                        let a_start = (i * k + j) * field_bytes;
                        let lo = u64::from_le_bytes(
                            a1_bytes[a_start..a_start + 8].try_into().unwrap(),
                        );
                        let hi = u64::from_le_bytes(
                            a1_bytes[a_start + 8..a_start + 16].try_into().unwrap(),
                        );
                        let alpha_1 = F::from((hi as u128) << 64 | lo as u128);
                        let gamma_bit = ((gamma >> j) & T::one()) == T::one();
                        F::from(gamma_bit as u64) - alpha_1
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        io.network().send_many(PartyID::ID2, &alpha_2_all)?;
    }

    let alpha2_flat: Vec<F> = if party_id == PartyID::ID2 {
        let _span = info_span!("recv_alpha2").entered();
        let alpha_2_all: Vec<F> = io.network().recv_many(PartyID::ID0)?;
        debug_assert_eq!(alpha_2_all.len(), num * k);
        alpha_2_all
    } else {
        Vec::new()
    };

    // Return truly lazy: P0/P1 store only seeds (~192 bytes), P2 stores alpha2_flat.
    // The temporary all_bytes1/all_bytes2 are dropped here.
    Ok(LazyEdaBits::new(
        seed1,
        pos1,
        seed2,
        pos2,
        num,
        alpha2_flat,
        party_id,
    ))
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
) -> eyre::Result<Rep3PrimeFieldShare<F>> {
    let c_share = x - eda.r_ring;
    let c_open: RingElement<T> = rep3_ring_arith::open(c_share, io)?;

    let c_f = F::from(Into::<u128>::into(c_open.0));
    let c_fp_share = rep3_arith::promote_to_trivial_share(io.id, c_f);

    Ok(eda.r_fp + c_fp_share)
}

pub fn ring_to_field_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x: &[Rep3RingShare<T>],
    eda: &[DaRing<T, F>],
    io: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
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
            let c_f = F::from(Into::<u128>::into(c_open.0));
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
) -> eyre::Result<Rep3PrimeFieldShare<F>>
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
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    if x_binary.len() != eda.len() {
        return Err(eyre::anyhow!("ring_to_field_b2a_many: length mismatch"));
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
                v += F::from(Into::<u128>::into(beta.0));
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
) -> eyre::Result<Rep3PrimeFieldShare<F>> {
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
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
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
/// EdaBits are stored lazily via [`LazyEdaBits`] (O(1) storage for P0/P1).
/// DaBits are stored lazily via [`PcgDaBitSetup`] (expanded on demand).
pub struct EdaBitsPool<F: PrimeField> {
    edabits_u8: LazyEdaBits<u8, F>,
    edabits_u16: LazyEdaBits<u16, F>,
    edabits_u32: LazyEdaBits<u32, F>,
    edabits_u64: LazyEdaBits<u64, F>,
    edabits_u128: LazyEdaBits<u128, F>,
    /// Lazy daBit source (same setup, expanded on demand).
    dabit_setup: PcgDaBitSetup,
    dabit_total: usize,
    dabit_cursor: usize,
}

impl<F: PrimeField> EdaBitsPool<F> {
    /// Create an empty pool.
    pub fn empty(party_id: PartyID) -> Self {
        Self {
            edabits_u8: LazyEdaBits::empty(party_id),
            edabits_u16: LazyEdaBits::empty(party_id),
            edabits_u32: LazyEdaBits::empty(party_id),
            edabits_u64: LazyEdaBits::empty(party_id),
            edabits_u128: LazyEdaBits::empty(party_id),
            dabit_setup: PcgDaBitSetup {
                party_id,
                seed_next: [0u8; 32],
                seed_prev: [0u8; 32],
                seed_third: None,
            },
            dabit_total: 0,
            dabit_cursor: 0,
        }
    }

    /// Create a pool from lazy edaBits sources and a daBit setup.
    pub fn new(
        edabits_u8: LazyEdaBits<u8, F>,
        edabits_u16: LazyEdaBits<u16, F>,
        edabits_u32: LazyEdaBits<u32, F>,
        edabits_u64: LazyEdaBits<u64, F>,
        edabits_u128: LazyEdaBits<u128, F>,
        dabit_setup: PcgDaBitSetup,
        dabit_total: usize,
    ) -> Self {
        Self {
            edabits_u8,
            edabits_u16,
            edabits_u32,
            edabits_u64,
            edabits_u128,
            dabit_setup,
            dabit_total,
            dabit_cursor: 0,
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
        let dabit_setup = dabit_gen::dealer_setup(rng);
        let setup = match party_id {
            PartyID::ID0 => dabit_setup.party0,
            PartyID::ID1 => dabit_setup.party1,
            PartyID::ID2 => dabit_setup.party2,
        };
        Self {
            edabits_u8: LazyEdaBits::empty(party_id),
            edabits_u16: LazyEdaBits::empty(party_id),
            edabits_u32: LazyEdaBits::empty(party_id),
            edabits_u64: LazyEdaBits::from_eager(trivial_edabits(num_u64, party_id, rng), party_id),
            edabits_u128: LazyEdaBits::from_eager(
                trivial_edabits(num_u128, party_id, rng),
                party_id,
            ),
            dabit_setup: setup,
            dabit_total: num_dabits,
            dabit_cursor: 0,
        }
    }

    /// Drain `n` edabits of type T from the pool. Panics if insufficient.
    pub fn take_edabits_u8(&mut self, n: usize) -> Vec<EdaBits<u8, F>> {
        self.edabits_u8.take(n)
    }

    pub fn take_edabits_u16(&mut self, n: usize) -> Vec<EdaBits<u16, F>> {
        self.edabits_u16.take(n)
    }

    pub fn take_edabits_u32(&mut self, n: usize) -> Vec<EdaBits<u32, F>> {
        self.edabits_u32.take(n)
    }

    pub fn take_edabits_u64(&mut self, n: usize) -> Vec<EdaBits<u64, F>> {
        self.edabits_u64.take(n)
    }

    pub fn take_edabits_u128(&mut self, n: usize) -> Vec<EdaBits<u128, F>> {
        self.edabits_u128.take(n)
    }

    /// Expand `n` daBits on demand from the lazy daBit source.
    #[tracing::instrument(skip(self))]
    pub fn take_dabits(&mut self, n: usize) -> Vec<DaBit<F>> {
        assert!(
            self.dabit_cursor + n <= self.dabit_total,
            "EdaBitsPool: need {n} dabits, have {}",
            self.dabit_total - self.dabit_cursor
        );
        let dabits = dabit_gen::expand_dabits(&self.dabit_setup, self.dabit_cursor, n);
        self.dabit_cursor += n;
        dabits
    }

    pub fn remaining_u64(&self) -> usize {
        self.edabits_u64.remaining()
    }

    pub fn remaining_u128(&self) -> usize {
        self.edabits_u128.remaining()
    }

    pub fn remaining_dabits(&self) -> usize {
        self.dabit_total - self.dabit_cursor
    }

    pub fn is_empty(&self) -> bool {
        self.edabits_u8.remaining() == 0
            && self.edabits_u16.remaining() == 0
            && self.edabits_u32.remaining() == 0
            && self.edabits_u64.remaining() == 0
            && self.edabits_u128.remaining() == 0
            && self.remaining_dabits() == 0
    }

    /// Generic edaBits drain, dispatched by `TypeId`.
    ///
    /// Panics if `T` is not one of u8, u16, u32, u64, u128.
    pub fn take_edabits<T: IntRing2k>(&mut self, n: usize) -> Vec<EdaBits<T, F>>
    where
        Standard: Distribution<T>,
    {
        use std::any::TypeId;
        // Safety: We transmute between Vec<EdaBits<concrete, F>> and Vec<EdaBits<T, F>>
        // only when TypeId confirms T == concrete. The EdaBits struct layout is
        // identical for the same concrete type, so the transmute is a no-op.
        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<u8>() {
            let v = self.edabits_u8.take(n);
            // SAFETY: T == u8 confirmed by TypeId check.
            unsafe { std::mem::transmute::<Vec<EdaBits<u8, F>>, Vec<EdaBits<T, F>>>(v) }
        } else if tid == TypeId::of::<u16>() {
            let v = self.edabits_u16.take(n);
            unsafe { std::mem::transmute::<Vec<EdaBits<u16, F>>, Vec<EdaBits<T, F>>>(v) }
        } else if tid == TypeId::of::<u32>() {
            let v = self.edabits_u32.take(n);
            unsafe { std::mem::transmute::<Vec<EdaBits<u32, F>>, Vec<EdaBits<T, F>>>(v) }
        } else if tid == TypeId::of::<u64>() {
            let v = self.edabits_u64.take(n);
            unsafe { std::mem::transmute::<Vec<EdaBits<u64, F>>, Vec<EdaBits<T, F>>>(v) }
        } else if tid == TypeId::of::<u128>() {
            let v = self.edabits_u128.take(n);
            unsafe { std::mem::transmute::<Vec<EdaBits<u128, F>>, Vec<EdaBits<T, F>>>(v) }
        } else {
            panic!("EdaBitsPool::take_edabits: unsupported ring type");
        }
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
                random_edabits::<u64, Fr, _>(NUM, &mut rng, &mut io_ctx).map_err(Into::into)
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
    fn lazy_edabits_consistent() {
        const NUM: usize = 32;
        let outputs: [Vec<EdaBits<u64, Fr>>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| i,
            || (),
            |party_idx, mut io_ctx| {
                assert_eq!(usize::from(io_ctx.party_idx()), party_idx);
                let mut lazy = random_edabits_lazy::<u64, Fr, _>(NUM, &mut io_ctx)?;
                Ok(lazy.take(NUM))
            },
            |(), _net| Ok(()),
        )
        .0;

        // P0 knows gamma. P0 and P1 hold alpha_1, P2 holds alpha_2.
        // Check: alpha_1 + alpha_2 = F::from(gamma_bit) for each bit.
        for i in 0..NUM {
            let gamma = outputs[0][i].gamma.0;
            for b in 0..u64::K {
                let gamma_bit = ((gamma >> b) & 1u64) == 1u64;
                let alpha_1 = outputs[0][i].alphas[b];
                let alpha_2 = outputs[2][i].alphas[b];
                assert_eq!(
                    alpha_1 + alpha_2,
                    Fr::from(gamma_bit as u64),
                    "lazy alpha mismatch at value {i}, bit {b}"
                );
                // P1 also holds alpha_1
                assert_eq!(outputs[1][i].alphas[b], alpha_1);
            }
        }
    }

    #[test]
    fn lazy_edabits_b2a_recovers_x_u64() {
        const NUM: usize = 16;
        let mut rng = ChaCha20Rng::seed_from_u64(0x1A2B_B2A_001);
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
                let mut lazy = random_edabits_lazy::<u64, Fr, _>(NUM, &mut io_ctx)?;
                let edas = lazy.take(NUM);
                ring_to_field_b2a_many::<u64, Fr, _>(&x_shares, edas, io_ctx.main())
                    .map_err(Into::into)
            },
            |(), _net| Ok(()),
        )
        .0;

        let combined = combine_field_elements(&outputs[0], &outputs[1], &outputs[2]);
        let expected = xs.into_iter().map(Fr::from).collect::<Vec<_>>();
        assert_eq!(combined, expected);
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
                let mut local_rng = ChaCha20Rng::seed_from_u64(0xB2A_EDA_0002);
                let eda = random_edabits::<u64, Fr, _>(1, &mut local_rng, &mut io_ctx)?
                    .into_iter()
                    .next()
                    .unwrap();
                ring_to_field_b2a::<u64, Fr, _>(x_share, eda, io_ctx.main()).map_err(Into::into)
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
                let mut local_rng = ChaCha20Rng::seed_from_u64(0xB2A_EDA_1002);
                let edas = random_edabits::<u64, Fr, _>(NUM, &mut local_rng, &mut io_ctx)?;
                ring_to_field_b2a_many::<u64, Fr, _>(&x_shares, edas, io_ctx.main())
                    .map_err(Into::into)
            },
            |(), _net| Ok(()),
        )
        .0;

        let combined = combine_field_elements(&outputs[0], &outputs[1], &outputs[2]);
        let expected = xs.into_iter().map(Fr::from).collect::<Vec<_>>();
        assert_eq!(combined, expected);
    }

    /// Test 3 sequential `ring_to_field_b2a_many` calls on the same io_ctx,
    /// mimicking the operand Q pattern: first u32, then u16, then u16.
    #[test]
    fn sequential_b2a_mixed_rings_on_same_io_ctx() {
        const NUM: usize = 16;
        let mut rng = ChaCha20Rng::seed_from_u64(0xB2A_5E0_001);

        // Generate u32 values for "identity"
        let id_vals: Vec<u32> = (0..NUM).map(|_| rng.next_u32()).collect();
        let id_shares: [Vec<Rep3RingShare<u32>>; 3] = {
            let per = id_vals
                .iter()
                .map(|&x| share_ring_element_binary::<u32, _>(RingElement(x), &mut rng))
                .collect::<Vec<_>>();
            std::array::from_fn(|pid| per.iter().map(|s| s[pid]).collect())
        };

        // Generate u16 values for "left" and "right"
        let left_vals: Vec<u16> = (0..NUM).map(|_| rng.next_u32() as u16).collect();
        let left_shares: [Vec<Rep3RingShare<u16>>; 3] = {
            let per = left_vals
                .iter()
                .map(|&x| share_ring_element_binary::<u16, _>(RingElement(x), &mut rng))
                .collect::<Vec<_>>();
            std::array::from_fn(|pid| per.iter().map(|s| s[pid]).collect())
        };

        let right_vals: Vec<u16> = (0..NUM).map(|_| rng.next_u32() as u16).collect();
        let right_shares: [Vec<Rep3RingShare<u16>>; 3] = {
            let per = right_vals
                .iter()
                .map(|&x| share_ring_element_binary::<u16, _>(RingElement(x), &mut rng))
                .collect::<Vec<_>>();
            std::array::from_fn(|pid| per.iter().map(|s| s[pid]).collect())
        };

        let outputs: [(
            Vec<Rep3PrimeFieldShare<Fr>>,
            Vec<Rep3PrimeFieldShare<Fr>>,
            Vec<Rep3PrimeFieldShare<Fr>>,
        ); 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| {
                (
                    id_shares[i].clone(),
                    left_shares[i].clone(),
                    right_shares[i].clone(),
                )
            },
            || (),
            |(id_sh, left_sh, right_sh), mut io_ctx| {
                // Generate lazy edaBits for u32 and u16
                let mut lazy_u32 = random_edabits_lazy::<u32, Fr, _>(NUM, &mut io_ctx)?;
                let mut lazy_u16 = random_edabits_lazy::<u16, Fr, _>(2 * NUM, &mut io_ctx)?;

                let io = io_ctx.main();

                // Call 1: u32 identity
                let identity = {
                    let edas = lazy_u32.take(NUM);
                    ring_to_field_b2a_many::<u32, Fr, _>(&id_sh, edas, io)?
                };

                // Call 2: u16 left
                let left = {
                    let edas = lazy_u16.take(NUM);
                    ring_to_field_b2a_many::<u16, Fr, _>(&left_sh, edas, io)?
                };

                // Call 3: u16 right
                let right = {
                    let edas = lazy_u16.take(NUM);
                    ring_to_field_b2a_many::<u16, Fr, _>(&right_sh, edas, io)?
                };

                Ok((identity, left, right))
            },
            |(), _net| Ok(()),
        )
        .0;

        // Verify identity (u32)
        let id_combined = combine_field_elements(&outputs[0].0, &outputs[1].0, &outputs[2].0);
        let id_expected: Vec<Fr> = id_vals.iter().map(|&x| Fr::from(x as u64)).collect();
        assert_eq!(id_combined, id_expected, "identity mismatch");

        // Verify left (u16)
        let left_combined = combine_field_elements(&outputs[0].1, &outputs[1].1, &outputs[2].1);
        let left_expected: Vec<Fr> = left_vals.iter().map(|&x| Fr::from(x as u64)).collect();
        assert_eq!(left_combined, left_expected, "left operand mismatch");

        // Verify right (u16)
        let right_combined = combine_field_elements(&outputs[0].2, &outputs[1].2, &outputs[2].2);
        let right_expected: Vec<Fr> = right_vals.iter().map(|&x| Fr::from(x as u64)).collect();
        assert_eq!(right_combined, right_expected, "right operand mismatch");
    }

    /// Test that lazy u16 edaBits work correctly when take() is called at
    /// non-zero cursor positions (regression test for word-alignment bug).
    #[test]
    fn lazy_u16_edabits_b2a_with_cursor_offset() {
        const BATCH1: usize = 7; // odd number to misalign cursor
        const BATCH2: usize = 5;
        let mut rng = ChaCha20Rng::seed_from_u64(0xAB0B_0016);

        // Values for batch 2 (the one we verify)
        let xs: Vec<u16> = (0..BATCH2).map(|_| rng.next_u32() as u16).collect();
        let per_val_shares = xs
            .iter()
            .map(|&x| share_ring_element_binary::<u16, _>(RingElement(x), &mut rng))
            .collect::<Vec<_>>();
        let x_bin_shares: [Vec<Rep3RingShare<u16>>; 3] =
            std::array::from_fn(|pid| per_val_shares.iter().map(|s| s[pid]).collect());

        let outputs: [Vec<Rep3PrimeFieldShare<Fr>>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| x_bin_shares[i].clone(),
            || (),
            |x_shares, mut io_ctx| {
                let mut lazy = random_edabits_lazy::<u16, Fr, _>(BATCH1 + BATCH2, &mut io_ctx)?;

                // Consume first batch (advances cursor by BATCH1, which is odd)
                let _discard = lazy.take(BATCH1);

                // Now take batch2 at cursor=BATCH1 (odd) — this tests the word alignment fix
                let edas = lazy.take(BATCH2);
                ring_to_field_b2a_many::<u16, Fr, _>(&x_shares, edas, io_ctx.main())
                    .map_err(Into::into)
            },
            |(), _net| Ok(()),
        )
        .0;

        let combined = combine_field_elements(&outputs[0], &outputs[1], &outputs[2]);
        let expected: Vec<Fr> = xs.iter().map(|&x| Fr::from(x as u64)).collect();
        assert_eq!(
            combined, expected,
            "u16 lazy B2A mismatch after cursor offset"
        );
    }
}
