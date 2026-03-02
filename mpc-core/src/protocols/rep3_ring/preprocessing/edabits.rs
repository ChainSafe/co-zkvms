//! edaBits helpers for Rep3 over rings.
//!
//! This module provides an opt-in conversion primitive to translate an
//! arithmetic Rep3 sharing over `Z_{2^K}` into an arithmetic Rep3 sharing over a
//! prime field `Fp`, using an edaBits mask that links the same random `r` across
//! both domains.

use super::backing_store;
use super::dabits::{DaBitBatch, LazyDaBits};
use crate::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use crate::protocols::rep3::{
    PartyID, Rep3PrimeFieldShare, arithmetic as rep3_arith,
    network::{IoContext, Rep3Network},
};
use crate::protocols::rep3_ring::arithmetic as rep3_ring_arith;
use eyre::Ok;
use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3_ring::{
    Rep3RingShare,
    ring::{int_ring::IntRing2k, ring_impl::RingElement},
};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::{RngCore, SeedableRng};
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

/// Flat batch of edaBits: avoids per-edaBit heap allocations.
///
/// For `n` edaBits with `K = T::K` bits each:
/// - `gammas[i]`: packed random bits for edaBit `i` (only meaningful for P0)
/// - `alphas_flat[i*K + j]`: alpha for edaBit `i`, bit `j`
pub struct EdaBitsBatch<T: IntRing2k, F: PrimeField> {
    pub gammas: Vec<RingElement<T>>,
    pub alphas_flat: Vec<F>,
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

    let _span = tracing::trace_span!("gen_gamma_alphas").entered();
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

    let _span = tracing::trace_span!("reshare").entered();

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
    /// P2-only storage: flat alpha_2 values (length = total * T::K).
    /// Empty for P0/P1.  May be backed by a memory-mapped file.
    alpha2_flat: backing_store::BackingStore<F>,
    /// This party's ID.
    party_id: PartyID,
    /// Path to the meta file on disk (set when loaded via `load()`).
    /// `None` for in-memory-only pools (freshly preprocessed).
    meta_path: Option<std::path::PathBuf>,
    _phantom: PhantomData<(T, F)>,
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
            alpha2_flat: backing_store::BackingStore::Empty,
            party_id,
            meta_path: None,
            _phantom: PhantomData,
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
            alpha2_flat: backing_store::BackingStore::from_vec(alpha2_flat),
            party_id,
            meta_path: None,
            _phantom: PhantomData,
        }
    }

    /// Number of remaining edaBits that can be produced.
    pub fn remaining(&self) -> usize {
        self.total - self.cursor
    }

    /// Drain `n` edaBits as a flat `EdaBitsBatch` with zero per-edaBit allocations.
    ///
    /// Two allocations total: `gammas` (len n) + `alphas_flat` (len n*K).
    pub fn take_batch(&mut self, n: usize) -> EdaBitsBatch<T, F> {
        assert!(
            self.cursor + n <= self.total,
            "LazyEdaBits<u{}>: need {n}, have {} (cursor={}, total={})",
            T::K,
            self.remaining(),
            self.cursor,
            self.total
        );

        if n == 0 {
            return EdaBitsBatch {
                gammas: Vec::new(),
                alphas_flat: Vec::new(),
            };
        }

        let t_bytes = std::mem::size_of::<T>();
        let k = T::K;
        let party_id = self.party_id;
        let cursor_base = self.cursor;
        let fb = self.field_bytes;

        // P2: slice from stored alpha2_flat, zero gammas.
        if party_id == PartyID::ID2 {
            let flat_start = cursor_base * k;
            let flat_end = flat_start + n * k;
            let alphas_flat = self.alpha2_flat.as_slice()[flat_start..flat_end].to_vec();
            let gammas = vec![RingElement(T::zero()); n];
            self.cursor += n;
            self.persist_cursor();
            self.alpha2_flat.consume(flat_start, flat_end);
            return EdaBitsBatch {
                gammas,
                alphas_flat,
            };
        }

        // P0/P1: regenerate from RNG seeds.
        let gamma_byte_offset = cursor_base * t_bytes;
        let alpha_byte_offset = self.total * t_bytes + cursor_base * k * fb;

        fn seek_and_generate(
            seed: [u8; crate::SEED_SIZE],
            base_pos: u128,
            byte_offset: usize,
            needed: usize,
        ) -> Vec<u8> {
            let word_offset = (byte_offset as u128) / 4;
            let skip = byte_offset % 4;
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

        let (g1_bytes, g2_bytes, alpha_bytes);
        if party_id == PartyID::ID0 {
            let _span = tracing::trace_span!("gen_gamma_alpha").entered();
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
            let _span = tracing::trace_span!("gen_alpha").entered();
            g1_bytes = Vec::new();
            g2_bytes = Vec::new();
            alpha_bytes =
                seek_and_generate(self.seed2, self.pos2, alpha_byte_offset, alpha_total_bytes);
        }

        // Build flat arrays: one gammas vec + one alphas_flat vec.
        let _span = tracing::trace_span!("build_batch").entered();
        let gammas: Vec<RingElement<T>> = if party_id == PartyID::ID0 {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let g_start = i * t_bytes;
                    let g1_val = T::from_le_bytes(&g1_bytes[g_start..g_start + t_bytes]);
                    let g2_val = T::from_le_bytes(&g2_bytes[g_start..g_start + t_bytes]);
                    RingElement(g1_val ^ g2_val)
                })
                .collect()
        } else {
            vec![RingElement(T::zero()); n]
        };

        let alphas_flat: Vec<F> = (0..n * k)
            .into_par_iter()
            .with_min_len(256)
            .map(|idx| {
                let a_start = idx * fb;
                let lo = u64::from_le_bytes(alpha_bytes[a_start..a_start + 8].try_into().unwrap());
                let hi =
                    u64::from_le_bytes(alpha_bytes[a_start + 8..a_start + 16].try_into().unwrap());
                F::from((hi as u128) << 64 | lo as u128)
            })
            .collect();

        self.cursor += n;
        EdaBitsBatch {
            gammas,
            alphas_flat,
        }
    }
}

// Persistence methods — no `Standard: Distribution<T>` bound needed.
impl<T: IntRing2k, F: PrimeField> LazyEdaBits<T, F> {
    /// Write this lazy source to `dir`.
    ///
    /// Creates `edabits_{K}.meta` (all parties) and `edabits_{K}.alpha2`
    /// (P2 only, when non-empty).
    pub fn save(&self, dir: &std::path::Path) -> std::io::Result<()> {
        const { backing_store::assert_field_layout::<F>() };
        std::fs::create_dir_all(dir)?;

        let suffix = T::K;
        let meta_path = dir.join(format!("edabits_{suffix}.meta"));
        backing_store::write_meta(
            &meta_path,
            &backing_store::MetaData {
                seed1: self.seed1,
                pos1: self.pos1,
                seed2: self.seed2,
                pos2: self.pos2,
                total: self.total,
                party_id_byte: backing_store::party_id_to_byte(self.party_id),
                cursor: self.cursor,
                field_bytes: self.field_bytes,
            },
        )?;

        if !self.alpha2_flat.is_empty() {
            let data_path = dir.join(format!("edabits_{suffix}.alpha2"));
            self.alpha2_flat.save_to_file(&data_path)?;
        }
        std::result::Result::Ok(())
    }

    /// Load a previously saved lazy source from `dir`.
    ///
    /// P2 memory-maps the alpha2 file for JIT retrieval.
    pub fn load(dir: &std::path::Path, party_id: PartyID) -> std::io::Result<Self> {
        const { backing_store::assert_field_layout::<F>() };

        let suffix = T::K;
        let meta_path = dir.join(format!("edabits_{suffix}.meta"));
        let meta = backing_store::read_meta(&meta_path)?;
        assert_eq!(
            meta.party_id_byte,
            backing_store::party_id_to_byte(party_id)
        );

        let alpha2_flat = if party_id == PartyID::ID2 && meta.total > 0 {
            let data_path = dir.join(format!("edabits_{suffix}.alpha2"));
            backing_store::BackingStore::load_from_file(&data_path)?
        } else {
            backing_store::BackingStore::Empty
        };

        std::result::Result::Ok(Self {
            seed1: meta.seed1,
            pos1: meta.pos1,
            seed2: meta.seed2,
            pos2: meta.pos2,
            total: meta.total,
            field_bytes: meta.field_bytes,
            cursor: meta.cursor,
            alpha2_flat,
            party_id,
            meta_path: Some(meta_path),
            _phantom: PhantomData,
        })
    }

    /// Persist the current cursor to the meta file on disk.
    ///
    /// No-op when `meta_path` is `None` (in-memory-only pool) or when the
    /// `reuse-preproc` feature is enabled.
    fn persist_cursor(&self) {
        if let Some(ref path) = self.meta_path {
            let _ = backing_store::update_cursor(path, self.cursor);
        }
    }
}

impl<T: IntRing2k, F: PrimeField> Drop for LazyEdaBits<T, F> {
    fn drop(&mut self) {
        #[cfg(not(feature = "reuse-preproc"))]
        self.persist_cursor();
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
        let _span = tracing::trace_span!("gen_rng_bytes").entered();
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

        let _span = tracing::trace_span!("compute_send_alpha2").entered();
        let mut alpha_2_all = vec![F::zero(); num * k];
        alpha_2_all
            .par_chunks_mut(k)
            .enumerate()
            .with_min_len(256)
            .for_each(|(i, chunk)| {
                let g_start = i * t_bytes;
                let g1_val = T::from_le_bytes(&g1_bytes[g_start..g_start + t_bytes]);
                let g2_val = T::from_le_bytes(&g2_bytes[g_start..g_start + t_bytes]);
                let gamma = g1_val ^ g2_val;
                for j in 0..k {
                    let a_start = (i * k + j) * field_bytes;
                    let lo = u64::from_le_bytes(a1_bytes[a_start..a_start + 8].try_into().unwrap());
                    let hi =
                        u64::from_le_bytes(a1_bytes[a_start + 8..a_start + 16].try_into().unwrap());
                    let alpha_1 = F::from((hi as u128) << 64 | lo as u128);
                    let gamma_bit = ((gamma >> j) & T::one()) == T::one();
                    chunk[j] = F::from(gamma_bit as u64) - alpha_1;
                }
            });
        io.network().send_many(PartyID::ID2, &alpha_2_all)?;
    }

    let alpha2_flat: Vec<F> = if party_id == PartyID::ID2 {
        let _span = tracing::trace_span!("recv_alpha2").entered();
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

/// Sequential preprocessing: generate all edaBits + daBits with natural TCP
/// pipelining. Each `random_edabits_lazy` call sends α₂ immediately after
/// computing it, so P2 starts receiving while P0 computes the next type.
#[tracing::instrument(skip_all, name = "preprocess_pool")]
pub fn preprocess_pool<F: PrimeField, N: Rep3NetworkWorker>(
    counts: [usize; 5], // [u8, u16, u32, u64, u128]
    num_dabits: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<EdaBitsPool<F>> {
    let e0 = random_edabits_lazy::<u8, F, _>(counts[0], io)?;
    let e1 = random_edabits_lazy::<u16, F, _>(counts[1], io)?;
    let e2 = random_edabits_lazy::<u32, F, _>(counts[2], io)?;
    let e3 = random_edabits_lazy::<u64, F, _>(counts[3], io)?;
    let e4 = random_edabits_lazy::<u128, F, _>(counts[4], io)?;
    let d = super::dabits::random_dabits_lazy::<F, _>(num_dabits, io)?;
    Ok(EdaBitsPool::new(e0, e1, e2, e3, e4, d))
}

/// Batched preprocessing: generate all edaBits + daBits in **2 network rounds**
/// instead of 7 sequential rounds (5 edaBit + 2 daBit).
///
/// Round 1: P0→P2 sends all edaBit α₂ + daBit α₂; P1→P2 sends daBit s₁₂.
/// Round 2: P2→P0 sends daBit s₂₀.
#[tracing::instrument(skip_all, name = "preprocess_pool_batched")]
pub fn preprocess_pool_batched<F: PrimeField, N: Rep3NetworkWorker>(
    counts: [usize; 5], // [u8, u16, u32, u64, u128]
    num_dabits: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<EdaBitsPool<F>> {
    use super::dabits;
    use crate::protocols::rep3::rngs::Rep3Rand;

    let party_id = io.party_id();
    let fb = usize::try_from(F::MODULUS_BIT_SIZE)
        .expect("u32 fits into usize")
        .div_ceil(8);

    // Phase 1: Fork 6 Rep3Rands and snapshot seeds (local, no communication).
    let mut rands: [Rep3Rand; 6] = std::array::from_fn(|_| io.main().rngs.rand.fork());
    let snaps: [_; 6] = std::array::from_fn(|i| rands[i].snapshot());

    // Helper: compute edaBit α₂ for P0 given a forked Rep3Rand.
    #[tracing::instrument(skip(eda_rand))]
    fn edabit_alpha2_p0<T: IntRing2k, Fp: PrimeField>(
        num: usize,
        eda_rand: &mut Rep3Rand,
        fb: usize,
    ) -> Vec<Fp>
    where
        Standard: Distribution<T>,
    {
        if num == 0 {
            return Vec::new();
        }
        let t_bytes = std::mem::size_of::<T>();
        let k = T::K;
        let gamma_total = num * t_bytes;
        let alpha_total = num * k * fb;
        let (all1, g2) = {
            let mut a = vec![0u8; gamma_total + alpha_total];
            let mut b = vec![0u8; gamma_total];
            rayon::join(
                || eda_rand.rng1.fill_bytes(&mut a),
                || eda_rand.rng2.fill_bytes(&mut b),
            );
            (a, b)
        };
        let g1 = &all1[..gamma_total];
        let a1 = &all1[gamma_total..];
        let mut out = vec![Fp::zero(); num * k];
        out.par_chunks_mut(k)
            .enumerate()
            .with_min_len(256)
            .for_each(|(i, chunk)| {
                let gs = i * t_bytes;
                let g1v = T::from_le_bytes(&g1[gs..gs + t_bytes]);
                let g2v = T::from_le_bytes(&g2[gs..gs + t_bytes]);
                let gamma = g1v ^ g2v;
                for j in 0..k {
                    let s = (i * k + j) * fb;
                    let alpha1 = dabits::parse_field::<Fp>(&a1, s);
                    let gbit = ((gamma >> j) & T::one()) == T::one();
                    chunk[j] = Fp::from(gbit as u64) - alpha1;
                }
            });
        out
    }

    // Phase 2: Compute data to send (local, parallelizable).
    match party_id {
        PartyID::ID0 => {
            // Compute all edaBit α₂ + daBit α₂.
            let ea0 = edabit_alpha2_p0::<u8, F>(counts[0], &mut rands[0], fb);
            let ea1 = edabit_alpha2_p0::<u16, F>(counts[1], &mut rands[1], fb);
            let ea2 = edabit_alpha2_p0::<u32, F>(counts[2], &mut rands[2], fb);
            let ea3 = edabit_alpha2_p0::<u64, F>(counts[3], &mut rands[3], fb);
            let ea4 = edabit_alpha2_p0::<u128, F>(counts[4], &mut rands[4], fb);

            // daBit α₂ (same logic as random_dabits_lazy P0 branch)
            let da_alpha2 = if num_dabits > 0 {
                let r = &mut rands[5];
                let slen1 = num_dabits + num_dabits * fb + num_dabits * fb;
                let slen2 = num_dabits;
                let (s1, s2) = {
                    let mut a = vec![0u8; slen1];
                    let mut b = vec![0u8; slen2];
                    rayon::join(|| r.rng1.fill_bytes(&mut a), || r.rng2.fill_bytes(&mut b));
                    (a, b)
                };
                let g1 = &s1[..num_dabits];
                let a1 = &s1[num_dabits..num_dabits + num_dabits * fb];
                (0..num_dabits)
                    .into_par_iter()
                    .with_min_len(256)
                    .map(|i| {
                        let gbit = ((g1[i] ^ s2[i]) & 1) != 0;
                        let alpha1: F = dabits::parse_field(&a1, i * fb);
                        F::from(gbit as u64) - alpha1
                    })
                    .collect::<Vec<F>>()
            } else {
                Vec::new()
            };

            // Round 1: P0 → P2 — distribute across parallel fork channels
            let mut combined: Vec<F> = Vec::with_capacity(
                ea0.len() + ea1.len() + ea2.len() + ea3.len() + ea4.len() + da_alpha2.len(),
            );
            combined.extend_from_slice(&ea0);
            combined.extend_from_slice(&ea1);
            combined.extend_from_slice(&ea2);
            combined.extend_from_slice(&ea3);
            combined.extend_from_slice(&ea4);
            combined.extend_from_slice(&da_alpha2);
            io.par_chunks(
                combined.into_par_iter(),
                None,
                |chunk: Vec<F>, ctx| -> eyre::Result<Vec<()>> {
                    ctx.network.send_many(PartyID::ID2, &chunk)?;
                    Ok(vec![])
                },
            )?;

            // Round 2: P0 ← P2 receives daBit s₂₀ across fork channels
            let s20: Vec<F> = if num_dabits > 0 {
                let _span = tracing::info_span!("resv_s20").entered();
                io.par_chunks(
                    0..num_dabits,
                    None,
                    |_: Vec<usize>, ctx| -> eyre::Result<Vec<F>> {
                        Ok(ctx.network.recv_many::<F>(PartyID::ID2)?)
                    },
                )?
            } else {
                Vec::new()
            };

            let mk = |i: usize| snaps[i];
            let (s1, p1, s2, p2) = mk(0);
            let e0 = LazyEdaBits::<u8, F>::new(s1, p1, s2, p2, counts[0], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(1);
            let e1 = LazyEdaBits::<u16, F>::new(s1, p1, s2, p2, counts[1], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(2);
            let e2 = LazyEdaBits::<u32, F>::new(s1, p1, s2, p2, counts[2], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(3);
            let e3 = LazyEdaBits::<u64, F>::new(s1, p1, s2, p2, counts[3], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(4);
            let e4 = LazyEdaBits::<u128, F>::new(s1, p1, s2, p2, counts[4], Vec::new(), party_id);
            let (ds1, dp1, ds2, dp2) = snaps[5];
            Ok(EdaBitsPool::new(
                e0,
                e1,
                e2,
                e3,
                e4,
                dabits::LazyDaBits::new(ds1, dp1, ds2, dp2, num_dabits, s20, party_id),
            ))
        }
        PartyID::ID1 => {
            // P1 only sends daBit s₁₂ to P2 (edaBits are local-only for P1).
            if num_dabits > 0 {
                let r = &mut rands[5];
                let slen2 = num_dabits + num_dabits * fb + num_dabits * fb;
                let slen1 = num_dabits;
                let (s2, s1) = {
                    let mut a = vec![0u8; slen2];
                    let mut b = vec![0u8; slen1];
                    rayon::join(|| r.rng2.fill_bytes(&mut a), || r.rng1.fill_bytes(&mut b));
                    (a, b)
                };
                let a1 = &s2[num_dabits..num_dabits + num_dabits * fb];
                let r1 = &s2[num_dabits + num_dabits * fb..];
                let theta_bytes = &s1;

                let s12: Vec<F> = (0..num_dabits)
                    .into_par_iter()
                    .with_min_len(256)
                    .map(|i| {
                        let theta = (theta_bytes[i] & 1) != 0;
                        let neg1_theta = if theta { -F::one() } else { F::one() };
                        let alpha1: F = dabits::parse_field(&a1, i * fb);
                        let r1_val: F = dabits::parse_field(&r1, i * fb);
                        neg1_theta * alpha1 - r1_val
                    })
                    .collect();
                io.network().send_many(PartyID::ID2, &s12)?;
            }

            let mk = |i: usize| snaps[i];
            let (s1, p1, s2, p2) = mk(0);
            let e0 = LazyEdaBits::<u8, F>::new(s1, p1, s2, p2, counts[0], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(1);
            let e1 = LazyEdaBits::<u16, F>::new(s1, p1, s2, p2, counts[1], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(2);
            let e2 = LazyEdaBits::<u32, F>::new(s1, p1, s2, p2, counts[2], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(3);
            let e3 = LazyEdaBits::<u64, F>::new(s1, p1, s2, p2, counts[3], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(4);
            let e4 = LazyEdaBits::<u128, F>::new(s1, p1, s2, p2, counts[4], Vec::new(), party_id);
            let (ds1, dp1, ds2, dp2) = snaps[5];
            Ok(EdaBitsPool::new(
                e0,
                e1,
                e2,
                e3,
                e4,
                dabits::LazyDaBits::new(ds1, dp1, ds2, dp2, num_dabits, Vec::new(), party_id),
            ))
        }
        PartyID::ID2 => {
            // P2 advances daBit RNGs (to stay in sync) but doesn't use them.
            if num_dabits > 0 {
                let r = &mut rands[5];
                let slen2 = num_dabits;
                let slen1 = num_dabits;
                let mut s2 = vec![0u8; slen2];
                let mut s1 = vec![0u8; slen1];
                rayon::join(|| r.rng2.fill_bytes(&mut s2), || r.rng1.fill_bytes(&mut s1));
            }

            // Round 1: receive combined α₂ from P0 + s₁₂ from P1
            let total_eda = counts
                .iter()
                .enumerate()
                .map(|(i, &c)| c * [u8::K, u16::K, u32::K, u64::K, u128::K][i])
                .sum::<usize>();
            let total_recv = total_eda + num_dabits;

            let combined: Vec<F> = if total_recv > 0 {
                let _span = tracing::info_span!("resv_combined").entered();
                io.par_chunks(
                    0..total_recv,
                    None,
                    |_: Vec<usize>, ctx| -> eyre::Result<Vec<F>> {
                        Ok(ctx.network.recv_many::<F>(PartyID::ID0)?)
                    },
                )?
            } else {
                Vec::new()
            };
            debug_assert_eq!(combined.len(), total_recv);

            let s12_recv: Vec<F> = if num_dabits > 0 {
                let _span = tracing::info_span!("s12_recv").entered();
                io.network().recv_many(PartyID::ID1)?
            } else {
                Vec::new()
            };

            // Split combined into 5 edaBit α₂ slices + 1 daBit α₂
            let mut offset = 0;
            let mut eda_alphas: [Vec<F>; 5] = Default::default();
            for (idx, &c) in counts.iter().enumerate() {
                let k = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                let len = c * k;
                eda_alphas[idx] = combined[offset..offset + len].to_vec();
                offset += len;
            }
            let dabit_alpha2 = &combined[offset..offset + num_dabits];

            // Compute daBit s₂₀ and send to P0
            let dabit_stored = if num_dabits > 0 {
                // Re-derive theta from P2↔P1 seed (rng2 was advanced above, use snapshot).
                let (_, _, ds2_seed, ds2_pos) = snaps[5];
                let theta_buf = dabits::seek_and_generate(ds2_seed, ds2_pos, 0, num_dabits);

                let s20: Vec<F> = (0..num_dabits)
                    .into_par_iter()
                    .with_min_len(256)
                    .map(|i| {
                        let theta = (theta_buf[i] & 1) != 0;
                        let neg1_theta = if theta { -F::one() } else { F::one() };
                        neg1_theta * dabit_alpha2[i]
                    })
                    .collect();

                // Interleave s20 + s12 for LazyDaBits stored format (before send consumes s20)
                let mut stored = Vec::with_capacity(2 * num_dabits);
                for i in 0..num_dabits {
                    stored.push(s20[i]);
                    stored.push(s12_recv[i]);
                }

                // Round 2: P2 → P0 sends s₂₀ across fork channels
                io.par_chunks(
                    s20.into_par_iter(),
                    None,
                    |chunk: Vec<F>, ctx| -> eyre::Result<Vec<()>> {
                        ctx.network.send_many(PartyID::ID0, &chunk)?;
                        Ok(vec![])
                    },
                )?;

                stored
            } else {
                Vec::new()
            };

            let [a0, a1, a2, a3, a4] = eda_alphas;
            let mk = |i: usize| snaps[i];
            let (s1, p1, s2, p2) = mk(0);
            let e0 = LazyEdaBits::<u8, F>::new(s1, p1, s2, p2, counts[0], a0, party_id);
            let (s1, p1, s2, p2) = mk(1);
            let e1 = LazyEdaBits::<u16, F>::new(s1, p1, s2, p2, counts[1], a1, party_id);
            let (s1, p1, s2, p2) = mk(2);
            let e2 = LazyEdaBits::<u32, F>::new(s1, p1, s2, p2, counts[2], a2, party_id);
            let (s1, p1, s2, p2) = mk(3);
            let e3 = LazyEdaBits::<u64, F>::new(s1, p1, s2, p2, counts[3], a3, party_id);
            let (s1, p1, s2, p2) = mk(4);
            let e4 = LazyEdaBits::<u128, F>::new(s1, p1, s2, p2, counts[4], a4, party_id);
            let (ds1, dp1, ds2, dp2) = snaps[5];
            Ok(EdaBitsPool::new(
                e0,
                e1,
                e2,
                e3,
                e4,
                dabits::LazyDaBits::new(ds1, dp1, ds2, dp2, num_dabits, dabit_stored, party_id),
            ))
        }
    }
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
    let batch = EdaBitsBatch {
        gammas: vec![eda.gamma],
        alphas_flat: eda.alphas,
    };
    let mut out = ring_to_field_b2a_many::<T, F, N>(&[x_binary], &batch, io)?;
    Ok(out.remove(0))
}

/// Batched Protocol Π₂ B2A conversion.
///
/// For each binary Rep3 share `x`, converts to an arithmetic Rep3 field share
/// using the correlated random tuple in `batch`.
///
/// **Online communication:**
/// - Round 1: P0 broadcasts N packed ring elements (K bits each) to P1 and P2.
/// - Round 2: ShareConvert via `reshare_many` (one field element per conversion).
pub fn ring_to_field_b2a_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x_binary: &[Rep3RingShare<T>],
    batch: &EdaBitsBatch<T, F>,
    io: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    let n = x_binary.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    debug_assert_eq!(batch.gammas.len(), n);
    debug_assert_eq!(batch.alphas_flat.len(), n * T::K);

    // Precompute powers of 2 in Fp.
    let k = T::K;
    let pow2 = {
        let mut pow2 = Vec::with_capacity(k);
        let mut cur = F::one();
        for _ in 0..k {
            pow2.push(cur);
            cur = cur + cur;
        }
        pow2
    };

    // --- Round 1: P0 broadcasts masked values ---
    let ms: Vec<RingElement<T>> = if io.id == PartyID::ID0 {
        let ms: Vec<_> = x_binary
            .iter()
            .zip(&batch.gammas)
            .map(|(x, gamma)| x.a ^ x.b ^ *gamma)
            .collect();
        io.network.send_many(PartyID::ID1, &ms)?;
        io.network.send_many(PartyID::ID2, &ms)?;
        ms
    } else {
        io.network.recv_many(PartyID::ID0)?
    };

    // --- Local computation: fused v_component + masking → s_selfs ---
    // Pre-generate masking elements (sequential, RNG is stateful).
    let maskings: Vec<F> = (0..n).map(|_| io.masking_field_element::<F>()).collect();
    let party_id = io.id;

    // Fused parallel computation: v_component + masking in one pass.
    let s_selfs: Vec<F> = ms
        .par_iter()
        .zip(x_binary.par_iter())
        .zip(maskings.par_iter())
        .enumerate()
        .with_min_len(256)
        .map(|(idx, ((m, x), z))| {
            if party_id == PartyID::ID0 {
                return *z;
            }

            let beta = match party_id {
                PartyID::ID0 => unreachable!(),
                PartyID::ID1 => *m ^ x.a,
                PartyID::ID2 => *m ^ x.b,
            };

            let mut v = F::zero();
            let alpha_base = idx * k;
            for i in 0..k {
                let beta_bit = ((beta.0 >> i) & T::one()) == T::one();
                let alpha = batch.alphas_flat[alpha_base + i];
                let signed_alpha = if beta_bit { -alpha } else { alpha };
                v += pow2[i] * signed_alpha;
            }

            if party_id == PartyID::ID1 {
                v += F::from(Into::<u128>::into(beta.0));
            }

            v + *z
        })
        .collect();
    let s_prevs = io.network.reshare_many(&s_selfs)?;

    Ok(s_selfs
        .into_iter()
        .zip(s_prevs)
        .map(|(s_self, s_prev)| Rep3PrimeFieldShare::new(s_self, s_prev))
        .collect())
}

// ---------------------------------------------------------------------------
// EdaBitsPool: pre-generated edaBits/daBits for batched conversions
// ---------------------------------------------------------------------------

/// A pool of pre-generated edaBits and daBits for batched binary→field conversions.
///
/// EdaBits are stored lazily via [`LazyEdaBits`] (O(1) storage for P0/P1).
/// DaBits are stored via [`LazyDaBits`] (Cheng23 Π₁ partial-lazy).
pub struct EdaBitsPool<F: PrimeField> {
    edabits_u8: LazyEdaBits<u8, F>,
    edabits_u16: LazyEdaBits<u16, F>,
    edabits_u32: LazyEdaBits<u32, F>,
    edabits_u64: LazyEdaBits<u64, F>,
    edabits_u128: LazyEdaBits<u128, F>,
    dabits: LazyDaBits<F>,
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
            dabits: LazyDaBits::empty(party_id),
        }
    }

    /// Create a pool from lazy edaBits sources and lazy daBits.
    pub fn new(
        edabits_u8: LazyEdaBits<u8, F>,
        edabits_u16: LazyEdaBits<u16, F>,
        edabits_u32: LazyEdaBits<u32, F>,
        edabits_u64: LazyEdaBits<u64, F>,
        edabits_u128: LazyEdaBits<u128, F>,
        dabits: LazyDaBits<F>,
    ) -> Self {
        Self {
            edabits_u8,
            edabits_u16,
            edabits_u32,
            edabits_u64,
            edabits_u128,
            dabits,
        }
    }

    /// Drain `n` daBit tuples (Cheng23 Π₁) from the lazy source.
    #[tracing::instrument(skip(self))]
    pub fn take_dabits(&mut self, n: usize) -> DaBitBatch<F> {
        self.dabits.take_batch(n)
    }

    pub fn remaining_dabits(&self) -> usize {
        self.dabits.remaining()
    }

    pub fn is_empty(&self) -> bool {
        self.edabits_u8.remaining() == 0
            && self.edabits_u16.remaining() == 0
            && self.edabits_u32.remaining() == 0
            && self.edabits_u64.remaining() == 0
            && self.edabits_u128.remaining() == 0
            && self.dabits.remaining() == 0
    }

    /// Generic edaBits drain as flat batch, dispatched by `TypeId`.
    ///
    /// Panics if `T` is not one of u8, u16, u32, u64, u128.
    #[tracing::instrument(skip(self))]
    pub fn take_edabits<T: IntRing2k>(&mut self, n: usize) -> EdaBitsBatch<T, F>
    where
        Standard: Distribution<T>,
    {
        use std::any::TypeId;
        // Safety: We transmute between EdaBitsBatch<concrete, F> and EdaBitsBatch<T, F>
        // only when TypeId confirms T == concrete. The struct layout is
        // identical for the same concrete type, so the transmute is a no-op.
        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<u8>() {
            let v = self.edabits_u8.take_batch(n);
            // SAFETY: T == u8 confirmed by TypeId check.
            unsafe { std::mem::transmute::<EdaBitsBatch<u8, F>, EdaBitsBatch<T, F>>(v) }
        } else if tid == TypeId::of::<u16>() {
            let v = self.edabits_u16.take_batch(n);
            unsafe { std::mem::transmute::<EdaBitsBatch<u16, F>, EdaBitsBatch<T, F>>(v) }
        } else if tid == TypeId::of::<u32>() {
            let v = self.edabits_u32.take_batch(n);
            unsafe { std::mem::transmute::<EdaBitsBatch<u32, F>, EdaBitsBatch<T, F>>(v) }
        } else if tid == TypeId::of::<u64>() {
            let v = self.edabits_u64.take_batch(n);
            unsafe { std::mem::transmute::<EdaBitsBatch<u64, F>, EdaBitsBatch<T, F>>(v) }
        } else if tid == TypeId::of::<u128>() {
            let v = self.edabits_u128.take_batch(n);
            unsafe { std::mem::transmute::<EdaBitsBatch<u128, F>, EdaBitsBatch<T, F>>(v) }
        } else {
            panic!("EdaBitsPool::take_edabits: unsupported ring type");
        }
    }

    /// Write all lazy sources to `dir`.
    #[tracing::instrument(skip_all, name = "Preprocessing::save")]
    pub fn save(&self, dir: &std::path::Path) -> std::io::Result<()> {
        self.edabits_u8.save(dir)?;
        self.edabits_u16.save(dir)?;
        self.edabits_u32.save(dir)?;
        self.edabits_u64.save(dir)?;
        self.edabits_u128.save(dir)?;
        self.dabits.save(dir)?;
        std::result::Result::Ok(())
    }

    /// Load all lazy sources from `dir`.
    pub fn load(dir: &std::path::Path, party_id: PartyID) -> std::io::Result<Self> {
        std::result::Result::Ok(Self {
            edabits_u8: LazyEdaBits::<u8, F>::load(dir, party_id)?,
            edabits_u16: LazyEdaBits::<u16, F>::load(dir, party_id)?,
            edabits_u32: LazyEdaBits::<u32, F>::load(dir, party_id)?,
            edabits_u64: LazyEdaBits::<u64, F>::load(dir, party_id)?,
            edabits_u128: LazyEdaBits::<u128, F>::load(dir, party_id)?,
            dabits: LazyDaBits::<F>::load(dir, party_id)?,
        })
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
        let outputs: [EdaBitsBatch<u64, Fr>; 3] = run_rep3_local_test_with_coordinator(
            1,
            |i| i,
            || (),
            |party_idx, mut io_ctx| {
                assert_eq!(usize::from(io_ctx.party_idx()), party_idx);
                let mut lazy = random_edabits_lazy::<u64, Fr, _>(NUM, &mut io_ctx)?;
                Ok(lazy.take_batch(NUM))
            },
            |(), _net| Ok(()),
        )
        .0;

        let k = u64::K;
        // P0 knows gamma. P0 and P1 hold alpha_1, P2 holds alpha_2.
        // Check: alpha_1 + alpha_2 = F::from(gamma_bit) for each bit.
        for i in 0..NUM {
            let gamma = outputs[0].gammas[i].0;
            for b in 0..k {
                let gamma_bit = ((gamma >> b) & 1u64) == 1u64;
                let alpha_1 = outputs[0].alphas_flat[i * k + b];
                let alpha_2 = outputs[2].alphas_flat[i * k + b];
                assert_eq!(
                    alpha_1 + alpha_2,
                    Fr::from(gamma_bit as u64),
                    "lazy alpha mismatch at value {i}, bit {b}"
                );
                // P1 also holds alpha_1
                assert_eq!(outputs[1].alphas_flat[i * k + b], alpha_1);
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
                let batch = lazy.take_batch(NUM);
                ring_to_field_b2a_many::<u64, Fr, _>(&x_shares, &batch, io_ctx.main())
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
                let batch = EdaBitsBatch {
                    gammas: edas.iter().map(|e| e.gamma).collect(),
                    alphas_flat: edas.into_iter().flat_map(|e| e.alphas).collect(),
                };
                ring_to_field_b2a_many::<u64, Fr, _>(&x_shares, &batch, io_ctx.main())
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
                    let batch = lazy_u32.take_batch(NUM);
                    ring_to_field_b2a_many::<u32, Fr, _>(&id_sh, &batch, io)?
                };

                // Call 2: u16 left
                let left = {
                    let batch = lazy_u16.take_batch(NUM);
                    ring_to_field_b2a_many::<u16, Fr, _>(&left_sh, &batch, io)?
                };

                // Call 3: u16 right
                let right = {
                    let batch = lazy_u16.take_batch(NUM);
                    ring_to_field_b2a_many::<u16, Fr, _>(&right_sh, &batch, io)?
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
                let _discard = lazy.take_batch(BATCH1);

                // Now take batch2 at cursor=BATCH1 (odd) — this tests the word alignment fix
                let batch = lazy.take_batch(BATCH2);
                ring_to_field_b2a_many::<u16, Fr, _>(&x_shares, &batch, io_ctx.main())
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

    /// Helper: run preprocess + B2A + bit-inject roundtrip with a given pool builder.
    fn preprocess_roundtrip_impl(use_batched: bool) {
        use crate::protocols::rep3_ring::dabits;

        const NUM_U64: usize = 8;
        const NUM_DABITS: usize = 16;

        let mut rng = ChaCha20Rng::seed_from_u64(0xB001_0001);

        // Random u64 values for B2A
        let xs: Vec<u64> = (0..NUM_U64).map(|_| rng.next_u64()).collect();
        let x_bin_shares: [Vec<Rep3RingShare<u64>>; 3] = {
            let per = xs
                .iter()
                .map(|&x| share_ring_element_binary::<u64, _>(RingElement(x), &mut rng))
                .collect::<Vec<_>>();
            std::array::from_fn(|pid| per.iter().map(|s| s[pid]).collect())
        };

        // Random bits for bit injection
        let bits: Vec<bool> = (0..NUM_DABITS).map(|_| (rng.next_u32() & 1) == 1).collect();
        let bit_shares: [Vec<Rep3RingShare<RingBit>>; 3] = {
            let per = bits
                .iter()
                .map(|&b| share_ring_element::<RingBit, _>(RingElement(RingBit::new(b)), &mut rng))
                .collect::<Vec<_>>();
            std::array::from_fn(|pid| per.iter().map(|s| s[pid]).collect())
        };

        let outputs: [(Vec<Rep3PrimeFieldShare<Fr>>, Vec<Rep3PrimeFieldShare<Fr>>); 3] =
            run_rep3_local_test_with_coordinator(
                1,
                |i| (x_bin_shares[i].clone(), bit_shares[i].clone()),
                || (),
                move |(x_sh, bit_sh): (Vec<Rep3RingShare<u64>>, Vec<Rep3RingShare<RingBit>>),
                      mut io_ctx| {
                    let mut pool = if use_batched {
                        preprocess_pool_batched::<Fr, _>(
                            [0, 0, 0, NUM_U64, 0],
                            NUM_DABITS,
                            &mut io_ctx,
                        )?
                    } else {
                        preprocess_pool::<Fr, _>([0, 0, 0, NUM_U64, 0], NUM_DABITS, &mut io_ctx)?
                    };

                    // B2A via edaBits
                    let batch = pool.take_edabits::<u64>(NUM_U64);
                    let b2a = ring_to_field_b2a_many::<u64, Fr, _>(&x_sh, &batch, io_ctx.main())?;

                    // Bit inject via daBits
                    let dbatch = pool.take_dabits(NUM_DABITS);
                    let inj =
                        dabits::bit_inject_field_many::<Fr, _>(&bit_sh, &dbatch, io_ctx.main())?;

                    Ok((b2a, inj))
                },
                |(): (), _net| Ok(()),
            )
            .0;

        // Verify B2A
        let b2a_combined = combine_field_elements(&outputs[0].0, &outputs[1].0, &outputs[2].0);
        let b2a_expected: Vec<Fr> = xs.iter().map(|&x| Fr::from(x)).collect();
        assert_eq!(b2a_combined, b2a_expected, "preprocess B2A mismatch");

        // Verify bit injection
        let inj_combined = combine_field_elements(&outputs[0].1, &outputs[1].1, &outputs[2].1);
        let inj_expected: Vec<Fr> = bits.iter().map(|&b| Fr::from(b as u64)).collect();
        assert_eq!(inj_combined, inj_expected, "preprocess bit inject mismatch");
    }

    #[test]
    fn preprocess_pool_roundtrip() {
        preprocess_roundtrip_impl(false);
    }

    #[test]
    fn preprocess_pool_batched_roundtrip() {
        preprocess_roundtrip_impl(true);
    }
}
