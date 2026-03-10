//! edaBits helpers for Rep3 over rings.
//!
//! This module provides an opt-in conversion primitive to translate an
//! arithmetic Rep3 sharing over `Z_{2^K}` into an arithmetic Rep3 sharing over a
//! prime field `Fp`, using an edaBits mask that links the same random `r` across
//! both domains.

use super::backing_store;
use super::dabits::{DaBitBatch, LazyDaBits};
use crate::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker, Rep3RawFieldTransport};
use crate::protocols::rep3::{
    PartyID, Rep3PrimeFieldShare, arithmetic as rep3_arith,
    network::{IoContext, Rep3Network},
};
use crate::protocols::rep3_ring::arithmetic as rep3_ring_arith;
use eyre::Ok;
use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3_ring::ring::u66::U66;
use mpc_types::protocols::rep3_ring::{
    Rep3RingShare,
    ring::{int_ring::IntRing2k, ring_impl::RingElement},
};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::path::Path;
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
    pub(crate) total: usize,
    /// Bytes per field element: ceil(MODULUS_BIT_SIZE / 8).
    field_bytes: usize,
    /// Consumption cursor: next edaBit index to produce.
    cursor: usize,
    /// P2-only storage: flat alpha_2 values (length = total * T::K).
    /// Empty for P0/P1.  May be backed by a memory-mapped file.
    pub(crate) alpha2_flat: backing_store::BackingStore<F>,
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
        Self::new_with_store(
            seed1,
            pos1,
            seed2,
            pos2,
            total,
            backing_store::BackingStore::from_vec(alpha2_flat),
            party_id,
        )
    }

    pub(crate) fn new_with_store(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        total: usize,
        alpha2_flat: backing_store::BackingStore<F>,
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
            meta_path: None,
            _phantom: PhantomData,
        }
    }

    /// Number of remaining edaBits that can be produced.
    pub fn remaining(&self) -> usize {
        self.total - self.cursor
    }

    /// Reset the consumption cursor to 0 in `reuse-preproc` mode.
    ///
    /// This is intended for benchmark harnesses that run multiple proofs in one process and
    /// want to re-use the same persisted preprocessing pool across iterations.
    #[cfg(feature = "reuse-preproc")]
    pub(crate) fn reset_cursor_for_reuse(&mut self) {
        self.cursor = 0;
    }

    /// Drain `n` edaBits as a flat `EdaBitsBatch` with zero per-edaBit allocations.
    ///
    /// Two allocations total: `gammas` (len n) + `alphas_flat` (len n*K).
    pub fn take_batch(&mut self, n: usize) -> eyre::Result<EdaBitsBatch<T, F>> {
        eyre::ensure!(
            self.cursor + n <= self.total,
            "LazyEdaBits<u{}>: need {n}, have {} (cursor={}, total={})",
            T::K,
            self.remaining(),
            self.cursor,
            self.total
        );

        if n == 0 {
            return Ok(EdaBitsBatch {
                gammas: Vec::new(),
                alphas_flat: Vec::new(),
            });
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
            let alphas_flat = {
                #[cfg(feature = "reuse-preproc")]
                {
                    self.alpha2_flat
                        .read_reuse(flat_start, flat_end)
                        .unwrap_or_else(|e| {
                            panic!(
                                "LazyEdaBits(P2): read_reuse({flat_start}..{flat_end}) failed: {e}"
                            );
                        })
                }
                #[cfg(not(feature = "reuse-preproc"))]
                {
                    self.alpha2_flat
                        .read_consume(flat_start, flat_end)
                        .unwrap_or_else(|e| {
                            panic!("LazyEdaBits(P2): read_consume({flat_start}..{flat_end}) failed: {e}");
                        })
                }
            };
            let gammas = vec![RingElement(T::zero()); n];
            self.cursor += n;
            self.persist_cursor();
            self.alpha2_flat.consume(flat_start, flat_end);
            return Ok(EdaBitsBatch {
                gammas,
                alphas_flat,
            });
        }

        // P0/P1: regenerate from RNG seeds.
        //
        // Per-item interleaved layout in rng1 (P0↔P1 stream):
        //   [γ₀(t_B) α_{0,0}(fb)..α_{0,k-1}(fb) | γ₁ α_{1,0}..α_{1,k-1} | ...]
        // stride = t_bytes + k * fb per item.  Offsets are independent of `total`.
        let stride = t_bytes + k * fb;
        let item_byte_offset = cursor_base * stride;
        let interleaved_bytes = n * stride;

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

        // rng2 (P0↔P2) only carries gammas — offset = cursor * t_bytes (unchanged).
        let g2_gamma_offset = cursor_base * t_bytes;
        let g2_gamma_bytes = n * t_bytes;

        let (interleaved, g2_bytes);
        if party_id == PartyID::ID0 {
            let _span = tracing::trace_span!("gen_gamma_alpha").entered();
            let (s1, s2) = rayon::join(
                || seek_and_generate(self.seed1, self.pos1, item_byte_offset, interleaved_bytes),
                || seek_and_generate(self.seed2, self.pos2, g2_gamma_offset, g2_gamma_bytes),
            );
            interleaved = s1;
            g2_bytes = s2;
        } else {
            // P1: same interleaved stream from seed2 (P1↔P0 = P0↔P1 shared stream).
            let _span = tracing::trace_span!("gen_alpha").entered();
            interleaved =
                seek_and_generate(self.seed2, self.pos2, item_byte_offset, interleaved_bytes);
            g2_bytes = Vec::new();
        }

        // Build flat arrays from per-item interleaved layout.
        let _span = tracing::trace_span!("build_batch").entered();
        let gammas: Vec<RingElement<T>> = if party_id == PartyID::ID0 {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let g1_off = i * stride;
                    let g2_off = i * t_bytes;
                    let g1_val = T::from_le_bytes(&interleaved[g1_off..g1_off + t_bytes]);
                    let g2_val = T::from_le_bytes(&g2_bytes[g2_off..g2_off + t_bytes]);
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
                let item = idx / k;
                let bit = idx % k;
                let a_start = item * stride + t_bytes + bit * fb;
                let lo = u64::from_le_bytes(interleaved[a_start..a_start + 8].try_into().unwrap());
                let hi =
                    u64::from_le_bytes(interleaved[a_start + 8..a_start + 16].try_into().unwrap());
                F::from((hi as u128) << 64 | lo as u128)
            })
            .collect();

        self.cursor += n;
        Ok(EdaBitsBatch {
            gammas,
            alphas_flat,
        })
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

        // Data file first (page-cache write, no fsync), then meta with fsync.
        // Meta fsync is the durability barrier: if it succeeds, data is at least
        // in the kernel page cache and will be written to disk before meta.
        if !self.alpha2_flat.is_empty() {
            let data_path = dir.join(format!("edabits_{suffix}.alpha2"));
            self.alpha2_flat.save_to_file(&data_path)?;
        }
        backing_store::write_meta(
            &dir.join(format!("edabits_{suffix}.meta")),
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

    /// Return the RNG seeds and positions for this lazy source.
    pub(crate) fn extension_seeds(
        &self,
    ) -> ([u8; crate::SEED_SIZE], u128, [u8; crate::SEED_SIZE], u128) {
        (self.seed1, self.pos1, self.seed2, self.pos2)
    }

    /// Current total number of items in this pool.
    pub(crate) fn total(&self) -> usize {
        self.total
    }

    /// Extend this pool by `deficit` additional items.
    ///
    /// For P2: appends the received alpha2 extension to stored backing.
    /// For P0/P1: only bumps `total` (regeneration is seed-based).
    pub(crate) fn apply_extension(&mut self, deficit: usize, alpha2_ext: Vec<F>) {
        if deficit == 0 {
            return;
        }
        if !alpha2_ext.is_empty() {
            self.alpha2_flat.extend(&alpha2_ext);
        }
        self.total += deficit;
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

    // Per-item interleaved layout in rng1:
    //   [γ₀(t_B) α_{0,0}(fb)..α_{0,k-1}(fb) | γ₁ α_{1,0}..α_{1,k-1} | ...]
    // stride = t_bytes + k * field_bytes.
    let stride = t_bytes + k * field_bytes;
    let gamma_total_bytes = num * t_bytes;

    // P0 → P2: send alpha_2 = F::from(gamma_bit) - alpha_1.
    // Only P0 needs RNG bytes; P1/P2 skip generation entirely.
    if party_id == PartyID::ID0 {
        let _span = tracing::trace_span!("gen_rng_bytes").entered();
        // rng1: interleaved gamma+alpha per item. rng2: gamma only.
        let (all_bytes1, g2_bytes) = {
            let mut a = vec![0u8; num * stride];
            let mut b = vec![0u8; gamma_total_bytes];
            rayon::join(
                || eda_rand.rng1.fill_bytes(&mut a),
                || eda_rand.rng2.fill_bytes(&mut b),
            );
            (a, b)
        };
        drop(_span);

        let _span = tracing::trace_span!("compute_send_alpha2").entered();
        let mut alpha_2_all = vec![F::zero(); num * k];
        alpha_2_all
            .par_chunks_mut(k)
            .enumerate()
            .with_min_len(256)
            .for_each(|(i, chunk)| {
                let base = i * stride;
                let g1_val = T::from_le_bytes(&all_bytes1[base..base + t_bytes]);
                let g2_val = T::from_le_bytes(&g2_bytes[i * t_bytes..(i + 1) * t_bytes]);
                let gamma = g1_val ^ g2_val;
                for j in 0..k {
                    let a_start = base + t_bytes + j * field_bytes;
                    let lo =
                        u64::from_le_bytes(all_bytes1[a_start..a_start + 8].try_into().unwrap());
                    let hi = u64::from_le_bytes(
                        all_bytes1[a_start + 8..a_start + 16].try_into().unwrap(),
                    );
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

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn preproc_max_msg_mb() -> usize {
    env_usize("PREPROC_MAX_MSG_MB", 2)
}

fn preproc_store_batch_mb() -> usize {
    env_usize("PREPROC_STORE_BATCH_MB", 16)
}

fn preproc_lanes() -> usize {
    env_usize("PREPROC_LANES", 8)
}

fn preproc_segment_mb() -> usize {
    env_usize("PREPROC_SEGMENT_MB", 64)
}

fn configured_transport_lanes() -> usize {
    std::env::var("MPC_QUIC_CONN_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .or_else(|| {
            std::env::var("NETWORK_FORKS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v > 0)
        })
        .unwrap_or(8)
}

fn preproc_max_elems_per_msg<F: PrimeField>() -> usize {
    let mb = preproc_max_msg_mb();
    let bytes = mb.saturating_mul(1024 * 1024);
    let elem = std::mem::size_of::<F>();
    if elem == 0 {
        1
    } else {
        bytes.div_ceil(elem).max(1)
    }
}

fn preproc_store_batch_elems<F: PrimeField>() -> usize {
    let mb = preproc_store_batch_mb();
    let bytes = mb.saturating_mul(1024 * 1024);
    let elem = std::mem::size_of::<F>();
    if elem == 0 {
        1
    } else {
        bytes.div_ceil(elem).max(1)
    }
}

fn preproc_segment_elems<F: PrimeField>() -> usize {
    let mb = preproc_segment_mb();
    let bytes = mb.saturating_mul(1024 * 1024);
    let elem = std::mem::size_of::<F>();
    if elem == 0 {
        1
    } else {
        bytes.div_ceil(elem).max(1)
    }
}

#[derive(Clone, Copy)]
struct SegmentPlan {
    start_item: usize,
    items: usize,
}

fn build_segment_plans(total_items: usize, segment_items: usize) -> Vec<SegmentPlan> {
    if total_items == 0 {
        return Vec::new();
    }
    let segment_items = segment_items.max(1);
    let mut start_item = 0usize;
    let mut out = Vec::with_capacity(total_items.div_ceil(segment_items));
    while start_item < total_items {
        let items = (total_items - start_item).min(segment_items);
        out.push(SegmentPlan { start_item, items });
        start_item += items;
    }
    out
}

fn par_segment_plans<N, MapFn>(
    forks: &mut [IoContext<N>],
    plans: Vec<SegmentPlan>,
    map: MapFn,
) -> eyre::Result<()>
where
    N: Rep3NetworkWorker,
    MapFn: Fn(SegmentPlan, &mut IoContext<N>) -> eyre::Result<()> + Sync + Send,
{
    let plans_len = plans.len();
    if plans_len == 0 {
        return Ok(());
    }

    let fork_count = forks.len().min(plans_len);
    if fork_count == 0 {
        return Ok(());
    }
    if fork_count == 1 {
        for plan in plans {
            map(plan, &mut forks[0])?;
        }
        return Ok(());
    }

    (0..fork_count)
        .into_par_iter()
        .zip(forks[..fork_count].par_iter_mut())
        .map(|(fork_idx, mut ctx)| {
            let start = (fork_idx * plans_len) / fork_count;
            let end = ((fork_idx + 1) * plans_len) / fork_count;
            for plan in &plans[start..end] {
                map(*plan, &mut ctx)?;
            }
            std::result::Result::<(), eyre::Report>::Ok(())
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    Ok(())
}

enum RawFieldChunk<F> {
    Borrowed { bytes: Vec<u8>, elems: usize },
    Owned(Vec<F>),
}

impl<F: PrimeField> RawFieldChunk<F> {
    fn as_slice(&self) -> &[F] {
        match self {
            Self::Borrowed { bytes, elems } => {
                const { backing_store::assert_field_layout::<F>() };
                let ptr = bytes.as_ptr();
                let align = std::mem::align_of::<F>();
                debug_assert_eq!(bytes.len(), elems * std::mem::size_of::<F>());
                debug_assert_eq!((ptr as usize) % align.max(1), 0);
                unsafe { std::slice::from_raw_parts(ptr.cast::<F>(), *elems) }
            }
            Self::Owned(vec) => vec.as_slice(),
        }
    }
}

fn recv_field_chunk_ctx<F: PrimeField, N: Rep3Network + Rep3RawFieldTransport>(
    from: PartyID,
    elems: usize,
    io: &mut IoContext<N>,
) -> eyre::Result<RawFieldChunk<F>> {
    let bytes = io.network.recv_field_bytes_raw::<F>(from, elems)?;
    let align = std::mem::align_of::<F>();
    if (bytes.as_ptr() as usize) % align.max(1) == 0 {
        return Ok(RawFieldChunk::Borrowed { bytes, elems });
    }
    Ok(RawFieldChunk::Owned(field_vec_from_raw_bytes::<F>(&bytes)?))
}

fn recv_field_chunk<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    from: PartyID,
    elems: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<RawFieldChunk<F>> {
    recv_field_chunk_ctx(from, elems, io.main())
}

fn send_field_superchunk_ctx<F: PrimeField, N: Rep3Network + Rep3RawFieldTransport>(
    mut data: Vec<F>,
    target: PartyID,
    io: &mut IoContext<N>,
    max_msg_elems: usize,
) -> eyre::Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let _span = tracing::trace_span!(
        "send_field_superchunk",
        party_id = ?io.id,
        target = ?target,
        elems = data.len(),
        bytes = data.len() * std::mem::size_of::<F>(),
    )
    .entered();
    if data.len() <= max_msg_elems {
        io.network.send_field_vec_raw(target, data)?;
        return Ok(());
    }

    while data.len() > max_msg_elems {
        let tail = data.split_off(max_msg_elems);
        io.network.send_field_vec_raw(target, data)?;
        data = tail;
    }
    io.network.send_field_vec_raw(target, data)?;
    Ok(())
}

fn send_field_superchunk<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    data: Vec<F>,
    target: PartyID,
    io: &mut IoContextPool<N>,
    max_msg_elems: usize,
) -> eyre::Result<()> {
    send_field_superchunk_ctx(data, target, io.main(), max_msg_elems)
}

fn recv_field_bytes_collect_ctx<F: PrimeField, N: Rep3Network + Rep3RawFieldTransport>(
    from: PartyID,
    expected_len: usize,
    io: &mut IoContext<N>,
    max_msg_elems: usize,
) -> eyre::Result<Vec<u8>> {
    if expected_len == 0 {
        return Ok(Vec::new());
    }

    let num_msgs = expected_len.div_ceil(max_msg_elems.max(1));
    let _span = tracing::trace_span!(
        "recv_field_messages_collect",
        party_id = ?io.id,
        from = ?from,
        elems = expected_len,
        bytes = expected_len * std::mem::size_of::<F>(),
        num_msgs
    )
    .entered();

    let mut out = Vec::with_capacity(expected_len * std::mem::size_of::<F>());
    let mut received = 0usize;
    while received < expected_len {
        let chunk_elems = (expected_len - received).min(max_msg_elems.max(1));
        let recv = io.network.recv_field_bytes_raw::<F>(from, chunk_elems)?;
        out.extend_from_slice(&recv);
        received += chunk_elems;
    }
    debug_assert_eq!(received, expected_len);
    Ok(out)
}

fn recv_field_messages_collect_ctx<F: PrimeField, N: Rep3Network + Rep3RawFieldTransport>(
    from: PartyID,
    expected_len: usize,
    io: &mut IoContext<N>,
    max_msg_elems: usize,
) -> eyre::Result<Vec<F>> {
    if expected_len == 0 {
        return Ok(Vec::new());
    }

    if expected_len <= max_msg_elems.max(1) {
        return Ok(io
            .network
            .recv_field_vec_raw_owned::<F>(from, expected_len)?);
    }

    let bytes = recv_field_bytes_collect_ctx::<F, _>(from, expected_len, io, max_msg_elems)?;
    field_vec_from_raw_bytes::<F>(&bytes)
}

fn recv_field_messages_collect<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    from: PartyID,
    expected_len: usize,
    io: &mut IoContextPool<N>,
    max_msg_elems: usize,
) -> eyre::Result<Vec<F>> {
    recv_field_messages_collect_ctx(from, expected_len, io.main(), max_msg_elems)
}

fn recv_field_messages_into_store<
    F: PrimeField + Copy,
    N: Rep3NetworkWorker + Rep3RawFieldTransport,
>(
    from: PartyID,
    start_elem: usize,
    expected_len: usize,
    io: &mut IoContextPool<N>,
    max_msg_elems: usize,
    store: &mut backing_store::BackingStore<F>,
) -> eyre::Result<()> {
    if expected_len == 0 {
        return Ok(());
    }

    let elem_size = std::mem::size_of::<F>();
    let num_msgs = expected_len.div_ceil(max_msg_elems.max(1));
    let _span = tracing::trace_span!(
        "recv_field_messages_into_store",
        party_id = ?io.party_id(),
        from = ?from,
        start_elem,
        elems = expected_len,
        bytes = expected_len * elem_size,
        num_msgs
    )
    .entered();

    let buf_elems = max_msg_elems.max(1).min(expected_len);
    let mut buf = vec![0u8; buf_elems * elem_size];
    let mut write_elem_offset = start_elem;
    let mut received = 0usize;
    while received < expected_len {
        let chunk_elems = (expected_len - received).min(buf_elems);
        let chunk_bytes = chunk_elems * elem_size;
        io.network()
            .recv_field_bytes_bulk_into::<F>(from, &mut buf[..chunk_bytes])?;
        write_bytes_to_store(store, write_elem_offset, &buf[..chunk_bytes])?;
        write_elem_offset += chunk_elems;
        received += chunk_elems;
    }
    debug_assert_eq!(received, expected_len);

    Ok(())
}

fn receive_field_store_batch<F: PrimeField + Copy, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    from: PartyID,
    start_elem: usize,
    expected_len: usize,
    io: &mut IoContextPool<N>,
    max_msg_elems: usize,
    store: &mut backing_store::BackingStore<F>,
) -> eyre::Result<()> {
    recv_field_messages_into_store(from, start_elem, expected_len, io, max_msg_elems, store)
}

fn receive_field_writer_batch_ctx<F: PrimeField + Copy, N: Rep3Network + Rep3RawFieldTransport>(
    from: PartyID,
    start_elem: usize,
    expected_len: usize,
    io: &mut IoContext<N>,
    max_msg_elems: usize,
    writer: &backing_store::FileBackedWriter<F>,
) -> eyre::Result<()> {
    recv_field_messages_into_writer_ctx(from, start_elem, expected_len, io, max_msg_elems, writer)
}

fn recv_field_messages_into_writer_ctx<
    F: PrimeField + Copy,
    N: Rep3Network + Rep3RawFieldTransport,
>(
    from: PartyID,
    start_elem: usize,
    expected_len: usize,
    io: &mut IoContext<N>,
    max_msg_elems: usize,
    writer: &backing_store::FileBackedWriter<F>,
) -> eyre::Result<()> {
    if expected_len == 0 {
        return Ok(());
    }

    let elem_size = std::mem::size_of::<F>();
    let num_msgs = expected_len.div_ceil(max_msg_elems.max(1));
    let _span = tracing::trace_span!(
        "recv_field_messages_into_store",
        party_id = ?io.id,
        from = ?from,
        start_elem,
        elems = expected_len,
        bytes = expected_len * elem_size,
        num_msgs
    )
    .entered();

    let buf_elems = max_msg_elems.max(1).min(expected_len);
    let mut buf = vec![0u8; buf_elems * elem_size];
    let mut write_elem_offset = start_elem;
    let mut received = 0usize;
    while received < expected_len {
        let chunk_elems = (expected_len - received).min(buf_elems);
        let chunk_bytes = chunk_elems * elem_size;
        io.network
            .recv_field_bytes_bulk_into::<F>(from, &mut buf[..chunk_bytes])?;
        let _span = tracing::trace_span!(
            "append_to_store",
            start_elem = write_elem_offset,
            elems = chunk_elems,
            bytes = chunk_bytes
        )
        .entered();
        writer.write_bytes_at(write_elem_offset, &buf[..chunk_bytes])?;
        write_elem_offset += chunk_elems;
        received += chunk_elems;
    }
    debug_assert_eq!(received, expected_len);
    Ok(())
}

fn write_to_store<F: PrimeField + Copy>(
    store: &mut backing_store::BackingStore<F>,
    start_elem: usize,
    data: &[F],
) -> eyre::Result<()> {
    let _span = tracing::trace_span!(
        "append_to_store",
        start_elem,
        elems = data.len(),
        bytes = data.len() * std::mem::size_of::<F>()
    )
    .entered();
    store.write_at(start_elem, data)?;
    Ok(())
}

fn write_bytes_to_store<F: PrimeField + Copy>(
    store: &mut backing_store::BackingStore<F>,
    start_elem: usize,
    bytes: &[u8],
) -> eyre::Result<()> {
    let elem_size = std::mem::size_of::<F>();
    debug_assert_eq!(bytes.len() % elem_size, 0);
    let _span = tracing::trace_span!(
        "append_to_store",
        start_elem,
        elems = bytes.len() / elem_size,
        bytes = bytes.len()
    )
    .entered();
    store.write_bytes_at(start_elem, bytes)?;
    Ok(())
}

fn field_vec_from_raw_bytes<F: PrimeField>(bytes: &[u8]) -> eyre::Result<Vec<F>> {
    const { backing_store::assert_field_layout::<F>() };
    let elem_size = std::mem::size_of::<F>();
    if bytes.len() % elem_size != 0 {
        eyre::bail!(
            "raw field payload length {} is not divisible by element size {} for {}",
            bytes.len(),
            elem_size,
            std::any::type_name::<F>()
        );
    }
    let elems = bytes.len() / elem_size;
    if elems == 0 {
        return Ok(Vec::new());
    }

    let mut out: Vec<std::mem::MaybeUninit<F>> = Vec::with_capacity(elems);
    unsafe { out.set_len(elems) };
    let out_bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, bytes.len()) };
    out_bytes.copy_from_slice(bytes);
    let out: Vec<F> = unsafe { std::mem::transmute(out) };
    Ok(out)
}

fn read_file_backed_range<F: PrimeField + Copy>(
    file: &std::fs::File,
    start: usize,
    end: usize,
) -> std::io::Result<Vec<F>> {
    const { backing_store::assert_field_layout::<F>() };
    if start >= end {
        return std::result::Result::Ok(Vec::new());
    }
    let count = end - start;
    let elem_size = std::mem::size_of::<F>();
    let byte_offset = start * elem_size;
    let byte_len = count * elem_size;

    // SAFETY: same contract as BackingStore::read_file_backed_range.
    let mut out: Vec<std::mem::MaybeUninit<F>> = Vec::with_capacity(count);
    unsafe { out.set_len(count) };
    let out_bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len) };

    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(out_bytes, byte_offset as u64)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(out_bytes, byte_offset as u64)?;
    }
    #[cfg(not(any(unix, windows)))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file.try_clone()?;
        f.seek(SeekFrom::Start(byte_offset as u64))?;
        f.read_exact(out_bytes)?;
    }

    let out: Vec<F> = unsafe { std::mem::transmute(out) };
    std::result::Result::Ok(out)
}

/// Batched preprocessing into a persistence directory, using bounded-memory chunking.
///
/// This is intended for large-trace benchmark harnesses. It avoids allocating
/// monolithic `Vec<F>` buffers for all α₂ values at once, and for P2 it writes
/// α₂/stored data directly to file-backed stores under `dir`.
#[tracing::instrument(skip_all, name = "preprocess_pool")]
fn preprocess_pool_base<F, N>(
    dir: &Path,
    counts: [usize; 5], // [u8, u16, u32, u64, u128]
    num_dabits: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<PreprocessingPool<F>>
where
    F: PrimeField + Copy,
    N: Rep3NetworkWorker + Rep3RawFieldTransport,
{
    use super::dabits;
    use crate::protocols::rep3::rngs::Rep3Rand;

    std::fs::create_dir_all(dir)?;

    let party_id = io.party_id();
    let fb = usize::try_from(F::MODULUS_BIT_SIZE)
        .expect("u32 fits into usize")
        .div_ceil(8);
    let max_msg_elems = preproc_max_elems_per_msg::<F>();
    let store_batch_elems = preproc_store_batch_elems::<F>();
    let segment_elems = preproc_segment_elems::<F>();
    let active_edabit_lanes = preproc_lanes()
        .max(1)
        .min(io.max_forks().max(1))
        .min(configured_transport_lanes().max(1));
    let active_dabit_lanes = 1; // daBits benefit more from intra-chunk rayon parallelism than from lane splitting

    // Phase 1: Fork 6 Rep3Rands and snapshot seeds (local, no communication).
    let mut rands: [Rep3Rand; 6] = std::array::from_fn(|_| io.main().rngs.rand.fork());
    let snaps: [_; 6] = std::array::from_fn(|i| rands[i].snapshot());

    // Helper: compute edaBit α2 for a contiguous item range from seed snapshots.
    fn edabit_alpha2_seed_chunk<T: IntRing2k, Fp: PrimeField>(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        start_item: usize,
        num: usize,
        fb: usize,
        parallel: bool,
    ) -> Vec<Fp>
    where
        Standard: Distribution<T>,
    {
        if num == 0 {
            return Vec::new();
        }
        let t_bytes = std::mem::size_of::<T>();
        let k = T::K;
        let stride = t_bytes + k * fb;
        let _span = tracing::trace_span!(
            "edabit_alpha2_p0_chunk",
            ring_bits = T::K,
            items = num,
            start_item,
            elems = num * k,
            bytes = num * k * std::mem::size_of::<Fp>()
        )
        .entered();
        let all1 = dabits::seek_and_generate(seed1, pos1, start_item * stride, num * stride);
        let g2 = dabits::seek_and_generate(seed2, pos2, start_item * t_bytes, num * t_bytes);
        let mut out = vec![Fp::zero(); num * k];
        let fill_chunk = |(i, chunk): (usize, &mut [Fp])| {
            let base = i * stride;
            let g1v = T::from_le_bytes(&all1[base..base + t_bytes]);
            let g2v = T::from_le_bytes(&g2[i * t_bytes..(i + 1) * t_bytes]);
            let gamma = g1v ^ g2v;
            for j in 0..k {
                let s = base + t_bytes + j * fb;
                let alpha1 = dabits::parse_field::<Fp>(&all1, s);
                let gbit = ((gamma >> j) & T::one()) == T::one();
                chunk[j] = Fp::from(gbit as u64) - alpha1;
            }
        };
        if parallel {
            out.par_chunks_mut(k)
                .enumerate()
                .with_min_len(256)
                .for_each(fill_chunk);
        } else {
            out.chunks_mut(k).enumerate().for_each(fill_chunk);
        }
        out
    }

    fn dabit_alpha2_seed_chunk<Fp: PrimeField>(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        start_item: usize,
        num: usize,
        fb: usize,
        parallel: bool,
    ) -> Vec<Fp> {
        if num == 0 {
            return Vec::new();
        }
        let da_stride = 1 + 2 * fb;
        let s1 = dabits::seek_and_generate(seed1, pos1, start_item * da_stride, num * da_stride);
        let s2 = dabits::seek_and_generate(seed2, pos2, start_item, num);
        let compute = |i: usize| {
            let gbit = ((s1[i * da_stride] ^ s2[i]) & 1) != 0;
            let alpha1: Fp = dabits::parse_field(&s1, i * da_stride + 1);
            Fp::from(gbit as u64) - alpha1
        };
        if parallel {
            (0..num)
                .into_par_iter()
                .with_min_len(256)
                .map(compute)
                .collect()
        } else {
            (0..num).map(compute).collect()
        }
    }

    fn dabit_s12_seed_chunk<Fp: PrimeField>(
        theta_seed: [u8; crate::SEED_SIZE],
        theta_pos: u128,
        interleaved_seed: [u8; crate::SEED_SIZE],
        interleaved_pos: u128,
        start_item: usize,
        num: usize,
        fb: usize,
        parallel: bool,
    ) -> Vec<Fp> {
        if num == 0 {
            return Vec::new();
        }
        let da_stride = 1 + 2 * fb;
        let theta_bytes = dabits::seek_and_generate(theta_seed, theta_pos, start_item, num);
        let s2 = dabits::seek_and_generate(
            interleaved_seed,
            interleaved_pos,
            start_item * da_stride,
            num * da_stride,
        );
        let compute = |i: usize| {
            let theta = (theta_bytes[i] & 1) != 0;
            let neg1_theta = if theta { -Fp::one() } else { Fp::one() };
            let alpha1: Fp = dabits::parse_field(&s2, i * da_stride + 1);
            let r1_val: Fp = dabits::parse_field(&s2, i * da_stride + 1 + fb);
            neg1_theta * alpha1 - r1_val
        };
        if parallel {
            (0..num)
                .into_par_iter()
                .with_min_len(256)
                .map(compute)
                .collect()
        } else {
            (0..num).map(compute).collect()
        }
    }

    fn dabit_s20_from_theta_seed<Fp: PrimeField>(
        theta_seed: [u8; crate::SEED_SIZE],
        theta_pos: u128,
        start_item: usize,
        alpha2: &[Fp],
        parallel: bool,
    ) -> Vec<Fp> {
        if alpha2.is_empty() {
            return Vec::new();
        }
        let theta_buf = dabits::seek_and_generate(theta_seed, theta_pos, start_item, alpha2.len());
        let compute = |(alpha2_i, theta_b): (&Fp, &u8)| {
            let theta = (theta_b & 1) != 0;
            let neg1_theta = if theta { -Fp::one() } else { Fp::one() };
            neg1_theta * *alpha2_i
        };
        if parallel {
            alpha2
                .par_iter()
                .zip(theta_buf.par_iter())
                .map(compute)
                .collect()
        } else {
            alpha2.iter().zip(theta_buf.iter()).map(compute).collect()
        }
    }

    match party_id {
        PartyID::ID0 => {
            // Round 1: P0 → P2 sends edaBit α₂ streams per type, then daBit α₂.
            let mut send_edabit_type = |idx: usize, num: usize| {
                if num == 0 {
                    return Ok(());
                }
                let _span = tracing::info_span!("edabits_send_alphas", k = idx, n = num).entered();
                let k = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                let seeds = snaps[idx];
                let max_items_per_msg = (max_msg_elems / k).max(1);
                let segment_items = (segment_elems / k).max(max_items_per_msg);
                let plans = build_segment_plans(num, segment_items);
                let chunk_parallel = active_edabit_lanes == 1;
                if active_edabit_lanes > 1 && plans.len() > 1 {
                    par_segment_plans(io.forks(active_edabit_lanes), plans, |plan, ctx| {
                        let mut done = 0usize;
                        while done < plan.items {
                            let items = (plan.items - done).min(max_items_per_msg);
                            let start_item = plan.start_item + done;
                            let alpha2 = match idx {
                                0 => edabit_alpha2_seed_chunk::<u8, F>(
                                    seeds.0,
                                    seeds.1,
                                    seeds.2,
                                    seeds.3,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                ),
                                1 => edabit_alpha2_seed_chunk::<u16, F>(
                                    seeds.0,
                                    seeds.1,
                                    seeds.2,
                                    seeds.3,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                ),
                                2 => edabit_alpha2_seed_chunk::<u32, F>(
                                    seeds.0,
                                    seeds.1,
                                    seeds.2,
                                    seeds.3,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                ),
                                3 => edabit_alpha2_seed_chunk::<u64, F>(
                                    seeds.0,
                                    seeds.1,
                                    seeds.2,
                                    seeds.3,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                ),
                                4 => edabit_alpha2_seed_chunk::<u128, F>(
                                    seeds.0,
                                    seeds.1,
                                    seeds.2,
                                    seeds.3,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                ),
                                _ => unreachable!(),
                            };
                            debug_assert_eq!(alpha2.len(), items * k);
                            send_field_superchunk_ctx(alpha2, PartyID::ID2, ctx, max_msg_elems)?;
                            done += items;
                        }
                        Ok(())
                    })?;
                } else {
                    let mut done = 0usize;
                    while done < num {
                        let items = (num - done).min(max_items_per_msg);
                        let alpha2 = match idx {
                            0 => edabit_alpha2_seed_chunk::<u8, F>(
                                seeds.0,
                                seeds.1,
                                seeds.2,
                                seeds.3,
                                done,
                                items,
                                fb,
                                chunk_parallel,
                            ),
                            1 => edabit_alpha2_seed_chunk::<u16, F>(
                                seeds.0,
                                seeds.1,
                                seeds.2,
                                seeds.3,
                                done,
                                items,
                                fb,
                                chunk_parallel,
                            ),
                            2 => edabit_alpha2_seed_chunk::<u32, F>(
                                seeds.0,
                                seeds.1,
                                seeds.2,
                                seeds.3,
                                done,
                                items,
                                fb,
                                chunk_parallel,
                            ),
                            3 => edabit_alpha2_seed_chunk::<u64, F>(
                                seeds.0,
                                seeds.1,
                                seeds.2,
                                seeds.3,
                                done,
                                items,
                                fb,
                                chunk_parallel,
                            ),
                            4 => edabit_alpha2_seed_chunk::<u128, F>(
                                seeds.0,
                                seeds.1,
                                seeds.2,
                                seeds.3,
                                done,
                                items,
                                fb,
                                chunk_parallel,
                            ),
                            _ => unreachable!(),
                        };
                        debug_assert_eq!(alpha2.len(), items * k);
                        send_field_superchunk(alpha2, PartyID::ID2, io, max_msg_elems)?;
                        done += items;
                    }
                }
                Ok(())
            };

            // Collect active (non-zero) edaBit types.
            let active_types: Vec<(usize, usize)> = counts
                .iter()
                .copied()
                .enumerate()
                .filter(|(_i, c)| *c > 0)
                .collect();

            if active_types.len() > 1 && active_edabit_lanes >= active_types.len() {
                // Pipeline: each edaBit type runs on its own fork in parallel.
                let forks = io.forks(active_types.len());
                active_types
                    .into_par_iter()
                    .zip(forks.par_iter_mut())
                    .try_for_each(|((idx, num), ctx)| {
                        let _span =
                            tracing::info_span!("edabits_send_alphas", k = idx, n = num).entered();
                        let k: usize = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                        let seeds = snaps[idx];
                        let max_items_per_msg: usize = (max_msg_elems / k).max(1);
                        let mut done = 0usize;
                        while done < num {
                            let items = (num - done).min(max_items_per_msg);
                            let alpha2 = match idx {
                                0 => edabit_alpha2_seed_chunk::<u8, F>(
                                    seeds.0, seeds.1, seeds.2, seeds.3, done, items, fb, true,
                                ),
                                1 => edabit_alpha2_seed_chunk::<u16, F>(
                                    seeds.0, seeds.1, seeds.2, seeds.3, done, items, fb, true,
                                ),
                                2 => edabit_alpha2_seed_chunk::<u32, F>(
                                    seeds.0, seeds.1, seeds.2, seeds.3, done, items, fb, true,
                                ),
                                3 => edabit_alpha2_seed_chunk::<u64, F>(
                                    seeds.0, seeds.1, seeds.2, seeds.3, done, items, fb, true,
                                ),
                                4 => edabit_alpha2_seed_chunk::<u128, F>(
                                    seeds.0, seeds.1, seeds.2, seeds.3, done, items, fb, true,
                                ),
                                _ => unreachable!(),
                            };
                            debug_assert_eq!(alpha2.len(), items * k);
                            send_field_superchunk_ctx(alpha2, PartyID::ID2, ctx, max_msg_elems)?;
                            done += items;
                        }
                        eyre::Result::<()>::Ok(())
                    })?;
            } else {
                send_edabit_type(0, counts[0])?;
                send_edabit_type(1, counts[1])?;
                send_edabit_type(2, counts[2])?;
                send_edabit_type(3, counts[3])?;
                send_edabit_type(4, counts[4])?;
            }
            let _span =
                tracing::trace_span!("edabits_to_dabits_sync", party_id = ?party_id).entered();
            io.sync_with_parties()?;

            let (ds1, dp1, ds2, dp2) = snaps[5];
            let dabits_store_path = dir.join("dabits.stored");
            let mut dabits_store = backing_store::BackingStore::create_file_backed_sized(
                &dabits_store_path,
                num_dabits,
            )?;
            if num_dabits > 0 {
                let max_items_per_msg = max_msg_elems.max(1);
                let segment_items = segment_elems.max(max_items_per_msg);
                let _span = tracing::info_span!("dabits_stream_chunks", n = num_dabits).entered();
                if let Some(writer) = dabits_store.writer()? {
                    let plans = build_segment_plans(num_dabits, segment_items);
                    let chunk_parallel = active_dabit_lanes == 1;
                    if active_dabit_lanes > 1 && plans.len() > 1 {
                        par_segment_plans(io.forks(active_dabit_lanes), plans, |plan, ctx| {
                            let mut offset = 0usize;
                            while offset < plan.items {
                                let items = (plan.items - offset).min(max_items_per_msg);
                                let start_item = plan.start_item + offset;
                                let da_alpha2 = dabit_alpha2_seed_chunk::<F>(
                                    ds1,
                                    dp1,
                                    ds2,
                                    dp2,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                );
                                send_field_superchunk_ctx(
                                    da_alpha2,
                                    PartyID::ID2,
                                    ctx,
                                    max_msg_elems,
                                )?;
                                recv_field_messages_into_writer_ctx::<F, _>(
                                    PartyID::ID2,
                                    start_item,
                                    items,
                                    ctx,
                                    max_msg_elems,
                                    &writer,
                                )?;
                                offset += items;
                            }
                            Ok(())
                        })?;
                    } else {
                        let mut offset = 0usize;
                        while offset < num_dabits {
                            let items = (num_dabits - offset).min(max_items_per_msg);
                            let da_alpha2 = dabit_alpha2_seed_chunk::<F>(
                                ds1,
                                dp1,
                                ds2,
                                dp2,
                                offset,
                                items,
                                fb,
                                chunk_parallel,
                            );
                            send_field_superchunk(da_alpha2, PartyID::ID2, io, max_msg_elems)?;

                            recv_field_messages_into_store::<F, _>(
                                PartyID::ID2,
                                offset,
                                items,
                                io,
                                max_msg_elems,
                                &mut dabits_store,
                            )?;
                            offset += items;
                        }
                    }
                }
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

            let d =
                LazyDaBits::new_with_store(ds1, dp1, ds2, dp2, num_dabits, dabits_store, party_id);
            let pool = PreprocessingPool::new(party_id, e0, e1, e2, e3, e4, d);
            pool.save(dir)?;
            Ok(pool)
        }
        PartyID::ID1 => {
            // P1 only sends daBit s₁₂ to P2 (edaBits are local-only for P1).
            let _span =
                tracing::trace_span!("edabits_to_dabits_sync", party_id = ?party_id).entered();
            io.sync_with_parties()?;
            if num_dabits > 0 {
                let _span = tracing::info_span!("dabits_send_thetas", n = num_dabits).entered();
                let (theta_seed, theta_pos, interleaved_seed, interleaved_pos) = snaps[5];
                let max_items_per_msg = max_msg_elems.max(1);
                let segment_items = segment_elems.max(max_items_per_msg);
                let plans = build_segment_plans(num_dabits, segment_items);
                let chunk_parallel = active_dabit_lanes == 1;
                if active_dabit_lanes > 1 && plans.len() > 1 {
                    par_segment_plans(io.forks(active_dabit_lanes), plans, |plan, ctx| {
                        let mut offset = 0usize;
                        while offset < plan.items {
                            let items = (plan.items - offset).min(max_items_per_msg);
                            let start_item = plan.start_item + offset;
                            let s12 = dabit_s12_seed_chunk::<F>(
                                theta_seed,
                                theta_pos,
                                interleaved_seed,
                                interleaved_pos,
                                start_item,
                                items,
                                fb,
                                chunk_parallel,
                            );
                            send_field_superchunk_ctx(s12, PartyID::ID2, ctx, max_msg_elems)?;
                            offset += items;
                        }
                        Ok(())
                    })?;
                } else {
                    let mut offset = 0usize;
                    while offset < num_dabits {
                        let items = (num_dabits - offset).min(max_items_per_msg);
                        let s12 = dabit_s12_seed_chunk::<F>(
                            theta_seed,
                            theta_pos,
                            interleaved_seed,
                            interleaved_pos,
                            offset,
                            items,
                            fb,
                            chunk_parallel,
                        );
                        send_field_superchunk(s12, PartyID::ID2, io, max_msg_elems)?;
                        offset += items;
                    }
                }
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
            let d = LazyDaBits::new(ds1, dp1, ds2, dp2, num_dabits, Vec::new(), party_id);
            let pool = PreprocessingPool::new(party_id, e0, e1, e2, e3, e4, d);
            pool.save(dir)?;
            Ok(pool)
        }
        PartyID::ID2 => {
            // Round 1: receive edaBit α₂ streams from P0 and append to file-backed stores.
            let mut eda_stores_vec: Vec<backing_store::BackingStore<F>> = Vec::with_capacity(5);
            for idx in 0..5 {
                let suffix = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                let path = dir.join(format!("edabits_{suffix}.alpha2"));
                let total = counts[idx] * suffix;
                eda_stores_vec.push(backing_store::BackingStore::create_file_backed_sized(
                    &path, total,
                )?);
            }
            let mut eda_stores_iter = eda_stores_vec.into_iter();
            let mut eda_stores: [backing_store::BackingStore<F>; 5] = [
                eda_stores_iter
                    .next()
                    .expect("eda store vec must have len=5"),
                eda_stores_iter
                    .next()
                    .expect("eda store vec must have len=5"),
                eda_stores_iter
                    .next()
                    .expect("eda store vec must have len=5"),
                eda_stores_iter
                    .next()
                    .expect("eda store vec must have len=5"),
                eda_stores_iter
                    .next()
                    .expect("eda store vec must have len=5"),
            ];
            debug_assert!(eda_stores_iter.next().is_none());

            // Collect active (non-zero) edaBit types with their writers.
            let active_types: Vec<(usize, usize)> = counts
                .iter()
                .copied()
                .enumerate()
                .filter(|(_i, c)| *c > 0)
                .collect();

            if active_types.len() > 1 && active_edabit_lanes >= active_types.len() {
                // Pipeline: each edaBit type receives on its own fork in parallel.
                let writers: Vec<_> = active_types
                    .iter()
                    .map(|(idx, _)| eda_stores[*idx].writer())
                    .collect::<std::io::Result<Vec<_>>>()?;
                let forks = io.forks(active_types.len());
                active_types
                    .into_par_iter()
                    .zip(writers.into_par_iter())
                    .zip(forks.par_iter_mut())
                    .try_for_each(|(((idx, c), writer), ctx)| {
                        let _span =
                            tracing::info_span!("edabits_resv_store", k = idx, n = c).entered();
                        let k: usize = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                        let max_items_per_msg: usize = (max_msg_elems / k).max(1);
                        let store_batch_items: usize =
                            (store_batch_elems / k).max(max_items_per_msg);
                        if let Some(writer) = writer {
                            let mut done = 0usize;
                            while done < c {
                                let items = (c - done).min(store_batch_items);
                                recv_field_messages_into_writer_ctx::<F, _>(
                                    PartyID::ID0,
                                    done * k,
                                    items * k,
                                    ctx,
                                    max_msg_elems,
                                    &writer,
                                )?;
                                done += items;
                            }
                        }
                        eyre::Result::<()>::Ok(())
                    })?;
            } else {
                for (idx, &c) in counts.iter().enumerate() {
                    let _span = tracing::info_span!("edabits_resv_store", k = idx, n = c).entered();
                    let k = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                    let total_elems = c * k;
                    if total_elems == 0 {
                        continue;
                    }

                    let max_items_per_msg = (max_msg_elems / k).max(1);
                    let store_batch_items = (store_batch_elems / k).max(max_items_per_msg);
                    let segment_items = (segment_elems / k).max(max_items_per_msg);
                    let plans = build_segment_plans(c, segment_items);
                    if let Some(writer) = eda_stores[idx].writer()? {
                        if active_edabit_lanes > 1 && plans.len() > 1 {
                            par_segment_plans(
                                io.forks(active_edabit_lanes),
                                plans,
                                |plan, ctx| {
                                    let mut offset = 0usize;
                                    while offset < plan.items {
                                        let items = (plan.items - offset).min(store_batch_items);
                                        let start_item = plan.start_item + offset;
                                        recv_field_messages_into_writer_ctx::<F, _>(
                                            PartyID::ID0,
                                            start_item * k,
                                            items * k,
                                            ctx,
                                            max_msg_elems,
                                            &writer,
                                        )?;
                                        offset += items;
                                    }
                                    Ok(())
                                },
                            )?;
                        } else {
                            let mut write_elem_offset = 0usize;
                            let mut remaining = c;
                            while remaining > 0 {
                                let items = remaining.min(store_batch_items);
                                let expected = items * k;
                                recv_field_messages_into_store::<F, _>(
                                    PartyID::ID0,
                                    write_elem_offset,
                                    expected,
                                    io,
                                    max_msg_elems,
                                    &mut eda_stores[idx],
                                )?;
                                write_elem_offset += expected;
                                remaining -= items;
                            }
                        }
                    }
                }
            }
            let _span =
                tracing::trace_span!("edabits_to_dabits_sync", party_id = ?party_id).entered();
            io.sync_with_parties()?;

            // daBits: receive alpha2 from P0 and s12 from P1 in matching chunks, compute s20,
            // append local stored data, and forward s20 to P0 immediately.
            let dabits_store_path = dir.join("dabits.stored");
            let mut dabits_store = backing_store::BackingStore::create_file_backed_sized(
                &dabits_store_path,
                num_dabits.saturating_mul(2),
            )?;

            if num_dabits > 0 {
                let _span = tracing::info_span!("dabits_resv_store", n = num_dabits).entered();
                let (_, _, ds2_seed, ds2_pos) = snaps[5];
                let max_items_per_msg = max_msg_elems.max(1);
                let segment_items = segment_elems.max(max_items_per_msg);
                let plans = build_segment_plans(num_dabits, segment_items);
                if let Some(writer) = dabits_store.writer()? {
                    let chunk_parallel = active_dabit_lanes == 1;
                    if active_dabit_lanes > 1 && plans.len() > 1 {
                        par_segment_plans(io.forks(active_dabit_lanes), plans, |plan, ctx| {
                            let mut buffered_s20 =
                                Vec::with_capacity(plan.items.min(store_batch_elems));
                            let mut buffered_s12 =
                                Vec::with_capacity(plan.items.min(store_batch_elems));
                            let mut pending_pair_start = plan.start_item;
                            let mut offset = 0usize;
                            while offset < plan.items {
                                let items = (plan.items - offset).min(max_items_per_msg);
                                let start_item = plan.start_item + offset;
                                let alpha2 =
                                    recv_field_chunk_ctx::<F, _>(PartyID::ID0, items, ctx)?;
                                let s12 = recv_field_chunk_ctx::<F, _>(PartyID::ID1, items, ctx)?;
                                let _span_chunk = tracing::trace_span!(
                                    "dabit_s20_forward_chunk",
                                    items,
                                    bytes = items * std::mem::size_of::<F>()
                                )
                                .entered();
                                let s20 = dabit_s20_from_theta_seed::<F>(
                                    ds2_seed,
                                    ds2_pos,
                                    start_item,
                                    alpha2.as_slice(),
                                    chunk_parallel,
                                );
                                if buffered_s20.is_empty() {
                                    pending_pair_start = start_item;
                                }
                                buffered_s20.extend_from_slice(&s20);
                                buffered_s12.extend_from_slice(s12.as_slice());
                                if buffered_s20.len() >= store_batch_elems {
                                    let _span = tracing::trace_span!(
                                        "append_to_store",
                                        start_elem = pending_pair_start * 2,
                                        elems = buffered_s20.len() * 2,
                                        bytes = buffered_s20.len() * 2 * std::mem::size_of::<F>()
                                    )
                                    .entered();
                                    writer.write_interleaved_at(
                                        pending_pair_start,
                                        &buffered_s20,
                                        &buffered_s12,
                                    )?;
                                    buffered_s20.clear();
                                    buffered_s12.clear();
                                }
                                send_field_superchunk_ctx(s20, PartyID::ID0, ctx, max_msg_elems)?;
                                offset += items;
                            }
                            if !buffered_s20.is_empty() {
                                let _span = tracing::trace_span!(
                                    "append_to_store",
                                    start_elem = pending_pair_start * 2,
                                    elems = buffered_s20.len() * 2,
                                    bytes = buffered_s20.len() * 2 * std::mem::size_of::<F>()
                                )
                                .entered();
                                writer.write_interleaved_at(
                                    pending_pair_start,
                                    &buffered_s20,
                                    &buffered_s12,
                                )?;
                            }
                            Ok(())
                        })?;
                    } else {
                        let mut offset = 0usize;
                        while offset < num_dabits {
                            let items = (num_dabits - offset).min(max_items_per_msg);

                            let alpha2 = recv_field_chunk::<F, _>(PartyID::ID0, items, io)?;
                            let s12 = recv_field_chunk::<F, _>(PartyID::ID1, items, io)?;

                            let _span_chunk = tracing::trace_span!(
                                "dabit_s20_forward_chunk",
                                items,
                                bytes = items * std::mem::size_of::<F>()
                            )
                            .entered();
                            let s20 = dabit_s20_from_theta_seed::<F>(
                                ds2_seed,
                                ds2_pos,
                                offset,
                                alpha2.as_slice(),
                                chunk_parallel,
                            );

                            dabits_store.write_interleaved_at(offset, &s20, s12.as_slice())?;
                            send_field_superchunk(s20, PartyID::ID0, io, max_msg_elems)?;

                            offset += items;
                        }
                    }
                }
            }

            let [a0, a1, a2, a3, a4] = eda_stores;
            let mk = |i: usize| snaps[i];
            let (s1, p1, s2, p2) = mk(0);
            let e0 = LazyEdaBits::<u8, F>::new_with_store(s1, p1, s2, p2, counts[0], a0, party_id);
            let (s1, p1, s2, p2) = mk(1);
            let e1 = LazyEdaBits::<u16, F>::new_with_store(s1, p1, s2, p2, counts[1], a1, party_id);
            let (s1, p1, s2, p2) = mk(2);
            let e2 = LazyEdaBits::<u32, F>::new_with_store(s1, p1, s2, p2, counts[2], a2, party_id);
            let (s1, p1, s2, p2) = mk(3);
            let e3 = LazyEdaBits::<u64, F>::new_with_store(s1, p1, s2, p2, counts[3], a3, party_id);
            let (s1, p1, s2, p2) = mk(4);
            let e4 =
                LazyEdaBits::<u128, F>::new_with_store(s1, p1, s2, p2, counts[4], a4, party_id);
            let (ds1, dp1, ds2, dp2) = snaps[5];
            let d =
                LazyDaBits::new_with_store(ds1, dp1, ds2, dp2, num_dabits, dabits_store, party_id);

            let pool = PreprocessingPool::new(party_id, e0, e1, e2, e3, e4, d);
            pool.save(dir)?;
            Ok(pool)
        }
    }
}

/// File-backed preprocessing: generate all edaBits + daBits into `dir`.
#[cfg(not(feature = "ring-msm"))]
pub fn preprocess_pool<F, N>(
    dir: &Path,
    counts: [usize; 5],
    num_dabits: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<PreprocessingPool<F>>
where
    F: PrimeField + Copy,
    N: Rep3NetworkWorker + Rep3RawFieldTransport,
{
    preprocess_pool_base(dir, counts, num_dabits, io)
}

/// File-backed preprocessing: generate edaBits + daBits + wrap masks + ring edaBits into `dir`.
#[cfg(feature = "ring-msm")]
pub fn preprocess_pool<F, N>(
    dir: &Path,
    counts: [usize; 5],
    num_dabits: usize,
    num_wrap_masks: usize,
    num_ring_edabits_u66: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<PreprocessingPool<F>>
where
    F: PrimeField + Copy,
    N: Rep3NetworkWorker + Rep3RawFieldTransport,
{
    let mut pool = preprocess_pool_base(dir, counts, num_dabits, io)?;
    if num_wrap_masks > 0 {
        pool.set_wrap_masks(super::wrap_mask::generate_wrap_masks_lazy(
            num_wrap_masks,
            io.main(),
        )?);
    }
    if num_ring_edabits_u66 > 0 {
        pool.set_ring_edabits_u66(random_edabits_ring_lazy::<U66, _>(
            num_ring_edabits_u66,
            io,
        )?);
    }
    pool.save(dir)?;
    Ok(pool)
}

/// Extend an existing preprocessing pool with additional items.
///
/// For each edaBit type and daBits where `deficit > 0`, generates additional
/// items by seeking into the existing RNG streams past the already-generated
/// region. Same 2-round communication pattern as `preprocess_pool_base`.
///
/// **Communication:** P0→P2 (combined alpha2 for deficit items), P1→P2 (daBit s₁₂),
/// P2→P0 (daBit s₂₀).
#[tracing::instrument(skip_all, name = "extend_pool_batched")]
fn extend_pool_batched_base<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    pool: &mut PreprocessingPool<F>,
    deficit_counts: [usize; 5],
    deficit_dabits: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<()> {
    use super::dabits;

    let party_id = io.party_id();
    let fb = usize::try_from(F::MODULUS_BIT_SIZE)
        .expect("u32 fits into usize")
        .div_ceil(8);
    let max_msg_elems = preproc_max_elems_per_msg::<F>();
    let forks_cap = io.max_forks().max(1);

    fn edabit_alpha2_ext_chunk<T: IntRing2k, Fp: PrimeField>(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        start_item: usize,
        n: usize,
        fb: usize,
    ) -> Vec<Fp>
    where
        Standard: Distribution<T>,
    {
        if n == 0 {
            return Vec::new();
        }
        let t_bytes = std::mem::size_of::<T>();
        let k = T::K;
        let stride = t_bytes + k * fb;

        let all1 = dabits::seek_and_generate(seed1, pos1, start_item * stride, n * stride);
        let g2 = dabits::seek_and_generate(seed2, pos2, start_item * t_bytes, n * t_bytes);

        let mut out = vec![Fp::zero(); n * k];
        out.par_chunks_mut(k)
            .enumerate()
            .with_min_len(256)
            .for_each(|(i, chunk)| {
                let base = i * stride;
                let g1v = T::from_le_bytes(&all1[base..base + t_bytes]);
                let g2v = T::from_le_bytes(&g2[i * t_bytes..(i + 1) * t_bytes]);
                let gamma = g1v ^ g2v;
                for j in 0..k {
                    let s = base + t_bytes + j * fb;
                    let alpha1 = dabits::parse_field::<Fp>(&all1, s);
                    let gbit = ((gamma >> j) & T::one()) == T::one();
                    chunk[j] = Fp::from(gbit as u64) - alpha1;
                }
            });
        out
    }

    // Collect active edaBit types with their metadata.
    let active_types: Vec<(
        usize,
        usize,
        usize,
        ([u8; crate::SEED_SIZE], u128, [u8; crate::SEED_SIZE], u128),
    )> = {
        let ks = [u8::K, u16::K, u32::K, u64::K, u128::K];
        let totals = [
            pool.edabits_u8.total(),
            pool.edabits_u16.total(),
            pool.edabits_u32.total(),
            pool.edabits_u64.total(),
            pool.edabits_u128.total(),
        ];
        let seeds = [
            pool.edabits_u8.extension_seeds(),
            pool.edabits_u16.extension_seeds(),
            pool.edabits_u32.extension_seeds(),
            pool.edabits_u64.extension_seeds(),
            pool.edabits_u128.extension_seeds(),
        ];
        (0..5)
            .filter(|&i| deficit_counts[i] > 0)
            .map(|i| (i, ks[i], totals[i], seeds[i]))
            .collect()
    };
    let active_edabit_lanes = preproc_lanes()
        .max(1)
        .min(forks_cap)
        .min(configured_transport_lanes().max(1));
    let segment_elems = preproc_segment_elems::<F>();

    match party_id {
        PartyID::ID0 => {
            // Round 1: send edaBit α₂ extensions per type, using intra-type
            // lane parallelism (par_segment_plans) just like initial preprocessing.
            for &(ty, k, old_total, seeds) in &active_types {
                let deficit = deficit_counts[ty];
                let (seed1, pos1, seed2, pos2) = seeds;
                let max_items_per_msg = (max_msg_elems / k).max(1);
                let segment_items = (segment_elems / k).max(max_items_per_msg);
                let plans = build_segment_plans(deficit, segment_items);

                if active_edabit_lanes > 1 && plans.len() > 1 {
                    par_segment_plans(io.forks(active_edabit_lanes), plans, |plan, ctx| {
                        let mut done = 0usize;
                        while done < plan.items {
                            let items = (plan.items - done).min(max_items_per_msg);
                            let start = old_total + plan.start_item + done;
                            let alpha2 = match ty {
                                0 => edabit_alpha2_ext_chunk::<u8, F>(
                                    seed1, pos1, seed2, pos2, start, items, fb,
                                ),
                                1 => edabit_alpha2_ext_chunk::<u16, F>(
                                    seed1, pos1, seed2, pos2, start, items, fb,
                                ),
                                2 => edabit_alpha2_ext_chunk::<u32, F>(
                                    seed1, pos1, seed2, pos2, start, items, fb,
                                ),
                                3 => edabit_alpha2_ext_chunk::<u64, F>(
                                    seed1, pos1, seed2, pos2, start, items, fb,
                                ),
                                4 => edabit_alpha2_ext_chunk::<u128, F>(
                                    seed1, pos1, seed2, pos2, start, items, fb,
                                ),
                                _ => unreachable!(),
                            };
                            send_field_superchunk_ctx(alpha2, PartyID::ID2, ctx, max_msg_elems)?;
                            done += items;
                        }
                        Ok(())
                    })?;
                } else {
                    let mut done = 0usize;
                    while done < deficit {
                        let items = (deficit - done).min(max_items_per_msg);
                        let start = old_total + done;
                        let alpha2 = match ty {
                            0 => edabit_alpha2_ext_chunk::<u8, F>(
                                seed1, pos1, seed2, pos2, start, items, fb,
                            ),
                            1 => edabit_alpha2_ext_chunk::<u16, F>(
                                seed1, pos1, seed2, pos2, start, items, fb,
                            ),
                            2 => edabit_alpha2_ext_chunk::<u32, F>(
                                seed1, pos1, seed2, pos2, start, items, fb,
                            ),
                            3 => edabit_alpha2_ext_chunk::<u64, F>(
                                seed1, pos1, seed2, pos2, start, items, fb,
                            ),
                            4 => edabit_alpha2_ext_chunk::<u128, F>(
                                seed1, pos1, seed2, pos2, start, items, fb,
                            ),
                            _ => unreachable!(),
                        };
                        send_field_superchunk(alpha2, PartyID::ID2, io, max_msg_elems)?;
                        done += items;
                    }
                }
            }

            // Barrier: all parties must finish edaBit phase before daBit phase.
            io.sync_with_parties()?;

            // daBit α₂ extension (chunked, interleaved send/recv to avoid deadlock).

            if deficit_dabits > 0 {
                let (ds1, dp1, ds2, dp2) = pool.dabits.extension_seeds();
                let old_total = pool.dabits.total();
                let da_stride = 1 + 2 * fb;
                let max_items_per_msg = max_msg_elems.max(1);
                let mut done = 0usize;
                while done < deficit_dabits {
                    let items = (deficit_dabits - done).min(max_items_per_msg);
                    let start = old_total + done;
                    let s1_buf =
                        dabits::seek_and_generate(ds1, dp1, start * da_stride, items * da_stride);
                    let g2_buf = dabits::seek_and_generate(ds2, dp2, start, items);
                    let da_alpha2 = (0..items)
                        .into_par_iter()
                        .with_min_len(256)
                        .map(|i| {
                            let base = i * da_stride;
                            let gbit = ((s1_buf[base] ^ g2_buf[i]) & 1) != 0;
                            let alpha1: F = dabits::parse_field(&s1_buf, base + 1);
                            F::from(gbit as u64) - alpha1
                        })
                        .collect::<Vec<F>>();
                    send_field_superchunk(da_alpha2, PartyID::ID2, io, max_msg_elems)?;
                    // Interleaved recv: must drain s20 each iteration to prevent
                    // circular buffer deadlock (P0 blocked sending, P2 blocked sending back).
                    let s20 = recv_field_messages_collect::<F, _>(
                        PartyID::ID2,
                        items,
                        io,
                        max_msg_elems,
                    )?;
                    pool.dabits.apply_extension(items, s20);
                    done += items;
                }
            }

            // Apply edaBits total extensions (P0 stores no α₂).
            pool.edabits_u8
                .apply_extension(deficit_counts[0], Vec::new());
            pool.edabits_u16
                .apply_extension(deficit_counts[1], Vec::new());
            pool.edabits_u32
                .apply_extension(deficit_counts[2], Vec::new());
            pool.edabits_u64
                .apply_extension(deficit_counts[3], Vec::new());
            pool.edabits_u128
                .apply_extension(deficit_counts[4], Vec::new());
        }
        PartyID::ID1 => {
            // Barrier: all parties must finish edaBit phase before daBit phase.
            io.sync_with_parties()?;

            // P1: send daBit s₁₂ extension in chunks.
            if deficit_dabits > 0 {
                let (ds1, dp1, ds2, dp2) = pool.dabits.extension_seeds();
                let old_total = pool.dabits.total();
                let da_stride = 1 + 2 * fb;
                let max_items_per_msg = max_msg_elems.max(1);
                let mut done = 0usize;
                while done < deficit_dabits {
                    let items = (deficit_dabits - done).min(max_items_per_msg);
                    let start = old_total + done;
                    let s2_buf =
                        dabits::seek_and_generate(ds2, dp2, start * da_stride, items * da_stride);
                    let theta_buf = dabits::seek_and_generate(ds1, dp1, start, items);
                    let s12: Vec<F> = (0..items)
                        .into_par_iter()
                        .with_min_len(256)
                        .map(|i| {
                            let base = i * da_stride;
                            let theta = (theta_buf[i] & 1) != 0;
                            let neg1_theta = if theta { -F::one() } else { F::one() };
                            let alpha1: F = dabits::parse_field(&s2_buf, base + 1);
                            let r1: F = dabits::parse_field(&s2_buf, base + 1 + fb);
                            neg1_theta * alpha1 - r1
                        })
                        .collect();
                    send_field_superchunk(s12, PartyID::ID2, io, max_msg_elems)?;
                    done += items;
                }
                pool.dabits.apply_extension(deficit_dabits, Vec::new());
            }
            // Apply edaBits total extensions (P1 stores no α₂).
            pool.edabits_u8
                .apply_extension(deficit_counts[0], Vec::new());
            pool.edabits_u16
                .apply_extension(deficit_counts[1], Vec::new());
            pool.edabits_u32
                .apply_extension(deficit_counts[2], Vec::new());
            pool.edabits_u64
                .apply_extension(deficit_counts[3], Vec::new());
            pool.edabits_u128
                .apply_extension(deficit_counts[4], Vec::new());
        }
        PartyID::ID2 => {
            // Receive edaBit α₂ extensions per type, using intra-type lane
            // parallelism (par_segment_plans) with FileBackedWriter for
            // parallel pwrite to different file offsets.
            for &(ty, k, _old_total, _seeds) in &active_types {
                let deficit = deficit_counts[ty];
                let bs = match ty {
                    0 => &mut pool.edabits_u8.alpha2_flat,
                    1 => &mut pool.edabits_u16.alpha2_flat,
                    2 => &mut pool.edabits_u32.alpha2_flat,
                    3 => &mut pool.edabits_u64.alpha2_flat,
                    4 => &mut pool.edabits_u128.alpha2_flat,
                    _ => unreachable!(),
                };
                let old_len = bs.len();
                let max_items_per_msg = (max_msg_elems / k).max(1);
                let segment_items = (segment_elems / k).max(max_items_per_msg);
                let plans = build_segment_plans(deficit, segment_items);
                if let Some(writer) = bs.pre_extended_writer(deficit * k)? {
                    if active_edabit_lanes > 1 && plans.len() > 1 {
                        par_segment_plans(io.forks(active_edabit_lanes), plans, |plan, ctx| {
                            let mut done = 0usize;
                            while done < plan.items {
                                let items = (plan.items - done).min(max_items_per_msg);
                                let write_start = old_len + (plan.start_item + done) * k;
                                recv_field_messages_into_writer_ctx::<F, _>(
                                    PartyID::ID0,
                                    write_start,
                                    items * k,
                                    ctx,
                                    max_msg_elems,
                                    &writer,
                                )?;
                                done += items;
                            }
                            Ok(())
                        })?;
                    } else {
                        let mut write_offset = old_len;
                        let mut remaining = deficit;
                        while remaining > 0 {
                            let items = remaining.min(max_items_per_msg);
                            recv_field_messages_into_writer_ctx::<F, _>(
                                PartyID::ID0,
                                write_offset,
                                items * k,
                                io.main(),
                                max_msg_elems,
                                &writer,
                            )?;
                            write_offset += items * k;
                            remaining -= items;
                        }
                    }
                }
                // Bump total (file was already extended by grow_and_writer).
                match ty {
                    0 => pool.edabits_u8.total += deficit,
                    1 => pool.edabits_u16.total += deficit,
                    2 => pool.edabits_u32.total += deficit,
                    3 => pool.edabits_u64.total += deficit,
                    4 => pool.edabits_u128.total += deficit,
                    _ => unreachable!(),
                }
            }

            // Barrier: all parties must finish edaBit phase before daBit phase.
            io.sync_with_parties()?;

            // daBits: receive alpha2 from P0 and s12 from P1 in matching chunks, compute s20,
            // append interleaved stored_ext, and forward s20 to P0 immediately.

            if deficit_dabits > 0 {
                let (_ds1, _dp1, ds2, dp2) = pool.dabits.extension_seeds();
                let old_total = pool.dabits.total();

                let max_items_per_msg = max_msg_elems.max(1);
                let mut done = 0usize;
                while done < deficit_dabits {
                    let items = (deficit_dabits - done).min(max_items_per_msg);
                    let alpha2 = recv_field_messages_collect::<F, _>(
                        PartyID::ID0,
                        items,
                        io,
                        max_msg_elems,
                    )?;
                    let s12 = recv_field_messages_collect::<F, _>(
                        PartyID::ID1,
                        items,
                        io,
                        max_msg_elems,
                    )?;
                    let theta_buf = dabits::seek_and_generate(ds2, dp2, old_total + done, items);

                    let _span = tracing::trace_span!(
                        "dabit_s20_forward_chunk",
                        party_id = ?party_id,
                        items,
                        bytes = items * std::mem::size_of::<F>()
                    )
                    .entered();
                    let s20: Vec<F> = (0..items)
                        .into_par_iter()
                        .with_min_len(256)
                        .map(|i| {
                            let theta = (theta_buf[i] & 1) != 0;
                            let neg1_theta = if theta { -F::one() } else { F::one() };
                            neg1_theta * alpha2[i]
                        })
                        .collect();

                    let mut stored_chunk = Vec::with_capacity(2 * items);
                    for i in 0..items {
                        stored_chunk.push(s20[i]);
                        stored_chunk.push(s12[i]);
                    }
                    pool.dabits.apply_extension(items, stored_chunk);
                    send_field_superchunk(s20, PartyID::ID0, io, max_msg_elems)?;
                    done += items;
                }
            }
        }
    }

    Ok(())
}

/// Extend an existing pool with additional edaBits + daBits.
#[cfg(not(feature = "ring-msm"))]
pub fn extend_pool_batched<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    pool: &mut PreprocessingPool<F>,
    deficit_counts: [usize; 5],
    deficit_dabits: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<()> {
    extend_pool_batched_base(pool, deficit_counts, deficit_dabits, io)
}

/// Extend an existing pool with additional edaBits + daBits + wrap masks + ring edaBits.
#[cfg(feature = "ring-msm")]
pub fn extend_pool_batched<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    pool: &mut PreprocessingPool<F>,
    deficit_counts: [usize; 5],
    deficit_dabits: usize,
    deficit_wrap_masks: usize,
    deficit_ring_edabits_u66: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<()> {
    extend_pool_batched_base(pool, deficit_counts, deficit_dabits, io)?;
    if deficit_wrap_masks > 0 {
        pool.set_wrap_masks(super::wrap_mask::generate_wrap_masks_lazy(
            deficit_wrap_masks,
            io.main(),
        )?);
    }
    if deficit_ring_edabits_u66 > 0 {
        pool.set_ring_edabits_u66(random_edabits_ring_lazy::<U66, _>(
            deficit_ring_edabits_u66,
            io,
        )?);
    }
    Ok(())
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
// Ring-domain EdaBits: B2A in Z_{2^K} via Protocol Π₂
// ---------------------------------------------------------------------------

/// EdaBits batch for ring-to-ring B2A (Protocol Π₂ in ring domain).
///
/// Each tuple links a random K-bit value γ (known only to P0) with a 2-of-2
/// arithmetic sharing of each γ bit in the ring Z_{2^K}.
pub struct EdaBitsRingBatch<T: IntRing2k> {
    pub gammas: Vec<RingElement<T>>,
    pub alphas_flat: Vec<RingElement<T>>,
}

// ---------------------------------------------------------------------------
// LazyEdaBitsRing<T> — lazy ring-domain edaBits with BackingStore persistence
// ---------------------------------------------------------------------------

/// Lazy ring-domain edaBits source, mirroring `LazyEdaBits<T, F>` but in ring domain.
///
/// P0/P1 store only RNG seeds (~192 bytes) and regenerate on demand.
/// P2 stores received alpha₂ values in a `BackingStore` (memory or mmap).
pub struct LazyEdaBitsRing<T: IntRing2k> {
    seed1: [u8; crate::SEED_SIZE],
    pos1: u128,
    seed2: [u8; crate::SEED_SIZE],
    pos2: u128,
    total: usize,
    cursor: usize,
    /// P2-only: flat alpha₂ values (length = total * T::K). Empty for P0/P1.
    alpha2_flat: backing_store::BackingStore<RingElement<T>>,
    party_id: PartyID,
    meta_path: Option<std::path::PathBuf>,
    _phantom: PhantomData<T>,
}

impl<T: IntRing2k> LazyEdaBitsRing<T>
where
    Standard: Distribution<T>,
{
    /// Create an empty lazy source.
    pub fn empty(party_id: PartyID) -> Self {
        Self {
            seed1: [0u8; crate::SEED_SIZE],
            pos1: 0,
            seed2: [0u8; crate::SEED_SIZE],
            pos2: 0,
            total: 0,
            cursor: 0,
            alpha2_flat: backing_store::BackingStore::Empty,
            party_id,
            meta_path: None,
            _phantom: PhantomData,
        }
    }

    /// Construct from RNG seeds + P2's alpha₂.
    pub fn new(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        total: usize,
        alpha2_flat: Vec<RingElement<T>>,
        party_id: PartyID,
    ) -> Self {
        Self {
            seed1,
            pos1,
            seed2,
            pos2,
            total,
            cursor: 0,
            alpha2_flat: backing_store::BackingStore::from_vec(alpha2_flat),
            party_id,
            meta_path: None,
            _phantom: PhantomData,
        }
    }

    pub fn remaining(&self) -> usize {
        self.total - self.cursor
    }

    /// Drain `n` ring edaBits. P0/P1 regenerate from seeds; P2 slices from store.
    pub fn take_batch(&mut self, n: usize) -> eyre::Result<EdaBitsRingBatch<T>> {
        eyre::ensure!(
            self.cursor + n <= self.total,
            "LazyEdaBitsRing<u{}>: need {n}, have {} (cursor={}, total={})",
            T::K,
            self.remaining(),
            self.cursor,
            self.total
        );

        if n == 0 {
            return Ok(EdaBitsRingBatch {
                gammas: Vec::new(),
                alphas_flat: Vec::new(),
            });
        }

        let t_bytes = std::mem::size_of::<T>();
        let k = T::K;
        let party_id = self.party_id;
        let cursor_base = self.cursor;

        // P2: slice from stored alpha2_flat.
        if party_id == PartyID::ID2 {
            let flat_start = cursor_base * k;
            let flat_end = flat_start + n * k;
            let alphas_flat = self.alpha2_flat.as_slice()[flat_start..flat_end].to_vec();
            let gammas = vec![RingElement(T::zero()); n];
            self.cursor += n;
            self.persist_cursor();
            self.alpha2_flat.consume(flat_start, flat_end);
            return Ok(EdaBitsRingBatch {
                gammas,
                alphas_flat,
            });
        }

        // P0/P1: regenerate from RNG seeds.
        // Per-item interleaved layout in rng1 (P0↔P1 stream):
        //   [γ₀(t_B) α_{0,0}(t_B)..α_{0,k-1}(t_B) | γ₁ α_{1,0}..α_{1,k-1} | ...]
        // stride = (1 + K) * t_bytes.
        let stride = (1 + k) * t_bytes;
        let item_byte_offset = cursor_base * stride;
        let interleaved_bytes = n * stride;

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

        // rng2 (P0↔P2) only carries gammas — stride = (1+K)*t_bytes (same as rng1
        // because random_elements advances BOTH rngs equally).
        let g2_gamma_offset = cursor_base * stride;
        let g2_gamma_bytes = n * stride;

        let (interleaved, g2_bytes);
        if party_id == PartyID::ID0 {
            let (s1, s2) = rayon::join(
                || seek_and_generate(self.seed1, self.pos1, item_byte_offset, interleaved_bytes),
                || seek_and_generate(self.seed2, self.pos2, g2_gamma_offset, g2_gamma_bytes),
            );
            interleaved = s1;
            g2_bytes = s2;
        } else {
            // P1: interleaved from seed2 (P1↔P0 = P0↔P1 shared stream).
            interleaved =
                seek_and_generate(self.seed2, self.pos2, item_byte_offset, interleaved_bytes);
            g2_bytes = Vec::new();
        }

        // Build gammas and alphas from interleaved layout.
        let gammas: Vec<RingElement<T>> = if party_id == PartyID::ID0 {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let g1_off = i * stride;
                    let g2_off = i * stride; // rng2 has same stride
                    let g1_val = T::from_le_bytes(&interleaved[g1_off..g1_off + t_bytes]);
                    let g2_val = T::from_le_bytes(&g2_bytes[g2_off..g2_off + t_bytes]);
                    RingElement(g1_val ^ g2_val)
                })
                .collect()
        } else {
            vec![RingElement(T::zero()); n]
        };

        let alphas_flat: Vec<RingElement<T>> = (0..n * k)
            .into_par_iter()
            .with_min_len(256)
            .map(|idx| {
                let item = idx / k;
                let bit = idx % k;
                let a_start = item * stride + t_bytes + bit * t_bytes;
                let val = T::from_le_bytes(&interleaved[a_start..a_start + t_bytes]);
                RingElement(val)
            })
            .collect();

        self.cursor += n;
        Ok(EdaBitsRingBatch {
            gammas,
            alphas_flat,
        })
    }
}

// Persistence methods for LazyEdaBitsRing.
impl<T: IntRing2k> LazyEdaBitsRing<T> {
    pub fn save(&self, dir: &std::path::Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let suffix = format!("ring_edabits_{}", T::K);
        if !self.alpha2_flat.is_empty() {
            let data_path = dir.join(format!("{suffix}.alpha2"));
            self.alpha2_flat.save_to_file(&data_path)?;
        }
        backing_store::write_meta(
            &dir.join(format!("{suffix}.meta")),
            &backing_store::MetaData {
                seed1: self.seed1,
                pos1: self.pos1,
                seed2: self.seed2,
                pos2: self.pos2,
                total: self.total,
                party_id_byte: backing_store::party_id_to_byte(self.party_id),
                cursor: self.cursor,
                field_bytes: std::mem::size_of::<T>(),
            },
        )?;
        std::result::Result::Ok(())
    }

    pub fn load(dir: &std::path::Path, party_id: PartyID) -> std::io::Result<Self> {
        let suffix = format!("ring_edabits_{}", T::K);
        let meta_path = dir.join(format!("{suffix}.meta"));
        if !meta_path.exists() {
            return std::result::Result::Ok(Self {
                seed1: [0u8; crate::SEED_SIZE],
                pos1: 0,
                seed2: [0u8; crate::SEED_SIZE],
                pos2: 0,
                total: 0,
                cursor: 0,
                alpha2_flat: backing_store::BackingStore::Empty,
                party_id,
                meta_path: None,
                _phantom: PhantomData,
            });
        }
        let meta = backing_store::read_meta(&meta_path)?;
        assert_eq!(
            meta.party_id_byte,
            backing_store::party_id_to_byte(party_id)
        );
        let alpha2_flat = if party_id == PartyID::ID2 && meta.total > 0 {
            let data_path = dir.join(format!("{suffix}.alpha2"));
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
            cursor: meta.cursor,
            alpha2_flat,
            party_id,
            meta_path: Some(meta_path),
            _phantom: PhantomData,
        })
    }

    fn persist_cursor(&self) {
        if let Some(ref path) = self.meta_path {
            let _ = backing_store::update_cursor(path, self.cursor);
        }
    }
}

/// Generate `num` lazy ring-domain edaBits (offline preprocessing).
///
/// Mirrors `random_edabits_lazy()` but in ring domain. P0/P1 store only seeds;
/// P2 stores received alpha₂.
/// Communication: P0 → P2: `num * K` ring elements (1 round).
pub fn random_edabits_ring_lazy<T: IntRing2k, N: Rep3NetworkWorker>(
    num: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<LazyEdaBitsRing<T>>
where
    Standard: Distribution<T>,
{
    let party_id = io.party_id();
    if num == 0 {
        return Ok(LazyEdaBitsRing::empty(party_id));
    }

    let t_bytes = std::mem::size_of::<T>();
    let k = T::K;

    // Fork a dedicated Rep3Rand and snapshot its state BEFORE generating bytes.
    let mut eda_rand = io.main().rngs.rand.fork();
    let (seed1, pos1, seed2, pos2) = eda_rand.snapshot();

    // Per-item interleaved layout in rng1:
    //   [γ₀(t_B) α_{0,0}(t_B)..α_{0,k-1}(t_B) | γ₁ α_{1,0}..α_{1,k-1} | ...]
    // stride = (1 + K) * t_bytes.
    let stride = (1 + k) * t_bytes;

    // P0 → P2: send alpha_2 = RingElement(gamma_bit) - alpha_1.
    if party_id == PartyID::ID0 {
        let (all_bytes1, g2_bytes) = {
            let mut a = vec![0u8; num * stride];
            let mut b = vec![0u8; num * stride]; // rng2 same stride (both rngs advance equally)
            rayon::join(
                || eda_rand.rng1.fill_bytes(&mut a),
                || eda_rand.rng2.fill_bytes(&mut b),
            );
            (a, b)
        };

        let mut alpha_2_all = vec![RingElement(T::zero()); num * k];
        alpha_2_all
            .par_chunks_mut(k)
            .enumerate()
            .with_min_len(256)
            .for_each(|(i, chunk)| {
                let base = i * stride;
                let g1_val = T::from_le_bytes(&all_bytes1[base..base + t_bytes]);
                let g2_val = T::from_le_bytes(&g2_bytes[base..base + t_bytes]);
                let gamma = g1_val ^ g2_val;
                for j in 0..k {
                    let a_start = base + t_bytes + j * t_bytes;
                    let alpha_1 =
                        RingElement(T::from_le_bytes(&all_bytes1[a_start..a_start + t_bytes]));
                    let gamma_bit = ((gamma >> j) & T::one()) == T::one();
                    chunk[j] = RingElement(T::from(gamma_bit)) - alpha_1;
                }
            });
        io.network().send_many(PartyID::ID2, &alpha_2_all)?;
    }

    let alpha2_flat: Vec<RingElement<T>> = if party_id == PartyID::ID2 {
        let alpha_2_all: Vec<RingElement<T>> = io.network().recv_many(PartyID::ID0)?;
        debug_assert_eq!(alpha_2_all.len(), num * k);
        alpha_2_all
    } else {
        Vec::new()
    };

    Ok(LazyEdaBitsRing::new(
        seed1,
        pos1,
        seed2,
        pos2,
        num,
        alpha2_flat,
        party_id,
    ))
}

/// Ring-domain B2A: convert binary XOR-shares to arithmetic ring shares.
///
/// Online communication: 2 rounds (1 broadcast + 1 reshare_many).
pub fn ring_b2a_many<T: IntRing2k, N: Rep3Network>(
    x_binary: &[Rep3RingShare<T>],
    batch: &EdaBitsRingBatch<T>,
    io: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3RingShare<T>>>
where
    Standard: Distribution<T>,
{
    let n = x_binary.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    debug_assert_eq!(batch.gammas.len(), n);
    debug_assert_eq!(batch.alphas_flat.len(), n * T::K);

    let k = T::K;

    // Precompute powers of 2 in the ring.
    let pow2: Vec<RingElement<T>> = {
        let mut pow2 = Vec::with_capacity(k);
        let mut cur = RingElement(T::one());
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

    // --- Local computation + masking ---
    let maskings: Vec<RingElement<T>> = (0..n)
        .map(|_| io.rngs.rand.masking_element::<RingElement<T>>())
        .collect();
    let party_id = io.id;

    let s_selfs: Vec<RingElement<T>> = ms
        .iter()
        .zip(x_binary.iter())
        .zip(maskings.iter())
        .enumerate()
        .map(|(idx, ((m, x), z))| {
            if party_id == PartyID::ID0 {
                return *z;
            }

            let beta = match party_id {
                PartyID::ID0 => unreachable!(),
                PartyID::ID1 => *m ^ x.a,
                PartyID::ID2 => *m ^ x.b,
            };

            let mut v = RingElement(T::zero());
            let alpha_base = idx * k;
            for i in 0..k {
                let beta_bit = ((beta.0 >> i) & T::one()) == T::one();
                let alpha = batch.alphas_flat[alpha_base + i];
                let signed_alpha = if beta_bit { -alpha } else { alpha };
                v = v + pow2[i] * signed_alpha;
            }

            if party_id == PartyID::ID1 {
                v = v + beta;
            }

            v + *z
        })
        .collect();

    // --- Round 2: reshare ---
    let s_prevs = io.network.reshare_many(&s_selfs)?;

    Ok(s_selfs
        .into_iter()
        .zip(s_prevs)
        .map(|(s_self, s_prev)| Rep3RingShare::new_ring(s_self, s_prev))
        .collect())
}

// ---------------------------------------------------------------------------
// EdaBitsPool: pre-generated edaBits/daBits for batched conversions
// ---------------------------------------------------------------------------

pub use super::pool::PreprocessingPool;

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

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(value) => unsafe { std::env::set_var(self.key, value) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

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
                Ok(lazy.take_batch(NUM)?)
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
                let batch = lazy.take_batch(NUM)?;
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
                    let batch = lazy_u32.take_batch(NUM)?;
                    ring_to_field_b2a_many::<u32, Fr, _>(&id_sh, &batch, io)?
                };

                // Call 2: u16 left
                let left = {
                    let batch = lazy_u16.take_batch(NUM)?;
                    ring_to_field_b2a_many::<u16, Fr, _>(&left_sh, &batch, io)?
                };

                // Call 3: u16 right
                let right = {
                    let batch = lazy_u16.take_batch(NUM)?;
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
    #[cfg(not(feature = "ring-msm"))]
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
                let _discard = lazy.take_batch(BATCH1)?;

                // Now take batch2 at cursor=BATCH1 (odd) — this tests the word alignment fix
                let batch = lazy.take_batch(BATCH2)?;
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

    /// Helper: run preprocess + B2A + bit-inject roundtrip.
    #[cfg(not(feature = "ring-msm"))]
    fn preprocess_roundtrip_impl() {
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
                    let pool_dir = std::env::temp_dir().join(format!(
                        "mpc-core-test-preproc-batched-{}",
                        io_ctx.party_idx()
                    ));
                    let mut pool = preprocess_pool::<Fr, _>(
                        &pool_dir,
                        [0, 0, 0, NUM_U64, 0],
                        NUM_DABITS,
                        &mut io_ctx,
                    )?;

                    // B2A via edaBits
                    let batch = pool.take_edabits::<u64>(NUM_U64)?;
                    let b2a = ring_to_field_b2a_many::<u64, Fr, _>(&x_sh, &batch, io_ctx.main())?;

                    // Bit inject via daBits
                    let dbatch = pool.take_dabits(NUM_DABITS)?;
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
    #[cfg(not(feature = "ring-msm"))]
    fn preprocess_pool_roundtrip() {
        use crate::protocols::rep3_ring::dabits;

        const NUM_U64: usize = 8;
        const NUM_DABITS: usize = 16;

        let mut rng = ChaCha20Rng::seed_from_u64(0xB002_0002);

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

        let base_dir = std::env::temp_dir().join(format!("co_jolt2_preproc_{}", rng.next_u64()));
        let base_dir_for_workers = base_dir.clone();
        std::fs::create_dir_all(&base_dir).expect("failed to create temp dir");

        type Input = (Vec<Rep3RingShare<u64>>, Vec<Rep3RingShare<RingBit>>);
        let outputs: [(Vec<Rep3PrimeFieldShare<Fr>>, Vec<Rep3PrimeFieldShare<Fr>>); 3] =
            run_rep3_local_test_with_coordinator(
                1,
                |i| (x_bin_shares[i].clone(), bit_shares[i].clone()),
                || (),
                move |(x_sh, bit_sh): Input, mut io_ctx| {
                    let party_id = io_ctx.party_id();
                    let pool_dir =
                        base_dir_for_workers.join(format!("party_{}", usize::from(party_id)));
                    std::fs::create_dir_all(&pool_dir)?;

                    let mut pool = preprocess_pool::<Fr, _>(
                        &pool_dir,
                        [0, 0, 0, NUM_U64, 0],
                        NUM_DABITS,
                        &mut io_ctx,
                    )?;

                    // Load back from disk and consume from the loaded pool.
                    let mut loaded = PreprocessingPool::<Fr>::load(&pool_dir, party_id)?;
                    // Ensure the returned pool and the loaded pool expose the same remaining counts.
                    assert_eq!(pool.remaining_counts(), loaded.remaining_counts());

                    let batch = loaded.take_edabits::<u64>(NUM_U64)?;
                    let b2a = ring_to_field_b2a_many::<u64, Fr, _>(&x_sh, &batch, io_ctx.main())?;

                    let dbatch = loaded.take_dabits(NUM_DABITS)?;
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
        assert_eq!(
            b2a_combined, b2a_expected,
            "preprocess_into_dir B2A mismatch"
        );

        // Verify bit injection
        let inj_combined = combine_field_elements(&outputs[0].1, &outputs[1].1, &outputs[2].1);
        let inj_expected: Vec<Fr> = bits.iter().map(|&b| Fr::from(b as u64)).collect();
        assert_eq!(
            inj_combined, inj_expected,
            "preprocess_into_dir bit inject mismatch"
        );

        let _ = std::fs::remove_dir_all(&base_dir);
    }

    #[test]
    #[cfg(not(feature = "ring-msm"))]
    fn preprocess_pool_small_chunks() {
        let _msg_guard = EnvVarGuard::set("PREPROC_MAX_MSG_MB", "1");
        let _fork_guard = EnvVarGuard::set("PREPROC_MIN_FORK_ELEMS", "1");
        preprocess_pool_roundtrip();
    }

    /// Test that extend_pool_batched produces correct results.
    ///
    /// 1. Create a pool with INITIAL items.
    /// 2. Consume all of them.
    /// 3. Extend the pool with EXTRA deficit items.
    /// 4. Consume the new items and verify B2A / bit-inject correctness.
    #[test]
    #[cfg(not(feature = "ring-msm"))]
    fn extend_pool_batched_roundtrip() {
        use crate::protocols::rep3_ring::dabits;

        const INITIAL_U64: usize = 4;
        const INITIAL_DABITS: usize = 8;
        const EXTRA_U64: usize = 6;
        const EXTRA_DABITS: usize = 10;

        let mut rng = ChaCha20Rng::seed_from_u64(0xE001_0001);

        // Values for initial consumption (just to advance cursors)
        let init_xs: Vec<u64> = (0..INITIAL_U64).map(|_| rng.next_u64()).collect();
        let init_x_shares: [Vec<Rep3RingShare<u64>>; 3] = {
            let per = init_xs
                .iter()
                .map(|&x| share_ring_element_binary::<u64, _>(RingElement(x), &mut rng))
                .collect::<Vec<_>>();
            std::array::from_fn(|pid| per.iter().map(|s| s[pid]).collect())
        };
        let init_bits: Vec<bool> = (0..INITIAL_DABITS)
            .map(|_| (rng.next_u32() & 1) == 1)
            .collect();
        let init_bit_shares: [Vec<Rep3RingShare<RingBit>>; 3] = {
            let per = init_bits
                .iter()
                .map(|&b| share_ring_element::<RingBit, _>(RingElement(RingBit::new(b)), &mut rng))
                .collect::<Vec<_>>();
            std::array::from_fn(|pid| per.iter().map(|s| s[pid]).collect())
        };

        // Values for extension verification
        let ext_xs: Vec<u64> = (0..EXTRA_U64).map(|_| rng.next_u64()).collect();
        let ext_x_shares: [Vec<Rep3RingShare<u64>>; 3] = {
            let per = ext_xs
                .iter()
                .map(|&x| share_ring_element_binary::<u64, _>(RingElement(x), &mut rng))
                .collect::<Vec<_>>();
            std::array::from_fn(|pid| per.iter().map(|s| s[pid]).collect())
        };
        let ext_bits: Vec<bool> = (0..EXTRA_DABITS)
            .map(|_| (rng.next_u32() & 1) == 1)
            .collect();
        let ext_bit_shares: [Vec<Rep3RingShare<RingBit>>; 3] = {
            let per = ext_bits
                .iter()
                .map(|&b| share_ring_element::<RingBit, _>(RingElement(RingBit::new(b)), &mut rng))
                .collect::<Vec<_>>();
            std::array::from_fn(|pid| per.iter().map(|s| s[pid]).collect())
        };

        type Input = (
            Vec<Rep3RingShare<u64>>,
            Vec<Rep3RingShare<RingBit>>,
            Vec<Rep3RingShare<u64>>,
            Vec<Rep3RingShare<RingBit>>,
        );

        let outputs: [(Vec<Rep3PrimeFieldShare<Fr>>, Vec<Rep3PrimeFieldShare<Fr>>); 3] =
            run_rep3_local_test_with_coordinator(
                1,
                |i| {
                    (
                        init_x_shares[i].clone(),
                        init_bit_shares[i].clone(),
                        ext_x_shares[i].clone(),
                        ext_bit_shares[i].clone(),
                    )
                },
                || (),
                move |(init_x, init_b, ext_x, ext_b): Input, mut io_ctx| {
                    // Phase 1: create initial pool and consume everything.
                    let pool_dir = std::env::temp_dir()
                        .join(format!("mpc-core-test-extend-{}", io_ctx.party_idx()));
                    let mut pool = preprocess_pool::<Fr, _>(
                        &pool_dir,
                        [0, 0, 0, INITIAL_U64, 0],
                        INITIAL_DABITS,
                        &mut io_ctx,
                    )?;
                    let batch = pool.take_edabits::<u64>(INITIAL_U64)?;
                    let _ = ring_to_field_b2a_many::<u64, Fr, _>(&init_x, &batch, io_ctx.main())?;
                    let dbatch = pool.take_dabits(INITIAL_DABITS)?;
                    let _ =
                        dabits::bit_inject_field_many::<Fr, _>(&init_b, &dbatch, io_ctx.main())?;

                    // Phase 2: extend pool to cover extra items.
                    let deficit_counts = [0, 0, 0, EXTRA_U64, 0];
                    extend_pool_batched(&mut pool, deficit_counts, EXTRA_DABITS, &mut io_ctx)?;

                    // Phase 3: consume extension items and verify.
                    let batch2 = pool.take_edabits::<u64>(EXTRA_U64)?;
                    let b2a = ring_to_field_b2a_many::<u64, Fr, _>(&ext_x, &batch2, io_ctx.main())?;
                    let dbatch2 = pool.take_dabits(EXTRA_DABITS)?;
                    let inj =
                        dabits::bit_inject_field_many::<Fr, _>(&ext_b, &dbatch2, io_ctx.main())?;

                    Ok((b2a, inj))
                },
                |(): (), _net| Ok(()),
            )
            .0;

        // Verify B2A
        let b2a_combined = combine_field_elements(&outputs[0].0, &outputs[1].0, &outputs[2].0);
        let b2a_expected: Vec<Fr> = ext_xs.iter().map(|&x| Fr::from(x)).collect();
        assert_eq!(b2a_combined, b2a_expected, "extend B2A mismatch");

        // Verify bit injection
        let inj_combined = combine_field_elements(&outputs[0].1, &outputs[1].1, &outputs[2].1);
        let inj_expected: Vec<Fr> = ext_bits.iter().map(|&b| Fr::from(b as u64)).collect();
        assert_eq!(inj_combined, inj_expected, "extend bit inject mismatch");
    }

    #[test]
    fn ring_b2a_many_u66_correct() {
        use crate::protocols::rep3::network::IoContextPool;
        use crate::protocols::rep3::test_utils::LocalRep3TestWorkerNet;
        use mpc_types::protocols::rep3_ring::ring::u66::U66;

        let mut rng = ChaCha20Rng::seed_from_u64(0x66B2A);
        let n = 16;
        let values: Vec<RingElement<U66>> = (0..n)
            .map(|_| {
                let lo = rng.next_u64() as u128;
                let hi = (rng.next_u64() as u128) & 3; // 2 extra bits
                RingElement(U66::new(lo | (hi << 64)))
            })
            .collect();

        // Create binary (XOR) shares of each value.
        let bin_shares: Vec<[Rep3RingShare<U66>; 3]> = values
            .iter()
            .map(|v| share_ring_element_binary(*v, &mut rng))
            .collect();

        let (results, _) = run_rep3_local_test_with_coordinator(
            0,
            |party_idx| {
                let party_bins: Vec<Rep3RingShare<U66>> =
                    bin_shares.iter().map(|s| s[party_idx]).collect();
                party_bins
            },
            || (),
            move |party_bins: Vec<Rep3RingShare<U66>>,
                  mut io_ctx: IoContextPool<LocalRep3TestWorkerNet>| {
                let mut lazy = random_edabits_ring_lazy::<U66, _>(n, &mut io_ctx)?;
                let batch = lazy.take_batch(n)?;
                let io = io_ctx.main();
                let arith = ring_b2a_many(&party_bins, &batch, io)?;
                Ok(arith)
            },
            |(), _net| Ok(()),
        );

        // Reconstruct arithmetic shares and verify.
        for i in 0..n {
            let combined = combine_ring_element(results[0][i], results[1][i], results[2][i]);
            assert_eq!(
                combined, values[i],
                "ring_b2a_many mismatch at index {i}: expected {:?}, got {:?}",
                values[i], combined
            );
        }
    }
}
