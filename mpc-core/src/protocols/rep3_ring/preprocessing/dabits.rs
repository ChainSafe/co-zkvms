//! Cheng23 Protocol Π₁ daBits: single-bit Boolean→Arithmetic conversion.
//!
//! Provides partial-lazy storage for daBit correlated tuples, with:
//! - P0 storing 1 field element per daBit (received s₂₀)
//! - P1 storing nothing (fully seed-regenerable)
//! - P2 storing 2 field elements per daBit (s₂₀ + s₁₂)
//!
//! Online cost: 3 bits / 1 round (same as standard bit injection).

use super::backing_store;
use crate::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use crate::protocols::rep3::{
    PartyID, Rep3PrimeFieldShare,
    network::{IoContext, Rep3Network},
};
use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3_ring::{Rep3RingShare, ring::bit::Bit};
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;

/// Batch of Cheng23 Π₁ correlated tuples for single-bit B2A (bit injection).
///
/// Each tuple contains:
/// - `gammas[i]`: random bit γ known to P0 only (false for P1/P2)
/// - `thetas[i]`: random bit θ known to P1 and P2 (false for P0)
/// - `v_shares[i]`: replicated arithmetic sharing of v = (−1)^θ · γ
pub struct DaBitBatch<F: PrimeField> {
    pub gammas: Vec<bool>,
    pub thetas: Vec<bool>,
    pub v_shares: Vec<Rep3PrimeFieldShare<F>>,
}

// ---------------------------------------------------------------------------
// LazyDaBits: Cheng23 Π₁ partial-lazy daBit storage
// ---------------------------------------------------------------------------

/// Partial-lazy storage for Π₁ daBit tuples.
///
/// After preprocessing (`random_dabits_lazy`), stores:
/// - P0: seed snapshots + `total` field elements (s₂₀ received from P2)
/// - P1: seed snapshots only (fully regenerable from P0↔P1 and P1↔P2 seeds)
/// - P2: seed snapshot + `2*total` field elements (interleaved s₂₀, s₁₂)
///
/// `take_batch(n)` regenerates daBit tuples on demand from the stored seeds
/// and received field elements.
pub struct LazyDaBits<F: PrimeField> {
    /// Snapshot of pairwise RNG shared with NEXT party.
    seed1: [u8; crate::SEED_SIZE],
    pos1: u128,
    /// Snapshot of pairwise RNG shared with PREV party.
    seed2: [u8; crate::SEED_SIZE],
    pos2: u128,
    /// Bytes per field element: ceil(MODULUS_BIT_SIZE / 8).
    field_bytes: usize,
    party_id: PartyID,
    total: usize,
    cursor: usize,
    /// Received field elements that cannot be regenerated from seeds.
    /// May be backed by a memory-mapped file.
    ///
    /// - P0: `[s₂₀_0, s₂₀_1, …]` — length `total`
    /// - P1: `[]` — empty
    /// - P2: `[s₂₀_0, s₁₂_0, s₂₀_1, s₁₂_1, …]` — interleaved, length `2*total`
    stored: backing_store::BackingStore<F>,
    /// Path to the meta file on disk (set when loaded via `load()`).
    meta_path: Option<std::path::PathBuf>,
}

impl<F: PrimeField> LazyDaBits<F> {
    pub fn empty(party_id: PartyID) -> Self {
        Self {
            seed1: [0u8; crate::SEED_SIZE],
            pos1: 0,
            seed2: [0u8; crate::SEED_SIZE],
            pos2: 0,
            field_bytes: usize::try_from(F::MODULUS_BIT_SIZE)
                .expect("u32 fits into usize")
                .div_ceil(8),
            party_id,
            total: 0,
            cursor: 0,
            stored: backing_store::BackingStore::Empty,
            meta_path: None,
        }
    }

    pub fn new(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        total: usize,
        stored: Vec<F>,
        party_id: PartyID,
    ) -> Self {
        Self::new_with_store(
            seed1,
            pos1,
            seed2,
            pos2,
            total,
            backing_store::BackingStore::from_vec(stored),
            party_id,
        )
    }

    pub(crate) fn new_with_store(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        total: usize,
        stored: backing_store::BackingStore<F>,
        party_id: PartyID,
    ) -> Self {
        Self {
            seed1,
            pos1,
            seed2,
            pos2,
            field_bytes: usize::try_from(F::MODULUS_BIT_SIZE)
                .expect("u32 fits into usize")
                .div_ceil(8),
            party_id,
            total,
            cursor: 0,
            stored,
            meta_path: None,
        }
    }

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

    /// Drain `n` daBit tuples from the pool.
    pub fn take_batch(&mut self, n: usize) -> DaBitBatch<F> {
        assert!(
            self.cursor + n <= self.total,
            "LazyDaBits: need {n}, have {}",
            self.total - self.cursor
        );

        let fb = self.field_bytes;
        let cursor_base = self.cursor;

        let party_id = self.party_id;

        // P2: stored values only, no RNG regeneration needed for v_shares.
        if party_id == PartyID::ID2 {
            // Regenerate theta from seed2 (P2↔P1 stream).
            let theta_offset = cursor_base; // 1 byte per daBit
            let theta_bytes = seek_and_generate(self.seed2, self.pos2, theta_offset, n);
            let thetas: Vec<bool> = theta_bytes.iter().map(|b| (b & 1) != 0).collect();

            // Stored layout: [s₂₀_0, s₁₂_0, s₂₀_1, s₁₂_1, …]
            let store_base = cursor_base * 2;
            let stored_slice = {
                #[cfg(feature = "reuse-preproc")]
                {
                    self.stored
                        .read_reuse(store_base, store_base + 2 * n)
                        .unwrap_or_else(|e| {
                            panic!(
                                "LazyDaBits(P2): read_reuse({}..{}) failed: {e}",
                                store_base,
                                store_base + 2 * n
                            );
                        })
                }
                #[cfg(not(feature = "reuse-preproc"))]
                {
                    self.stored
                        .read_consume(store_base, store_base + 2 * n)
                        .unwrap_or_else(|e| {
                            panic!(
                                "LazyDaBits(P2): read_consume({}..{}) failed: {e}",
                                store_base,
                                store_base + 2 * n
                            );
                        })
                }
            };
            let v_shares: Vec<Rep3PrimeFieldShare<F>> = (0..n)
                .map(|i| {
                    let s20 = stored_slice[2 * i]; // v.a for P2
                    let s12 = stored_slice[2 * i + 1]; // v.b for P2
                    Rep3PrimeFieldShare::new(s20, s12)
                })
                .collect();

            self.cursor += n;
            self.persist_cursor();
            self.stored.consume(store_base, store_base + 2 * n);
            return DaBitBatch {
                gammas: vec![false; n],
                thetas,
                v_shares,
            };
        }

        // P0 and P1: regenerate from seeds.
        // P0↔P1 stream uses per-item interleaved layout:
        //   [g₀(1B), a1₀(fb), r1₀(fb) | g₁(1B), a1₁(fb), r1₁(fb) | ...]
        let da_stride = 1 + 2 * fb;
        let interleaved_offset = cursor_base * da_stride;
        let interleaved_needed = n * da_stride;

        match party_id {
            PartyID::ID0 => {
                // seed1 = P0↔P1 (interleaved), seed2 = P0↔P2
                let s1_buf = seek_and_generate(
                    self.seed1,
                    self.pos1,
                    interleaved_offset,
                    interleaved_needed,
                );
                // g2 from seed2 (P0↔P2 stream)
                let g2_bytes = seek_and_generate(self.seed2, self.pos2, cursor_base, n);

                let gammas: Vec<bool> = (0..n)
                    .map(|i| ((s1_buf[i * da_stride] ^ g2_bytes[i]) & 1) != 0)
                    .collect();

                let stored_slice = {
                    #[cfg(feature = "reuse-preproc")]
                    {
                        self.stored
                            .read_reuse(cursor_base, cursor_base + n)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "LazyDaBits(P0): read_reuse({}..{}) failed: {e}",
                                    cursor_base,
                                    cursor_base + n
                                );
                            })
                    }
                    #[cfg(not(feature = "reuse-preproc"))]
                    {
                        self.stored
                            .read_consume(cursor_base, cursor_base + n)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "LazyDaBits(P0): read_consume({}..{}) failed: {e}",
                                    cursor_base,
                                    cursor_base + n
                                );
                            })
                    }
                };
                let v_shares: Vec<Rep3PrimeFieldShare<F>> = (0..n)
                    .map(|i| {
                        let r1: F = parse_field(&s1_buf, i * da_stride + 1 + fb);
                        let s20 = stored_slice[i]; // received from P2
                        Rep3PrimeFieldShare::new(r1, s20) // (.a=s₀₁=r₁, .b=s₂₀)
                    })
                    .collect();

                self.cursor += n;
                self.persist_cursor();
                self.stored.consume(cursor_base, cursor_base + n);
                DaBitBatch {
                    gammas,
                    thetas: vec![false; n],
                    v_shares,
                }
            }
            PartyID::ID1 => {
                // seed1 = P1↔P2, seed2 = P1↔P0 (same stream as P0's seed1)
                // Theta from seed1 (P1↔P2)
                let theta_bytes = seek_and_generate(self.seed1, self.pos1, cursor_base, n);
                let thetas: Vec<bool> = theta_bytes.iter().map(|b| (b & 1) != 0).collect();

                // alpha1, r1 from seed2 (P1↔P0 = P0↔P1 interleaved stream)
                let s2_buf = seek_and_generate(
                    self.seed2,
                    self.pos2,
                    interleaved_offset,
                    interleaved_needed,
                );

                let v_shares: Vec<Rep3PrimeFieldShare<F>> = (0..n)
                    .map(|i| {
                        let alpha1: F = parse_field(&s2_buf, i * da_stride + 1);
                        let r1: F = parse_field(&s2_buf, i * da_stride + 1 + fb);
                        let theta = thetas[i];
                        let neg1_theta = if theta { -F::one() } else { F::one() };
                        let v1 = neg1_theta * alpha1;
                        let s12 = v1 - r1;
                        Rep3PrimeFieldShare::new(s12, r1) // (.a=s₁₂, .b=s₀₁=r₁)
                    })
                    .collect();

                self.cursor += n;
                DaBitBatch {
                    gammas: vec![false; n],
                    thetas,
                    v_shares,
                }
            }
            PartyID::ID2 => unreachable!(), // handled above
        }
    }

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    /// Write this lazy source to `dir`.
    ///
    /// Creates `dabits.meta` (all parties) and `dabits.stored` (P0/P2 only).
    pub fn save(&self, dir: &std::path::Path) -> std::io::Result<()> {
        const { backing_store::assert_field_layout::<F>() };
        std::fs::create_dir_all(dir)?;

        // Data file first (page-cache write, no fsync), then meta with fsync.
        if !self.stored.is_empty() {
            self.stored.save_to_file(&dir.join("dabits.stored"))?;
        }
        backing_store::write_meta(
            &dir.join("dabits.meta"),
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
    /// P0/P2 memory-map the stored data file for JIT retrieval.
    pub fn load(dir: &std::path::Path, party_id: PartyID) -> std::io::Result<Self> {
        const { backing_store::assert_field_layout::<F>() };

        let meta_path = dir.join("dabits.meta");
        let meta = backing_store::read_meta(&meta_path)?;
        assert_eq!(
            meta.party_id_byte,
            backing_store::party_id_to_byte(party_id)
        );

        let stored = if meta.total > 0 && party_id != PartyID::ID1 {
            let data_path = dir.join("dabits.stored");
            let bs = backing_store::BackingStore::load_from_file(&data_path)?;
            // Validate stored data length matches meta.total.
            // P0 stores 1 field element per daBit, P2 stores 2.
            let expected = match party_id {
                PartyID::ID0 => meta.total,
                PartyID::ID2 => meta.total * 2,
                _ => 0,
            };
            if bs.len() < expected {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "dabits.stored: expected at least {} elements, got {} (total={}, cursor={}, party={:?})",
                        expected,
                        bs.len(),
                        meta.total,
                        meta.cursor,
                        party_id,
                    ),
                ));
            }
            bs
        } else {
            backing_store::BackingStore::Empty
        };

        std::result::Result::Ok(Self {
            seed1: meta.seed1,
            pos1: meta.pos1,
            seed2: meta.seed2,
            pos2: meta.pos2,
            field_bytes: meta.field_bytes,
            party_id,
            total: meta.total,
            cursor: meta.cursor,
            stored,
            meta_path: Some(meta_path),
        })
    }

    /// Persist the current cursor to the meta file on disk.
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
    /// For P0: appends received s₂₀ to stored backing.
    /// For P1: only bumps `total` (fully seed-regenerable).
    /// For P2: appends interleaved s₂₀+s₁₂ to stored backing.
    pub(crate) fn apply_extension(&mut self, deficit: usize, stored_ext: Vec<F>) {
        if deficit == 0 {
            return;
        }
        if !stored_ext.is_empty() {
            self.stored.extend(&stored_ext);
        }
        self.total += deficit;
    }
}

impl<F: PrimeField> Drop for LazyDaBits<F> {
    fn drop(&mut self) {
        #[cfg(not(feature = "reuse-preproc"))]
        self.persist_cursor();
    }
}

// ---------------------------------------------------------------------------
// Preprocessing: random_dabits_lazy
// ---------------------------------------------------------------------------

/// Generate Cheng23 Π₁ daBit tuples with partial-lazy storage.
///
/// Produces correlated tuples {γ, θ, ⟦v⟧^A} where v = (−1)^θ·γ, with:
/// - P0/P1 storing only RNG seeds (P1 fully regenerable)
/// - P0 additionally storing 1 field element per daBit (received from P2)
/// - P2 storing 2 field elements per daBit
///
/// **Communication:** 2 rounds. Round 1: P0→P2 (n F), P1→P2 (n F). Round 2: P2→P0 (n F).
#[tracing::instrument(skip_all, name = "dabits_preprocess_lazy")]
pub fn random_dabits_lazy<F: PrimeField, N: Rep3NetworkWorker>(
    num: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<LazyDaBits<F>> {
    let party_id = io.party_id();
    if num == 0 {
        return Ok(LazyDaBits::empty(party_id));
    }

    let fb = usize::try_from(F::MODULUS_BIT_SIZE)
        .expect("u32 fits into usize")
        .div_ceil(8);

    // Fork a dedicated Rep3Rand and snapshot state BEFORE generating bytes.
    let mut eda_rand = io.main().rngs.rand.fork();
    let (seed1, pos1, seed2, pos2) = eda_rand.snapshot();

    // Byte budget per stream:
    // P0↔P1 (rng1 for P0, rng2 for P1): num + num*fb (alpha) + num*fb (r1)
    // P0↔P2 (rng2 for P0, rng1 for P2): num (gamma g2)
    // P1↔P2 (rng1 for P1, rng2 for P2): num (theta)

    let stored: Vec<F> = match party_id {
        PartyID::ID0 => {
            // Generate interleaved [g₀(1B), a1₀(fb), r1₀(fb), ...] from rng1.
            // g2 from rng2 (P0↔P2).
            let da_stride = 1 + 2 * fb;
            let stream1_len = num * da_stride;
            let stream2_len = num;
            let (stream1, stream2) = {
                let mut s1 = vec![0u8; stream1_len];
                let mut s2 = vec![0u8; stream2_len];
                rayon::join(
                    || eda_rand.rng1.fill_bytes(&mut s1),
                    || eda_rand.rng2.fill_bytes(&mut s2),
                );
                (s1, s2)
            };
            let g2_bytes = &stream2;

            // Compute alpha2 = F(gamma) - alpha1 for each daBit, and send to P2.
            let alpha2: Vec<F> = (0..num)
                .into_par_iter()
                .with_min_len(256)
                .map(|i| {
                    let base = i * da_stride;
                    let gamma_bit = ((stream1[base] ^ g2_bytes[i]) & 1) != 0;
                    let alpha1: F = parse_field(&stream1, base + 1);
                    F::from(gamma_bit as u64) - alpha1
                })
                .collect();

            // Round 1: P0 → P2 sends alpha2
            io.network().send_many(PartyID::ID2, &alpha2)?;
            // Round 2: P0 ← P2 receives s20
            let s20: Vec<F> = io.network().recv_many(PartyID::ID2)?;
            debug_assert_eq!(s20.len(), num);
            s20
        }
        PartyID::ID1 => {
            // Generate interleaved from rng2 (P1↔P0): g1, alpha1, r1
            // Generate from rng1 (P1↔P2): theta
            let da_stride = 1 + 2 * fb;
            let stream2_len = num * da_stride; // P1↔P0 interleaved stream
            let stream1_len = num; // P1↔P2 stream
            let (stream2, stream1) = {
                let mut s2 = vec![0u8; stream2_len];
                let mut s1 = vec![0u8; stream1_len];
                rayon::join(
                    || eda_rand.rng2.fill_bytes(&mut s2),
                    || eda_rand.rng1.fill_bytes(&mut s1),
                );
                (s2, s1)
            };
            let theta_bytes = &stream1;

            // Compute s12 = v1 - r1 for each daBit, and send to P2.
            let s12: Vec<F> = (0..num)
                .into_par_iter()
                .with_min_len(256)
                .map(|i| {
                    let base = i * da_stride;
                    let theta = (theta_bytes[i] & 1) != 0;
                    let neg1_theta = if theta { -F::one() } else { F::one() };
                    let alpha1: F = parse_field(&stream2, base + 1);
                    let r1: F = parse_field(&stream2, base + 1 + fb);
                    let v1 = neg1_theta * alpha1;
                    v1 - r1
                })
                .collect();

            // Round 1: P1 → P2 sends s12
            io.network().send_many(PartyID::ID2, &s12)?;
            // P1 stores nothing
            Vec::new()
        }
        PartyID::ID2 => {
            // Generate from rng2 (P2↔P1): theta
            // Advance rng1 (P2↔P0) by num bytes to keep in sync.
            let stream2_len = num; // theta
            let stream1_len = num; // g2 (not used, just advance RNG)
            let (stream2, _stream1) = {
                let mut s2 = vec![0u8; stream2_len];
                let mut s1 = vec![0u8; stream1_len];
                rayon::join(
                    || eda_rand.rng2.fill_bytes(&mut s2),
                    || eda_rand.rng1.fill_bytes(&mut s1),
                );
                (s2, s1)
            };
            let theta_bytes = &stream2;

            // Round 1: receive alpha2 from P0 and s12 from P1.
            let alpha2: Vec<F> = io.network().recv_many(PartyID::ID0)?;
            let s12_recv: Vec<F> = io.network().recv_many(PartyID::ID1)?;
            debug_assert_eq!(alpha2.len(), num);
            debug_assert_eq!(s12_recv.len(), num);

            // Compute s20 = v2 = (-1)^theta * alpha2 and send to P0.
            let s20: Vec<F> = (0..num)
                .into_par_iter()
                .with_min_len(256)
                .map(|i| {
                    let theta = (theta_bytes[i] & 1) != 0;
                    let neg1_theta = if theta { -F::one() } else { F::one() };
                    neg1_theta * alpha2[i]
                })
                .collect();

            // Round 2: P2 → P0 sends s20
            io.network().send_many(PartyID::ID0, &s20)?;

            // Store interleaved [s20_0, s12_0, s20_1, s12_1, ...]
            let mut stored = Vec::with_capacity(2 * num);
            for i in 0..num {
                stored.push(s20[i]);
                stored.push(s12_recv[i]);
            }
            stored
        }
    };

    Ok(LazyDaBits::new(
        seed1, pos1, seed2, pos2, num, stored, party_id,
    ))
}

// ---------------------------------------------------------------------------
// Online: bit_inject_field_many (Π₁)
// ---------------------------------------------------------------------------

/// Convert binary Rep3 bit shares to arithmetic field shares using Π₁ daBits.
///
/// **Protocol (1 round, 3 bits):**
/// 1. P0 broadcasts m₀ = b.a ⊕ b.b ⊕ γ to P1,P2
/// 2. P1 sends m₁ = θ ⊕ b.a to P0
/// 3. All compute σ, then ⟦b⟧^A = (−1)^σ · ⟦v⟧^A + ⟦β⟧^A_{G₁}
pub fn bit_inject_field_many<F: PrimeField, N: Rep3Network>(
    x: &[Rep3RingShare<Bit>],
    batch: &DaBitBatch<F>,
    io: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    debug_assert_eq!(batch.gammas.len(), n);
    debug_assert_eq!(batch.thetas.len(), n);
    debug_assert_eq!(batch.v_shares.len(), n);

    let party_id = io.id;

    // --- Round 1: P0 broadcasts m₀, P1 sends m₁ ---
    // Use u8 for network (Bit lacks CanonicalSerialize).
    let (m0s, m1s): (Vec<u8>, Vec<u8>) = match party_id {
        PartyID::ID0 => {
            // m₀ = x.a ⊕ x.b ⊕ γ
            let m0s: Vec<u8> = x
                .iter()
                .zip(&batch.gammas)
                .map(|(xi, &gamma)| (xi.a.0.convert() ^ xi.b.0.convert() ^ gamma) as u8)
                .collect();
            io.network.send_many(PartyID::ID1, &m0s)?;
            io.network.send_many(PartyID::ID2, &m0s)?;
            let m1s: Vec<u8> = io.network.recv_many(PartyID::ID1)?;
            (m0s, m1s)
        }
        PartyID::ID1 => {
            let m0s: Vec<u8> = io.network.recv_many(PartyID::ID0)?;
            // m₁ = θ ⊕ x.a (P1.a = s₁₂)
            let m1s: Vec<u8> = x
                .iter()
                .zip(&batch.thetas)
                .map(|(xi, &theta)| (xi.a.0.convert() ^ theta) as u8)
                .collect();
            io.network.send_many(PartyID::ID0, &m1s)?;
            (m0s, m1s)
        }
        PartyID::ID2 => {
            let m0s: Vec<u8> = io.network.recv_many(PartyID::ID0)?;
            (m0s, vec![])
        }
    };

    // --- Local computation ---
    let results: Vec<Rep3PrimeFieldShare<F>> = match party_id {
        PartyID::ID0 => {
            x.iter()
                .zip(m0s.iter())
                .zip(m1s.iter())
                .zip(batch.gammas.iter())
                .zip(batch.v_shares.iter())
                .map(|((((xi, &_m0), &m1), &gamma), v)| {
                    // σ = m₁ ⊕ x.a ⊕ x.b ⊕ γ
                    let sigma = (m1 != 0) ^ xi.a.0.convert() ^ xi.b.0.convert() ^ gamma;
                    let neg1_sigma = if sigma { -F::one() } else { F::one() };
                    // P0: no β addition (β unknown to P0)
                    Rep3PrimeFieldShare::new(v.a * neg1_sigma, v.b * neg1_sigma)
                })
                .collect()
        }
        PartyID::ID1 => {
            m0s.iter()
                .zip(x.iter())
                .zip(batch.thetas.iter())
                .zip(batch.v_shares.iter())
                .map(|(((&m0, xi), &theta), v)| {
                    // β = m₀ ⊕ x.a (P1.a = s₁₂)
                    let beta = (m0 != 0) ^ xi.a.0.convert();
                    let sigma = beta ^ theta;
                    let neg1_sigma = if sigma { -F::one() } else { F::one() };
                    // P1: add β to .a (s₁₂ component)
                    Rep3PrimeFieldShare::new(
                        v.a * neg1_sigma + F::from(beta as u64),
                        v.b * neg1_sigma,
                    )
                })
                .collect()
        }
        PartyID::ID2 => {
            m0s.iter()
                .zip(x.iter())
                .zip(batch.thetas.iter())
                .zip(batch.v_shares.iter())
                .map(|(((&m0, xi), &theta), v)| {
                    // β = m₀ ⊕ x.b (P2.b = s₁₂)
                    let beta = (m0 != 0) ^ xi.b.0.convert();
                    let sigma = beta ^ theta;
                    let neg1_sigma = if sigma { -F::one() } else { F::one() };
                    // P2: add β to .b (s₁₂ component)
                    Rep3PrimeFieldShare::new(
                        v.a * neg1_sigma,
                        v.b * neg1_sigma + F::from(beta as u64),
                    )
                })
                .collect()
        }
    };

    Ok(results)
}

// ---------------------------------------------------------------------------
// Helpers (shared with edabits.rs)
// ---------------------------------------------------------------------------

/// Seek RNG to `byte_offset` from snapshotted position and generate `needed` bytes.
pub(crate) fn seek_and_generate(
    seed: [u8; crate::SEED_SIZE],
    base_pos: u128,
    byte_offset: usize,
    needed: usize,
) -> Vec<u8> {
    use rand::RngCore;
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

/// Parse a field element from 16 raw bytes at `start` offset (128-bit entropy).
pub(crate) fn parse_field<Fp: PrimeField>(bytes: &[u8], start: usize) -> Fp {
    let lo = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
    let hi = u64::from_le_bytes(bytes[start + 8..start + 16].try_into().unwrap());
    Fp::from((hi as u128) << 64 | lo as u128)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::protocols::rep3::test_utils::run_rep3_local_test_with_coordinator;
    use ark_bn254::Fr;
    use mpc_types::protocols::rep3::combine_field_elements;
    use mpc_types::protocols::rep3_ring::{
        ring::{bit::Bit as RingBit, ring_impl::RingElement},
        share_ring_element,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    type Rep3RingShare<T> = mpc_types::protocols::rep3_ring::Rep3RingShare<T>;

    #[test]
    fn random_dabits_lazy_roundtrip() {
        const NBITS: usize = 64;
        let mut rng = ChaCha20Rng::seed_from_u64(0xDAB1_3001);
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
            |x_shares: Vec<Rep3RingShare<RingBit>>, mut io_ctx| {
                let n = x_shares.len();
                let mut lazy = random_dabits_lazy::<Fr, _>(n, &mut io_ctx)?;
                let batch = lazy.take_batch(n);
                bit_inject_field_many::<Fr, _>(&x_shares, &batch, io_ctx.main()).map_err(Into::into)
            },
            |(): (), _net| Ok(()),
        )
        .0;

        let combined = combine_field_elements(&outputs[0], &outputs[1], &outputs[2]);
        let expected = bits
            .into_iter()
            .map(|b| Fr::from(b as u64))
            .collect::<Vec<_>>();
        assert_eq!(combined, expected);
    }
}
